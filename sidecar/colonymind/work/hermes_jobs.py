"""Read-only job projections. No scheduler or store initialization on this path."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import shlex
import sqlite3
import time
import threading

from colonymind.health_counters import bump

log = logging.getLogger(__name__)
_process_read_warnings: set[tuple] = set()
_process_warning_lock = threading.Lock()
ACTIVE = {'running', 'claimed', 'started', 'working', 'dispatched', 'checking', 'integrating'}


def failed(error: Exception, site: str = 'reader', *, once: bool = False) -> None:
    key = (site, type(error), getattr(error, 'sqlite_errorname', None))
    with _process_warning_lock:
        report = not once or key not in _process_read_warnings
        if once:
            _process_read_warnings.add(key)
    if report:
        log.warning('jobwatch %s failed: %s sqlite=%s', site, type(error).__name__,
                    getattr(error, 'sqlite_errorname', None))
    bump('context_reads_failed' if site == 'reader' else f'writes_dropped.{site}')


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def timestamp(value) -> float | None:
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()


@dataclass
class HermesJob:
    id: str
    kind: str
    label: str
    state: str
    started_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None
    end_reason: str | None = None
    owner_task_id: str | None = None
    limits: dict[str, float] = field(default_factory=dict)
    usage: dict[str, float] = field(default_factory=dict)
    signature: str | None = None
    error_hash: str | None = None
    pid: int | None = None
    host_start_time: int | None = None
    cpu: float | None = None
    output: str | None = None
    source: str = ''

    @property
    def subject(self) -> str:
        return digest(f'{self.kind}:{self.source}:{self.id}')


class JobSnapshot(list[HermesJob]):
    """Availability is separate from an empty or unreadable source."""
    def __init__(self):
        super().__init__()
        self.available: set[str] = set()


def _json(path: Path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('record_too_large')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _command(value: str) -> tuple[str, str]:
    try:
        words = shlex.split(value)
    except ValueError:
        words = value.split(maxsplit=1)
    program = Path(words[0]).name if words else 'process'
    # Never use command arguments or arbitrary shell text as a display label.
    return (program if re.fullmatch(r'[A-Za-z0-9_.+-]{1,64}', program) else 'process', digest(value))


def _process(row: dict, *, ended: float | None = None) -> HermesJob:
    label, signature = _command(row.get('command', ''))
    last = next((line.strip() for line in reversed(row.get('output', '').splitlines()) if line.strip()), '')
    return HermesJob(
        id=str(row.get('id') or row['session_id']), kind='process', label=label,
        state='running' if ended is None else 'finished', started_at=timestamp(row.get('started_at')),
        ended_at=ended, exit_code=row.get('exit_code'),
        end_reason=row.get('completion_reason'), owner_task_id=row.get('owner_task_id') or row.get('task_id'),
        signature=signature, error_hash=digest(last) if last else None,
        pid=row.get('pid') if row.get('pid_scope', 'host') == 'host' else None,
        host_start_time=row.get('host_start_time'), source='process')


def rows(path: Path, table: str, columns: tuple[str, ...]) -> list[dict]:
    """The same mode=ro, query_only and bounded snapshot as turns.hermes_work."""
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        deadline = time.monotonic() + 2
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        conn.execute('BEGIN')
        present = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if not present:
            raise ValueError('table_missing')
        selected = ','.join(f'"{name}"' for name in columns if name in present)
        return [dict(row) for row in conn.execute(f'SELECT {selected} FROM "{table}"')]


def _limits(value) -> dict[str, float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        return {}
    return {key: float(value[key]) for key in ('wall_seconds', 'max_turns', 'turns')
            if isinstance(value.get(key), (int, float))}


def read_jobs(hermes_home: str | Path, now: float, *, data_dir: Path | None = None,
              since: float | None = None, processes_only: bool = False) -> JobSnapshot:
    """Read the selected profile's live records and retained completion receipts."""
    home = Path(hermes_home).expanduser()
    result = JobSnapshot()
    checkpoint = home / 'processes.json'
    if checkpoint.is_file():
        try:
            values = _json(checkpoint)
            if not isinstance(values, list):
                raise ValueError('checkpoint_shape')
            complete = True
            for row in values:
                try:
                    result.append(_process(row))
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    complete = False
                    failed(error, once=processes_only)
            if complete:
                result.available.add('process')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            failed(error, once=processes_only)
    receipts = home / 'logs' / 'process-results'
    checkpoint_ids = {job.id for job in result}
    try:
        for path in sorted(receipts.glob('proc_*.json')):
            try:
                ended = path.stat().st_mtime
                # Keep old receipts for stale checkpoints during the existing scan.
                if ended > now or (since is not None and ended < since and path.stem not in checkpoint_ids):
                    continue
                row = _json(path)
                result.append(_process(row, ended=ended))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                failed(error, once=processes_only)
    except OSError as error:
        failed(error, once=processes_only)

    if processes_only:
        return result

    # Native cron's ledger is also projected by turns.hermes_work.cron_view.
    # Read here with an explicit home; cron_view binds process-global configuration
    # and truncates active work, which is inappropriate for a complete watch pass.
    cron = home / 'cron' / 'executions.db'
    declarations = {}
    jobs_path = home / 'cron' / 'jobs.json'
    if jobs_path.is_file():
        try:
            declarations = {str(job['id']): job for job in _json(jobs_path).get('jobs', [])}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            failed(error)
    if cron.is_file():
        try:
            for row in rows(cron, 'executions', ('id', 'job_id', 'status', 'claimed_at', 'started_at',
                                                 'finished_at', 'pid', 'process_started_at')):
                declaration = declarations.get(str(row['job_id']), {})
                limits = _limits(declaration.get('limits', {}))
                for key in ('max_runtime_seconds', 'timeout_seconds'):
                    if isinstance(declaration.get(key), (int, float)):
                        limits['wall_seconds'] = declaration[key]
                result.append(HermesJob(str(row['id']), 'cron', 'scheduled task', row['status'],
                    timestamp(row.get('started_at') or row.get('claimed_at')), timestamp(row.get('finished_at')),
                    owner_task_id=str(row['job_id']), limits=limits, pid=row.get('pid'), source='cron'))
            result.available.add('cron')
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
            failed(error)

    paths = [home / 'kanban.db', *sorted((home / 'kanban' / 'boards').glob('*/kanban.db'))]
    for path in paths:
        if not path.is_file():
            continue
        source = 'kanban:' + digest(str(path.resolve()))
        try:
            for row in rows(path, 'task_runs', ('id', 'task_id', 'status', 'worker_pid', 'started_at',
                                               'ended_at', 'max_runtime_seconds')):
                limit = row.get('max_runtime_seconds')
                result.append(HermesJob(str(row['id']), 'kanban', 'task', row['status'],
                    timestamp(row.get('started_at')), timestamp(row.get('ended_at')),
                    owner_task_id=str(row['task_id']), limits={'wall_seconds': limit} if limit else {},
                    pid=row.get('worker_pid'), source=source))
            result.available.add(source)
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
            failed(error)

    if data_dir is not None:
        # HandsStore.open initializes/migrates the database. Only its file contract
        # is reused here; no controller or writable connection is constructed.
        from colonymind.hands.store import DB_FILE
        path = data_dir / DB_FILE
        if path.is_file():
            try:
                for row in rows(path, 'hands', ('id', 'task_id', 'state', 'started_at', 'ended_at',
                                                'budgets', 'usage', 'executor_ref')):
                    result.append(HermesJob(str(row['id']), 'hand', 'delegated task', row['state'],
                        timestamp(row.get('started_at')), timestamp(row.get('ended_at')),
                        owner_task_id=str(row['task_id']), limits=_limits(row.get('budgets')),
                        usage=_limits(row.get('usage')), source='hands'))
                result.available.add('hands')
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
                failed(error)
    return result
