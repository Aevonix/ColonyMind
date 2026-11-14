"""Bounded activity probes; retain only metadata and output digests."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import sqlite3
import subprocess

from colonymind.work.hermes_jobs import ACTIVE, HermesJob, JobSnapshot, _json, digest, failed, rows

SHELLS = {'sh', 'bash', 'zsh', 'fish', 'dash', 'ksh', 'tcsh', 'csh', 'login'}


def record_activity(jobs: JobSnapshot, hermes_home: str | Path) -> None:
    """Complete non-/proc samples from the selected profile's progress records."""
    if Path('/proc/self/stat').is_file():
        return
    pending = [job for job in jobs if job.state in ACTIVE and job.pid and
               job.cpu is not None and job.output is None]
    if not pending:
        return
    home = Path(hermes_home).expanduser()
    processes = {job.id: job for job in pending if job.kind == 'process'}
    if processes:
        try:
            values = _json(home / 'processes.json')
            if not isinstance(values, list):
                raise ValueError('checkpoint_shape')
            for row in values:
                try:
                    job = processes.get(str(row.get('id') or row['session_id']))
                    if (job and row.get('pid') == job.pid and row.get('pid_scope', 'host') == 'host'
                            and isinstance(row.get('output'), str)):
                        job.output = digest(row['output'])
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    failed(error)
        except (OSError, ValueError, TypeError) as error:
            failed(error)
    cron = [job for job in pending if job.kind == 'cron']
    if cron and (home / 'cron' / 'jobs.json').is_file():
        try:
            declarations = _json(home / 'cron' / 'jobs.json')['jobs']
            for row in declarations:
                try:
                    matches = [job for job in jobs if job.kind == 'cron' and job.state in ACTIVE
                               and job.owner_task_id == str(row['id'])]
                    claim = row.get('run_claim')
                    # A schedule's claim cannot identify overlapping attempts.
                    if (len(matches) == 1 and matches[0] in cron and row.get('schedule', {}).get('kind') == 'once'
                            and isinstance(claim, dict) and claim.get('at') and claim.get('by')):
                        matches[0].output = digest(str((claim['by'], claim['at'])))
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    failed(error)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            failed(error)
    kanban = {(job.source, job.id): job for job in pending if job.kind == 'kanban'}
    if kanban:
        try:
            paths = [home / 'kanban.db', *sorted((home / 'kanban' / 'boards').glob('*/kanban.db'))]
            for path in paths:
                source = 'kanban:' + digest(str(path.resolve()))
                if not any(key[0] == source for key in kanban):
                    continue
                try:
                    for row in rows(path, 'task_runs', ('id', 'worker_pid', 'last_heartbeat_at')):
                        job = kanban.get((source, str(row['id'])))
                        heartbeat = row.get('last_heartbeat_at')
                        if job and row.get('worker_pid') == job.pid and isinstance(heartbeat, (int, float)):
                            job.output = digest(str(heartbeat))
                except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
                    failed(error)
        except OSError as error:
            failed(error)
    # Bind record progress to the current process-tree identities, just as /proc
    # output is bound below. A recycled PID must start a fresh idle interval.
    measured = JobSnapshot()
    measured.extend(job for job in pending if job.output is not None)
    for job in measured:
        job.cpu = None
    probe(measured, tmux=False)
    for job in measured:
        if job.cpu is None:
            job.output = None


def _seconds(value: str) -> float:
    days, _, clock = value.rpartition('-')
    parts = [float(part) for part in clock.split(':')]
    total = 0.
    for part in parts:
        total = total * 60 + part
    return total + (int(days) * 86400 if days else 0)


def probe(jobs: JobSnapshot, *, tmux: bool = True) -> None:
    shells = set()
    if tmux and shutil.which('tmux'):
        try:
            response = subprocess.run(['tmux', 'list-panes', '-a', '-F',
                '#{session_id}|#{pane_id}|#{pane_pid}|#{session_created}|#{history_size}|#{history_bytes}|#{cursor_x}|#{cursor_y}|#{pane_dead}|#{session_attached}|#{pane_current_command}|#{==:#{pane_start_command},}'],
                capture_output=True, text=True, timeout=5, check=False)
            if response.returncode == 0:
                for line in response.stdout.splitlines():
                    session, pane, pid, created, size, size_bytes, x, y, dead, attached, command, default_shell = line.split('|')
                    if attached != '0':
                        continue
                    # Explicit pane commands may be noninteractive shell scripts.
                    if command in SHELLS and default_shell == '1':
                        shells.add(session + ':' + pane)
                    jobs.append(HermesJob(session + ':' + pane, 'tmux', 'terminal job',
                        'finished' if dead == '1' else 'running', float(created), pid=int(pid),
                        output=digest(':'.join((size, size_bytes, x, y))), source='tmux'))
                jobs.available.add('tmux')
            elif 'no server running' in response.stderr or 'No such file or directory' in response.stderr:
                jobs.available.add('tmux')
            else:
                failed(subprocess.CalledProcessError(response.returncode, 'tmux'))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            failed(error)
    if not any(job.pid for job in jobs):
        return
    try:
        foreground = True
        try:
            response = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,tpgid=,time=,lstart='],
                                      capture_output=True, text=True, timeout=5, check=True)
        except subprocess.CalledProcessError as error:
            failed(error)
            foreground = False
            response = subprocess.run(['ps', '-axo', 'pid=,ppid=,time=,lstart='],
                                      capture_output=True, text=True, timeout=5, check=True)
        processes = {}
        at_prompt = {}
        for line in response.stdout.splitlines():
            if foreground:
                pid, parent, group, terminal, cpu, started = line.strip().split(None, 5)
                if terminal not in {'-', '?'} and int(terminal) > 0:
                    at_prompt[int(pid)] = int(group) == int(terminal)
            else:
                pid, parent, cpu, started = line.strip().split(None, 3)
            processes[int(pid)] = (int(parent), _seconds(cpu), started)
        parents = {value[0] for value in processes.values()}
        jobs[:] = [job for job in jobs if not (job.kind == 'tmux' and job.id in shells
                    and job.pid in processes and at_prompt.get(job.pid, job.pid not in parents))]
        for job in jobs:
            if not job.pid or job.state not in {'running', 'claimed'} or job.pid not in processes:
                continue
            tree = {job.pid}
            while True:
                children = {pid for pid, value in processes.items() if value[0] in tree}
                if children <= tree:
                    break
                tree |= children
            job.cpu = sum(processes[pid][1] for pid in tree)
            identities = [(pid, processes[pid][2]) for pid in sorted(tree)]
            try:
                proc = Path('/proc/self/stat').is_file()
                if proc:
                    # Include waited-for children: a compile can finish a short
                    # subprocess entirely between two samples.
                    ticks = os.sysconf('SC_CLK_TCK')
                    for pid in tree:
                        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                        job.cpu += (int(stat[13]) + int(stat[14])) / ticks
                        if pid == job.pid and job.host_start_time is not None and int(stat[19]) != job.host_start_time:
                            job.cpu = None
                            job.output = None
                            break
                    if job.cpu is None:
                        continue
                if job.output is None:
                    if not proc:
                        continue
                    # Linux wchar includes pipe output, even when Hermes keeps the
                    # rolling buffer only in memory.
                    output = []
                    for pid in sorted(tree):
                        fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
                        output.append(int(fields['wchar']))
                    job.output = digest(str((identities, output)))
                else:
                    job.output = digest(str((identities, job.output)))
            except FileNotFoundError:
                job.cpu = None  # a process exited during the probe
                job.output = None
            except (OSError, ValueError, KeyError) as error:
                failed(error)
                job.cpu = None
                job.output = None
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        failed(error)
