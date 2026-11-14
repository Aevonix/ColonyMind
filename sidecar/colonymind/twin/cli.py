"""``colonymind twin``: init | run | status | promote | reseed | query."""

from __future__ import annotations

import json
import shlex
import shutil
import signal
import sqlite3
import sys
import time
from pathlib import Path

from colonymind.twin import config as twin_config

_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def add_parser(sub) -> None:
    twin = sub.add_parser('twin', help='Live replica of this instance on a second machine (see docs/TWIN.md)')
    commands = twin.add_subparsers(dest='twin_command')
    init = commands.add_parser('init', help='Make this instance a primary (with a target) or a standby')
    init.add_argument('--role', choices=twin_config.ROLES, help='primary ships; standby applies and can be promoted')
    init.add_argument('--target', default='', help='primary: the standby instance directory, a path or ssh://[user@]host[:port]/path')
    init.add_argument('--external', action='append', default=None, metavar='PATH',
                      help='primary: a store outside the instance to replicate (default: the plugin turn outbox)')
    init.add_argument('--external-dir', default='', help='standby: where promotion places outside stores')
    init.add_argument('--max-lag', type=float, default=None, help='lag alarm threshold in seconds (default 60)')
    init.add_argument('--max-spool', type=int, default=None, help='spool cap in bytes before a re-seed (default 1 GiB)')
    init.add_argument('--interval', type=float, default=None, help='seconds between polls (default 0.5)')
    init.add_argument('--alarm-command', default=None, help='command run on each alarm raise/clear')
    init.add_argument('--ssh-command', default=None, help='ssh program and options for ssh:// targets')
    init.add_argument('--off', action='store_true', help='stop being a twin: remove the configuration, keep data')
    run = commands.add_parser('run', help='Ship (primary) or apply (standby) continuously')
    run.add_argument('--once', action='store_true', help='one tick or cycle, then exit')
    status = commands.add_parser('status', help='Role, lag, alarms; exit 1 when an alarm is raised')
    status.add_argument('--json', action='store_true')
    promote = commands.add_parser('promote', help='Standby: apply what has arrived, fence the primary, become primary')
    promote.add_argument('--force', action='store_true', help='promote even with stale or diverged stores')
    promote.add_argument('--timeout', type=float, default=30.0, help='seconds to wait for a running applier to stop')
    reseed = commands.add_parser('reseed', help='Primary: start a new generation; standby: ask the primary for one')
    reseed.add_argument('--reason', default='operator request')
    query = commands.add_parser('query', help='Standby: run a read-only query against one replica')
    query.add_argument('store', help='store name, e.g. turn-idempotency.db or @external/<file>')
    query.add_argument('sql')


def _home() -> Path:
    from colonymind.config import instance_home
    return Path(instance_home())


def run(args) -> int:
    command = getattr(args, 'twin_command', None)
    handlers = {'init': _init, 'run': _run, 'status': _status, 'promote': _promote, 'reseed': _reseed, 'query': _query}
    if command not in handlers:
        print('usage: colonymind twin {init,run,status,promote,reseed,query} (see docs/TWIN.md)', file=sys.stderr)
        return 2
    try:
        return handlers[command](args, _home())
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f'twin {command} failed: {error}', file=sys.stderr)
        return 2


def _init(args, home: Path) -> int:
    path = twin_config.config_path(home)
    if args.off:
        if path.exists():
            path.rename(path.with_name('twin.json.off'))
        print('twin: configuration removed; replicas and spool left in place under .twin')
        return 0
    if not args.role:
        raise ValueError('--role is required')
    previous = twin_config.load(home)
    config = twin_config.TwinConfig(role=args.role, target=args.target if args.role == 'primary' else '')
    if previous is not None:
        config.promoted = previous.promoted
    if args.role == 'primary' and args.external is not None:
        config.external = [str(Path(p).expanduser().resolve()) for p in args.external]
    if args.external_dir:
        config.external_dir = str(Path(args.external_dir).expanduser().resolve())
    for name, value in (('max_lag_seconds', args.max_lag), ('max_spool_bytes', args.max_spool),
                        ('interval', args.interval), ('ssh_command', args.ssh_command)):
        if value is not None:
            setattr(config, name, value)
    if args.alarm_command:
        config.alarm_command = shlex.split(args.alarm_command)
    if args.role == 'primary' and config.target and not config.target.startswith('ssh://'):
        target = Path(config.target.removeprefix('file://')).expanduser().resolve()
        mine = home.resolve()
        if target == mine or target.is_relative_to(mine) or mine.is_relative_to(target):
            raise ValueError('the target must be another instance directory, neither inside nor around this one')
        config.target = str(target)
    twin_config.save(home, config)
    if config.promoted:
        # A promoted instance takes no more shipments: drop what a racing mirror left behind.
        shutil.rmtree(twin_config.twin_dir(home) / 'inbox', ignore_errors=True)
    twin_config.Journal(home, config).event('init', role=config.role, target=config.target)
    print(f'twin: this instance is now a {config.role}' + (f' shipping to {config.target}' if config.target else ''))
    return 0


def _require(home: Path) -> twin_config.TwinConfig:
    config = twin_config.load(home)
    if config is None:
        raise ValueError('not configured: run `colonymind twin init` first')
    return config


def _run(args, home: Path) -> int:
    config = _require(home)
    if config.role == 'primary' and not config.target:
        print('twin: promoted primary with no standby attached; run `colonymind twin init --role primary --target ...`')
        return 0
    lock = twin_config.RunLock(home)
    if not lock.acquire():
        print('twin: another runner holds this instance', file=sys.stderr)
        return 2
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    request = twin_config.twin_dir(home) / 'promote.request'
    worker = None
    try:
        if config.role == 'primary':
            from colonymind.twin.primary import Shipper
            worker = Shipper(home, config)
            step = worker.tick
        else:
            from colonymind.twin.standby import Standby
            worker = Standby(home, config)
            step = worker.cycle
        while True:
            try:
                step()
            except Exception as error:     # one bad tick must not stop replication
                worker.journal.event('cycle-error', error=type(error).__name__, detail=str(error)[:200])
            if args.once or stopping or request.exists():
                break
            time.sleep(config.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None and hasattr(worker, 'close'):
            worker.close()
        lock.release()
    return 0


def _status(args, home: Path) -> int:
    config = _require(home)
    live = twin_config.read_json(twin_config.twin_dir(home) / 'status.json') or {}
    now = time.time()
    alarms = dict(live.get('alarms') or {})
    age = now - float(live.get('at') or 0)
    if config.target or config.role == 'standby':
        if not live or age > max(10 * config.interval, 30):
            alarms['runner_down'] = 'no runner has reported for %.0fs' % age if live else 'no runner has reported'
    report = {**{k: v for k, v in live.items() if k not in ('alarms', 'role', 'promoted')},
              'role': config.role, 'target': config.target, 'promoted': config.promoted or None,
              'status_age_seconds': round(age, 1) if live else None, 'alarms': alarms}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"role: {config.role}" + (f" -> {config.target}" if config.target else ''))
        if config.promoted:
            print(f"promoted at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(config.promoted['at']))}")
        for name in ('generation', 'seq', 'acked_seq', 'lag_seconds', 'spool_bytes', 'fenced'):
            if name in live:
                print(f'{name}: {live[name]}')
        if live.get('stale'):
            print(f"stale stores: {', '.join(live['stale'])}")
        for name, detail in sorted(alarms.items()):
            print(f'ALARM {name}: {detail}')
    return 1 if alarms else 0


def _promote(args, home: Path) -> int:
    from colonymind.twin.standby import PromoteError, Standby
    config = _require(home)
    if config.role != 'standby':
        raise ValueError('only a standby can be promoted')
    request = twin_config.twin_dir(home) / 'promote.request'
    lock = twin_config.RunLock(home)
    request.touch()
    try:
        if not lock.acquire(timeout=args.timeout):
            print('twin: the running applier did not stop in time', file=sys.stderr)
            return 2
        try:
            report = Standby(home, config).promote(force=args.force)
        except PromoteError as error:
            print(f'twin promote refused: {error}\n(use --force to promote anyway)', file=sys.stderr)
            return 1
        finally:
            lock.release()
    finally:
        request.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print('Promoted. Start the sidecar on this machine (`colonymind service start`), point the body at it, '
          'and stop the old primary if it is still running.')
    return 0


def _reseed(args, home: Path) -> int:
    config = _require(home)
    if config.role == 'standby':
        from colonymind.twin.standby import Standby
        standby = Standby(home, config)
        standby.state['reseed'] = None
        standby.request_reseed(args.reason)
        print('twin: re-seed requested; the primary starts a new generation when it next reads the ack')
        return 0
    request = twin_config.twin_dir(home) / 'reseed.request'
    request.write_text(args.reason[:200] + '\n')
    print('twin: the running shipper starts a new generation on its next tick')
    return 0


def _query(args, home: Path) -> int:
    from colonymind.twin.standby import connect
    conn = connect(home, args.store)
    conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY)
    try:
        cursor = conn.execute(args.sql)
        columns = [d[0] for d in cursor.description or ()]
        for row in cursor:
            print(json.dumps(dict(zip(columns, row)), default=lambda v: v.hex() if isinstance(v, bytes) else str(v)))
    finally:
        conn.close()
    return 0
