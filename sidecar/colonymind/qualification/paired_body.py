"""The body tick and the capture outbox, identical in every arm.

One tick runs the arm's own step first (the ColonyMind tick in plugin arms,
making the heartbeat job due or the curator pass in the comparator arms, nothing
in base), then Hermes cron ``tick()``, then kanban ``dispatch_once`` with the
ready workers run in-process and awaited up to a bound. ``advance_clock`` shifts the two wall
clocks Hermes reads, faketime-style, without touching monotonic clocks; every "now" in the sidecar
and the plugins reads ``time.time`` (``temporal.now_utc``) and moves with them. A generated
family also pins the clock's start (``start_offset``): every episode of every arm begins
at the same UTC time of day.

``set_clock`` is the same clock for a history replayed on an absolute timeline (``benchmarks/public_memory``, and a
paired family's history in the replay mode, ``paired_history.replay_ledger``): it puts "now" at a given time, back
once before the replay's state exists and only forward after that, and holds it there until the next move, so the
replayed timeline never depends on how long processing takes; ``run_clock_from`` lets it run again.
"""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

PROTOCOL = 'paired-body-tick-1'
PLUGIN = 'capture'
OWNER = 'owner'
DEFAULT_WORKER_WAIT_SECONDS = 120
MAX_SPAWN_PER_TICK = 8
MAX_SNAPSHOT_TASKS = 64
TEXT_BOUND = 2048
DISPATCH_FIELDS = ('spawned', 'promoted', 'reclaimed', 'crashed', 'timed_out', 'stale', 'auto_blocked',
                   'skipped_unassigned', 'skipped_nonspawnable', 'respawn_guarded')
_ORIGINAL = {}
_OFFSET = [0.0]
# The replay clock's time while it stands (``set_clock``); None while the clock runs with real time.
_STANDING = [None]
# A generated family starts every episode's body clock at one UTC time of day, in every arm
# (paired_cases declares it, the plan records it): day-boundary rules such as the mind's
# nightly consolidation then fire, or not, by the scenario's clock advances alone, never by
# the hour a container happened to start. Forward only: the clock never runs back.
CLOCK_START_PROTOCOL = 'paired-clock-start-1'
CLOCK_STARTS = ('12:00',)


def plugin_source():
    return Path(__file__).resolve().parents[3] / 'benchmarks' / 'paired' / 'capture_platform'


def capture_module(source=None):
    """The plugin's own outbox writer, loaded from its directory."""
    source = Path(source or plugin_source())
    spec = importlib.util.spec_from_file_location('paired_capture_platform', source / '__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_capture_platform(home, config, outbox, *, source=None):
    """Copy the plugin into the isolated profile and enable it; same config in every arm."""
    source = Path(source or plugin_source())
    target = Path(home) / 'plugins' / PLUGIN
    if not target.exists():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.md'))
    module = capture_module(source)
    # An episode that never delivers is observed as an empty outbox, not a missing
    # one; a restarted phase keeps the entries already recorded.
    if not Path(outbox).exists():
        Path(outbox).write_text('[]')
    os.environ[module.OUTBOX_ENV] = str(outbox)
    os.environ[module.HOME_CHANNEL_ENV] = OWNER
    plugins = config.setdefault('plugins', {})
    enabled = plugins.setdefault('enabled', [])
    if PLUGIN not in enabled:
        enabled.append(PLUGIN)
    config['platforms'] = {**config.get('platforms', {}), PLUGIN: {'enabled': True}}
    # Unassigned model-created tasks dispatch to the only profile that exists,
    # and delivered text is the agent's own message, not a cron wrapper.
    config['kanban'] = {**config.get('kanban', {}), 'default_assignee': 'default'}
    config['cron'] = {**config.get('cron', {}), 'wrap_response': False}
    # Cron jobs run on the episode's configured endpoint: a job's creation
    # snapshot collapses a named provider to bare "custom", which has no URL here.
    provider = config.get('model', {}).get('provider')
    if provider:
        config['cron'].update(model_provider=provider, model=config['model']['default'])
    return module


def read_outbox(path):
    """The outbox array, or None when it is missing or not an array."""
    try:
        rows = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return rows if isinstance(rows, list) else None


# A module first imported while the clock is installed can bind the shifted function itself (a default argument,
# say): after ``uninstall_clock`` it reads the real clock, which ``time.time`` and ``hermes_time.now`` are again.

def _shifted_time():
    if 'time' not in _ORIGINAL:
        return time.time()
    standing = _STANDING[0]
    return standing if standing is not None else _ORIGINAL['time']() + _OFFSET[0]


def _shifted_now():
    if 'now' not in _ORIGINAL:
        import hermes_time
        return hermes_time.now()
    standing = _STANDING[0]
    if standing is not None:
        return datetime.fromtimestamp(standing, _ORIGINAL['now']().tzinfo)
    return _ORIGINAL['now']() + timedelta(seconds=_OFFSET[0])


def _rebind_clock_aliases():
    # Modules bind ``from hermes_time import now as _hermes_now`` at import;
    # anything imported after install already sees the shifted function.
    if 'now' not in _ORIGINAL:
        return
    for module in list(sys.modules.values()):
        for name, value in list(getattr(module, '__dict__', {}).items()):
            if value is _ORIGINAL['now']:
                setattr(module, name, _shifted_now)


def _install(*, hermes):
    """Shift ``time.time``, and ``hermes_time.now`` where Hermes is installed (``hermes`` requires it); once."""
    if 'time' in _ORIGINAL:
        return
    try:
        import hermes_time
    except ImportError:
        if hermes:
            raise
        hermes_time = None
    _ORIGINAL['time'] = time.time
    time.time = _shifted_time
    if hermes_time is not None:
        _ORIGINAL['now'] = hermes_time.now
        hermes_time.now = _shifted_now


def install_clock(offset_seconds=0):
    """Shift ``time.time`` (kanban timestamps, claims) and ``hermes_time.now`` (cron due times)."""
    _install(hermes=True)
    _STANDING[0] = None
    _OFFSET[0] = float(offset_seconds)
    _rebind_clock_aliases()
    return _OFFSET[0]


class ClockRefused(RuntimeError):
    """A move of the replay clock its rules refuse (``set_clock``)."""


# The replay whose state the clock serves, and whether its one move back was taken.
_REPLAY = {'state': None, 'rewound': False}


def _holds_anything(state):
    return state.exists() and any(state.iterdir())


def _moved_to(epoch):
    _STANDING[0] = epoch
    _OFFSET[0] = epoch - _ORIGINAL['time']()
    _rebind_clock_aliases()
    return epoch


def _replay_state(state):
    state = Path(state).resolve()
    if _REPLAY['state'] != state:
        _REPLAY.update(state=state, rewound=False)
    return state


def set_clock(epoch, *, state):
    """Put the wall clock (``time.time``, and Hermes' where it is installed) at ``epoch``, seconds since the epoch,
    for the replay whose state is the directory ``state``, and hold it there; returns ``epoch``.

    The clock stands between moves: processing a replayed turn, a drain or a night takes real time but no replayed
    time, so where the timeline is never depends on how fast the model answered. Waits and deadlines run on the
    monotonic clock, which this never touches; a retry the sidecar schedules on the wall clock waits for the next
    move.

    A history replayed on its own dates starts by moving the clock back to its first moment. That move is allowed
    once, and only while ``state`` holds nothing: no store and no credential exists yet. A store stamped on one
    timeline and then read on an earlier one would see its own rows in the future, and a credential would be
    issued after the time it is used. After it the clock moves only forward: an ``epoch`` at or before the clock's
    own time raises ``ClockRefused`` (``clock_at_least`` never moves back). Installs the clock first, as the
    paired worker does, and needs no Hermes."""
    epoch = float(epoch)
    if not math.isfinite(epoch):
        raise ValueError('The clock needs a finite time')
    _install(hermes=False)
    state = _replay_state(state)
    if epoch <= _shifted_time():
        if _REPLAY['rewound'] or _holds_anything(state):
            raise ClockRefused('The replay clock moves back only once, before its state exists')
        _REPLAY['rewound'] = True
    return _moved_to(epoch)


def clock_at_least(epoch, *, state):
    """Move the clock forward to ``epoch`` when it is earlier, and never back; returns the clock's time, where it
    stands (``set_clock``)."""
    epoch = float(epoch)
    if not math.isfinite(epoch):
        raise ValueError('The clock needs a finite time')
    _install(hermes=False)
    _replay_state(state)
    now = _shifted_time()
    return _moved_to(epoch) if epoch > now else _moved_to(now)


def run_clock_from(epoch):
    """Let the wall clock run again, from ``epoch`` (after a replayed history: the episode's pinned start); forward
    only, since the store holds stamps up to the clock's own time. Returns ``epoch``."""
    epoch = float(epoch)
    if not math.isfinite(epoch):
        raise ValueError('The clock needs a finite time')
    _install(hermes=False)
    if epoch < _shifted_time():
        raise ClockRefused('The clock runs again only from its own time or later')
    _STANDING[0] = None
    _OFFSET[0] = epoch - _ORIGINAL['time']()
    _rebind_clock_aliases()
    return epoch


def start_offset(clock_start, now=None):
    """Seconds from the real clock (``now``, epoch seconds) forward to the next ``clock_start`` UTC."""
    if clock_start not in CLOCK_STARTS:
        raise ValueError('Unknown clock start')
    hour, minute = (int(part) for part in clock_start.split(':'))
    real = datetime.fromtimestamp(_ORIGINAL.get('time', time.time)() if now is None else now, timezone.utc)
    start = real.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if start < real:
        start += timedelta(days=1)
    return (start - real).total_seconds()


def advance_clock(seconds):
    if 'time' not in _ORIGINAL:
        raise RuntimeError('Clock is not installed')
    if _STANDING[0] is not None:
        _STANDING[0] += float(seconds)
    _OFFSET[0] += float(seconds)
    _rebind_clock_aliases()
    return _OFFSET[0]


def clock_offset():
    return _OFFSET[0]


def uninstall_clock():
    if 'time' in _ORIGINAL:
        time.time = _ORIGINAL.pop('time')
    if 'now' in _ORIGINAL:
        import hermes_time
        hermes_time.now = _ORIGINAL.pop('now')
    _OFFSET[0] = 0.0
    _STANDING[0] = None
    _REPLAY.update(state=None, rewound=False)


def _colonymind_entry(name):
    """A callable the loaded ColonyMind plugin module defines; None in every other arm."""
    try:
        from hermes_cli.plugins import get_plugin_manager
        loaded = get_plugin_manager()._plugins.get('colonymind')
    except Exception:
        return None
    if loaded is None or not loaded.enabled:
        return None
    entry = getattr(loaded.module, name, None)
    return entry if callable(entry) else None


def colonymind_tick_entry():
    """The loaded ColonyMind plugin's ``tick`` when it defines one; None in every other arm."""
    return _colonymind_entry('tick')


def colonymind_flush_entry():
    """The adapter's ``flush``: deliver captured turns now, as its body thread would shortly."""
    return _colonymind_entry('flush')


def _bounded(text):
    text = '' if text is None else str(text)
    return text[:TEXT_BOUND]


def kanban_snapshot(conn):
    from hermes_cli import kanban_db
    rows = []
    for task in kanban_db.list_tasks(conn, include_archived=True):
        rows.append({'id': task.id, 'title': _bounded(task.title), 'body': _bounded(task.body),
                     'status': task.status, 'assignee': task.assignee, 'created_by': task.created_by,
                     'created_at': task.created_at, 'completed_at': task.completed_at,
                     'result': _bounded(task.result)})
    rows.sort(key=lambda row: (row['created_at'] or 0, row['id']))
    return rows[:MAX_SNAPSHOT_TASKS]


def run_tick(*, outbox, arm_tick=None, run_task=None, wait_seconds=DEFAULT_WORKER_WAIT_SECONDS):
    """One body tick. ``arm_tick()`` is the arm's own step; ``run_task(task, workspace, seconds)``
    runs a claimed task in-process."""
    from cron.scheduler import tick as cron_tick
    from hermes_cli import kanban_db_connect, kanban_db_dispatch
    before = read_outbox(outbox)
    row = {'outbox_before': len(before or []), 'arm_tick': None, 'workers': []}
    queued = []

    def spawn(task, workspace, board=None):
        # No worker pid: the task runs in this process, and Hermes signals a recorded
        # pid when it reclaims an expired claim. Without one, a task a worker left
        # running is released on expiry the way a vanished worker's would be.
        queued.append((task, workspace))
        return None

    with kanban_db_connect.connect_closing() as conn:
        # Tasks created by the arm's step or by a cron job count as this tick's effects
        # and are dispatched by this same tick.
        existing = {task['id'] for task in kanban_snapshot(conn)}
        if arm_tick is not None:
            row['arm_tick'] = arm_tick()
        row['cron_jobs_run'] = int(cron_tick(verbose=False, sync=True))
        result = kanban_db_dispatch.dispatch_once(conn, spawn_fn=spawn, max_spawn=MAX_SPAWN_PER_TICK,
                                                  default_assignee='default')
        row['dispatch'] = {name: len(getattr(result, name)) if isinstance(getattr(result, name), list)
                           else getattr(result, name) for name in DISPATCH_FIELDS}
        deadline = time.monotonic() + wait_seconds
        for task, workspace in queued:
            remaining = max(0.0, deadline - time.monotonic())
            outcome = run_task(task, workspace, remaining) if run_task is not None else 'no_worker'
            row['workers'].append({'task_id': task.id, 'outcome': outcome})
        row['kanban'] = kanban_snapshot(conn)
    row['created_task_ids'] = [task['id'] for task in row['kanban'] if task['id'] not in existing]
    row['outbox_after'] = len(read_outbox(outbox) or [])
    return row
