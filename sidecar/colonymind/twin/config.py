"""The twin's configuration, audit log, status file and alarms, all under ``<home>/.twin``."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

TWIN_DIR = '.twin'
PROMOTED = 'promoted'      # marker in the twin directory: this standby became a primary
ROLES = ('primary', 'standby')
EVENTS_MAX_BYTES = 8 << 20


@dataclass
class TwinConfig:
    role: str
    target: str = ''                  # primary: the standby's instance directory, local or ssh://
    external: list[str] = field(default_factory=list)   # primary: stores outside the instance (the turn outbox)
    external_dir: str = ''            # standby: where promotion puts those stores (the plugin's state directory)
    interval: float = 0.5             # seconds between polls of every store
    push_interval: float = 1.0        # seconds between transfers to the standby
    transfer_wait: float = 0.25       # longest a tick waits for a transfer; the rest runs on its own thread
    heartbeat_interval: float = 5.0   # an empty segment when nothing changed, so lag stays measurable
    max_lag_seconds: float = 60.0     # the lag alarm
    max_spool_bytes: int = 1 << 30    # unacknowledged segments beyond this: drop them and re-seed
    min_free_bytes: int = 512 << 20   # stop spooling (and re-seed later) before the disk fills
    release_bytes: int = 16 << 20     # a log this long is released so the application can restart it
    anti_entropy_interval: float = 900.0
    vector_interval: float = 300.0
    files_interval: float = 300.0
    rediscover_interval: float = 30.0
    ssh_command: str = 'ssh'
    alarm_command: list[str] = field(default_factory=list)
    keep_local: list[str] = field(default_factory=lambda: ['colonymind.yaml'])
    promoted: dict = field(default_factory=dict)

    def validate(self) -> 'TwinConfig':
        if self.role not in ROLES:
            raise ValueError(f'role must be one of {ROLES}')
        if self.role == 'primary' and not self.target and not self.promoted:
            raise ValueError('a primary needs a target: the standby instance directory')
        for name in ('interval', 'push_interval', 'heartbeat_interval', 'max_lag_seconds'):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')
        if float(self.transfer_wait) < 0:
            raise ValueError('transfer_wait must not be negative')
        if int(self.max_spool_bytes) < 1 << 20:
            raise ValueError('max_spool_bytes must be at least 1 MiB')
        return self


def twin_dir(home: Path) -> Path:
    return Path(home) / TWIN_DIR


def config_path(home: Path) -> Path:
    return twin_dir(home) / 'twin.json'


def load(home: Path) -> TwinConfig | None:
    path = config_path(home)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    known = {f.name for f in fields(TwinConfig)}
    return TwinConfig(**{k: v for k, v in raw.items() if k in known}).validate()


def save(home: Path, config: TwinConfig) -> None:
    write_json(config_path(home), asdict(config.validate()))


def write_json(path: Path, value, *, durable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        if durable:
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


class Journal:
    """Append-only audit log (``events.jsonl``) and the alarm set, with an optional hook."""

    def __init__(self, home: Path, config: TwinConfig, *, clock=time.time):
        self.path = twin_dir(home) / 'events.jsonl'
        self.config, self.clock = config, clock
        self.alarms: dict[str, str] = {}

    def event(self, name: str, /, **detail) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            if self.path.stat().st_size > EVENTS_MAX_BYTES:
                os.replace(self.path, self.path.with_suffix('.jsonl.1'))
        except FileNotFoundError:
            pass
        line = json.dumps({**detail, 'at': round(self.clock(), 3), 'event': name}, sort_keys=True)
        with open(self.path, 'a') as handle:
            handle.write(line + '\n')

    def alarm(self, name: str, raised: bool, detail: str = '') -> None:
        """Raise or clear one alarm; transitions are logged and run ``alarm_command``."""
        was = name in self.alarms
        if raised:
            self.alarms[name] = detail
        else:
            self.alarms.pop(name, None)
        if was == raised:
            return
        self.event('alarm', name=name, state='raised' if raised else 'cleared', detail=detail)
        if self.config.alarm_command:
            env = {**os.environ, 'COLONYMIND_TWIN_ALARM': name,
                   'COLONYMIND_TWIN_STATE': 'raised' if raised else 'cleared', 'COLONYMIND_TWIN_DETAIL': detail}
            try:
                subprocess.run(self.config.alarm_command, env=env, timeout=10, check=False,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError) as error:
                self.event('alarm-hook-failed', name=name, error=type(error).__name__)


class RunLock:
    """One runner per twin directory; ``promote`` takes the same lock."""

    def __init__(self, home: Path):
        self.path = twin_dir(home) / 'run.lock'
        self.handle = None

    def acquire(self, timeout: float = 0.0) -> bool:
        import fcntl                 # POSIX only; imported here so `colonymind --help` never needs it
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.handle = open(self.path, 'a+')
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        if self.handle is not None:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
