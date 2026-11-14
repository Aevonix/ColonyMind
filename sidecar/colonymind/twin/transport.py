"""Moving the outbox to the standby: a directory mirror plus reading the standby's ack.

Every file in the outbox is immutable and uniquely named (segments and seeds by their digest,
tree blobs by content), so mirroring is "copy what is missing, delete what is gone". A local
target (a path, or a mounted share) is mirrored in Python; ``ssh://[user@]host[:port]/path``
uses ``rsync`` over ``ssh`` (an external program, invoked and never linked). Either way the
standby's ``inbox`` is written only by the primary and read only by the standby, and the
standby's ``ack.json`` is written only by the standby: the twin never writes the primary's
stores and the primary never writes the standby's replicas.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from colonymind.twin.config import PROMOTED, TWIN_DIR, read_json
from colonymind.twin.segment import fsync_dir


class TransportError(RuntimeError):
    pass


def _visible(root: Path):
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in files:
            if not name.startswith('.'):
                path = Path(directory) / name
                yield path.relative_to(root).as_posix(), path


class LocalTransport:
    def __init__(self, target_home: Path):
        self.target = Path(target_home)
        self.inbox = self.target / TWIN_DIR / 'inbox'

    def describe(self) -> str:
        return 'local'

    def mirror(self, outbox: Path) -> int:
        if not (self.target / TWIN_DIR).is_dir():
            raise TransportError('the standby is not initialised (no .twin directory at the target)')
        if (self.target / TWIN_DIR / PROMOTED).exists():
            raise TransportError('the standby was promoted; nothing more is shipped to it')
        self.inbox.mkdir(parents=True, exist_ok=True)
        wanted, copied = {}, 0
        for relative, path in _visible(outbox):
            wanted[relative] = path
        for relative, path in wanted.items():
            destination = self.inbox / relative
            try:
                if destination.stat().st_size == path.stat().st_size:
                    continue
            except FileNotFoundError:
                pass
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.part')
            try:
                shutil.copyfile(path, temporary)
            except FileNotFoundError:
                continue      # acknowledged and removed while mirroring
            with open(temporary, 'rb+') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            fsync_dir(destination.parent)
            copied += destination.stat().st_size
        for relative, path in list(_visible(self.inbox)):
            if relative not in wanted:
                path.unlink(missing_ok=True)
        for directory, _, _ in sorted(os.walk(self.inbox), key=lambda item: -len(item[0])):
            if Path(directory) != self.inbox and not (outbox / Path(directory).relative_to(self.inbox)).is_dir():
                try:
                    Path(directory).rmdir()
                except OSError:
                    pass
        return copied

    def read_ack(self) -> dict | None:
        return read_json(self.target / TWIN_DIR / 'ack.json')


class RsyncTransport:
    """``rsync`` over ``ssh``; one persistent ssh connection is reused between calls."""

    def __init__(self, spec: str, *, ssh_command: str = 'ssh', control_dir: Path | None = None, timeout: float = 120):
        parsed = urlparse(spec)
        if parsed.scheme != 'ssh' or not parsed.hostname or not parsed.path.startswith('/'):
            raise ValueError('an ssh target is ssh://[user@]host[:port]/absolute/path')
        self.host = (f'{parsed.username}@' if parsed.username else '') + parsed.hostname
        self.path = parsed.path.rstrip('/')
        self.timeout = timeout
        ssh = shlex.split(ssh_command) + ['-o', 'BatchMode=yes']
        if parsed.port:
            ssh += ['-p', str(parsed.port)]
        if control_dir is not None and len(str(control_dir)) < 80:
            control_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            ssh += ['-o', 'ControlMaster=auto', '-o', f'ControlPath={control_dir}/%C', '-o', 'ControlPersist=60']
        self.shell = shlex.join(ssh)

    def describe(self) -> str:
        return f'rsync over ssh to {self.host}'

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(['rsync', '-e', self.shell, *args], capture_output=True, text=True,
                                    timeout=self.timeout, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as error:
            raise TransportError(f'rsync failed to run: {type(error).__name__}') from None
        return result

    def mirror(self, outbox: Path) -> int:
        inbox = f'{self.path}/{TWIN_DIR}/inbox/'
        result = self._run(['-rt', '--delete', '--delay-updates', '--exclude=.*', '--stats',
                            f'--rsync-path=test -d {shlex.quote(self.path)}/{TWIN_DIR} && '
                            f'test ! -e {shlex.quote(self.path)}/{TWIN_DIR}/{PROMOTED} && '
                            f'mkdir -p {shlex.quote(inbox)} && rsync',
                            f'{outbox}/', f'{self.host}:{shlex.quote(inbox)}'])
        # 24: source files vanished during the transfer. The spool is pruned while a transfer
        # runs (acknowledged segments, superseded generations and snapshots); what vanished is
        # no longer wanted, and everything else was transferred.
        if result.returncode not in (0, 24):
            raise TransportError(f'rsync exited {result.returncode}: {result.stderr.strip()[-200:]}')
        for line in result.stdout.splitlines():
            if line.startswith('Total transferred file size:'):
                digits = ''.join(ch for ch in line.split(':', 1)[1] if ch.isdigit())
                return int(digits or 0)
        return 0

    def read_ack(self) -> dict | None:
        with tempfile.TemporaryDirectory(prefix='ptwin-ack-') as tmp:
            local = Path(tmp) / 'ack.json'
            result = self._run(['-t', f'{self.host}:{shlex.quote(self.path + "/" + TWIN_DIR + "/ack.json")}', str(local)])
            if result.returncode != 0:
                if 'No such file' in result.stderr:
                    return None
                raise TransportError(f'rsync exited {result.returncode} reading the ack')
            return read_json(local)


def for_target(target: str, *, ssh_command: str = 'ssh', control_dir: Path | None = None):
    if target.startswith('ssh://'):
        return RsyncTransport(target, ssh_command=ssh_command, control_dir=control_dir)
    if target.startswith('file://'):
        target = urlparse(target).path
    return LocalTransport(Path(target).expanduser())
