"""Keep full restores offline for the entire lifetime of serving readers."""
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def state_access(state_dir, *, restore=False):
    """Serving shares a lease; restore requires all serving processes to exit.

    Embedded readers must also close before restore and reopen afterwards.
    Never unlink this lock file: all processes must lock the same inode.
    """
    import fcntl

    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    if restore:
        # Also recognize sidecars started by older CLI versions, before leases.
        pid_file = state_dir / 'sidecar.pid'
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if pid <= 0:
                    raise ValueError()
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            except (ValueError, PermissionError):
                raise ValueError('Stop the sidecar before restoring; verify sidecar.pid') from None
            else:
                raise ValueError('Stop the sidecar before restoring, then restart it to reopen readers')
    with (state_dir / '.state-access.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, (fcntl.LOCK_EX if restore else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Stop the sidecar before restoring; restart readers after restore') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
