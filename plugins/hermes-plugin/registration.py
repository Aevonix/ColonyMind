"""Private, atomic evidence of this process's successful hook registration."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time

logger = logging.getLogger(__name__)
REGISTRATION_FILE = "hermes-plugin-registration.json"


def process_start() -> float:
    """Match the checker's ps lstart epoch on both supported host platforms."""
    result = subprocess.run(
        ["/bin/ps", "-p", str(os.getpid()), "-o", "lstart="],
        env={**os.environ, "LC_ALL": "C"}, capture_output=True, text=True, check=True, timeout=2,
    )
    _, month, day, clock, year = result.stdout.split()
    month_number = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec").index(month) + 1
    hour, minute, second = map(int, clock.split(":"))
    return time.mktime((int(year), month_number, int(day), hour, minute, second, -1, -1, -1))


class RegistrationEvidence:
    """Forward the public context while recording hooks actually registered."""

    def __init__(self, context):
        self.context = context
        self.hooks: set[str] = set()

    def __getattr__(self, name):
        return getattr(self.context, name)

    def register_hook(self, name, callback):
        result = self.context.register_hook(name, callback)
        # Hermes acknowledges an accepted registration with a lease. An
        # abandoned plugin load returns None and must not publish evidence.
        if result is not None:
            self.hooks.add(name)
        return result

    def write(self, home: Path) -> None:
        from . import guard
        if "pre_tool_call" not in self.hooks:
            logger.warning("registration_evidence_unavailable reason=guard_hook_unregistered")
            return
        temporary = None
        try:
            record = dict(schema=1, pid=os.getpid(), process_start=process_start(),
                          module_path=str(Path(guard.__file__).resolve()), interpreter=sys.executable,
                          hooks=sorted(self.hooks), written_at=time.time())
            fd, temporary = tempfile.mkstemp(prefix=".hermes-plugin-registration-", dir=home)
            with os.fdopen(fd, "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                if os.fstat(stream.fileno()).st_nlink != 1:
                    raise OSError("registration link count")
                json.dump(record, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, Path(home) / REGISTRATION_FILE)
            temporary = None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            logger.warning("registration_evidence_unavailable exception_type=%s", type(error).__name__)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError as error:
                    logger.warning("registration_cleanup_unavailable exception_type=%s", type(error).__name__)
