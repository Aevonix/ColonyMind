"""Read one saved-login password through Hermes' existing vault API.

Hermes has its own interpreter. Use the same configured interpreter selection as
the secrets CLI; no vault format, key handling or credential store is duplicated.
The password crosses an anonymous pipe only, never argv, files or diagnostics.
"""
from __future__ import annotations

import json
import subprocess


class SavedPasswordNotFound(LookupError):
    pass


class SavedPasswordUnavailable(RuntimeError):
    pass


_READ_PASSWORD = """
import contextlib, json, logging, os, sys
wire = sys.stdout
logging.disable(logging.CRITICAL)
status, password = 3, None
try:
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        from pathlib import Path
        from agent.vault_store import VaultStore
        home, handle = json.load(sys.stdin)
        store = VaultStore(base_dir=Path(home) / 'vault')
        meta = store.get_meta(handle)
        if meta is None or meta.kind != 'login':
            status = 4
        else:
            secret = store.resolve_secret(handle)
            try:
                value = secret.get('password')
                if isinstance(value, str) and value:
                    password, status = value, 0
                else:
                    status = 4
            finally:
                secret.clear()
except Exception:
    status = 3
if status == 0:
    wire.write(json.dumps({'password': password}))
sys.exit(status)
"""


def saved_password(handle: str) -> str:
    """Called only after the HTTP route checks the owner relay and hand session."""
    from colonymind.config import load_config
    from colonymind.secrets.cli import _hermes_python

    try:
        python = _hermes_python()
        if python is None:
            raise SavedPasswordUnavailable()
        done = subprocess.run(
            [str(python), "-B", "-c", _READ_PASSWORD],
            input=json.dumps([str(load_config().hermes_home), handle]),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15,
        )
        if done.returncode == 4:
            raise SavedPasswordNotFound()
        if done.returncode != 0:
            raise SavedPasswordUnavailable()
        value = json.loads(done.stdout)
        if (not isinstance(value, dict) or set(value) != {"password"}
                or not isinstance(value["password"], str) or not value["password"]):
            raise SavedPasswordUnavailable()
        return value["password"]
    except SavedPasswordNotFound:
        raise
    except Exception:
        # Never propagate upstream exceptions, stdout or stderr to logs/errors.
        raise SavedPasswordUnavailable() from None
