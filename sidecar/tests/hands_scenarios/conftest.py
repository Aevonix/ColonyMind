"""Isolated fixture imports and explicit integrator-only socket/live markers."""

import os
from pathlib import Path
import socket
import sys

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "hermes_adapter"))

from .support import runtime_factory  # noqa: E402,F401


def pytest_configure(config):
    config.addinivalue_line("markers", "hands_mutation: disables a named production guard")
    config.addinivalue_line("markers", "hands_sockets: integrator must run with local socket pairs enabled")
    config.addinivalue_line("markers", "hands_clone: integrator must supply an isolated live Hermes clone driver")


def pytest_collection_modifyitems(items):
    if os.environ.get("HANDS_TEST_POLL_LOOP"):
        return
    candidates = [item for item in items if item.get_closest_marker("hands_sockets")]
    if not candidates:
        return
    pair = ()
    try:
        pair = socket.socketpair()
        pair[0].send(b"x")
        assert pair[1].recv(1) == b"x"
    except OSError:
        marker = pytest.mark.skip(reason="Integrator: local socket pairs required by asyncio/aiosqlite are denied")
        for item in candidates:
            item.add_marker(marker)
    finally:
        for connection in pair:
            connection.close()
