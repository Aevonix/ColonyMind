"""Apply the existing offline polling mode to adapter-test event loops."""
import asyncio
import os

_original = None


def _polling_run_forever(loop):
    selector = getattr(loop, '_selector', None)
    if selector is None:
        return _original(loop)
    original_select = selector.select

    def select(timeout=None):
        # Match sidecar/tests/conftest.py, including tests that freeze monotonic time.
        return original_select(0.001 if timeout is None else min(timeout, 0.001))

    selector.select = select
    try:
        return _original(loop)
    finally:
        selector.select = original_select


def pytest_configure(config):
    global _original
    if os.environ.get('COLONYMIND_TEST_POLL_LOOP') == '1' and asyncio.BaseEventLoop.run_forever is not _polling_run_forever:
        _original = asyncio.BaseEventLoop.run_forever
        asyncio.BaseEventLoop.run_forever = _polling_run_forever


def pytest_report_header(config):
    if asyncio.BaseEventLoop.run_forever is _polling_run_forever:
        return 'Qualification environment: adapter asyncio loops poll every 1 ms; socket restrictions remain active'


def pytest_unconfigure(config):
    if asyncio.BaseEventLoop.run_forever is _polling_run_forever:
        asyncio.BaseEventLoop.run_forever = _original
