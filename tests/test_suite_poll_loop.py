"""A blocked socketpair wakeup must not strand adapter fixtures in offline mode."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('enabled', [False, True])
def test_cross_thread_work_with_a_blocked_wakeup(enabled):
    program = '''
import asyncio, concurrent.futures, threading
from scripts import suite_poll_loop
suite_poll_loop.pytest_configure(None)
loop = asyncio.new_event_loop()
loop._write_to_self = lambda: None
parked = threading.Event()
original_select = loop._selector.select
def select(timeout=None):
    if timeout is None or timeout > 0:
        parked.set()
    return original_select(timeout)
loop._selector.select = select
thread = threading.Thread(target=loop.run_forever, daemon=True)
thread.start()
assert parked.wait(2)
async def answer():
    return 42
try:
    assert asyncio.run_coroutine_threadsafe(answer(), loop).result(1) == 42
except concurrent.futures.TimeoutError:
    print('cross_thread_timeout', flush=True)
    raise SystemExit(1)
loop.call_soon_threadsafe(loop.stop)
thread.join(2)
assert not thread.is_alive()
loop.close()
suite_poll_loop.pytest_unconfigure(None)
print('cross_thread_complete', flush=True)
'''
    result = subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT,
                            env={**os.environ, 'COLONYMIND_TEST_POLL_LOOP': '1' if enabled else '0'},
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == (0 if enabled else 1), result.stdout + result.stderr
    assert ('cross_thread_complete' if enabled else 'cross_thread_timeout') in result.stdout
