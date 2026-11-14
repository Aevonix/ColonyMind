"""Watchdog diagnostics contain code locations and one warning per stall."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
import logging
import sys
import time

import pytest

from colonymind import health_counters, loop_lag


@pytest.mark.asyncio
async def test_blocking_callback_records_one_stall_without_values(caplog):
    before = health_counters.snapshot().get('loop_stalls', 0)
    location = []

    def blocking_callback():
        private_value = 'private-callback-value'
        location.append(f'{__file__}:{sys._getframe().f_lineno + 1}:blocking_callback')
        time.sleep(1)
        assert private_value

    with caplog.at_level(logging.INFO, logger=loop_lag.__name__):
        async with loop_lag.loop_lag_lifespan() as recorder:
            assert recorder._thread.daemon
            await asyncio.sleep(.15)
            asyncio.get_running_loop().call_soon(blocking_callback)
            await asyncio.sleep(1.2)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING
                and record.name == loop_lag.__name__]
    assert len(warnings) == 1
    assert location[0] in warnings[0].getMessage()
    assert 'private-callback-value' not in caplog.text
    recovered = [record for record in caplog.records if 'Event loop recovered' in record.getMessage()]
    assert len(recovered) == 1
    assert .9 <= recovered[0].args[0] <= 1.3
    assert health_counters.snapshot().get('loop_stalls', 0) == before + 1
    assert not recorder._thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize('startup_fails', [False, True])
async def test_server_stops_watchdog_even_when_startup_fails(monkeypatch, tmp_path, startup_fails):
    from colonymind import server

    recorders = []
    original = loop_lag.LoopLagRecorder

    def recorder(loop):
        result = original(loop)
        recorders.append(result)
        return result

    @asynccontextmanager
    async def subsystem(app):
        assert recorders[0]._thread.is_alive()
        if startup_fails:
            raise RuntimeError('startup failed')
        yield

    monkeypatch.setattr(loop_lag, 'LoopLagRecorder', recorder)
    monkeypatch.setattr(server, '_lifespan', subsystem)
    monkeypatch.setattr(server, '_state_dir', lambda: tmp_path)
    monkeypatch.setattr('colonymind.util.state_access.state_access', lambda path: nullcontext())
    if startup_fails:
        with pytest.raises(RuntimeError, match='startup failed'):
            async with server.lifespan(None):
                pytest.fail('startup should fail')
    else:
        async with server.lifespan(None):
            assert recorders[0]._thread.is_alive()
    assert not recorders[0]._thread.is_alive()
