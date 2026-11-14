"""Record event-loop stalls from a watchdog thread, without retaining values."""

import asyncio
from contextlib import asynccontextmanager
import logging
import sys
import threading
import time

from colonymind.health_counters import bump

logger = logging.getLogger(__name__)


class LoopLagRecorder:
    """Keep one heartbeat pending and capture only the loop's code locations."""

    def __init__(self, loop):
        self.loop = loop
        self.loop_thread = threading.get_ident()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='loop-lag', daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _stack(self):
        frame = sys._current_frames().get(self.loop_thread)
        locations = []
        while frame is not None:
            locations.append(f'{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}')
            frame = frame.f_back
        return ' <- '.join(locations)

    def _run(self):
        while not self._stop.is_set():
            acknowledged = threading.Event()
            ended = []

            def heartbeat():
                ended.append(time.monotonic())
                acknowledged.set()

            started = time.monotonic()
            try:
                self.loop.call_soon_threadsafe(heartbeat)
            except RuntimeError as error:
                if not self._stop.is_set():
                    logger.warning('Loop heartbeat unavailable: %s', type(error).__name__)
                    bump('loop_stalls')
                return
            stalled = False
            while not acknowledged.is_set():
                if self._stop.wait(.1):
                    return
                lag = time.monotonic() - started
                if not acknowledged.is_set() and lag >= .5 and not stalled:
                    stalled = True
                    bump('loop_stalls')
                    logger.warning('Event loop stalled lag_seconds=%.3f stack=%s', lag, self._stack())
            if stalled:
                logger.info('Event loop recovered stall_seconds=%.3f', ended[0] - started)
            if self._stop.wait(max(0., .1 - (time.monotonic() - started))):
                return


@asynccontextmanager
async def loop_lag_lifespan():
    recorder = LoopLagRecorder(asyncio.get_running_loop())
    recorder.start()
    try:
        yield recorder
    finally:
        await asyncio.to_thread(recorder.stop)
