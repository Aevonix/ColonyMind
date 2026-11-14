"""Offline qualification guard, also loaded by test subprocess sitecustomize."""
from __future__ import annotations

from contextlib import contextmanager
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

import pytest

BLOCKED_CONNECTIONS = []
_INSTALLED = False
_CASE_TIMEOUT = 0.0
_REPORT_JOURNAL = None
_JOURNAL_FSYNC = os.fsync


def pytest_configure(config):
    global _CASE_TIMEOUT, _REPORT_JOURNAL
    raw_timeout = os.environ.get('COLONYMIND_TEST_CASE_TIMEOUT') or '0'
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise pytest.UsageError('COLONYMIND_TEST_CASE_TIMEOUT must be a nonnegative number') from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise pytest.UsageError('COLONYMIND_TEST_CASE_TIMEOUT must be a finite nonnegative number')
    if timeout and (sys.platform != 'linux' or not hasattr(signal, 'setitimer')
                    or threading.current_thread() is not threading.main_thread()):
        raise pytest.UsageError('COLONYMIND_TEST_CASE_TIMEOUT requires the Linux main thread')
    _CASE_TIMEOUT = timeout
    journal = os.environ.get('COLONYMIND_TEST_REPORT_JOURNAL')
    _REPORT_JOURNAL = Path(journal) if journal else None
    if _REPORT_JOURNAL is not None:
        _REPORT_JOURNAL.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _case_deadline(nodeid, phase):
    if not _CASE_TIMEOUT:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Offline qualification deadlines require the main thread')
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        # Failed inherits BaseException, so application exception recovery cannot
        # mistake a timed out test for a successful fallback.
        pytest.fail(f'{phase} exceeded {_CASE_TIMEOUT:g}s: {nodeid}', pytrace=False)

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, _CASE_TIMEOUT)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        remaining, interval = previous_timer
        if remaining:
            remaining = max(0.000001, remaining - (time.monotonic() - started))
        signal.setitimer(signal.ITIMER_REAL, remaining, interval)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    with _case_deadline(item.nodeid, 'setup'):
        yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    with _case_deadline(item.nodeid, 'call'):
        yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    with _case_deadline(item.nodeid, 'teardown'):
        yield


def pytest_runtest_logreport(report):
    if _REPORT_JOURNAL is None:
        return
    entry = {'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome,
             'duration': report.duration}
    if report.outcome != 'passed':
        entry['longrepr'] = str(report.longrepr)
    # Flush each phase so an interrupted native call cannot erase earlier results.
    with _REPORT_JOURNAL.open('a', encoding='utf-8') as journal:
        journal.write(json.dumps(entry, ensure_ascii=True) + '\n')
        journal.flush()
        _JOURNAL_FSYNC(journal.fileno())


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    original_home = Path.home.__func__
    fallback = (os.environ.get('COLONYMIND_TEST_HOME') or os.environ.get('HERMES_HOME')
                or tempfile.mkdtemp(prefix='step0-test-home-'))
    def test_home(cls):
        if os.environ.get('HOME'):
            return original_home(cls)
        return cls(os.environ.get('COLONYMIND_TEST_HOME') or fallback)
    Path.home = classmethod(test_home)

    def offline(event, args):
        if event not in {'socket.connect', 'socket.getaddrinfo'}:
            return
        address = args[1] if event == 'socket.connect' else (args[0],)
        if not isinstance(address, tuple):  # local Unix socket
            return
        host = address[0]
        if host in {'localhost', None}:
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        BLOCKED_CONNECTIONS.append(event)
        raise AssertionError('Unexpected outbound call in offline qualification')
    sys.addaudithook(offline)


@pytest.fixture(autouse=True)
def no_unexpected_outbound_calls():
    BLOCKED_CONNECTIONS.clear()
    yield
    assert not BLOCKED_CONNECTIONS, 'Unexpected outbound calls were blocked by offline qualification'


def pytest_report_header():
    import colonymind
    import colonymind_hermes
    import hermes_cli
    return [f'ColonyMind adapter: {colonymind_hermes.__file__}', f'ColonyMind sidecar: {colonymind.__file__}',
            f'Hermes: {hermes_cli.__file__}']
