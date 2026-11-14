"""Root pytest configuration.

The ``e2e`` and ``integration`` suites talk to a *running* ColonyMind sidecar over
HTTP (httpx against ``COLONYMIND_URL``). They are skipped unless a live sidecar is
configured (``COLONYMIND_URL`` + ``COLONYMIND_API_KEY``) and actually reachable, so the
default ``pytest tests/`` run exercises the unit suite for real and is
honest-green without any external services.

This replaces an earlier ``tests/integration/conftest.py`` hook that skipped the
*entire* session (not just integration tests) whenever ``COLONYMIND_URL`` was unset —
which silently skipped every unit test in CI.
"""

import atexit
import asyncio
import importlib.util
import os
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# Use the adapter suite's hardware latency policy for sidecar measurements too.
_latency_spec = importlib.util.spec_from_file_location(
    "_sidecar_hardware_latency", Path(__file__).resolve().parents[2] / "scripts/hardware_latency.py")
_latency = importlib.util.module_from_spec(_latency_spec)
_latency_spec.loader.exec_module(_latency)
hardware_latency = _latency.hardware_latency
pytest_addoption = _latency.pytest_addoption
pytest_sessionfinish = _latency.pytest_sessionfinish
pytest_terminal_summary = _latency.pytest_terminal_summary


def pytest_configure(config):
    _latency.configure(config)
    config.addinivalue_line("markers", "hardware_latency: latency enforced locally and reported on GitHub Actions")


@pytest.fixture(scope="module")
def isolated_hermes_python(request):
    """Install/uninstall adapters in a disposable venv, borrowing only Hermes' dependencies."""
    source = os.environ.get("COLONYMIND_TEST_HERMES_PYTHON") or sys.executable
    with tempfile.TemporaryDirectory(prefix="hermes-test-") as directory:
        subprocess.run([source, "-m", "venv", directory], check=True)
        python = str(Path(directory) / "bin" / "python")
        site_query = "import sysconfig; print(sysconfig.get_path('purelib'))"
        source_site = subprocess.check_output([source, "-I", "-c", site_query], text=True).strip()
        target_site = subprocess.check_output([python, "-I", "-c", site_query], text=True).strip()
        (Path(target_site) / "hermes-dependencies.pth").write_text(
            f"import site; site.addsitedir({source_site!r})\n")
        # Upgrade tests import _args from test_init, so bind its globals too.
        import test_init

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(request.module, "HERMES_PYTHON", python)
            patch.setattr(test_init, "HERMES_PYTHON", python)
            yield python

# Collection imports the production ASGI module, which normally loads the
# operator's ~/.colonymind/.env and opens state stores immediately.  Unit tests
# must never inherit credentials or mutate live state, and release trees may
# intentionally be read-only.  Establish an isolated, writable process state
# before any test module is imported.
_TEST_STATE_DIR = tempfile.mkdtemp(prefix="colonymind-pytest-")
atexit.register(shutil.rmtree, _TEST_STATE_DIR, ignore_errors=True)
_TEST_HOME_DIR = os.path.join(_TEST_STATE_DIR, "home")
os.mkdir(_TEST_HOME_DIR, mode=0o700)
os.environ["COLONYMIND_TEST_HOME"] = _TEST_HOME_DIR
# HOME itself, not a patched expanduser: the processes a test starts inherit it, and a product path that is
# fixed when its module is imported (util/session_safety.py) is built from it. A test that sets its own
# HOME keeps it. Outside a virtual environment the interpreter has packages under the home (its user site):
# the processes a test starts keep finding them where this interpreter has them.
if site.ENABLE_USER_SITE:
    os.environ.setdefault("PYTHONUSERBASE", site.getuserbase())
os.environ["HOME"] = _TEST_HOME_DIR
for variable in ("HERMES_HOME", "COLONYMIND_HOME", "COLONYMIND_INSTANCE_SELECTED"):
    os.environ.pop(variable, None)
for variable, leaf in (
    ("XDG_CONFIG_HOME", "config"),
    ("XDG_DATA_HOME", "data"),
    ("XDG_CACHE_HOME", "cache"),
):
    directory = os.path.join(_TEST_HOME_DIR, leaf)
    os.mkdir(directory, mode=0o700)
    os.environ[variable] = directory
os.environ.pop("XDG_STATE_HOME", None)  # its default lies under HOME
os.environ["COLONYMIND_SKIP_DOTENV"] = "1"
os.environ["COLONYMIND_STATE_DIR"] = _TEST_STATE_DIR
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

# Some offline sandboxes permit socketpair creation but deny the send that asyncio
# uses to wake a loop from another thread. Poll only when qualification explicitly
# requests it; socket permissions and application behavior remain unchanged.
_POLL_LOOP = os.environ.get("COLONYMIND_TEST_POLL_LOOP") == "1"
_ORIGINAL_RUN_FOREVER = asyncio.BaseEventLoop.run_forever


def _polling_run_forever(loop):
    selector = getattr(loop, "_selector", None)
    if selector is None:
        return _ORIGINAL_RUN_FOREVER(loop)
    original_select = selector.select

    def select(timeout=None):
        # A timer cannot wake a selector after a test replaces time.monotonic.
        # Bound the wait itself so background loops survive those clock fixtures.
        return original_select(0.001 if timeout is None else min(timeout, 0.001))

    selector.select = select
    try:
        return _ORIGINAL_RUN_FOREVER(loop)
    finally:
        selector.select = original_select


if _POLL_LOOP:
    asyncio.BaseEventLoop.run_forever = _polling_run_forever


def pytest_report_header(config):
    if _POLL_LOOP:
        return "Qualification environment: asyncio loops poll every 1 ms; socket restrictions remain active"


def pytest_unconfigure(config):
    if asyncio.BaseEventLoop.run_forever is _polling_run_forever:
        asyncio.BaseEventLoop.run_forever = _ORIGINAL_RUN_FOREVER

# Directories (relative to this file) whose tests require a live sidecar.
_LIVE_DIRS = ("e2e", "integration")


def _live_sidecar_status() -> tuple[bool, str]:
    """Return (enabled, skip_reason) for the live-sidecar test suites."""
    url = os.environ.get("COLONYMIND_URL")
    key = os.environ.get("COLONYMIND_API_KEY")
    if not url or not key:
        return False, (
            "live sidecar tests skipped — set COLONYMIND_URL and COLONYMIND_API_KEY "
            "to a running sidecar to run the e2e/integration suites"
        )
    try:
        import httpx

        # Any HTTP response (even 401/404) means a sidecar is listening.
        httpx.get(url, timeout=2.0)
        return True, ""
    except Exception:
        return False, f"live sidecar tests skipped — no sidecar reachable at {url}"


def pytest_collection_modifyitems(config, items):
    enabled, reason = _live_sidecar_status()
    if enabled:
        return
    tests_root = os.path.dirname(__file__)
    live_roots = tuple(os.path.join(tests_root, d) + os.sep for d in _LIVE_DIRS)
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        path = str(getattr(item, "path", "") or getattr(item, "fspath", ""))
        # str.startswith accepts a tuple of prefixes.
        if path.startswith(live_roots):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_served_context_window():
    """The per-conversation record of served context is process-wide; no test inherits another's."""
    from colonymind.memory import compass
    compass.SERVED.clear()
    yield
