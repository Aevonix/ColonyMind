"""The wheel carries exactly the thin adapter and the memory provider."""

import configparser
from email.parser import BytesParser
import tarfile
import subprocess
import sys
import zipfile

import yaml
from conftest import ROOT

ADAPTER_MODULES = {"__init__", "client", "capture", "body", "guard", "commands", "tools", "reminders", "answers",
                   "sms_ingress", "busy_ingress", "native_settlement", "work", "voice_writeback", "standing",
                   "authority_diagnostics", "registration", "write_origin", "ideas", "archive"}
# The sealed store's Hermes half (docs/SECRETS.md), counted apart from the thin adapter's ceiling below.
SECRETS_MODULES = {"secrets"}
PROVIDER_MODULES = {"__init__", "provider", "cli"}


def test_wheel_contains_the_adapter_modules_and_nothing_else(artifacts):
    _, wheel, source = artifacts
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        adapter = {n.split("/", 1)[1] for n in names if n.startswith("colonymind_hermes/")}
        provider = {n.split("/", 1)[1] for n in names if n.startswith("colonymind_memory/")}
        core = {'archive_core/' + name for name in ('__init__.py', 'reparent.py', 'store.py')}
        assert adapter == {f"{m}.py" for m in ADAPTER_MODULES | SECRETS_MODULES} | {"plugin.yaml"} | core, adapter
        for name in ('__init__.py', 'reparent.py', 'store.py'):
            assert archive.read('colonymind_hermes/archive_core/' + name) == (ROOT / 'sidecar/colonymind/session_archive' / name).read_bytes()
        assert provider == {f"{m}.py" for m in PROVIDER_MODULES} | {"plugin.yaml", "SKILL.md"}, provider
        for module in ADAPTER_MODULES | SECRETS_MODULES:
            assert archive.read(f"colonymind_hermes/{module}.py") == (ROOT / f"plugins/hermes-plugin/{module}.py").read_bytes()
        assert not any(n.startswith("colonymind/") or "/ops/" in n or "bundled_skills" in n
                       for n in names)
        metadata = BytesParser().parsebytes(archive.read(next(n for n in names if n.endswith(".dist-info/METADATA"))))
        for package in ("colonymind_hermes", "colonymind_memory"):
            manifest = yaml.safe_load(archive.read(f"{package}/plugin.yaml"))
            assert manifest["version"] == metadata["Version"], package
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(next(n for n in names if n.endswith("/entry_points.txt"))).decode())
        # The optional gates are a second plugin with its own entry point (test_instinct_gates.py covers it).
        assert dict(entries["hermes_agent.plugins"]) == {"colonymind": "colonymind_hermes",
                                                          "colonymind-gates": "colonymind_gates"}
        assert dict(entries["hermes_agent.memory_providers"]) == {"colonymind-memory": "colonymind_memory"}
    with tarfile.open(source) as archive:
        names = archive.getnames()
        assert any(n.endswith("/plugins/hermes-plugin/guard.py") for n in names)
        assert not any("/ops/" in n for n in names)


def test_packaged_gateway_journal_api_uses_only_its_injected_clock_without_sidecar(artifacts):
    output, wheel, _ = artifacts
    code = r'''
import importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1])
class NoSidecar(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'colonymind' or fullname.startswith('colonymind.'):
            raise AssertionError('gateway journal imported sidecar: ' + fullname)
sys.meta_path.insert(0, NoSidecar())
from colonymind_hermes.archive_core.store import ArchiveStore
from colonymind_hermes.archive_core.reparent import archive_reparent
clock = [1000.0]
store = ArchiveStore(pathlib.Path(sys.argv[2]) / 'gateway-journal.db', clock=lambda: clock[0])
def engine(**kwargs):
    if kwargs['phase'] == 'plan': return {'jobs': []}
    if kwargs['phase'] == 'apply': return {'reparented': {'cron_jobs': 0}, 'tasks_reparented': True}
    if kwargs['phase'] == 'rollback': return {'restored': {'cron_jobs': 0}, 'half_applied': False, 'jobs_left_moved': []}
body = {'operation_id': 'package-check', 'archived': True, 'reparent_to_main': True}
store.enqueue('session', body, {'owner_id': 'owner'})
store.heartbeat()
assert store.worker_available()
plan = store.claim_plan('worker', 'owner')
assert plan['created_at'] == '1970-01-01T00:16:40+00:00'
receipt = store.archive('session', body, engine)
assert receipt['archived_at'] == plan['created_at']
assert store.archive('session', body, engine) == receipt
store.accept_plan_result(body['operation_id'], 'worker', plan['claim'], receipt)
assert store.plan_result(body['operation_id']) == receipt
assert store.receipt('session', engine)['receipt'] == receipt
assert store.inventory()['snapshot_at'] == plan['created_at']
clock[0] += 60
assert not store.worker_available()
assert archive_reparent(phase='validate', session_id='session', request={}, intent=None) is None
assert not any(name == 'colonymind' or name.startswith('colonymind.') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(wheel), str(output)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_the_sealed_store_module_line_count_is_reported():
    """Tracked, not gated, as the adapter's own count: ``secrets.py`` carries a byte-identical copy of the sidecar's
    capture detector (about 220 lines, kept equal by ``sidecar/tests/test_secrets_detect.py``), the gateway seal,
    the fills and masks on tool calls, the send-time resolve and the owner's ``colonymind_secret`` (635 lines when
    it arrived). The ceiling moved from 700 to 1,050 with the round-1 review fixes: a value placed in a command or
    in code as a private file read at run time (the shell and Python placement and the files' lifetime), the ask
    bound to the complete call and put to the owner directly, the quoted and backfilled words sealed in place, the
    section validated as the sidecar validates it, and a reveal sent to the owner's chat (987 after). It moved to
    1,150 with the round-2 review fixes, which replaced the placement parser: a value reaches a command or code
    only through that call's environment (the shell and Python binding scripts, and the files kept while the program
    that took them runs), references resolved in batches, freshness bound to the chat, and a reveal without a chat
    put in a private file."""
    total = sum(len((ROOT / f"plugins/hermes-plugin/{m}.py").read_text().splitlines()) for m in SECRETS_MODULES)
    print(f"sealed store lines: {total}")
    assert total < 1150


def test_adapter_line_count_is_reported():
    """Counts and prints the source lines of the thin adapter's modules (``ADAPTER_MODULES``, the files of
    ``plugins/hermes-plugin``; the sealed store's module is counted by the test above). The ceiling exists so that
    the adapter, which is meant to stay host integration with the mind's behaviour kept in the sidecar, cannot grow
    far past its present size unless someone raises the number on purpose."""
    total = sum(len((ROOT / f"plugins/hermes-plugin/{m}.py").read_text().splitlines()) for m in ADAPTER_MODULES)
    print(f"plugin lines: {total}")
    assert total < 10000
