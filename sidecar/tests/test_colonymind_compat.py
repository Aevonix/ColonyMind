"""Canonical package entry points and explicit environment scope."""
import os
from pathlib import Path
import subprocess
import sys

import pytest



def child(tmp_path, code, *arguments):
    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('COLONYMIND_')}
    env.update(HOME=str(tmp_path), COLONYMIND_SKIP_DOTENV='1',
               PYTHONPATH=str(root/'sidecar'))
    return subprocess.run([sys.executable, '-c', code, *arguments], env=env,
        cwd=tmp_path, capture_output=True, text=True, timeout=20)


def test_canonical_imports_share_ledger_cache(tmp_path):
    result = child(tmp_path, """
from colonymind.turns.idempotency import get_turn_idempotency_ledger
import sys
assert get_turn_idempotency_ledger(sys.argv[1]) is get_turn_idempotency_ledger(sys.argv[1])
print('same ledger')
""", str(tmp_path/'state'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'same ledger'






@pytest.mark.parametrize('module', ['colonymind', 'colonymind.cli'])
def test_module_entrypoints_have_canonical_help_without_contacting_services(tmp_path, module):
    result = child(tmp_path, '''
import runpy,socket,sys
def no_network(*args,**kwargs): raise AssertionError('CLI help cannot contact services')
socket.create_connection=no_network
socket.socket.connect=no_network
module=sys.argv[1];sys.argv=[module,'--help']
runpy.run_module(module,run_name='__main__')
''', module)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage: colonymind ' in result.stdout


def test_concurrent_state_reads_never_rewrite_process_environment(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import colonymind
    selected = tmp_path / 'selected'
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(selected))
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(selected))
    before = dict(os.environ)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: colonymind.get_state_dir(), range(128)))
    assert results == [selected] * 128
    assert dict(os.environ) == before
