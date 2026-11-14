"""OC4 checks run locally; none of these cases needs a socket or a model."""
import importlib
import subprocess
import sys

import pytest

from hands_fake_hermes import load_hands


@pytest.fixture
def checks():
    return importlib.import_module(load_hands().__name__ + '.checks')


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)
    git('init')
    (tmp_path / 'output.txt').write_text('first')
    git('add', '.')
    git('-c', 'user.name=Owner', '-c', 'user.email=owner@example.invalid', 'commit', '-m', 'fixture')
    return tmp_path


def test_code_receipt_bound_to_head_porcelain_and_final_bytes(checks, repo):
    command = [sys.executable, '-c', 'print("checked")']
    runner = checks.CheckRunner()
    receipts = runner.run(kind='coding', acceptance={'test_command': command}, workdir=repo)
    receipt = receipts[-1]
    assert receipt['passed'] is True and receipt['exit_code'] == 0
    assert receipt['scope']['command'] == command
    assert receipt['scope']['reference'] == subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
    assert receipt['scope']['dirty_tree_sha256']
    assert runner.is_current(receipt, repo)
    (repo / 'output.txt').write_text('second')
    assert not runner.is_current(receipt, repo)
    dirty = runner.run(kind='coding', acceptance={'test_command': command}, workdir=repo)[-1]
    (repo / 'output.txt').write_text('third')
    assert not runner.is_current(dirty, repo)  # porcelain itself has not changed


def test_only_plan_commands_run_and_no_criteria_means_no_command(checks, repo):
    runner = checks.CheckRunner()
    assert runner.run(kind='coding', acceptance={}, workdir=repo) == []
    command = [sys.executable, '-c', 'open("planned.txt","w").write("ok")']
    receipt = runner.run(kind='coding', acceptance={'test_command': command}, workdir=repo)[-1]
    assert (repo / 'planned.txt').read_text() == 'ok'
    assert receipt['passed'] is False  # a test that changes the artifact does not attest the final bytes


def test_quote_must_occur_in_text_fetched_by_this_hand(checks, tmp_path):
    runner = checks.CheckRunner()
    fetched = {'https://one.example/page': 'Exact quoted words.', 'https://two.example/page': 'Independent words.'}
    acceptance = {'required_sources': 2, 'independent_sources': True, 'quotes_required': True}
    quotes = [{'url': url, 'quote': text} for url, text in fetched.items()]
    receipt = runner.run(kind='research', acceptance=acceptance, workdir=tmp_path,
                         fetched=fetched, quotes=quotes)[-1]
    assert receipt['passed'] and receipt['scope']['details']['requirement_passed']
    quotes[0]['quote'] = 'Invented quotation'
    bad = runner.run(kind='research', acceptance=acceptance, workdir=tmp_path, fetched=fetched, quotes=quotes)[-1]
    assert bad['passed'] is False


def test_ops_exit_zero_does_not_establish_named_state(checks, tmp_path):
    runner = checks.CheckRunner()
    acceptance = {'probe': [sys.executable, '-c', 'print("ok")'], 'requirements': ['service=ready']}
    bad = runner.run(kind='ops', acceptance=acceptance, workdir=tmp_path)[-1]
    assert bad['exit_code'] == 0 and not bad['passed']
    acceptance['probe'] = [sys.executable, '-c', 'print("service=ready")']
    good = runner.run(kind='ops', acceptance=acceptance, workdir=tmp_path)[-1]
    assert good['passed'] and good['scope']['probe'] == acceptance['probe']


@pytest.mark.parametrize('workdir', [None, ''])
def test_ops_stopped_at_hold_without_probe_has_no_check_receipt(checks, workdir):
    # A held call can know its tool workdir even when the task has no planned
    # probe or workdir. No command ran, so there is no failed result check.
    assert checks.CheckRunner().run(kind='ops', acceptance={}, workdir=workdir) == []


def test_planned_probe_without_workdir_fails_closed_with_validation_error(checks):
    receipt, = checks.CheckRunner().run(kind='ops', workdir=None,
        acceptance={'probe': [sys.executable, '-c', 'print("ready")'], 'requirements': ['ready']})
    assert not receipt['passed'] and receipt['exit_code'] is None
    assert receipt['scope']['details']['error'] == 'ValueError'


def test_existing_undecodable_media_fails(checks, tmp_path):
    path = tmp_path / 'image.png'
    path.write_bytes(b'not an image')
    receipt = checks.CheckRunner().run(kind='media', acceptance={'output_type': 'image'},
                                     workdir=tmp_path, artifacts=[str(path)])[-1]
    assert receipt['passed'] is False


def test_timeout_cannot_pass(checks, tmp_path):
    receipt = checks.CheckRunner(timeout=.05).run(kind='ops', workdir=tmp_path,
        acceptance={'probe': [sys.executable, '-c', 'import time; time.sleep(1)'], 'requirements': ['ready']})[-1]
    assert not receipt['passed'] and receipt['scope']['details']['timed_out']


def test_a_python_test_importing_the_code_still_attests_a_repository_without_bytecode_ignores(checks, repo):
    (repo / 'module.py').write_text('VALUE = 1\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'module.py'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Owner', '-c', 'user.email=owner@example.invalid',
                    'commit', '-m', 'module'], check=True, capture_output=True)
    command = [sys.executable, '-c', 'import module; assert module.VALUE == 1']
    receipt = checks.CheckRunner().run(kind='coding', acceptance={'test_command': command}, workdir=repo)[-1]
    assert not (repo / '__pycache__').exists()
    assert receipt['passed'] is True and receipt['scope']['details']['current'] is True
