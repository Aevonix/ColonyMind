"""Final-state regression cases, collected before implementing the OC4 runner."""
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
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    (tmp_path / 'output.txt').write_text('first')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Owner', '-c',
                    'user.email=owner@example.invalid', 'commit', '-m', 'fixture'], check=True, capture_output=True)
    return tmp_path


def test_nested_workdir_binding_includes_files_outside_subdirectory(checks, repo):
    nested = repo / 'nested'
    nested.mkdir()
    runner = checks.CheckRunner()
    receipt = runner.run(kind='coding', acceptance={'test_command': [sys.executable, '-c', 'pass']}, workdir=nested)[0]
    assert receipt['passed']
    (repo / 'output.txt').write_text('changed outside workdir')
    assert not runner.is_current(receipt, nested)


def test_finalize_revokes_receipt_after_last_write(checks, repo):
    runner = checks.CheckRunner()
    receipts = runner.run(kind='coding', acceptance={'test_command': [sys.executable, '-c', 'pass']}, workdir=repo)
    (repo / 'output.txt').write_text('last write after passing test')
    final = runner.finalize(receipts, repo)
    assert final[0]['passed'] is False
    assert final[0]['scope']['details']['current'] is False


def test_named_state_requires_exact_complete_lines(checks, tmp_path):
    receipt = checks.CheckRunner().run(kind='ops', workdir=tmp_path, acceptance={
        'probe': [sys.executable, '-c', 'print("not-service=ready")'], 'requirements': ['service=ready']})[0]
    assert receipt['exit_code'] == 0 and not receipt['passed']


def test_research_same_host_does_not_meet_independent_source_rule(checks, tmp_path):
    fetched = {'https://one.example/a': 'Sam reports one.', 'https://www.one.example/b': 'Robin reports two.'}
    receipt = checks.CheckRunner().run(kind='research', acceptance={}, workdir=tmp_path,
        fetched=fetched, quotes=[{'url': url, 'quote': words} for url, words in fetched.items()])[0]
    assert receipt['passed']
    assert not receipt['scope']['details']['requirement_passed']


@pytest.mark.parametrize('output_type,suffix', [('image', '.png'), ('pdf', '.pdf'), ('audio', '.wav'), ('video', '.mp4')])
def test_corrupt_media_cannot_validate(checks, tmp_path, output_type, suffix):
    path = tmp_path / ('artifact' + suffix)
    path.write_bytes(b'exists, but does not decode')
    receipt = checks.CheckRunner().run(kind='media', acceptance={'output_type': output_type},
                                     workdir=tmp_path, artifacts=[path])[0]
    assert not receipt['passed']
    assert not receipt['scope']['details']['artifact_validated']


def test_media_receipt_tracks_file_hash_after_validation(checks, tmp_path):
    from PIL import Image
    path = tmp_path / 'artifact.png'
    Image.new('RGB', (2, 3)).save(path)
    runner = checks.CheckRunner()
    receipt = runner.run(kind='media', acceptance={'output_type': 'image'}, workdir=tmp_path, artifacts=[path])[0]
    assert receipt['passed']
    assert receipt['scope']['details']['validation'] == [{'dimensions': [2, 3]}]
    assert runner.is_current(receipt, tmp_path)
    path.write_bytes(b'changed')
    assert not runner.finalize([receipt], tmp_path)[0]['passed']


def test_invalid_plan_commands_fail_closed(checks, tmp_path):
    receipt = checks.CheckRunner().run(kind='ops', acceptance={'probe': [42]}, workdir=tmp_path)[0]
    assert not receipt['passed']
    assert receipt['scope']['details']['error'] == 'ValueError'
