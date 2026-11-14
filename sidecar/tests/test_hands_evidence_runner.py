"""Actual local checker receipts must survive the fixed result wire model."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

from pydantic import TypeAdapter
import pytest

from colonymind.hands.evidence import assess, evaluate_result, scope_hash
from colonymind.hands.types import CheckReceipt


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / 'plugins/colonymind-hands/checks.py'
    spec = spec_from_file_location('hands_evidence_runner', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CheckRunner()


def evidence(runner, kind, acceptance, workdir, **kwargs):
    receipts = runner.finalize(runner.run(kind=kind, acceptance=acceptance, workdir=workdir, **kwargs), workdir)
    wire = TypeAdapter(list[CheckReceipt]).validate_python(receipts)
    return assess(kind=kind, acceptance=acceptance, receipts=wire)


def test_real_code_receipt_attests_final_commit_and_revokes_later_write(runner, tmp_path):
    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)
    git('init')
    (tmp_path / 'value.txt').write_text('Owner')
    git('add', '.')
    git('-c', 'user.name=Owner', '-c', 'user.email=owner@example.invalid', 'commit', '-m', 'fixture')
    plan = {'test_command': [sys.executable, '-c', 'assert open("value.txt").read() == "Owner"']}
    receipts = runner.run(kind='coding', acceptance=plan, workdir=tmp_path)
    assert assess(kind='coding', acceptance=plan, receipts=receipts).level == 'E3'
    (tmp_path / 'value.txt').write_text('Sam')
    assert assess(kind='coding', acceptance=plan, receipts=runner.finalize(receipts, tmp_path)).level == 'E1'


@pytest.mark.parametrize('state,expected', [('ready', 'E3'), ('unknown', 'E1')])
def test_real_ops_probe_named_state(runner, tmp_path, state, expected):
    plan = {'probe': [sys.executable, '-c', f'print("service={state}")'], 'requirements': ['service=ready']}
    assert evidence(runner, 'ops', plan, tmp_path).level == expected


@pytest.mark.parametrize('words,expected', [('Sam reports the result.', 'E2'), ('Missing quote', 'E1')])
def test_real_research_quote_validation_is_separate_from_source_requirement(runner, tmp_path, words, expected):
    url = 'https://source.example/fixture'
    assert evidence(runner, 'research', {}, tmp_path, fetched={url: 'Sam reports the result.'},
                    quotes=[{'url': url, 'quote': words}]).level == expected


@pytest.mark.parametrize('output_type', ['pdf', 'audio'])
def test_real_parsed_media_passes_its_planned_output_type(runner, tmp_path, output_type):
    if output_type == 'pdf':
        from pypdf import PdfWriter
        path = tmp_path / 'artifact.pdf'
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(path)
    else:
        path = tmp_path / 'artifact.wav'
        with wave.open(str(path), 'wb') as stream:
            stream.setparams((1, 2, 8000, 8, 'NONE', 'not compressed'))
            stream.writeframes(b'\0\0' * 8)
    assert evidence(runner, 'media', {'output_type': output_type}, tmp_path, artifacts=[path]).level == 'E3'


@pytest.mark.parametrize('kind', ['deep', 'plan', 'synthesize'])
def test_analysis_text_opens_without_establishing_its_requirements(runner, tmp_path, kind):
    path = tmp_path / 'analysis.txt'
    path.write_text('Sam reports an unreviewed analysis for Owner.')
    plan = {'output_type': 'text'}
    receipts = runner.run(kind=kind, acceptance=plan, workdir=tmp_path, artifacts=[path])
    assert receipts[0]['passed']
    assert receipts[0]['scope']['details']['requirement_passed'] is False
    # Even a caller supplying a successful output-type flag cannot claim that
    # opening analysis text tested the analysis's substantive requirements.
    receipts[0]['scope']['details']['requirement_passed'] = True
    assert assess(kind=kind, acceptance=plan, receipts=receipts).level == 'E2'


@pytest.mark.parametrize('effort', [0, 1])
def test_low_effort_deep_artifact_requires_independent_review(runner, tmp_path, effort):
    path = tmp_path / 'analysis.txt'
    path.write_text('Sam reports an unreviewed analysis for Owner.')
    plan = {'output_type': 'text'}
    receipts = runner.run(kind='deep', acceptance=plan, workdir=tmp_path, artifacts=[path])
    base = assess(kind='deep', effort=effort, acceptance=plan, receipts=receipts, producer_families=['sam-route'])
    assert base.level == 'E1'
    assert base.scope.artifacts[0].sha256
    review = {'verdict': 'pass', 'family': 'robin-route', 'scope_hash': scope_hash(base.scope)}
    assert assess(kind='deep', effort=effort, acceptance=plan, receipts=receipts,
                  producer_families=['sam-route'], review=review).level == 'E4'
    review['family'] = 'sam-route'
    assert assess(kind='deep', effort=effort, acceptance=plan, receipts=receipts,
                  producer_families=['sam-route'], review=review).level == 'E1'
    body = SimpleNamespace(receipts=receipts, tool_digest=[], families_used=['sam-route'],
                           contracts_used=[], summary_ref='branch:analysis', summary_hash='a' * 64)
    task = {'intents': {'1': {'acceptance': plan}}}
    hand = {'kind': 'deep', 'effort': effort, 'intent_version': 1}
    assert evaluate_result(hand, task, body)['level'] == 'E1'
