"""One fixture follows checked and unchecked work through capture and erasure."""
from contextlib import closing
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from colonymind.hands.evidence import evaluate_result
from colonymind.hands.provenance import Provenance
from colonymind.hands.wire import DigestRequest, ResultRequest
from colonymind.memory.recall import render_memory_context, source_candidates
from colonymind.mind.lessons import Lessons
from colonymind.mind.outcomes import Autobiography
from test_hands_controller import EPOCH, c, finish, running
from test_hands_provenance import FIELDS
from test_hands_provenance_capture import captured


@pytest.mark.asyncio
async def test_check_capture_recall_lesson_expiry_and_erasure(c, captured, tmp_path):
    module, capture, outbox, settings, ledger = captured
    lessons = Lessons(ledger=ledger, store=None, owner_id='owner',
                      autobiography=Autobiography(ledger, owner_id='owner', clock=c.clock))
    provenance = Provenance(c, ledger, owner_id='owner', lessons=lessons)
    c.evidence_evaluator = evaluate_result
    unchecked = await running(c, key='unchecked', name='Analysis')
    assert (await finish(c, unchecked))['evidence']['level'] == 'E0'
    assert await provenance.record_accepted(unchecked['hand_id'], 'The valve seems ready.') is None

    repo = tmp_path / 'work'
    repo.mkdir()
    (repo / 'valve.txt').write_text('ready')
    for args in [('init',), ('add', '.'), ('-c', 'user.name=Owner', '-c',
                 'user.email=owner@example.invalid', 'commit', '-m', 'fixture')]:
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    command = [sys.executable, '-c',
               'from pathlib import Path; assert Path("valve.txt").read_text() == "ready"']
    plan = {'test_command': command}
    checked = await running(c, key='checked', kind='coding', name='Check',
                            workdir=str(repo), accept=plan, risk='high')
    task = await c.get_task(checked['task_id'])
    criteria = task['intents']['1']['acceptance']
    path = Path(__file__).resolve().parents[2] / 'plugins/colonymind-hands/checks.py'
    spec = spec_from_file_location('oc4_lifecycle_checks', path)
    runner_module = module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    runner = runner_module.CheckRunner()
    receipts = runner.finalize(runner.run(kind='coding', acceptance=criteria, workdir=repo), repo)
    result = await c.result(checked['hand_id'], ResultRequest(
        epoch=EPOCH, intent_version=1, status='done', summary_ref='branch-checked',
        summary_hash='a' * 64, tool_digest=[], receipts=receipts, families_used=['family-a']),
        'checked:result')
    assert result['outcome'] == 'done' and result['evidence']['level'] == 'E3'
    entry = await provenance.record_accepted(checked['hand_id'], 'The valve passed the planned check.')
    assert entry
    assert await provenance.record_accepted(checked['hand_id'], 'A repeated result.') == entry
    lesson = await provenance.admit_lesson(checked['hand_id'], FIELDS)
    assert lesson is not None
    assert await provenance.admit_lesson(unchecked['hand_id'], FIELDS) is None

    digest = await c.digest(DigestRequest(session='parent'))
    capture.record_hand_refs(session_id='parent', turn_id='relay', digest=digest)
    capture.post_llm_call(session_id='parent', turn_id='relay',
                          assistant_response='My valve analysis and planned check are complete.')
    payload = outbox.rows()[0]['payload']
    await provenance.record_relay('relay', [ref['hand_id'] for ref in payload['hand_refs']])

    class Client:
        def post(self, path, *, json, timeout):
            assert path == '/v1/host/turns/sync'
            ledger.record_source('relay', contact_id='owner', session_id='parent',
                                 messages=[json['assistant_message']])
            return SimpleNamespace(status_code=200, is_success=True, json=lambda: {})

    assert module.deliver(Client(), capture.sessions, settings, payload)

    def recalled_relay():
        from colonymind.memory.curated import current_candidates
        candidates = source_candidates(ledger.search_sources(
            'valve', contact_id='owner', session_id='parent', limit=20))
        candidates = current_candidates(ledger, candidates, contact_id='owner')
        return render_memory_context([row for row in candidates if row['source_turn_id'] == 'relay'])

    rendered = recalled_relay()
    assert 'own analysis, not checked' in rendered and 'requirement tested' in rendered
    with closing(ledger._connect()) as db:
        assert db.execute('SELECT count(*) FROM source_claims').fetchone()[0] == 0
    await provenance.expire_branch('branch-checked')
    assert 'evidence expired' in recalled_relay()
    assert lessons.get(lesson.id) is not None
    with closing(ledger._connect()) as db:
        assert db.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (entry,)).fetchone()
    await provenance.erase_parent('parent')
    assert not recalled_relay()
    assert lessons.get(lesson.id) is None
    with closing(ledger._connect()) as db:
        assert db.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (entry,)).fetchone()
