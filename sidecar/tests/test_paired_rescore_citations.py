"""Reach from what the host assembled: the probe turn's cited sources, mapped to message hashes by the ledger map.

A paired plugin arm's trace now records the sources each assembled context cited (``context_route`` events) and,
at the end of the episode, which messages each ledger source holds (``ledger_sources``: turn id, session and the
message hashes the ledger computes, never text). The re-score reads the probe turn's citations as their own reach
surface, ``cited``: structural on the gold messages' hashes, whatever the rendered section shows of them, and only
for a section whose body reached a model request of the turn. Every citation the turn's assemblies made, delivered
or not, is the ``assembled`` surface: coverage reported apart, never reach. A citation names a source, and a source
may hold many messages (a checkpoint holds a whole conversation): it reaches only the messages its section names by
hash, or the one message a single-message source holds, never every message of the source.
"""
import hashlib
import sqlite3

from colonymind.qualification import native_memory_worker, paired_worker
from colonymind.turns import TurnIdempotencyLedger
from colonymind.turns.idempotency import source_message_hash
from test_public_memory_rescore import (H_NEW, H_OLD, NEW, OLD, context, event, quotation, request,  # noqa: F401
                                        rescore, trace, update_case)

PROSE = 'The review moved; see the earlier notes.'          # a rendered line that cites nothing


def cited(*sources, path='/v1/host/context/assemble', body=None):
    """The assembly's trace event: the memory section's citations and, as the recorder writes them, its body's
    SHA-256 and length and the messages its records name (``body``: the section body the assembly returned)."""
    section = {'section': 'colonymind-memory', 'refs': [{'source_id': source, 'source_version': 'v1'}
                                                       for source in sources]}
    if body is not None:
        section.update(body_sha256=hashlib.sha256(body.encode()).hexdigest(), body_chars=len(body),
                       messages=native_memory_worker.shown_messages(body)[0])
    return event('context_route', {'method': 'POST', 'path': path, 'status': 200, 'source_refs': [section]},
                 thread='Thread-3 (serve)')


LEDGER = event('ledger_sources', {'sources': [
    {'turn_id': 'hist-1', 'session_id': 'chat-01', 'message_hashes': [H_OLD]},
    {'turn_id': 'hist-5', 'session_id': 'sms-03', 'message_hashes': [H_NEW]}], 'truncated': False})


def test_the_probe_turns_citations_reach_the_gold_by_its_hash(rescore):  # noqa: F811
    lines = [*trace(cited('hist-5', body='- ' + PROSE), request([{'role': 'user', 'content': context('- ' + PROSE)}])),
             LEDGER]
    row = rescore.episode(update_case(), lines, completion=True, answer='{"place": "main hall"}')
    assert row['reach_surfaces']['injected']['any'] is False           # the rendered line cites nothing
    assert row['reach_surfaces']['cited']['any'] is True and row['reach_surfaces']['cited']['basis'] == 'structural'
    assert row['reach']['any'] is True


def test_a_citation_of_another_message_is_a_miss_and_an_earlier_turns_does_not_count(rescore):  # noqa: F811
    lines = [*trace(cited('hist-1', body='- ' + PROSE), request([{'role': 'user', 'content': context('- ' + PROSE)}]),
                    before=['- ' + PROSE]), LEDGER]
    lines.insert(1, cited('hist-5', body='- ' + PROSE))                # an earlier turn's assembly cited the gold
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach_surfaces']['cited']['any'] is False and row['reach']['any'] is False


def test_without_citations_or_a_ledger_map_the_cited_surface_is_unscored(rescore):  # noqa: F811
    probe = request([{'role': 'user', 'content': context('- ' + PROSE)}])
    older = rescore.episode(update_case(), trace(probe), completion=True, answer='{"place": "main hall"}')
    assert older['reach_surfaces']['cited'] is None                    # a trace recorded before citations were
    unmapped = rescore.episode(update_case(), trace(cited('hist-5'), probe), completion=True,
                               answer='{"place": "main hall"}')
    assert unmapped['reach_surfaces']['cited'] is None and unmapped['reach']['any'] is False


def test_the_worker_maps_each_ledger_source_to_its_message_hashes_without_text(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    ledger.record_source('hist-1', contact_id='p-01', session_id='chat-01',
                         messages=[{'role': 'user', 'content': OLD}, {'role': 'assistant', 'content': 'Noted.'}])
    ledger.record_source('hist-5', contact_id='p-01', session_id='sms-03', messages=[{'role': 'user', 'content': NEW}])
    mapped = paired_worker.ledger_sources(tmp_path / 'turn-idempotency.db')
    assert mapped == {'sources': [
        {'turn_id': 'hist-1', 'session_id': 'chat-01', 'message_hashes': [
            H_OLD, source_message_hash('chat-01', {'role': 'assistant', 'content': 'Noted.'})]},
        {'turn_id': 'hist-5', 'session_id': 'sms-03', 'message_hashes': [H_NEW]}], 'truncated': False}
    assert OLD not in repr(mapped) and NEW not in repr(mapped)
    assert paired_worker.ledger_sources(tmp_path / 'missing.db') is None
    with sqlite3.connect(tmp_path / 'turn-idempotency.db') as conn:           # read-only: nothing written
        assert conn.execute('SELECT count(*) FROM turn_sources').fetchone()[0] == 2


def test_the_ledger_map_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_worker, 'LEDGER_MAP_SOURCES', 1)
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    for index in range(3):
        ledger.record_source(f't-{index}', contact_id='p-01', session_id='s', messages=[
            {'role': 'user', 'content': f'message {index}'}])
    mapped = paired_worker.ledger_sources(tmp_path / 'turn-idempotency.db')
    assert len(mapped['sources']) == 1 and mapped['truncated'] is True


def test_the_summary_shows_structural_reach_by_surface(rescore):  # noqa: F811
    lines = [*trace(cited('hist-5', body='- ' + PROSE), request([{'role': 'user', 'content': context('- ' + PROSE)}])),
             LEDGER]
    row = rescore.episode(update_case(), lines, completion=True, answer='{"place": "main hall"}')
    older = rescore.episode(update_case(), trace(request([{'role': 'user', 'content': context('- ' + PROSE)}])),
                            completion=True, answer='{"place": "main hall"}')
    late = rescore.episode(update_case(), [*trace(request([{'role': 'user', 'content': 'Where?'}]),
                                                  cited('hist-5', body='- ' + PROSE)), LEDGER],
                           completion=False, answer='{"place": "east room"}')
    table = rescore.structural_surfaces([row, older, late])
    assert table == {'injected': (0, 3), 'tools': (0, 3), 'cited': (1, 2), 'assembled': (2, 2)}


def test_an_assembly_the_model_never_received_is_not_reach(rescore):  # noqa: F811
    """The assembly answered after the turn's only model request, and the provider went on without it: the host
    cited the gold, the model never saw it. That is assembly coverage, reported apart, and never reach."""
    alone = request([{'role': 'user', 'content': 'Where is the monthly budget draft review held now?'}])
    lines = [*trace(alone, cited('hist-5', body='- ' + PROSE)), LEDGER]
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach_surfaces']['cited']['any'] is False and row['reach']['any'] is False
    assert row['reach_surfaces']['assembled']['any'] is True


def test_a_section_the_request_did_not_carry_whole_is_not_reach(rescore):  # noqa: F811
    """The request carried a memory section, but not the body this assembly returned (held back, spilled or cut):
    its citations are not reach."""
    shown = request([{'role': 'user', 'content': context('- ' + PROSE)}])
    lines = [*trace(cited('hist-5', body='- ' + PROSE + ' And more.'), shown), LEDGER]
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach_surfaces']['cited']['any'] is False and row['reach_surfaces']['assembled']['any'] is True


def test_without_the_bodys_hash_delivery_cannot_be_told_and_the_cited_surface_is_unscored(rescore):  # noqa: F811
    import json
    data = json.loads(cited('hist-5', body='- ' + PROSE))
    for name in ('body_sha256', 'body_chars'):
        data['data']['source_refs'][0].pop(name)
    probe = request([{'role': 'user', 'content': context('- ' + PROSE)}])
    row = rescore.episode(update_case(), [*trace(json.dumps(data), probe), LEDGER], completion=True,
                          answer='{"place": "main hall"}')
    assert row['reach_surfaces']['cited'] is None and row['reach']['any'] is False
    assert row['reach_surfaces']['assembled']['any'] is True
    # Without the messages each section named, neither surface can tell what a citation reached.
    bare = rescore.episode(update_case(), [*trace(cited('hist-5'), probe), LEDGER], completion=True,
                           answer='{"place": "main hall"}')
    assert bare['reach_surfaces']['cited'] is None and bare['reach_surfaces']['assembled'] is None


CHECKPOINT = event('ledger_sources', {'sources': [
    {'turn_id': 'ckpt-1', 'session_id': 'owner-1', 'message_hashes': [H_OLD, H_NEW]}], 'truncated': False})


def test_a_citation_reaches_only_the_messages_its_section_shows_not_its_whole_source(rescore):  # noqa: F811
    """A checkpoint holds the old statement and its correction; the section shows the old one only. Citing the
    checkpoint does not show the correction."""
    body = quotation('ckpt-1', H_OLD, OLD)
    lines = [*trace(cited('ckpt-1', body=body), request([{'role': 'user', 'content': context(body)}])), CHECKPOINT]
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach_surfaces']['cited']['any'] is False and row['reach']['any'] is False
    assert row['reach_surfaces']['assembled']['any'] is False
    shown = quotation('ckpt-1', H_NEW, NEW)
    lines = [*trace(cited('ckpt-1', body=shown), request([{'role': 'user', 'content': context(shown)}])), CHECKPOINT]
    row = rescore.episode(update_case(), lines, completion=True, answer='{"place": "main hall"}')
    assert row['reach_surfaces']['cited']['any'] is True


def test_a_prose_section_citing_a_source_of_several_messages_reaches_none_of_them(rescore):  # noqa: F811
    lines = [*trace(cited('ckpt-1', body='- ' + PROSE), request([{'role': 'user', 'content': context('- ' + PROSE)}])),
             CHECKPOINT]
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach_surfaces']['cited']['any'] is False


def test_a_section_whose_named_messages_were_cut_short_leaves_the_cited_surface_unscored(rescore):  # noqa: F811
    body = quotation('ckpt-1', H_OLD, OLD)
    event_line = cited('ckpt-1', body=body)
    import json
    data = json.loads(event_line)
    data['data']['source_refs'][0]['messages_truncated'] = True
    lines = [*trace(json.dumps(data), request([{'role': 'user', 'content': context(body)}])), CHECKPOINT]
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach_surfaces']['cited'] is None and row['reach_surfaces']['assembled'] is None


def test_the_recorder_names_the_messages_a_sections_records_show_by_their_hashes():
    body = '\n'.join([quotation('turn-7', 'a' * 64, 'Quoted words.'), '- ' + PROSE,
                      '- ' + json_line({'evidence_ref': 'q1', 'source': 'turn:turn-9', 'source_message_hash': 'b' * 64,
                                        'quote': 'Passage words.'}),
                      '- ' + json_line({'kind': 'conversation_pair', 'source': 'turn:turn-11', 'content': {
                          'input': {'source_id': 'turn-11', 'source_message_hash': 'c' * 64, 'quote': 'In.'},
                          'response': {'source_id': 'turn-11', 'quote': 'Out.'}}})])
    messages, truncated = native_memory_worker.shown_messages(body)
    assert messages == [['turn-7', 'a' * 64], ['turn-9', 'b' * 64], ['turn-11', 'c' * 64]] and truncated is False
    assert 'words' not in repr(messages)


def json_line(value):
    import json
    return json.dumps(value)


def cut_map(*rows):
    """A ledger map the worker cut at its bound: sources past it are not listed."""
    return event('ledger_sources', {'sources': [{'turn_id': turn, 'session_id': session, 'message_hashes': hashes}
                                                for turn, session, hashes in rows], 'truncated': True})


def test_a_citation_a_cut_ledger_map_cannot_resolve_leaves_reach_unscored_not_missed(rescore):  # noqa: F811
    """The map was cut before the cited gold source: which messages it holds is unknown, so the citation is
    unresolved, and a question nothing else reached is unscored, never a miss."""
    lines = [*trace(cited('hist-5', body='- ' + PROSE), request([{'role': 'user', 'content': context('- ' + PROSE)}])),
             cut_map(('hist-1', 'chat-01', [H_OLD]))]
    row = rescore.episode(update_case(), lines, completion=True, answer='{"place": "main hall"}')
    assert row['reach_surfaces']['cited'] is None and row['reach_surfaces']['assembled'] is None
    assert row['reach'] is None


def test_an_unresolved_citation_does_not_unscore_a_question_every_gold_message_reached(rescore):  # noqa: F811
    body = '\n'.join(['- ' + PROSE, quotation('hist-5', H_NEW, NEW)])
    lines = [*trace(cited('hist-9', 'hist-5', body=body), request([{'role': 'user', 'content': context(body)}])),
             cut_map(('hist-1', 'chat-01', [H_OLD]))]
    row = rescore.episode(update_case(), lines, completion=True, answer='{"place": "main hall"}')
    assert row['reach_surfaces']['cited'] is None and row['reach']['all'] is True


def test_a_complete_map_without_the_cited_source_is_a_miss(rescore):  # noqa: F811
    """A map that was not cut lists every source the ledger holds: a cited source it lacks holds nothing now."""
    lines = [*trace(cited('hist-9', body='- ' + PROSE), request([{'role': 'user', 'content': context('- ' + PROSE)}])),
             LEDGER]
    row = rescore.episode(update_case(), lines, completion=False, answer='{"place": "east room"}')
    assert row['reach_surfaces']['cited']['any'] is False and row['reach']['any'] is False


def test_the_worker_maps_the_cited_sources_first_so_a_cut_keeps_them(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_worker, 'LEDGER_MAP_SOURCES', 2)
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    for index in range(4):
        ledger.record_source(f't-{index}', contact_id='p-01', session_id='s', messages=[
            {'role': 'user', 'content': f'message {index}'}])
    mapped = paired_worker.ledger_sources(tmp_path / 'turn-idempotency.db', first=['t-3', 'missing'])
    assert [row['turn_id'] for row in mapped['sources']] == ['t-3', 't-0'] and mapped['truncated'] is True
    monkeypatch.setattr(paired_worker, 'LEDGER_MAP_SOURCES', 4)
    whole = paired_worker.ledger_sources(tmp_path / 'turn-idempotency.db', first=['t-3'])
    assert [row['turn_id'] for row in whole['sources']] == ['t-3', 't-0', 't-1', 't-2'] and whole['truncated'] is False


def test_the_recorder_keeps_the_sources_the_assemblies_cited_latest_first():
    import asyncio
    from types import SimpleNamespace

    cited_now = {}
    recorder = native_memory_worker.route_recorder([], None, cited=cited_now)

    def body(*sources):
        import json
        return json.dumps({'sections': [{'id': 'm', 'body': 'x', 'citations': [
            {'source_id': source, 'source_version': 'v'} for source in sources]}]}).encode()

    async def assemble(*sources):
        async def iterate():
            yield body(*sources)
        response = SimpleNamespace(status_code=200, body_iterator=iterate(), headers={}, background=None)
        request = SimpleNamespace(method='POST', url=SimpleNamespace(path='/v1/host/context/assemble'))
        return await recorder(request, lambda _: _respond(response))

    async def _respond(response):
        return response

    asyncio.run(assemble('t-1', 't-2'))
    asyncio.run(assemble('t-1'))
    assert paired_worker.latest_cited(cited_now) == ['t-1', 't-2']
