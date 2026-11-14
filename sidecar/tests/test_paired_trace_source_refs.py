"""The paired trace records which sources each assembled context cited, not only the route and its status.

A context_route event carried the method, the path and the status. Whether the evidence that answers a probe
reached the model is scored structurally: the gold messages against what the injected memory cited. The citations
of each section ``/v1/host/context/assemble`` returned are now recorded in its trace event (identifiers only, never
text), and the response reaches the host unchanged.
"""
import hashlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification.paired_trace import DiagnosticTrace

SECTIONS = {'sections': [
    {'id': 'colonymind-memory', 'title': 'Relevant Memories', 'body': '- a remembered line', 'priority': 90,
     'citations': [{'source_id': 'turn-7', 'source_version': 'v1'}, {'source_id': 'turn-9', 'source_version': 'v2'}]},
    {'id': 'colonymind-status', 'title': 'Status', 'body': 'quiet', 'priority': 10, 'citations': None}],
    'notices': None}


def served(status=200, payload=SECTIONS):
    lines, routes = [], []
    trace = DiagnosticTrace(sink=lines.append)
    app = FastAPI()
    app.middleware('http')(worker.route_recorder(routes, trace))

    @app.post('/v1/host/context/assemble')
    async def assemble():
        from fastapi.responses import JSONResponse
        return JSONResponse(payload, status_code=status)

    @app.post('/v1/host/turns/sync')
    async def sync():
        return {'ok': True}
    return TestClient(app), lines, routes


def events(lines):
    return [json.loads(line) for line in lines]


def test_an_assembled_contexts_citations_are_recorded_per_section_and_the_response_is_unchanged():
    client, lines, routes = served()
    response = client.post('/v1/host/context/assemble', json={})
    assert response.status_code == 200 and response.json() == SECTIONS
    event, = events(lines)
    assert event['kind'] == 'context_route'
    body = SECTIONS['sections'][0]['body']
    assert event['data'] == {'method': 'POST', 'path': '/v1/host/context/assemble', 'status': 200,
                             'source_refs': [{'section': 'colonymind-memory',
                                              'refs': [{'source_id': 'turn-7', 'source_version': 'v1'},
                                                       {'source_id': 'turn-9', 'source_version': 'v2'}],
                                              'body_sha256': hashlib.sha256(body.encode()).hexdigest(),
                                              'body_chars': len(body), 'messages': []}]}
    assert routes == [{'path': '/v1/host/context/assemble', 'status': 200}]
    assert 'remembered line' not in lines[0]        # identifiers only, never the text


def test_other_routes_and_failed_assemblies_record_what_they_did_before():
    client, lines, routes = served(status=503, payload={'detail': 'down'})
    assert client.post('/v1/host/context/assemble', json={}).json() == {'detail': 'down'}
    assert client.post('/v1/host/turns/sync', json={}).json() == {'ok': True}
    assert [event['data'] for event in events(lines)] == [
        {'method': 'POST', 'path': '/v1/host/context/assemble', 'status': 503},
        {'method': 'POST', 'path': '/v1/host/turns/sync', 'status': 200}]


def test_a_citation_carries_only_its_identifiers():
    odd = {'sections': [{'id': 'm', 'body': 'x', 'citations': [
        {'source_id': 't-1', 'source_version': 'v', 'content': 'secret words', 'score': 0.9}, 'not a ref', {}]}]}
    client, lines, _ = served(payload=odd)
    client.post('/v1/host/context/assemble', json={})
    assert [{key: value for key, value in section.items() if key in ('section', 'refs')}
            for section in events(lines)[0]['data']['source_refs']] == [
        {'section': 'm', 'refs': [{'source_id': 't-1', 'source_version': 'v'}]}]
