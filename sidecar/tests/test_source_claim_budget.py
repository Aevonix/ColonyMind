"""Configured extraction deadlines stay inside an owned, renewable job lease."""
import asyncio
from contextlib import contextmanager
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from colonymind.beliefs.source_claims import extraction_timeout_seconds, projection_timeout_seconds, extract_claims
from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.turns.idempotency import TurnIdempotencyLedger
from test_function_routing import config, router
from test_source_claim_projection import claim, Model


@pytest.fixture
def endpoint(monkeypatch):
    """Exercise the HTTP client and role deadlines without opening a network socket."""
    from colonymind.router import router as module

    @contextmanager
    def serve(*, content):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append({'payload': payload, 'authorization': request.headers.get('Authorization')})
            return httpx.Response(200, json={
                'id': 'neutral', 'object': 'chat.completion', 'created': 1, 'model': payload['model'],
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content(payload)},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}})

        monkeypatch.setattr(module, '_local_http_client',
                            lambda config, pinned_ip: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        yield 'http://127.0.0.1:1/v1', requests

    return serve


@pytest.mark.asyncio
@pytest.mark.parametrize('task_override', [False, True])
async def test_actual_role_request_uses_configured_outer_deadline(monkeypatch, task_override, endpoint):
    text = 'My office is in Alder.'
    observed = []
    original_wait_for = asyncio.wait_for

    async def inspect_wait_for(awaitable, timeout):
        if getattr(getattr(awaitable, 'cr_code', None), 'co_name', '') == 'complete':
            observed.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, 'wait_for', inspect_wait_for)
    def answer(payload):
        if 'proposals' in json.loads(payload['messages'][1]['content']):
            return json.dumps({'0': {'keep': True, 'reason': 'Controlled valid location.'}})
        return json.dumps([claim(text, 'Alder')])
    with endpoint(content=answer) as (url, requests):
        cfg = config(url, url, deadlineSeconds=80)
        cfg['functionRoles']['judging'] = ['interactive']
        if task_override:
            cfg['taskRoles'] = {'source_claim_extraction': 'reasoning'}
        r = router(cfg)
        rows, _ = await extract_claims(r, {'occurred_at': None}, {'role': 'user', 'content': text}, [])
        assert len(rows) == 1 and len(requests) == 2
        extraction_bound = 185 if task_override else 85
        assert requests[0]['payload']['model'] == ('strong-neutral' if task_override else 'fast-neutral')
        assert requests[1]['payload']['model'] == 'fast-neutral'  # Review keeps its own role.
        assert observed == [extraction_bound, 185]
        assert extraction_timeout_seconds(r) == extraction_bound
        assert projection_timeout_seconds(r) == extraction_bound + 185
        r.configure(config(url, url, deadlineSeconds=120))
        assert extraction_timeout_seconds(r) == 125


@pytest.mark.asyncio
async def test_memory_review_task_override_uses_selected_binding_and_deadline(monkeypatch, endpoint):
    text = 'My office is in Alder.'
    bounds = []
    original_wait_for = asyncio.wait_for

    async def observe_wait_for(awaitable, timeout):
        if getattr(getattr(awaitable, 'cr_code', None), 'co_name', '') == 'complete':
            bounds.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, 'wait_for', observe_wait_for)

    def answer(payload):
        if 'proposals' in json.loads(payload['messages'][1]['content']):
            return json.dumps({'0': {'keep': True, 'reason': 'Reported office location.'}})
        return json.dumps([claim(text, 'Alder')])

    with endpoint(content=answer) as (url, requests):
        cfg = config(url, url, timeoutSeconds=60, deadlineSeconds=60)
        cfg['functionRoles']['extraction']['candidates'] = ['interactive']
        cfg['functionRoles']['judging'] = ['deliberate']
        cfg['modelPool']['interactive']['maxTokens'] = 2200
        cfg['taskRoles'] = {'source_claim_extraction': 'extraction',
                            'source_claim_review': 'extraction'}
        r = router(cfg)
        rows, _ = await extract_claims(r, {'occurred_at': None}, {'role': 'user', 'content': text}, [])
        assert len(rows) == 1
        assert [call['payload']['model'] for call in requests] == ['fast-neutral', 'fast-neutral']
        assert [call['payload']['max_tokens'] for call in requests] == [2200, 1400]
        assert bounds == [65, 65]
        assert projection_timeout_seconds(r) == 130
        review = rows[0]['admission_review']['model_provenance']
        assert review['function_role'] == 'extraction' and review['binding'] == 'interactive'
        # Only this task changes selection. Other judgment and planning callers
        # retain their declared model, and configuration reload restores defaults.
        assert r.routing_status()['roles']['judging'] == ['deliberate']
        assert r.routing_status()['roles']['planning'] == ['deliberate', 'interactive']
        cfg['taskRoles'].pop('source_claim_review')
        r.configure(cfg)
        assert projection_timeout_seconds(r) == 250
        rows, _ = await extract_claims(r, {'occurred_at': None}, {'role': 'user', 'content': text}, [])
        assert requests[-1]['payload']['model'] == 'strong-neutral'
        assert rows[0]['admission_review']['model_provenance']['function_role'] == 'judging'


@pytest.mark.asyncio
async def test_long_multi_message_job_renews_after_role_reload(tmp_path, monkeypatch):
    from colonymind.beliefs import source_projection as module
    clock = [1000.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda: clock[0]))
    ledger = TurnIdempotencyLedger(tmp_path / 'turns.db')
    first, second = 'My office is in Alder.', 'My office is in Birch.'
    ledger.record_source('neutral', contact_id='person', session_id='sms', messages=[
        {'role': 'user', 'content': first}, {'role': 'user', 'content': second}])
    projection, competitor = SourceClaimProjection(ledger), SourceClaimProjection(ledger)
    r = router(config('http://127.0.0.1:1/v1', 'http://127.0.0.1:1/v1', deadlineSeconds=80))
    leases, waits = [], []
    original_wait_for = asyncio.wait_for

    async def inspect_wait_for(awaitable, timeout):
        waits.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, 'wait_for', inspect_wait_for)

    class SlowModel(Model):
        supports_function_routing = True
        function_deadline_seconds = r.function_deadline_seconds

        async def complete(self, messages, **kwargs):
            with sqlite3.connect(ledger.db_path) as conn:
                leases.append(conn.execute('SELECT lease_until FROM source_claim_jobs').fetchone()[0])
            # A simulated 61-second request outlives the old fixed lease. The
            # real SQLite competitor must not steal either active message.
            clock[0] += 61
            assert competitor.claim_job() is None
            r.configure(config('http://127.0.0.1:1/v1', 'http://127.0.0.1:1/v1', deadlineSeconds=120))
            return await super().complete(messages, **kwargs)

    model = SlowModel({first: claim(first, 'Alder'), second: claim(second, 'Birch')})
    assert await projection.process_one(model)
    assert waits == [270, 85, 185, 310, 125, 185]
    assert leases == [1300, 1300, 1462, 1462]
    assert projection.status('person')[0]['status'] == 'complete'
    assert projection.status('person')[0]['claim_count'] == 2


def test_legacy_and_old_function_adapters_keep_their_existing_bounds():
    assert extraction_timeout_seconds(SimpleNamespace()) == 20
    assert extraction_timeout_seconds(SimpleNamespace(supports_function_routing=True)) == 40

from colonymind.memory import curated

def test_old_lease_cannot_be_renewed_after_reclaim_or_forget(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turns.db')
    projection = SourceClaimProjection(ledger)
    ledger.record_source('neutral', contact_id='person', session_id='sms', messages=[
        {'role': 'user', 'content': 'My office is in Alder.'}])
    old = projection.claim_job()
    with sqlite3.connect(ledger.db_path) as conn:
        conn.execute('UPDATE source_claim_jobs SET lease_until=0')
    new = projection.claim_job()
    assert not projection.renew_job(old, 85)
    assert projection.renew_job(new, 85)
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, 'person', source_ids=['neutral'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert not projection.renew_job(new, 85)
