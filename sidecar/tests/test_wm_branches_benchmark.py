"""Working-memory arms reach the same persisted records through the benchmark API."""
import json
from contextlib import closing
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from onekey import KEY
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import branches, host
from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification import native_memory_batch, paired, paired_container, paired_worker
from colonymind.turns import get_turn_idempotency_ledger
from test_qualification_paired_runner import fixture  # noqa: F401


@pytest.mark.parametrize("arm", ["full-working_memory", "all-on-working_memory", "wm.summary_only"])
def test_working_memory_arms_refuse_an_image_without_the_branch_worker(fixture, monkeypatch, arm):
    # Image capability tests isolate the unrelated filesystem permission preflight.
    monkeypatch.setattr(native_memory_batch, "preflight_output", lambda output: None)
    with pytest.raises(ValueError, match="working.memory"):
        paired.plan(fixture.output, native_binding="candidate", evidence_mode="controlled",
                    arms=["all-on", arm], **fixture.resources)


def test_working_memory_capability_is_inspected_and_frozen(fixture, monkeypatch):
    from colonymind.qualification import native_identity, native_memory_identity
    monkeypatch.setattr(native_identity, "inspect_runtime", lambda: {})
    monkeypatch.setattr(native_memory_identity, "inspect_runtime", lambda: {})
    monkeypatch.setattr(native_memory_batch, "preflight_output", lambda output: None)
    payload = paired_worker.inspect_payload()
    capability = payload["working_memory"]
    original = paired_container.configuration

    def current_image(*args, **kwargs):
        config, recipe = original(*args, **kwargs)
        recipe["container_payload"]["working_memory"] = capability
        return config, recipe

    monkeypatch.setattr(paired_container, "configuration", current_image)
    manifest = paired.plan(fixture.output, native_binding="candidate", evidence_mode="controlled",
                           arms=["all-on", "all-on-working_memory", "wm.summary_only"], **fixture.resources)
    assert manifest["comparison"]["profiles"]["wm.summary_only"]["wm_summary_only"] is True
    assert manifest["comparison"]["profiles"]["all-on-working_memory"]["minus_working_memory"] is True


@pytest.mark.parametrize("profile", [
    {"plugin": False, "full": True, "minus_working_memory": True},
    {"plugin": True, "minus_working_memory": True},
])
def test_custom_working_memory_ablation_cannot_be_a_noop(profile):
    with pytest.raises(ValueError, match="working.memory"):
        paired.validate_profiles({"custom-wm": profile})


from colonymind.memory import curated
@pytest.mark.asyncio
@pytest.mark.parametrize('arm', ['colonymind', 'all-on', 'all-on-working_memory', 'wm.summary_only'])
async def test_benchmark_routes_share_host_records_and_apply_the_arm(tmp_path, monkeypatch, arm):
    profile = paired.PROFILES[arm]
    switches = paired_worker.mind_switches(profile)
    section = worker.mind_section(switches)
    instance = tmp_path / 'colonymind-instance'
    instance.mkdir()
    (instance / 'colonymind.yaml').write_text(json.dumps(worker.instance_config('owner', section, profile=profile)))
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'owner')
    ledger = get_turn_idempotency_ledger(tmp_path / 'memory-state')
    ledger.record_source('parent-turn', contact_id='owner', session_id='parent', messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'The sample is blue.'}]}])
    calls = []

    class Router:

        async def complete(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return SimpleNamespace(content='The sample is blue.', tool_calls=[])
    monkeypatch.setattr(host, '_llm_router', Router())
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=worker.benchmark_credentials())
    worker.mount_routes(app, mind=bool(switches))
    original_store = branches._store
    with worker.working_memory_store(tmp_path):
        store = branches._store()
        assert store.ledger is ledger
        assert store.enabled is (arm != 'all-on-working_memory')
        assert store.summary_only is (arm == 'wm.summary_only')
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test', headers={'Authorization': 'Bearer ' + KEY}) as client:
            body = {'requester': 'owner', 'sender': {'platform': 'cli'}, 'session_id': 'parent'}
            response = await client.post('/v1/memory/branches/session', json=body)
            if arm == 'all-on-working_memory':
                assert response.status_code == 409, response.text
                assert response.json()['detail']['code'] == 'disabled'
                return
            assert response.status_code == 200, response.text
            branch = response.json()['branch']
            assert [span['record_id'] for span in branch['spans']] == ['parent-turn']
            response = await client.post('/v1/memory/branches/consult', json={**body, 'handles': [branch['id']], 'question': 'What color?'})
            if arm == 'wm.summary_only':
                assert response.status_code == 409, response.text
                assert response.json()['detail']['code'] == 'summary_only'
                assert calls == []
            else:
                assert response.status_code == 200, response.text
                assert response.json()['answer'] == 'The sample is blue.'
                assert calls[0][1]['tools'] == []
                assert calls[0][1]['context']['task'] == 'consult'
            response = await client.post('/v1/memory/branches', json={'requester': 'owner', 'sender': {'platform': 'cli'}, 'parent_session': 'parent', 'child_session': 'child', 'brief': 'Read the sample', 'messages': [{'role': 'assistant', 'content': 'The sample is blue.'}]})
            assert response.status_code == 200, response.text
            span_ids = [span['record_id'] for span in response.json()['branch']['spans']]
            with closing(ledger._connect()) as conn:
                assert all((conn.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (key,)).fetchone() for key in span_ids))
            from test_curated_memory import historical_bytes
            _native_before_withdrawal = historical_bytes(ledger)
            curated.forget_sources(ledger, 'owner', source_ids=['parent-turn'])
            assert historical_bytes(ledger) == _native_before_withdrawal
            assert store.list_branches('parent', requester='owner') == []
            with closing(ledger._connect()) as conn:
                assert all(conn.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (key,)).fetchone() for key in span_ids)
    assert branches._store is original_store
