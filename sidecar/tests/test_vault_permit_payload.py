"""Capture schemas use the held-call ledger; typed values never pass through this API."""
import json
import uuid
from datetime import timedelta

import pytest

from colonymind.mind import permit_asks
from test_hands_permit_answers import rig, app_answer  # noqa: F401
from test_hands_permits import fx  # noqa: F401
from test_mind_loop import pinned_clock  # noqa: F401


def payload(kind="vault_save_login"):
    value = {"kind": kind, "origin": "https://marker-" + uuid.uuid4().hex + ".invalid",
            "site_hint": "marker-" + uuid.uuid4().hex, "ttl_s": 120,
            "save_toggle_default": kind != "vault_code",
            "fields": ["code"] if kind == "vault_code" else ["username", "password"]}
    if kind == "vault_fill_confirm":
        value.update(label="Example login", fields=[], save_toggle_default=False)
    return value


@pytest.mark.parametrize("kind", sorted(permit_asks.CAPTURE_KINDS))
async def test_create_view_answer_and_expiry(rig, kind):
    schema = payload(kind)
    result = await rig.client.post("/v1/mind/vault-captures", json={
        "authorization_id": str(uuid.uuid4()), "payload": schema})
    assert result.status_code == 200
    ident = result.json()["ask_id"]
    row = rig.mind.store.get(ident)
    assert row.kind == "note"  # never dispatch a kanban continuation
    assert row.expires_at == rig.mind.clock() + timedelta(seconds=120)
    view = (await rig.hub.get("/v1/mind/permit-asks/" + ident)).json()
    assert view["payload"] == schema
    assert [x["value"] for x in view["choices"]] == ["yes", "no"]
    assert (await app_answer(rig, ident)).status_code == 200
    state = (await rig.client.get("/v1/mind/vault-captures/" + ident)).json()
    assert state["open"] is False
    assert state["winner"]["choice"] == "yes"
    assert (await rig.client.get("/v1/mind/permit-asks/" + ident)).status_code == 403


@pytest.mark.parametrize("change", [
    {"kind": "other"}, {"origin": "http://example.invalid"},
    {"origin": "https://user:pass@example.invalid"}, {"origin": "https://example.invalid/path"},
    {"origin": "https://example.invalid?query"}, {"origin": "https://example.invalid:999999"},
    {"origin": "https://example.invalid/#"}, {"ttl_s": True}, {"ttl_s": 0}, {"ttl_s": 121},
    {"fields": ["password", "username"]}, {"fields": [{"name": "password", "value": ""}]},
    {"save_toggle_default": "yes"}, {"site_hint": "é" * 129},
])
def test_invalid_schema_is_finite_and_does_not_echo(change):
    with pytest.raises(ValueError, match="^invalid_capture_payload$"):
        permit_asks.validate_payload(payload() | change)


async def test_secret_rejection_and_credential_isolation(rig, caplog):
    secret = "marker-" + str(uuid.uuid4())
    schema = payload()
    replies = []
    for body in ({"authorization_id": str(uuid.uuid4()), "payload": schema, "password": secret},
                 {"authorization_id": str(uuid.uuid4()), "payload": schema | {"password": secret}}):
        reply = await rig.client.post("/v1/mind/vault-captures", json=body)
        assert reply.status_code == 422
        replies.append(reply.text)
    assert secret not in json.dumps(replies) + caplog.text
    assert not any("payload" in (r.context or {}) for r in rig.mind.store.intentions(limit=500))
    for client in (rig.hub,):
        response = await client.post("/v1/mind/vault-captures", json={
            "authorization_id": str(uuid.uuid4()), "payload": schema})
        assert response.status_code == 403


async def test_cancel_timeout_and_no_typed_answer(rig):
    row = permit_asks.create_capture(rig.mind, payload("vault_code"), str(uuid.uuid4()))
    reply = await rig.hub.post("/v1/mind/permit-asks/" + row.id + "/answer",
                              json={"choice": "no", "platform": "companion", "contact_id": rig.mind.owner_id,
                                    "code": True}, headers={"Idempotency-Key": str(uuid.uuid4())})
    assert reply.status_code == 422
    # Existing validation contract is untouched: the gateway never sends this request.
    assert (await app_answer(rig, row.id, "no")).status_code == 200
    row = permit_asks.create_capture(rig.mind, payload("vault_code"), str(uuid.uuid4()))
    rig.mind.store.update(row.id, expires_at=rig.mind.clock() - timedelta(seconds=1))
    assert (await app_answer(rig, row.id)).status_code == 409


def test_companion_phase_two_fill_payload_with_optional_site_hint():
    # Companion Vault._prompt builds this exact envelope for confirm_fill.
    schema = {"kind": "vault_fill_confirm", "origin": "https://example.invalid",
              "site_hint": "example.invalid", "label": "Example login", "ttl_s": 120,
              "save_toggle_default": False, "fields": []}
    assert permit_asks.validate_payload(schema) == schema
    without_hint = {key: value for key, value in schema.items() if key != "site_hint"}
    assert permit_asks.validate_payload(without_hint) == without_hint


@pytest.mark.parametrize("change", [{"fields": ["password"]}, {"fields": ["code"]},
    {"fields": [{"name": "password"}]}, {"password": "private"}, {"save_toggle_default": True},
    {"label": "a" * 121}, {"label": "password: synthetic-password"}, {"label": "bad\nlabel"},
    {"label": None}, {"label": ""}, {"ttl_s": 121}])
def test_fill_rejects_fields_extra_keys_and_invalid_labels(change):
    with pytest.raises(ValueError, match="^invalid_capture_payload$"):
        permit_asks.validate_payload(payload("vault_fill_confirm") | change)


async def test_fill_question_is_plain_yes_no(rig):
    schema = payload("vault_fill_confirm")
    row = permit_asks.create_capture(rig.mind, schema, str(uuid.uuid4()))
    state = (await rig.hub.get("/v1/mind/permit-asks/" + row.id)).json()
    assert state["terms"]["text"] == f"Fill your saved login for Example login on {schema['origin']}?"
    assert state["payload"]["fields"] == [] and len(state["choices"]) == 2
