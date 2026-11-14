"""The MCP process can claim attestations and ask, and has no other authority."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from colonymind.api.credentials import validate_credentials, secret_digest, CredentialTable, describe
from colonymind.config import ConfigError
from colonymind.mind.tool_origins import ToolOrigins
from test_hands_permit_answers import (rig, ORIGIN_SECRET, LANE_SECRET, GUEST_SECRET, HUB_SECRET,
                                     RCS_ONLY_SECRET)  # noqa: F401
from test_hands_permits import fx  # noqa: F401
from test_mind_loop import pinned_clock  # noqa: F401

REGISTER, CLAIM = "/v1/mind/tool-origins", "/v1/mind/tool-origins/claim"
AUTH = {"Authorization": "Bearer " + ORIGIN_SECRET}
BINDING = {"tool": "google_send", "args_hash": "a" * 64}
STAMP = {**BINDING, "origin": "owner_channel", "channel": "companion", "turn_id": "turn",
         "session_id": "session", "untrusted_addresses": ["read@example.test"]}


async def test_register_claim_single_use_and_unknown(rig):
    assert (await rig.client.post(REGISTER, json=STAMP)).status_code == 200
    claimed = await rig.client.post(CLAIM, json=BINDING, headers=AUTH)
    assert claimed.status_code == 200, claimed.text
    assert claimed.json() == {"origin": "owner_channel", "channel": "companion", "turn_id": "turn",
                              "verified": True, "untrusted_addresses": ["read@example.test"]}
    assert (await rig.client.post(CLAIM, json=BINDING, headers=AUTH)).status_code == 404
    assert (await rig.client.post(CLAIM, json=BINDING | {"tool": "unknown"}, headers=AUTH)).status_code == 404


async def test_ttl_and_autonomy(rig):
    now = [100.0]
    rig.mind.tool_origins = ToolOrigins(clock=lambda: now[0])
    stamp = STAMP | {"origin": "autonomy", "run_id": "run"}
    assert (await rig.client.post(REGISTER, json=stamp)).status_code == 200
    now[0] += 119.999
    assert (await rig.client.post(CLAIM, json=BINDING, headers=AUTH)).json()["run_id"] == "run"
    assert (await rig.client.post(REGISTER, json=stamp)).status_code == 200
    now[0] += 120
    assert (await rig.client.post(CLAIM, json=BINDING, headers=AUTH)).status_code == 404


def test_bounded_copy_and_atomic_claim():
    origins = ToolOrigins(capacity=1)
    origins.register(STAMP)
    origins.register(STAMP | {"tool": "other"})
    assert origins.claim(BINDING) is None
    origins.register(STAMP)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: origins.claim(BINDING), range(8)))
    assert sum(result is not None for result in results) == 1


@pytest.mark.parametrize("secret", [LANE_SECRET, GUEST_SECRET, HUB_SECRET, RCS_ONLY_SECRET])
async def test_other_credentials_cannot_register_claim_or_ask(rig, secret):
    for path in (REGISTER, CLAIM, "/v1/mind/owner-actions"):
        response = await rig.client.post(path, json={}, headers={"Authorization": "Bearer " + secret})
        assert response.status_code == 403, (path, response.text)


async def test_narrow_grant_and_adapter_separation(rig):
    for path in (CLAIM, "/v1/mind/owner-actions"):
        assert (await rig.client.post(path, json={})).status_code == 403
    assert (await rig.client.post(REGISTER, json=STAMP, headers=AUTH)).status_code == 403
    for method, path in [("GET", "/v1/mind/state"), ("GET", "/v1/mind/permit-asks"),
                         ("GET", "/v1/mind/permit-asks/unknown"), ("GET", "/v1/secrets"),
                         ("POST", "/v1/mind/vault-captures"), ("POST", "/v1/mind/answers"),
                         ("POST", "/v1/mind/permit-asks/unknown/answer"),
                         ("POST", "/v1/host/turns/sync"), ("POST", "/v1/host/memory/read")]:
        assert (await rig.client.request(method, path, headers=AUTH)).status_code == 403, path


@pytest.mark.parametrize("change", [{"owner": ["companion"]}, {"owner": "any"},
    {"access": ["tool-origin", "manage"]}, {"access": ["tool-origin", "read"]},
    {"contacts": "all"}, {"scopes": ["*"]}])
def test_tool_origin_stands_alone(change):
    with pytest.raises(ConfigError):
        validate_credentials({"mcp": {"sha256": secret_digest("synthetic"), "access": ["tool-origin"], **change}})


def test_describe_narrow_credential():
    grant = CredentialTable.from_config({"mcp": {"sha256": secret_digest("synthetic"),
                                                 "access": ["tool-origin"]}}).get("mcp")
    assert grant.owner == () and grant.access == frozenset({"tool-origin"})
    assert "claim tool origins" in describe(grant)


def test_instance_key_cannot_receive_tool_origin():
    with pytest.raises(ConfigError, match="tool-origin stands alone"):
        validate_credentials({"api-key": {"owner": [], "contacts": "all", "access": ["tool-origin"]}})


@pytest.mark.parametrize("change", [{"body": "private"}, {"verified": True}, {"origin": "owner"},
    {"args_hash": "bad"}, {"untrusted_addresses": ["bad"]}, {"origin": "autonomy"},
    {"run_id": "spoof"}])
async def test_invalid_registration_does_not_create_a_claim(rig, change):
    assert (await rig.client.post(REGISTER, json=STAMP | change)).status_code == 422
    assert (await rig.client.post(CLAIM, json=BINDING, headers=AUTH)).status_code == 404


@pytest.mark.parametrize("owner_last", [False, True])
def test_identical_inflight_calls_cannot_borrow_another_turn(owner_last):
    origins = ToolOrigins()
    guest = STAMP | {"origin": "other", "session_id": "guest-session", "turn_id": "guest-turn"}
    first, newest = (guest, STAMP) if owner_last else (STAMP, guest)
    origins.register(first)
    origins.register(newest)
    claimed = origins.claim(BINDING)
    # Stdio cannot carry a nonce. Keep the newest metadata, but require a tap for ambiguity.
    assert claimed["origin"] == "other"
    assert claimed["turn_id"] == newest["turn_id"]
    assert claimed["untrusted_addresses"] == ["*"]
    assert origins.claim(BINDING) is None


@pytest.mark.parametrize("change", [{"session_id": "another-session"}, {"turn_id": "another-turn"},
    {"untrusted_addresses": []}, {"origin": "autonomy", "run_id": "run"}])
def test_conflicts_stay_downgraded_until_claim_or_expiry(change):
    origins = ToolOrigins()
    origins.register(STAMP)
    origins.register(STAMP | change)
    origins.register(STAMP | change)
    claimed = origins.claim(BINDING)
    assert claimed["origin"] == "other" and "run_id" not in claimed
    assert claimed["untrusted_addresses"] == ["*"]
    origins.register(STAMP)
    assert origins.claim(BINDING)["origin"] == "owner_channel"


async def test_rejected_registration_invalidates_an_older_owner_record(rig):
    assert (await rig.client.post(REGISTER, json=STAMP)).status_code == 200
    assert (await rig.client.post(REGISTER, json=STAMP | {"origin": "invalid"})).status_code == 422
    assert (await rig.client.post(CLAIM, json=BINDING, headers=AUTH)).status_code == 404


async def test_address_overflow_survives_the_claim_wire(rig):
    assert (await rig.client.post(REGISTER, json=STAMP | {"untrusted_addresses": ["*"]})).status_code == 200
    claimed = (await rig.client.post(CLAIM, json=BINDING, headers=AUTH)).json()
    assert claimed["untrusted_addresses"] == ["*"]
