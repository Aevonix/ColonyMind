"""Signed bridge identity stays separate from SMS body claims and guest identity."""

from copy import deepcopy
import ipaddress
from types import SimpleNamespace

import pytest

from colonymind_hermes.sms_ingress import owner_phone_sender, owner_rcs_sender


OWNER = "+12025550123"
GUEST = "+12025550124"
NOW = 2_000_000_000


@pytest.fixture(params=["sms", "rcs"])
def envelope(request):
    return {
        "platform": request.param, "senderId": OWNER, "chatId": OWNER, "isGroup": False, "body": "approve",
        "transportAttestation": {
            "verified": True, "method": "ecdsa-p256-sha256" if request.param == "rcs" else "hmac-sha256", "keyId": "fixture-key",
            "receivedAt": NOW, "eventTime": NOW, "nonce": "fixture-nonce-00000001",
            "bodySha256": "a" * 64,
        },
    }


def resolve(envelope, owner=OWNER, bridge_url="http://localhost"):
    platform = envelope.get("platform", "sms") if isinstance(envelope, dict) else "sms"
    return owner_phone_sender(envelope, platform=platform, bridge_url=bridge_url,
                              settings=SimpleNamespace(owner_sms_handle=lambda: owner,
                                                       owner_rcs_handle=lambda: owner), now=NOW)


def test_signed_direct_owner_is_admitted(envelope):
    assert resolve(envelope) == OWNER


def test_phone_device_key_attestation_is_admitted(envelope):
    # The bridge's P-256 device-key path (the phone's AndroidKeyStore key): unpadded base64url nonce.
    envelope["transportAttestation"].update(
        method="ecdsa-p256-sha256", keyId="phone-primary", nonce="AAECAwQFBgcICQoLDA0ODw")
    assert resolve(envelope) == OWNER


@pytest.mark.parametrize("owner", ["", GUEST])
def test_unconfigured_or_changed_owner_is_not_admitted(envelope, owner):
    assert resolve(envelope, owner=owner) is None


@pytest.mark.parametrize(("field", "value"), [
    ("senderId", GUEST), ("chatId", GUEST), ("isGroup", True), ("isGroup", "false"),
    ("senderId", " " + OWNER), ("senderId", OWNER[1:]), ("transportAttestation", {"verified": True}),
])
def test_other_senders_groups_and_body_identity_flags_remain_guests(envelope, field, value):
    envelope[field] = value
    assert resolve(envelope) is None


@pytest.mark.parametrize(("field", "value"), [
    ("verified", False), ("verified", 1), ("method", "body-claim"), ("method", "ecdsa-p384-sha384"), ("keyId", ""),
    ("nonce", "short"), ("bodySha256", "bad-digest"), ("receivedAt", NOW - 301),
    ("eventTime", NOW - 301), ("eventTime", NOW + 31), ("receivedAt", float("nan")),
    ("receivedAt", True), ("eventTime", "2000000000"),
])
def test_unverified_malformed_or_stale_attestation_remains_guest(envelope, field, value):
    envelope["transportAttestation"][field] = value
    assert resolve(envelope) is None


@pytest.mark.parametrize("bridge_url", ["https://bridge.invalid", "http://bridge.invalid", "file:bridge", "http://user@localhost"])
def test_only_local_bridge_envelopes_can_authenticate_owner(envelope, bridge_url):
    assert resolve(envelope, bridge_url=bridge_url) is None


def test_no_input_is_mutated(envelope):
    before = deepcopy(envelope)
    resolve(envelope)
    assert envelope == before


@pytest.mark.parametrize("missing", ["senderId", "chatId", "isGroup", "transportAttestation"])
def test_incomplete_envelope_remains_guest(envelope, missing):
    envelope.pop(missing)
    assert resolve(envelope) is None


@pytest.mark.parametrize("missing", ["verified", "method", "keyId", "receivedAt", "eventTime", "nonce", "bodySha256"])
def test_incomplete_attestation_remains_guest(envelope, missing):
    envelope["transportAttestation"].pop(missing)
    assert resolve(envelope) is None


@pytest.mark.parametrize("malformed", [None, [], "body-only", 1])
def test_malformed_envelope_remains_guest(malformed):
    assert resolve(malformed) is None


@pytest.mark.parametrize("host", ["localhost", str(ipaddress.IPv4Address(2130706433)),
                                 "[" + str(ipaddress.IPv6Address(1)) + "]"])
def test_loopback_bridge_forms_are_supported(envelope, host):
    assert resolve(envelope, bridge_url="http://" + host) == OWNER


def test_rcs_requires_device_key_and_its_own_owner_binding(envelope):
    envelope["transportAttestation"]["method"] = "ecdsa-p256-sha256"
    settings = SimpleNamespace(owner_sms_handle=lambda: OWNER, owner_rcs_handle=lambda: "")
    assert owner_rcs_sender(envelope, bridge_url="http://localhost", settings=settings, now=NOW) is None
    settings.owner_rcs_handle = lambda: OWNER
    assert owner_rcs_sender(envelope, bridge_url="http://localhost", settings=settings, now=NOW) == OWNER
    envelope["transportAttestation"]["method"] = "hmac-sha256"
    assert owner_rcs_sender(envelope, bridge_url="http://localhost", settings=settings, now=NOW) is None
