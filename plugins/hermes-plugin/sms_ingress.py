"""Owner identity at the trusted local SMS/RCS bridge boundary.

The bridge must generate transportAttestation after verifying the signed webhook body,
timestamp and nonce. Never call this helper on a caller-supplied webhook body: attestation
fields are trusted only in the configured local bridge's drained message envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
import ipaddress
import math
import re
import time
from urllib.parse import urlsplit

from .client import load_settings


def owner_sms_sender(message: Mapping, *, bridge_url: str, settings=None, now=None) -> str | None:
    """Return the configured owner for authenticated direct SMS."""
    return owner_phone_sender(message, platform="sms", bridge_url=bridge_url, settings=settings, now=now)


def owner_rcs_sender(message: Mapping, *, bridge_url: str, settings=None, now=None) -> str | None:
    """Return the configured owner for authenticated direct RCS."""
    return owner_phone_sender(message, platform="rcs", bridge_url=bridge_url, settings=settings, now=now)


def owner_phone_sender(message: Mapping, *, platform: str, bridge_url: str, settings=None, now=None) -> str | None:
    """Return the configured owner only for a fresh, signed, direct phone message from a local bridge.

    Unsigned, stale, malformed, group, non-owner and remotely bridged messages remain guests.
    This helper grants identity only; normal owner authorization still governs every action.
    """
    try:
        if platform not in {"sms", "rcs"}:
            return None
        endpoint = urlsplit(bridge_url)
        host = endpoint.hostname or ""
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        if endpoint.scheme != "http" or not local or endpoint.username or endpoint.password:
            return None
        configured = settings or load_settings()
        owner = getattr(configured, f"owner_{platform}_handle")()
        sender = message.get("senderId")
        if not owner or sender != owner or message.get("chatId") != sender or message.get("isGroup") is not False:
            return None
        proof = message.get("transportAttestation")
        methods = ("ecdsa-p256-sha256",) if platform == "rcs" else ("hmac-sha256", "ecdsa-p256-sha256")
        if not isinstance(proof, Mapping) or proof.get("verified") is not True or proof.get("method") not in methods:
            return None
        if not isinstance(proof.get("keyId"), str) or not proof["keyId"].strip() or len(proof["keyId"]) > 128:
            return None
        if not isinstance(proof.get("nonce"), str) or not re.fullmatch(r"[A-Za-z0-9._:-]{16,128}", proof["nonce"]):
            return None
        if not isinstance(proof.get("bodySha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", proof["bodySha256"]):
            return None
        observed = time.time() if now is None else now
        received, event = proof.get("receivedAt"), proof.get("eventTime")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (observed, received, event)):
            return None
        if not -30 <= received - event <= 300 or not -30 <= observed - received <= 300:
            return None
        return sender
    except (AttributeError, OSError, TypeError, ValueError):
        return None
