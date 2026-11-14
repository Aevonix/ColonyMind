"""Recognize canonical voice imports by provenance, never by text similarity.

Only trusted Hermes row metadata with a confirmed original source receipt is
excluded. This does not modify/delete the original voice capture or its outbox.
"""
from collections.abc import Mapping
import re


def is_bridged_voice_row(message):
    if not isinstance(message, Mapping) or message.get("role") not in {"user", "assistant"}:
        return False
    meta = message.get("display_metadata")
    if not isinstance(meta, Mapping):
        return False
    receipt = meta.get("source_receipt")
    return (
        meta.get("bridge") == "voice.writeback.v1"
        and meta.get("origin") == "voice"
        and isinstance(meta.get("voice_exchange_id"), str) and bool(meta["voice_exchange_id"])
        and isinstance(meta.get("payload_digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", meta["payload_digest"]) is not None
        and meta.get("generation_complete") is True and meta.get("playback_complete") is True
        and isinstance(receipt, Mapping) and receipt.get("source_recorded") is True
        and all(isinstance(receipt.get(key), str) and bool(receipt[key]) for key in ("source_id", "turn_id"))
    )
