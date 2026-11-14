"""Tests for Phase 2 — sidecar subsystems broadcast typed events."""

from __future__ import annotations


import pytest

from colonymind.events import broadcaster
from colonymind.events.broadcaster import emit, reset_broadcaster_for_tests


class _Collector:
    """Captures emitted events for assertions."""

    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)


@pytest.fixture
def collector():
    c = _Collector()
    reset_broadcaster_for_tests(c)
    yield c
    reset_broadcaster_for_tests(None)
    broadcaster._broadcast_fn = None  # force lazy-import on next call


def test_emit_wraps_payload_with_type_and_timestamp(collector):
    emit("custom", {"foo": 1})
    assert len(collector.events) == 1
    e = collector.events[0]
    assert e["type"] == "custom"
    assert e["payload"] == {"foo": 1}
    assert "occurred_at" in e


def test_emit_with_no_payload_defaults_to_empty_dict(collector):
    emit("pinged")
    assert collector.events[0]["payload"] == {}


def test_emit_swallows_broadcaster_exceptions(collector):
    def _boom(_event):
        raise RuntimeError("sink dead")
    reset_broadcaster_for_tests(_boom)
    emit("custom", {"foo": 1})  # must not raise

