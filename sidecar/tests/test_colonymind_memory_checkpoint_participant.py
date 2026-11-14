"""The memory provider's pre-compress checkpoint keeps its participant through a transient lookup failure.

Before Hermes compresses a session, the provider commits the direct evidence as a checkpoint bound to the turn's
exact participant, resolved from the sender. The positive lookup is cached for a minute; past that a lookup the
sidecar cannot answer right now (unreachable, out of time, its breaker open, a server error) used to leave the
checkpoint with no participant, so it was deferred and the compressed turns' evidence was lost. It now keeps the
participant the same sender resolved to earlier in the same session. A definitive answer (the sidecar says the
sender has no participant) never falls back, and neither does another session or another sender.
"""
import types

import pytest

from test_colonymind_memory_provider import (_FakeHttpx, _FakeResponse, _RESOLVE, _install_session_context,
                                            _instance, _make_provider, provider_mod)  # noqa: F401  (fixtures)

MESSAGES = [{"role": "user", "content": "The invoice goes to the Lisbon office."},
            {"role": "assistant", "content": "Noted: the Lisbon office."}]


class Sidecar:
    """The resolve route: a contact, or a transient failure, or a definitive "no such participant"."""

    def __init__(self):
        self.mode = "ok"

    def __call__(self, request):
        if self.mode == "down":
            raise _FakeHttpx.HTTPError("sidecar unreachable")
        if self.mode == "busy":
            return _FakeResponse(status_code=503)
        if self.mode == "unknown":
            return {}
        if self.mode == "rejected":
            return _FakeResponse(status_code=404)
        if self.mode == "moved":
            return {"contact_id": "cid-split"}
        return {"contact_id": "cid-tg"}


@pytest.fixture
def world(provider_mod, monkeypatch, tmp_path):  # noqa: F811
    import colonymind_hermes.client as client
    from colonymind_hermes.capture import TurnOutbox
    outbox = tmp_path / "outbox.db"
    monkeypatch.setattr(client, "load_settings", lambda *args, **kwargs: types.SimpleNamespace(outbox_path=outbox))
    set_turn = _install_session_context(monkeypatch)
    set_turn(platform="telegram", sender="tg-1", chat="chat-1")
    sidecar = Sidecar()
    fake = _FakeHttpx(routes={_RESOLVE: sidecar, ("POST", "/v1/host/context/assemble"): {"sections": []}})
    provider = _make_provider(provider_mod, fake, monkeypatch)
    provider.initialize("s-1", platform="telegram")
    return types.SimpleNamespace(provider=provider, sidecar=sidecar, set_turn=set_turn,
                                 rows=lambda: [row["payload"] for row in TurnOutbox(outbox).rows()])


def expire(provider):
    """The minute the positive lookup is cached for has passed."""
    for key, (at, contact) in list(provider._handle_cache.items()):
        provider._handle_cache[key] = (at - 3600.0, contact)


def resolved_once(world):
    assert world.provider._resolve_handle("telegram", "tg-1") == "cid-tg"
    expire(world.provider)


@pytest.mark.parametrize("mode", ["down", "busy"])
def test_a_transient_lookup_failure_keeps_the_participant(world, mode):
    resolved_once(world)
    world.sidecar.mode = mode
    world.provider.on_pre_compress(MESSAGES, require_checkpoint=True)
    [payload] = world.rows()
    assert payload["contact_id"] == "cid-tg" and payload["session_id"] == "s-1"
    assert (payload["platform"], payload["sender_id"]) == ("telegram", "tg-1")
    assert world.provider._last_checkpoint["state"] == "pending"
    assert world.provider._last_checkpoint["participant"] == "kept"


def test_the_breaker_open_keeps_the_participant_too(world):
    resolved_once(world)
    for _ in range(3):
        world.provider._record_connection_failure()
    world.provider.on_pre_compress(MESSAGES)
    assert [payload["contact_id"] for payload in world.rows()] == ["cid-tg"]


def test_a_definitive_answer_never_falls_back(world):
    resolved_once(world)
    world.sidecar.mode = "unknown"
    world.provider.on_pre_compress(MESSAGES)
    assert world.rows() == [] and world.provider._last_checkpoint["state"] == "failed"


@pytest.mark.parametrize("answer", ["rejected", "unknown"])
def test_a_rejection_on_a_later_turn_is_never_undone_by_an_outage(world, answer):
    """The owner split the identity: the sender resolved once, then the sidecar says it has no participant. An
    outage on the next turn must not bring the rejected participant back."""
    resolved_once(world)
    world.provider.on_turn_start(2, "hello")
    world.sidecar.mode = answer
    assert world.provider._resolve_handle("telegram", "tg-1") is None
    world.provider.on_turn_start(3, "hello again")
    world.sidecar.mode = "down"
    world.provider.on_pre_compress(MESSAGES)
    assert world.rows() == [] and world.provider._last_checkpoint["state"] == "failed"


def test_a_rejection_reaches_every_session_and_compressed_copy(world):
    """The rejection is about the sender, so no session that saw the old participant keeps it: not the compressed
    copy, not the session it came from, however the rejection was learned."""
    resolved_once(world)
    world.provider.on_session_switch("s-1-compressed", parent_session_id="s-1", reason="compression")
    world.provider.on_session_switch("s-2")
    world.sidecar.mode = "rejected"
    assert world.provider._resolve_handle("telegram", "tg-1") is None
    world.sidecar.mode = "down"
    for session in ("s-1", "s-1-compressed", "s-2"):
        world.provider.on_session_switch(session)
        world.provider.on_turn_start(4, "hello")
        world.provider.on_pre_compress(MESSAGES)
        assert world.rows() == [], session


def test_a_sender_that_now_resolves_elsewhere_drops_the_old_participant(world):
    """After a split the sender resolves to another participant in one session: no other session keeps the old one
    for an outage."""
    resolved_once(world)
    world.provider.on_session_switch("s-1-compressed", parent_session_id="s-1", reason="compression")
    world.provider.on_session_switch("s-2")
    world.sidecar.mode = "moved"
    assert world.provider._resolve_handle("telegram", "tg-1") == "cid-split"
    expire(world.provider)
    world.sidecar.mode = "down"
    world.provider.on_turn_start(5, "hello")
    world.provider.on_pre_compress(MESSAGES)
    assert [payload["contact_id"] for payload in world.rows()] == ["cid-split"]
    for session in ("s-1", "s-1-compressed"):
        world.provider.on_session_switch(session)
        world.provider.on_turn_start(6, "hello")
        world.provider.on_pre_compress(MESSAGES)
        assert [payload["contact_id"] for payload in world.rows()] == ["cid-split"], session


def test_another_session_or_sender_never_inherits_the_participant(world):
    resolved_once(world)
    world.sidecar.mode = "down"
    world.provider.on_session_switch("s-2")
    world.provider.on_pre_compress(MESSAGES)
    assert world.rows() == []
    world.provider.on_session_switch("s-1")
    world.set_turn(platform="telegram", sender="tg-2", chat="chat-1")
    world.provider.on_pre_compress(MESSAGES)
    assert world.rows() == []


def test_a_compressed_session_keeps_its_participant_under_the_new_id(world):
    """Hermes compacts under a new session id and the same turn goes on: the participant goes with it."""
    resolved_once(world)
    world.provider.on_session_switch("s-1-compressed", parent_session_id="s-1", reason="compression")
    world.sidecar.mode = "down"
    world.provider.on_pre_compress(MESSAGES)
    [payload] = world.rows()
    assert payload["contact_id"] == "cid-tg" and payload["session_id"] == "s-1-compressed"


def test_a_lookup_that_answers_is_used_and_nothing_is_kept_without_one(world):
    world.provider.on_pre_compress(MESSAGES)
    [payload] = world.rows()
    assert payload["contact_id"] == "cid-tg" and "participant" not in world.provider._last_checkpoint
    fresh = world.provider
    fresh._handle_cache.clear()
    fresh._session_participants.clear()
    world.sidecar.mode = "down"
    fresh.on_pre_compress(MESSAGES + [{"role": "user", "content": "And the copy to Porto."}])
    assert len(world.rows()) == 1 and fresh._last_checkpoint["state"] == "failed"
