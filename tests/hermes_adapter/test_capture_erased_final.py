"""A turn the sidecar refuses because its source was erased is final in the outbox, and its text goes with it."""

from copy import deepcopy
import json
import time
from types import SimpleNamespace

import pytest

from colonymind_hermes.capture import Capture, SessionMap, TurnOutbox, checkpoint, deliver
from colonymind_hermes.client import Settings, SidecarUnavailable

USER, ASSISTANT = "synthetic user words 4471", "synthetic assistant words 9026"
MESSAGES = [{"role": "user", "content": USER}, {"role": "assistant", "content": ASSISTANT}]
ERASED = {"accepted": False, "source_recorded": False, "continuity_updated": False, "skipped_reason": "source_erased"}
ERASED_WITH_WORK = {**ERASED, "work_recorded": False, "work_skipped_reason": "ineligible_origin"}
SETTLED = "erased: source_erased"
WORK = {"version": 1, "root_turn_id": "turn-1", "parent_turn_id": "", "origin_ref": "", "operations": [],
        "children": [], "truncated": False}
V1, V2 = ("POST", "/v1/host/turns/sync"), ("PUT", "/v2/host/turns/turn-1")


class Sidecar:
    """Answers every turn alike, or one turn id its own way, and keeps what it was sent."""

    def __init__(self, answer=None, status=200, *, answers=None, failure=None):
        self.calls, self.failure = [], failure
        self.default, self.answers = (status, {"accepted": True} if answer is None else answer), answers or {}

    def resolve_contact(self, *args, **kwargs):
        raise AssertionError("these turns name no sender")

    def request(self, method, path, **kwargs):
        body = deepcopy(kwargs["json"])
        self.calls.append((method, path, body))
        if self.failure is not None:
            raise self.failure
        status, answer = self.answers.get((body.get("context") or {}).get("turn_id"), self.default)
        return SimpleNamespace(status_code=status, is_success=200 <= status < 300, json=lambda: answer)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)

    def put(self, path, **kwargs):
        return self.request("PUT", path, **kwargs)


@pytest.fixture
def box(tmp_path):
    settings = Settings("http://unreachable.invalid", tmp_path / "api.key", "test-key", tmp_path, tmp_path,
                        tmp_path / "outbox.sqlite3")
    sessions, outbox = SessionMap(settings, Sidecar()), TurnOutbox(settings.outbox_path)
    return SimpleNamespace(settings=settings, sessions=sessions, outbox=outbox, drain=lambda sidecar: outbox.drain(
        lambda payload: deliver(sidecar, sessions, settings, payload)))


def turn(ident="turn-1", **extra):
    return {"session_id": "session-1", "turn_id": ident, "task_id": "", "platform": "cli", "sender_id": "",
            "user_message": USER, "assistant_message": ASSISTANT, "model": "", **extra}


def sql(outbox, statement, *values):
    connection = outbox._connect()
    try:
        return connection.execute(statement, values).fetchall()
    finally:
        connection.close()    # an open connection would keep the write-ahead log, and old pages in it


def stored(outbox):
    """Every byte the outbox has on disk, its write-ahead log included."""
    return b"".join(path.read_bytes() for path in sorted(outbox.path.parent.glob(outbox.path.name + "*")))


def by_id(outbox):
    return {row["turn_id"]: row for row in outbox.rows()}


# -- new: the erased answer settles the row, and the row keeps nothing of the turn --------------------------------

@pytest.mark.parametrize("answer,extra", [(ERASED, {}), (ERASED_WITH_WORK, {"work": WORK})], ids=["plain", "work"])
@pytest.mark.parametrize("version,status,sent", [(1, 200, V1), (2, 200, V2), (2, 201, V2)],
                         ids=["v1-200", "v2-200", "v2-201"])
def test_an_erased_turn_is_settled_by_the_pass_that_hears_it_and_never_posted_again(box, version, status, sent,
                                                                                  answer, extra):
    box.outbox.enqueue("turn-1", turn(transport_version=version, **extra))
    sidecar = Sidecar(answer, status)
    drained = box.drain(sidecar)
    row, = box.outbox.rows()
    assert row["state"] == "delivered", "the erased turn is still pending"
    assert (drained, row["attempts"], row["last_error"], box.outbox.pending_count()) == (1, 1, SETTLED, 0)
    assert [call[:2] for call in sidecar.calls] == [sent] and sidecar.calls[0][2].get("work") == extra.get("work")
    sql(box.outbox, "UPDATE turn_outbox SET lease_expires_at = 0")    # as if every back-off had run out
    assert box.drain(sidecar) == 0 and len(sidecar.calls) == 1
    assert box.outbox.rows() == [row]


def test_an_erased_turn_leaves_none_of_its_text_in_the_outbox_file(box):
    box.outbox.enqueue("turn-1", turn())
    assert USER.encode() in stored(box.outbox) and ASSISTANT.encode() in stored(box.outbox)
    box.drain(Sidecar(ERASED))
    row, = box.outbox.rows()
    assert row["payload"] == {}, "the erased turn's text is still in its outbox row"
    assert (row["state"], row["last_error"]) == ("delivered", SETTLED)
    held = stored(box.outbox)
    assert USER.encode() not in held and ASSISTANT.encode() not in held


def test_an_erased_checkpoint_leaves_none_of_its_messages_in_the_outbox_file(box):
    receipt = checkpoint(MESSAGES, session_id="session-1", contact_id="p-01", outbox=box.outbox)
    assert USER.encode() in stored(box.outbox) and ASSISTANT.encode() in stored(box.outbox)
    sidecar = Sidecar(ERASED)
    box.drain(sidecar)
    row, = box.outbox.rows()
    assert sidecar.calls[0][2]["checkpoint_messages"] == MESSAGES
    assert row["payload"] == {}, "the erased checkpoint's messages are still in its outbox row"
    assert (row["turn_id"], row["state"], row["last_error"]) == (receipt["turn_id"], "delivered", SETTLED)
    held = stored(box.outbox)
    assert USER.encode() not in held and ASSISTANT.encode() not in held


def test_a_captured_turn_with_a_work_part_is_settled_and_nothing_sends_it_again(box):
    capture = Capture(box.sessions, box.outbox, box.settings)
    values = {"session_id": "session-1", "turn_id": "turn-1", "platform": "cli", "sender_id": "",
              "task_id": "task-1", "user_message": USER}
    capture.pre_llm_call(**values)
    capture.work.post_tool_call(session_id="session-1", turn_id="turn-1", tool_call_id="call-1",
                                tool_name="write_file", args={"path": "example.txt", "content": "synthetic contents"},
                                result={"verified": True, "bytes_written": 18}, status="ok")
    capture.post_llm_call(**values, assistant_response=ASSISTANT)
    work = by_id(box.outbox)["turn-1"]["payload"]["work"]
    assert work["root_turn_id"] == "turn-1" and work["operations"]
    assert USER.encode() in stored(box.outbox) and ASSISTANT.encode() in stored(box.outbox)
    sidecar = Sidecar(ERASED_WITH_WORK)
    box.drain(sidecar)
    row = by_id(box.outbox)["turn-1"]
    assert row["state"] == "delivered", "the erased turn is still pending"
    assert (row["payload"], row["last_error"], box.outbox.pending_count()) == ({}, SETTLED, 0)
    assert [call[:2] for call in sidecar.calls] == [V1] and sidecar.calls[0][2]["work"] == work
    held = stored(box.outbox)
    assert USER.encode() not in held and ASSISTANT.encode() not in held
    sql(box.outbox, "UPDATE turn_outbox SET lease_expires_at = 0")
    assert box.drain(sidecar) == 0 and len(sidecar.calls) == 1


def test_a_turn_retried_many_times_before_is_settled_at_its_next_attempt(box):
    box.outbox.enqueue("turn-1", turn())
    sql(box.outbox, "UPDATE turn_outbox SET attempts = 5000, lease_expires_at = ?", time.time() - 1)
    sidecar = Sidecar(ERASED)
    drained = box.drain(sidecar)
    row, = box.outbox.rows()
    assert row["state"] == "delivered", "the erased turn is still pending"
    assert (drained, row["attempts"], row["last_error"], row["payload"]) == (1, 5001, SETTLED, {})
    assert len(sidecar.calls) == 1


def test_an_erased_turn_does_not_end_the_pass(box):
    box.outbox.enqueue("turn-1", turn())
    box.outbox.enqueue("turn-2", turn("turn-2"))
    sidecar = Sidecar(answers={"turn-1": (200, ERASED)})
    drained = box.drain(sidecar)
    first, second = (by_id(box.outbox)[ident] for ident in ("turn-1", "turn-2"))
    assert first["state"] == "delivered", "the erased turn is still pending"
    assert (drained, first["last_error"]) == (2, SETTLED)
    assert (second["state"], second["last_error"], second["payload"]) == ("delivered", "", {})
    assert [call[2]["context"]["turn_id"] for call in sidecar.calls] == ["turn-1", "turn-2"]


def test_start_up_keeps_an_erased_row_empty_and_it_is_no_rejected_row(box):
    box.outbox.enqueue("turn-1", turn())
    box.drain(Sidecar(ERASED))
    restarted = TurnOutbox(box.outbox.path)
    row, = restarted.rows()
    assert row["state"] == "delivered", "the erased turn is still pending"
    assert (row["payload"], row["last_error"]) == ({}, SETTLED)
    # The start-up scrub covers the row: a payload on it would be dropped, as on any delivered row but a rejected one.
    sql(restarted, "UPDATE turn_outbox SET payload_json = ?", json.dumps(turn()))
    assert USER.encode() in stored(restarted)
    assert TurnOutbox(box.outbox.path).rows() == [row]
    assert USER.encode() not in stored(restarted)
    assert restarted.refused_count() == 0
    restarted.prune(keep_delivered=0)
    assert restarted.rows() == []


# -- kept: every other answer and failure leaves the row as before ------------------------------------------------

def failed_pass(box, sidecar, version=1):
    """One pass over two queued turns that delivers neither: both stay pending with their text, the first backed off."""
    queued = {ident: turn(ident, transport_version=version) for ident in ("turn-1", "turn-2")}
    for ident, payload in queued.items():
        box.outbox.enqueue(ident, payload)
    before = time.time()
    assert box.drain(sidecar) == 0
    after = time.time()
    found = by_id(box.outbox)
    assert {ident: (row["state"], row["payload"]) for ident, row in found.items()} == {
        ident: ("pending", payload) for ident, payload in queued.items()}
    (lease,), = sql(box.outbox, "SELECT lease_expires_at FROM turn_outbox WHERE turn_id = 'turn-1'")
    assert before + 2.0 <= lease <= after + 2.0    # the first back-off
    return found["turn-1"], found["turn-2"]


@pytest.mark.parametrize("version,sent", [(1, V1), (2, V2)], ids=["v1", "v2"])
def test_a_server_error_is_retried_as_before(box, version, sent):
    sidecar = Sidecar({"detail": {"code": "turn_ingestion_ambiguous"}}, 503)
    first, second = failed_pass(box, sidecar, version)
    assert (first["attempts"], first["last_error"]) == (1, "")
    assert second["attempts"] == 0 and [call[:2] for call in sidecar.calls] == [sent]    # the pass ended there


@pytest.mark.parametrize("status,answer", [
    (202, {"accepted": False, "skipped_reason": "identical_retry_in_progress"}),
    (202, {"accepted": False, "skipped_reason": "source_input_parent_pending"}),
    (200, {"accepted": False, "skipped_reason": "work_source_erased"}),    # not a turn answer: only the exact one is final
    (200, {"accepted": False}),
], ids=["retry-in-progress", "parent-pending", "near-miss", "no-reason"])
def test_another_refusal_is_retried_as_before(box, status, answer):
    sidecar = Sidecar(answer, status)
    first, second = failed_pass(box, sidecar)
    assert (first["attempts"], first["last_error"]) == (1, "")
    assert second["attempts"] == 0 and len(sidecar.calls) == 1    # the pass ended there


@pytest.mark.parametrize("failure,posts", [(SidecarUnavailable("down"), 1), (RuntimeError("synthetic"), 2)],
                         ids=["sidecar-down", "exception"])
def test_a_failed_post_is_retried_as_before(box, failure, posts):
    sidecar = Sidecar(failure=failure)
    first, second = failed_pass(box, sidecar)
    assert (first["attempts"], first["last_error"]) == (1, type(failure).__name__)
    assert second["attempts"] == posts - 1 and len(sidecar.calls) == posts    # only a sidecar that is down ends it


@pytest.mark.parametrize("status,answer,reason", [
    (200, {"accepted": False, "skipped_reason": "owner_not_granted"},
     "rejected: attribution refused: owner_not_granted"),
    (422, {"detail": {"code": "invalid_turn"}}, "rejected: HTTP 422"),
], ids=["attribution", "http-422"])
def test_a_rejected_turn_still_keeps_its_payload(box, status, answer, reason):
    box.outbox.enqueue("turn-1", turn())
    assert box.drain(Sidecar(answer, status)) == 1
    row, = TurnOutbox(box.outbox.path).rows()
    assert (row["state"], row["last_error"], row["payload"]) == ("delivered", reason, turn())
