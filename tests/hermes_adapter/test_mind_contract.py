"""Both halves of the loop together: the sidecar's own ``/v1/mind`` over a real Mind, the body on stock kanban.

The other body tests run against ``FakeMind``, the body's reading of the
contract. Here the sidecar's router (``colonymind.api.routers.mind``) serves a
real ``colonymind.mind.Mind`` over real stores in a temporary state directory,
and the plugin's body, tools and ``/mind`` command run against it in fresh
interpreters with stock ``hermes_cli.kanban_db``. What one half sends is what
the other half parses (docs/HERMES-ADAPTER.md, docs/MIND.md).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import yaml

from conftest import API_KEY, MIND_PRELUDE, OWNER, build_home, probe, refused
from test_answers_plugin import GATEWAY_CODE
from test_tools_commands import TOOL_CODE

pytestmark = pytest.mark.requires_sidecar


class Contacts:
    """The contact store the Mind asks about recipients and their handles."""

    def __init__(self, owner):
        self.owner = owner
        self.records = {owner: {"contact_id": owner, "may_contact": "auto"},
                        "p-03": {"contact_id": "p-03", "may_contact": "ask"}}

    async def get(self, contact_id):
        record = self.records.get(contact_id)
        return SimpleNamespace(to_dict=lambda: record) if record else None

    async def get_handles(self, contact_id):
        if contact_id == "p-03":
            return [SimpleNamespace(gateway="telegram", address="2003", is_primary=True, verified=True)]
        return []

    async def resolve_handle(self, gateway, address):
        return SimpleNamespace(contact_id="p-03") if address == "2003" else None

    async def resolve_messaging_handle(self, gateway, address):
        """Who a sender is (the answer route checks the sender of an owner message again)."""
        people = {("telegram", "1001"): self.owner, ("telegram", "2003"): "p-03"}
        contact = people.get((gateway, address))
        return SimpleNamespace(contact_id=contact) if contact else None


@pytest.fixture
def real_sidecar(tmp_path, monkeypatch):
    """The real router and Mind on 127.0.0.1, with a clock the test can shift."""
    from fastapi import FastAPI
    import uvicorn
    from colonymind.api.credentials import INSTANCE_CREDENTIAL, CredentialTable
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import mind as mind_router
    from colonymind.commitments.store import CommitmentStore
    from colonymind.feedback import TypeFeedbackStore
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.mind import Mind
    from colonymind.self_model.expectations import ExpectationEngine, ExpectationStore
    from colonymind.turns.idempotency import TurnIdempotencyLedger

    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    state = tmp_path / "sidecar-state"
    state.mkdir()
    clock = {"now": datetime.now(timezone.utc).replace(microsecond=0)}
    store = InitiativeStore(state_dir=state)
    commitments = CommitmentStore(state / "colonymind-commitments.db")
    ledger = TurnIdempotencyLedger(state / "turn-idempotency.db")
    mind = Mind(config={"autonomy": "standard", "quiet_hours": ""}, store=store, state_dir=state, owner_id=OWNER,
                commitments=commitments, feedback=TypeFeedbackStore(str(state / "colonymind-feedback.db")),
                expectations=ExpectationEngine(ExpectationStore(str(state / "colonymind-expectations.db"))),
                contacts=Contacts(OWNER), ledger=ledger, clock=lambda: clock["now"], backups=False,
                persist=lambda changes: None)
    mind.digest_hour = 25
    app = FastAPI()
    # The owner speaks on Telegram here: the instance key is granted the owner there, as ``colonymind init`` does for
    # the owner's handles.
    app.add_middleware(ApiKeyMiddleware, api_key=API_KEY, credentials=CredentialTable.from_config(
        {INSTANCE_CREDENTIAL: {"owner": ["telegram", "whatsapp", "sms", "rcs", "voice"], "contacts": "all"}}))
    app.include_router(mind_router.router)

    @app.get("/v1/host/health")
    async def health():
        return {"status": "ok", "capabilities": ["memory", "mind"]}

    @app.get("/v1/host/contacts/resolve")
    async def resolve(gateway: str = "", address: str = ""):
        from fastapi import HTTPException
        if gateway == "telegram" and address == "1001":
            return {"contact_id": OWNER, "display_name": "Owner", "may_contact": "auto"}
        if gateway == "telegram" and address == "2003":
            return {"contact_id": "p-03", "display_name": "Friend", "may_contact": "ask"}
        raise HTTPException(status_code=404, detail="No contact for that handle")

    mind_router.set_mind(mind)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", access_log=False,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    started = time.monotonic()
    while not server.started:
        assert thread.is_alive() and time.monotonic() - started < 10, "the sidecar did not start"
        time.sleep(0.02)

    def shift(**delta):
        clock["now"] += timedelta(**delta)

    try:
        yield SimpleNamespace(port=port, url=f"http://127.0.0.1:{port}", mind=mind, store=store,
                              commitments=commitments, ledger=ledger, clock=clock, shift=shift)
    finally:
        server.should_exit = True
        thread.join(10)
        mind_router.set_mind(None)
        store.close()


@pytest.fixture
def real_home(tmp_path, real_sidecar):
    return build_home(tmp_path, real_sidecar.port, real_sidecar.url)


def overdue(real, description, *, hours=1, person=OWNER):
    """A commitment due in a minute; the mind's clock then moves past it."""
    due = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    row = real.commitments.create(person_id=person, description=description, due_at=due)
    real.shift(hours=hours)
    return row


# The parked body, the kanban helpers, the tools and the command; ``tick()`` here is the
# plugin's own: ``POST /v1/mind/tick`` then one body pass.
PRELUDE = MIND_PRELUDE + TOOL_CODE + "def tick():\n    return colonymind_hermes.tick()\n"
TICK = "result = tick()\n"
# The owner's chat as the gateway runs it, with the plan's five seconds between a reply and a shown ask.
ANSWER_CODE = GATEWAY_CODE.replace("answers_module.AFTER_REPLY_SECONDS = 0.3", "")


def test_commitment_becomes_one_task_whose_outcome_comes_back(real_home, real_sidecar):
    """Build plan M2, end to end across the seam: one task, no duplicate after a restart, the outcome recorded."""
    real = real_sidecar
    overdue(real, "send the owner the report")
    first = probe(TICK + "emit(tick=result, tasks=tasks(), state=body.client.mind_state())",
                  real_home, prelude=PRELUDE)
    formed = first["tick"]["mind_tick"]["formed"]
    assert [item["type"] for item in formed] == ["commitment_overdue"] and formed[0]["decision"] == "act"
    assert first["tick"]["mind"] is True and first["tick"]["dispatched"] == 1
    intention_id = formed[0]["id"]
    task, = first["tasks"].values()
    assert list(first["tasks"]) == [f"mind:{intention_id}"]
    assert task["assignee"] == "colonymind-act" and task["status"] == "ready" and task["created_by"] == "colonymind"
    assert "Report what you did" in task["body"] and "duty drive" in task["body"]
    row = real.store.get(intention_id)
    assert row.status == "dispatched" and row.hermes_ref == task["id"] and row.hermes_kind == "kanban"
    assert first["state"]["enabled"] is True and first["state"]["dispatched"] == 1 and first["state"]["asks"] == []

    # A restart: the sidecar never re-offers a dispatched intention, the tick forms nothing new.
    second = probe(TICK + "emit(tick=result, tasks=tasks())", real_home, prelude=PRELUDE)
    assert second["tick"]["mind_tick"]["formed"] == [] and second["tick"]["dispatched"] == 0
    assert list(second["tasks"]) == [f"mind:{intention_id}"]

    # The worker finishes; reconciliation posts the body's report once and the sidecar settles the row.
    third = probe('''
conn = connect()
kb.claim_task(conn, tasks()["mind:%s"]["id"], claimer="worker-a")
kb.complete_task(conn, tasks()["mind:%s"]["id"], summary="Sent the report. Evidence: message id 7. It worked.")
conn.close()
after = tick()
again = tick()
emit(after=after, again=again, why=body.client.get("/v1/mind/why/%s").json())
''' % (intention_id, intention_id, intention_id), real_home, prelude=PRELUDE)
    assert third["after"]["reconciled"] == 1 and third["again"]["reconciled"] == 0
    row = real.store.get(intention_id)
    assert row.status == "done" and row.outcome == "done" and row.result.startswith("Sent the report")
    # Nothing outside the intention verified it (the worker cannot mark the row fulfilled itself), so
    # the outcome is unverified and the body's report is what closes the commitment.
    assert row.verified == "none" and "check" not in row.result_metadata
    assert row.result_metadata["run"]["outcome"] == "completed"
    commitment, = real.commitments.list(person_id=OWNER, status=["fulfilled"])["commitments"]
    assert commitment["metadata"]["resolution"]["by"] == "body" and intention_id in commitment["metadata"]["resolution"]["note"]
    why = third["why"]
    assert why["drive"] == "duty" and why["decision"] == "act" and why["hermes_ref"] == task["id"]
    assert why["outcome"] == "done" and why["verified"] == "none" and why["evidence"]
    assert any(item["action"] == "outcome_done" for item in why["history"])
    # The autobiography entry a later session recalls (architecture 4.1).
    references = real.ledger.source_references([f"mind:{intention_id}:outcome_done"], contact_id=OWNER,
                                               session_id="mind")
    assert references, "no autobiography entry in the ledger"


def notice_row(real):
    """The ask's own message as the sidecar recorded its delivery (the body reports its platform id)."""
    rows = [row for row in real.store.intentions(kind=["message"], limit=50)
            if row.type == "ask_notice" and row.status == "sent" and row.hermes_ref]
    assert rows, "no delivered ask message"
    return rows[0]


def test_ask_path_across_the_seam(real_home, real_sidecar):
    """A floor task becomes an ask: one notice, no task; the owner's yes on the ask's own message approves it, in
    their own chat, read by the sidecar before the model runs. A guest's yes, a tool call and the owner's yes to the
    agent's own question approve nothing; the model can only have the sidecar show the ask again, in the sidecar's
    own words. The floor is
    what a task can do, never its words: here the worker would carry a shell, so the task asks at every level (a
    task with the default toolsets and the same words acts)."""
    real = real_sidecar
    toolsets = real.mind.policy.worker_toolsets
    real.mind.policy.worker_toolsets = (*toolsets, "terminal")
    overdue(real, "wire $500 to the vendor")
    first = probe('''
_smt.send_message_tool = fake_send
result = tick()
emit(tick=result, tasks=tasks(), sends=SENDS, asks=command("asks"), status=command("status"))
''', real_home, prelude=PRELUDE)
    asked = [row for row in real.store.intentions(status=["asked"], limit=10)]
    assert len(asked) == 1 and asked[0].ask_code and first["tasks"] == {}
    code = asked[0].ask_code
    # The ask reaches the owner's chat in words with its reply line; the message's own format is the sidecar's.
    notice, = first["sends"]
    assert notice["target"] == "telegram:1001" and "wire $500 to the vendor" in notice["message"].lower()
    assert "yes or no" in notice["message"].lower()
    assert "wire $500 to the vendor" in first["asks"] and code not in first["asks"] and "asks:" in first["status"]
    assert first["tick"]["sent"] == 1

    # A guest's yes posts nothing; a tool call answers nothing; the owner's yes to the agent's own question (its
    # reply came after the ask) is conversation, and the model is told the ask is still open.
    second = probe(ANSWER_CODE + '''
dispatched("yes", sender="2003", message_id="g-1")
turn("guest-1", "yes", sender="2003")
dispatched("what is the weather tomorrow?", message_id="o-1", reply_to="unrelated-weather-message")
turn("owner-1", "what is the weather tomorrow?")
replied("owner-1", "what is the weather tomorrow?", "Sunny. Want the hourly forecast too?")
time.sleep(3.5)
dispatched("yes", message_id="o-2")
context = turn("owner-2", "yes")
emit(context=context, tool=call("colonymind_self", {"operation": "yes", "code": "%(code)s"}, "owner-2"))
''' % {"code": code}, real_home, prelude=PRELUDE)
    assert real.store.get(asked[0].id).status == "asked"
    assert "owner's own message" in refused(second["tool"])
    assert "not read as an answer" in second["context"][0]["context"]

    # The model can only ask the sidecar to show it again, in the sidecar's own words (a sidecar without that route
    # shows nothing): the plugin never words an ask itself. The reply wakes the body for what the turn caused; the
    # body is parked here, so the probe runs the pass that wake starts, once the reply's gap has passed. The owner's
    # yes quoting the ask's own message approves it (dated after it: the sidecar's clock here stands where
    # ``overdue`` moved it, an hour past the wall clock).
    shown = probe(ANSWER_CODE + '''
_smt.send_message_tool = fake_send
dispatched("what was that payment again?", message_id="o-3")
turn("owner-3", "what was that payment again?")
listed = call("colonymind_self", {"operation": "asks"}, "owner-3")
show = call("colonymind_self", {"operation": "show", "id": "1"}, "owner-3")
replied("owner-3", "what was that payment again?", "It waits for your OK.")
time.sleep(8)
body.send_outbox()
dispatched("yes", message_id="o-4", reply_to="%(ask_message)s", ago=-7200)
context = turn("owner-4", '[Replying to: "Needs your OK"]\\n\\nyes')
replied("owner-4", "yes", "Approved, I will send it.")
emit(listed=listed, show=show, sends=SENDS, context=context)
''' % {"ask_message": notice_row(real).hermes_ref}, real_home, prelude=PRELUDE, timeout=300)
    assert code not in json.dumps(shown["listed"]) and "wire $500 to the vendor" in json.dumps(shown["listed"])
    sidecar_shows = [row for row in real.store.intentions(kind=["message"], limit=50) if row.type == "ask_show"]
    assert len(shown["sends"]) == len(sidecar_shows) and all(code not in send["message"] for send in shown["sends"])
    assert all(row.status == "sent" for row in sidecar_shows)
    assert real.store.get(asked[0].id).status == "approved", shown["context"]
    assert "approved" in shown["context"][0]["context"]
    third = probe(TICK + "emit(tick=result, tasks=tasks())", real_home, prelude=PRELUDE)
    assert list(third["tasks"]) == [f"mind:{asked[0].id}"] and third["tick"]["dispatched"] == 1
    assert real.store.get(asked[0].id).status == "dispatched"

    # Silence: a second ask expires after 72 h while unrelated duty work proceeds. Every turn above has replied and
    # its delivery gap has passed, so the body holds no ask's notice behind the chat's last turn
    # (``answers.HELD_TYPES``, ``AFTER_REPLY_SECONDS``): the notice goes, and only the ask itself expires.
    time.sleep(6)
    overdue(real, "wire $900 to the landlord")
    probe(TICK + "emit(tick=result)", real_home, prelude=PRELUDE)
    real.mind.policy.worker_toolsets = toolsets
    overdue(real, "send the owner the slides")
    probe(TICK + "emit(tick=result)", real_home, prelude=PRELUDE)
    pending = [row for row in real.store.intentions(status=["asked"], limit=10)]
    assert len(pending) == 1 and "landlord" in pending[0].description
    assert any(row.status == "dispatched" and "slides" in row.description
               for row in real.store.intentions(kind=["task"], limit=20))
    real.shift(hours=73)
    fourth = probe(TICK + "emit(tick=result, tasks=tasks())", real_home, prelude=PRELUDE)
    assert fourth["tick"]["mind_tick"]["expired_asks"] == 1
    assert real.store.get(pending[0].id).status == "expired"
    assert f"mind:{pending[0].id}" not in fourth["tasks"]


def test_off_switch_across_the_seam(real_home, real_sidecar):
    """``/mind off`` in chat: the sidecar stops offering work and the body archives what was not started."""
    real = real_sidecar
    overdue(real, "send the owner the report")
    overdue(real, "send the owner the minutes")
    first = probe(TICK + "emit(tick=result, tasks=tasks())", real_home, prelude=PRELUDE)
    assert first["tick"]["dispatched"] == 2
    real.mind.outbox.notice(type="stale_task", title="n", text="late", dedup_key="late")
    second = probe('''
_smt.send_message_tool = fake_send
off = command("off")
after = tick()
emit(off=off, after=after, tasks=tasks(), sends=SENDS, state=body.client.mind_state())
''', real_home, prelude=PRELUDE)
    assert second["off"].startswith("Mind off.") and "sidecar: off" in second["off"]
    assert second["state"]["enabled"] is False and second["sends"] == []
    assert {task["status"] for task in second["tasks"].values()} == {"archived"}
    assert real.mind.enabled is False and real.store.intentions(status=["approved"], kind=["message"], limit=5) == []
    real.mind.on()
    overdue(real, "send the owner the agenda")
    third = probe(TICK + "emit(tick=result, tasks=tasks())", real_home, prelude=PRELUDE)
    assert third["tick"]["dispatched"] == 1
    assert sum(task["status"] == "ready" for task in third["tasks"].values()) == 1


def test_observations_and_the_owner_handle_shape_init_writes(real_home, real_sidecar):
    """Board observations reach the mind in the body's shape; ``identity.yaml`` handles as ``init`` writes them."""
    real = real_sidecar
    identity = yaml.safe_load((real_home.instance / "identity.yaml").read_text())
    identity["owner"]["handles"] = [{"platform": "telegram", "id": "1001"}]
    (real_home.instance / "identity.yaml").write_text(yaml.safe_dump(identity))
    first = probe('''
conn = connect()
old = kb.create_task(conn, title="owner task from long ago", assignee="default", idempotency_key="owner:old")
conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (int(time.time()) - 80 * 3600, old))
conn.commit()
conn.close()
body._last_observation = None
result = tick()
emit(tick=result, observed=body.client.get("/v1/mind/state").json()["body"])
''', real_home, prelude=PRELUDE)
    assert first["tick"]["observed"] is True and first["observed"]["ticks"] >= 1
    assert [item["id"] for item in real.mind.observations["stale_task"]] and \
        real.mind.observations["stale_task"][0]["age_hours"] >= 80
    second = probe('''
_smt.send_message_tool = fake_send
result = tick()
emit(tick=result, sends=SENDS)
''', real_home, prelude=PRELUDE)
    assert [item["type"] for item in second["tick"]["mind_tick"]["formed"]] == ["stale_task"]
    notice, = second["sends"]
    assert notice["target"] == "telegram:1001" and "owner task from long ago" in notice["message"]
