"""Standing allows: "always" on an ask (docs/MIND.md, Standing grants and allows).

An ask that can stop asking offers "always" in its own message, naming exactly what it would allow. The owner's own
"always" (their yes, read as strictly as any widening answer, bound to that ask by a reaction, a quoted reply or their
next message within twelve hours) approves the ask and adds one standing allow for exactly that class of work:

- a contact on ``ask``: its permission becomes ``auto`` (performed as the owner's answer to the ask);
- an external task: tasks with exactly those extra toolsets act at ``standard``;
- a held structured floor call: a standing grant for exactly its class, tool, target and amount, once a day.

A task with an executor toolset offers none: nothing confines its commands to its workspace, so it keeps its one ask.
Allows never apply at ``suggest`` and never lift the off switch, the deny list, the breaker or the budgets. The console
and the API list and revoke them (a narrowing); nothing but the owner's own answer adds one.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import mind as mind_router
from colonymind.contacts.store import ContactChanged
from colonymind.mind import allows as allow_rules
from colonymind.mind import answers as answer_words
from colonymind.mind import asks_report
from colonymind.mind import cli as mind_cli
from colonymind.mind import digest as digest_text
from owner_answer import CHAT, PLATFORM, owner_answers
from test_mind_cli import _Transport, _parse, home  # noqa: F401  (home: the CLI's instance fixture)
from test_mind_loop import AUTH, CONTACT, OWNER, FakeContacts, Fixture, pinned_clock  # noqa: F401  (pinned clock)

DECLARED = {"pay_invoice": {"class": "money_movement", "amount": "amount", "currency": "currency", "target": "payee"}}
PAY = {"amount": 40, "currency": "usd", "payee": "acme"}
# A command the floor recognizers name (an irreversible deletion), spelled apart so no shell guard reads it here.
DELETION = {"command": " ".join(["rm", "-rf", "/srv/backups/2019"])}


class Contacts(FakeContacts):
    """The fake store with the owner's permission write: compare-and-set against ``expected``, recorded."""

    def __init__(self, records):
        super().__init__(records)
        self.writes = []

    async def set_may_contact(self, contact_id, value, *, by, reason="", expected=None, expected_handles=None):
        record = self.records[contact_id]
        if expected is not None and record.get("may_contact") != expected:
            raise ContactChanged(contact_id)
        record["may_contact"] = value
        self.writes.append((contact_id, value, by))


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    built = []

    def build(**kwargs):
        state = tmp_path / f"instance-{len(built)}"
        state.mkdir()
        fx = Fixture(state, **kwargs)
        fx.contacts = Contacts(fx.contacts.records)
        fx.restart()
        built.append(fx)
        return fx
    yield build
    mind_router.set_mind(None)
    for fx in built:
        fx.store.close()


async def formed(fx, person, description):
    """A commitment owed to ``person``, past due: the one intention the tick forms for it."""
    created = fx.commitments.create(person_id=person, description=description,
                                    due_at=(fx.now + timedelta(minutes=1)).isoformat())
    fx.shift(minutes=10)
    summary = await fx.mind.tick(force=True)
    row, = [row for row in (fx.store.get(item["id"]) for item in summary["formed"])
            if str(row.source_id) == str(created["id"])]
    return row


def ask_messages(fx, row):
    return [message for message in fx.store.intentions(kind=["message"], limit=200)
            if message.type == "ask_notice" and row.id in (message.context.get("asks") or [])]


def allows(fx):
    return fx.mind.allows_state()["allows"]


def deliver(fx, message, message_id):
    fx.mind.outbox.sending(message.id, target=f"{PLATFORM}:{CHAT}")
    return fx.mind.outbox.sent(message.id, result="sent", hermes_ref=message_id)


def _dispatched(fx, title):
    row, _ = fx.store.create_intention(kind="task", type="upkeep", title=title, drive="upkeep", cls="internal",
                                       decision="act", decision_reason="standard: internal -> act", status="approved",
                                       dedup_key=f"task:{title}", context={"body": title}, created_at=fx.now)
    fx.store.transition(row.id, "approved", action="queued", at=fx.now)
    return fx.mind.bound(row.id, f"kanban-{row.id[:6]}")


# -- the offer ---------------------------------------------------------------------------------------------------------

async def test_an_ask_to_reach_a_contact_offers_always_in_its_own_message(make):
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    assert row.status == "asked" and row.cls == "contact"
    assert {key: row.context["always"][key] for key in ("kind", "target")} == {"kind": "contact", "target": CONTACT}
    message, = ask_messages(fx, row)
    text = message.context["text"]
    assert "Reply yes or no, or 'always' to let me contact" in text and text.endswith("without asking first.")
    assert row.ask_code not in text


async def test_always_is_offered_only_where_a_standing_permission_exists(make):
    """No offer at ``suggest`` (the owner reviews everything there), and none for a shell, code or browser call:
    there "always" answers nothing and the ask stays open."""
    fx = make(autonomy="suggest")
    row = await formed(fx, CONTACT, "send the contact the slides")
    assert row.status == "asked" and "always" not in row.context
    assert (await owner_answers(fx.mind, row.ask_code, "always")).status == "asked"
    fx = make(autonomy="trusted")
    task = _dispatched(fx, "clear the old backups")
    held = await fx.mind.guard(tool="terminal", run="mind", task_id=task.hermes_ref, args=DELETION)
    shell = fx.store.get_by_ask_code(held["held"])
    assert shell.type == "floor_effect" and "always" not in shell.context
    assert (await owner_answers(fx.mind, shell.ask_code, "always")).status == "asked"
    assert allows(fx) == []


# -- a contact -----------------------------------------------------------------------------------------------------------

async def test_always_approves_the_ask_and_stops_asking_before_contacting_that_person(make):
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    approved = await owner_answers(fx.mind, row.ask_code, "always")
    assert approved.status == "approved"
    assert fx.contacts.records[CONTACT]["may_contact"] == "auto"
    assert fx.contacts.writes == [(CONTACT, "auto", f"owner-ask:{row.ask_code}")]
    allow, = allows(fx)
    assert (allow["kind"], allow["target"], allow["ask"]) == ("contact", CONTACT, row.id)
    assert allow["by"].startswith("owner-turn:") and allow["uses"] == 0
    again = await formed(fx, CONTACT, "send the contact the agenda")
    assert again.status == "approved" and again.decision == "act"
    assert allows(fx)[0]["uses"] == 1 and allows(fx)[0]["last_used"]


async def test_always_needs_the_bar_of_a_widening_answer(make):
    """"always" stops asking, so it is read like any widening answer: strict, and by a plain next message only within
    twelve hours of the ask's own message. Short of that bar it approves nothing and adds nothing; a plain yes
    still approves the ask alone."""
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    for text in ("always, but not on weekends", "always?", "maybe always", "always. How long will it take?"):
        assert (await owner_answers(fx.mind, row.ask_code, text)).status == "asked", text
    assert allows(fx) == [] and fx.contacts.records[CONTACT]["may_contact"] == "ask"
    message, = ask_messages(fx, row)
    deliver(fx, message, "m-ask")
    fx.shift(hours=13)
    result = await fx.mind.read_answer({
        "platform": PLATFORM, "contact_id": OWNER, "chat_id": CHAT, "chat_type": "dm", "message_id": "late-1",
        "turn_id": "t-late-1", "arrived_at": fx.now.isoformat(), "text": "always",
        "ledger": [{"at": fx.store.get(message.id).completed_at.isoformat(), "outbox_id": message.id}]})
    assert result["path"] == "follow_up" and result["reading"]["verdict"] == "unclear"      # one plain question
    assert result["settled"] == [] and fx.store.get(row.id).status == "asked"
    assert allows(fx) == []
    assert (await owner_answers(fx.mind, row.ask_code, "yes")).status == "approved"
    assert allows(fx) == [] and fx.contacts.records[CONTACT]["may_contact"] == "ask"


async def test_a_contact_that_changed_since_the_ask_gets_no_allow(make):
    """The permission is written compare-and-set against what the ask showed: a contact who opted out meanwhile stays
    out, whatever "always" said."""
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    fx.contacts.records[CONTACT]["may_contact"] = "never"
    await owner_answers(fx.mind, row.ask_code, "always")
    assert fx.contacts.records[CONTACT]["may_contact"] == "never" and allows(fx) == []


# -- a toolset, and no executor kind -----------------------------------------------------------------------------------

async def test_always_on_an_external_task_lets_tasks_with_that_toolset_act_at_standard(make):
    fx = make(config={"worker_toolsets": ["web", "file", "image_gen"]})
    row = await formed(fx, OWNER, "draw the quarterly chart")
    assert row.status == "asked" and row.cls == "external"
    assert (row.context["always"]["kind"], row.context["always"]["target"]) == ("toolset", "image_gen")
    await owner_answers(fx.mind, row.ask_code, "always")
    allow, = allows(fx)
    again = await formed(fx, OWNER, "draw the yearly chart")
    assert again.status == "approved" and again.context["allow"] == allow["id"]
    assert allow["id"] in again.decision_reason
    fx.mind.set_level("suggest")
    held = await formed(fx, OWNER, "draw the monthly chart")
    assert held.status == "asked"                                  # never at suggest


async def test_an_executor_task_offers_no_always_and_keeps_its_one_ask(make):
    """A standing permission covers work the owner never saw, so it may only let calls run that stay inside the task's
    workspace, and nothing confines what a shell, code, browser or delegation call reaches there by structure (the
    plugin confines ``write_file`` and ``patch`` only). So a task with an executor toolset offers no "always", an
    "always" to it approves nothing, an executor allow in the store is no allow, and the owner's yes to the task is
    what lets its commands run, as before."""
    fx = make(config={"worker_toolsets": ["web", "file", "terminal"]})
    row = await formed(fx, OWNER, "rebuild the site")
    assert row.status == "asked" and row.cls == "floor" and "always" not in row.context
    message, = ask_messages(fx, row)
    assert message.context["text"].endswith("Reply yes or no.")
    assert (await owner_answers(fx.mind, row.ask_code, "always")).status == "asked" and allows(fx) == []
    fx.mind.mind_state.set(f"allow:executor-{row.drive}-terminal", text=json.dumps({
        "id": f"executor-{row.drive}-terminal", "kind": "executor", "target": f"{row.drive}:terminal",
        "words": "let me use terminal without asking first", "ask": row.id, "toolsets": ["terminal"]}))
    assert allows(fx) == []
    task = await formed(fx, OWNER, "rebuild the docs")
    assert task.status == "asked" and "always" not in task.context
    assert (await owner_answers(fx.mind, task.ask_code, "yes")).status in {"approved", "dispatched"}
    fx.mind.bound(task.id, "kanban-docs")
    ran = await fx.mind.guard(tool="terminal", run="mind", task_id="kanban-docs", args={"command": "make docs"})
    assert ran["allow"] is True and "approved this task" in ran["reason"]


async def test_a_toolset_allow_never_lets_an_executor_call_run(make):
    """The calls a standing toolset allow lets run are its own tools' calls; a shell call in a task formed under it
    (whatever put it there) waits for the owner's yes to exactly it."""
    fx = make(config={"worker_toolsets": ["web", "file", "image_gen"]})
    row = await formed(fx, OWNER, "draw the quarterly chart")
    await owner_answers(fx.mind, row.ask_code, "always")
    allow, = allows(fx)
    task = await formed(fx, OWNER, "draw the yearly chart")
    assert task.status == "approved" and task.context["allow"] == allow["id"]
    fx.mind.dispatch()
    fx.mind.bound(task.id, "kanban-chart")
    tool = await fx.mind.guard(tool="chart_render", run="mind", task_id="kanban-chart", args={"data": "q3"})
    assert tool["allow"] is True and allow["id"] in tool["reason"]
    shell = await fx.mind.guard(tool="terminal", run="mind", task_id="kanban-chart",
                                args={"command": "touch /srv/shared/report.txt"})
    assert shell["allow"] is False and shell["held"]


# -- a structured floor call ---------------------------------------------------------------------------------------------

async def test_always_on_a_structured_floor_call_is_a_grant_for_exactly_that_call(make):
    fx = make(config={"floor": {"tools": DECLARED}})
    task = _dispatched(fx, "pay the hosting invoice")
    held = await fx.mind.guard(tool="pay_invoice", args=PAY, run="mind", task_id=task.hermes_ref)
    ask = fx.store.get_by_ask_code(held["held"])
    assert ask.context["always"]["kind"] == "grant"
    await owner_answers(fx.mind, ask.ask_code, "always")
    allow, = allows(fx)
    grant, = [item for item in fx.mind.grants_state()["grants"] if item["id"] == allow["grant"]]
    assert (grant["class"], grant["tools"], grant["match"]) == ("money_movement", ["pay_invoice"], ["acme"])
    assert grant["max_amount"] == 40 and grant["per_day"] == 1 and grant["active"]
    later = _dispatched(fx, "pay next month's hosting invoice")
    assert (await fx.mind.guard(tool="pay_invoice", args=PAY, run="mind", task_id=later.hermes_ref))["allow"]
    for args in ({**PAY, "amount": 60}, {**PAY, "payee": "someone-else"}):
        other = _dispatched(fx, f"pay {args}")
        assert (await fx.mind.guard(tool="pay_invoice", args=args, run="mind", task_id=other.hermes_ref))["held"]


async def test_a_grant_allow_counts_each_call_its_grant_lets_run(make):
    """A grant allow lets work go without asking where its grant covers a call, at execution: each such call is a use
    of the allow, so the listing shows it and the digest offers for removal only one that let nothing go for 90
    days."""
    fx = make(config={"floor": {"tools": DECLARED}})
    task = _dispatched(fx, "pay the hosting invoice")
    held = await fx.mind.guard(tool="pay_invoice", args=PAY, run="mind", task_id=task.hermes_ref)
    await owner_answers(fx.mind, held["held"], "always")
    assert allows(fx)[0]["uses"] == 0
    fx.shift(days=80)
    later = _dispatched(fx, "pay a later hosting invoice")
    assert (await fx.mind.guard(tool="pay_invoice", args=PAY, run="mind", task_id=later.hermes_ref))["allow"]
    allow, = allows(fx)
    assert allow["uses"] == 1 and allow["last_used"] == fx.now.isoformat()
    assert "1 uses" in fx.mind.allows_state()["text"]
    fx.shift(days=20)                                                   # 100 days since the "always", 20 since use
    assert not allow_rules.unused(fx.mind.allows.in_force()[0], fx.now)
    assert "90 days" not in allow_rules.digest_line(fx.mind.allows.in_force(), fx.now)
    fx.shift(days=71)
    assert "90 days" in allow_rules.digest_line(fx.mind.allows.in_force(), fx.now)


# -- revoking, undoing, and nothing else adds one ------------------------------------------------------------------------

async def test_the_console_lists_and_revokes_an_allow_and_no_route_adds_one(make):
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    allow, = allows(fx)
    async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url="http://test") as client:
        listed = (await client.get("/v1/mind/allows", headers=AUTH)).json()
        assert [item["id"] for item in listed["allows"]] == [allow["id"]] and allow["id"] in listed["text"]
        assert (await client.get("/v1/mind/allows", headers=AUTH, params={"viewer": CONTACT})).json()["allows"] == []
        assert (await client.post("/v1/mind/allows", headers=AUTH, json={"kind": "contact",
                                                                          "target": "p-03"})).status_code == 405
        assert (await client.post("/v1/mind/allows/nope/revoke", headers=AUTH, json={})).status_code == 404
        revoked = await client.post(f"/v1/mind/allows/{allow['id']}/revoke", headers=AUTH, json={"by": "cli"})
        assert revoked.status_code == 200 and revoked.json()["status"] == "revoked"
    assert allows(fx) == [] and fx.contacts.records[CONTACT]["may_contact"] == "ask"
    assert (await formed(fx, CONTACT, "send the contact the agenda")).status == "asked"


async def test_revoking_leaves_a_permission_changed_since_as_it_is(make):
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    fx.contacts.records[CONTACT]["may_contact"] = "never"               # the contact opted out since
    await fx.mind.revoke_allow(allows(fx)[0]["id"], by="cli")
    assert fx.contacts.records[CONTACT]["may_contact"] == "never" and allows(fx) == []


async def test_a_revocation_whose_permission_write_fails_keeps_asking_until_the_reset_lands(make):
    """A contact's permission reset that fails (the contacts database is locked, say) takes nothing back half way: the
    allow is gone at once, the mind asks before contacting that person while their stored permission still says
    ``auto``, and the reset is kept and tried again at each tick until it lands (or the permission changed since)."""
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    allow, = allows(fx)
    working = fx.contacts.set_may_contact

    async def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    fx.contacts.set_may_contact = locked
    done = await fx.mind.revoke_allow(allow["id"], by="cli")
    assert done["status"] == "revoked" and "try again" in done["text"]
    assert allows(fx) == [] and fx.contacts.records[CONTACT]["may_contact"] == "auto"      # the write did not land
    pending, = fx.mind.allows_state()["resetting"]
    assert pending["id"] == allow["id"] and "resetting" in fx.mind.allows_state()["text"]
    assert (await formed(fx, CONTACT, "send the contact the agenda")).status == "asked"
    sent = await fx.mind.guard(tool="send_message", args={"contact_id": CONTACT, "message": "hi"}, run="mind",
                               task_id=_dispatched(fx, "say hi").hermes_ref)
    assert sent["allow"] is False
    await fx.mind.tick(force=True)
    assert fx.mind.allows_state()["resetting"] and fx.contacts.records[CONTACT]["may_contact"] == "auto"
    fx.contacts.set_may_contact = working
    await fx.mind.tick(force=True)
    assert fx.contacts.records[CONTACT]["may_contact"] == "ask" and fx.mind.allows_state()["resetting"] == []
    assert fx.contacts.writes[-1] == (CONTACT, "ask", f"allow-revoked:{allow['id']}")


async def test_a_pending_reset_leaves_a_permission_changed_since_as_it_is(make):
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    working = fx.contacts.set_may_contact

    async def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    fx.contacts.set_may_contact = locked
    await fx.mind.revoke_allow(allows(fx)[0]["id"], by="cli")
    fx.contacts.records[CONTACT]["may_contact"] = "never"               # the contact opted out meanwhile
    fx.contacts.set_may_contact = working
    await fx.mind.tick(force=True)
    assert fx.contacts.records[CONTACT]["may_contact"] == "never" and fx.mind.allows_state()["resetting"] == []


async def test_a_new_always_while_a_reset_waits_keeps_the_permission_and_drops_the_reset(make):
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    working = fx.contacts.set_may_contact

    async def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    fx.contacts.set_may_contact = locked
    await fx.mind.revoke_allow(allows(fx)[0]["id"], by="cli")
    fx.contacts.set_may_contact = working
    again = await formed(fx, CONTACT, "send the contact the agenda")
    assert again.status == "asked" and again.context["always"]["kind"] == "contact"
    await owner_answers(fx.mind, again.ask_code, "always")
    allow, = allows(fx)
    assert allow["ask"] == again.id and allow["before"] == "ask" and fx.mind.allows_state()["resetting"] == []
    await fx.mind.tick(force=True)
    assert fx.contacts.records[CONTACT]["may_contact"] == "auto"


async def test_revoking_a_grant_allow_withdraws_its_grant(make):
    fx = make(config={"floor": {"tools": DECLARED}})
    task = _dispatched(fx, "pay the hosting invoice")
    held = await fx.mind.guard(tool="pay_invoice", args=PAY, run="mind", task_id=task.hermes_ref)
    await owner_answers(fx.mind, held["held"], "always")
    allow, = allows(fx)
    await fx.mind.revoke_allow(allow["id"], by="cli")
    assert allows(fx) == [] and allow["grant"] not in {grant.id for grant in fx.mind.grants()}


async def test_undo_takes_an_always_back(make):
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    fx.shift(minutes=2)
    result = await fx.mind.read_answer({"platform": PLATFORM, "contact_id": OWNER, "chat_id": CHAT, "chat_type": "dm",
                                        "message_id": "undo-1", "turn_id": "t-undo-1",
                                        "arrived_at": fx.now.isoformat(), "text": "undo"})
    assert result["undone"] and allows(fx) == []
    assert fx.contacts.records[CONTACT]["may_contact"] == "ask"


# -- revoking in the owner's own words -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, thing", [
    ("ask me again about Sam", "sam"),
    ("Please ask me again before you contact Sam Rivera.", "you contact sam rivera"),
    ("From now on always ask me first before using terminal", "using terminal"),
    ("start asking again about image_gen, thanks", "image_gen, thanks"),
    ("Could you check with me again before paying acme?", "paying acme"),
])
def test_ask_me_again_names_the_thing(text, thing):
    assert answer_words.ask_again(text) == thing


@pytest.mark.parametrize("text", [
    "don't ask me again about Sam", "No need to ask me again before contacting Sam", "never ask again before you "
    "message Sam", "stop asking me first about Sam", "Did you ask me again about Sam?", "ask me about Sam",
    "ask me again", 'She wrote "ask me again about Sam" in the doc', "> ask me again about Sam",
])
def test_words_that_do_not_ask_to_be_asked_again_name_nothing(text):
    assert answer_words.ask_again(text) is None


def _says(fx, text, number, *, contact_id=OWNER):
    return fx.mind.read_answer({"platform": PLATFORM, "contact_id": contact_id, "chat_id": CHAT, "chat_type": "dm",
                                "message_id": f"words-{number}", "turn_id": f"t-words-{number}",
                                "arrived_at": fx.now.isoformat(), "text": text})


async def test_ask_me_again_about_a_person_revokes_their_allow(make):
    """"Ask me again about <thing>" in the owner's own message revokes the allows that thing names (a narrowing):
    the asking comes back and the contact goes back to ask. Anyone else's message, a negation, a thing no allow
    names or someone else's words the owner passes on revoke nothing."""
    fx = make()
    fx.contacts.records[CONTACT]["display_name"] = "Sam Rivera"
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    allow, = allows(fx)
    fx.shift(minutes=30)                       # past the undo window: this is no undo
    kept = ("don't ask me again about Sam", "ask me again about Kim", 'Kim wrote: "ask me again about Sam"',
            "ask me again about tasks")
    for number, text in enumerate(kept):
        result = await _says(fx, text, number)
        assert result.get("revoked", []) == [] and len(allows(fx)) == 1, text
    assert (await _says(fx, "ask me again about Sam", 10, contact_id=CONTACT))["read"] is False
    assert len(allows(fx)) == 1
    result = await _says(fx, "Please ask me again before you contact Sam.", 11)
    assert [item["allow"] for item in result["revoked"]] == [allow["id"]]
    assert allows(fx) == [] and fx.contacts.records[CONTACT]["may_contact"] == "ask"
    assert "ask again before I contact Sam Rivera" in result["context"]
    assert fx.mind.answer_turns.find(result["key"]).as_answer        # the question readers skip this message
    assert (await formed(fx, CONTACT, "send the contact the agenda")).status == "asked"


async def test_ask_me_again_names_a_kind_of_work_or_everything(make):
    fx = make(config={"worker_toolsets": ["web", "file", "image_gen"]})
    row = await formed(fx, OWNER, "draw the quarterly chart")
    await owner_answers(fx.mind, row.ask_code, "always")
    fx.shift(minutes=30)
    assert (await _says(fx, "ask me again before you browse the web", 1)).get("revoked", []) == []
    result = await _says(fx, "ask me again before you use image gen", 2)
    assert allows(fx) == [] and len(result["revoked"]) == 1
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    fx.shift(minutes=30)
    result = await _says(fx, "Always ask me first about everything.", 3)
    assert allows(fx) == [] and len(result["revoked"]) == 1


async def test_the_digest_lists_the_allows_once_a_week_and_offers_to_remove_one_left_unused(make):
    """One line of the digest, once a week, says what goes without asking and how to bring the asking back; an allow
    that let nothing go for 90 days is offered for removal there. The line rides along in a digest; it never makes
    one on its own."""
    fx = make()
    fx.contacts.records[CONTACT]["display_name"] = "Sam Rivera"
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    fx.mind.digest_hour, fx.mind.quiet = 0, None

    async def digest(code):
        fx.store.create_intention(
            kind="task", type="research", title=f"Re-run export {code}", drive="curiosity", cls="self",
            context={"notice": True}, decision="ask", decision_reason="an ask", status="asked",
            dedup_key=f"test:{code}", ask_code=code, hermes_kind="none", created_at=fx.now,
            expires_at=fx.now + timedelta(days=3))
        ident = await fx.mind._digest(fx.now)
        text = fx.store.get(ident).context["text"]
        return [line for line in digest_text.lines_of(text, digest_text.DONE) if "ask me again about" in line]

    line, = await digest("DA1")
    assert "contact Sam Rivera" in line and "90 days" not in line
    fx.shift(days=1)
    assert await digest("DB2") == []                                     # once a week
    fx.shift(days=7)
    line, = await digest("DC3")
    assert "contact Sam Rivera" in line and "90 days" not in line
    fx.shift(days=90)
    line, = await digest("DD4")
    assert "contact Sam Rivera" in line and "90 days" in line and "remove" in line


def test_the_cli_lists_and_revokes_allows(home, monkeypatch, capsys):
    transport = _Transport({
        ("GET", "/v1/mind/allows"): {"allows": [], "text": "contact-p-02: let me contact p-02 without asking first"},
        ("POST", "/v1/mind/allows/contact-p-02/revoke"): {"ok": True, "status": "revoked", "text": "revoked"},
    })
    monkeypatch.setattr(httpx, "Client", transport.client)
    assert mind_cli.run(_parse(["mind", "allows"])) == 0
    assert "contact-p-02" in capsys.readouterr().out
    assert mind_cli.run(_parse(["mind", "allows", "revoke", "contact-p-02"])) == 0
    assert transport.calls[-1][:2] == ("POST", "/v1/mind/allows/contact-p-02/revoke")
    assert json.loads(transport.calls[-1][2]) == {"by": "cli"}
    assert mind_cli.run(_parse(["mind", "allows", "revoke"])) == 2


async def test_the_asks_report_counts_the_asks_an_always_stopped(make):
    """The instrument approvals are measured with counts each "always" (an ask the owner turned into a standing allow)
    beside the shows, clarifications and undos, and a proposal is an ask of its own type."""
    fx = make()
    row = await formed(fx, CONTACT, "send the contact the slides")
    await owner_answers(fx.mind, row.ask_code, "always")
    value = asks_report.report(fx.store, since=fx.now - timedelta(days=1), now=fx.now)
    assert value["events"]["allows"] == 1
    assert "standing allows added 1" in asks_report.render(value)


def test_an_ask_about_who_someone_is_offers_no_always():
    """Only a plain ask to reach a contact on ``ask`` offers "always": not one that asks whether a name the store
    matched is the person the owner meant, not work the owner's own request already allowed, never the owner."""
    def row(**context):
        return SimpleNamespace(kind="message", type="commitment_notice", entity_id=CONTACT, drive="duty",
                               context={"may_contact": "ask", **context})
    offer = allow_rules.offer_for(row(), cls="contact", level="standard", owner_id=OWNER, toolsets=[], executors=[])
    assert offer is not None and offer.key == f"contact:{CONTACT}"
    for context in ({"ask_owner": True}, {"grant": "owner"}, {"may_contact": "auto"}):
        assert allow_rules.offer_for(row(**context), cls="contact", level="standard", owner_id=OWNER, toolsets=[],
                                     executors=[]) is None, context
    assert allow_rules.offer_for(row(), cls="contact", level="standard", owner_id=CONTACT, toolsets=[],
                                 executors=[]) is None
