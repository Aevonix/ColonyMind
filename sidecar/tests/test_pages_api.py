"""The pages routes: the owner reads every page, and the agent sets the lines that are its own.

Its boundaries with a person are set from the owner's conversation for anyone and from a person's own
conversation for that person; its likes and dislikes from any conversation. A line names the turn it was
written in.
"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind import get_state_dir
from colonymind.api.credentials import CredentialTable
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host as host_mod
from colonymind.api.routers import people as people_mod
from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore
from colonymind.pages import api as pages_api
from colonymind.pages import store as pages
from colonymind.turns import get_turn_idempotency_ledger
from onekey import AUTH, KEY


@pytest.fixture
async def world(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path / "state"))
    contacts = SQLiteContactStore(ContactsConfig(sqlite_path=str(tmp_path / "contacts.db")))
    await contacts.connect()
    owner = await contacts.create(display_name="Owner", trust_tier="inner_circle", may_contact="auto")
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner.contact_id)
    casey = await contacts.create(display_name="Casey Park", trust_tier="regular")
    sam = await contacts.create(display_name="Sam Rivera", trust_tier="regular")
    monkeypatch.setattr(host_mod, "_contacts_store", contacts)
    app = FastAPI()
    app.include_router(people_mod.router)
    app.include_router(pages_api.router)
    # The plugin's side of the API: the instance key, granted the owner; a turn's person is the viewer it names.
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["sms"], "contacts": "all"}}))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH) as client:
        yield SimpleNamespace(client=client, casey=casey.contact_id, sam=sam.contact_id,
                              ledger=get_turn_idempotency_ledger(get_state_dir()))
    await contacts.close()


async def set_line(w, subject, **body):
    return await w.client.post(f"/v1/mind/pages/{subject}/lines", json=body)


async def test_the_agent_sets_a_boundary_and_the_owner_reads_the_page(world):
    w = world
    made = await set_line(w, w.casey, kind="boundary", text="Never raise money with them.", source="turn-1")
    assert made.status_code == 200 and made.json()["ok"], made.text
    line = made.json()["added"]
    page = (await w.client.get(f"/v1/mind/pages/{w.casey}")).json()
    assert page["text"] == f"Your boundaries with them:\n- #{line} Never raise money with them. [turn-1]"
    # The owner reads the page with the person, through the people route.
    person = (await w.client.get(f"/v1/mind/people/{w.casey}")).json()
    assert person["contact"]["page"] == page["text"] and page["text"] in person["text"]
    # A change drops the old line and writes the new one.
    changed = (await set_line(w, w.casey, kind="boundary", text="Money only when they raise it.", drop=str(line),
                              source="turn-2")).json()
    assert changed["dropped"] and [row["text"] for row in changed["lines"]] == ["Money only when they raise it."]


async def test_a_persons_conversation_reaches_only_the_page_about_them(world):
    w = world
    own = await set_line(w, w.casey, kind="boundary", text="Keep replies short.", source="turn-1",
                         contact_id=w.casey)
    assert own.status_code == 200 and own.json()["ok"], own.text
    assert (await w.client.get(f"/v1/mind/pages/{w.casey}", params={"contact_id": w.casey})).status_code == 200
    other = await set_line(w, w.sam, kind="boundary", text="Be curt.", source="turn-2", contact_id=w.casey)
    assert other.status_code == 403 and other.json()["detail"]["code"] == "not_your_page"
    assert (await w.client.get(f"/v1/mind/pages/{w.sam}", params={"contact_id": w.casey})).status_code == 403
    assert pages.lines(w.ledger, w.sam) == []


async def test_the_agents_tastes_are_its_own_to_set_in_any_conversation(world):
    w = world
    liked = (await set_line(w, "self", kind="like", text="Long walks.", source="turn-1", contact_id=w.casey)).json()
    assert liked["ok"] and liked["text"] == f"Likes:\n- #{liked['added']} Long walks. [turn-1]"
    # A line goes only as the kind it is.
    kept = (await set_line(w, "self", kind="dislike", drop=str(liked["added"]), contact_id=w.casey)).json()
    assert not kept["ok"] and [row["text"] for row in kept["lines"]] == ["Long walks."]
    gone = (await set_line(w, "self", kind="like", drop=str(liked["added"]))).json()
    assert gone["dropped"] and gone["lines"] == []


async def test_a_line_with_no_turn_and_a_line_that_is_the_nights_are_refused(world):
    w = world
    no_turn = await set_line(w, w.casey, kind="boundary", text="Keep replies short.")
    assert no_turn.status_code == 422 and no_turn.json()["detail"]["code"] == "no_source"
    nights = await set_line(w, w.casey, kind="about", text="They are kind.", source="turn-1")
    assert nights.status_code == 422 and nights.json()["detail"]["code"] == "kind"
    assert pages.lines(w.ledger, w.casey) == []


async def test_a_merge_moves_the_dropped_records_page_to_the_kept_one(world):
    w = world
    await set_line(w, w.sam, kind="boundary", text="Be patient.", source="turn-1")
    assert people_mod._reattribute_pages in people_mod.reattribute_hooks(reconcile=False)
    assert people_mod._reattribute_pages(w.sam, w.casey) == 1
    assert [row["text"] for row in pages.lines(w.ledger, w.casey)] == ["Be patient."]
