"""History follows the existing hold/answer ledger, its dedupe and session links."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3

import pytest

from colonymind.api.routers import host
from colonymind.hands.sessions import open_store as production_open_store
from colonymind.mind import ask_origin
from test_hands_permit_answers import (app_answer, fx, hold_other, pinned_clock, rig as permit_rig)  # noqa: F401
from test_mind_loop import OWNER

PATH = "/v1/host/queue/approvals/history"


@pytest.fixture
async def history_rig(permit_rig, monkeypatch):
    rig = permit_rig
    monkeypatch.setenv("COLONYMIND_HOME", str(Path(rig.store.path).parent))
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", OWNER)
    rig.client._transport.app.include_router(host.router)
    return rig


async def test_real_holds_and_answers_have_counts_detail_and_idempotent_receipts(history_rig, tmp_path):
    rig = history_rig
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    assert held == await rig.service.hold("h-000000000001", rig.body, "hold-retry")
    other, _ = await hold_other(rig, tmp_path, hand="h-000000000002", task="t-000000000002", name="second")
    await rig.store.db.execute("UPDATE hands SET parent_session='other-session' WHERE id='h-000000000002'")
    await rig.store.db.commit()
    first = (await rig.client.get(PATH, params={"limit": 1})).json()
    assert first["counts"] == {"pending": 2, "settled": 0, "total": 2}
    assert first["sessions"] == [
        {"session_id": "other-session", "pending": 1, "settled": 0, "total": 1},
        {"session_id": "owner-session", "pending": 1, "settled": 0, "total": 1},
    ]
    pending = (await rig.client.get(PATH + "/" + held.ask_id)).json()["approval"]
    assert pending["hand_id"] == "h-000000000001" and pending["session_id"] == "owner-session"
    assert pending["terms"]["call"]["arguments"] == rig.body.call.arguments
    assert pending["terms"]["call"]["targets"] == [item.model_dump() for item in rig.body.targets]
    assert pending["settlement"] is None
    answered = await app_answer(rig, held.ask_id, key="same-answer")
    assert answered.status_code == 200
    detail = (await rig.client.get(PATH + "/" + held.ask_id)).json()
    assert detail["approval"]["outcome"] == "allowed"
    assert detail["approval"]["settlement"] == answered.json()["settlement"]
    assert detail["approval"]["settled_at"] == answered.json()["winner"]["at"]
    assert (await app_answer(rig, held.ask_id, key="same-answer")).json()["replayed"] is True
    replay = (await rig.client.get(PATH + "/" + held.ask_id)).json()
    assert replay == detail
    changed = await rig.client.get(PATH, params={"limit": 1, "cursor": first["next_cursor"]})
    assert changed.status_code == 409
    denied = await app_answer(rig, other.ask_id, "no")
    assert denied.status_code == 200
    after = (await rig.client.get(PATH)).json()
    assert after["counts"] == {"pending": 0, "settled": 2, "total": 2}
    assert {item["outcome"] for item in after["items"]} == {"allowed", "denied"}
    filtered = (await rig.client.get(PATH, params={"session_id": "owner-session", "state": "pending"})).json()
    assert filtered["items"] == [] and filtered["counts"] == {"pending": 0, "settled": 1, "total": 1}


async def test_verified_session_survives_controller_retention_and_unknown_links_stay_null(history_rig):
    rig = history_rig
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    db = rig.fx.store._db
    ask_origin.initialize(db)
    db.execute("INSERT INTO hand_ask_origins VALUES (?,?,?,?,?)", (held.ask_id, "h-000000000001", OWNER,
                                                                "owner-session", json.dumps({})))
    db.commit()
    before = (await rig.client.get(PATH)).json()
    await rig.store.db.execute("DELETE FROM permits")
    await rig.store.db.execute("DELETE FROM hands")
    await rig.store.db.commit()
    assert (await rig.client.get(PATH)).json() == before
    db.execute("UPDATE hand_ask_origins SET owner_id='another-owner'")
    db.commit()
    after = (await rig.client.get(PATH)).json()
    assert after["items"][0]["session_id"] is None
    assert after["sessions"] == [{"session_id": None, "pending": 1, "settled": 0, "total": 1}]
    assert after["revision"] != before["revision"]


async def test_session_link_change_invalidates_cursor(history_rig, tmp_path):
    rig = history_rig
    await rig.service.hold("h-000000000001", rig.body, "hold")
    await hold_other(rig, tmp_path, hand="h-000000000002", task="t-000000000002", name="second")
    first = (await rig.client.get(PATH, params={"limit": 1})).json()
    await rig.store.db.execute("UPDATE hands SET parent_session='new-session' WHERE id='h-000000000001'")
    await rig.store.db.commit()
    response = await rig.client.get(PATH, params={"limit": 1, "cursor": first["next_cursor"]})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "approval_history_changed"


async def test_existing_automatic_reconsideration_is_mapped_only_after_a_recorded_ask(history_rig):
    from colonymind.mind.authority import Verdict
    rig = history_rig
    row, _ = rig.fx.store.create_intention(kind="task", type="research", title="Prior owner ask",
        drive="curiosity", cls="self", decision="ask", decision_reason="needs a decision", status="asked",
        dedup_key="prior-ask", ask_code="ABC234", created_at=rig.fx.now)
    rig.mind._apply_decision(row, Verdict(decision="act", reason="existing policy allows this", cls="self"),
                             rig.fx.now, reconsidered=True)
    response = await rig.client.get(PATH + "/" + row.id)
    assert response.status_code == 200
    assert response.json()["approval"]["outcome"] == "auto"
    assert response.json()["approval"]["settled_at"] == rig.fx.now.isoformat()


async def test_registered_hand_cannot_use_an_owner_credential_to_read_history(history_rig):
    from colonymind.hands import sessions
    from test_hands_permits import EPOCH
    rig = history_rig
    await sessions.register(rig.store, hand_id="h-000000000001", session_id="child-session",
                             parent_session="owner-session", epoch=EPOCH)
    for path in (PATH, PATH + "/unknown"):
        response = await rig.client.get(path, headers={"X-ColonyMind-Session": "child-session"})
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "hand_session_refused"
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("injected", [False, True])
async def test_gets_leave_database_files_and_change_counters_unchanged(history_rig, monkeypatch, injected):
    from colonymind.hands import sessions
    from test_hands_permits import EPOCH

    rig = history_rig
    monkeypatch.setattr(sessions, "open_store", production_open_store)
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    await sessions.register(rig.store, hand_id="h-000000000001", session_id="child-session",
                            parent_session="owner-session", epoch=EPOCH)
    if not injected:
        del rig.client._transport.app.state.hands_store
    paths = [Path(rig.store.path), Path(rig.fx.store._db_path)]

    def files():
        return {str(path) + suffix: (Path(str(path) + suffix).read_bytes()
                                    if Path(str(path) + suffix).exists() else None)
                for path in paths for suffix in ("", "-wal")}

    with closing(sqlite3.connect(paths[0])) as hands, closing(sqlite3.connect(paths[1])) as initiatives:
        databases = (hands, initiatives)
        versions = [db.execute("PRAGMA data_version").fetchone()[0] for db in databases]
        before = files()
        counters = [int.from_bytes(path.read_bytes()[24:28], "big") for path in paths]
        for path in (PATH, PATH + "/" + held.ask_id):
            for session, expected in (("owner-session", 200), ("child-session", 403)):
                response = await rig.client.get(path, params={"session_id": session})
                assert response.status_code == expected, response.text
                if expected == 403:
                    assert response.json()["detail"]["code"] == "hand_session_refused"
                assert files() == before
                assert [db.execute("PRAGMA data_version").fetchone()[0] for db in databases] == versions
                assert [int.from_bytes(p.read_bytes()[24:28], "big") for p in paths] == counters
