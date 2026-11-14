"""Contact store persistence + env config (v0.17.0).

The production server must not run the contact store in :memory: — it is
the source of truth for who people are, so it has to survive restarts.
"""

import os
import sqlite3

import pytest

from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore


def test_from_env_defaults_to_state_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("COLONYMIND_CONTACTS_DB", raising=False)
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    cfg = ContactsConfig.from_env()
    assert cfg.sqlite_path == os.path.join(str(tmp_path), "colonymind-contacts.db")


def test_from_env_explicit_path_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_CONTACTS_DB", str(tmp_path / "custom.db"))
    assert ContactsConfig.from_env().sqlite_path == str(tmp_path / "custom.db")


@pytest.mark.asyncio
async def test_contacts_survive_reconnect(tmp_path):
    path = str(tmp_path / "contacts.db")

    store = SQLiteContactStore(config=ContactsConfig(sqlite_path=path))
    await store.connect()
    contact = await store.create(display_name="Owner Test", trust_tier="inner_circle")
    await store.add_handle(contact.contact_id, gateway="whatsapp",
                           address="12345@lid", is_primary=True)
    await store.close()

    # Fresh connection on the same file — the restart scenario.
    store2 = SQLiteContactStore(config=ContactsConfig(sqlite_path=path))
    await store2.connect()
    loaded = await store2.get(contact.contact_id)
    assert loaded is not None
    assert loaded.display_name == "Owner Test"
    by_handle = await store2.resolve_handle("whatsapp", "12345@lid")
    assert by_handle is not None and by_handle.contact_id == contact.contact_id
    await store2.close()


@pytest.mark.asyncio
async def test_startup_preserves_corrected_contacts_without_graph(tmp_path, monkeypatch):
    from colonymind import server
    from colonymind.api.routers import host

    monkeypatch.setenv("COLONYMIND_CONTACTS_DB", str(tmp_path / "contacts.db"))
    monkeypatch.setattr(host, "_contacts_store", None)
    store = await server._initialize_contacts_store()
    target = await store.create(
        display_name="Morgan", import_source="world_model", trust_tier="trusted",
        notes="Owner corrected the identity; preserve this contact.",
    )
    await store.update(target.contact_id, person_node_id="person-without-graph-node")
    provisional = await store.create(display_name="Unknown sender")
    await store.add_handle(provisional.contact_id, "sms", "+12125550123", verified=True)
    correction = await store.correct_handle_identity(
        operation_id="owner-contact-correction", performed_by="owner-test",
        gateway="sms", address="+12125550123",
        expected_contact_id=provisional.contact_id, contact_id=target.contact_id,
        evidence_refs=["source:owner-confirmation"], affected_source_ids=["turn:one"],
    )
    assert correction["authority_granted"] is False
    with sqlite3.connect(tmp_path / "contacts.db") as db:
        db.execute("UPDATE contacts SET relationship_score=? WHERE contact_id=?",
                   (0.35, target.contact_id))
    expected_contacts = {c.contact_id: c.to_dict() for c in await store.list()}
    expected_evidence = await store.identity_evidence(target.contact_id)
    expected_audit = await store.get_audit_log(target.contact_id)
    await store.close()

    # Exercise the same initialization boundary used by the server twice.
    # A missing graph node must never delete the corrected canonical identity.
    for _ in range(2):
        store = await server._initialize_contacts_store()
        try:
            assert host._contacts_store is store
            assert {c.contact_id: c.to_dict() for c in await store.list()} == expected_contacts
            assert await store.identity_evidence(target.contact_id) == expected_evidence
            assert await store.get_audit_log(target.contact_id) == expected_audit
            contact = await store.resolve_messaging_handle(
                "whatsapp", "12125550123@s.whatsapp.net",
            )
            assert contact.contact_id == target.contact_id
            assert contact.relationship_score == 0.35
        finally:
            await store.close()
