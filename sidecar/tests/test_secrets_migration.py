"""Sealing what was stored before capture sealed: counted by a dry run, sealed once, lineage kept attached.

The migration seals the ledger's turns (their digests recomputed, their index rewritten, their vectors queued
again), re-points every derived record that cites a changed message or version, seals the other memory stores in
place, never touches a store that is not memory, is idempotent and resumable, and reports counts only.

secret-scan: synthetic fixtures (every value below is made up and assembled at run time)
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from colonymind.contacts.comms import CommsLog
from colonymind.secrets.config import SecretsSettings, set_settings
from colonymind.secrets.detect import REFERENCE
from colonymind.secrets.migrate import MARKER, REPORT_NAME, migrate, migrate_on_start, release
from colonymind.secrets.refs import references_in
from colonymind.secrets.vault import Vault, reset_vaults
from colonymind.turns import TurnIdempotencyLedger, canonical_turn_digest
from colonymind.turns.idempotency import source_message_hash

KEY = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
WIFI = "Hunter" + "22!" + "wifi"
OWNER = "p-owner"


def dump(path) -> str:
    with sqlite3.connect(path) as conn:
        return "\n".join(conn.iterdump())


def messages_of(path, turn_id):
    with sqlite3.connect(path) as conn:
        return json.loads(conn.execute("SELECT messages_json FROM turn_sources WHERE turn_id=?",
                                       (turn_id,)).fetchone()[0])


@pytest.fixture
def old_state(tmp_path):
    """An instance as an earlier release left it: turns stored as they arrived, with what derives from them."""
    reset_vaults()
    set_settings(SecretsSettings(capture_enabled=False))      # the earlier release stored every word
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    raw = [{"role": "user", "content": f"the wifi password is {WIFI} and my token is {KEY}"},
           {"role": "assistant", "content": "Saved."}]
    ledger.record_source("turn-raw", contact_id=OWNER, session_id="s-1", messages=raw)
    version = canonical_turn_digest(raw)
    ledger.record_source("turn-child", contact_id=OWNER, session_id="s-1", messages=[
        {"role": "assistant", "content": "About that wifi: noted.",
         "_supplied_sources": [{"source_id": "turn-raw", "source_version": version}]}])
    ledger.record_source("turn-plain", contact_id=OWNER, session_id="s-1",
                         messages=[{"role": "user", "content": "Order 1234567890123, commit a3f5c9e."}])
    ledger.append_source_annotation(contact_id=OWNER, session_id="s-1", annotation_id="note-1",
                                    source_id="turn-raw", source_version=version, excerpt="wifi password",
                                    correction="it changed in May", author_principal="owner")
    log = CommsLog(str(tmp_path / "colonymind-comms.db"), source_ledger=ledger)
    log.log(OWNER, summary=f"The owner said the wifi password is {WIFI}.", session_id="s-1")
    with sqlite3.connect(tmp_path / "colonymind-guard-audit.db") as conn:   # not memory: never rewritten
        conn.execute("CREATE TABLE audit (detail TEXT)")
        conn.execute("INSERT INTO audit VALUES (?)", (f"token {KEY}",))
    with sqlite3.connect(tmp_path / "colonymind-facts.db") as conn:     # append-only: reported, never forced
        conn.execute("CREATE TABLE facts_log (contact_id TEXT, text TEXT)")
        conn.execute("CREATE TRIGGER facts_log_frozen BEFORE UPDATE ON facts_log BEGIN "
                     "SELECT RAISE(ABORT, 'append only'); END")
        conn.execute("INSERT INTO facts_log VALUES (?, ?)", (OWNER, f"the wifi password is {WIFI}"))
    set_settings(None)
    yield tmp_path, raw, version
    set_settings(None)
    reset_vaults()


def test_a_dry_run_counts_and_writes_nothing(old_state):
    state, _, _ = old_state
    before = {path.name: dump(path) for path in state.glob("*.db")}
    report = migrate(state, dry_run=True, owner_id=OWNER)
    assert report["dry_run"] is True
    assert report["turn_sources"]["turns_changed"] == 1 and report["turn_sources"]["messages_changed"] == 1
    assert report["rules"] == {"labeled": 3, "prefix": 1}      # the turn, the log's summary, the append-only row
    assert {(item["db"], item["table"]) for item in report["stores"]} >= {("colonymind-comms.db", "communications")}
    assert {path.name: dump(path) for path in state.glob("*.db")} == before
    assert not (state / "secrets-vault.db").exists() and not (state / REPORT_NAME).exists()
    assert WIFI not in json.dumps(report) and KEY not in json.dumps(report)


def test_the_migration_seals_every_memory_store_and_keeps_lineage_attached(old_state):
    state, raw, version = old_state
    report = migrate(state, owner_id=OWNER)
    ledger_db = state / "turn-idempotency.db"
    sealed = messages_of(ledger_db, "turn-raw")
    assert [ref.label for ref in references_in(sealed[0]["content"])] == ["wifi password", "Hugging Face token"]
    for name in ("turn-idempotency.db", "colonymind-comms.db"):
        assert WIFI not in dump(state / name) and KEY not in dump(state / name), name
    assert KEY in dump(state / "colonymind-guard-audit.db")               # not a memory store
    assert WIFI in dump(state / "colonymind-facts.db")                     # append-only: left and reported
    assert report["left_in_append_only_tables"] == [
        {"db": "colonymind-facts.db", "table": "facts_log", "column": "text", "rows": 1}]
    new_version = canonical_turn_digest(sealed)
    with sqlite3.connect(ledger_db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM turn_sources WHERE turn_id='turn-raw'").fetchone()
        assert row["content_sha256"] == canonical_turn_digest({
            "messages": sealed, "contact_id": OWNER, "session_id": "s-1", "scope": "person",
            "occurred_at": None})
        note = conn.execute("SELECT * FROM source_annotations").fetchone()
        assert note["target_version"] == new_version
        assert json.loads(note["target_message_hashes_json"]) == [source_message_hash("s-1", sealed[0])]
        indexed = [r[0] for r in conn.execute("SELECT content FROM turn_source_search WHERE turn_id='turn-raw'")]
        assert any(REFERENCE.search(text) for text in indexed) and not any(WIFI in text for text in indexed)
        assert conn.execute("SELECT count(*) FROM source_vector_jobs WHERE turn_id='turn-raw'").fetchone()[0]
    child = messages_of(ledger_db, "turn-child")[0]
    assert child["_supplied_sources"] == [{"source_id": "turn-raw", "source_version": new_version}]
    assert messages_of(ledger_db, "turn-plain")[0]["content"] == "Order 1234567890123, commit a3f5c9e."
    vault = Vault(state)
    assert {vault.resolve(ref.ref_id) for ref in references_in(sealed[0]["content"])} == {WIFI, KEY}
    assert vault.marker(MARKER)
    written = (state / REPORT_NAME).read_text()
    assert WIFI not in written and KEY not in written
    assert oct((state / REPORT_NAME).stat().st_mode & 0o777) == "0o600"
    hits = TurnIdempotencyLedger(ledger_db).search_sources("wifi password", contact_id=OWNER, session_id="s-9")
    assert hits and WIFI not in json.dumps(hits)


def test_a_second_run_changes_nothing_and_a_new_run_picks_up_where_it_stopped(old_state):
    state, _, _ = old_state
    migrate(state, owner_id=OWNER)
    frozen = {path.name: dump(path) for path in state.glob("*.db") if path.name != "secrets-vault.db"}
    again = migrate(state, owner_id=OWNER)
    assert again["turn_sources"]["turns_changed"] == 0 and again["stores"] == []
    assert {path.name: dump(path) for path in state.glob("*.db") if path.name != "secrets-vault.db"} == frozen
    # Stopped before it marked itself done: the next start runs it again and it finishes the rest.
    set_settings(SecretsSettings(capture_enabled=False))
    TurnIdempotencyLedger(state / "turn-idempotency.db").record_source(
        "turn-late", contact_id=OWNER, session_id="s-2", messages=[{"role": "user", "content": f"key {KEY}"}])
    set_settings(None)
    with sqlite3.connect(state / "secrets-vault.db") as conn:
        conn.execute("DELETE FROM vault_meta")
    thread = migrate_on_start(state, settings=SecretsSettings())
    thread.join(60)
    assert KEY not in dump(state / "turn-idempotency.db")
    assert migrate_on_start(state, settings=SecretsSettings()) is None      # marked: never again


def test_release_puts_a_false_positive_back_everywhere_and_it_is_never_sealed_again(old_state):
    state, _, _ = old_state
    migrate(state, owner_id=OWNER)
    ledger_db = state / "turn-idempotency.db"
    ref = next(item.ref_id for item in references_in(messages_of(ledger_db, "turn-raw")[0]["content"])
               if item.label == "wifi password")
    result = release(ref, state)
    assert result["turns"] == 1 and result["other_rows"] >= 1     # the turn, and the log's summary
    assert WIFI in messages_of(ledger_db, "turn-raw")[0]["content"]
    assert WIFI in dump(state / "colonymind-comms.db")
    TurnIdempotencyLedger(ledger_db).record_source(
        "turn-after", contact_id=OWNER, session_id="s-1",
        messages=[{"role": "user", "content": f"the wifi password is {WIFI}"}])
    assert WIFI in messages_of(ledger_db, "turn-after")[0]["content"]


def test_hermes_transcripts_are_counted_read_only(old_state, tmp_path_factory):
    state, _, _ = old_state
    hermes = tmp_path_factory.mktemp("hermes")
    with sqlite3.connect(hermes / "state.db") as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT)")
        conn.executemany("INSERT INTO messages (session_id, content) VALUES (?, ?)",
                         [("a", f"key {KEY}"), ("a", "hello"), ("b", f"the wifi password is {WIFI}")])
    before = (hermes / "state.db").read_bytes()
    report = migrate(state, dry_run=True, hermes_home=hermes, owner_id=OWNER)
    counted = report["hermes_transcripts"]
    assert counted["messages_with_values"] == 2 and counted["sessions_with_values"] == 2
    assert "hermes sessions delete" in counted["next_step"]
    assert (hermes / "state.db").read_bytes() == before
    assert sorted(path.name for path in hermes.iterdir()) == ["state.db"]


def test_the_cli_uses_and_lists_without_ever_printing_a_value(old_state, monkeypatch, capsys):
    import sys
    from colonymind import cli
    from colonymind.secrets import cli as secrets_cli
    state, _, _ = old_state
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(state))
    monkeypatch.setattr(secrets_cli, "_owner", lambda: OWNER)
    monkeypatch.setattr("colonymind.secrets.migrate._owner_id", lambda: OWNER)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["colonymind", "secrets", *argv])
        cli.main()
        return capsys.readouterr().out

    counted = json.loads(run("migrate", "--dry-run"))
    assert counted["dry_run"] is True and counted["turn_sources"]["turns_changed"] == 1
    assert json.loads(run("migrate"))["turn_sources"]["turns_changed"] == 1
    listing = run("list")
    assert "wifi password" in listing and "Hugging Face token" in listing
    ref = next(item.ref_id for item in references_in(messages_of(state / "turn-idempotency.db", "turn-raw")[0]
                                                   ["content"]) if item.label == "wifi password")
    target = state / "router.conf"
    target.write_text(f"psk=[secret {ref}: wifi password]\n")
    target.chmod(0o600)
    used = run("use", ref, "to_file", str(target))
    assert target.read_text() == f"psk={WIFI}\n" and oct(target.stat().st_mode & 0o777) == "0o600"
    shown = run("show", ref)
    assert json.loads(shown)["uses"] == 1
    run("forget", ref)
    assert Vault(state).entry(ref)["state"] == "forgotten"
    for printed in (counted, listing, used, shown):
        assert WIFI not in str(printed) and KEY not in str(printed)


def test_a_vault_that_cannot_take_a_value_stops_the_migration_and_changes_nothing(old_state):
    """A locked keyring or an unreadable key: nothing is withheld over what was stored, no marker is written, and
    the next run (the key back) seals everything."""
    from colonymind.secrets.vault import VaultUnavailable
    state, _, _ = old_state
    before = {path.name: dump(path) for path in state.glob("*.db")}

    class Locked(Vault):
        def _load_key(self, *, create):
            raise VaultUnavailable("the keyring is locked")
    locked = Locked(state)
    with pytest.raises(VaultUnavailable):
        migrate(state, vault=locked, owner_id=OWNER, write_report=False)
    after = {path.name: dump(path) for path in state.glob("*.db") if path.name in before}
    assert after == before
    assert locked.marker(MARKER) is None
    report = migrate(state, vault=Vault(state), owner_id=OWNER)
    assert report["withheld"] == 0 and report["turn_sources"]["turns_changed"] == 1
    assert WIFI not in dump(state / "turn-idempotency.db")


def test_a_value_inside_json_is_sealed_as_its_bytes(tmp_path):
    """A key block kept inside a JSON column is stored in the vault with its line breaks, not their escapes."""
    reset_vaults()
    pem = "-----BEGIN " + "PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END " + "PRIVATE KEY-----"
    with sqlite3.connect(tmp_path / "colonymind-facts.db") as conn:
        conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, contact_id TEXT, body TEXT)")
        conn.execute("INSERT INTO facts (contact_id, body) VALUES (?, ?)", (OWNER, json.dumps({"note": pem})))
    vault = Vault(tmp_path)
    migrate(tmp_path, vault=vault, owner_id=OWNER, write_report=False)
    with sqlite3.connect(tmp_path / "colonymind-facts.db") as conn:
        body = json.loads(conn.execute("SELECT body FROM facts").fetchone()[0])
    ref, = references_in(body["note"])
    assert vault.resolve(ref.ref_id) == pem
    reset_vaults()


def test_a_historical_tool_observation_is_sealed_without_changing_its_id(tmp_path):
    """Historical observation text still receives generic source sealing."""
    reset_vaults()
    set_settings(SecretsSettings(capture_enabled=False))
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    instruction = [{"role": "user", "content": "check the service"}]
    ledger.record_source("task-instruction:1", contact_id=OWNER, session_id="s-1", messages=instruction)
    origin = {"source_id": "task-instruction:1", "source_version": canonical_turn_digest(instruction)}
    content = f"service is up; token {KEY}"
    ledger.record_source('historical-observation', contact_id=OWNER, session_id='s-1', derive_claims=False,
        messages=[{'role': 'tool', 'content': content, '_native_tool_observation': 'native-tool-observation-v1',
                   '_observation_sources': [origin]}])
    set_settings(None)
    migrate(tmp_path, vault=Vault(tmp_path), owner_id=OWNER, write_report=False)
    text = dump(tmp_path / "turn-idempotency.db")
    assert KEY not in text
    assert messages_of(tmp_path / 'turn-idempotency.db', 'historical-observation')
    reset_vaults()


def test_a_value_named_only_by_its_json_field_is_sealed(tmp_path):
    """A password recognizable only by its field name (``{"password": ...}``) is sealed when its JSON is decoded,
    as it was when the serialized text was read, and stored as its exact bytes."""
    reset_vaults()
    value = "Synthetic" + "Pass8!Demo"
    with sqlite3.connect(tmp_path / "colonymind-facts.db") as conn:
        conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, contact_id TEXT, body TEXT)")
        conn.execute("INSERT INTO facts (contact_id, body) VALUES (?, ?)",
                     (OWNER, json.dumps({"service": "router", "password": value})))
    vault = Vault(tmp_path)
    migrate(tmp_path, vault=vault, owner_id=OWNER, write_report=False)
    with sqlite3.connect(tmp_path / "colonymind-facts.db") as conn:
        body = json.loads(conn.execute("SELECT body FROM facts").fetchone()[0])
    assert body["service"] == "router"
    ref, = references_in(body["password"])
    assert vault.resolve(ref.ref_id) == value
    reset_vaults()


def test_release_puts_the_value_back_in_every_field_sealing_reached(tmp_path):
    """Sealing reaches a message's metadata and an observation's executed arguments; release returns the value
    to each of them."""
    reset_vaults()
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    ledger.record_source("turn-p", contact_id=OWNER, session_id="s-1", messages=[
        {"role": "tool", "content": "ok", "metadata": {"note": f"used {KEY}"},
         "provenance": {"input": {"arguments": {"command": f"login --token {KEY}"}}}}], derive_claims=False)
    vault = Vault(tmp_path)
    ref, = [entry["ref"] for entry in vault.entries()]
    release(ref, tmp_path, vault=vault)
    message, = messages_of(tmp_path / "turn-idempotency.db", "turn-p")
    assert message["metadata"]["note"] == f"used {KEY}"
    assert message["provenance"]["input"]["arguments"]["command"] == f"login --token {KEY}"
    assert ref not in json.dumps(message)
    reset_vaults()
