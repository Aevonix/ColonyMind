"""The sealed store: encrypted at rest, private files, deterministic references, no key in backups.

secret-scan: synthetic fixtures (every value below is made up)
"""

from __future__ import annotations

import base64
import os
import sqlite3
import tarfile

import pytest

from colonymind.secrets.detect import REFERENCE
from colonymind.secrets.refs import FORGOTTEN_LABEL, reference_text, references_in
from colonymind.secrets.vault import (KEY_NAME, SecretForgotten, UnknownReference, Vault, VaultUnavailable)

VALUE = "Hunter22!wifi"
OWNER = "p-owner"


def _sealed(vault: Vault, value: str = VALUE, contact: str = OWNER) -> str:
    return vault.seal(value, contact_id=contact, session_id="s-1", kind="labeled", label="wifi password",
                      source_turn="turn-1")


def test_a_value_round_trips_and_is_never_stored_in_the_clear(tmp_path):
    vault = Vault(tmp_path)
    ref = _sealed(vault)
    assert REFERENCE.fullmatch(reference_text(ref, "wifi password"))
    assert vault.resolve(ref) == VALUE
    with sqlite3.connect(vault.db_path) as conn:
        dump = "\n".join(conn.iterdump())
    assert VALUE not in dump
    for path in tmp_path.iterdir():
        assert VALUE.encode() not in path.read_bytes(), path.name
    assert Vault(tmp_path).resolve(ref) == VALUE   # another process, the same key file


def test_files_are_private(tmp_path):
    vault = Vault(tmp_path)
    _sealed(vault)
    assert oct(vault.db_path.stat().st_mode & 0o777) == "0o600"
    assert oct(vault.key_path.stat().st_mode & 0o777) == "0o600"


def test_the_same_value_and_scope_give_the_same_reference_and_two_scopes_two(tmp_path):
    vault = Vault(tmp_path)
    first, again = _sealed(vault), _sealed(vault)
    other = _sealed(vault, contact="p-guest")
    assert first == again and first != other
    assert Vault(tmp_path).seal(VALUE, contact_id=OWNER, kind="labeled", label="x") == first
    assert [entry["ref"] for entry in vault.entries(contact_id=OWNER)] == [first]


def test_a_missing_key_reports_values_unavailable_and_never_invents_them(tmp_path):
    vault = Vault(tmp_path)
    ref = _sealed(vault)
    vault.key_path.unlink()
    reopened = Vault(tmp_path)
    with pytest.raises(VaultUnavailable):
        reopened.resolve(ref)
    assert reopened.entry(ref)["state"] == "unavailable"
    fresh = reopened.seal("Another1!", contact_id=OWNER, kind="labeled", label="x")   # a new key seals new values
    assert reopened.resolve(fresh) == "Another1!"
    with pytest.raises(VaultUnavailable):
        reopened.resolve(ref)


def test_a_wrong_key_reports_unavailable(tmp_path):
    vault = Vault(tmp_path)
    ref = _sealed(vault)
    vault.key_path.write_text(base64.b64encode(os.urandom(32)).decode() + "\n")
    with pytest.raises(VaultUnavailable):
        Vault(tmp_path).resolve(ref)


def test_a_key_file_others_can_read_is_refused(tmp_path):
    vault = Vault(tmp_path)
    ref = _sealed(vault)
    vault.key_path.chmod(0o644)
    reopened = Vault(tmp_path)
    assert reopened.available is False
    with pytest.raises(VaultUnavailable):
        reopened.resolve(ref)


def test_release_returns_the_value_once_and_it_is_never_sealed_again(tmp_path):
    vault = Vault(tmp_path)
    ref = _sealed(vault)
    assert vault.release(ref) == VALUE
    assert _sealed(vault) is None
    assert vault.entry(ref)["state"] == "released"


def test_forget_deletes_the_value_and_keeps_the_reference(tmp_path):
    vault = Vault(tmp_path)
    ref = _sealed(vault)
    assert vault.forget(ref) is True
    with pytest.raises(SecretForgotten):
        vault.resolve(ref)
    entry = vault.entry(ref)
    assert entry["label"] == FORGOTTEN_LABEL and entry["state"] == "forgotten"
    # A new sighting is sealed again, under a new reference: the forgotten one stays forgotten, so no older
    # message that carried it resolves again.
    again = _sealed(vault)
    assert again != ref and vault.resolve(again) == VALUE
    with pytest.raises(SecretForgotten):
        vault.resolve(ref)


def test_planning_a_reference_stores_nothing(tmp_path):
    vault = Vault(tmp_path)
    ref, state = vault.plan(VALUE, contact_id=OWNER)
    assert state == "new" and vault.entries() == []
    assert _sealed(vault) == ref and vault.plan(VALUE, contact_id=OWNER) == (ref, "sealed")
    vault.release(ref)
    assert vault.plan(VALUE, contact_id=OWNER) == (ref, "released")


def test_a_staged_value_forgotten_before_its_source_is_admitted_is_never_stored_again(tmp_path):
    """The plan named a reference the owner then forgot: the commit refuses before it writes anything, so no new
    row holds the value they deleted."""
    from colonymind.secrets.seal import Scope, Staged, seal_text
    vault = Vault(tmp_path)
    ref = _sealed(vault)
    stage = Staged()
    seal_text(f"the wifi password is {VALUE}", scope=Scope(OWNER), vault=vault, stage=stage)
    vault.forget(ref)
    with pytest.raises(VaultUnavailable):
        stage.commit()
    assert [entry["state"] for entry in vault.entries()] == ["forgotten"]


def test_every_reference_a_value_has_had_is_known_forgotten_ones_included(tmp_path):
    """What an erasure's identity check needs: the references a value had in a scope, in the order they were
    given, a forgotten one too (its reference is worked out from the value, never from a stored copy)."""
    vault = Vault(tmp_path)
    first = _sealed(vault)
    vault.forget(first)
    second = _sealed(vault)
    assert vault.identities(VALUE, contact_id=OWNER) == [first, second]
    assert vault.identities(VALUE, contact_id="p-other") == []
    assert vault.identities("never sealed", contact_id=OWNER) == []


def test_two_first_users_of_a_keyring_keep_one_key(tmp_path, monkeypatch):
    """Both find the keyring empty at once; the key is made under a lock by one of them and both use it."""
    import sys
    import threading
    import types
    stored, barrier = {}, threading.Barrier(2)
    keyring = types.ModuleType("keyring")

    def get_password(service, account):
        value = stored.get((service, account))
        if value is None:
            try:
                barrier.wait(timeout=2)   # both first readers see an empty keyring
            except threading.BrokenBarrierError:
                pass
        return value
    keyring.get_password = get_password
    keyring.set_password = lambda service, account, value: stored.__setitem__((service, account), value)
    monkeypatch.setitem(sys.modules, "keyring", keyring)
    keys = []
    threads = [threading.Thread(target=lambda: keys.append(Vault(tmp_path, key_store="keyring")._load_key(create=True)))
               for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(keys) == 2 and keys[0] == keys[1]
    ref = _sealed(Vault(tmp_path, key_store="keyring"))
    assert Vault(tmp_path, key_store="keyring").resolve(ref) == VALUE


def test_unknown_references_and_use_bookkeeping(tmp_path):
    vault = Vault(tmp_path)
    with pytest.raises(UnknownReference):
        vault.resolve("S-AAAAAAAA")
    ref = _sealed(vault)
    vault.note_use(ref, "fill", "send_message")
    entry = vault.entry(ref)
    assert entry["uses"] == 1 and entry["last_used_at"]
    assert vault.uses(ref)[0]["purpose"] == "fill"
    assert references_in(f"see {reference_text(ref, 'wifi password')}")[0].ref_id == ref


def test_a_backup_carries_the_ciphertext_and_never_the_key(tmp_path):
    from colonymind.backup import create_full_backup
    state = tmp_path / "state"
    state.mkdir()
    vault = Vault(state)
    _sealed(vault)
    archive = create_full_backup(state, tmp_path / "out", include_vectors=False)
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        payload = b"".join(bundle.extractfile(member).read() for member in bundle.getmembers() if member.isfile())
    assert any(name.endswith("secrets-vault.db") for name in names)
    assert not any(name.endswith(KEY_NAME) for name in names)
    assert VALUE.encode() not in payload
    assert vault.key_path.read_bytes().strip() not in payload


def test_concurrent_seals_of_one_new_value_store_it_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    vault = Vault(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(lambda _: Vault(tmp_path).seal(VALUE, contact_id=OWNER, kind="labeled", label="x"),
                             range(16)))
    assert len(set(refs)) == 1
    assert len(vault.entries()) == 1


def test_an_encrypted_backup_carries_the_key_only_when_the_owner_turned_that_on(tmp_path):
    from colonymind.backup import create_full_backup, restore_full_backup
    from colonymind.secrets.config import SecretsSettings, set_settings
    state = tmp_path / "state"
    state.mkdir()
    ref = _sealed(Vault(state))
    passphrase = b"correct horse battery staple"
    try:
        for carried in (False, True):
            set_settings(SecretsSettings(key_in_backups=carried))
            archive = create_full_backup(state, tmp_path / f"out-{carried}", passphrase=passphrase,
                                         include_vectors=False)
            restored = tmp_path / f"restored-{carried}"
            restored.mkdir()
            restore_full_backup(archive, restored, passphrase=passphrase)
            again = Vault(restored)
            assert again.entry(ref) is not None
            if carried:
                assert again.resolve(ref) == VALUE
            else:
                assert not again.key_path.exists()
                with pytest.raises(VaultUnavailable):
                    again.resolve(ref)
    finally:
        set_settings(None)


def test_the_secrets_section_is_read_and_a_malformed_one_never_weakens_capture(tmp_path, monkeypatch):
    from colonymind.secrets.config import SecretsSettings, current_settings, parse_settings, set_settings
    from colonymind.secrets.detect import RULES
    assert parse_settings(None) == SecretsSettings()
    settings = parse_settings({"capture": {"ingress": False, "rules": {"card": False}},
                               "vault": {"key_store": "keyring", "key_in_backups": True}})
    assert settings.ingress is False and "card" not in settings.rules and len(settings.rules) == len(RULES) - 1
    assert settings.key_store == "keyring" and settings.key_in_backups is True
    for broken in ({"capture": {"rules": ["no_such_rule"]}}, {"vault": {"key_store": "cloud"}}, "on",
                   {"capture": {"enabled": "maybe"}}):
        with pytest.raises(ValueError):
            parse_settings(broken)
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    (tmp_path / "colonymind.yaml").write_text("secrets:\n  capture:\n    enabled: maybe\n")
    set_settings(None)
    try:
        assert current_settings() == SecretsSettings()
    finally:
        set_settings(None)
