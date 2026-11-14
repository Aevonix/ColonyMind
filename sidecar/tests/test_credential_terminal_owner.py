"""A credential that writes the owner from the terminal says so (docs/CREDENTIALS.md; docs/MIND.md, Asks).

Granting the owner on ``cli`` (or on every platform) means anything that can start a Hermes chat on the machine
writes as the owner, an agent's own shell included, and so can answer the owner's asks. The grant stays the
operator's to make; ``describe`` (the credentials listing, the upgrade report, the sidecar log) and ``colonymind
credentials add|set`` say what it means. No credential approves anything by itself: there is no ``approve``
operation to grant, on the instance key or on any other.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from colonymind import config as configuration
from colonymind.api.credentials import OPERATIONS, CredentialTable, describe, terminal_owner
from colonymind.config import ConfigError, DEFAULTS, save_config

NOTE = "anything that can start a Hermes chat on this machine writes as the owner"


def test_a_grant_on_the_terminal_or_everywhere_says_what_it_means():
    table = CredentialTable.from_config({
        "api-key": {"owner": ["cli", "telegram"], "contacts": "all"},
        "desk": {"sha256": "b" * 64, "owner": "any", "contacts": "all"},
        "voice-gw": {"sha256": "c" * 64, "owner": ["voice"], "contacts": "all"}})
    assert terminal_owner(table.instance) and terminal_owner(table.get("desk"))
    assert not terminal_owner(table.get("voice-gw")) and not terminal_owner(CredentialTable.from_config(None).instance)
    assert NOTE in describe(table.instance) and "answer the owner's asks" in describe(table.instance)
    assert describe(table.instance).startswith("may write as the owner from cli and telegram")
    assert NOTE in describe(table.get("desk"))
    assert NOTE not in describe(table.get("voice-gw"))
    assert NOTE not in describe(CredentialTable.from_config(None).instance)


def test_no_credential_can_be_granted_an_approval():
    """The console credential of the first design is not built: an agent with a shell could mint it too."""
    assert "approve" not in OPERATIONS
    for entry in ({"api-key": {"owner": ["whatsapp"], "access": ["read", "write", "manage", "approve"]}},
                  {"console": {"sha256": "d" * 64, "owner": ["cli"], "contacts": "all", "access": ["approve"]}}):
        with pytest.raises(ConfigError, match="access"):
            configuration.validate(configuration._merge(configuration.DEFAULTS, {"credentials": entry}))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    save_config({**DEFAULTS, "owner": {"contact_id": "p-01"}}, tmp_path)
    return tmp_path


def _cli(home, **argv):
    from colonymind.credentials_cli import run
    return run(SimpleNamespace(home=str(home), **argv))


def test_the_credentials_cli_says_so_when_it_grants_the_terminal(home, capsys):
    assert _cli(home, command="set", name="api-key", owner_platform=["whatsapp", "cli"], contacts=None) == 0
    out = capsys.readouterr().out
    assert NOTE in out and "colonymind doctor" in out
    assert _cli(home, command="add", name="desk", owner_platform=["cli"], contacts="all", key_file=None) == 0
    out = capsys.readouterr().out
    assert NOTE in out
    assert _cli(home, command="set", name="api-key", owner_platform=["whatsapp"], contacts=None) == 0
    out = capsys.readouterr().out
    assert NOTE not in out
    assert yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["api-key"]["owner"] == ["whatsapp"]
