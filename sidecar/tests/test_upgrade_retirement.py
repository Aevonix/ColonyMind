"""``colonymind upgrade`` retires a store only with its writers stopped, and last.

These run the whole upgrade (``init.run_upgrade``) without a Hermes install: the interpreter
resolution, the adapter install and ``pip check`` are stubbed, and the instance's user service is a
fake that holds the running sidecar's open store connection. Stores are created with the schema the
release before the prune wrote, in WAL mode, with committed rows.
"""

from __future__ import annotations

from contextlib import closing
import copy
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from colonymind import init

# The channel registry as the release before the prune created it (channels/migrations/001).
V1_CHANNELS = """CREATE TABLE IF NOT EXISTS channels (
    channel_key    TEXT PRIMARY KEY,
    display_name   TEXT NOT NULL,
    gateway_family TEXT NOT NULL,
    manifest_json  TEXT NOT NULL,
    channel_token  TEXT NOT NULL,
    registered_at  TEXT NOT NULL,
    last_seen_at   TEXT,
    status         TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'inactive', 'revoked'))
);"""


def open_v1_channels(path: Path) -> sqlite3.Connection:
    """The registry as the previous release's sidecar held it: WAL, no automatic checkpoint."""
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.executescript(V1_CHANNELS)
    return connection


def register(connection: sqlite3.Connection, key: str) -> None:
    connection.execute("INSERT INTO channels (channel_key, display_name, gateway_family, manifest_json, "
                       "channel_token, registered_at) VALUES (?, ?, 'hermes', '{}', 'token', '2026-09-01')",
                       (key, key.title()))
    connection.commit()


def rows(path: Path, sql: str) -> list[tuple]:
    """Read ``path`` and close it: a connection left open would keep the store's WAL beside it."""
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(sql).fetchall()


def channel_keys(path: Path) -> list[str]:
    return sorted(row[0] for row in rows(path, "SELECT channel_key FROM channels"))


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ManagedSidecar:
    """The instance's user service, running the previous release with a store open."""

    def __init__(self, events: list[str], writer: sqlite3.Connection | None) -> None:
        self.events, self.writer, self.live = events, writer, True

    def _owned(self) -> bool:
        return True

    def status(self) -> dict:
        return {"manager": "systemd-user", "installed": True, "loaded": True, "running": self.live,
                "state": "active" if self.live else "inactive", "pid": 4242 if self.live else None}

    def stop(self) -> dict:
        self.events.append("stop")
        if self.writer is not None:
            self.writer.close()          # the process exits: its connections close
        self.live = False
        return self.status()

    def start(self, *, restart: bool = False, timeout: int = 30) -> dict:
        self.events.append("start")
        self.live = True
        return {**self.status(), "ready": True, "health": "ok", "problems": []}


class NoService:
    def _owned(self) -> bool:
        return False


@pytest.fixture
def instance(tmp_path, monkeypatch):
    from colonymind.config import DEFAULTS, save_config, write_api_key
    home, hermes_home = tmp_path / "colonymind", tmp_path / "hermes"
    home.mkdir(mode=0o700)
    hermes_home.mkdir()
    monkeypatch.setenv("COLONYMIND_HOME", str(home))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    data = copy.deepcopy(DEFAULTS)
    data["hermes"].update(home=str(hermes_home), python=sys.executable)
    data["sidecar"]["port"] = free_port()
    save_config(data, home)
    write_api_key("upgrade-test-key", home)
    events: list[str] = []

    def recorded(name, function):
        def call(*args, **kwargs):
            events.append(name)
            return function(*args, **kwargs)
        return call

    def install_adapter(python, version, *, source=None):
        events.append("install_adapter")
        return True

    monkeypatch.setattr(init, "resolve_hermes_python", lambda explicit, configured, home=None: (Path(sys.executable), "0.21.3"))
    monkeypatch.setattr(init, "installed_adapter_version", lambda python: "0.0.0")
    monkeypatch.setattr(init, "install_adapter", install_adapter)
    monkeypatch.setattr(init, "pip_check", lambda python: (True, ""))
    monkeypatch.setattr(init, "colonymind_in_hermes_environment", lambda python: False)
    for name in ("backup_instance", "retire_state", "retire_tables"):
        monkeypatch.setattr(init, name, recorded(name, getattr(init, name)))
    service: dict = {"current": NoService()}
    monkeypatch.setattr(init, "_service", lambda cfg: service["current"])
    return SimpleNamespace(
        home=home, hermes_home=hermes_home, events=events, port=data["sidecar"]["port"], service=service,
        args=SimpleNamespace(home=str(home), hermes_home=None, hermes_python=None, adapter_source=None))


def test_the_upgrade_stops_the_running_service_before_its_backup_and_retires_last(instance, capsys):
    """The service's process holds a retired store open with committed rows in its WAL: the upgrade stops it and
    waits before copying anything, moves the store aside only after the adapter and the Hermes keys are in place,
    and then starts the service again. Every committed row is in the moved-aside store."""
    store = instance.home / "colonymind-channels.db"
    writer = open_v1_channels(store)
    register(writer, "whatsapp")
    register(writer, "sms")
    assert (instance.home / "colonymind-channels.db-wal").stat().st_size > 0
    instance.service["current"] = ManagedSidecar(instance.events, writer)

    assert init.run_upgrade(instance.args) == 0
    out = capsys.readouterr().out
    events = instance.events
    assert events.index("stop") < events.index("backup_instance") < events.index("install_adapter") \
        < events.index("retire_state") < events.index("start"), events
    assert events.count("stop") == 1 and events[-1] == "start"
    assert "stopped for the upgrade and started again" in out and "retired colonymind-channels.db" in out
    assert not store.exists()
    retired = next(instance.home.glob("backups/*/retired/colonymind-channels.db"))
    assert channel_keys(retired) == ["sms", "whatsapp"]
    assert init.run_upgrade(instance.args) == 0     # adapter still "outdated" here: nothing left to retire
    assert "retired" not in capsys.readouterr().out


def _record_this_process(home: Path) -> None:
    from colonymind.cli import _process_signature
    (home / "sidecar.pid").write_text(str(os.getpid()))
    (home / "sidecar-process.json").write_text(json.dumps({"pid": os.getpid(),
                                                            "signature": _process_signature(os.getpid())}))


@pytest.mark.parametrize("evidence", ["port", "pid record"])
def test_a_sidecar_running_outside_its_service_blocks_the_retirement(instance, capsys, evidence):
    """A sidecar started by hand ('colonymind start') still writes the store: the upgrade refuses before it takes a
    backup or changes anything, and says how to stop it. Once it is stopped, the same upgrade goes through."""
    store = instance.home / "colonymind-channels.db"
    writer = open_v1_channels(store)
    register(writer, "whatsapp")
    listener = None
    if evidence == "port":
        listener = socket.socket()
        listener.bind(("127.0.0.1", instance.port))
        listener.listen()
    else:
        _record_this_process(instance.home)

    assert init.run_upgrade(instance.args) == 1
    out = capsys.readouterr().out
    assert "outside the sidecar service" in out and "colonymind stop" in out and "colonymind-channels.db" in out
    assert "failed while checking that no other sidecar writes the stores" in out and "nothing was changed" in out
    assert instance.events == [] and not (instance.home / "backups").exists()
    assert store.exists() and channel_keys(store) == ["whatsapp"]

    if listener is not None:
        listener.close()
    (instance.home / "sidecar-process.json").unlink(missing_ok=True)
    (instance.home / "sidecar.pid").unlink(missing_ok=True)
    writer.close()
    assert init.run_upgrade(instance.args) == 0
    assert not store.exists()
    assert channel_keys(next(instance.home.glob("backups/*/retired/colonymind-channels.db"))) == ["whatsapp"]


def test_a_failed_adapter_install_leaves_every_store_in_place_and_the_service_running(instance, capsys, monkeypatch):
    """The adapter install fails after the service stopped and the backup was taken: no store has moved, every
    committed row is readable where the sidecar opens it, and the service runs again. The next upgrade retires."""
    store = instance.home / "colonymind-channels.db"
    writer = open_v1_channels(store)
    register(writer, "whatsapp")
    register(writer, "sms")
    instance.service["current"] = ManagedSidecar(instance.events, writer)

    def failing(python, version, *, source=None):
        instance.events.append("install_adapter")
        raise init.InitError("adapter install failed in Hermes' environment: no matching distribution")

    monkeypatch.setattr(init, "install_adapter", failing)
    assert init.run_upgrade(instance.args) == 1
    out = capsys.readouterr().out
    assert "colonymind upgrade failed while installing the adapter: adapter install failed" in out
    assert "started again" in out and "no store was moved aside" in out
    assert "retire_state" not in instance.events and instance.events[-1] == "start"
    assert store.exists() and channel_keys(store) == ["sms", "whatsapp"]
    assert not list(instance.home.glob("backups/*/retired"))

    instance.service["current"] = ManagedSidecar(instance.events, None)
    monkeypatch.setattr(init, "install_adapter", lambda python, version, source=None: True)
    assert init.run_upgrade(instance.args) == 0
    assert not store.exists()
    assert channel_keys(next(instance.home.glob("backups/*/retired/colonymind-channels.db"))) == ["sms", "whatsapp"]



def test_a_corrupt_retired_store_fails_the_upgrade_and_the_service_runs_again(instance, capsys):
    """SQLite cannot read a retired store to back it up. The upgrade fails before changing anything, names the
    store, leaves it as it is, and starts the service it stopped."""
    corrupt = b"not a database, " * 64
    (instance.home / "colonymind-channels.db").write_bytes(corrupt)
    instance.service["current"] = ManagedSidecar(instance.events, None)

    assert init.run_upgrade(instance.args) == 1
    out = capsys.readouterr().out
    assert "colonymind upgrade failed while taking the backup: colonymind-channels.db could not be read" in out
    assert "nothing was changed" in out and "no store was moved aside" in out
    assert "not done: taking the backup, applying the store migrations, installing the adapter" in out
    assert instance.events == ["stop", "backup_instance", "start"] and "started again" in out
    assert (instance.home / "colonymind-channels.db").read_bytes() == corrupt


def test_an_unexpected_error_while_retiring_is_reported_and_the_service_runs_again(instance, capsys, monkeypatch):
    """Any exception, not only the upgrade's own: the report says what changed, which stores moved aside and
    which did not, and the service the upgrade stopped runs again."""
    for name in ("colonymind-channels.db", "colonymind-facts.db"):
        writer = open_v1_channels(instance.home / name)
        register(writer, "whatsapp")
        writer.close()
    instance.service["current"] = ManagedSidecar(instance.events, None)
    real = init.retire_state

    def retire_one_then_fail(home, backup_dir):
        monkeypatch.setattr(init, "RETIRED_STATE", ("colonymind-channels.db",))
        real(home, backup_dir)
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(init, "retire_state", retire_one_then_fail)
    assert init.run_upgrade(instance.args) == 1
    out = capsys.readouterr().out
    assert "colonymind upgrade failed while moving aside the retired stores: RuntimeError: the disk went away" in out
    assert "changed before the failure:" in out and "backup taken in" in out and "plugins.enabled += colonymind" in out
    assert "moved aside before the failure: colonymind-channels.db; left in place: colonymind-facts.db" in out
    assert instance.events[0] == "stop" and instance.events[-1] == "start" and "started again" in out
    assert (instance.home / "colonymind-facts.db").exists()


def test_an_interrupted_upgrade_starts_the_service_again_and_still_stops(instance, capsys, monkeypatch):
    """Ctrl-C during the adapter install: the report and the service's start come first, then the interrupt
    goes on, so the command still stops."""
    instance.service["current"] = ManagedSidecar(instance.events, None)

    def interrupted(python, version, *, source=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(init, "install_adapter", interrupted)
    with pytest.raises(KeyboardInterrupt):
        init.run_upgrade(instance.args)
    out = capsys.readouterr().out
    assert "colonymind upgrade failed while installing the adapter: KeyboardInterrupt" in out
    assert "no store was moved aside" in out and "started again" in out
    assert instance.events[0] == "stop" and instance.events[-1] == "start"

def as_110(home: Path) -> None:
    """``colonymind.yaml`` as 1.10 wrote it (one key, no credential grants), with a 1.9.0 keyring 1.10 no longer
    honoured left on disk: the live shape an upgrade from 1.10 meets."""
    import yaml
    path = home / "colonymind.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.pop("credentials", None)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    (home / "api-keyring.json").write_text(json.dumps({"version": 1, "principals": [
        {"principal": "voice", "status": "active", "scopes": ["*"],
         "credentials": [{"id": "c", "secret": "stale-keyring-secret-0000000000", "status": "active"}]}]}))


def grants(home: Path) -> dict | None:
    import yaml
    return (yaml.safe_load((home / "colonymind.yaml").read_text()) or {}).get("credentials")


def test_one_upgrade_from_110_records_the_grants_and_retires_with_the_service_stopped(instance, capsys, monkeypatch):
    """The credential migration and the retirement compose: one upgrade stops the service, takes the backup,
    records the instance key's grants (and retires the stale keyring), installs the adapter, moves the retired
    store aside last and starts the service again. A second upgrade has neither left to do."""
    from colonymind.api.credentials import default_credentials
    as_110(instance.home)
    store = instance.home / "colonymind-channels.db"
    writer = open_v1_channels(store)
    register(writer, "whatsapp")
    instance.service["current"] = ManagedSidecar(instance.events, writer)
    real = init.record_credential_grants

    def recorded(home):
        instance.events.append("record_credential_grants")
        return real(home)

    monkeypatch.setattr(init, "record_credential_grants", recorded)
    assert init.run_upgrade(instance.args) == 0
    out = capsys.readouterr().out
    events = instance.events
    assert events.index("stop") < events.index("backup_instance") < events.index("record_credential_grants") \
        < events.index("install_adapter") < events.index("retire_state") < events.index("start"), events
    assert events.count("stop") == 1 and events[-1] == "start"
    assert grants(instance.home) == default_credentials()
    assert "credential grants recorded in colonymind.yaml" in out and "retired colonymind-channels.db" in out
    assert "api-keyring.json is a 1.9.0 keyring this instance no longer honoured" in out
    assert "stale-keyring-secret" not in out and not store.exists()
    assert init.run_upgrade(instance.args) == 0
    again = capsys.readouterr().out
    assert "credential grants recorded" not in again and "retired" not in again


@pytest.mark.parametrize("failing", ["recording the grants", "installing the adapter"])
def test_a_failure_after_the_stop_starts_the_service_and_the_next_upgrade_finishes(instance, capsys, monkeypatch,
                                                                                   failing):
    """Whatever fails once the service is stopped, the credential step or a later one, the service runs again and
    no store has moved; the grants stay recorded only when their own step finished. The next upgrade does the
    rest."""
    from colonymind.api.credentials import default_credentials
    as_110(instance.home)
    store = instance.home / "colonymind-channels.db"
    writer = open_v1_channels(store)
    register(writer, "whatsapp")
    instance.service["current"] = ManagedSidecar(instance.events, writer)

    def refused(*args, **kwargs):
        raise OSError("the disk went away")

    owner, name = (init.configuration, "update_config") if failing == "recording the grants" \
        else (init, "install_adapter")
    working = getattr(owner, name)
    monkeypatch.setattr(owner, name, refused)
    assert init.run_upgrade(instance.args) == 1
    out = capsys.readouterr().out
    step = "applying the store migrations" if failing == "recording the grants" else "installing the adapter"
    assert f"colonymind upgrade failed while {step}: OSError: the disk went away" in out
    assert "started again" in out and "no store was moved aside" in out
    assert instance.events[0] == "stop" and instance.events[-1] == "start" and "retire_state" not in instance.events
    assert store.exists() and channel_keys(store) == ["whatsapp"]
    assert grants(instance.home) == (None if failing == "recording the grants" else default_credentials())

    monkeypatch.setattr(owner, name, working)
    instance.service["current"] = ManagedSidecar(instance.events, None)
    assert init.run_upgrade(instance.args) == 0
    assert grants(instance.home) == default_credentials() and not store.exists()


class InterruptedWhileStopping(ManagedSidecar):
    """The stop command went through, and Ctrl-C lands while the service manager is still being polled."""

    def stop(self) -> dict:
        super().stop()
        raise KeyboardInterrupt


class InterruptedBeforeStarting(ManagedSidecar):
    """Ctrl-C lands while the start is being prepared, before the start command is issued."""

    def start(self, *, restart: bool = False, timeout: int = 30) -> dict:
        if not getattr(self, "interrupted", False):
            self.interrupted = True
            raise KeyboardInterrupt
        return super().start(restart=restart, timeout=timeout)


@pytest.mark.parametrize("service", [InterruptedWhileStopping, InterruptedBeforeStarting])
def test_an_interrupt_while_stopping_or_starting_still_leaves_the_service_running(instance, capsys, service):
    """Wherever Ctrl-C lands after the upgrade began stopping a running service, the service runs again."""
    managed = service(instance.events, None)
    instance.service["current"] = managed
    with pytest.raises(KeyboardInterrupt):
        init.run_upgrade(instance.args)
    assert managed.live, "the service the upgrade stopped runs again"
    assert instance.events[0] == "stop" and instance.events[-1] == "start"


# What 'colonymind mcp setup' wrote before the prune: a server registration that launches 'colonymind mcp', and a
# diagnostic skill in the coding harnesses' skill directories.
V1_MCP_ENTRY = {"command": "colonymind", "args": ["mcp"],
                "env": {"COLONYMIND_URL": "http://127.0.0.1:7777", "COLONYMIND_MCP_CONTACT_ID": "owner-contact",
                        "COLONYMIND_MCP_SOURCE": "hermes"}}
V1_DIAGNOSTIC_SKILL = """---
name: colonymind-diagnose
description: Diagnose ColonyMind connection failures and missing recalled context using the selected instance.
---

# ColonyMind diagnostics

1. Run `colonymind status` and `colonymind doctor`. Read the reported endpoint and instance
   instead of assuming a directory, machine, port, or database backend.
2. For coding harnesses, run `colonymind mcp detect`. Confirm one `colonymind` MCP server
   is configured using `colonymind mcp setup --harness <name>`.
3. Call `colonymind_health`. If it fails, report the actual transport or authentication
   error. Use the selected instance's credential reference; never guess a key,
   print its value, or dump a whole configuration into a transcript.
4. If health passes, use `colonymind_get_context` for the authenticated contact and
   `colonymind_lookup_facts` for a known source-backed fact. Report what was actually
   retrieved. Do not insert test facts into ordinary memory.
5. For Hermes, inspect its selected profile's plugin and memory-provider status.
   Correct the indicated connection or configuration, then repeat the failed
   probe once. A successful HTTP request alone does not prove useful recollection.

"""


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    home = tmp_path / "user"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _hermes_config(instance, **extra):
    import yaml
    path = instance.hermes_home / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"provider": "custom", "default": "test-model"}, **extra},
                                   sort_keys=False))
    return path


def test_the_upgrade_removes_the_generated_mcp_registration_and_keeps_a_backup(instance, user_home, capsys):
    """Hermes would launch 'colonymind mcp', which no longer exists, for every session: the upgrade removes the
    entry the setup wrote, keeps the previous config in the backup, leaves every other server alone and names
    the retired integration. A second upgrade has nothing left to remove."""
    import yaml
    other = {"command": "uvx", "args": ["some-mcp-server"]}
    path = _hermes_config(instance, mcp_servers={"colonymind": V1_MCP_ENTRY, "files": other})

    assert init.run_upgrade(instance.args) == 0
    out = capsys.readouterr().out
    assert "mcp_servers.colonymind removed" in out and "retired ColonyMind MCP server" in out
    assert yaml.safe_load(path.read_text())["mcp_servers"] == {"files": other}
    saved = [yaml.safe_load(p.read_text()) for p in (instance.home / "backups").glob("hermes-config-*.yaml")]
    assert any((cfg.get("mcp_servers") or {}).get("colonymind") == V1_MCP_ENTRY for cfg in saved)

    assert init.run_upgrade(instance.args) == 0
    assert "mcp_servers.colonymind" not in capsys.readouterr().out


def test_an_edited_or_foreign_mcp_registration_is_kept_and_named(instance, user_home, capsys, monkeypatch):
    """An entry the owner edited after the setup wrote it is left as it is and named, and so is one in another
    harness's config: those files belong to applications that may be running, so the upgrade does not edit them.
    The report repeats while they remain, including when there is nothing else to do."""
    import yaml
    edited = {**V1_MCP_ENTRY, "env": {**V1_MCP_ENTRY["env"], "COLONYMIND_API_KEY": "literal"}}
    path = _hermes_config(instance, mcp_servers={"colonymind": edited})
    codex = user_home / ".codex" / "config.toml"
    codex.parent.mkdir()
    codex.write_text('model = "x"\n\n[mcp_servers.colonymind]\ncommand = "colonymind"\nargs = ["mcp"]\n')
    before = codex.read_bytes()

    assert init.run_upgrade(instance.args) == 0
    out = capsys.readouterr().out
    assert f"kept mcp_servers.colonymind in {path}" in out
    assert f"{codex} registers the retired ColonyMind MCP server (mcp_servers.colonymind)" in out
    assert yaml.safe_load(path.read_text())["mcp_servers"] == {"colonymind": edited}
    assert codex.read_bytes() == before

    monkeypatch.setattr(init, "installed_adapter_version", lambda python: init.__version__)
    assert init.run_upgrade(instance.args) == 0
    out = capsys.readouterr().out
    assert "nothing to do" in out and "kept mcp_servers.colonymind" in out and str(codex) in out


def test_the_upgrade_moves_the_generated_diagnostic_skills_aside_and_keeps_edited_ones(instance, user_home, capsys,
                                                                                      monkeypatch):
    """The diagnostic skill the setup installed prescribes 'colonymind mcp' and tools that no longer exist: an
    unedited copy moves into the upgrade backup (a pending move is work, not "nothing to do"); an edited one
    stays and is named."""
    monkeypatch.setattr(init, "installed_adapter_version", lambda python: init.__version__)
    _hermes_config(instance)
    assert init.run_upgrade(instance.args) == 0          # the Hermes keys and the worker profile
    assert init.run_upgrade(instance.args) == 0
    assert "nothing to do" in capsys.readouterr().out
    codex = user_home / ".codex" / "skills" / "colonymind-diagnose"
    opencode = user_home / ".config" / "opencode" / "skills" / "colonymind-diagnose"
    for directory in (codex, opencode):
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(V1_DIAGNOSTIC_SKILL)
    (opencode / "SKILL.md").write_text(V1_DIAGNOSTIC_SKILL + "\nAlso check the proxy.\n")

    assert init.run_upgrade(instance.args) == 0
    out = capsys.readouterr().out
    assert "nothing to do" not in out
    assert f"retired the diagnostic skill {codex}" in out and not codex.exists()
    moved = next(instance.home.glob("backups/*/harness-skills/.codex/skills/colonymind-diagnose/SKILL.md"))
    assert moved.read_text() == V1_DIAGNOSTIC_SKILL
    assert f"kept {opencode}" in out and (opencode / "SKILL.md").exists()
    assert init.run_upgrade(instance.args) == 0
    out = capsys.readouterr().out
    assert "nothing to do" in out and f"kept {opencode}" in out and "retired the diagnostic skill" not in out


def test_uninstall_strips_the_generated_mcp_registration_but_not_an_edited_one(tmp_path):
    stock = {"model": {"provider": "custom"}}
    stripped, changes = init.strip_hermes_config({**stock, "mcp_servers": {"colonymind": V1_MCP_ENTRY}}, skills_dir=None)
    assert stripped == stock and changes == [f"mcp_servers.colonymind removed: {init.MCP_RETIRED}"]
    custom = {**V1_MCP_ENTRY, "command": "/opt/tools/colonymind-wrapper"}
    kept, changes = init.strip_hermes_config({**stock, "mcp_servers": {"colonymind": custom}}, skills_dir=None)
    assert kept["mcp_servers"] == {"colonymind": custom} and changes == []


# The shared facts as the release before the prune created them (tom/facts.py).
V1_FACTS = """CREATE TABLE IF NOT EXISTS shared_facts (
    id TEXT PRIMARY KEY, contact_id TEXT NOT NULL, fact TEXT NOT NULL, source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.8, created_at TEXT NOT NULL, expires_at TEXT, metadata TEXT,
    source_lineage_json TEXT);
CREATE INDEX IF NOT EXISTS idx_shared_facts_contact ON shared_facts(contact_id);"""


def _left_by_a_crashed_writer(staging: Path, home: Path, name: str, write) -> None:
    """``write`` commits to ``staging/name``; the store lands in ``home`` as a writer that stopped without
    closing leaves it: the committed rows only in its WAL."""
    holder = sqlite3.connect(staging / name)          # keeps the last connection from checkpointing
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("PRAGMA wal_autocheckpoint=0")
    holder.execute("SELECT count(*) FROM sqlite_master").fetchall()
    write()
    for suffix in ("", "-wal"):
        import shutil
        shutil.copy2(staging / (name + suffix), home / (name + suffix))
    holder.close()
    assert (home / (name + "-wal")).stat().st_size > 0


def _roll_back(home: Path, backup: Path) -> None:
    """INSTALL.md's rollback, with the sidecar stopped: the backup's files back into the instance directory, each
    store without the WAL and shared-memory files of the store it replaces, then what ``retired/`` holds that is
    not back yet, except SQLite's side files."""
    import shutil
    for path in backup.iterdir():
        if path.is_file():
            for suffix in ("-wal", "-shm"):
                (home / (path.name + suffix)).unlink(missing_ok=True)
            shutil.copy2(path, home / path.name)
    for path in (backup / "retired").iterdir():
        if not (home / path.name).exists() and not path.name.endswith(("-wal", "-shm", "-journal")):
            shutil.move(str(path), str(home / path.name))


def _modes(home: Path, names) -> dict:
    return {name: (home / name).stat().st_mode & 0o777 for name in names}


def test_an_upgrade_and_its_rollback_keep_every_record_the_live_readers_serve(instance, tmp_path, capsys):
    """Surviving stores written through their live writers (the source ledger with committed rows only in its WAL,
    the initiatives, the contacts' affect, the action journal) and the previous release's retired stores. After
    the upgrade the live readers serve every record, scoped as before, with the stores' permissions unchanged, and
    the retired stores sit in the backup with their rows. The documented rollback brings back every retired store,
    table, tree and file with its rows and permissions; upgrading again retires them again and loses nothing."""
    from colonymind.contacts.affect import AffectStore
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.self_model.journal import ActionJournal
    from colonymind.turns import TurnIdempotencyLedger
    home = instance.home
    staging = tmp_path / "staging"
    staging.mkdir()

    def sources():
        ledger = TurnIdempotencyLedger(staging / "turn-idempotency.db")
        with closing(sqlite3.connect(staging / "turn-idempotency.db")) as db, db:   # a table the upgrade retires
            db.execute("CREATE TABLE self_attention (slot INTEGER PRIMARY KEY, snapshot_json TEXT, observed_at REAL)")
            db.execute("INSERT INTO self_attention VALUES (1, '{\"focus\": \"garden\"}', 1.0)")
        ledger.record_source("turn-a", contact_id="contact-a", session_id="session-a",
                             messages=[{"role": "user", "content": "The boiler service is booked for Thursday."}])
        ledger.record_source("turn-b", contact_id="contact-b", session_id="session-b",
                             messages=[{"role": "user", "content": "My sister lands on Friday evening."}])

    def channels():
        with closing(sqlite3.connect(staging / "colonymind-channels.db")) as db:
            db.executescript(V1_CHANNELS)
        writer = sqlite3.connect(staging / "colonymind-channels.db")
        register(writer, "whatsapp")
        register(writer, "sms")
        writer.close()

    for name in ("turn-idempotency.db", "colonymind-channels.db"):
        sqlite3.connect(staging / name).close()
    _left_by_a_crashed_writer(staging, home, "turn-idempotency.db", sources)
    _left_by_a_crashed_writer(staging, home, "colonymind-channels.db", channels)
    initiatives = InitiativeStore(state_dir=home)
    intention, _ = initiatives.create_intention(
        kind="task", type="t", title="renew the passport", drive="duty", cls="owner", decision="act",
        decision_reason="r", status="approved", dedup_key="k-passport", recipient="contact-a")
    initiatives.close()
    affect = AffectStore(db_path=str(home / "colonymind-affect.db"))
    affect.create_event(contact_id="contact-b", valence=-0.4, source="explicit", trigger="missed the train")
    affect.close()
    journal = ActionJournal(db_path=str(home / "colonymind-action-journal.db"))
    journal.record("expectation", "prediction hit: the parcel arrives today", decision="noted", outcome="hit")
    journal._conn.close()
    with closing(sqlite3.connect(home / "colonymind-facts.db")) as db, db:
        db.executescript(V1_FACTS)
        db.execute("INSERT INTO shared_facts (id, contact_id, fact, source, created_at) VALUES "
                   "('f-1', 'contact-a', 'prefers tea', 'manual', '2026-08-01')")
    (home / "repo-mirrors" / "notes").mkdir(parents=True)
    (home / "repo-mirrors" / "notes" / "HEAD").write_text("ref: main\n")
    (home / "autonomy_checkin.json").write_text('{"last": "2026-09-01"}')
    stores = ("turn-idempotency.db", "colonymind-channels.db", "initiatives.db", "colonymind-affect.db",
              "colonymind-action-journal.db", "colonymind-facts.db")
    for name in stores:
        (home / name).chmod(0o600)
    modes = _modes(home, stores)

    def readers_serve_every_record():
        ledger = TurnIdempotencyLedger(home / "turn-idempotency.db")
        found = ledger.search_sources("boiler service", contact_id="contact-a", session_id="session-a")
        assert [row["turn_id"] for row in found] == ["turn-a"]
        assert ledger.search_sources("boiler service", contact_id="contact-b", session_id="session-b") == []
        assert [row["turn_id"] for row in ledger.search_sources("sister", contact_id="contact-b",
                                                                session_id="session-b")] == ["turn-b"]
        store = InitiativeStore(state_dir=home)
        assert store.get(intention.id).description == "renew the passport"
        store.close()
        feelings = AffectStore(db_path=str(home / "colonymind-affect.db"))
        assert [event["trigger"] for event in feelings.list_events(contact_id="contact-b")] == ["missed the train"]
        feelings.close()
        entries = ActionJournal(db_path=str(home / "colonymind-action-journal.db"))
        assert [entry["description"] for entry in entries.recent()] == ["prediction hit: the parcel arrives today"]
        entries._conn.close()

    assert init.run_upgrade(instance.args) == 0
    capsys.readouterr()
    backup = next(path for path in (home / "backups").iterdir() if (path / "retired").is_dir())
    readers_serve_every_record()
    surviving = ("turn-idempotency.db", "initiatives.db", "colonymind-affect.db", "colonymind-action-journal.db")
    assert _modes(home, surviving) == {name: modes[name] for name in surviving}
    assert (backup.stat().st_mode & 0o777) == 0o700 and ((backup / "retired").stat().st_mode & 0o777) == 0o700
    assert ("self_attention",) not in rows(home / "turn-idempotency.db", "SELECT name FROM sqlite_master")
    for name in ("colonymind-channels.db", "colonymind-facts.db", "repo-mirrors", "autonomy_checkin.json"):
        assert not (home / name).exists() and (backup / "retired" / name).exists()
    assert channel_keys(backup / "retired" / "colonymind-channels.db") == ["sms", "whatsapp"]

    # The documented rollback stops the sidecar, closing its journal monitors.
    from colonymind.memory import curated
    curated._cache.cache_clear()
    _roll_back(home, backup)
    assert channel_keys(home / "colonymind-channels.db") == ["sms", "whatsapp"]
    assert rows(home / "colonymind-facts.db", "SELECT fact FROM shared_facts") == [("prefers tea",)]
    assert rows(home / "turn-idempotency.db", "SELECT snapshot_json FROM self_attention") == [('{"focus": "garden"}',)]
    assert (home / "repo-mirrors" / "notes" / "HEAD").read_text() == "ref: main\n"
    assert (home / "autonomy_checkin.json").read_text() == '{"last": "2026-09-01"}'
    assert _modes(home, stores) == modes
    readers_serve_every_record()

    assert init.run_upgrade(instance.args) == 0
    assert "retired colonymind-channels.db" in capsys.readouterr().out
    readers_serve_every_record()
    again = next(path for path in (home / "backups").iterdir() if path != backup and (path / "retired").is_dir())
    assert channel_keys(again / "retired" / "colonymind-channels.db") == ["sms", "whatsapp"]
    assert init.retired_state_present(home) == [] and init.retired_tables_present(home) == []


def test_the_stores_of_the_owner_gate_learning_drift_twin_and_sealed_store_survive_an_upgrade(instance, tmp_path,
                                                                                            capsys):
    """The owner gate's audit (the mind store), the drift check's state (the mind store), the learning log (the
    source ledger), the sealed store and the twin's configuration, written through their own writers. The running
    sidecar writes all but the twin's, and the upgrade stops it before its backup; the twin's shipper is a process
    of its own beside the sidecar and keeps its read transaction (its pin) on the stores throughout. After the
    upgrade every one is where its reader opens it and serves its record, none is moved aside, the backup holds a
    copy of each store with its rows and never the sealed store's key; the documented rollback keeps them too."""
    from datetime import datetime, timezone
    from colonymind.memory import learning
    from colonymind.mind import drift, gate
    from colonymind.mind.concerns import MIND_DB, Concerns, open_mind_db
    from colonymind.secrets.vault import get_vault, reset_vaults
    from colonymind.turns import TurnIdempotencyLedger
    from colonymind.twin import config as twin_config
    home, now = instance.home, datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    TurnIdempotencyLedger(home / "turn-idempotency.db")
    with closing(sqlite3.connect(home / "turn-idempotency.db")) as conn, conn:
        learning.initialize(conn)
        learning.log(conn, store="claim", action="rejected", stage="admission", reason="no source")
    sidecar = open_mind_db(home / MIND_DB)            # the running sidecar's mind store connection
    gate.Ledger(sidecar).record(gate.Verdict(decision="digest", reason="quiet hours", stage="prior", id="v-1"),
                                gate.Message(type="notice", topic="boiler", text="The boiler service is booked.",
                                             ref="i-1"), now)
    Concerns(sidecar, clock=lambda: now).state.set(drift.OWNER_KEY, text="call me by my first name", now=now)
    reset_vaults()
    value = "-".join(["garden", "gate", str(4400 + 17)])
    ref = get_vault(home, key_store="file").seal(value, contact_id="contact-a", kind="password", label="gate code")
    twin_config.save(home, twin_config.TwinConfig(role="primary", target=str(tmp_path / "standby")))
    with closing(sqlite3.connect(home / "colonymind-facts.db")) as db, db:      # a retired store, so one moves aside
        db.executescript(V1_FACTS)
    pins = [sqlite3.connect(home / name) for name in (MIND_DB, "colonymind-facts.db")]
    for pin in pins:                                   # the twin's pins: read transactions held throughout
        pin.execute("BEGIN")
        pin.execute("SELECT count(*) FROM sqlite_master").fetchall()
    instance.service["current"] = ManagedSidecar(instance.events, sidecar)
    kept = ("turn-idempotency.db", MIND_DB, "secrets-vault.db")

    def readers_serve_every_record():
        with closing(open_mind_db(home / MIND_DB)) as conn:
            assert gate.Ledger(conn).last_for("i-1")["reason"] == "quiet hours"
            assert Concerns(conn, clock=lambda: now).state.get(drift.OWNER_KEY)["text"] == "call me by my first name"
        with closing(sqlite3.connect(home / "turn-idempotency.db")) as conn:
            conn.row_factory = sqlite3.Row
            assert [(e["store"], e["action"]) for e in learning.entries(conn)] == [("claim", "rejected")]
        reset_vaults()
        assert get_vault(home, key_store="file").resolve(ref) == value
        assert twin_config.load(home).target == str(tmp_path / "standby")

    assert init.run_upgrade(instance.args) == 0
    assert "retired colonymind-facts.db" in capsys.readouterr().out
    events = instance.events
    assert events.index("stop") < events.index("backup_instance") < events.index("retire_state") < events.index("start")
    readers_serve_every_record()
    backup = next(path for path in (home / "backups").iterdir() if (path / "retired").is_dir())
    assert sorted(path.name for path in (backup / "retired").iterdir()) == ["colonymind-facts.db"]
    assert {name for name in kept} <= {path.name for path in backup.iterdir()}
    assert not (backup / "secrets-vault.key").exists() and (home / "secrets-vault.key").exists()
    assert rows(backup / MIND_DB, "SELECT reason FROM owner_gate") == [("quiet hours",)]
    assert rows(backup / MIND_DB, f"SELECT text FROM mind_state WHERE key='{drift.OWNER_KEY}'") \
        == [("call me by my first name",)]
    assert rows(backup / "turn-idempotency.db", "SELECT store, action FROM learning_log") == [("claim", "rejected")]
    assert rows(backup / "secrets-vault.db", "SELECT ref FROM sealed") == [(ref,)]
    for pin in pins:
        pin.rollback()
        pin.close()

    _roll_back(home, backup)
    readers_serve_every_record()


def test_two_backups_in_the_same_second_keep_separate_directories(tmp_path, monkeypatch):
    """A retry right after a failed upgrade must not overwrite the backup the first attempt took before it
    changed anything: each backup gets a directory of its own."""
    home = tmp_path / "colonymind"
    home.mkdir()
    with closing(sqlite3.connect(home / "initiatives.db")) as db, db:
        db.execute("CREATE TABLE initiatives (id TEXT)")
        db.execute("INSERT INTO initiatives VALUES ('before')")
    monkeypatch.setattr(init, "_utc_stamp", lambda: "20260927T120000Z")
    first = init.backup_instance(home)
    with closing(sqlite3.connect(home / "initiatives.db")) as db, db:
        db.execute("INSERT INTO initiatives VALUES ('after')")
    second = init.backup_instance(home)
    assert first != second and first.name == "20260927T120000Z"
    assert rows(first / "initiatives.db", "SELECT id FROM initiatives") == [("before",)]
    assert rows(second / "initiatives.db", "SELECT id FROM initiatives") == [("before",), ("after",)]
