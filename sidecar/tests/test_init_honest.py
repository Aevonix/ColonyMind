"""``colonymind init`` says what it changes in Hermes and puts back what it replaced.

Runs against a temporary Hermes home with stock Hermes installed in this interpreter, like ``test_init.py``,
whose fixtures it shares. Another memory provider is always the made-up ``example-memory``.
"""

from __future__ import annotations

import builtins
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest
import yaml

from colonymind import init
from colonymind.config import apply_environment, load_config
from colonymind.init import INSTALL_STATE, read_install_state

from test_init import HERMES_PYTHON, STOCK_CONFIG, _args, homes  # noqa: F401  (pytest fixture)


def _hermes(hermes_home):
    return yaml.safe_load((hermes_home / "config.yaml").read_text())


def _write_hermes(hermes_home, config):
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


# -- (a) the provider switch ------------------------------------------------------------------------------------


def test_init_names_the_provider_it_replaces_and_uninstall_restores_it(homes, capsys):
    home, hermes_home = homes
    _write_hermes(hermes_home, {**STOCK_CONFIG, "memory": {"provider": "example-memory", "memory_char_limit": 2200}})
    assert init.run_init(_args(home, hermes_home)) == 0
    out = capsys.readouterr().out
    assert "replaces memory provider example-memory; 'colonymind init --uninstall' restores it" in out
    assert _hermes(hermes_home)["memory"]["provider"] == "colonymind-memory"
    state = read_install_state(home)
    assert state["replaced_memory_provider"] == "example-memory"
    # The backup of the replaced config names it too.
    backup = home / state["hermes_config_backup"]
    assert yaml.safe_load(backup.read_text())["memory"]["provider"] == "example-memory"
    assert oct((home / INSTALL_STATE).stat().st_mode & 0o777) == "0o600"

    # A second init finds its own provider selected and keeps the record.
    assert init.run_init(_args(home, hermes_home)) == 0
    assert "replaces memory provider" not in capsys.readouterr().out
    assert read_install_state(home)["replaced_memory_provider"] == "example-memory"

    assert init.run_init(_args(home, hermes_home, uninstall=True, hermes_python=None)) == 0
    out = capsys.readouterr().out
    assert "memory.provider restored to example-memory" in out
    assert _hermes(hermes_home)["memory"] == {"provider": "example-memory", "memory_char_limit": 2200}
    assert "replaced_memory_provider" not in read_install_state(home)


def test_uninstall_leaves_a_provider_chosen_after_install_alone(homes, capsys):
    """Restore only while the key still names ColonyMind's provider: a provider the user picked since is theirs."""
    home, hermes_home = homes
    _write_hermes(hermes_home, {**STOCK_CONFIG, "memory": {"provider": "example-memory"}})
    assert init.run_init(_args(home, hermes_home)) == 0
    config = _hermes(hermes_home)
    config["memory"]["provider"] = "another-memory"
    _write_hermes(hermes_home, config)
    capsys.readouterr()
    assert init.run_init(_args(home, hermes_home, uninstall=True, hermes_python=None)) == 0
    out = capsys.readouterr().out
    assert _hermes(hermes_home)["memory"]["provider"] == "another-memory"
    assert "memory.provider is another-memory" in out and "left as it is" in out
    assert "restored" not in out


def test_a_backup_carries_the_install_record_and_restores_the_replaced_provider(homes, capsys):
    """The backup an upgrade takes first holds hermes-install.json: an instance restored from it still puts the
    replaced provider back on uninstall, instead of deleting the selection."""
    import shutil
    home, hermes_home = homes
    _write_hermes(hermes_home, {**STOCK_CONFIG, "memory": {"provider": "example-memory"}})
    assert init.run_init(_args(home, hermes_home)) == 0
    backup = init.backup_instance(home)
    assert json.loads((backup / INSTALL_STATE).read_text()) == read_install_state(home)
    assert oct((backup / INSTALL_STATE).stat().st_mode & 0o777) == "0o600"

    restored = home.parent / "restored"
    restored.mkdir(mode=0o700)
    for item in backup.iterdir():
        shutil.copy2(item, restored / item.name)
    capsys.readouterr()
    assert init.run_init(_args(restored, hermes_home, uninstall=True, hermes_python=None)) == 0
    assert "memory.provider restored to example-memory" in capsys.readouterr().out
    assert _hermes(hermes_home)["memory"] == {"provider": "example-memory"}


def test_init_on_a_home_without_a_provider_says_uninstall_removes_the_key(homes, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home)) == 0
    out = capsys.readouterr().out
    assert "memory.provider: colonymind-memory (none was set; 'colonymind init --uninstall' removes it)" in out
    assert read_install_state(home)["replaced_memory_provider"] == ""
    assert init.run_init(_args(home, hermes_home, uninstall=True, hermes_python=None)) == 0
    assert "memory" not in _hermes(hermes_home)


def test_init_records_the_sidecar_and_adapter_versions_it_installed(homes):
    """The memory provider compares its own version with the sidecar's, from this file, without the network."""
    from colonymind import __version__
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home)) == 0
    state = read_install_state(home)
    assert state["sidecar_version"] == __version__
    assert state["adapter_version"] == init.installed_adapter_version(Path(HERMES_PYTHON)) == __version__


def test_init_states_the_hermes_wide_callback_timeout(homes, capsys):
    """Flagged, not removed: ``plugins.hook_callback_timeout: 0`` holds for every plugin in this Hermes."""
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home)) == 0
    out = capsys.readouterr().out
    assert ("plugins.hook_callback_timeout: 0 is Hermes-wide: every plugin's callbacks run inline, with no "
            "timeout") in out
    assert _hermes(hermes_home)["plugins"]["hook_callback_timeout"] == 0


# -- (g) the four init bugs -------------------------------------------------------------------------------------


def test_init_writes_quiet_hours_where_the_mind_reads_them(homes):
    """The mind reads ``mind.quiet_hours``; init wrote only ``identity.yaml agent.quiet_hours``."""
    from colonymind.mind.authority import Policy
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, quiet_hours="23:30-06:15")) == 0
    cfg = load_config(home, environ={})
    assert cfg.get("mind.quiet_hours") == "23:30-06:15"
    assert Policy.from_config(cfg.get("mind")).quiet_hours == "23:30-06:15"
    # The follow-up scheduler's reader (identity.yaml) says the same.
    assert yaml.safe_load((home / "identity.yaml").read_text())["agent"]["quiet_hours"] == "23:30-06:15"
    # A re-run without the flag keeps the answer.
    assert init.run_init(_args(home, hermes_home, quiet_hours=None)) == 0
    assert load_config(home, environ={}).get("mind.quiet_hours") == "23:30-06:15"


def _quiet(home):
    return (load_config(home, environ={}).get("mind.quiet_hours"),
            yaml.safe_load((home / "identity.yaml").read_text())["agent"]["quiet_hours"])


def test_a_rerun_keeps_quiet_hours_the_owner_changed_since(homes):
    """The mind's policy is the live one: a rerun without a new answer keeps what the owner set there after
    install, and brings identity.yaml into line, instead of writing the older answer back over it."""
    from colonymind.config import update_config
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, quiet_hours="23:30-06:15")) == 0
    update_config({"mind": {"quiet_hours": "21:00-08:00"}}, home)
    assert init.run_init(_args(home, hermes_home, quiet_hours=None)) == 0
    assert _quiet(home) == ("21:00-08:00", "21:00-08:00")
    # Also when the owner sets the mind's default window itself.
    update_config({"mind": {"quiet_hours": "22:00-07:00"}}, home)
    assert init.run_init(_args(home, hermes_home, quiet_hours=None)) == 0
    assert _quiet(home) == ("22:00-07:00", "22:00-07:00")
    # An answer changed in identity.yaml since is the newer one.
    identity = yaml.safe_load((home / "identity.yaml").read_text())
    identity["agent"]["quiet_hours"] = "20:00-06:00"
    (home / "identity.yaml").write_text(yaml.safe_dump(identity))
    assert init.run_init(_args(home, hermes_home, quiet_hours=None)) == 0
    assert _quiet(home) == ("20:00-06:00", "20:00-06:00")
    # A given answer always wins.
    assert init.run_init(_args(home, hermes_home, quiet_hours="23:00-05:00")) == 0
    assert _quiet(home) == ("23:00-05:00", "23:00-05:00")


def test_a_rerun_carries_an_earlier_releases_answer_to_the_mind(homes):
    """Releases before this one wrote the answer to identity.yaml only, and the mind kept its default."""
    from colonymind.config import update_config
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, quiet_hours="23:30-06:15")) == 0
    update_config({"mind": {"quiet_hours": "22:00-07:00"}}, home)
    init.write_install_state(home, quiet_hours=None)              # no record: an instance from before
    assert init.run_init(_args(home, hermes_home, quiet_hours=None)) == 0
    assert _quiet(home) == ("23:30-06:15", "23:30-06:15")


def test_init_defaults_the_time_zone_to_the_systems(homes, monkeypatch):
    home, hermes_home = homes
    monkeypatch.setattr(init, "system_timezone", lambda: "Europe/Paris")
    assert init.run_init(_args(home, hermes_home, timezone=None)) == 0
    assert yaml.safe_load((home / "identity.yaml").read_text())["agent"]["timezone"] == "Europe/Paris"
    environ: dict[str, str] = {}
    apply_environment(load_config(home, environ={}), environ=environ)
    assert environ["COLONYMIND_AGENT_TIMEZONE"] == "Europe/Paris"
    # A zone name that does not exist is refused before anything changes.
    before = (home / "identity.yaml").read_text()
    assert init.run_init(_args(home, hermes_home, timezone="Mars/Olympus")) == 1
    assert (home / "identity.yaml").read_text() == before


def test_the_system_time_zone_comes_from_tz_or_the_localtime_link(tmp_path, monkeypatch):
    zoneinfo = tmp_path / "usr" / "share" / "zoneinfo" / "America"
    zoneinfo.mkdir(parents=True)
    (zoneinfo / "Chicago").write_bytes(b"")
    link = tmp_path / "localtime"
    link.symlink_to(zoneinfo / "Chicago")
    monkeypatch.delenv("TZ", raising=False)
    assert init.system_timezone(localtime=link, timezone_file=tmp_path / "none") == "America/Chicago"
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    assert init.system_timezone(localtime=link, timezone_file=tmp_path / "none") == "Asia/Tokyo"
    monkeypatch.setenv("TZ", "not-a-zone")
    (tmp_path / "timezone").write_text("Europe/Berlin\n")
    assert init.system_timezone(localtime=tmp_path / "missing", timezone_file=tmp_path / "timezone") == "Europe/Berlin"
    assert init.system_timezone(localtime=tmp_path / "missing", timezone_file=tmp_path / "none") == ""


def test_init_resolves_a_named_providers_base_url(homes):
    """Hermes may name a provider (an entry under ``providers:``) instead of giving ``model.base_url``."""
    home, hermes_home = homes
    _write_hermes(hermes_home, {"model": {"provider": "house", "default": "house-model"},
                                "providers": {"house": {"base_url": "http://127.0.0.1:9/v1", "api_key": "house-secret",
                                                        "default_model": "house-model"}}})
    assert init.run_init(_args(home, hermes_home)) == 0
    llm = json.loads((home / ".colonymind-llm-config.json").read_text())
    assert llm["baseUrl"] == "http://127.0.0.1:9/v1" and llm["apiKey"] == "house-secret"
    assert llm["models"]["medium"] == "house-model"


def test_init_resolves_a_hermes_registry_provider(homes):
    """A provider Hermes knows by name (its registry), keyed by the variable Hermes names, in its .env."""
    home, hermes_home = homes
    _write_hermes(hermes_home, {"model": {"provider": "deepseek", "default": "deepseek-chat"}})
    (hermes_home / ".env").write_text("DEEPSEEK_API_KEY=ds-secret\n")
    assert init.run_init(_args(home, hermes_home)) == 0
    llm = json.loads((home / ".colonymind-llm-config.json").read_text())
    assert llm["baseUrl"] == "https://api.deepseek.com/v1" and llm["apiKey"] == "ds-secret"


@pytest.mark.parametrize("config, env", [
    # A Hermes registry provider whose endpoint is a native Messages API.
    ({"model": {"provider": "minimax", "default": "some-model"}}, "MINIMAX_API_KEY=mm-secret\n"),
    # The same with the endpoint and key in the model block: Hermes still reaches it over that transport.
    ({"model": {"provider": "anthropic", "default": "some-model", "base_url": "https://api.anthropic.com",
                "api_key": "an-secret"}}, ""),
    # A named entry that says so.
    ({"model": {"provider": "house", "default": "house-model"},
      "providers": {"house": {"base_url": "http://127.0.0.1:9/v1", "api_key": "house-secret",
                              "transport": "anthropic_messages"}}}, ""),
    # A plain endpoint Hermes reaches over that transport by its path.
    ({"model": {"provider": "custom", "default": "some-model", "base_url": "http://127.0.0.1:9/anthropic"}},
     "OPENAI_API_KEY=model-secret\n"),
], ids=["registry", "model-block", "named-entry", "plain-url"])
def test_init_refuses_an_endpoint_hermes_reaches_over_another_transport(homes, capsys, config, env):
    """The router speaks OpenAI-compatible chat completions: an endpoint Hermes reaches over another transport
    would take requests in the wrong API, so init stops and asks for --model-url instead."""
    home, hermes_home = homes
    _write_hermes(hermes_home, config)
    (hermes_home / ".env").write_text(env)
    assert init.run_init(_args(home, hermes_home)) == 1
    out = capsys.readouterr().out
    assert "anthropic_messages" in out and "--model-url" in out and "secret" not in out
    assert not (home / "colonymind.yaml").exists() and not (home / ".colonymind-llm-config.json").exists()
    assert init.run_init(_args(home, hermes_home, model_url="http://127.0.0.1:9/v1", model="some-model")) == 0
    assert json.loads((home / ".colonymind-llm-config.json").read_text())["baseUrl"] == "http://127.0.0.1:9/v1"


def test_init_takes_a_registry_provider_whose_host_also_answers_chat_completions(homes):
    home, hermes_home = homes
    _write_hermes(hermes_home, {"model": {"provider": "xai", "default": "some-model"}})
    (hermes_home / ".env").write_text("XAI_API_KEY=xai-secret\n")
    assert init.run_init(_args(home, hermes_home)) == 0
    llm = json.loads((home / ".colonymind-llm-config.json").read_text())
    assert llm["baseUrl"] == "https://api.x.ai/v1" and llm["apiKey"] == "xai-secret"


def test_init_stops_when_no_endpoint_resolves(homes, capsys):
    home, hermes_home = homes
    _write_hermes(hermes_home, {"model": {"provider": "example-cloud", "default": "some-model"}})
    (hermes_home / ".env").write_text("")
    assert init.run_init(_args(home, hermes_home)) == 1
    out = capsys.readouterr().out
    assert "example-cloud" in out and "--model-url" in out
    assert not (home / "colonymind.yaml").exists() and not (home / "identity.yaml").exists()
    # Given the endpoint, it goes ahead.
    assert init.run_init(_args(home, hermes_home, model_url="http://127.0.0.1:9/v1", model="some-model")) == 0


def test_owner_platforms_follow_the_owners_handles_with_hermes_names(homes, capsys):
    """Not the fixed whatsapp, sms, rcs, voice: the platforms of the handles the owner gave, and the CLI."""
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, owner_handle=["telegram=1001", "Signal=+15550100"])) == 0
    out = capsys.readouterr().out
    raw = yaml.safe_load((home / "colonymind.yaml").read_text())
    assert raw["credentials"]["api-key"]["owner"] == ["cli", "telegram", "signal"]
    assert "may not write the owner from" not in out
    # A re-run keeps the recorded list; an explicit list replaces it.
    assert init.run_init(_args(home, hermes_home, owner_handle=None)) == 0
    assert yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["api-key"]["owner"] == [
        "cli", "telegram", "signal"]
    assert init.run_init(_args(home, hermes_home, owner_platform=["discord"])) == 0
    assert yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["api-key"]["owner"] == ["discord"]


def test_an_owner_platform_hermes_does_not_know_is_named(homes, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, owner_handle=["carrier-pigeon=7"])) == 0
    out = capsys.readouterr().out
    assert "carrier-pigeon" in out and "Hermes has no platform named carrier-pigeon" in out
    assert yaml.safe_load((home / "colonymind.yaml").read_text())["credentials"]["api-key"]["owner"] == [
        "cli", "carrier-pigeon"]


def test_hermes_platform_names_come_from_hermes():
    names = init.hermes_platform_names(Path(HERMES_PYTHON))
    assert {"cli", "telegram", "discord", "slack", "signal", "whatsapp", "sms"} <= set(names)
    assert "rcs" not in names and "voice" not in names


# -- (d) recall mode: a consented probe of the embeddings and rerank routes only --------------------------------


class _Endpoint:
    """An OpenAI-compatible endpoint that records every request and serves embeddings and rerank."""

    def __init__(self, *, embeddings=True):
        self.requests: list[tuple[str, str, dict]] = []
        self.embeddings = embeddings
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _answer(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                endpoint.requests.append(("GET", self.path, {}))
                self._answer(404, {"error": "not here"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                endpoint.requests.append(("POST", self.path, {**body, "authorization": self.headers.get("Authorization")}))
                if self.path == "/v1/embeddings" and endpoint.embeddings:
                    return self._answer(200, {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}], "model": body["model"]})
                if self.path == "/v1/rerank":
                    return self._answer(200, {"results": [{"index": 0, "relevance_score": 0.9}]})
                self._answer(404, {"error": "not here"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.root = f"http://127.0.0.1:{self.server.server_address[1]}"

    def paths(self):
        return [(method, path) for method, path, _ in self.requests]


@pytest.fixture
def endpoint(homes):
    served = _Endpoint()
    _, hermes_home = homes
    _write_hermes(hermes_home, {**STOCK_CONFIG, "model": {"provider": "custom", "default": "chat-model",
                                                          "base_url": served.root + "/v1"}})
    yield served
    served.server.shutdown()


def _scripted(monkeypatch, answers):
    """Interactive init with scripted answers: the first key contained in a prompt answers it; else the default."""
    asked = []

    def answer(prompt):
        asked.append(prompt)
        return next((value for key, value in answers.items() if key in prompt), "")
    monkeypatch.setattr(builtins, "input", answer)
    return asked


def test_init_probes_nothing_without_consent(homes, endpoint, monkeypatch, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home)) == 0                      # non-interactive, no --embed-url auto
    assert endpoint.requests == []
    asked = _scripted(monkeypatch, {"embeddings": "n"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert any("never a chat request" in prompt for prompt in asked), asked
    assert endpoint.requests == []
    assert load_config(home, environ={}).get("router.embed_url") == ""
    assert "keyword recall only" in capsys.readouterr().out


def test_init_embed_url_auto_probes_only_the_embeddings_and_rerank_routes(homes, endpoint, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model",
                               rerank_model="rerank-model")) == 0
    assert endpoint.paths() == [("POST", "/v1/embeddings"), ("POST", "/v1/rerank")]
    assert not any("chat" in path or "completions" in path for _, path in endpoint.paths())
    embed, rerank = (body for _, _, body in endpoint.requests)
    assert embed["model"] == "embed-model" and embed["authorization"] == "Bearer model-secret"
    assert rerank["model"] == "rerank-model"
    cfg = load_config(home, environ={})
    assert cfg.get("router.embed_url") == endpoint.root + "/v1"
    assert cfg.get("router.embed_model") == "embed-model" and cfg.get("router.embed_dims") == 3
    assert cfg.get("router.rerank_url") == endpoint.root and cfg.get("router.rerank_model") == "rerank-model"
    assert cfg.get("mind.faculties.semantic_recall") is True
    assert cfg.get("environment")["COLONYMIND_EMBED_API_KEY"] == "model-secret"
    assert cfg.get("environment")["COLONYMIND_RERANKER_API_KEY"] == "model-secret"
    out = capsys.readouterr().out
    assert "embeddings answered at" in out and "3 dimensions" in out and "model-secret" not in out
    assert "no calibrated cutoff" in out


def test_init_probe_with_consent_in_interactive_init(homes, endpoint, monkeypatch):
    home, hermes_home = homes
    _scripted(monkeypatch, {"embeddings": "y", "Embedding model": "embed-model"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert endpoint.paths() == [("POST", "/v1/embeddings")]         # no rerank model given: that route is not called
    assert load_config(home, environ={}).get("router.embed_dims") == 3


def test_an_endpoint_without_embeddings_leaves_recall_keyword_only(homes, endpoint, capsys):
    home, hermes_home = homes
    endpoint.embeddings = False
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model")) == 0
    assert endpoint.paths() == [("POST", "/v1/embeddings")]
    cfg = load_config(home, environ={})
    assert cfg.get("router.embed_url") == "" and "COLONYMIND_EMBED_API_KEY" not in (cfg.get("environment") or {})
    out = capsys.readouterr().out
    assert "no embeddings at " + endpoint.root + "/v1 (HTTP 404)" in out and "keyword recall only" in out


def test_a_replaced_recall_endpoint_takes_the_key_it_was_probed_with(homes, endpoint):
    """The recorded key belongs to the endpoint it was probed at: moving the route moves or drops the key, so an
    earlier endpoint's secret never reaches a new one."""
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model",
                               rerank_model="rerank-model")) == 0
    assert load_config(home, environ={}).get("environment")["COLONYMIND_EMBED_API_KEY"] == "model-secret"

    moved = _Endpoint()
    try:
        assert init.run_init(_args(home, hermes_home, model_url=moved.root + "/v1", model_key="moved-secret",
                                   embed_url="auto")) == 0
        assert [body["authorization"] for _, _, body in moved.requests] == ["Bearer moved-secret"] * 2
        cfg = load_config(home, environ={})
        assert cfg.get("router.embed_url") == moved.root + "/v1" and cfg.get("router.rerank_url") == moved.root
        assert cfg.get("environment")["COLONYMIND_EMBED_API_KEY"] == "moved-secret"
        assert cfg.get("environment")["COLONYMIND_RERANKER_API_KEY"] == "moved-secret"
    finally:
        moved.server.shutdown()

    # An endpoint that answers with no key keeps none: the one recorded for the earlier endpoint is dropped.
    keyless = _Endpoint()
    try:
        (hermes_home / ".env").write_text("")
        assert init.run_init(_args(home, hermes_home, model_url=keyless.root + "/v1", embed_url="auto")) == 0
        assert [body["authorization"] for _, _, body in keyless.requests] == [None] * 2
        cfg = load_config(home, environ={})
        assert cfg.get("router.embed_url") == keyless.root + "/v1"
        assert "COLONYMIND_EMBED_API_KEY" not in (cfg.get("environment") or {})
        assert "COLONYMIND_RERANKER_API_KEY" not in (cfg.get("environment") or {})
    finally:
        keyless.server.shutdown()


def test_an_embed_url_given_by_hand_drops_the_key_recorded_for_another_endpoint(homes, endpoint, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model")) == 0
    capsys.readouterr()
    # The same endpoint again keeps its key.
    assert init.run_init(_args(home, hermes_home, embed_url=endpoint.root + "/v1")) == 0
    assert load_config(home, environ={}).get("environment")["COLONYMIND_EMBED_API_KEY"] == "model-secret"
    assert init.run_init(_args(home, hermes_home, embed_url="http://127.0.0.1:9/v1")) == 0
    cfg = load_config(home, environ={})
    assert cfg.get("router.embed_url") == "http://127.0.0.1:9/v1"
    assert "COLONYMIND_EMBED_API_KEY" not in (cfg.get("environment") or {})
    out = capsys.readouterr().out
    assert "environment.COLONYMIND_EMBED_API_KEY" in out and "model-secret" not in out


# -- (f) non-interactive init from the environment ---------------------------------------------------------------

_ANSWERS = {
    "Your name": "Robin", "Your messaging handles": "telegram=1001, signal=+15550100", "Agent name": "Wren",
    "Guiding values": "care, candour", "Boundaries": "never send money", "Time zone": "Europe/Paris",
    "Quiet hours": "23:00-06:30", "Autonomy level": "standard", "Platforms": "cli, telegram, signal, discord",
    "embeddings": "n",
}
_ENVIRONMENT = {
    "COLONYMIND_INIT_OWNER_NAME": "Robin", "COLONYMIND_INIT_OWNER_HANDLES": "telegram=1001, signal=+15550100",
    "COLONYMIND_INIT_AGENT_NAME": "Wren", "COLONYMIND_INIT_AGENT_VALUES": "care, candour",
    "COLONYMIND_INIT_AGENT_BOUNDARIES": "never send money", "COLONYMIND_INIT_TIMEZONE": "Europe/Paris",
    "COLONYMIND_INIT_QUIET_HOURS": "23:00-06:30", "COLONYMIND_INIT_AUTONOMY": "standard",
    "COLONYMIND_INIT_OWNER_PLATFORMS": "cli, telegram, signal, discord", "COLONYMIND_INIT_NON_INTERACTIVE": "1",
}
_UNANSWERED = dict(owner_name=None, owner_handle=None, agent_name=None, agent_values=None, agent_boundaries=None,
                   timezone=None, quiet_hours=None, autonomy=None)


def _second_homes(tmp_path):
    home, hermes_home = tmp_path / "b" / "colonymind", tmp_path / "b" / "hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(STOCK_CONFIG, sort_keys=False))
    (hermes_home / ".env").write_text("OPENAI_API_KEY=model-secret\n")
    return home, hermes_home


def _comparable(home):
    config = yaml.safe_load((home / "colonymind.yaml").read_text())
    identity = yaml.safe_load((home / "identity.yaml").read_text())
    config["owner"].pop("contact_id")
    config["hermes"].pop("home")
    identity["owner"].pop("contact_id", None)
    return config, identity


def test_non_interactive_init_from_the_environment_matches_the_interactive_answers(homes, tmp_path, monkeypatch):
    home, hermes_home = homes
    _scripted(monkeypatch, _ANSWERS)
    assert init.run_init(_args(home, hermes_home, non_interactive=False, **_UNANSWERED)) == 0
    interactive = _comparable(home)
    assert interactive[1]["agent"]["timezone"] == "Europe/Paris"
    assert interactive[0]["credentials"]["api-key"]["owner"] == ["cli", "telegram", "signal", "discord"]

    monkeypatch.setattr(builtins, "input", lambda prompt: pytest.fail(f"asked {prompt!r}"))
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    second, second_hermes = _second_homes(tmp_path)
    assert init.run_init(_args(second, second_hermes, non_interactive=False, **_UNANSWERED)) == 0
    assert _comparable(second) == interactive


def test_a_flag_wins_over_the_environment(homes, monkeypatch):
    home, hermes_home = homes
    monkeypatch.setenv("COLONYMIND_INIT_AGENT_NAME", "Wren")
    monkeypatch.setenv("COLONYMIND_INIT_PORT", "7911")
    assert init.run_init(_args(home, hermes_home, agent_name="Kestrel", port=None)) == 0
    assert yaml.safe_load((home / "identity.yaml").read_text())["agent"]["name"] == "Kestrel"
    assert load_config(home, environ={}).get("sidecar.port") == 7911
    monkeypatch.setenv("COLONYMIND_INIT_PORT", "seven")
    assert init.run_init(_args(home, hermes_home, port=None)) == 1


# -- (e) day-one history: the owner's own direct sessions, bound by the handles given to init ------------------


def _hermes_history(hermes_home):
    """A Hermes ``state.db``: the owner's direct session on the handle given to init, and four that are not."""
    with sqlite3.connect(hermes_home / "state.db") as conn:
        conn.executescript("""
            CREATE TABLE sessions(id TEXT PRIMARY KEY, source TEXT, user_id TEXT, chat_id TEXT, chat_type TEXT,
                                  model TEXT, origin_json TEXT);
            CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL);
        """)
        for sid, source, actor, chat, kind in [("owner-dm", "telegram", "1001", "1001", "dm"),
                                               ("sam-dm", "telegram", "2002", "2002", "dm"),
                                               ("owner-group", "telegram", "1001", "-100500", "group"),
                                               ("owner-cron", "cron", "1001", "1001", "dm"),
                                               ("owner-signal", "signal", "+15550100", "+15550100", "dm")]:
            origin = {"platform": source, "user_id": actor, "chat_id": chat, "chat_type": kind}
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                         (sid, source, actor, chat, kind, "a-model", json.dumps(origin)))
        for mid, sid, role, text in [(1, "owner-dm", "user", "The ferry to the island leaves at nine on Fridays."),
                                     (2, "owner-dm", "assistant", "Noted: Fridays at nine."),
                                     (3, "sam-dm", "user", "Sam keeps a private note here."),
                                     (4, "owner-group", "user", "A group message is not a direct one."),
                                     (5, "owner-cron", "user", "A scheduled prompt is not the owner."),
                                     (6, "owner-signal", "user", "A handle init was not given.")]:
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?)", (mid, sid, role, text, 1788600000.0))


def _imported(home):
    path = home / "turn-idempotency.db"
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        try:
            rows = conn.execute("SELECT contact_id, session_id, messages_json FROM turn_sources "
                                "WHERE turn_id LIKE 'hermes-history:%' ORDER BY turn_id").fetchall()
        except sqlite3.OperationalError:
            return []
    return [(contact, session, json.loads(messages)[0]["content"]) for contact, session, messages in rows]


def test_the_history_offer_lists_only_the_owners_own_direct_sessions(homes, monkeypatch, capsys):
    home, hermes_home = homes
    _hermes_history(hermes_home)
    sessions = init.owner_history_sessions(hermes_home / "state.db", [{"platform": "telegram", "id": "1001"}])
    assert [item["session_id"] for item in sessions] == ["owner-dm"]
    asked = _scripted(monkeypatch, {"Import": "y"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    offer = capsys.readouterr().out
    assert "1 direct session from your own handles (telegram 1001: 2 messages)" in offer
    assert any("quotations only" in prompt for prompt in asked)
    owner = load_config(home, environ={}).get("owner.contact_id")
    assert _imported(home) == [(owner, "owner-dm", "The ferry to the island leaves at nine on Fridays."),
                               (owner, "owner-dm", "Noted: Fridays at nine.")]
    # The snapshot init read from is gone; Hermes' own database is untouched.
    assert not list(home.glob(".history-*"))
    assert [item["session_id"] for item in init.owner_history_sessions(
        hermes_home / "state.db", [{"platform": "telegram", "id": "1001"}])] == ["owner-dm"]


def test_the_history_import_needs_a_yes_or_the_flag(homes, monkeypatch, capsys):
    home, hermes_home = homes
    _hermes_history(hermes_home)
    assert init.run_init(_args(home, hermes_home)) == 0                          # non-interactive, no flag
    assert "--import-history" in capsys.readouterr().out and _imported(home) == []
    _scripted(monkeypatch, {"Import": "n"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert _imported(home) == []
    assert init.run_init(_args(home, hermes_home, import_history=True)) == 0
    assert [session for _, session, _ in _imported(home)] == ["owner-dm", "owner-dm"]
    # Again: the same quotations, not new ones.
    assert init.run_init(_args(home, hermes_home, import_history=True)) == 0
    assert len(_imported(home)) == 2
