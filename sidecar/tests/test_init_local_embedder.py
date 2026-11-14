"""``colonymind init`` offers the local embedder when the model endpoint has no embeddings route.

Runs against a temporary Hermes home with stock Hermes, like ``test_init_honest.py``, whose endpoint it shares. The
download is a stand-in that writes the model's files where it is told (and, like the real libraries, into the cache
its environment names), so a test sees every file a download would leave. No model is loaded and no network reached.
"""
from __future__ import annotations

import builtins
from pathlib import Path

import pytest
import yaml

from colonymind import init
from colonymind.config import apply_environment, load_config
from colonymind.vector import local_embedder, scanner
from colonymind.vector.scanner import HardwareProfile

from test_init import _args, homes  # noqa: F401  (pytest fixture)
from test_init_honest import _scripted, endpoint  # noqa: F401  (pytest fixture)
from test_local_embedder import LIBRARIES, OLD_LIBRARIES, Download, releases

SMALL = "Qwen/Qwen3-Embedding-0.6B"
SMALL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROBE = {"embeddings route": "y", "Embedding model": "embed-model", "Rerank model": ""}


@pytest.fixture
def machine(monkeypatch):
    """A 16 GB laptop without a GPU, with the extra installed at releases that load every pinned model, and a
    download that writes where it is told."""
    monkeypatch.setattr(scanner, "scan", lambda: HardwareProfile(gpu_type="none", gpu_name="", vram_gb=0, ram_gb=16))
    monkeypatch.setattr(local_embedder, "extra_available", lambda: True)
    releases(monkeypatch, LIBRARIES)
    download = Download()
    monkeypatch.setattr(local_embedder, "_download", download)
    return download


def _router(home):
    return load_config(home, environ={}).data["router"]


def test_init_offers_the_local_embedder_when_the_endpoint_has_no_embeddings_route(homes, endpoint, machine,
                                                                                 monkeypatch, capsys):
    home, hermes_home = homes
    endpoint.embeddings = False
    asked = _scripted(monkeypatch, {**PROBE, "in the sidecar": "y"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    offer = [prompt for prompt in asked if "in the sidecar" in prompt]
    assert len(offer) == 1 and SMALL in offer[0] and "1.2 GB" in offer[0] and str(home) in offer[0]
    router = _router(home)
    assert router["embed_provider"] == "local" and router["embed_url"] == ""
    assert router["embed_model"] == SMALL and router["embed_revision"] == SMALL_REVISION
    assert router["embed_dims"] == 1024                           # the width, recorded like an endpoint's
    assert load_config(home, environ={}).get("mind.faculties.semantic_recall") is True
    [call] = machine.calls
    assert (call["model"], call["revision"]) == (SMALL, SMALL_REVISION)
    assert local_embedder.installed(local_embedder.model_dir(home, SMALL, SMALL_REVISION))
    out = capsys.readouterr().out
    assert "semantic recall on (local embedder " + SMALL in out
    assert endpoint.paths() == [("POST", "/v1/embeddings")]       # the probe; the offer asks the endpoint nothing


def test_a_declined_offer_leaves_recall_keyword_only_and_downloads_nothing(homes, endpoint, machine, monkeypatch,
                                                                          capsys):
    home, hermes_home = homes
    endpoint.embeddings = False
    _scripted(monkeypatch, {**PROBE, "in the sidecar": "n"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    router = _router(home)
    assert router["embed_provider"] == "endpoint" and router["embed_url"] == "" and router["embed_revision"] == ""
    assert machine.calls == [] and not (home / "models").exists()
    out = capsys.readouterr().out
    assert "keyword recall only" in out and "--embed-url local" in out


def test_declining_the_probe_still_offers_the_local_embedder(homes, endpoint, machine, monkeypatch):
    """No endpoint was asked: the local embedder needs none, so it is offered all the same."""
    home, hermes_home = homes
    asked = _scripted(monkeypatch, {"embeddings route": "n", "in the sidecar": "y"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert endpoint.requests == []
    assert any("in the sidecar" in prompt for prompt in asked)
    assert _router(home)["embed_provider"] == "local"


def test_non_interactive_auto_takes_the_local_embedder_when_the_endpoint_has_none(homes, endpoint, machine, capsys):
    home, hermes_home = homes
    endpoint.embeddings = False
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model")) == 0
    router = _router(home)
    assert router["embed_provider"] == "local" and router["embed_model"] == SMALL
    assert "COLONYMIND_EMBED_API_KEY" not in (load_config(home, environ={}).get("environment") or {})
    out = capsys.readouterr().out
    assert "no embeddings at " + endpoint.root + "/v1 (HTTP 404)" in out and "local embedder " + SMALL in out


def test_a_small_machine_gets_no_offer_and_can_name_the_smallest_pinned_model(homes, endpoint, machine,
                                                                              monkeypatch, capsys):
    home, hermes_home = homes
    endpoint.embeddings = False
    monkeypatch.setattr(scanner, "scan", lambda: HardwareProfile(gpu_type="none", gpu_name="", vram_gb=0, ram_gb=4))
    asked = _scripted(monkeypatch, {**PROBE, "in the sidecar": "y"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert not any("in the sidecar" in prompt for prompt in asked) and machine.calls == []
    out = capsys.readouterr().out
    assert f"--embed-model {SMALL}" in out and "keyword recall only" in out
    assert init.run_init(_args(home, hermes_home, embed_url="local", embed_model=SMALL)) == 0
    assert (_router(home)["embed_provider"], _router(home)["embed_model"]) == ("local", SMALL)


def test_no_offer_without_the_extra_and_the_way_to_add_it_is_named(homes, endpoint, machine, monkeypatch, capsys):
    home, hermes_home = homes
    endpoint.embeddings = False
    monkeypatch.setattr(local_embedder, "extra_available", lambda: False)
    asked = _scripted(monkeypatch, PROBE)
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert not any("in the sidecar" in prompt for prompt in asked)
    assert _router(home)["embed_provider"] == "endpoint" and machine.calls == []
    out = capsys.readouterr().out
    assert "colonymind[vectors]" in out and "--embed-url local" in out and "keyword recall only" in out


def test_an_endpoint_with_embeddings_is_used_and_nothing_is_offered(homes, endpoint, machine, monkeypatch):
    home, hermes_home = homes
    asked = _scripted(monkeypatch, {**PROBE, "in the sidecar": "y"})
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert not any("in the sidecar" in prompt for prompt in asked)
    router = _router(home)
    assert router["embed_provider"] == "endpoint" and router["embed_url"] == endpoint.root + "/v1"
    assert machine.calls == []


def test_embed_url_local_asks_the_endpoint_nothing(homes, endpoint, machine, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 0
    assert endpoint.requests == []
    router = _router(home)
    assert (router["embed_provider"], router["embed_model"], router["embed_dims"]) == ("local", SMALL, 1024)
    environ: dict[str, str] = {}
    apply_environment(load_config(home, environ={}), environ=environ)
    assert environ["COLONYMIND_EMBED_PROVIDER"] == "cpu"
    assert environ["COLONYMIND_EMBED_MODEL"] == str(local_embedder.model_dir(home, SMALL, SMALL_REVISION))


def test_embed_url_local_takes_another_pinned_model_and_refuses_an_unpinned_one(homes, endpoint, machine, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="local", embed_model="Qwen/Qwen3-Embedding-4B")) == 0
    router = _router(home)
    assert router["embed_model"] == "Qwen/Qwen3-Embedding-4B" and router["embed_dims"] == 2560
    assert router["embed_revision"] == local_embedder.pinned("Qwen/Qwen3-Embedding-4B").revision
    capsys.readouterr()
    assert init.run_init(_args(home, hermes_home, embed_url="local", embed_model="example/unpinned")) == 1
    out = capsys.readouterr().out
    assert "example/unpinned" in out and SMALL in out
    assert _router(home)["embed_model"] == "Qwen/Qwen3-Embedding-4B"          # nothing written


def test_embed_url_local_without_the_extra_stops_and_says_how_to_add_it(homes, endpoint, machine, monkeypatch,
                                                                        capsys):
    home, hermes_home = homes
    monkeypatch.setattr(local_embedder, "extra_available", lambda: False)
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 1
    out = capsys.readouterr().out
    assert "colonymind[vectors]" in out and "uv tool install" in out
    assert machine.calls == [] and not (home / "colonymind.yaml").exists()


def test_embed_url_local_with_libraries_too_old_for_the_model_stops_before_anything_downloads(homes, endpoint,
                                                                                            machine, monkeypatch,
                                                                                            capsys):
    """sentence-transformers 3.0 admits transformers 4.34, which cannot build a Qwen3 model: the sidecar would fail
    to load it at every start. init names what to upgrade instead of downloading 1.2 GB it cannot use."""
    home, hermes_home = homes
    releases(monkeypatch, OLD_LIBRARIES)
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 1
    out = capsys.readouterr().out
    assert "transformers 4.34.0 (needs 4.51.0 or later)" in out and "'transformers>=4.51.0'" in out
    assert machine.calls == [] and not (home / "models").exists() and not (home / "colonymind.yaml").exists()


def test_no_offer_with_libraries_too_old_for_the_model_and_the_upgrade_is_named(homes, endpoint, machine,
                                                                               monkeypatch, capsys):
    home, hermes_home = homes
    endpoint.embeddings = False
    releases(monkeypatch, OLD_LIBRARIES)
    asked = _scripted(monkeypatch, PROBE)
    assert init.run_init(_args(home, hermes_home, non_interactive=False)) == 0
    assert not any("in the sidecar" in prompt for prompt in asked)
    assert _router(home)["embed_provider"] == "endpoint" and machine.calls == []
    out = capsys.readouterr().out
    assert "transformers 4.34.0 (needs 4.51.0 or later)" in out and "'transformers>=4.51.0'" in out
    assert "keyword recall only" in out


def test_a_download_that_fails_stops_an_explicit_request_and_leaves_an_offer_keyword_only(homes, endpoint,
                                                                                         monkeypatch, capsys):
    home, hermes_home = homes
    monkeypatch.setattr(scanner, "scan", lambda: HardwareProfile(gpu_type="none", gpu_name="", vram_gb=0, ram_gb=16))
    monkeypatch.setattr(local_embedder, "extra_available", lambda: True)
    releases(monkeypatch, LIBRARIES)
    monkeypatch.setattr(local_embedder, "_download", Download(fails=True))
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 1
    assert "download" in capsys.readouterr().out
    endpoint.embeddings = False
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model")) == 0
    router = _router(home)
    assert router["embed_provider"] == "endpoint" and router["embed_url"] == ""
    assert not local_embedder.installed(local_embedder.model_dir(home, SMALL, SMALL_REVISION))
    assert "keyword recall only" in capsys.readouterr().out


def test_a_rerun_keeps_the_local_embedder_and_fetches_only_what_is_missing(homes, endpoint, machine, monkeypatch):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 0
    assert len(machine.calls) == 1
    monkeypatch.setattr(builtins, "input", lambda prompt: pytest.fail(f"asked {prompt!r}"))
    assert init.run_init(_args(home, hermes_home)) == 0                      # files there: nothing fetched
    assert len(machine.calls) == 1 and _router(home)["embed_provider"] == "local"
    target = local_embedder.model_dir(home, SMALL, SMALL_REVISION)
    (target / "model.safetensors").unlink()
    assert init.run_init(_args(home, hermes_home)) == 0                      # the weights went: fetched again
    assert len(machine.calls) == 2 and local_embedder.installed(target)
    (target / "config.json").unlink()                                       # a file beside them went
    machine.hub.written.clear()
    assert init.run_init(_args(home, hermes_home)) == 0
    assert len(machine.calls) == 3 and machine.hub.written == ["config.json"] and local_embedder.installed(target)
    assert _router(home)["embed_model"] == SMALL and endpoint.requests == []


def test_a_probe_that_finds_nothing_keeps_the_local_embedder_recorded(homes, endpoint, machine):
    home, hermes_home = homes
    endpoint.embeddings = False
    assert init.run_init(_args(home, hermes_home, embed_url="local", embed_model="Qwen/Qwen3-Embedding-4B")) == 0
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model")) == 0
    assert _router(home)["embed_model"] == "Qwen/Qwen3-Embedding-4B" and len(machine.calls) == 1


def test_an_endpoint_given_later_replaces_the_local_embedder(homes, endpoint, machine, capsys):
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 0
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model")) == 0
    router = _router(home)
    assert router["embed_provider"] == "endpoint" and router["embed_url"] == endpoint.root + "/v1"
    assert router["embed_model"] == "embed-model" and router["embed_dims"] == 3 and router["embed_revision"] == ""
    assert str(home / "models") in capsys.readouterr().out     # the unused files are named, not deleted
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 0
    assert init.run_init(_args(home, hermes_home, embed_url="http://127.0.0.1:9101/v1")) == 0
    router = _router(home)
    assert router["embed_provider"] == "endpoint" and router["embed_url"] == "http://127.0.0.1:9101/v1"
    assert router["embed_model"] == "" and router["embed_dims"] == 0 and router["embed_revision"] == ""


@pytest.mark.parametrize("setting, value", [("COLONYMIND_EMBED_REVISION", "main"),
                                            ("COLONYMIND_EMBED_REQUEST_DIMS", "3"),
                                            ("COLONYMIND_EMBED_QUERY_INSTRUCTION", "Represent the question: ")])
def test_switching_an_endpoint_to_the_local_embedder_drops_what_only_the_endpoint_read(homes, endpoint, machine,
                                                                                      capsys, setting, value):
    """An endpoint's own settings under ``environment`` (its weights' revision, the width asked of it, its model's
    query instruction) describe the endpoint's model: the local embedder, a different model, drops them and says so,
    and the sidecar it configures can start."""
    from colonymind.vector.config import EmbeddingConfig
    home, hermes_home = homes
    assert init.run_init(_args(home, hermes_home, embed_url="auto", embed_model="embed-model")) == 0
    data = yaml.safe_load((home / "colonymind.yaml").read_text())
    data.setdefault("environment", {})[setting] = value
    (home / "colonymind.yaml").write_text(yaml.safe_dump(data))
    assert load_config(home, environ={}).get("router.embed_dims") == 3       # the endpoint's, with the setting
    capsys.readouterr()
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 0
    cfg = load_config(home, environ={})
    assert setting not in (cfg.get("environment") or {})
    assert f"environment.{setting}" in capsys.readouterr().out
    environ: dict[str, str] = {}
    apply_environment(cfg, environ=environ)
    assert environ.get("COLONYMIND_EMBED_REVISION") == SMALL_REVISION
    assert "COLONYMIND_EMBED_REQUEST_DIMS" not in environ and "COLONYMIND_EMBED_QUERY_INSTRUCTION" not in environ
    EmbeddingConfig(provider=environ["COLONYMIND_EMBED_PROVIDER"], model_id=environ["COLONYMIND_EMBED_MODEL"],
                    dimensions=int(environ["COLONYMIND_EMBED_DIMS"]), revision=environ["COLONYMIND_EMBED_REVISION"])


def test_non_interactive_init_from_the_environment_takes_the_local_embedder(homes, endpoint, machine, monkeypatch):
    home, hermes_home = homes
    monkeypatch.setenv("COLONYMIND_INIT_EMBED_URL", "local")
    assert init.run_init(_args(home, hermes_home)) == 0
    assert _router(home)["embed_provider"] == "local" and endpoint.requests == []


def test_uninstall_leaves_no_model_files_outside_the_instance(homes, endpoint, machine, tmp_path, monkeypatch,
                                                              capsys):
    """Every file the download and the sidecar's libraries write goes under the instance directory, even with the
    user's own cache settings in the environment; uninstall keeps the instance data and says where the model is."""
    home, hermes_home = homes
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("HF_HUB_CACHE", str(user / ".cache" / "huggingface" / "hub"))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(user / ".cache" / "torch" / "sentence_transformers"))
    assert init.run_init(_args(home, hermes_home, embed_url="local")) == 0
    [call] = machine.calls
    assert Path(call["local_dir"]).is_relative_to(home) and Path(call["env"]["HF_HOME"]).is_relative_to(home)
    assert "HF_HUB_CACHE" not in call["env"] and "SENTENCE_TRANSFORMERS_HOME" not in call["env"]
    environ: dict[str, str] = {}
    apply_environment(load_config(home, environ={}), environ=environ)
    assert Path(environ["HF_HOME"]).is_relative_to(home)
    assert Path(environ["COLONYMIND_EMBED_MODEL"]).is_relative_to(home)
    capsys.readouterr()
    assert init.run_init(_args(home, hermes_home, uninstall=True)) == 0
    assert str(home / "models") in capsys.readouterr().out
    outside = [path for path in tmp_path.rglob("*") if path.is_file()
               and not path.is_relative_to(home) and not path.is_relative_to(hermes_home)]
    assert outside == []
    assert local_embedder.installed(local_embedder.model_dir(home, SMALL, SMALL_REVISION))
    assert yaml.safe_load((home / "colonymind.yaml").read_text())["router"]["embed_provider"] == "local"
