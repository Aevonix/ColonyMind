"""The front pages' init instructions do what they say: Get started's local embedder is the one init records.

Runs ``colonymind init`` against a temporary Hermes home with stock Hermes, like ``test_init.py``, and skips where
those tests skip. No model is loaded or downloaded and no network is reached.
"""
from __future__ import annotations

import argparse
import json
import shlex

import pytest

from colonymind import init
from colonymind.config import apply_environment, load_config

from test_front_pages import README, _commands, _Parsed, _section, _text
from test_init import _args, homes  # noqa: F401  (homes: a temporary instance and Hermes home)
from test_init_honest import endpoint  # noqa: F401  (a model endpoint that records every request)


def _namespace(monkeypatch, command: str) -> argparse.Namespace:
    """What the CLI parses from *command*, as a reader types it; nothing runs."""
    from colonymind import cli
    parse = argparse.ArgumentParser.parse_args
    parsed = []

    def parse_then_stop(self, args=None, namespace=None):
        parsed.append(parse(self, args, namespace))
        raise _Parsed

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
        patch.setattr("sys.argv", shlex.split(command))
        with pytest.raises(_Parsed):
            cli.main()
    return parsed[0]


def test_the_local_embedder_get_started_shows_is_the_one_init_records(homes, endpoint, monkeypatch):
    """Get started's ``colonymind init --embed-url local`` on a machine with the ``vectors`` extra records the local
    embedder, never an embeddings endpoint named "local": the provider ``local`` with ``router.embed_url`` empty,
    the pinned model, commit and width, semantic recall on, and the sidecar handed the copy in the instance
    directory. The model endpoint is asked nothing. A finished download of the pinned files (with the list of
    files it writes last) is put in place first, so nothing is downloaded."""
    from colonymind.vector import local_embedder, scanner
    from colonymind.vector.scanner import HardwareProfile
    [command] = [command for command in _commands(_section(_text(README), "Get started"))
                 if "--embed-url local" in command]
    given = _namespace(monkeypatch, command)
    home, hermes_home = homes
    monkeypatch.setattr(scanner, "scan", lambda: HardwareProfile(gpu_type="none", gpu_name="", vram_gb=0, ram_gb=16))
    monkeypatch.setattr(local_embedder, "extra_available", lambda: True)
    choice = local_embedder.for_this_machine()
    pinned = local_embedder.model_dir(home, choice.model, choice.revision)
    pinned.mkdir(parents=True)
    files = {"modules.json": b"[]", "model.safetensors": b"weights"}
    for name, body in files.items():
        (pinned / name).write_bytes(body)
    listed = {name: len(body) for name, body in files.items()}
    (pinned / local_embedder.MANIFEST).write_text(json.dumps({"files": listed}))
    assert local_embedder.installed(pinned)                       # a finished download: nothing left to fetch
    monkeypatch.setattr(local_embedder, "fetch", lambda *_, **__: pytest.fail("init downloaded a model in place"))

    assert init.run_init(_args(home, hermes_home, embed_url=given.embed_url)) == 0

    config = load_config(home, environ={})
    router = config.data["router"]
    assert (router["embed_provider"], router["embed_url"]) == ("local", "")
    assert (router["embed_model"], router["embed_revision"], router["embed_dims"]) == (choice.model, choice.revision,
                                                                                      choice.dims)
    assert config.get("mind.faculties.semantic_recall") is True
    environ: dict[str, str] = {}
    apply_environment(config, environ=environ)
    assert environ["COLONYMIND_EMBED_PROVIDER"] == "cpu" and "COLONYMIND_EMBED_BASE_URL" not in environ
    assert environ["COLONYMIND_EMBED_MODEL"] == str(pinned)
    assert endpoint.requests == []
