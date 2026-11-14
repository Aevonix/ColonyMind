"""The local embedder: recall by meaning for an instance whose model endpoint has no embeddings route.

``router.embed_provider: local`` runs a sentence-transformers model inside the sidecar (the ``colonymind[vectors]``
extra). The model comes from the tier table for the machine, pinned to one revision, and its files live in the
instance directory: the sidecar loads that directory, so the index identity names the exact weights and a start
needs no network. These tests never load a model or reach a network: the model is a stand-in module, the hardware
is a stated profile and the download is a function that writes where it is told.
"""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from colonymind import config, doctor
from colonymind.api.routers import host
from colonymind.config import ConfigError, apply_environment, load_config, save_config, write_api_key
from colonymind.doctor import FAIL, PASS, SKIP, WARN, check_semantic_recall
from colonymind.vector import local_embedder, scanner
from colonymind.vector.scanner import HardwareProfile

SMALL = "Qwen/Qwen3-Embedding-0.6B"
SMALL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
TINY = "sentence-transformers/all-MiniLM-L6-v2"
OWNER = "contact-owner"


def hardware(monkeypatch, *, gpu="none", vram=0, ram=16):
    """The machine the scanner reports; nothing is probed."""
    profile = HardwareProfile(gpu_type=gpu, gpu_name="", vram_gb=vram, ram_gb=ram)
    monkeypatch.setattr(scanner, "scan", lambda: profile)
    return profile


#: A model repository at one commit: what the sidecar loads, another runtime's copy and the card (both skipped).
FILES = {"modules.json": b"[]", "config.json": b"{}", "tokenizer.json": b"{}", "1_Pooling/config.json": b"{}",
         "model.safetensors": b"weights", "onnx/model.onnx": b"another runtime's weights", "README.md": b"the card"}
SHARDED = {**{name: data for name, data in FILES.items() if name != "model.safetensors"},
           "model.safetensors.index.json": json.dumps({"weight_map": {
               "embed": "model-00001-of-00002.safetensors", "layers": "model-00002-of-00002.safetensors"}}).encode(),
           "model-00001-of-00002.safetensors": b"first half", "model-00002-of-00002.safetensors": b"second half"}


class Hub:
    """The model's repository as the download sees it through ``huggingface_hub``: its file list with each size, and
    a snapshot that writes every file not skipped and not already there unchanged, as the library resumes one. The
    download's own body (``local_embedder._snapshot``) runs against it."""

    def __init__(self, files=None):
        self.files = dict(FILES if files is None else files)
        self.written: list[str] = []

    def snapshot(self, model, revision, directory, skipped):
        hub = self

        class HfApi:
            def model_info(self, repo_id, *, revision, files_metadata):
                assert files_metadata and repo_id == model
                return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=name, size=len(data))
                                                               for name, data in hub.files.items()])

        def snapshot_download(*, repo_id, revision, local_dir, ignore_patterns):
            for name, data in hub.files.items():
                path = Path(local_dir) / name
                if any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns):
                    continue
                if path.is_file() and path.read_bytes() == data:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                hub.written.append(name)

        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(sys.modules, "huggingface_hub",
                          SimpleNamespace(HfApi=HfApi, snapshot_download=snapshot_download))
            local_embedder._snapshot(model, revision, directory, skipped)


class Download:
    """The download ``fetch`` starts: what it was asked for, and the files a finished one leaves, written by the
    download's own body against a stand-in repository (``Hub``); nothing reaches a network."""

    def __init__(self, hub=None, *, fails=False):
        self.hub = hub or Hub()
        self.calls: list[dict] = []
        self.fails = fails

    def __call__(self, command, env):
        assert command[1] == "-c" and "_snapshot(" in command[2]
        skipped, model, revision, local_dir = command[-4:]
        self.calls.append({"model": model, "revision": revision, "local_dir": local_dir, "env": dict(env)})
        if self.fails:
            return 1
        self.hub.snapshot(model, revision, local_dir, skipped.split(","))
        # What the hub library keeps beside a download: its cache under HF_HOME.
        cache = Path(env["HF_HOME"]) / "xet" / "chunk-cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "chunk").write_bytes(b"chunk")
        return 0


def place_model(home, model=SMALL, revision=SMALL_REVISION, hub=None):
    """The files a finished download leaves: what the sidecar loads."""
    target = local_embedder.model_dir(home, model, revision)
    target.mkdir(parents=True)
    (hub or Hub()).snapshot(model, revision, str(target), local_embedder._SKIPPED)
    return target


#: What the extra's floor alone admits: sentence-transformers 3.0 declares transformers 4.34 or later, and the Qwen3
#: architecture arrived in transformers 4.51.
OLD_LIBRARIES = {"sentence-transformers": "3.0.0", "transformers": "4.34.0"}
#: The oldest releases that load every pinned model: the tests' environment, whatever this interpreter has.
LIBRARIES = {"sentence-transformers": "3.0.0", "transformers": "4.51.0"}


def releases(monkeypatch, versions):
    """The sidecar's environment reports these releases (``None``: not installed); every other distribution is as
    installed. Nothing is imported."""
    from importlib import metadata
    real = metadata.version

    def version(name):
        if name not in versions:
            return real(name)
        if versions[name] is None:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(metadata, "version", version)


def local_config(home, **router):
    save_config({**config.DEFAULTS, "router": {**config.DEFAULTS["router"], "embed_provider": "local",
                                               "embed_model": SMALL, "embed_revision": SMALL_REVISION,
                                               "embed_dims": 1024, **router}}, home)
    return load_config(home, environ={})


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(local_embedder, "extra_available", lambda: True)
    releases(monkeypatch, LIBRARIES)
    hardware(monkeypatch)
    return tmp_path


# -- the choice: the tier table's model for this machine, pinned ------------------------------------------------------

@pytest.mark.parametrize("profile, model", [
    ({"ram": 16}, SMALL),                                  # a laptop: the 8-16 GB row
    ({"gpu": "cuda", "vram": 24, "ram": 64}, SMALL),       # a larger row is not the default (the ceiling)
    ({"gpu": "mlx", "vram": 32, "ram": 32}, SMALL),
])
def test_the_default_is_the_tier_tables_model_for_the_machine_pinned_to_a_revision(monkeypatch, profile, model):
    hardware(monkeypatch, **profile)
    chosen = local_embedder.for_this_machine()
    assert chosen.model == model
    assert len(chosen.revision) == 40 and all(ch in "0123456789abcdef" for ch in chosen.revision)
    from colonymind.vector.tiers import TIERS
    widths = {tier.text_embedder.model_id: tier.text_embedder.dims for tier in TIERS if tier.text_embedder}
    assert chosen.dims == widths[model]                   # the width is the tier table's


def _chunk_tokens():
    """The most tokens one memory chunk can take: its characters at up to four UTF-8 bytes each, a token per byte
    at most, and the model's own markers (the count the query bound uses)."""
    from colonymind.turns import source_vectors
    from colonymind.vector import embedder
    return source_vectors.CHUNK * 4 // embedder.BYTES_PER_TOKEN_FLOOR + embedder.QUERY_SPECIAL_TOKENS


def test_every_pinned_model_reads_a_whole_memory_chunk():
    """A model reads only the first tokens of its input: past its window a chunk's words never reach its vector.
    The chunker keeps ``'a ' * 300 + 'Hydrofoil departures: Friday, nine sharp.'`` in one chunk, and a model that
    reads 256 tokens never sees the departure, so recall by meaning cannot find it. Every pinned model's window
    (the tier table's) holds the longest chunk the chunker cuts."""
    from colonymind.vector.tiers import TIERS
    windows = {tier.text_embedder.model_id: tier.text_embedder.context for tier in TIERS if tier.text_embedder}
    for model in local_embedder.PINNED:
        assert windows[model] >= _chunk_tokens(), model
        local_embedder.pinned(model)


def test_a_model_whose_window_cuts_a_memory_chunk_is_never_a_local_embedder(monkeypatch):
    """The tier table's smallest model reads 256 tokens; pinned or not, it is refused, and no machine gets it."""
    monkeypatch.setitem(local_embedder.PINNED, TINY, local_embedder.Pin("0" * 40, 0.09))
    with pytest.raises(ValueError, match=f"{TINY} reads 256 tokens"):
        local_embedder.pinned(TINY)
    hardware(monkeypatch, ram=2)
    with pytest.raises(local_embedder.LocalEmbedderError):
        local_embedder.for_this_machine()


@pytest.mark.parametrize("ram", [2, 4])
def test_a_machine_below_the_smallest_pinned_models_row_is_told_how_to_name_it(monkeypatch, ram):
    """Below 8 GB the tier table's models are one that cuts chunks and one that runs another repository's code: no
    model is chosen for such a machine, and the smallest pinned one is named for an owner who wants it anyway."""
    hardware(monkeypatch, ram=ram)
    with pytest.raises(local_embedder.LocalEmbedderError) as refused:
        local_embedder.for_this_machine()
    assert f"--embed-model {SMALL}" in str(refused.value) and "1.2 GB" in str(refused.value)


def test_every_pinned_model_is_in_the_tier_table_and_loads_without_remote_code():
    from colonymind.vector.tiers import TIERS
    listed = {tier.text_embedder.model_id for tier in TIERS if tier.text_embedder}
    assert set(local_embedder.PINNED) <= listed
    assert "nomic-ai/nomic-embed-text-v1.5" not in local_embedder.PINNED   # its code comes from another repository
    assert local_embedder.pinned("Qwen/Qwen3-Embedding-4B").dims == 2560
    with pytest.raises(ValueError, match=SMALL):
        local_embedder.pinned("example/unpinned-embedder")


@pytest.mark.parametrize("versions, model, unmet", [
    (OLD_LIBRARIES, SMALL, ["transformers 4.34.0 (needs 4.51.0 or later)"]),
    (OLD_LIBRARIES, "Qwen/Qwen3-Embedding-8B", ["transformers 4.34.0 (needs 4.51.0 or later)"]),
    ({"sentence-transformers": "2.2.2", "transformers": "4.51.0"}, SMALL,
     ["sentence-transformers 2.2.2 (needs 3.0 or later)"]),
    ({"sentence-transformers": "3.0.0", "transformers": None}, SMALL,
     ["transformers (not installed; needs 4.51.0 or later)"]),
    ({"sentence-transformers": "5.1.0", "transformers": "4.51.0.dev0"}, SMALL, []),
])
def test_what_keeps_the_sidecars_libraries_from_loading_a_pinned_model(monkeypatch, versions, model, unmet):
    """The extra only needs sentence-transformers to be there; a model's architecture can need newer libraries than
    the extra's floor admits, and the sidecar would then fail to load it at start and recall by words only."""
    releases(monkeypatch, versions)
    assert local_embedder.unmet(model) == unmet
    command = local_embedder.upgrade_command(model)
    assert command.startswith("uv tool install --force --python 3.12 'colonymind[vectors]'")
    for line in unmet:
        name = line.split(" ")[0]
        assert f"'{name}>=" in command


def test_the_model_directory_is_inside_the_instance_and_names_its_revision(tmp_path):
    target = local_embedder.model_dir(tmp_path, SMALL, SMALL_REVISION)
    assert target.is_relative_to(tmp_path / "models") and target.name == SMALL_REVISION
    assert not local_embedder.installed(target)
    place_model(tmp_path)
    assert local_embedder.installed(target)


def test_the_download_lists_what_the_repository_holds_but_the_skipped_files(tmp_path):
    target = place_model(tmp_path)
    manifest = json.loads((target / local_embedder.MANIFEST).read_text())
    assert (manifest["model"], manifest["revision"]) == (SMALL, SMALL_REVISION)
    kept = {name: len(data) for name, data in FILES.items() if name not in ("onnx/model.onnx", "README.md")}
    assert manifest["files"] == kept
    assert not (target / "onnx").exists() and not (target / "README.md").exists()


@pytest.mark.parametrize("files, damage, lacking", [
    (FILES, "config.json", ["config.json"]),                                  # a file beside the weights
    (FILES, "1_Pooling/config.json", ["1_Pooling/config.json"]),              # a module's own file
    (FILES, "truncate model.safetensors", ["model.safetensors"]),             # weights cut short
    (FILES, local_embedder.MANIFEST, [local_embedder.MANIFEST]),              # a download that never finished
    (SHARDED, "model-00002-of-00002.safetensors", ["model-00002-of-00002.safetensors"]),   # one shard of two
])
def test_a_model_missing_anything_its_download_listed_is_not_installed(tmp_path, files, damage, lacking):
    """Every file the repository lists at the pinned commit, at its size, and every shard the weights' index names:
    one missing file makes the copy incomplete, so doctor fails it and init fetches it."""
    target = place_model(tmp_path, hub=Hub(files))
    assert local_embedder.installed(target) and local_embedder.missing(target) == []
    if damage.startswith("truncate "):
        name = damage.split(" ", 1)[1]
        (target / name).write_bytes((target / name).read_bytes()[:-1])
    else:
        (target / damage).unlink()
    assert not local_embedder.installed(target)
    assert local_embedder.missing(target) == lacking


def test_weights_the_download_skipped_leave_the_model_incomplete(tmp_path):
    """A shard the index names but the download left out (another runtime's format) is missing too."""
    files = {**SHARDED, "model-00002-of-00002.bin": SHARDED["model-00002-of-00002.safetensors"]}
    del files["model-00002-of-00002.safetensors"]
    index = json.loads(files["model.safetensors.index.json"])
    index["weight_map"]["layers"] = "model-00002-of-00002.bin"
    files["model.safetensors.index.json"] = json.dumps(index).encode()
    target = place_model(tmp_path, hub=Hub(files))
    assert local_embedder.missing(target) == ["model-00002-of-00002.bin"]


def test_fetch_completes_an_incomplete_copy_by_fetching_only_what_it_lacks(tmp_path, monkeypatch):
    hub = Hub(SHARDED)
    target = place_model(tmp_path, hub=hub)
    (target / "config.json").unlink()
    (target / "model-00002-of-00002.safetensors").unlink()
    hub.written.clear()
    download = Download(hub)
    monkeypatch.setattr(local_embedder, "_download", download)
    assert local_embedder.fetch(tmp_path, SMALL, SMALL_REVISION) == target
    assert len(download.calls) == 1 and sorted(hub.written) == ["config.json", "model-00002-of-00002.safetensors"]
    assert local_embedder.installed(target) and not target.with_name(target.name + ".partial").exists()
    assert local_embedder.fetch(tmp_path, SMALL, SMALL_REVISION) == target and len(download.calls) == 1


STAND_IN_HUB = """
from pathlib import Path
from types import SimpleNamespace
FILES = {"modules.json": b"[]", "config.json": b"{}", "model.safetensors": b"weights", "README.md": b"card"}
class HfApi:
    def model_info(self, repo_id, *, revision, files_metadata):
        return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=name, size=len(data))
                                                       for name, data in FILES.items()])
def snapshot_download(*, repo_id, revision, local_dir, ignore_patterns):
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for name, data in FILES.items():
        if name != "README.md":
            Path(local_dir, name).write_bytes(data)
"""


def test_the_download_runs_in_an_interpreter_of_its_own(tmp_path, monkeypatch):
    """The command ``fetch`` runs, in a fresh interpreter, with a stand-in hub library ahead of the real one on its
    path: the model lands in the instance with its manifest."""
    library = tmp_path / "library" / "huggingface_hub"
    library.mkdir(parents=True)
    (library / "__init__.py").write_text(STAND_IN_HUB)
    monkeypatch.setenv("PYTHONPATH", str(library.parent))
    home = tmp_path / "instance"
    target = local_embedder.fetch(home, SMALL, SMALL_REVISION)
    assert target == local_embedder.model_dir(home, SMALL, SMALL_REVISION) and local_embedder.installed(target)
    assert json.loads((target / local_embedder.MANIFEST).read_text())["files"] == {
        "modules.json": 2, "config.json": 2, "model.safetensors": 7}


def test_a_stopped_repair_keeps_what_was_there_for_the_next_one(tmp_path, monkeypatch):
    target = place_model(tmp_path)
    (target / "config.json").unlink()
    monkeypatch.setattr(local_embedder, "_download", Download(fails=True))
    with pytest.raises(local_embedder.LocalEmbedderError, match="resume"):
        local_embedder.fetch(tmp_path, SMALL, SMALL_REVISION)
    partial = target.with_name(target.name + ".partial")
    assert (partial / "model.safetensors").is_file() and not target.exists()
    hub = Hub()
    monkeypatch.setattr(local_embedder, "_download", Download(hub))
    assert local_embedder.fetch(tmp_path, SMALL, SMALL_REVISION) == target
    assert hub.written == ["config.json"] and local_embedder.installed(target)


# -- the setting: validated, exported instead of skip -----------------------------------------------------------------

def test_the_setting_defaults_to_the_endpoint(home):
    cfg = load_config(home, environ={})
    assert cfg.get("router.embed_provider") == "endpoint" and cfg.get("router.embed_revision") == ""


def test_a_local_embedder_is_exported_instead_of_skip(home):
    environ: dict[str, str] = {}
    apply_environment(local_config(home), environ=environ)
    target = local_embedder.model_dir(home, SMALL, SMALL_REVISION)
    assert environ["COLONYMIND_EMBED_PROVIDER"] == "cpu"
    assert environ["COLONYMIND_EMBED_MODEL"] == str(target)       # the pinned directory, never a name to download
    assert environ["COLONYMIND_EMBED_REVISION"] == SMALL_REVISION
    assert environ["COLONYMIND_EMBED_DIMS"] == "1024"
    assert "COLONYMIND_EMBED_BASE_URL" not in environ
    assert "COLONYMIND_EMBED_QUERY_INSTRUCTION" not in environ   # the pipeline's own: the Qwen3 embedders' format
    # Anything the libraries would fetch or cache stays in the instance as well.
    assert Path(environ["HF_HOME"]).is_relative_to(home)


@pytest.mark.parametrize("gpu, provider", [("cuda", "cuda"), ("mlx", "mlx"), ("none", "cpu")])
def test_the_local_embedder_runs_on_the_device_the_machine_has(home, monkeypatch, gpu, provider):
    hardware(monkeypatch, gpu=gpu, vram=24 if gpu != "none" else 0)
    environ: dict[str, str] = {}
    apply_environment(local_config(home), environ=environ)
    assert environ["COLONYMIND_EMBED_PROVIDER"] == provider


def test_the_semantic_recall_arm_switches_the_local_embedder_off(home, monkeypatch):
    """The faculty's ablation arm covers the local embedder like an endpoint: off, nothing is loaded (and the
    hardware is not even scanned); the model stays exported, so a calibration stamp is the same in both arms."""
    cfg = local_config(home)
    cfg.data["mind"]["faculties"]["semantic_recall"] = False
    monkeypatch.setattr(scanner, "scan", lambda: pytest.fail("scanned with semantic recall off"))
    environ: dict[str, str] = {}
    apply_environment(cfg, environ=environ)
    assert environ["COLONYMIND_EMBED_PROVIDER"] == "skip"
    assert environ["COLONYMIND_EMBED_MODEL"] == str(local_embedder.model_dir(home, SMALL, SMALL_REVISION))


def test_the_derived_local_exports_are_reserved_or_not_the_mappings(home):
    environ: dict[str, str] = {}
    apply_environment(local_config(home), environ=environ)
    derived = set(environ) - {"HERMES_HOME", "HF_HOME", "COLONYMIND_EMBED_REVISION"}
    assert derived <= set(config.RESERVED_ENVIRONMENT), derived - set(config.RESERVED_ENVIRONMENT)
    # The pinned revision has one place: the mapping may not restate it for a local embedder.
    (home / "colonymind.yaml").write_text(yaml.safe_dump({
        "router": {"embed_provider": "local", "embed_model": SMALL, "embed_revision": SMALL_REVISION,
                   "embed_dims": 1024},
        "environment": {"COLONYMIND_EMBED_REVISION": "main"}}))
    with pytest.raises(ConfigError, match=r"router\.embed_revision"):
        load_config(home, environ={})


@pytest.mark.parametrize("router, message", [
    ({"embed_provider": "remote"}, "router.embed_provider must be endpoint or local"),
    ({"embed_provider": "local", "embed_url": "http://127.0.0.1:9101/v1", "embed_model": SMALL,
      "embed_revision": SMALL_REVISION, "embed_dims": 1024}, "router.embed_url must be empty"),
    ({"embed_provider": "local", "embed_model": SMALL, "embed_revision": "main", "embed_dims": 1024},
     "router.embed_revision"),
    ({"embed_provider": "local", "embed_model": "../../elsewhere", "embed_revision": SMALL_REVISION,
      "embed_dims": 1024}, "router.embed_model"),
    ({"embed_provider": "local", "embed_model": SMALL, "embed_revision": SMALL_REVISION}, "router.embed_dims"),
    ({"embed_provider": "endpoint", "embed_url": "http://127.0.0.1:9101/v1", "embed_revision": SMALL_REVISION},
     "router.embed_revision"),
    ({"embed_provider": "local", "embed_model": SMALL, "embed_revision": SMALL_REVISION, "embed_dims": 1024,
      "environment": {"COLONYMIND_EMBED_REQUEST_DIMS": "1024"}}, "COLONYMIND_EMBED_REQUEST_DIMS"),
])
def test_a_local_embedder_setting_is_checked_when_the_file_loads(home, router, message):
    """Including what only an endpoint reads: the width asked of an endpoint stops an in-process provider from
    starting, so a hand-written file with both is refused when it loads, not by a sidecar that recalls by words."""
    router = dict(router)
    environment = router.pop("environment", None)
    (home / "colonymind.yaml").write_text(yaml.safe_dump({"router": router,
                                                         **({"environment": environment} if environment else {})}))
    with pytest.raises(ConfigError, match=message.replace(".", r"\.")):
        load_config(home, environ={})


# -- doctor recognizes it -----------------------------------------------------------------------------------------

def _serve(monkeypatch, routes):
    calls = []

    def fake_get(url, api_key="", timeout=10.0):
        calls.append(url)
        for suffix, answer in routes.items():
            if url.endswith(suffix):
                return answer
        return 404, {"detail": "not found"}

    monkeypatch.setattr(doctor, "_http_get", fake_get)
    return calls


def test_doctor_passes_a_local_embedder_whose_pinned_files_are_there(home):
    local_config(home)
    place_model(home)
    result = doctor.check_recall_mode()
    assert result.status == PASS
    assert SMALL in result.detail and SMALL_REVISION[:12] in result.detail and "1024 dimensions" in result.detail


def test_doctor_fails_a_local_embedder_whose_files_are_missing(home):
    local_config(home)
    result = doctor.check_recall_mode()
    assert result.status == FAIL
    assert str(local_embedder.model_dir(home, SMALL, SMALL_REVISION)) in result.detail
    assert "re-run 'colonymind init'" in result.remedy


def test_doctor_fails_a_local_embedder_missing_one_file_and_names_it(home):
    local_config(home)
    target = place_model(home)
    (target / "config.json").unlink()
    result = doctor.check_recall_mode()
    assert result.status == FAIL and "incomplete (config.json missing)" in result.detail
    assert "re-run 'colonymind init'" in result.remedy and "missing" in result.remedy


def test_doctor_fails_a_local_embedder_without_the_extra(home, monkeypatch):
    local_config(home)
    place_model(home)
    monkeypatch.setattr(local_embedder, "extra_available", lambda: False)
    result = doctor.check_recall_mode()
    assert result.status == FAIL and "colonymind[vectors]" in result.detail
    assert "colonymind[vectors]" in result.remedy


def test_doctor_fails_a_local_embedder_its_libraries_cannot_load(home, monkeypatch):
    local_config(home)
    place_model(home)
    releases(monkeypatch, OLD_LIBRARIES)
    result = doctor.check_recall_mode()
    assert result.status == FAIL and "transformers 4.34.0 (needs 4.51.0 or later)" in result.detail
    assert SMALL in result.detail and "'transformers>=4.51.0'" in result.remedy


def test_doctor_warns_when_the_switch_is_off_with_a_local_embedder(home):
    cfg = local_config(home)
    cfg.data["mind"]["faculties"]["semantic_recall"] = False
    save_config(cfg.data, home)
    result = doctor.check_recall_mode()
    assert result.status == WARN and result.detail.startswith("keyword recall only")
    assert "router.embed_provider is local" in result.detail


def test_doctor_asks_the_sidecar_whether_the_local_embedder_serves(home, monkeypatch):
    local_config(home)
    _serve(monkeypatch, {"/v1/host/embed/health": (200, {"status": "ok", "model": "local", "dims": 1024,
                                                        "latency_ms": 12.0})})
    assert check_semantic_recall("http://127.0.0.1:7777", "key", 5).status == PASS
    _serve(monkeypatch, {"/v1/host/embed/health": (200, {"status": "error",
                                                        "error": "the embedder did not initialise"})})
    result = check_semantic_recall("http://127.0.0.1:7777", "key", 5)
    assert result.status == FAIL and "re-run 'colonymind init'" in result.remedy
    (home / "colonymind.yaml").write_text(yaml.safe_dump({"router": {"embed_url": ""}}))
    assert check_semantic_recall("http://127.0.0.1:7777", "key", 5).status == SKIP


def test_the_keyword_only_warning_names_the_local_embedder(home):
    result = doctor.check_recall_mode()
    assert result.status == WARN and "--embed-url local" in result.remedy


# -- recall runs hybrid ---------------------------------------------------------------------------------------------

class StandInModel:
    """A sentence-transformers model that loads nothing: boats and hydrofoils share one direction."""

    loaded: list[tuple[str, dict]] = []

    def __init__(self, model_name_or_path, **kwargs):
        StandInModel.loaded.append((model_name_or_path, kwargs))

    def encode(self, texts, **_):
        rows = np.zeros((len(texts), 1024), dtype=np.float32)
        for row, text in zip(rows, texts):
            row[0 if any(word in text.casefold() for word in ("hydrofoil", "boat")) else 1] = 1.0
        return rows


ANSWER = "Hydrofoil departures: Friday, nine sharp."
QUESTION = "when is my boat leaving"


def _search(client, query):
    response = client.post("/v1/host/memory/search", headers={"Authorization": "Bearer file-key"}, json={
        "identity": {"host_id": "fixture"}, "person_id": OWNER, "session_id": "session-b", "query": query})
    assert response.status_code == 200, response.text
    return response.json()


def _recall(home, monkeypatch, environ):
    """Start the sidecar with *environ* (what the instance config exports), record the answer, index it the way the
    running sidecar does, and ask the question that shares no word with it."""
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    for name in ("_contacts_store", "_telemetry", "_context_recall_selector", "_comms_log", "_embedder",
                 "_reranker"):
        monkeypatch.setattr(host, name, None)
    from colonymind import vector
    from colonymind.server import create_app
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.turns.source_vectors import SourceVectors
    with TestClient(create_app(), base_url="http://127.0.0.1:7777") as client:
        ledger = get_turn_idempotency_ledger(home)
        ledger.record_source("answer", contact_id=OWNER, session_id="session-a",
                             messages=[{"role": "user", "content": ANSWER}])
        store, pipeline = vector.get_store(), vector.get_pipeline()
        if store is not None and pipeline is not None:
            async def index():
                projection = SourceVectors(ledger, store, pipeline)
                projection.backfill()
                while await projection.process_one():
                    pass
            client.portal.call(index)
        return _search(client, QUESTION)


def test_recall_runs_hybrid_with_the_local_embedder(home, monkeypatch):
    """What the setting exports starts an in-process embedder on the instance's pinned files, and recall uses it:
    a question that shares no word with the answer finds it by meaning, which keyword recall cannot."""
    write_api_key("file-key", home)
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(home))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=StandInModel))
    StandInModel.loaded.clear()
    target = place_model(home)
    environ: dict[str, str] = {}
    cfg = local_config(home)
    cfg.data["owner"]["contact_id"] = OWNER
    apply_environment(cfg, environ=environ)

    body = _recall(home, monkeypatch, environ)
    assert StandInModel.loaded and StandInModel.loaded[0][0] == str(target)
    assert StandInModel.loaded[0][1].get("device") == "cpu"
    assert body["retrieval"]["semantic"] == "ready"
    assert "nine sharp" in body["content"]


def test_without_the_local_embedder_the_same_question_finds_nothing(home, monkeypatch):
    """The control: keyword recall alone (the default without an embeddings route) cannot answer it."""
    write_api_key("file-key", home)
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(home))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    save_config({**config.DEFAULTS, "owner": {"contact_id": OWNER}}, home)
    environ: dict[str, str] = {}
    apply_environment(load_config(home, environ={}), environ=environ)
    assert environ["COLONYMIND_EMBED_PROVIDER"] == "skip"
    body = _recall(home, monkeypatch, environ)
    assert body["retrieval"]["semantic"] != "ready" and "nine sharp" not in body["content"]
