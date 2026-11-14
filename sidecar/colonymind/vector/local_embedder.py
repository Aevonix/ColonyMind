"""The local embedder: a sentence-transformers model the sidecar runs itself (``router.embed_provider: local``).

It gives recall by meaning to an instance whose model endpoint has no embeddings route. The model comes from the
tier table (``vector.tiers``) for the machine's memory, no larger than ``CEILING``, and is pinned to one revision
(``PINNED``); every pinned model reads a whole memory chunk (``chunk_tokens``). Its files are downloaded once, by ``colonymind init``, into the instance directory
(``model_dir``), and the sidecar loads that directory: the index identity and any calibration stamp name the exact
weights, a start needs no network, and nothing lands outside the instance. The in-process providers are the ones
``vector.embedder`` already has, installed by the ``colonymind[vectors]`` extra.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

EXTRA = "colonymind[vectors]"
INSTALL = ("uv tool install --force --python 3.12 'colonymind[vectors]' "
           "(or pipx install --force 'colonymind[vectors]')")
#: The instance's model directory; downloads and the libraries' caches stay under it.
MODELS = "models"

#: The largest tier-table model chosen without being named. Rows above it assume memory set aside for retrieval,
#: while this model shares the machine with the agent and its model: the 4B and 8B models hold about 8 and 15 GB
#: (twice that where the libraries load full precision), this one about 1.2 GB. A larger pinned model is chosen by
#: name (``colonymind init --embed-url local --embed-model <model>``). The ceiling is a maintainer's choice, not a
#: measurement.
CEILING = "Qwen/Qwen3-Embedding-0.6B"


#: The oldest sentence-transformers the local embedder runs on: the ``colonymind[vectors]`` extra's own floor.
LIBRARY_FLOOR: tuple[tuple[str, str], ...] = (("sentence-transformers", "3.0"),)


@dataclass(frozen=True)
class Pin:
    """One commit of a model's repository, the size of what the download keeps (GB), and the libraries its
    architecture needs beyond ``LIBRARY_FLOOR``, as (distribution, oldest release)."""

    revision: str
    download_gb: float
    requires: tuple[tuple[str, str], ...] = ()


#: The Qwen3 architecture arrived in transformers 4.51.0 (an earlier release cannot build the model: the models'
#: cards). The ``colonymind[vectors]`` extra declares it since 1.12; an environment installed before that can still
#: hold an older transformers, which init and doctor name (``unmet``).
_QWEN3 = (("transformers", "4.51.0"),)
#: The tier-table text embedders the in-process providers load as they are (no code from the model's repository),
#: each pinned to one commit, whose input window holds a whole memory chunk (``chunk_tokens``): the tier table's
#: smallest model reads 256 tokens, and the rest of a longer chunk would never reach its vector. A new pin changes
#: the weights, so an instance keeps the revision it recorded until init records another. The pipeline's default
#: query instruction is the Qwen3 embedders' own format.
PINNED: dict[str, Pin] = {
    "Qwen/Qwen3-Embedding-0.6B": Pin("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", 1.2, requires=_QWEN3),
    "Qwen/Qwen3-Embedding-4B": Pin("5cf2132abc99cad020ac570b19d031efec650f2b", 8.1, requires=_QWEN3),
    "Qwen/Qwen3-Embedding-8B": Pin("1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", 15.2, requires=_QWEN3),
}

#: What a download leaves out: other runtimes' copies of the weights, and code, which is never run.
_SKIPPED = ["onnx/*", "openvino/*", "*.bin", "*.h5", "*.ot", "*.msgpack", "*.onnx", "*.py", "*.md"]

#: Cache locations a user's environment may point outside the instance; a download runs without them.
_OUTSIDE_CACHES = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "HF_MODULES_CACHE",
                   "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME")

#: What a finished download holds: every file the repository lists at the pinned commit but the skipped ones, with
#: its size. The download writes it last (``_snapshot``), and the model counts as installed only while every file it
#: lists is there at that size (``missing``).
MANIFEST = ".colonymind-files.json"

#: The download, run as ``python -c _FETCH <skipped patterns> <model> <revision> <directory>``.
_FETCH = (
    "import sys\n"
    "from colonymind.vector.local_embedder import _snapshot\n"
    "_snapshot(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[1].split(','))\n"
)


class LocalEmbedderError(RuntimeError):
    """The model could not be put in place."""


@dataclass(frozen=True)
class Choice:
    """One pinned model: its repository, the commit, its width (the tier table's) and its size."""

    model: str
    revision: str
    dims: int
    params: str
    download_gb: float


def _spec(model: str):
    from colonymind.vector.tiers import TIERS
    for tier in TIERS:
        if tier.text_embedder is not None and tier.text_embedder.model_id == model:
            return tier.text_embedder
    return None


def chunk_tokens() -> int:
    """The most tokens one memory chunk (``turns.source_vectors``) can take, counted as the query bound counts a
    query (``vector.embedder``): its characters at up to four UTF-8 bytes each, at most one token per
    ``BYTES_PER_TOKEN_FLOOR`` bytes, and the model's own markers."""
    from colonymind.turns.source_vectors import CHUNK
    from colonymind.vector.embedder import BYTES_PER_TOKEN_FLOOR, QUERY_SPECIAL_TOKENS
    return -(-CHUNK * 4 // BYTES_PER_TOKEN_FLOOR) + QUERY_SPECIAL_TOKENS


def _refusal(model: str) -> str | None:
    """Why *model* is not a local embedder (``None``: it is one)."""
    spec = _spec(model)
    if model not in PINNED or spec is None:
        return f"{model} is not a pinned local embedder; the pinned ones are {', '.join(PINNED)}"
    if spec.context < chunk_tokens():
        return (f"{model} reads {spec.context} tokens, and a memory chunk can take {chunk_tokens()}: the rest of a "
                "longer chunk would never reach its vector")
    return None


def pinned(model: str) -> Choice:
    """The pinned model *model*; ``ValueError`` saying why for any other."""
    reason = _refusal(model)
    if reason:
        raise ValueError(reason)
    spec, pin = _spec(model), PINNED[model]
    return Choice(model, pin.revision, spec.dims, spec.params, pin.download_gb)


def choose(vram_gb: int, ram_gb: int) -> Choice:
    """The tier table's text embedder for this much memory, at most ``CEILING``'s row, else the nearest row below
    whose model is a local embedder. Below the smallest such row nothing is chosen, and the refusal names that
    model for an owner who wants it anyway (the rows' memory ranges are the tier table's, not a measurement)."""
    from colonymind.vector.tiers import TIERS, get_tier_by_memory
    ceiling = next(index for index, tier in enumerate(TIERS)
                   if tier.text_embedder is not None and tier.text_embedder.model_id == CEILING)
    index = min(TIERS.index(get_tier_by_memory(vram_gb, ram_gb)), ceiling)
    for tier in reversed(TIERS[:index + 1]):
        if tier.text_embedder is not None and _refusal(tier.text_embedder.model_id) is None:
            return pinned(tier.text_embedder.model_id)
    smallest = min((pinned(model) for model in PINNED if _refusal(model) is None), key=lambda c: c.download_gb)
    raise LocalEmbedderError(f"no local embedder is chosen for a machine with {ram_gb} GB of memory; the smallest, "
                             f"{smallest.model} (about {smallest.download_gb:g} GB, twice that at full precision), "
                             f"runs when named: 'colonymind init --embed-url local --embed-model {smallest.model}'")


def for_this_machine() -> Choice:
    from colonymind.vector import scanner
    hardware = scanner.scan()
    return choose(hardware.vram_gb, hardware.ram_gb)


def device() -> str:
    """The in-process provider for this machine, by the same scan the sidecar's tier detection uses: ``cuda``,
    ``mlx`` (Apple silicon) or ``cpu``. All three load the model with sentence-transformers and normalize its
    vectors, so the index identity is the same on each."""
    from colonymind.vector import scanner
    kind = scanner.scan().gpu_type
    return kind if kind in ("cuda", "mlx") else "cpu"


def extra_available() -> bool:
    """``colonymind[vectors]`` is installed in this interpreter (the sidecar's)."""
    try:
        return find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return False


def _release(text: str) -> tuple[int, ...]:
    """A release's leading numbers (``4.51.0.dev0`` is 4.51.0)."""
    found = re.match(r"\d+(?:\.\d+)*", text.strip())
    return tuple(int(part) for part in found.group(0).split(".")) if found else ()


def _needs(model: str) -> list[tuple[str, str, str | None]]:
    """What the sidecar's environment lacks for *model*: (distribution, oldest release, installed release or
    ``None``) for each library missing or older than the model needs."""
    pin = PINNED.get(model)
    lacking = []
    for name, oldest in LIBRARY_FLOOR + (pin.requires if pin is not None else ()):
        try:
            installed_release = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed_release = None
        if installed_release is None or _release(installed_release) < _release(oldest):
            lacking.append((name, oldest, installed_release))
    return lacking


def unmet(model: str) -> list[str]:
    """Why this interpreter (the sidecar's) cannot load *model*, one line per library, such as ``transformers 4.34.0
    (needs 4.51.0 or later)``; empty when it can. The extra's presence (``extra_available``) says nothing of
    this: an environment the extra was installed into before it declared a model's needs can hold releases the
    model's architecture postdates, and the sidecar would fail to load the model at every start and recall by
    words only."""
    return [f"{name} {have} (needs {oldest} or later)" if have is not None
            else f"{name} (not installed; needs {oldest} or later)" for name, oldest, have in _needs(model)]


def upgrade_command(model: str) -> str:
    """The install that gives the sidecar's environment what *model* needs."""
    wanted = [f"'{name}>={oldest}'" for name, oldest, _ in _needs(model)]
    uv = "uv tool install --force --python 3.12 'colonymind[vectors]'" + "".join(f" --with {w}" for w in wanted)
    pipx = "pipx install --force 'colonymind[vectors]'" + (f", then pipx inject colonymind {' '.join(wanted)}"
                                                           if wanted else "")
    return f"{uv} (or {pipx})"


def models_dir(home: str | os.PathLike) -> Path:
    return Path(home) / MODELS


def cache_dir(home: str | os.PathLike) -> Path:
    """``HF_HOME`` for the download and the sidecar: whatever the libraries cache stays in the instance."""
    return models_dir(home) / "cache"


def model_dir(home: str | os.PathLike, model: str, revision: str) -> Path:
    """Where *model* at *revision* lives in the instance: the directory the sidecar loads."""
    return models_dir(home) / "embeddings" / model.replace("/", "--") / revision


def _skipped(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def missing(path: Path) -> list[str]:
    """What the model directory *path* lacks: every file its manifest lists, at the listed size, the module list,
    and every weight shard the weights' index names. Empty when the download is complete; the manifest alone when
    no download finished there."""
    try:
        listed = json.loads((path / MANIFEST).read_text(encoding="utf-8"))["files"]
        if not isinstance(listed, dict) or not all(isinstance(name, str) and _relative(name) for name in listed):
            raise ValueError("not a file list")
    except (OSError, ValueError, KeyError, TypeError):
        return [MANIFEST]
    absent = [name for name, size in sorted(listed.items()) if not _present(path / name, size)]
    required = {"modules.json"}
    if "model.safetensors.index.json" in listed:
        try:
            shards = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
            required |= {str(shard) for shard in shards.values()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            required.add("model.safetensors.index.json")
    else:
        required.add("model.safetensors")
    absent += sorted(name for name in required - set(listed) if name not in absent)
    return absent


def _relative(name: str) -> bool:
    parts = Path(name).parts
    return bool(parts) and not Path(name).is_absolute() and ".." not in parts


def _present(path: Path, size) -> bool:
    try:
        return path.is_file() and (size is None or path.stat().st_size == int(size))
    except (OSError, TypeError, ValueError):
        return False


def installed(path: Path) -> bool:
    """A finished, complete download (``missing`` finds nothing)."""
    return not missing(path)


def environment(home: str | os.PathLike, model: str, revision: str) -> dict[str, str]:
    """What the sidecar reads for the local embedder, beside its provider and width: the pinned directory, its
    commit and the libraries' cache in the instance."""
    return {"COLONYMIND_EMBED_MODEL": str(model_dir(home, model, revision)),
            "COLONYMIND_EMBED_REVISION": revision,
            "HF_HOME": str(cache_dir(home))}


def _snapshot(model: str, revision: str, directory: str, skipped: list[str]) -> None:
    """The download itself, in the interpreter ``fetch`` starts: every file of *model* at *revision* but the
    *skipped* ones into *directory* (the library keeps a file already there unchanged, so a resumed or repairing
    download fetches only what is missing), then the manifest of what the repository lists, written last."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(model, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise SystemExit(f"{model} at {revision} resolved to commit {info.sha}")
    listed = {sibling.rfilename: sibling.size for sibling in info.siblings or ()
              if not _skipped(sibling.rfilename, skipped)}
    snapshot_download(repo_id=model, revision=revision, local_dir=directory, ignore_patterns=skipped)
    manifest = {"model": model, "revision": revision, "files": listed}
    (Path(directory) / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _download(command: list[str], env: dict[str, str]) -> int:
    """Run the download in its own interpreter, so the cache locations in *env* hold from its first import; its
    progress goes to the terminal."""
    return subprocess.run(command, env=env, check=False).returncode


def fetch(home: str | os.PathLike, model: str, revision: str) -> Path:
    """Put *model* at *revision* in the instance, or complete it. The download runs in a ``.partial`` directory,
    which a stopped download leaves for the next to resume; a copy missing files moves there first, so only what it
    lacks is fetched. The directory is moved into place only once nothing is missing."""
    target = model_dir(home, model, revision)
    if installed(target):
        return target
    partial = target.with_name(target.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() and not partial.exists():
        os.replace(target, partial)
    env = {name: value for name, value in os.environ.items() if name not in _OUTSIDE_CACHES}
    env.update(HF_HOME=str(cache_dir(home)), HF_HUB_DISABLE_TELEMETRY="1")
    code = _download([sys.executable, "-c", _FETCH, ",".join(_SKIPPED), model, revision, str(partial)], env)
    if code != 0:
        raise LocalEmbedderError(f"the download of {model} at revision {revision[:12]} stopped (exit {code}); "
                                 "run the same command again to resume it")
    lacking = missing(partial)
    if lacking:
        raise LocalEmbedderError(f"the download of {model} left {partial} without {', '.join(lacking[:5])}"
                                 + (f" and {len(lacking) - 5} more" if len(lacking) > 5 else ""))
    if target.exists():
        shutil.rmtree(target)    # an earlier copy missing files (a stopped repair left both): the complete one wins
    os.replace(partial, target)
    return target
