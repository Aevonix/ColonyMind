"""``colonymind doctor``: install checks.

These things can be wrong with an install, and each has one remedy:

- the Hermes version is outside the supported range
- ``colonymind.yaml``, ``api.key`` or the Hermes keys ``init`` writes are missing
- the adapter version in Hermes' environment does not match the sidecar
- ``pip check`` in Hermes' environment is not clean
- the sidecar is not reachable with the key
- no user service keeps the sidecar up (a sidecar started by hand is not restarted
  after a crash or started at login), or the service unit is an earlier release's
- the plugin is not loaded (not enabled, or not installed where Hermes runs)
- semantic recall is configured (``router.embed_url``, or the local embedder) but the embedder is not serving
- the local embedder (``router.embed_provider: local``) is missing its pinned files, the ``colonymind[vectors]``
  extra, or library releases its model needs
- recall is keyword-only (no embedder, or the switch off), or has no calibrated cutoff (it never abstains):
  warnings, each with the way to change it
- the vector store library is missing from the sidecar's own environment
- Instinct: whether a fast decision provider is configured, its last day of decisions and the points by state;
  a warning for a retired ``COLONYMIND_DECISIONS_*`` variable still set (nothing reads it)
- the instance's own secret (``instinct.key``) holds no whole key (decisions and model usage key nothing with it),
  cannot be read, or is readable by other users
- the agent could approve its own asks: an owner lane whose terminal, code or file tools run on this machine (where
  the instance key can be read), the terminal (``cli``) as an owner platform on such a host, or a WhatsApp bridge
  in shared-account mode (warnings, each with its remedy; docs/MIND.md, Asks)
- no build stamp records the commit the install was built from, or the stamp says the tree had uncommitted
  changes (warnings; ``scripts/write_build_stamp.py`` writes the stamp)

Local checks read files and run Hermes' Python; the sidecar check talks HTTP
and degrades to a failure with the start command when the sidecar is down.
Every check runs defensively: an exception inside a check becomes a ``fail``
result, never a crashed run.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

#: Providers that route through LiteLLM's openai/* path; their baseUrl must end with /v1.
OPENAI_COMPAT_PROVIDERS = frozenset({"zai", "local", "custom", "lmstudio", "vllm", "openai"})


@dataclass
class CheckResult:
    """One diagnostic verdict."""

    name: str
    status: str  # "pass" | "warn" | "fail" | "skip"
    detail: str = ""
    remedy: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _run(name: str, fn: Callable[..., Any], *args: Any) -> List[CheckResult]:
    """Run one check defensively; an exception becomes a fail result."""
    try:
        result = fn(*args)
    except Exception as exc:  # noqa: BLE001 - the whole point
        return [CheckResult(name=name, status=FAIL,
                            detail=f"check crashed: {type(exc).__name__}: {exc}")]
    if isinstance(result, CheckResult):
        return [result]
    return list(result)


def _http_get(url: str, api_key: str = "", timeout: float = 10.0) -> Tuple[int, Any]:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


# ---------------------------------------------------------------------------
# Local checks
# ---------------------------------------------------------------------------

def check_config() -> CheckResult:
    """``colonymind.yaml`` exists and validates."""
    from colonymind.config import ConfigError, config_path, load_config
    path = config_path()
    if not path.is_file():
        return CheckResult("config", FAIL, detail=f"{path} is missing",
                           remedy="run 'colonymind init'")
    try:
        config = load_config(required=True)
    except ConfigError as exc:
        return CheckResult("config", FAIL, detail=str(exc), remedy=f"fix {path} and re-run 'colonymind doctor'")
    environment = config.get("environment") or {}
    detail = f"{path} valid"
    if environment:
        detail += f" (environment: {len(environment)} entr{'y' if len(environment) == 1 else 'ies'})"
    return CheckResult("config", PASS, detail=detail)


def check_api_key() -> CheckResult:
    """``api.key`` exists, has one token and is private."""
    from colonymind.config import key_file_is_private, key_path, read_api_key
    path = key_path()
    if os.environ.get("COLONYMIND_API_KEY", "").strip():
        return CheckResult("api-key", PASS, detail="COLONYMIND_API_KEY is set in the environment")
    if not path.is_file():
        return CheckResult("api-key", FAIL, detail=f"{path} is missing", remedy="run 'colonymind init'")
    if not read_api_key():
        return CheckResult("api-key", FAIL, detail=f"{path} is empty", remedy="run 'colonymind init'")
    if not key_file_is_private(path):
        return CheckResult("api-key", WARN, detail=f"{path} is readable by other users",
                           remedy=f"chmod 600 {path}")
    return CheckResult("api-key", PASS, detail=f"{path} present (mode 600)")


def check_identity() -> CheckResult:
    """``identity.yaml`` names the owner and the agent; the constitution (name, values, boundaries) fits
    its 1,500-character budget, and the report says whether boundaries are declared."""
    from colonymind.config import CONSTITUTION_CHARS, constitution_length, constitution_list, identity_path, load_identity
    path = identity_path()
    if not path.is_file():
        return CheckResult("identity", WARN, detail=f"{path} is missing", remedy="run 'colonymind init'")
    identity = load_identity()
    if not identity.get("owner", {}).get("name") or not identity.get("agent", {}).get("name"):
        return CheckResult("identity", WARN, detail=f"{path} lacks the owner or agent name",
                           remedy="run 'colonymind init'")
    length = constitution_length(identity)
    boundaries = constitution_list(identity.get("agent", {}).get("boundaries"))
    summary = (f"constitution {length}/{CONSTITUTION_CHARS} characters, "
               f"{len(boundaries)} boundar{'y' if len(boundaries) == 1 else 'ies'}")
    if length > CONSTITUTION_CHARS:
        return CheckResult("identity", WARN, detail=f"{path}: {summary}; the prompt clips it",
                           remedy="shorten agent.values or agent.boundaries (colonymind init --agent-values/--agent-boundaries)")
    return CheckResult("identity", PASS, detail=f"{path} names the owner and the agent; {summary}"
                       + ("" if boundaries else " (none declared: colonymind init --agent-boundaries)"))


def check_llm_config() -> CheckResult:
    """The router's persisted host config points at an OpenAI-compatible root."""
    from colonymind import get_state_dir
    from colonymind.config import LLM_CONFIG_FILE
    path = get_state_dir() / LLM_CONFIG_FILE
    if not path.exists():
        return CheckResult("llm-config", WARN, detail=f"{path} not found; the router has no endpoint",
                           remedy="run 'colonymind init --model-url <root>' or POST /v1/host/configure")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("top-level JSON value is not an object")
    except (OSError, ValueError) as exc:
        return CheckResult("llm-config", FAIL, detail=f"{path} is unreadable: {exc}",
                           remedy="fix or delete the file, then re-run 'colonymind init'")
    provider = str(cfg.get("provider", "") or "").strip().lower()
    base_url = str(cfg.get("baseUrl", "") or "").strip()
    if provider in OPENAI_COMPAT_PROVIDERS and base_url and not base_url.rstrip("/").endswith("/v1"):
        return CheckResult("llm-config", WARN,
                           detail=f"baseUrl {base_url!r} does not end with /v1",
                           remedy="run 'colonymind doctor --fix'")
    if provider in OPENAI_COMPAT_PROVIDERS and not str(cfg.get("apiKey") or "").strip():
        return CheckResult("llm-config", WARN, detail="apiKey is empty", remedy="run 'colonymind doctor --fix'")
    return CheckResult("llm-config", PASS, detail=f"{path} parsed (provider={provider or 'unknown'})")


def _hermes_python() -> Optional[Path]:
    from colonymind.config import load_config
    from colonymind.init import InitError, hermes_python_candidates, resolve_hermes_home
    config = load_config()
    configured = str(config.get("hermes.python") or "")
    home = resolve_hermes_home(None, config.get("hermes.home"))
    try:
        return next((p for p in hermes_python_candidates(None, configured, home) if p.exists()), None)
    except InitError:
        return None


def check_hermes_version() -> CheckResult:
    from colonymind.init import InitError, SUPPORTED_HERMES_RANGE, hermes_python_version, hermes_version_supported
    python = _hermes_python()
    if python is None:
        return CheckResult("hermes-version", FAIL, detail="no Hermes interpreter found",
                           remedy="install Hermes and re-run 'colonymind init'")
    try:
        version = hermes_python_version(python)
    except InitError as error:
        return CheckResult("hermes-version", FAIL, detail=str(error),
                           remedy="install a supported Hermes release, then 'colonymind upgrade'")
    if not version:
        return CheckResult("hermes-version", FAIL, detail=f"hermes-agent is not installed in {python}",
                           remedy="re-run 'colonymind init --hermes-python <interpreter>'")
    if not hermes_version_supported(version):
        return CheckResult("hermes-version", FAIL,
                           detail=f"Hermes {version} is outside {SUPPORTED_HERMES_RANGE}",
                           remedy="install a supported Hermes release, then 'colonymind upgrade'")
    return CheckResult("hermes-version", PASS, detail=f"Hermes {version} at {python}")


def check_adapter_version() -> CheckResult:
    from colonymind import __version__
    from colonymind.config import load_config
    from colonymind.init import (ADAPTER_DISTRIBUTION, MEMORY_PROVIDER, installed_adapter_version,
                                 read_hermes_config, resolve_hermes_home, selected_memory_provider)
    python = _hermes_python()
    if python is None:
        return CheckResult("adapter-version", SKIP, detail="no Hermes interpreter found")
    installed = installed_adapter_version(python)
    if installed is None:
        import shlex
        home = resolve_hermes_home(None, load_config().get("hermes.home"))
        config = read_hermes_config(home / "config.yaml")
        if (selected_memory_provider(config) == MEMORY_PROVIDER
                or "colonymind" in (config.get("plugins") or {}).get("enabled", [])):
            return CheckResult("adapter-version", FAIL,
                               detail=f"{ADAPTER_DISTRIBUTION} configured but not installed in {python} "
                                      f"for Hermes home {home}",
                               remedy=f"run 'colonymind init --hermes-home {shlex.quote(str(home))}' "
                                      "then 'hermes gateway restart' for that home")
        return CheckResult("adapter-version", FAIL, detail=f"{ADAPTER_DISTRIBUTION} is not installed in {python}",
                           remedy="run 'colonymind init' (or 'colonymind upgrade')")
    if installed != __version__:
        return CheckResult("adapter-version", FAIL,
                           detail=f"{ADAPTER_DISTRIBUTION} {installed} does not match the sidecar {__version__}",
                           remedy="run 'colonymind upgrade', then 'hermes gateway restart'")
    return CheckResult("adapter-version", PASS, detail=f"{ADAPTER_DISTRIBUTION} {installed} matches the sidecar")


def check_build_stamp() -> CheckResult:
    """The commit this install was built from, as the stamp beside the package says.

    Without a stamp the row is a warning, because a warning is the one status that changes neither the exit status
    nor ``ok``: a failed row changes the first, and a skipped or passed row counts in :func:`skip_dominated`.
    """
    from colonymind import build_stamp
    stamp = build_stamp.read()
    if stamp is None:
        return CheckResult("build-stamp", WARN,
                           detail="no readable build stamp: the commit this install was built from is not recorded",
                           remedy="in a source checkout run 'python scripts/write_build_stamp.py'; an install from a "
                                  "release wheel carries the stamp")
    built = f"commit {stamp['commit'][:12]}, " + (f"branch {stamp['branch']}" if stamp["branch"] else "detached")
    if stamp["dirty"]:
        return CheckResult("build-stamp", WARN, detail=f"{built}, changed: the tree had uncommitted changes",
                           remedy="commit the changes or take them out of the tree, run "
                                  "'python scripts/write_build_stamp.py' again, and build the wheel again if this "
                                  "install came from one")
    return CheckResult("build-stamp", PASS, detail=f"{built}, clean")


def check_pip_check() -> CheckResult:
    from colonymind.init import InitError, dependency_conflict_detail, pip_check
    python = _hermes_python()
    if python is None:
        return CheckResult("pip-check", SKIP, detail="no Hermes interpreter found")
    try:
        ok, output = pip_check(python)
    except InitError as exc:
        return CheckResult("pip-check", WARN, detail=str(exc))
    if not ok:
        unrelated, detail = dependency_conflict_detail(python, output)
        if unrelated:
            return CheckResult("pip-check", WARN, detail=f"dependency conflict in Hermes' own environment:\n{detail}",
                               remedy="check these packages with Hermes' dependency manager")
        return CheckResult("pip-check", FAIL, detail=detail,
                           remedy="run 'colonymind upgrade'; check the conflicting package requirements if it persists")
    return CheckResult("pip-check", PASS, detail=f"pip check clean in {python}")


def check_hermes_keys() -> CheckResult:
    """The Hermes config carries every key ``init`` writes."""
    from colonymind.config import KEY_FILE, load_config
    from colonymind.init import SKILLS_DIR, read_hermes_config, reconcile_hermes_config
    config = load_config()
    path = config.hermes_home / "config.yaml"
    if not path.is_file():
        return CheckResult("hermes-keys", FAIL, detail=f"{path} is missing", remedy="run 'colonymind init'")
    current = read_hermes_config(path)
    _, changes = reconcile_hermes_config(current, sidecar_url=config.sidecar_url,
                                         key_file=config.home / KEY_FILE,
                                         skills_dir=config.home / SKILLS_DIR)
    if changes:
        return CheckResult("hermes-keys", FAIL, detail="missing: " + "; ".join(changes),
                           remedy="run 'colonymind upgrade', then 'hermes gateway restart'")
    return CheckResult("hermes-keys", PASS, detail=f"{path} carries the adapter keys")


def check_worker_profile() -> CheckResult:
    """The worker profile matches the main model, and stock Hermes gives a dispatched worker that model
    and no toolset beyond the mind's."""
    from colonymind.config import KEY_FILE, load_config
    from colonymind.init import (PLUGIN_NAME, WORKER_PROFILE, InitError, profiles_root, read_hermes_config,
                                resolve_worker_profile, worker_profile_config, worker_profile_pending)
    config = load_config()
    path = profiles_root(config.hermes_home) / WORKER_PROFILE / "config.yaml"
    if not path.is_file():
        return CheckResult("worker-profile", FAIL, detail=f"{path} is missing",
                           remedy="run 'colonymind upgrade'")
    try:
        expected = worker_profile_config(read_hermes_config(config.hermes_home / "config.yaml"), config,
                                         sidecar_url=config.sidecar_url, key_file=config.home / KEY_FILE)
    except InitError as error:
        return CheckResult("worker-profile", FAIL, detail=str(error),
                           remedy="name an entry of the main Hermes config in mind.worker_model, or set it to null")
    if worker_profile_pending(profiles_root(config.hermes_home), expected):
        return CheckResult("worker-profile", FAIL, detail=f"profile {WORKER_PROFILE} differs from the model it "
                           "follows or the mind settings", remedy="run 'colonymind upgrade'")
    python = _hermes_python()
    if python is None:
        return CheckResult("worker-profile", SKIP, detail="no Hermes interpreter found")
    worker = resolve_worker_profile(python, config.hermes_home)
    if worker.get("error"):
        return CheckResult("worker-profile", FAIL, detail=f"a mind task cannot resolve its model: {worker['error']}",
                           remedy="fix the main model in Hermes, then 'colonymind upgrade'")
    extra = sorted(set(worker["toolsets"]) - set(config.get("mind.worker_toolsets") or []) - {PLUGIN_NAME})
    if extra:
        return CheckResult("worker-profile", FAIL, detail=f"a mind task would get {', '.join(extra)}",
                           remedy="run 'colonymind upgrade'")
    return CheckResult("worker-profile", PASS, detail=f"profile {WORKER_PROFILE}: {worker.get('provider')} "
                       f"at {worker.get('base_url')}, toolsets {', '.join(worker['toolsets'])}")


def check_plugin_loaded() -> CheckResult:
    """The plugin is enabled in Hermes' config and registered where Hermes runs."""
    from colonymind.config import load_config
    from colonymind.init import PLUGIN_NAME, read_hermes_config
    config = load_config()
    current = read_hermes_config(config.hermes_home / "config.yaml")
    enabled = (current.get("plugins") or {}).get("enabled") if isinstance(current.get("plugins"), dict) else None
    if not isinstance(enabled, list) or PLUGIN_NAME not in enabled:
        return CheckResult("plugin-loaded", FAIL, detail="plugins.enabled lacks colonymind",
                           remedy="run 'colonymind upgrade', then 'hermes gateway restart'")
    python = _hermes_python()
    if python is None:
        return CheckResult("plugin-loaded", SKIP, detail="no Hermes interpreter found")
    import subprocess
    result = subprocess.run(
        [str(python), "-I", "-c",
         "import importlib.metadata as m, json; "
         "print(json.dumps(sorted({e.name for e in m.entry_points(group='hermes_agent.plugins')} | "
         "{e.name for e in m.entry_points(group='hermes_agent.memory_providers')})))"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    names = set()
    if result.returncode == 0 and result.stdout.strip():
        try:
            names = set(json.loads(result.stdout.strip().splitlines()[-1]))
        except ValueError:
            names = set()
    missing = {PLUGIN_NAME, "colonymind-memory"} - names
    if missing:
        return CheckResult("plugin-loaded", FAIL,
                           detail=f"entry points not registered in {python}: {', '.join(sorted(missing))}",
                           remedy="run 'colonymind upgrade', then 'hermes gateway restart'")
    return CheckResult("plugin-loaded", PASS, detail="plugin enabled and its entry points registered")


def check_vector_store() -> CheckResult:
    """The vector store library imports in the sidecar's own interpreter."""
    import sys
    from colonymind.init import VECTOR_STORE_MODULE, VECTOR_STORE_REMEDY, vector_store_available
    if not vector_store_available():
        return CheckResult("vector-store", FAIL, detail=f"{VECTOR_STORE_MODULE} is not importable in {sys.executable}",
                           remedy=VECTOR_STORE_REMEDY)
    return CheckResult("vector-store", PASS, detail=f"{VECTOR_STORE_MODULE} importable in {sys.executable}")


def check_service() -> CheckResult:
    """This instance's user service (launchd, systemd --user) is installed, running and current.

    A sidecar started by hand (``colonymind start --detach``) passes every HTTP check while
    nothing restarts it after a crash or starts it at login; this check says so.
    """
    from colonymind import init
    from colonymind.config import load_config
    from colonymind.services.instance import ServiceError
    try:
        service = init._service(load_config())
    except ServiceError as exc:
        return CheckResult("service", SKIP, detail=str(exc))
    manager = "launchd" if service.platform == "darwin" else "systemd --user"
    try:
        installed = service._owned()
    except ServiceError as exc:
        return CheckResult("service", FAIL, detail=str(exc),
                           remedy="remove or move the file named, then 'colonymind service install'")
    if not installed:
        return CheckResult("service", WARN,
                           detail=f"no user service is installed for this instance ({manager}): a sidecar started "
                                  "by hand is not restarted after a crash or started at login",
                           remedy="'colonymind service install', then 'colonymind service start' (stop a sidecar "
                                  "started by hand first)")
    try:
        status = service.status()
    except ServiceError as exc:
        return CheckResult("service", WARN, detail=f"{service.label} is installed but {manager} did not answer: {exc}")
    if not status["running"]:
        state = "loaded" if status["loaded"] else "not loaded"
        return CheckResult("service", FAIL, detail=f"{manager} {service.label} is installed but not running ({state})",
                           remedy="'colonymind service start' (stop a sidecar started by hand first); "
                                  f"the logs are {service.log} and {service.manager_log}")
    stale = service.outdated()
    if stale:
        return CheckResult("service", WARN,
                           detail=f"{service.label} runs from a unit an earlier release wrote: {'; '.join(stale)}",
                           remedy="'colonymind service stop', 'colonymind service install', 'colonymind service start'")
    autostart = status.get("enabled", True) is not False
    detail = f"{manager} {service.label} running (pid {status['pid']})"
    if not autostart:
        return CheckResult("service", WARN, detail=detail + ", but not enabled: it does not start at login",
                           remedy="'colonymind service install' enables it")
    return CheckResult("service", PASS, detail=detail + ", restarted after a crash and started at login")


KEYWORD_ONLY_REMEDY = ("colonymind init --embed-url auto (asks the model endpoint Hermes uses for an embeddings "
                       "route; never a chat request), --embed-url <root> --embed-model <model>, or --embed-url "
                       "local (an embedder the sidecar runs itself; needs colonymind[vectors])")
LOCAL_REFETCH = ("re-run 'colonymind init' (it keeps the recorded local embedder and fetches whichever of its pinned "
                 "files are missing), then 'colonymind service restart'")


def check_recall_mode() -> CheckResult:
    """Whether recall can find meaning or only words, from ``colonymind.yaml``: a default install has no
    embedder and recalls by keywords only. For the local embedder, whether its pinned files and the extra that
    runs them are there (no model is loaded)."""
    from colonymind.config import load_config
    config = load_config()
    local = config.get("router.embed_provider") == "local"
    if not local and not config.get("router.embed_url"):
        return CheckResult("recall-mode", WARN,
                           detail="keyword recall only: no embedding endpoint (router.embed_url is empty), so "
                                  "recall finds words, not meaning",
                           remedy=KEYWORD_ONLY_REMEDY)
    if config.get("mind.faculties.semantic_recall") is False:
        # Releases before the switch was live wrote false whenever init found no endpoint.
        return CheckResult("recall-mode", WARN,
                           detail="keyword recall only: "
                                  + ("router.embed_provider is local" if local else "router.embed_url is set")
                                  + " but mind.faculties.semantic_recall is false",
                           remedy="set mind.faculties.semantic_recall: true in colonymind.yaml (or re-run 'colonymind "
                                  "init --embed-url ...'), then 'colonymind service restart'")
    if local:
        return _local_embedder_mode(config)
    return CheckResult("recall-mode", PASS, detail=f"semantic and keyword recall (embeddings at "
                                                   f"{config.get('router.embed_url')})")


def _local_embedder_mode(config) -> CheckResult:
    from colonymind.vector import local_embedder
    model, revision = str(config.get("router.embed_model")), str(config.get("router.embed_revision"))
    if not local_embedder.extra_available():
        return CheckResult("recall-mode", FAIL,
                           detail=f"the local embedder needs {local_embedder.EXTRA}, which is not installed in the "
                                  "sidecar's environment: recall finds words only",
                           remedy=f"{local_embedder.INSTALL}, then 'colonymind service restart'")
    unmet = local_embedder.unmet(model)
    if unmet:
        return CheckResult("recall-mode", FAIL,
                           detail=f"the sidecar's environment cannot load the local embedder {model} "
                                  f"({', '.join(unmet)}): recall finds words only",
                           remedy=f"{local_embedder.upgrade_command(model)}, then 'colonymind service restart'")
    target = local_embedder.model_dir(config.home, model, revision)
    lacking = local_embedder.missing(target)
    if lacking:
        if not target.is_dir():
            where = f"the local embedder's files are not in {target}"
        elif lacking == [local_embedder.MANIFEST]:
            where = f"the local embedder's download into {target} did not finish"
        else:
            where = (f"the local embedder's files in {target} are incomplete ({', '.join(lacking[:5])}"
                     + (f" and {len(lacking) - 5} more" if len(lacking) > 5 else "") + " missing)")
        return CheckResult("recall-mode", FAIL,
                           detail=f"{where} ({model} at revision {revision[:12]}): the embedder cannot start, so "
                                  "recall finds words only",
                           remedy=LOCAL_REFETCH)
    return CheckResult("recall-mode", PASS,
                       detail=f"semantic and keyword recall (local embedder {model} at revision {revision[:12]}, "
                              f"{config.get('router.embed_dims')} dimensions, run by the sidecar from {target})")


CALIBRATE_REMEDY = ("calibrate a cutoff for this reranker on your own questions (docs/RECALL-HYBRID.md, Returning "
                    "no useful memory) and set COLONYMIND_RECALL_RERANK_MIN_SCORE and "
                    "COLONYMIND_RECALL_RERANK_CALIBRATION under environment in colonymind.yaml")


def configured_calibration_fingerprint(derived: dict) -> Optional[str]:
    """The calibration fingerprint the sidecar computes for the reranker *derived* (the environment
    ``colonymind.yaml`` gives the sidecar) configures: a reranker endpoint, whose stamp the sidecar builds from
    the same values, with no model loaded and no request. None when the sidecar picks the reranker itself (an
    in-process one, by its hardware), so the stamp cannot be known here."""
    import unittest.mock
    model = str(derived.get("COLONYMIND_RERANKER_MODEL") or "")
    base_url = str(derived.get("COLONYMIND_RERANKER_BASE_URL") or "")
    remote = base_url or derived.get("COLONYMIND_RERANKER_PROVIDER") == "openai_api"
    if not remote or model.lower() in ("", "none", "null"):
        return None
    from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
    from colonymind.vector.reranker import OpenAIAPIRerankerProvider
    provider = OpenAIAPIRerankerProvider(model)
    provider.configure(base_url, "", str(derived.get("COLONYMIND_RERANKER_PROMPT_STYLE") or ""))
    # The stamp reads the sidecar's environment: the one this configuration gives it, not doctor's own.
    with unittest.mock.patch.dict(os.environ, {key: str(value) for key, value in derived.items()}, clear=True):
        return calibration_fingerprint(provider_calibration_metadata(provider))


def check_recall_cutoff() -> CheckResult:
    """Whether recall may return nothing: only a reranker with a calibrated cutoff lets it abstain, and the
    sidecar applies the cutoff only when it is a finite number and its fingerprint matches the reranker's
    configuration (``colonymind.memory.selection``). Read from ``colonymind.yaml`` (the keys and its
    ``environment``); no request is sent."""
    import math
    from colonymind.config import apply_environment, load_config
    derived = apply_environment(load_config(), environ={})
    reranking = derived.get("COLONYMIND_RECALL_RERANK", "off").strip().lower() == "on"
    raw = str(derived.get("COLONYMIND_RECALL_RERANK_MIN_SCORE") or "").strip()
    expected = str(derived.get("COLONYMIND_RECALL_RERANK_CALIBRATION") or "").strip()
    never = "recall never abstains, so every turn gets its best candidates even when none of them answers"
    if not (reranking and raw and expected):
        missing = [name for name, given in (("a reranker (router.rerank_url)", reranking),
                                            ("COLONYMIND_RECALL_RERANK_MIN_SCORE", raw),
                                            ("COLONYMIND_RECALL_RERANK_CALIBRATION", expected)) if not given]
        partial = f" ({', '.join(missing)} not set)" if len(missing) < 3 else ""
        return CheckResult("recall-cutoff", WARN, detail=f"no calibrated cutoff: {never}{partial}",
                           remedy="configure a reranker (router.rerank_url, router.rerank_model), " + CALIBRATE_REMEDY)
    try:
        score = float(raw)
    except ValueError:
        score = math.nan
    if not math.isfinite(score):
        return CheckResult("recall-cutoff", WARN,
                           detail=f"the calibrated cutoff {raw!r} is not a finite number: the sidecar ignores it, so "
                                  + never,
                           remedy="set COLONYMIND_RECALL_RERANK_MIN_SCORE to the score your calibration chose (a "
                                  "number), under environment in colonymind.yaml")
    actual = configured_calibration_fingerprint(derived)
    if actual is None:
        return CheckResult("recall-cutoff", WARN,
                           detail=f"calibrated cutoff {raw} set, but its fingerprint cannot be checked here: the "
                                  "sidecar picks the reranker itself (no router.rerank_url), and applies the cutoff "
                                  "only while the fingerprint matches that reranker; otherwise " + never,
                           remedy="look for 'Rerank abstention calibration' in the sidecar's log (mismatch or "
                                  "unverified means the cutoff is off), or name the reranker (router.rerank_url, "
                                  "router.rerank_model) so doctor can check it; recalibrate if it does not match "
                                  "(docs/RECALL-HYBRID.md)")
    if actual != expected:
        return CheckResult("recall-cutoff", WARN,
                           detail=f"calibrated cutoff {raw} does not apply: COLONYMIND_RECALL_RERANK_CALIBRATION does "
                                  "not match the configured reranker's fingerprint (its model, endpoint, prompt "
                                  "style, embedding model and width, index generation or candidate format changed "
                                  "since the calibration), so the sidecar switches it off and " + never,
                           remedy="recalibrate for this configuration: " + CALIBRATE_REMEDY)
    return CheckResult("recall-cutoff", PASS,
                       detail=f"calibrated cutoff {raw}; its fingerprint matches the configured reranker, so recall "
                              "may return nothing when no candidate answers")


#: The 1.11 decision layer's variables: retired with it (Instinct reads ``COLONYMIND_INSTINCT``).
RETIRED_DECISION_VARIABLES = ("COLONYMIND_DECISIONS_URL", "COLONYMIND_DECISIONS_TIMEOUT_MS",
                              "COLONYMIND_DECISIONS_POINTS")


def check_instinct() -> CheckResult:
    """Instinct: the fast provider configured or not (its kind, never its address), the decisions of the last day
    and the points by state; a warning names a retired decision variable still set in the environment."""
    from datetime import timedelta

    from colonymind.config import ConfigError, instance_home, load_config
    from colonymind.util.temporal import now_utc
    retired = [name for name in RETIRED_DECISION_VARIABLES if str(os.environ.get(name) or "").strip()]
    try:
        cfg = load_config()
    except ConfigError:
        return CheckResult("instinct", SKIP, detail="colonymind.yaml does not load (see the config check)")
    from colonymind.instinct import REGISTRY, settings, state as point_state
    from colonymind.instinct.log import DB_FILE, Store
    section = settings.parse(cfg.get("instinct") or {})
    provider = section["providers"].get(section["fast"]) if section["fast"] else None
    parts = [f"fast provider {provider['kind']} configured" if provider and provider.get("url")
             else "no fast provider configured: every point keeps its existing path"]
    path = instance_home() / DB_FILE
    overrides, recent, last = {}, 0, None
    if path.is_file():
        store = Store(path, create=False)
        overrides = store.states()
        rows = store.rows(since=now_utc() - timedelta(days=1))
        recent, last = len(rows), (rows[0]["at"] if rows else None)
    by_state: dict = {}
    for point in REGISTRY.values():
        for tier, state in point_state.states_of(point, overrides).items():
            by_state[state] = by_state.get(state, 0) + 1
    parts.append("model tiers: " + ", ".join(f"{count} {state}" for state, count in sorted(by_state.items())))
    parts.append(f"{recent} decisions in the last day" + (f" (the last at {last})" if last else ""))
    detail = "; ".join(parts)
    if retired:
        return CheckResult("instinct", WARN, detail=f"{', '.join(retired)} {'is' if len(retired) == 1 else 'are'} "
                                                    f"set, but nothing reads {'it' if len(retired) == 1 else 'them'} "
                                                    f"since the decision layer became Instinct; {detail}",
                           remedy="unset it and configure instinct.providers in colonymind.yaml ('colonymind upgrade' "
                                  "carries a decisions section there)")
    return CheckResult("instinct", PASS, detail=detail)


def check_instance_secret() -> CheckResult:
    """The instance's own secret (``instinct.key``): not made yet, or a whole 32-byte key only its owner can read. A
    file that holds no whole key keys nothing, and nothing else says so but a log line: every decision hashes its
    state under a key of its process alone, which nothing later matches, and model usage drops each row that names a
    role or an endpoint by its alias. The file is never rewritten here."""
    from colonymind.config import instance_home, key_file_is_private
    from colonymind.instance import SECRET_BYTES, SECRET_FILE, instance_secret
    home = instance_home()
    path = home / SECRET_FILE
    if path.is_symlink() and not path.exists():
        # A link whose target is gone holds the name: the key the sidecar makes is never linked into place.
        return CheckResult("instance-secret", FAIL,
                           detail=f"{path} is a link to {os.readlink(path)}, which is not there, and the sidecar "
                                  "cannot make a key in its place: decisions hash their state under a key of their "
                                  "process alone, which nothing later matches, and model usage drops every call it "
                                  "would name by an alias",
                           remedy=f"restore the key the link points to, or remove the link and restart the sidecar "
                                  f"to make a new key (what the old key hashed then matches nothing): rm {path} && "
                                  f"colonymind service restart")
    if not path.exists():
        return CheckResult("instance-secret", PASS, detail=f"{path} not made yet (the first decision or model usage "
                                                           "row that needs it makes it)")
    try:
        instance_secret(home, create=False)
    except ValueError:
        return CheckResult("instance-secret", FAIL,
                           detail=f"{path} holds no {SECRET_BYTES}-byte key, so it keys nothing: decisions hash their "
                                  "state under a key of their process alone, which nothing later matches, and model "
                                  "usage drops every call it would name by an alias",
                           remedy=f"restore it from a backup ('colonymind restore' of an encrypted one), or remove it "
                                  f"and restart the sidecar to make a new one (what the old key hashed then matches "
                                  f"nothing): rm {path} && colonymind service restart")
    except OSError as error:
        return CheckResult("instance-secret", FAIL, detail=f"{path} cannot be read ({type(error).__name__})",
                           remedy=f"make it this user's and readable by it alone: chmod 600 {path}")
    if not key_file_is_private(path):
        return CheckResult("instance-secret", WARN, detail=f"{path} is readable by other users",
                           remedy=f"chmod 600 {path}")
    return CheckResult("instance-secret", PASS, detail=f"{path} holds the instance's key (mode 600)")


# ---------------------------------------------------------------------------
# Could the agent approve its own asks? (docs/MIND.md, Asks: console and API answers)
# ---------------------------------------------------------------------------

#: The tools with which an agent reads a file on the machine its tools run on: a shell, code, a process, the file
#: tools. On the local backend they reach the instance directory, and so the key that writes as the owner.
KEY_READING_TOOLS = ("terminal", "process_manage", "execute_code", "read_file", "search_files")
#: The ones that start a program (a ``hermes chat`` of its own, say).
SHELL_TOOLS = frozenset({"terminal", "process_manage", "execute_code"})
CLI_PLATFORM = "cli"
# Run under Hermes' interpreter with its own configuration and ``.env``: the facts only, as names and booleans.
_OWNER_LANE_PROBE = """
import json, os, sys
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=sys.argv[1])
wanted, reading = json.loads(sys.argv[2]), set(json.loads(sys.argv[3]))
out = {"backend": (os.getenv("TERMINAL_ENV") or "local").strip().lower() or "local", "lanes": {}, "whatsapp": None}
try:
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import resolve_toolset
    config = load_config()
    for platform in wanted:
        try:
            tools = set()
            for name in _get_platform_tools(config, platform):
                try:
                    tools.update(resolve_toolset(name))
                except Exception:
                    pass
            out["lanes"][platform] = sorted(tools & reading)
        except Exception:
            out["lanes"][platform] = None
except Exception as error:
    out["error"] = type(error).__name__
try:
    from gateway.config import Platform, load_gateway_config
    bridge = load_gateway_config().platforms.get(Platform.WHATSAPP)
    out["whatsapp"] = {"enabled": bool(bridge is not None and bridge.enabled),
                       "bot": (os.getenv("WHATSAPP_MODE") or "self-chat").strip().lower() == "bot"}
except Exception:
    pass
print(json.dumps(out))
"""


def owner_lane_facts(python: Path, hermes_home: Path, platforms: List[str]) -> dict:
    """What Hermes runs for these platforms, asked of Hermes itself: ``backend`` (where its tools run), ``lanes``
    (each platform's key-reading tools; None when Hermes could not say) and ``whatsapp`` (``{enabled, bot}``, or
    None). ``error`` when Hermes could not be asked. Names and booleans only: nothing from its configuration."""
    import subprocess
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("TERMINAL_", "WHATSAPP_"))}
    environment["HERMES_HOME"] = str(hermes_home)
    try:
        result = subprocess.run([str(python), "-c", _OWNER_LANE_PROBE, str(hermes_home),
                                 json.dumps(sorted(set(platforms))), json.dumps(list(KEY_READING_TOOLS))],
                                capture_output=True, text=True, timeout=120, check=False, env=environment)
        found = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError) as exc:
        return {"error": type(exc).__name__}
    except Exception as exc:  # noqa: BLE001 - a timeout, whatever Hermes did: never a guess
        return {"error": type(exc).__name__}
    return found if isinstance(found, dict) else {"error": "unexpected answer"}


def _terminal_remedy(owners: Mapping[str, set], served: Iterable[str]) -> str:
    """``colonymind credentials set`` for each credential that may write the owner from the terminal: the terminal
    goes and every other platform it had stays (``owner: any`` becomes the platforms Hermes serves, ``served``);
    one that had only the terminal writes as the owner no more. Credentials that never had it are left alone."""
    from colonymind.api.credentials import ANY_PLATFORM
    commands, remaining, served = [], set(), set(served)
    for name in sorted(owners):
        platforms = owners[name]
        kept = sorted((served if ANY_PLATFORM in platforms else set(platforms)) - {CLI_PLATFORM, ANY_PLATFORM})
        remaining.update(kept)
        if CLI_PLATFORM in platforms or ANY_PLATFORM in platforms:
            flags = " ".join(f"--owner-platform {platform}" for platform in kept) or "--no-owner"
            commands.append(f"colonymind credentials set {name} {flags}")
    after = ("the owner then answers in their own chat" if remaining else
             "then no credential writes the owner's turns: add the platforms the owner chats on with "
             "--owner-platform")
    return "; ".join(commands) + f" ({after})"


def owner_lane_results(facts: dict, owners: Mapping[str, Iterable[str]]) -> List[CheckResult]:
    """The three warnings from Hermes' facts and, per credential, the platforms it may write the owner from (``*``:
    every one). Each names its remedy, for the credentials it concerns; a fact Hermes could not give is skipped,
    never guessed."""
    from colonymind.api.credentials import ANY_PLATFORM
    if not isinstance(facts, dict) or facts.get("error") or not isinstance(facts.get("lanes"), dict):
        return [CheckResult("owner-lanes", SKIP, detail="Hermes could not say what the owner's lanes run"
                            + (f" ({facts.get('error')})" if isinstance(facts, dict) and facts.get("error") else ""))]
    owners = {str(name): set(platforms) for name, platforms in owners.items() if platforms}
    owner_platforms = set().union(*owners.values())
    anywhere = ANY_PLATFORM in owner_platforms
    backend = str(facts.get("backend") or "local")
    local = backend == "local"
    lanes = {str(name): tools for name, tools in facts["lanes"].items()
             if anywhere or name in owner_platforms}
    results: List[CheckResult] = []
    unknown = sorted(name for name, tools in lanes.items() if tools is None)
    exposed = {name: sorted(tools) for name, tools in lanes.items() if tools}
    if not exposed and unknown:
        results.append(CheckResult("owner-lane-tools", SKIP,
                                   detail=f"Hermes could not say what the owner's lanes on {', '.join(unknown)} run"))
    elif not exposed:
        results.append(CheckResult("owner-lane-tools", PASS,
                                   detail="the owner's lanes have no terminal, code or file tool"))
    elif not local:
        results.append(CheckResult("owner-lane-tools", PASS,
                                   detail=f"the owner's lanes run their tools on the {backend} backend, away from "
                                          "the instance key (as long as it does not mount the instance directory)"))
    else:
        tools = sorted({tool for names in exposed.values() for tool in names})
        results.append(CheckResult(
            "owner-lane-tools", WARN,
            detail=(f"the owner's lanes on {', '.join(sorted(exposed))} run {', '.join(tools)} on this machine "
                    "(terminal backend local), where the instance key can be read: an agent that reads it can forge "
                    "the owner's turn and answer its own asks (every approval is still told to the owner in their "
                    "chat)"),
            remedy=("run the owner lane's tools where the instance directory cannot be read: set terminal.backend "
                    "in Hermes' config.yaml to a container or remote backend (docker, ssh, modal), or remove the "
                    "terminal, code and file tools from the owner's platforms (platform_toolsets)")))
    if anywhere or CLI_PLATFORM in owner_platforms:
        shells = sorted(name for name, names in lanes.items() if names and SHELL_TOOLS.intersection(names))
        if local and shells:
            granting = sorted(name for name, platforms in owners.items()
                              if CLI_PLATFORM in platforms or ANY_PLATFORM in platforms)
            which = ("credentials " if len(granting) > 1 else "credential ") + ", ".join(granting)
            results.append(CheckResult(
                "cli-owner-platform", WARN,
                detail=(f"the terminal (cli) is an owner platform ({which}) and the owner's lanes on "
                        f"{', '.join(shells)} run a shell on this machine: a hermes chat the agent starts from it is "
                        "written as the owner, and can answer the owner's asks"),
                remedy=_terminal_remedy(owners, (name for name in facts["lanes"] if name != CLI_PLATFORM))))
        else:
            results.append(CheckResult("cli-owner-platform", PASS,
                                       detail="the terminal (cli) is an owner platform, and no owner lane runs a "
                                              "shell on this machine"))
    bridge = facts.get("whatsapp")
    if (anywhere or "whatsapp" in owner_platforms) and isinstance(bridge, dict) and bridge.get("enabled"):
        if bridge.get("bot"):
            results.append(CheckResult("whatsapp-mode", PASS, detail="the WhatsApp bridge runs in bot mode, on the "
                                                                     "agent's own number"))
        else:
            results.append(CheckResult(
                "whatsapp-mode", WARN,
                detail=("the WhatsApp bridge runs in shared-account mode: the agent answers from the owner's own "
                        "account, so what the owner types in other chats and what the agent sends share one sender "
                        "with the owner's answers"),
                remedy=("give the agent its own number and run the bridge in bot mode: WHATSAPP_MODE=bot in Hermes' "
                        ".env, then 'hermes whatsapp' to pair it")))
    return results


def check_owner_lanes() -> List[CheckResult]:
    """Could the agent approve its own asks here? Hermes is asked what the owner's lanes run
    (``owner_lane_facts``), for every platform a credential may write the owner from."""
    unreadable = [CheckResult("owner-lanes", SKIP, detail="colonymind.yaml could not be read (see the config check)")]
    try:
        python = _hermes_python()
    except Exception:  # noqa: BLE001 - the config check reports what does not parse; this one quotes nothing
        return unreadable
    if python is None:
        return [CheckResult("owner-lanes", SKIP, detail="no Hermes interpreter found")]
    try:
        from colonymind.api.credentials import ANY_PLATFORM, CredentialTable
        from colonymind.config import load_config
        config = load_config()
        grants = CredentialTable.from_config(config.get("credentials")).grants()
        owners = {grant.name: set(grant.owner) for grant in grants if grant.owner}
        hermes_home = config.hermes_home
    except Exception:  # noqa: BLE001
        return unreadable
    if not owners:
        return [CheckResult("owner-lanes", PASS, detail="no credential may write as the owner")]
    platforms = set().union(*owners.values())
    if ANY_PLATFORM in platforms:
        from colonymind.init import hermes_platform_names
        platforms = (platforms - {ANY_PLATFORM}) | set(hermes_platform_names(python)) | {CLI_PLATFORM}
    return owner_lane_results(owner_lane_facts(python, hermes_home, sorted(platforms)), owners)


# ---------------------------------------------------------------------------
# Could a contact run this machine? (docs/HERMES-ADAPTER.md, Owner and guest)
# ---------------------------------------------------------------------------

#: What a non-owner session must not have on the machine the instance key is on: what runs a program or code, drives
#: the desktop or a browser (and every ``browser_`` tool), writes files, or does one of those later in the owner's name
#: (the plugin guard's executor table, ``plugins/hermes-plugin/guard.py``). The guard refuses them in every non-owner
#: session unless ``guests.executor_platforms`` opts its platform in.
GUEST_EXECUTOR_TOOLS = (
    "terminal", "process_manage", "process", "execute_code", "delegate_task", "computer_use", "browser_exec",
    "read_terminal", "close_terminal", "desktop_preview", "drive_preview", "annotate_preview", "read_window_below",
    "focus_pane", "setup_mcp", "gui_tour", "show_tip", "apply_layout", "open_preview", "close_preview", "read_preview",
    "write_file", "patch", "cronjob_manage", "skill_manage", "kanban_create", "colonymind_action")
#: Hermes' own lanes, where no contact speaks: its terminal and its scheduled runs.
INTERNAL_LANES = frozenset({CLI_PLATFORM, "cron"})
# Run under Hermes' interpreter with its own configuration and ``.env``: the platforms it serves (its gateway's enabled
# ones and every ``platforms.<name>.enabled`` in config.yaml), each with its executor tools. Names only; None for a
# platform whose tools Hermes cannot say without loading a plugin (its default toolset resolves to nothing).
_GUEST_LANE_PROBE = """
import json, os, sys
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=sys.argv[1])
internal, executors = set(json.loads(sys.argv[2])), set(json.loads(sys.argv[3]))
out = {"backend": (os.getenv("TERMINAL_ENV") or "local").strip().lower() or "local", "lanes": {}}
try:
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    from toolsets import resolve_toolset
    config = load_config()
    from toolsets import validate_toolset
    served = set()
    try:
        from gateway.config import load_gateway_config
        for platform, settings in load_gateway_config().platforms.items():
            if getattr(settings, "enabled", False):
                served.add(str(getattr(platform, "value", platform)).lower())
    except Exception:
        out["partial"] = True
    for name, settings in (config.get("platforms") or {}).items():
        if isinstance(settings, dict) and settings.get("enabled") is True:
            served.add(str(name).lower())
    for platform in sorted(served - internal):
        try:
            tools, whole = set(), True
            for name in _get_platform_tools(config, platform):
                try:
                    found = resolve_toolset(name)
                except Exception:
                    found, whole = [], False
                if not found and not validate_toolset(name):
                    whole = False
                tools.update(found)
            exposed = sorted(t for t in tools if t in executors or t.startswith("browser_"))
            out["lanes"][platform] = exposed if exposed or whole else None
        except Exception:
            out["lanes"][platform] = None
except Exception as error:
    out["error"] = type(error).__name__
print(json.dumps(out))
"""


def guest_lane_facts(python: Path, hermes_home: Path) -> dict:
    """What Hermes offers on the platforms it serves, asked of Hermes itself: ``backend`` and ``lanes`` (each served
    platform but its terminal and cron, with its executor tools, or None when Hermes could not say). ``error`` when
    Hermes could not be asked. Names only: nothing from its configuration."""
    import subprocess
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("TERMINAL_", "WHATSAPP_"))}
    environment["HERMES_HOME"] = str(hermes_home)
    try:
        result = subprocess.run([str(python), "-c", _GUEST_LANE_PROBE, str(hermes_home),
                                 json.dumps(sorted(INTERNAL_LANES)), json.dumps(list(GUEST_EXECUTOR_TOOLS))],
                                capture_output=True, text=True, timeout=120, check=False, env=environment)
        found = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError) as exc:
        return {"error": type(exc).__name__}
    except Exception as exc:  # noqa: BLE001 - a timeout, whatever Hermes did: never a guess
        return {"error": type(exc).__name__}
    return found if isinstance(found, dict) else {"error": "unexpected answer"}


def guest_lane_results(facts: dict, opted: Iterable[str]) -> List[CheckResult]:
    """``guest-lane-tools``: the non-owner lanes whose platforms offer executor tools on this machine. One the operator
    opted in (``guests.executor_platforms``) warns, with its remedy; the rest pass, named, since the guard refuses
    those tools there to everyone but the owner. A lane Hermes cannot say about is named, never guessed; opted in, it
    warns too. When Hermes could not list every platform it serves (``partial``), nothing passes: the lanes it listed
    are named and the check skips, and an opted-in platform missing from the list warns, since it may be served with
    every tool."""
    name = "guest-lane-tools"
    if not isinstance(facts, dict) or facts.get("error") or not isinstance(facts.get("lanes"), dict):
        why = f" ({facts.get('error')})" if isinstance(facts, dict) and facts.get("error") else ""
        return [CheckResult(name, SKIP, detail=f"Hermes could not say what its platforms run{why}")]
    opted = {str(item).strip().lower() for item in opted}
    backend = str(facts.get("backend") or "local")
    lanes = {str(platform): tools for platform, tools in facts["lanes"].items()}
    unknown = sorted(platform for platform, tools in lanes.items() if tools is None)
    exposed = {platform: sorted(tools) for platform, tools in lanes.items() if tools}
    unsaid = (f"; Hermes could not say what {', '.join(unknown)} run" if unknown else "") + \
        ("; Hermes could not list its gateway's platforms, so a platform may be missing" if facts.get("partial") else "")
    if backend != "local":
        return [CheckResult(name, PASS, detail=f"Hermes runs its tools on the {backend} backend, away from the "
                                               "instance key (as long as it does not mount the instance directory)")]
    partial = bool(facts.get("partial"))
    missed = sorted(opted - set(lanes)) if partial else []
    open_lanes = sorted((set(exposed) | set(unknown)) & opted)
    if open_lanes or missed:
        tools = sorted({tool for platform in open_lanes for tool in exposed.get(platform) or ["any tool"]}
                       | ({"any tool"} if missed else set()))
        where = open_lanes + [f"{platform} (if Hermes serves it)" for platform in missed]
        return [CheckResult(
            name, WARN,
            detail=(f"non-owner sessions on {', '.join(where)} keep {', '.join(tools)} on this machine "
                    "(guests.executor_platforms in colonymind.yaml): a contact or caller there runs a shell, code or "
                    "file tools where the instance key and every file of this user can be read; the guard still "
                    "refuses a call that names the key or an answer route, which a shell can spell another way"
                    + unsaid),
            remedy=("remove those platforms from guests.executor_platforms, or run Hermes' tools on a container or "
                    "remote backend (terminal.backend in Hermes' config.yaml)"))]
    if exposed:
        tools = sorted({tool for names in exposed.values() for tool in names})
        return [CheckResult(name, SKIP if partial else PASS, detail=(
            f"Hermes offers {', '.join(tools)} on {', '.join(sorted(exposed))}; the guard refuses them there to "
            "everyone but the owner (guests.executor_platforms opts a platform in)" + unsaid))]
    if unknown or partial:
        return [CheckResult(name, SKIP, detail=unsaid.lstrip("; "))]
    return [CheckResult(name, PASS, detail="no non-owner lane has a terminal, code, file-writing, browser or desktop "
                                           "tool")]


def check_guest_lanes() -> List[CheckResult]:
    """Could a contact run this machine? Hermes is asked what the platforms it serves run (``guest_lane_facts``);
    ``guests.executor_platforms`` in colonymind.yaml says which of them the operator opted in."""
    try:
        python = _hermes_python()
    except Exception:  # noqa: BLE001 - the config check reports what does not parse; this one quotes nothing
        return [CheckResult("guest-lanes", SKIP, detail="colonymind.yaml could not be read (see the config check)")]
    if python is None:
        return [CheckResult("guest-lanes", SKIP, detail="no Hermes interpreter found")]
    try:
        from colonymind.config import load_config
        config = load_config()
        opted = config.get("guests.executor_platforms") or []
        hermes_home = config.hermes_home
    except Exception:  # noqa: BLE001
        return [CheckResult("guest-lanes", SKIP, detail="colonymind.yaml could not be read (see the config check)")]
    return guest_lane_results(guest_lane_facts(python, hermes_home), opted)


def check_vector_jobs() -> CheckResult:
    """Report projection backlog from the ledger without changing or retrying jobs."""
    import sqlite3
    import time
    from contextlib import closing
    from colonymind import get_state_dir
    from colonymind.memory import curated
    path = get_state_dir() / "turn-idempotency.db"
    if not path.is_file():
        return CheckResult("vector-jobs", SKIP, detail="no source ledger")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"source_vector_jobs", "turn_sources"} <= tables:
            return CheckResult("vector-jobs", SKIP, detail="no source vector jobs")
        active = conn.execute("SELECT generation_id FROM vector_active WHERE slot=1").fetchone() if "vector_active" in tables else None
        now = time.time()
        pending = failed = 0
        for row in conn.execute("""SELECT j.turn_id,j.error,
            (j.status<>'complete' OR j.generation_id<>?) AND j.lease_until<=? AND j.next_attempt<=? AS claimable
            FROM source_vector_jobs j JOIN turn_sources s ON s.turn_id=j.turn_id""",
            (active[0] if active else "", now, now)):
            if curated.is_forgotten(conn, row['turn_id']):
                continue
            pending += bool(row['claimable'])
            failed += bool(row['error'])
    return CheckResult("vector-jobs", WARN if pending or failed else PASS,
                       detail=f"pending vector turns: {pending}; failed vector turns: {failed}")


LOCAL_CHECKS: List[Tuple[str, Callable[[], Any]]] = [
    ("vector-store", check_vector_store),
    ("vector-jobs", check_vector_jobs),
    ("config", check_config),
    ("api-key", check_api_key),
    ("identity", check_identity),
    ("llm-config", check_llm_config),
    ("recall-mode", check_recall_mode),
    ("recall-cutoff", check_recall_cutoff),
    ("instinct", check_instinct),
    ("instance-secret", check_instance_secret),
    ("hermes-version", check_hermes_version),
    ("adapter-version", check_adapter_version),
    ("build-stamp", check_build_stamp),
    ("pip-check", check_pip_check),
    ("hermes-keys", check_hermes_keys),
    ("worker-profile", check_worker_profile),
    ("plugin-loaded", check_plugin_loaded),
    ("owner-lanes", check_owner_lanes),
    ("guest-lanes", check_guest_lanes),
    ("service", check_service),
]


def run_local_checks() -> List[CheckResult]:
    results: List[CheckResult] = []
    for name, check in LOCAL_CHECKS:
        results += _run(name, check)
    return results


# ---------------------------------------------------------------------------
# Sidecar check (HTTP against the running sidecar)
# ---------------------------------------------------------------------------

def check_sidecar(base_url: str, api_key: str, timeout: float) -> List[CheckResult]:
    base_url = base_url.rstrip("/")
    try:
        status, body = _http_get(f"{base_url}/v1/host/health", api_key, timeout)
    except Exception as exc:  # noqa: BLE001 - URLError, OSError, timeouts
        return [CheckResult("sidecar", FAIL, detail=f"sidecar not reachable at {base_url}: {exc}",
                            remedy="start it with 'colonymind service start' (or 'colonymind start')")]
    if status != 200 or not isinstance(body, dict):
        return [CheckResult("sidecar", FAIL, detail=f"/v1/host/health returned HTTP {status}")]
    detail = f"sidecar at {base_url} reports status={body.get('status', 'unknown')}"
    problems = [str(item) for item in body.get("problems") or []]
    if problems:
        detail += ": " + "; ".join(problems)
    results = [CheckResult("sidecar", PASS if body.get("status") == "ok" else WARN, detail=detail)]
    results.append(_check_open_files((body.get("notes") or {}).get("fd_limit")))
    try:
        status, _ = _http_get(f"{base_url}/v1/mind/state", api_key, timeout)
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult("sidecar-auth", FAIL, detail=f"authenticated request failed: {exc}"))
        return results
    if status in (401, 403):
        results.append(CheckResult("sidecar-auth", FAIL, detail=f"the key was rejected (HTTP {status})",
                                   remedy="the running sidecar reads another api.key; restart it with "
                                          "'colonymind service restart'"))
    elif not 200 <= status < 300:
        results.append(CheckResult("sidecar-auth", WARN, detail=f"authenticated request returned HTTP {status}"))
    else:
        results.append(CheckResult("sidecar-auth", PASS,
                                   detail="authenticated request accepted" + ("" if api_key else " (no key: dev mode)")))
    return results


def _check_open_files(reported: Any) -> CheckResult:
    """The running sidecar's open-file limit, as its health reports it."""
    from colonymind.resources import OPEN_FILES
    text = str(reported or "").strip()
    if not text:
        return CheckResult("open-files", SKIP, detail="the sidecar did not report its open file limit")
    if text == "unlimited":
        return CheckResult("open-files", PASS, detail="open file limit unlimited")
    try:
        limit = int(text)
    except ValueError:
        return CheckResult("open-files", SKIP, detail=f"unreadable open file limit {text!r}")
    if limit < OPEN_FILES:
        return CheckResult("open-files", WARN,
                           detail=f"the sidecar runs with {limit} open files; the vector store wants {OPEN_FILES} "
                                  "under load",
                           remedy="raise the hard limit of the session the sidecar starts from (the generated "
                                  "service unit asks for it; 'colonymind service install' then 'colonymind "
                                  "service restart' apply it)")
    return CheckResult("open-files", PASS, detail=f"open file limit {limit}")


def check_semantic_recall(base_url: str, api_key: str, timeout: float) -> CheckResult:
    """When ``router.embed_url`` is set or the local embedder is, the running sidecar's embedder answers.

    A configured embedder that failed to initialise leaves the sidecar serving
    with keyword recall only; this check makes that a failure with the reason.
    """
    from colonymind.config import load_config
    config = load_config()
    local = config.get("router.embed_provider") == "local"
    if not local and not config.get("router.embed_url"):
        return CheckResult("semantic-recall", SKIP, detail="off: no router.embed_url in colonymind.yaml")
    if config.get("mind.faculties.semantic_recall") is False:
        return CheckResult("semantic-recall", SKIP, detail="off: mind.faculties.semantic_recall is false (the "
                                                           "recall-mode check says how to turn it on)")
    base_url = base_url.rstrip("/")
    try:
        status, body = _http_get(f"{base_url}/v1/host/embed/health", api_key, timeout)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("semantic-recall", FAIL, detail=f"embed/health not reachable: {exc}")
    if status != 200 or not isinstance(body, dict):
        return CheckResult("semantic-recall", FAIL, detail=f"/v1/host/embed/health returned HTTP {status}")
    if body.get("status") != "ok" and local:
        return CheckResult(
            "semantic-recall", FAIL,
            detail=f"the local embedder is set but not serving: {body.get('error') or 'unknown'}",
            remedy="the recall-mode check says whether its files and colonymind[vectors] are there (the sidecar log "
                   f"holds the first failure); {LOCAL_REFETCH}")
    if body.get("status") != "ok":
        return CheckResult(
            "semantic-recall", FAIL,
            detail=f"router.embed_url is set but the embedder is not serving: {body.get('error') or 'unknown'}",
            remedy="check router.embed_url, router.embed_model and router.embed_dims in colonymind.yaml against "
                   "the endpoint (the sidecar log holds the first failure), then 'colonymind service restart'")
    return CheckResult("semantic-recall", PASS,
                       detail=f"embedder serving (model={body.get('model') or 'unknown'}, dims={body.get('dims')}, "
                              f"{body.get('latency_ms', 0)} ms)")


# ---------------------------------------------------------------------------
# Engine entry point + reporting
# ---------------------------------------------------------------------------

def run_doctor(
    colonymind_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> List[CheckResult]:
    """Run every install check; never raises."""
    url = colonymind_url or default_colonymind_url()
    key = api_key if api_key is not None else default_api_key()
    results = run_local_checks()
    sidecar = _run("sidecar", check_sidecar, url, key, timeout)
    results += sidecar
    if sidecar and sidecar[0].status != FAIL:
        results += _run("semantic-recall", check_semantic_recall, url, key, timeout)
    return results


def default_api_key() -> str:
    from colonymind.config import read_api_key
    return read_api_key() or ""


def default_colonymind_url() -> str:
    """Resolve the sidecar URL the same way the other CLI commands do."""
    explicit = os.environ.get("COLONYMIND_URL") or os.environ.get("COLONYMIND_SIDECAR_URL")
    if explicit:
        return explicit
    from colonymind.config import load_config
    return load_config().sidecar_url


def summarize(results: List[CheckResult]) -> dict:
    counts = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def exit_code(results: List[CheckResult]) -> int:
    """0 when nothing failed (warns are OK), 1 otherwise."""
    return 1 if any(r.status == FAIL for r in results) else 0


def skip_dominated(results: List[CheckResult]) -> bool:
    """True when more checks skipped than passed; the run verified almost nothing."""
    counts = summarize(results)
    return counts[SKIP] > counts[PASS]


def results_to_json(results: List[CheckResult]) -> dict:
    counts = summarize(results)
    return {
        "results": [r.to_dict() for r in results],
        "summary": counts,
        "ok": exit_code(results) == 0 and not skip_dominated(results),
    }


_ICONS = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "skip"}
_COLORS = {PASS: "\033[92m", WARN: "\033[93m", FAIL: "\033[91m", SKIP: "\033[90m"}
_RESET = "\033[0m"


def format_report(results: List[CheckResult], colonymind_url: str = "", color: bool = True) -> str:
    """Human-readable report: aligned status lines, remedies indented."""
    lines: List[str] = []
    header = "ColonyMind doctor"
    if colonymind_url:
        header += f" ({colonymind_url})"
    lines.append(header)
    lines.append("")

    width = max((len(r.name) for r in results), default=0)
    for r in results:
        label = _ICONS.get(r.status, r.status)
        if color:
            label = f"{_COLORS.get(r.status, '')}{label}{_RESET}"
        line = f"  {label} {r.name.ljust(width)}"
        if r.detail:
            line += f"  {r.detail}"
        lines.append(line)
        if r.remedy and r.status in (WARN, FAIL):
            lines.append(f"       -> {r.remedy}")

    counts = summarize(results)
    lines.append("")
    lines.append(
        f"  {counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail, {counts[SKIP]} skip"
    )
    if counts[FAIL]:
        verdict = f"  {counts[FAIL]} check(s) failing; fix the remedies above"
    elif skip_dominated(results):
        verdict = f"  inconclusive: {counts[SKIP]} check(s) skipped, only {counts[PASS]} verified"
    elif counts[WARN]:
        verdict = f"  healthy with {counts[WARN]} warning(s)"
    else:
        verdict = "  all checks healthy"
    lines.append(verdict)
    return "\n".join(lines)
