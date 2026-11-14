"""``colonymind init``, ``colonymind upgrade`` and ``colonymind init --uninstall``.

Stock Hermes plus one package plus one command plus a restart::

    uv tool install --python 3.12 colonymind && colonymind init    && hermes gateway restart   # install
    uv tool upgrade colonymind               && colonymind upgrade && hermes gateway restart   # update

``hermes memory setup`` (pick colonymind-memory) is the other front door: the memory provider's
``post_setup`` runs this same ``init`` as a subprocess.

``init`` performs eight idempotent steps: it asks for the identity, the owner's
platforms and the autonomy level, writes ``colonymind.yaml``, ``identity.yaml``
and ``api.key``, points the router at the endpoint Hermes uses (and, with
consent, asks it for embeddings and rerank routes), offers the owner's own
Hermes history, installs the matching ``colonymind-hermes`` adapter into the
Python of the ``hermes`` executable and runs ``pip check`` there, writes the
Hermes configuration keys the adapter needs (saying which memory provider it
replaces), creates the ``colonymind-act`` worker profile and installs the
sidecar user service. It writes no Hermes admin lists. Nothing here restarts a
running gateway; the command to do that is printed at the end.
"""

from __future__ import annotations

import asyncio
import ast
import copy
import dataclasses
import datetime as _dt
import getpass
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

import yaml

from colonymind import __version__
from colonymind import config as configuration
from colonymind.config import (
    AUTONOMY_LEVELS,
    CONFIG_FILE,
    CONSTITUTION_CHARS,
    IDENTITY_FILE,
    KEY_FILE,
    LLM_CONFIG_FILE,
    Config,
    ConfigError,
    constitution_length,
    load_config,
    load_identity,
    read_api_key,
    render_constitution,
    save_config,
    save_identity,
    write_api_key,
)
from colonymind.api.credentials import (
    ANY_OWNER,
    ANY_PLATFORM,
    DEVELOPMENT_PRINCIPAL,
    INSTANCE_CREDENTIAL,
    Authority,
    CredentialTable,
    authority_of,
    authority_text,
    default_credentials,
    describe,
    lifetime_note,
    secret_digest,
)
from colonymind.util.temporal import now_utc

ADAPTER_DISTRIBUTION = "colonymind-hermes"
HERMES_DISTRIBUTION = "hermes-agent"
SUPPORTED_HERMES_MIN = (0, 21, 3)
SUPPORTED_HERMES_MAX_EXCLUSIVE = (0, 24, 0)
SUPPORTED_HERMES_RANGE = ">=0.21.3,<0.24"
WORKER_PROFILE = "colonymind-act"
PROTECTED_PATTERNS = (CONFIG_FILE, IDENTITY_FILE, KEY_FILE)
PLUGIN_NAME = "colonymind"
MEMORY_PROVIDER = "colonymind-memory"
SKILLS_DIR = "skills"
BACKUPS_DIR = "backups"
LEGACY_MANIFEST = "instance.json"
LEGACY_KEYRING = "api-keyring.json"
#: What init recorded about the Hermes it attached to: the memory provider it replaced (restored by
#: ``--uninstall``), the backup holding the replaced config, and the sidecar and adapter versions it installed
#: (the memory provider reads them, without the network, to tell a mismatched adapter); and the quiet hours it
#: last wrote to both the mind and identity.yaml (``current_quiet_hours``).
INSTALL_STATE = "hermes-install.json"

_MODEL_ROLES = ("small", "medium", "large")


class InitError(RuntimeError):
    """A step cannot proceed; nothing after it ran."""


def _say(message: str = "") -> None:
    print(message)


def _utc_stamp() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def _run(command: list[str], *, timeout: int = 900, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(part) for part in command], capture_output=True, text=True, timeout=timeout,
        env=env, check=False,
    )


# ---------------------------------------------------------------------------
# Hermes discovery
# ---------------------------------------------------------------------------

def parse_version(text: str) -> tuple[int, ...]:
    """Read the release, ignoring an optional local build stamp (PEP 440)."""
    match = re.fullmatch(
        r"v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\+[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*)?",
        (text or "").strip(),
    )
    return tuple(int(part or 0) for part in match.groups()) if match else (0, 0, 0)


def hermes_version_supported(version: str) -> bool:
    parsed = parse_version(version)
    return SUPPORTED_HERMES_MIN <= parsed < SUPPORTED_HERMES_MAX_EXCLUSIVE


def find_hermes_executable(explicit: str | None = None) -> Path | None:
    candidate = explicit or shutil.which("hermes")
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    return path if path.is_file() else None


def _installer_root(executable: Path) -> Path | None:
    """Read the installer's shell forwarders and literal bootstrap, never execute them."""
    seen: set[Path] = set()
    while executable not in seen:
        seen.add(executable)
        try:
            text = executable.read_text(encoding="utf-8")
            if not text.startswith("#!/bin/sh\nexec "):
                return None
            command = shlex.split(text.split("\n", 1)[1])
            if command[:1] != ["exec"]:
                return None
            if command[2:] == ["$@"]:
                executable = Path(command[1])
                continue
            if command[2:4] != ["-I", "-c"]:
                return None
            for node in ast.walk(ast.parse(command[4])):
                if (isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.path.insert"
                        and len(node.args) == 2 and ast.literal_eval(node.args[0]) == 0):
                    root = ast.literal_eval(node.args[1])
                    if isinstance(root, str) and Path(root).is_absolute():
                        return Path(root).resolve()
        except (OSError, UnicodeError, ValueError, SyntaxError, IndexError):
            return None
        return None
    return None


def _recorded_dependency_python(root: Path, home: Path) -> Path | None:
    # pm.environments.install_key / _recorded_venv: the completion record, not
    # the store interpreter embedded in the launcher, selects dependencies.
    if home.parent.name == "profiles":
        home = home.parent.parent
    key = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    state = home / "installs" / key
    try:
        data = json.loads((state / "facts.json").read_text(encoding="utf-8-sig"))
        value = data.get("packages", {}).get("venv", {}).get("environment")
        if value is None:
            return None
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("invalid environment path")
        environment = Path(value).resolve()
        if (not environment.is_relative_to((state / "environments").resolve())
                or not (environment / "pyvenv.cfg").is_file()):
            raise ValueError("missing environment or outside this home")
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise ValueError("missing Python")
        return python
    except FileNotFoundError:
        return None
    except (OSError, ValueError, AttributeError) as exc:
        raise InitError(f"cannot read Hermes dependency environment in {state}: {exc}") from exc


def installer_python(executable: Path, home: Path | None = None, *, provision: bool = False) -> Path | None:
    """The interpreter of the dependency environment Hermes' installer recorded for ``home``. None when
    ``executable`` is no installer launcher, and when nothing is recorded and ``provision`` is off: only
    ``init`` and ``upgrade`` let Hermes build the environment of a new home; every other command reads the
    record and goes on without it."""
    root = _installer_root(executable)
    if root is None:
        return None
    home = home or resolve_hermes_home(None, None)
    python = _recorded_dependency_python(root, home)
    if python is None and provision:
        # A new home has no selection yet. Hermes' environment query provisions
        # it through its own bootstrap, then we read the same completion record.
        env = dict(os.environ, HERMES_HOME=str(home))
        result = _run([str(executable), "--run-module", "pm.environments", "--format", "json"],
                      timeout=900, env=env)
        if result.returncode == 0:
            python = _recorded_dependency_python(root, home)
        if python is None:
            raise InitError(f"no Hermes dependency environment recorded for {home}; "
                            "run 'hermes --version' in that home and retry")
    return python


def python_of_executable(executable: Path) -> Path | None:
    """The interpreter a console script runs with, read from its shebang."""
    try:
        with open(executable, "rb") as stream:
            first = stream.readline(512)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    line = first[2:].decode("utf-8", "replace").strip()
    if not line:
        return None
    parts = line.split()
    if Path(parts[0]).name in {"sh", "bash", "dash"}:
        return installer_python(executable)
    # ``#!/usr/bin/env python`` resolves through PATH; everything else is a path.
    if parts[0].endswith("/env") and len(parts) > 1:
        found = shutil.which(parts[1])
        return Path(found) if found else None
    # pipx and uv write ``#!<venv>/bin/python`` (optionally with flags).
    path = Path(parts[0])
    return path if path.exists() else None


def hermes_python_version(python: Path) -> str | None:
    """Read package metadata, or Hermes' own install stamp for placeholder metadata."""
    result = _run([python, "-I", "-c",
                   "import importlib.metadata as m; print(m.version('hermes-agent'))"], timeout=60)
    if result.returncode != 0:
        return None
    version = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None
    if version != "0.0.0":
        return version
    # PM's venv imports a workspace snapshot without the checkout's stamp.
    # Its inputs/.project-root records the checkout that owns that environment
    # (pm.environments.record_activation_inputs). Do not search other homes.
    stamp = _run([python, "-I", "-c", """
import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path
root = Path(importlib.util.find_spec('hermes_cli').origin).resolve().parents[1]
prefix = Path(sys.prefix).resolve()
if (prefix.name == 'venv' and prefix.parent.parent.name == 'environments'
        and prefix.parent.parent.parent.parent.name == 'installs'):
    state = prefix.parent.parent.parent
    root = Path((state / 'inputs' / '.project-root').read_text(encoding='utf-8')).resolve()
    if hashlib.sha256(str(root).encode('utf-8')).hexdigest()[:16] != state.name:
        raise ValueError('Hermes checkout does not match its install key')
data = json.loads((root / 'install-stamp.json').read_text(encoding='utf-8-sig'))
# Mirror version_info._stamp_version_info's provenance requirements. Reading
# this exact path never consults the caller's HERMES_INSTALL_ROOT override.
commit = data.get('commit')
if not isinstance(commit, str) or not commit or set(commit) == {'0'}:
    raise ValueError('missing stamp commit')
if data.get('updateMechanism') not in ('self', 'app-installer', 'electron-updater', 'external', 'microsoft-store'):
    raise ValueError('missing or invalid updateMechanism')
if data.get('payload') == 'light':
    raise ValueError('light payload has no Python runtime')
if data.get('source') == 'git' and (root / '.git').exists():
    live = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, timeout=30, check=False)
    if live.returncode != 0 or live.stdout.strip() != commit:
        raise ValueError('stamp does not match the live checkout')
print(data.get('baseVersion') or '0.0.0')
"""], timeout=60)
    version = stamp.stdout.strip().splitlines()[-1] if stamp.stdout.strip() else None
    source = f"Hermes install-stamp.json baseVersion (hermes-agent metadata 0.0.0) in {python}"
    if stamp.returncode != 0 or not version or parse_version(version) == (0, 0, 0):
        raise InitError(f"no version could be read from Hermes' install stamp through {python}; "
                        f"supported range {SUPPORTED_HERMES_RANGE}")
    if not hermes_version_supported(version):
        raise InitError(f"Hermes {version} from {source} is outside the supported range {SUPPORTED_HERMES_RANGE}")
    return version


def hermes_python_candidates(explicit: str | None, configured: str | None,
                             home: Path | None = None, *, provision: bool = False) -> list[Path]:
    """Explicit overrides win; a managed launcher's current selection beats a saved generation. A managed
    launcher with nothing recorded for ``home`` leaves the saved interpreter only (``provision`` off): the
    default home's record is not asked in its place."""
    if explicit:
        return [Path(explicit).expanduser()]
    executable = find_hermes_executable()
    managed = installer_python(executable, home, provision=provision) if executable else None
    if managed is not None:
        return [managed]
    candidates = [Path(configured).expanduser()] if configured else []
    if executable is not None and _installer_root(executable) is None:
        python = python_of_executable(executable)
        if python is not None:
            candidates.append(python)
    return candidates


def resolve_hermes_python(explicit: str | None, configured: str | None,
                          home: Path | None = None) -> tuple[Path, str]:
    """Find the Python that runs ``hermes`` and check its version. This is ``init`` and ``upgrade``: the two
    commands that may have Hermes provision the dependency environment of a new home."""
    candidates = hermes_python_candidates(explicit, configured, home, provision=True)
    for python in candidates:
        if not python.exists():
            continue
        version = hermes_python_version(python)
        if not version:
            continue
        if not hermes_version_supported(version):
            raise InitError(
                f"Hermes {version} in {python} is outside the supported range {SUPPORTED_HERMES_RANGE}"
            )
        return python, version
    raise InitError(
        "no Hermes installation found: put the 'hermes' executable on PATH, or pass "
        "--hermes-python with the interpreter that runs it"
    )


def resolve_hermes_home(explicit: str | None, configured: str | None) -> Path:
    selected = explicit or os.environ.get("HERMES_HOME") or configured or "~/.hermes"
    return Path(selected).expanduser()


def profiles_root(hermes_home: Path) -> Path:
    """Named profiles live under the Hermes root even when the home is itself a profile."""
    if hermes_home.parent.name == "profiles":
        return hermes_home.parent
    return hermes_home / "profiles"


# ---------------------------------------------------------------------------
# Adapter install in Hermes' environment
# ---------------------------------------------------------------------------

def installed_adapter_version(python: Path) -> str | None:
    result = _run([python, "-I", "-c",
                   f"import importlib.metadata as m; print(m.version('{ADAPTER_DISTRIBUTION}'))"], timeout=60)
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None


def _package_manager(python: Path) -> Callable[..., list[str]]:
    """A command builder: ``pip`` inside *python* when it has one, else ``uv pip`` targeting it."""
    if _run([python, "-m", "pip", "--version"], timeout=60).returncode == 0:
        return lambda verb, *args: [str(python), "-m", "pip", verb, *args]
    uv = shutil.which("uv")
    if uv:
        return lambda verb, *args: [uv, "pip", verb, "--python", str(python), *args]
    raise InitError(
        f"{python} has no pip and 'uv' is not on PATH; install one of them to manage Hermes' environment"
    )


def install_adapter(python: Path, version: str, *, source: str | None = None) -> bool:
    """Install ``colonymind-hermes==version`` (or *source*) into *python*; True when it changed."""
    current = installed_adapter_version(python)
    if source is None and current == version:
        return False
    manager = _package_manager(python)
    requirement = source or f"{ADAPTER_DISTRIBUTION}=={version}"
    commands = [manager("install", "--quiet", requirement)]
    if source is not None:
        # A wheel of the same version must still replace what is installed; do
        # that without touching dependencies, which Hermes pins exactly.
        commands.insert(0, manager("install", "--quiet", "--no-deps", "--force-reinstall", requirement))
    for command in commands:
        result = _run(command, timeout=900)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise InitError("adapter install failed in Hermes' environment: "
                            + (detail[-1] if detail else "unknown error"))
    return source is not None or installed_adapter_version(python) != current


def uninstall_adapter(python: Path) -> bool:
    if installed_adapter_version(python) is None:
        return False
    manager = _package_manager(python)
    command = manager("uninstall", ADAPTER_DISTRIBUTION)
    if command[1] == "-m":  # pip asks for confirmation; uv does not
        command.insert(4, "--yes")
    result = _run(command, timeout=300)
    return result.returncode == 0


def pip_check(python: Path) -> tuple[bool, str]:
    result = _run(_package_manager(python)("check"), timeout=300)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def adapter_dependency_names(python: Path) -> set[str] | None:
    """Packages the adapter brings into Hermes, including active transitive requirements."""
    result = _run([python, "-I", "-c", """
import importlib.metadata as m
import json
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
pending = ['colonymind-hermes']
seen = set()
while pending:
    name = canonicalize_name(pending.pop())
    if name in seen:
        continue
    seen.add(name)
    for text in m.requires(name) or []:
        requirement = Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
            pending.append(requirement.name)
print(json.dumps(sorted(seen)))
"""], timeout=60)
    try:
        return set(json.loads(result.stdout)) if result.returncode == 0 else None
    except (ValueError, TypeError):
        return None


def dependency_conflict_detail(python: Path, output: str) -> tuple[bool, str]:
    """Return whether a failed check is entirely Hermes-owned, and its conflict lines.

    Unknown diagnostics or unavailable metadata stay failures. Recognize both
    pip and uv's requirement diagnostics rather than mistaking uv's banner for
    the conflict or treating a dependent of one of our packages as unrelated.
    """
    ours = adapter_dependency_names(python)
    lines = [line.strip() for line in output.splitlines() if line.strip()
             and not re.match(r"^(Using Python |Checked \d+ packages? |Found \d+ incompatibilit)", line)]
    unrelated = bool(lines) and ours is not None
    for line in lines:
        match = re.match(r"The package `([^`]+)` requires `([A-Za-z0-9_.-]+)", line)
        if match is None:
            match = re.match(r"([A-Za-z0-9_.-]+) \S+ (?:has requirement|requires) ([A-Za-z0-9_.-]+)", line)
        names = {re.sub(r"[-_.]+", "-", name).lower() for name in match.groups()} if match else set()
        if not names or ours is None or names & ours:
            unrelated = False
    return unrelated, "\n".join(lines) or "dependency check failed"


VECTOR_STORE_MODULE = "lancedb"


def vector_store_available() -> bool:
    """The vector store library imports in this interpreter (the sidecar's own).

    It is a base dependency, so a missing module means a broken install, and
    a sidecar started without it would serve keyword recall only.
    """
    import importlib.util
    try:
        return importlib.util.find_spec(VECTOR_STORE_MODULE) is not None
    except (ImportError, ValueError):
        return False


VECTOR_STORE_REMEDY = ("the vector store library (lancedb) is not importable in this interpreter; reinstall the "
                       "sidecar: uv tool install --force --python 3.12 colonymind (or pipx install --force colonymind)")


def colonymind_in_hermes_environment(python: Path) -> bool:
    result = _run([python, "-I", "-c",
                   "import importlib.metadata as m; m.version('colonymind')"], timeout=60)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Hermes configuration
# ---------------------------------------------------------------------------

def read_hermes_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        from colonymind.config import _yaml_problem
        raise InitError(f"{path} is not valid YAML: {_yaml_problem(exc)}") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise InitError(f"{path} must contain a YAML mapping")
    return loaded


def read_install_state(home: Path) -> dict[str, Any]:
    """``hermes-install.json`` in the instance directory; {} when absent or unreadable."""
    try:
        loaded = json.loads((home / INSTALL_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_install_state(home: Path, **fields: Any) -> None:
    """Merge ``fields`` into ``hermes-install.json`` (mode 600); a field given as None is removed."""
    state = read_install_state(home)
    for key, value in fields.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    configuration._atomic_write(home / INSTALL_STATE,
                                (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8"), mode=0o600)


def selected_memory_provider(config: dict[str, Any]) -> str:
    """The ``memory.provider`` a Hermes config selects, or ""."""
    memory = config.get("memory")
    return str(memory.get("provider") or "").strip() if isinstance(memory, dict) else ""


def _mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if value is None:
        value = parent[key] = {}
    if not isinstance(value, dict):
        raise InitError(f"Hermes config key '{key}' must be a mapping")
    return value


def _string_list(parent: dict, key: str) -> list:
    value = parent.get(key)
    if value is None:
        value = parent[key] = []
    if not isinstance(value, list):
        raise InitError(f"Hermes config key '{key}' must be a list")
    return value


# The retired MCP integration: what 'colonymind mcp setup' wrote before the prune. Its server command
# ('colonymind mcp') is gone, so a registration makes its harness launch a command that fails, and the
# diagnostic skill it installed prescribes that command and tools that no longer exist.
MCP_SERVER = "colonymind"
MCP_GENERATED_ENV = frozenset({"COLONYMIND_URL", "COLONYMIND_MCP_CONTACT_ID", "COLONYMIND_MCP_SOURCE"})
MCP_RETIRED = "the retired ColonyMind MCP server ('colonymind mcp' no longer exists)"
# Where the setup registered the server in the harnesses other than the Hermes config ``init`` reconciles
# (its Hermes entry always went to ~/.hermes), and the key that holds the servers there.
MCP_HARNESS_CONFIGS = (("~/.claude.json", "mcpServers"), ("~/.codex/config.toml", "mcp_servers"),
                       ("~/.crush.json", "mcp"), ("~/.config/opencode/opencode.json", "mcp"),
                       ("~/.hermes/config.yaml", "mcp_servers"))
DIAGNOSTIC_SKILL_DIRS = ("~/.codex/skills/colonymind-diagnose", "~/.config/crush/skills/colonymind-diagnose",
                         "~/.config/opencode/skills/colonymind-diagnose")
# SHA-256 of the SKILL.md the setup wrote: a copy with these exact bytes was never edited.
DIAGNOSTIC_SKILL_SHA256 = frozenset({"b8b4b7c752e2044228c44827c8a40a3078e6ad3cb8439c0a695e300615e0560a"})


def generated_mcp_registration(entry: Any) -> bool:
    """Whether ``entry`` is what 'colonymind mcp setup' wrote: 'colonymind mcp' with the generated
    environment and nothing the owner added (a literal credential, a custom command, another key)."""
    if not isinstance(entry, dict) or set(entry) - {"command", "args", "env", "type"}:
        return False
    env = entry.get("env", {})
    return (entry.get("command") == "colonymind" and entry.get("args") == ["mcp"]
            and entry.get("type", "stdio") == "stdio" and isinstance(env, dict)
            and not set(env) - MCP_GENERATED_ENV)


def _drop_generated_mcp_registration(config: dict[str, Any]) -> list[str]:
    servers = config.get("mcp_servers")
    if not isinstance(servers, dict) or not generated_mcp_registration(servers.get(MCP_SERVER)):
        return []
    del servers[MCP_SERVER]
    if not servers:
        del config["mcp_servers"]
    return [f"mcp_servers.colonymind removed: {MCP_RETIRED}"]


def retired_mcp_left(hermes_config: Path) -> list[str]:
    """What the retired MCP integration left that no upgrade changes, in words: a registration in the
    Hermes config edited after the setup wrote it, and one in another harness's config (those files
    belong to applications that may be running). Each still launches 'colonymind mcp'."""
    notes: list[str] = []
    try:
        servers = read_hermes_config(hermes_config).get("mcp_servers")
    except (InitError, OSError, yaml.YAMLError):
        servers = None
    if isinstance(servers, dict) and MCP_SERVER in servers:
        notes.append(f"kept mcp_servers.colonymind in {hermes_config}: it differs from what 'colonymind mcp "
                     f"setup' wrote; remove it if it still launches {MCP_RETIRED}")
    for raw, key in MCP_HARNESS_CONFIGS:
        path = Path(raw).expanduser()
        if not path.is_file() or path.resolve() == hermes_config.resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            data = (tomllib.loads(text) if path.suffix == ".toml" else yaml.safe_load(text) if path.suffix == ".yaml"
                    else json.loads(text))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        servers = data.get(key) if isinstance(data, dict) else None
        if isinstance(servers, dict) and MCP_SERVER in servers:
            notes.append(f"{path} registers the retired ColonyMind MCP server ({key}.colonymind); remove that "
                         "entry: 'colonymind mcp' no longer exists")
    return notes


def diagnostic_skills() -> tuple[list[Path], list[str]]:
    """The diagnostic skills the setup installed: the unedited copies (an upgrade moves them into its
    backup) and, in words, the edited ones, which stay."""
    unedited: list[Path] = []
    edited: list[str] = []
    for raw in DIAGNOSTIC_SKILL_DIRS:
        directory = Path(raw).expanduser()
        skill = directory / "SKILL.md"
        try:
            if not skill.is_file():
                continue
            if ([path.name for path in directory.iterdir()] == ["SKILL.md"]
                    and hashlib.sha256(skill.read_bytes()).hexdigest() in DIAGNOSTIC_SKILL_SHA256):
                unedited.append(directory)
                continue
        except OSError:
            continue
        edited.append(f"kept {directory}: edited since 'colonymind mcp setup' installed it, and it prescribes "
                      f"{MCP_RETIRED} and its tools; remove it yourself")
    return unedited, edited


def retire_diagnostic_skills(backup_dir: Path) -> list[str]:
    """Move the unedited diagnostic skills into ``backup_dir/harness-skills`` (never delete them)."""
    notes: list[str] = []
    user_home = Path("~").expanduser()
    for directory in diagnostic_skills()[0]:
        destination = backup_dir / "harness-skills" / directory.relative_to(user_home)
        from colonymind.backup import _make_tree_private, _private_directory
        _private_directory(destination.parent)
        shutil.move(str(directory), str(destination))
        _make_tree_private(destination)
        notes.append(f"retired the diagnostic skill {directory} (moved to {destination}): it prescribes "
                     f"{MCP_RETIRED}")
    return notes


def reconcile_hermes_config(config: dict[str, Any], *, sidecar_url: str, key_file: Path,
                            skills_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Return the config with every key the adapter needs, and the changes made."""
    result = copy.deepcopy(config)
    changes: list[str] = _drop_generated_mcp_registration(result)
    plugins = _mapping(result, "plugins")
    enabled = _string_list(plugins, "enabled")
    if PLUGIN_NAME not in enabled:
        enabled.append(PLUGIN_NAME)
        changes.append("plugins.enabled += colonymind")
    disabled = plugins.get("disabled")
    if isinstance(disabled, list) and PLUGIN_NAME in disabled:
        disabled.remove(PLUGIN_NAME)
        changes.append("plugins.disabled -= colonymind")
    if plugins.get("hook_callback_timeout") != 0:
        plugins["hook_callback_timeout"] = 0
        changes.append("plugins.hook_callback_timeout: 0")
    settings = _mapping(plugins, PLUGIN_NAME)
    if settings.get("sidecar_url") != sidecar_url:
        settings["sidecar_url"] = sidecar_url
        changes.append("plugins.colonymind.sidecar_url")
    if settings.get("key_file") != str(key_file):
        settings["key_file"] = str(key_file)
        changes.append("plugins.colonymind.key_file")
    memory = _mapping(result, "memory")
    if memory.get("provider") != MEMORY_PROVIDER:
        memory["provider"] = MEMORY_PROVIDER
        changes.append("memory.provider: colonymind-memory")
    kanban = _mapping(result, "kanban")
    if kanban.get("dispatch_in_gateway") is not True:
        kanban["dispatch_in_gateway"] = True
        changes.append("kanban.dispatch_in_gateway: true")
    skills = _mapping(result, "skills")
    external = _string_list(skills, "external_dirs")
    if str(skills_dir) not in external:
        external.append(str(skills_dir))
        changes.append("skills.external_dirs += " + str(skills_dir))
    security = _mapping(result, "security")
    patterns = _string_list(security, "protected_instruction_extra_patterns")
    for pattern in PROTECTED_PATTERNS:
        if pattern not in patterns:
            patterns.append(pattern)
            changes.append("security.protected_instruction_extra_patterns += " + pattern)
    return result, changes


def strip_hermes_config(config: dict[str, Any], *, skills_dir: Path | None,
                        restore_provider: str = "") -> tuple[dict[str, Any], list[str]]:
    """Remove every key ``init`` wrote, and the retired MCP server's generated entry; the rest of the
    file is untouched. ``memory.provider`` goes back to ``restore_provider`` (the one init replaced) while
    it still names ColonyMind's provider; a provider chosen since is left as it is."""
    result = copy.deepcopy(config)
    changes: list[str] = _drop_generated_mcp_registration(result)
    plugins = result.get("plugins")
    if isinstance(plugins, dict):
        enabled = plugins.get("enabled")
        if isinstance(enabled, list) and PLUGIN_NAME in enabled:
            enabled.remove(PLUGIN_NAME)
            changes.append("plugins.enabled -= colonymind")
        if PLUGIN_NAME in plugins:
            del plugins[PLUGIN_NAME]
            changes.append("plugins.colonymind removed")
        if plugins.get("hook_callback_timeout") == 0:
            del plugins["hook_callback_timeout"]
            changes.append("plugins.hook_callback_timeout removed")
        for key in ("enabled",):
            if plugins.get(key) == []:
                del plugins[key]
        if not plugins:
            del result["plugins"]
    memory = result.get("memory")
    if isinstance(memory, dict) and memory.get("provider") == MEMORY_PROVIDER and restore_provider:
        memory["provider"] = restore_provider
        changes.append(f"memory.provider restored to {restore_provider}")
    elif isinstance(memory, dict) and memory.get("provider") == MEMORY_PROVIDER:
        del memory["provider"]
        changes.append("memory.provider removed")
        if not memory:
            del result["memory"]
    kanban = result.get("kanban")
    if isinstance(kanban, dict) and "dispatch_in_gateway" in kanban:
        del kanban["dispatch_in_gateway"]
        changes.append("kanban.dispatch_in_gateway removed")
        if not kanban:
            del result["kanban"]
    skills = result.get("skills")
    if isinstance(skills, dict):
        external = skills.get("external_dirs")
        if isinstance(external, list) and skills_dir is not None and str(skills_dir) in external:
            external.remove(str(skills_dir))
            changes.append("skills.external_dirs -= " + str(skills_dir))
            if not external:
                del skills["external_dirs"]
        if not skills:
            del result["skills"]
    security = result.get("security")
    if isinstance(security, dict):
        patterns = security.get("protected_instruction_extra_patterns")
        if isinstance(patterns, list):
            kept = [pattern for pattern in patterns if pattern not in PROTECTED_PATTERNS]
            if kept != patterns:
                changes.append("security.protected_instruction_extra_patterns trimmed")
                if kept:
                    security["protected_instruction_extra_patterns"] = kept
                else:
                    del security["protected_instruction_extra_patterns"]
        if not security:
            del result["security"]
    return result, changes


def write_hermes_config(path: Path, config: dict[str, Any], *, backup_dir: Path) -> Path | None:
    """Atomically replace the Hermes config, keeping the previous bytes in *backup_dir*; returns that copy
    (None when there was no file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise InitError(f"{path} is a symlink; point it at a regular file first")
    backup = None
    if path.is_file():
        from colonymind.backup import _copy_private, _private_directory
        _private_directory(backup_dir)
        backup = backup_dir / f"hermes-config-{_utc_stamp()}.yaml"
        _copy_private(path, backup)
    mode = path.stat().st_mode & 0o777 if path.is_file() else 0o600
    temporary = path.with_name(f".{path.name}.colonymind-{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


# ---------------------------------------------------------------------------
# Worker profile
# ---------------------------------------------------------------------------

def worker_request_fields(cfg: Config) -> dict[str, Any]:
    """``mind.worker_request`` without its nulls: the fields every worker request carries."""
    fields = cfg.get("mind.worker_request") or {}
    return {str(key): copy.deepcopy(value) for key, value in fields.items() if value is not None} \
        if isinstance(fields, dict) else {}


def _cap_worker_requests(profile: dict[str, Any], fields: dict[str, Any]) -> None:
    """Put ``fields`` into the ``extra_body`` of every custom endpoint the worker may use.

    Stock Hermes sends a custom endpoint's ``extra_body`` with each request (the named provider's
    runtime, and the agent's own merge by base URL for ``provider: custom``); it sends no output
    limit otherwise, so one runaway completion could hold the model for a run's whole budget. A
    model or fallback given as a bare ``base_url`` gets an entry of its own for that. The worker's
    fields win over the provider's; a field set to null in ``colonymind.yaml`` is left to the provider.
    """
    if not fields:
        return
    providers = profile.get("providers")
    for entry in providers.values() if isinstance(providers, dict) else ():
        if isinstance(entry, dict):
            entry["extra_body"] = {**(entry.get("extra_body") if isinstance(entry.get("extra_body"), dict) else {}),
                                   **copy.deepcopy(fields)}
    listed = profile.get("custom_providers")
    for entry in listed if isinstance(listed, list) else ():
        if isinstance(entry, dict):
            entry["extra_body"] = {**(entry.get("extra_body") if isinstance(entry.get("extra_body"), dict) else {}),
                                   **copy.deepcopy(fields)}
    urls: list[str] = []
    for key in ("model", "fallback_model", "fallback_providers"):
        value = profile.get(key)
        for source in value if isinstance(value, list) else [value]:
            url = str(source.get("base_url") or "").strip() if isinstance(source, dict) else ""
            if url and url not in urls:
                urls.append(url)
    if urls and (listed is None or isinstance(listed, list)):
        profile["custom_providers"] = [*(listed or []), *(
            {"name": WORKER_PROFILE if number == 1 else f"{WORKER_PROFILE}-{number}", "base_url": url,
             "extra_body": copy.deepcopy(fields)} for number, url in enumerate(urls, 1))]


def worker_model(main_config: dict[str, Any], name: str) -> dict[str, Any]:
    """The worker's ``model`` block for ``mind.worker_model``: the main config's ``providers`` entry (or
    ``custom_providers`` entry) of that name, with its default model, address and context length."""
    providers = main_config.get("providers")
    entry = providers.get(name) if isinstance(providers, dict) else None
    default = entry.get("default_model") if isinstance(entry, dict) else None
    if not isinstance(entry, dict):
        listed = main_config.get("custom_providers")
        entry = next((item for item in listed if isinstance(item, dict) and item.get("name") == name), None) \
            if isinstance(listed, list) else None
        default = entry.get("model") if isinstance(entry, dict) else None
    if not isinstance(entry, dict):
        raise InitError(f"mind.worker_model names {name!r}, which is not a providers or custom_providers entry "
                        "of the main Hermes config")
    models = entry.get("models")
    if not default and isinstance(models, dict) and models:
        default = next(iter(models))
    if not isinstance(default, str) or not default.strip():
        raise InitError(f"mind.worker_model names {name!r}, whose entry has no default model")
    block: dict[str, Any] = {"provider": name, "default": default}
    base_url = entry.get("api") or entry.get("url") or entry.get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        block["base_url"] = base_url
    if entry.get("context_length"):
        block["context_length"] = entry["context_length"]
    return block


def worker_profile_config(main_config: dict[str, Any], cfg: Config, *, sidecar_url: str,
                          key_file: Path) -> dict[str, Any]:
    """The ``colonymind-act`` profile: the main model and its providers, or the model ``mind.worker_model``
    names, capped per request (``mind.worker_request``), the mind toolsets, the deny list and its own
    gateway."""
    profile: dict[str, Any] = {}
    # The model and every provider entry it or its fallbacks can name: a profile reads only its own config.
    for key in ("model", "providers", "custom_providers", "fallback_providers", "fallback_model"):
        if main_config.get(key):
            profile[key] = copy.deepcopy(main_config[key])
    named = cfg.get("mind.worker_model")
    if named:
        # The named entry instead of the main model, and none of the main model's fallbacks. Both fallback keys
        # are written empty rather than left out, so no fallback an earlier profile had is kept on the rewrite.
        profile["model"] = worker_model(main_config, named)
        profile["fallback_providers"] = []
        profile["fallback_model"] = []
    _cap_worker_requests(profile, worker_request_fields(cfg))
    # The dispatcher pins a worker's tools from ``platform_toolsets.cli``; a top-level ``toolsets`` is ignored.
    profile["platform_toolsets"] = {"cli": list(cfg.get("mind.worker_toolsets") or [])}
    profile["approvals"] = {"deny": list(cfg.get("mind.deny.commands") or [])}
    profile["memory"] = {"provider": MEMORY_PROVIDER}
    profile["plugins"] = {
        "enabled": [PLUGIN_NAME],
        "hook_callback_timeout": 0,
        PLUGIN_NAME: {"sidecar_url": sidecar_url, "key_file": str(key_file)},
    }
    profile["security"] = {"protected_instruction_extra_patterns": list(PROTECTED_PATTERNS)}
    # Hermes 0.21.7+ multiplexes every named profile into the host gateway unless the profile's own
    # config opts out; the worker profile is a dispatch target only, so it always opts out.
    profile["gateway"] = {"standalone": True}
    return profile


# Run under the Hermes interpreter: the dispatcher's own toolset pin and provider ladder, with the
# main home's .env loaded because a dispatched worker inherits the gateway's environment.
_WORKER_PROBE = """
import json, sys
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=sys.argv[1])
from hermes_cli.kanban_db_dispatch import _resolve_worker_cli_toolsets
from hermes_cli.runtime_provider import resolve_runtime_provider
out = {"toolsets": _resolve_worker_cli_toolsets(sys.argv[2]) or []}
try:
    runtime = resolve_runtime_provider()
    out.update(provider=runtime.get("provider"), base_url=runtime.get("base_url"),
               api_key=bool(runtime.get("api_key")))
except Exception as error:
    out["error"] = f"{type(error).__name__}: {error}"
print(json.dumps(out))
"""


def resolve_worker_profile(python: Path, hermes_home: Path) -> dict[str, Any]:
    """What stock Hermes gives a dispatched ``colonymind-act`` worker: its pinned toolsets and its model
    (``provider``, ``base_url``, whether it has a key), or ``error`` when the model does not resolve."""
    profile_home = profiles_root(hermes_home) / WORKER_PROFILE
    try:
        result = subprocess.run([str(python), "-c", _WORKER_PROBE, str(hermes_home), str(profile_home)],
                                capture_output=True, text=True, timeout=120, check=False,
                                env={**os.environ, "HERMES_HOME": str(profile_home)})
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"toolsets": [], "error": f"could not ask Hermes ({type(error).__name__})"}


def render_worker_profile(profile: dict[str, Any]) -> str:
    return yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)


def read_worker_profile(root: Path) -> dict[str, Any]:
    """``profiles/colonymind-act/config.yaml`` as a mapping; {} when it is missing or not a YAML mapping."""
    try:
        loaded = yaml.safe_load((root / WORKER_PROFILE / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def keep_unowned_keys(generated: dict[str, Any], current: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """*generated* plus every key that only *current* (the profile on disk) has, and the dotted paths of those
    keys' values (paths only, never values).

    The worker profile is ColonyMind's to write, but a key someone added by hand is not: it is kept. Where both
    have a key the generated value wins; two mappings are merged key by key, and anything else, a list
    included, is the generated value whole.
    Known limit: nothing tells a hand-added key from one the main config or ColonyMind no longer writes, so a
    key later removed from the main config stays in the worker profile until it is removed there by hand.
    """
    merged = copy.deepcopy(generated)
    kept: list[str] = []

    def leaves(value: Any, path: str) -> list[str]:
        if isinstance(value, dict) and value:
            return [leaf for key, item in value.items() for leaf in leaves(item, f"{path}.{key}")]
        return [path]

    def walk(target: dict[str, Any], existing: dict[str, Any], prefix: str) -> None:
        for key, value in existing.items():
            path = f"{prefix}{key}"
            if key not in target:
                target[key] = copy.deepcopy(value)
                kept.extend(leaves(value, path))
            elif isinstance(target[key], dict) and isinstance(value, dict):
                walk(target[key], value, path + ".")

    walk(merged, current if isinstance(current, dict) else {}, "")
    return merged, kept


def merged_worker_profile(root: Path, profile: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The profile as it is written over the one on disk (``keep_unowned_keys``), and the paths kept."""
    return keep_unowned_keys(profile, read_worker_profile(root))


def worker_profile_pending(root: Path, profile: dict[str, Any]) -> bool:
    """True when ``profiles/colonymind-act/config.yaml`` is missing or differs from *profile* with the keys only
    the file has kept (``keep_unowned_keys``)."""
    config_path = root / WORKER_PROFILE / "config.yaml"
    try:
        return config_path.read_text(encoding="utf-8") != render_worker_profile(merged_worker_profile(root, profile)[0])
    except OSError:
        return True


def write_worker_profile(root: Path, profile: dict[str, Any]) -> bool:
    """Create or update ``profiles/colonymind-act``, keeping the keys only the file on disk has
    (``keep_unowned_keys``); True when something changed."""
    profile = merged_worker_profile(root, profile)[0]
    directory = root / WORKER_PROFILE
    tombstone = root / ".deleted" / WORKER_PROFILE
    changed = False
    if tombstone.exists():
        tombstone.unlink()
        changed = True
    for name in ("memories", "sessions", "skills", "logs", "workspace"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.yaml"
    content = render_worker_profile(profile)
    if not config_path.is_file() or config_path.read_text(encoding="utf-8") != content:
        config_path.write_text(content, encoding="utf-8")
        os.chmod(config_path, 0o600)
        changed = True
    env_path = directory / ".env"
    if not env_path.exists():
        env_path.write_text("# Profile-scoped credentials; the model's key is inherited from the main profile.\n",
                            encoding="utf-8")
        os.chmod(env_path, 0o600)
        changed = True
    return changed


def remove_worker_profile(root: Path, *, backup_dir: Path) -> bool:
    """Move ``profiles/colonymind-act`` (its sessions, memories and logs included) into
    ``backup_dir/profiles`` and leave Hermes' tombstone; True when the profile existed."""
    directory = root / WORKER_PROFILE
    if not directory.exists():
        return False
    destination = backup_dir / "profiles" / WORKER_PROFILE
    from colonymind.backup import _make_tree_private, _private_directory
    _private_directory(destination.parent)
    shutil.move(str(directory), str(destination))
    _make_tree_private(destination)
    tombstone = root / ".deleted" / WORKER_PROFILE
    tombstone.parent.mkdir(parents=True, exist_ok=True)
    tombstone.write_text("deleted\n", encoding="utf-8")
    return True


def adapter_users(hermes_home: Path) -> list[str]:
    """The other configs under the same Hermes root that still enable the adapter.

    Profiles share the ``hermes`` executable's Python, so the package stays installed
    while any of them lists the plugin or selects the memory provider.
    """
    root = profiles_root(hermes_home)
    candidates = [root.parent / "config.yaml", *sorted(root.glob("*/config.yaml"))]
    users: list[str] = []
    for path in candidates:
        if path.parent in {hermes_home, root / WORKER_PROFILE}:
            continue
        try:
            config = read_hermes_config(path)
        except InitError:
            continue
        plugins = config.get("plugins") if isinstance(config.get("plugins"), dict) else {}
        memory = config.get("memory") if isinstance(config.get("memory"), dict) else {}
        enabled = plugins.get("enabled") if isinstance(plugins.get("enabled"), list) else []
        if PLUGIN_NAME in enabled or memory.get("provider") == MEMORY_PROVIDER:
            users.append(str(path.parent))
    return users


# ---------------------------------------------------------------------------
# Router and identity
# ---------------------------------------------------------------------------

def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if sep and key.strip():
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
    return values


#: ``model.provider`` values that name no endpoint of their own: the base URL is ``model.base_url`` or
#: ``OPENAI_BASE_URL``.
UNNAMED_PROVIDERS = frozenset({"", "custom", "auto"})

# Run under the Hermes interpreter, with the home's .env in its environment: Hermes' own static lookup of the
# endpoint a named provider stands for (an entry under ``providers:`` or ``custom_providers:``, else an API-key
# provider of its registry), and of the transport (``api_mode``) Hermes reaches the model's endpoint over, by the
# rules its runtime resolver uses. ``argv[3]`` is the base URL the model block (or .env) gives, if any. No
# request is sent anywhere.
_ENDPOINT_PROBE = """
import json, os, sys
name, given, found = sys.argv[2], sys.argv[3], {}
try:
    from hermes_cli import runtime_provider as rp
    model_cfg = rp.load_config().get("model")
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
except Exception:
    rp, model_cfg = None, {}


def mode(how):
    try:
        return how() or ""
    except Exception:
        return ""


if name.strip().lower() in ("", "custom", "auto"):
    found = {"api_mode": mode(lambda: rp._resolve_plain_custom_api_mode(model_cfg, given)) if given else ""}
else:
    try:
        from hermes_cli.runtime_provider_custom import _get_named_custom_provider
        entry = _get_named_custom_provider(name) or {}
    except Exception:
        entry = {}
    known = None
    if not entry.get("base_url"):
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
            known = PROVIDER_REGISTRY.get(name)
        except Exception:
            known = None
    if entry.get("base_url"):
        url = given or entry["base_url"]
        found = {"base_url": entry["base_url"], "api_key": entry.get("api_key") or "",
                 "model": entry.get("model") or "",
                 "api_mode": entry.get("api_mode") or mode(lambda: rp._detect_api_mode_for_url(url))
                 or "chat_completions"}
    elif known is not None and known.auth_type == "api_key":
        base = (os.environ.get(known.base_url_env_var) if known.base_url_env_var else "") or known.inference_base_url
        url = given or base
        found = {"base_url": base,
                 "api_key": next((os.environ[v] for v in known.api_key_env_vars if os.environ.get(v)), ""),
                 "model": "",
                 "api_mode": mode(lambda: rp._configured_or_fallback_api_mode(
                     name, model_cfg, url, model_cfg.get("default") or "", opencode_by_model=True))}
print(json.dumps(found))
"""


def resolve_hermes_endpoint(python: Path, hermes_home: Path, provider: str, base_url: str = "") -> dict[str, str]:
    """``{base_url, api_key, model, api_mode}`` as Hermes itself looks them up: the endpoint of a provider it
    names (empty for an unnamed one), and the transport it reaches *base_url* (else that endpoint) over. Empty
    values when it cannot say."""
    empty = {"base_url": "", "api_key": "", "model": "", "api_mode": ""}
    try:
        result = subprocess.run([str(python), "-c", _ENDPOINT_PROBE, str(hermes_home), provider, base_url],
                                capture_output=True, text=True, timeout=120, check=False,
                                env={**os.environ, **_read_env_file(hermes_home / ".env"),
                                     "HERMES_HOME": str(hermes_home)})
        found = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return empty
    return {key: str(found.get(key) or "") for key in empty} if isinstance(found, dict) else empty


def _expand(value: str) -> str:
    """A ``${NAME}`` value from the environment, as Hermes reads its config; anything else as it is."""
    return os.environ.get(value[2:-1], "") if value.startswith("${") and value.endswith("}") else value


def hermes_model_endpoint(hermes_home: Path, main_config: dict[str, Any], *,
                          python: Path | None = None) -> dict[str, str]:
    """The model endpoint Hermes uses: the ``model`` block first; a provider it names (``providers:``,
    ``custom_providers:`` or Hermes' registry, looked up by Hermes' interpreter *python*); then ``.env``.
    ``provider`` is the name the ``model`` block gives, if any; ``api_mode`` the transport Hermes reaches the
    endpoint over, as Hermes decides it ("" when *python* is not given or Hermes cannot say)."""
    model = main_config.get("model")
    env = _read_env_file(hermes_home / ".env")
    found = {"base_url": "", "model": "", "api_key": "", "provider": "", "api_mode": ""}
    if isinstance(model, dict):
        found["base_url"] = str(model.get("base_url") or "")
        found["model"] = str(model.get("default") or model.get("model") or "")
        found["api_key"] = str(model.get("api_key") or "")
        found["provider"] = str(model.get("provider") or "").strip()
    elif isinstance(model, str):
        found["model"] = model
    named = found["provider"].lower() not in UNNAMED_PROVIDERS
    if not found["base_url"] and not named:
        found["base_url"] = env.get("OPENAI_BASE_URL", "")
    if python is not None and (named or found["base_url"]):
        entry = resolve_hermes_endpoint(python, hermes_home, found["provider"], _expand(found["base_url"]))
        found["base_url"] = found["base_url"] or entry["base_url"]
        found["api_key"] = found["api_key"] or entry["api_key"]
        found["model"] = found["model"] or entry["model"]
        found["api_mode"] = entry["api_mode"]
    if not found["api_key"]:
        found["api_key"] = env.get("OPENAI_API_KEY", "")
    return {key: _expand(value) for key, value in found.items()}


#: The transports (Hermes' ``api_mode``) whose endpoint answers the OpenAI-compatible chat completions route the
#: router calls. A Responses host serves that route too; a native Messages endpoint does not.
CHAT_TRANSPORTS = frozenset({"chat_completions", "codex_responses"})


def endpoint_problem(endpoint: dict[str, str]) -> str:
    """Why init cannot point the router at the endpoint Hermes uses, and the flag that fixes it; "" when it
    can. An endpoint Hermes reaches over a transport the router does not speak would take the router's
    requests in the wrong API, so it is refused rather than written."""
    if not endpoint["base_url"]:
        return no_endpoint_problem(endpoint["provider"])
    transport = endpoint.get("api_mode") or ""
    if transport and transport not in CHAT_TRANSPORTS:
        whose = f"the model provider '{endpoint['provider']}'" if endpoint["provider"] else "its model endpoint"
        return (f"Hermes reaches {whose} ({endpoint['base_url']}) over its {transport} transport, not the "
                "OpenAI-compatible chat completions route the router calls: pass --model-url with an "
                "OpenAI-compatible root for the agent's model (and --model and --model-key when it needs them)")
    return ""


def no_endpoint_problem(provider: str) -> str:
    """Why init cannot point the router anywhere, and the flag that fixes it."""
    where = (f"Hermes names the model provider '{provider}' without a base URL, and Hermes does not know one "
             "for it" if provider and provider.lower() not in UNNAMED_PROVIDERS
             else "Hermes' config names no model endpoint (model.base_url, or OPENAI_BASE_URL in its .env)")
    return (f"{where}: pass --model-url with the OpenAI-compatible root the agent's model answers on (and --model "
            "and --model-key when it needs them)")


#: ``--embed-url auto``: ask the model endpoint for an embeddings route instead of naming one.
EMBED_AUTO = "auto"
_PROBE_TEXT = "recall probe"


def _probe_post(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> tuple[int | None, Any, str]:
    """``(status, body, error)`` of one JSON POST; no proxy from the environment."""
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout, trust_env=False)
    except httpx.HTTPError as exc:
        return None, None, type(exc).__name__
    try:
        body = response.json()
    except ValueError:
        body = None
    return response.status_code, body, "" if response.status_code == 200 else f"HTTP {response.status_code}"


def probe_recall_routes(base_url: str, api_key: str, *, embed_model: str, rerank_model: str = "",
                        timeout: float = 15.0) -> dict[str, dict[str, Any]]:
    """One small request to the embeddings route of an OpenAI-compatible *base_url*, and one to its rerank route
    when a rerank model is named; never a chat request. ``embeddings``: ``{url, model, dims}`` or ``{url, error}``
    (``url`` the root to record, ending in /v1); ``rerank`` (when asked): ``{url, model, ok}`` or ``{url, error}``
    (``url`` without /v1, as the reranker appends it)."""
    root = base_url.rstrip("/")
    v1 = root if root.endswith("/v1") else root + "/v1"
    plain = v1[: -len("/v1")]
    status, body, error = _probe_post(v1 + "/embeddings", {"model": embed_model, "input": [_PROBE_TEXT]},
                                      api_key, timeout)
    rows = body.get("data") if isinstance(body, dict) else None
    vector = rows[0].get("embedding") if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
    if status == 200 and isinstance(vector, list) and vector:
        found: dict[str, dict[str, Any]] = {"embeddings": {"url": v1, "model": embed_model, "dims": len(vector)}}
    else:
        found = {"embeddings": {"url": v1, "error": error or "no embedding in the answer"}}
    if rerank_model:
        status, body, error = _probe_post(plain + "/v1/rerank", {"model": rerank_model, "query": _PROBE_TEXT,
                                                                 "documents": [_PROBE_TEXT], "top_n": 1},
                                          api_key, timeout)
        results = body.get("results") if isinstance(body, dict) else None
        found["rerank"] = ({"url": plain, "model": rerank_model, "ok": True}
                           if status == 200 and isinstance(results, list) and results
                           else {"url": plain, "error": error or "no ranking in the answer"})
    return found


#: The key each recall route's client sends (``environment``). It belongs to the endpoint it was recorded for, so it
#: moves with the route: an earlier endpoint's secret never reaches a new one.
ROUTE_KEYS = {"embed_url": "COLONYMIND_EMBED_API_KEY", "rerank_url": "COLONYMIND_RERANKER_API_KEY"}


def _same_root(first: str, second: str) -> bool:
    return first.rstrip("/") == second.rstrip("/")


def _bind_route(router: dict[str, Any], environment: dict[str, Any], route: str, url: str, api_key: str) -> None:
    """Record *url* as the router's *route* with the key it answered to: *api_key*, or none when it answered
    without one. The key the route had before is replaced, never kept for the new endpoint."""
    router[route] = url
    if api_key:
        environment[ROUTE_KEYS[route]] = api_key
    else:
        environment.pop(ROUTE_KEYS[route], None)


#: ``--embed-url local``: the local embedder (``colonymind.vector.local_embedder``) instead of an endpoint.
EMBED_LOCAL = "local"


def _recall_routes(args, data: dict[str, Any], *, home: Path, base_url: str, model: str, api_key: str,
                   non_interactive: bool) -> list[str]:
    """Step 3's recall half: an embedder (semantic recall) and a reranker. ``--embed-url <root>`` records an
    endpoint as given, dropping the key recorded for a different earlier one. ``--embed-url auto``, or a yes in
    interactive init, asks the model endpoint's embeddings and rerank routes (``probe_recall_routes``) and records
    what answered with the key it was asked with (``_bind_route``); without either nothing is asked. When no
    embeddings route answers, or none was asked for, the local embedder is offered (``_offer_local``);
    ``--embed-url local`` takes it at once, and a rerun keeps it. Returns what to say."""
    requested = str(getattr(args, "embed_url", None) or "").strip()
    router = data["router"]
    if requested.lower() == EMBED_LOCAL:
        return _local_embedder(args, data, home, keep=False)
    if not requested and router.get("embed_provider") == EMBED_LOCAL:
        return _local_embedder(args, data, home, keep=True)
    auto = requested.lower() == EMBED_AUTO
    asked = False
    if not requested and not router.get("embed_url") and not non_interactive:
        asked = True
        auto = _ask(f"Ask {base_url} for an embeddings route (and a rerank route), so recall finds meaning and not "
                    "only words? Only those two routes are called, never a chat request (y/n)", "n",
                    non_interactive).lower().startswith("y")
    notes: list[str] = []
    if not auto:
        if asked:
            return _offer_local(args, data, home, non_interactive)
        previous = str(router.get("embed_url") or "")
        router["embed_url"] = requested or previous
        if requested:
            # Giving init an endpoint is asking for semantic recall: it also clears the ``false`` the
            # releases before the switch was live recorded whenever init found no endpoint.
            data["mind"]["faculties"]["semantic_recall"] = True
            notes += _leave_local(data, home)
            key = ROUTE_KEYS["embed_url"]
            environment = data.get("environment") or {}
            if previous and not _same_root(previous, requested) and key in environment:
                del environment[key]
                notes.append(f"environment.{key} dropped: it was the key for the earlier embedding endpoint "
                             f"{previous}; set it again if {requested} needs one")
        if getattr(args, "embed_model", None):
            router["embed_model"] = str(args.embed_model)
        if getattr(args, "embed_dims", None) is not None:
            router["embed_dims"] = int(args.embed_dims)
        return notes
    recorded = router.get("embed_model") if router.get("embed_provider") != EMBED_LOCAL else ""
    embed_model = (getattr(args, "embed_model", None) or recorded
                   or _ask("Embedding model at that endpoint", model, non_interactive) or model)
    rerank_model = (getattr(args, "rerank_model", None) or router.get("rerank_model")
                    or _ask("Rerank model at that endpoint (empty for none)", "", non_interactive))
    found = probe_recall_routes(base_url, api_key, embed_model=embed_model, rerank_model=rerank_model)
    environment = data.setdefault("environment", {})
    embeddings = found["embeddings"]
    if embeddings.get("dims"):
        notes += _leave_local(data, home)
        _bind_route(router, environment, "embed_url", embeddings["url"], api_key)
        router.update(embed_model=embed_model, embed_dims=embeddings["dims"])
        data["mind"]["faculties"]["semantic_recall"] = True
        notes.append(f"embeddings answered at {embeddings['url']} (model {embed_model}, {embeddings['dims']} "
                     "dimensions): recorded as router.embed_url")
    elif router.get("embed_provider") == EMBED_LOCAL:
        notes.append(f"no embeddings at {embeddings['url']} ({embeddings['error']}): the local embedder stays")
        notes += _local_embedder(args, data, home, keep=True)
    else:
        notes.append(f"no embeddings at {embeddings['url']} ({embeddings['error']})")
        notes += _offer_local(args, data, home, non_interactive, named_model=False)
    rerank = found.get("rerank")
    if rerank and rerank.get("ok"):
        _bind_route(router, environment, "rerank_url", rerank["url"], api_key)
        router["rerank_model"] = rerank_model
        notes.append(f"rerank answered at {rerank['url']} (model {rerank_model}): recall reranks there; no calibrated "
                     "cutoff yet, so recall never abstains ('colonymind doctor' says how to calibrate one)")
    elif rerank:
        notes.append(f"no rerank at {rerank['url']} ({rerank['error']}): recall keeps its candidate order")
    return notes


KEYWORD_ONLY_NOTE = "recall stays keyword-only ('colonymind init --embed-url local' sets up the local embedder later)"


def _offer_local(args, data: dict[str, Any], home: Path, non_interactive: bool, *,
                 named_model: bool = True) -> list[str]:
    """No embeddings route: offer the local embedder when its extra is installed and can load the model
    (non-interactive init, which only gets here with ``--embed-url auto``, takes it). A download that fails leaves
    recall keyword-only.
    ``named_model``: ``--embed-model`` names a local model here, not the endpoint's (after a probe it named the
    endpoint's)."""
    from colonymind.vector import local_embedder
    if not local_embedder.extra_available():
        return [f"recall stays keyword-only; for the local embedder, install {local_embedder.EXTRA} "
                f"({local_embedder.INSTALL}), then run 'colonymind init --embed-url local'"]
    try:
        chosen = _local_choice(args if named_model else None)
    except InitError as exc:
        return [f"recall stays keyword-only: {exc}"]
    unmet = local_embedder.unmet(chosen.model)
    if unmet:
        return [f"recall stays keyword-only; the sidecar's environment cannot load the local embedder {chosen.model} "
                f"({', '.join(unmet)}): {local_embedder.upgrade_command(chosen.model)}, then run 'colonymind init "
                "--embed-url local'"]
    target = local_embedder.model_dir(home, chosen.model, chosen.revision)
    if not non_interactive:
        answer = _ask(f"Run an embedder in the sidecar instead: {chosen.model} ({chosen.params} parameters, "
                      f"{chosen.dims} dimensions; about {chosen.download_gb:g} GB downloaded once into {target})? "
                      "(y/n)", "y", non_interactive)
        if not answer.lower().startswith("y"):
            return [f"local embedder declined: {KEYWORD_ONLY_NOTE}"]
    try:
        return _local_embedder(args if named_model else None, data, home, keep=False, chosen=chosen)
    except InitError as exc:
        return [f"{exc}; {KEYWORD_ONLY_NOTE}"]


def _local_choice(args):
    """The model ``--embed-model`` names, which must be pinned, else the tier table's for this machine."""
    from colonymind.vector import local_embedder
    named = str(getattr(args, "embed_model", None) or "").strip() if args is not None else ""
    try:
        return local_embedder.pinned(named) if named else local_embedder.for_this_machine()
    except (ValueError, local_embedder.LocalEmbedderError) as exc:
        raise InitError(str(exc)) from None


def _local_embedder(args, data: dict[str, Any], home: Path, *, keep: bool, chosen=None) -> list[str]:
    """Record the local embedder once its pinned files are in the instance (downloaded here when missing).
    ``keep``: a rerun without ``--embed-url`` keeps the model and commit recorded, and the owner's switch. Raises
    ``InitError`` when the extra is missing, its libraries cannot load the model, or the download fails; nothing
    is recorded then."""
    from colonymind.vector import local_embedder
    if not local_embedder.extra_available():
        raise InitError(f"the local embedder runs in the sidecar, which needs {local_embedder.EXTRA}: "
                        f"{local_embedder.INSTALL}")
    router = data["router"]
    if keep:
        model, revision, dims = str(router["embed_model"]), str(router["embed_revision"]), int(router["embed_dims"])
    else:
        chosen = chosen or _local_choice(args)
        model, revision, dims = chosen.model, chosen.revision, chosen.dims
        given = getattr(args, "embed_dims", None) if args is not None else None
        if given is not None and int(given) != dims:
            raise InitError(f"--embed-dims {given} is not {model}'s width ({dims} dimensions)")
    unmet = local_embedder.unmet(model)
    if unmet:
        raise InitError(f"the sidecar's environment cannot load the local embedder {model} ({', '.join(unmet)}): "
                        f"{local_embedder.upgrade_command(model)}")
    target = local_embedder.model_dir(home, model, revision)
    notes: list[str] = []
    if not local_embedder.installed(target):
        _say(f"  downloading {model} at revision {revision[:12]} into {target}")
        try:
            local_embedder.fetch(home, model, revision)
        except (local_embedder.LocalEmbedderError, OSError) as exc:
            raise InitError(f"the local embedder's download failed: {exc}") from None
        notes.append(f"{model} downloaded into {target}")
    notes += _drop_endpoint_settings(data)
    router.update(embed_provider=EMBED_LOCAL, embed_url="", embed_model=model, embed_revision=revision,
                  embed_dims=dims)
    if not keep:
        data["mind"]["faculties"]["semantic_recall"] = True
    notes.append(f"local embedder {model} (revision {revision[:12]}, {dims} dimensions) in {target}: recall finds "
                 f"meaning with no embeddings endpoint; the sidecar runs it on {local_embedder.device()}")
    return notes


def _drop_endpoint_settings(data: dict[str, Any]) -> list[str]:
    """The local embedder replaces an endpoint: what only the endpoint read under ``environment``
    (``config.ENDPOINT_ENVIRONMENT``) goes, each named with why, so the file loads and the sidecar starts."""
    from colonymind.config import ENDPOINT_ENVIRONMENT
    environment = data.get("environment") or {}
    endpoint = str(data["router"].get("embed_url") or "")
    notes = []
    for name, reason in ENDPOINT_ENVIRONMENT.items():
        if environment.pop(name, None) is not None:
            notes.append(f"environment.{name} dropped: {reason}" + (f" {endpoint}" if endpoint and
                                                                   name == ROUTE_KEYS["embed_url"] else "")
                         + ", which the local embedder replaces")
    return notes


def _leave_local(data: dict[str, Any], home: Path) -> list[str]:
    """An endpoint replaces the local embedder: its model, commit and width go (the endpoint's are recorded or
    learned), and its files are named, not deleted."""
    router = data["router"]
    if router.get("embed_provider") != EMBED_LOCAL:
        return []
    from colonymind.vector import local_embedder
    router.update(embed_provider="endpoint", embed_model="", embed_revision="", embed_dims=0)
    return [f"the local embedder is replaced; its files stay in {local_embedder.models_dir(home)} (delete them to "
            "free the space)"]


def write_llm_config(home: Path, *, base_url: str, model: str, api_key: str) -> bool:
    """Persist the router's host config once; an existing file is the operator's."""
    path = home / LLM_CONFIG_FILE
    if path.is_file():
        return False
    from colonymind.setup import apply_llm_config_fixes
    document = {
        "provider": "local",
        "baseUrl": base_url,
        "apiKey": api_key or "local-no-key",
        "models": {role: model for role in _MODEL_ROLES},
    }
    fixed, _ = apply_llm_config_fixes(document)
    configuration._atomic_write(path, (json.dumps(fixed, indent=2) + "\n").encode("utf-8"), mode=0o600)
    return True


def _contacts_db(home: Path) -> Path:
    db_path = home / "contacts.db"
    return db_path if db_path.exists() else home / "colonymind-contacts.db"


def _unresolved_owner(contact_id: str, where: str) -> str:
    return (f"owner.contact_id {contact_id} in colonymind.yaml does not resolve to a live contact ({where}); "
            "correct the id, or remove owner.contact_id to create a new owner contact")


def seeded_owner_problem(home: Path, contact_id: str) -> str | None:
    """Why a seeded ``owner.contact_id`` cannot be the owner, or None when it names a live contact.

    Checked before init writes anything: a typo, or an id naming a missing or deleted row,
    must not turn into a second, empty owner contact that the mind and recall then treat as
    the owner.
    """
    db_path = _contacts_db(home)
    if not db_path.exists():
        return _unresolved_owner(contact_id, f"there is no contacts store in {home}")
    from colonymind.contacts.config import ContactsConfig
    from colonymind.contacts.store import SQLiteContactStore

    async def live() -> bool:
        store = SQLiteContactStore(ContactsConfig(sqlite_path=str(db_path)))
        await store.connect()
        try:
            return await store.get(contact_id) is not None
        finally:
            await store.close()

    return None if asyncio.run(live()) else _unresolved_owner(contact_id, f"not in {db_path.name}, or deleted")


def ensure_owner_contact(home: Path, cfg: Config, identity: dict[str, Any]) -> tuple[str, bool]:
    """Create the owner contact once and return ``(contact_id, created)``.

    A recorded ``owner.contact_id`` that does not resolve is refused, never replaced.
    """
    from colonymind.contacts.config import ContactsConfig
    from colonymind.contacts.store import SQLiteContactStore
    from colonymind.setup import build_owner_contact

    db_path = _contacts_db(home)
    existing = str(cfg.get("owner.contact_id") or "")
    owner = identity.get("owner", {})
    handles = [(str(item.get("platform") or ""), str(item.get("id") or ""))
               for item in owner.get("handles") or [] if isinstance(item, dict)]

    async def work() -> tuple[str, bool]:
        store = SQLiteContactStore(ContactsConfig(sqlite_path=str(db_path)))
        await store.connect()
        try:
            if existing:
                if await store.get(existing) is None:
                    raise InitError(_unresolved_owner(existing, f"not in {db_path.name}, or deleted"))
                return existing, False
            contact_id = await build_owner_contact(store, str(owner.get("name") or "Owner"), handles)
            return contact_id, True
        finally:
            await store.close()

    return asyncio.run(work())


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def _service(cfg: Config):
    from colonymind.services.instance import InstanceService
    return InstanceService(cfg.home, cfg.hermes_home, host=cfg.get("sidecar.host"), port=cfg.get("sidecar.port"))


def install_service(cfg: Config) -> str:
    """Install and start the sidecar user service; a message when no manager exists."""
    from colonymind.services.instance import ServiceError
    if os.environ.get("COLONYMIND_INIT_NO_SERVICE"):
        return "sidecar service skipped (COLONYMIND_INIT_NO_SERVICE)"
    try:
        service = _service(cfg)
    except ServiceError as exc:
        return f"sidecar service skipped: {exc}"
    try:
        service._manager_ready()
    except Exception:
        manager = "launchd" if sys.platform == "darwin" else "systemd --user"
        return f"sidecar service skipped: {manager} is not available in this session; start it with 'colonymind start'"
    try:
        status = service.install()
    except (ServiceError, OSError) as exc:
        return f"sidecar service not installed: {exc}"
    installed = f"sidecar service installed ({status['manager']}: {status['label']})"
    try:
        started = service.start()  # enable alone starts nothing before the next login
    except (ServiceError, OSError) as exc:
        return f"{installed}, not running: {exc}; start it with 'colonymind service start'"
    return f"{installed} and running{_health_words(started)}"


def _health_words(result: Any) -> str:
    """The served health verdict when it is not ``ok``: the status and its reasons, in words."""
    health = (result or {}).get("health") if isinstance(result, dict) else None
    if not health or health == "ok":
        return ""
    problems = "; ".join(str(item) for item in result.get("problems") or []) or "run 'colonymind doctor'"
    return f"; health {health}: {problems}"


def service_status(cfg: Config) -> dict[str, Any] | None:
    from colonymind.services.instance import ServiceError
    try:
        service = _service(cfg)
        if not service._owned():
            return None
        return service.status()
    except (ServiceError, OSError):
        return None


def restart_service(cfg: Config) -> str:
    from colonymind.services.instance import ServiceError
    status = service_status(cfg)
    if status is None:
        return "sidecar service is not installed; restart the sidecar yourself if it is running"
    if not status.get("running"):
        return "sidecar service is installed but not running; start it with 'colonymind service start'"
    try:
        started = _service(cfg).start(restart=True)
    except (ServiceError, OSError) as exc:
        return f"sidecar service restart failed: {exc}"
    return f"sidecar service restarted{_health_words(started)}"


def _service_live(status: dict[str, Any]) -> bool:
    """Whether the manager runs this instance, or starts it again by itself: a running process, a unit
    still activating or waiting to restart, or a job launchd keeps loaded."""
    if status.get("manager") == "launchd":
        return bool(status.get("running") or status.get("loaded"))
    return bool(status.get("running") or status.get("state") in {"active", "activating", "deactivating", "reloading"})


def service_running_for_upgrade(cfg: Config) -> bool:
    """Whether this instance's user service is running: the upgrade stops it, and starts it again when it
    ends, whether it succeeded or not."""
    status = service_status(cfg)
    return status is not None and _service_live(status)


def stop_service_for_upgrade(cfg: Config) -> bool:
    """Stop this instance's user service and wait until its process has exited, so no writer holds a
    store the upgrade copies, migrates or moves aside. True when it was running."""
    from colonymind.services.instance import ServiceError
    if not service_running_for_upgrade(cfg):
        return False
    try:
        _service(cfg).stop()
    except (ServiceError, OSError) as exc:
        raise InitError(f"the sidecar service did not stop ({exc}); stop it with 'colonymind service stop' and "
                        "run 'colonymind upgrade' again") from None
    return True


def start_service_after_upgrade(cfg: Config) -> str:
    from colonymind.services.instance import ServiceError
    try:
        started = _service(cfg).start()
    except (ServiceError, OSError) as exc:
        return (f"sidecar service stopped for the upgrade and not started again: {exc}; start it with "
                "'colonymind service start'")
    return f"sidecar service stopped for the upgrade and started again{_health_words(started)}"


def running_outside_service(cfg: Config) -> str | None:
    """What shows this instance's sidecar running without its user service, in words, or None: the
    process ``colonymind start`` recorded (its pid and signature, as ``colonymind stop`` checks them),
    or a listener on the instance's port."""
    try:
        record = json.loads((cfg.home / "sidecar-process.json").read_text(encoding="utf-8"))
        pid, expected = int(record.get("pid")), str(record.get("signature") or "")
    except (OSError, ValueError, TypeError, AttributeError):
        pid, expected = 0, ""
    if pid > 0 and expected:
        from colonymind.cli import _process_signature
        try:
            if _process_signature(pid) == expected:
                return f"the sidecar process {pid} that 'colonymind start' recorded is running"
        except (OSError, subprocess.SubprocessError):
            pass
    host = str(cfg.get("sidecar.host") or "127.0.0.1")
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    port = int(cfg.get("sidecar.port") or 7777)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return f"a process is listening on the instance's port ({host}:{port})"
    except OSError:
        return None


def uninstall_service(cfg: Config) -> str:
    from colonymind.services.instance import ServiceError
    try:
        service = _service(cfg)
        if not service._owned():
            return "sidecar service was not installed"
        service.uninstall()
    except (ServiceError, OSError) as exc:
        return f"sidecar service removal failed: {exc}"
    return "sidecar service removed"


# ---------------------------------------------------------------------------
# Backups and migrations
# ---------------------------------------------------------------------------

def backup_instance(home: Path) -> Path:
    """Copy every store and configuration file, and ``hermes-install.json`` (the provider init replaced, which
    uninstall puts back), into ``backups/<stamp>/``, a directory of its own: a second backup within the same
    second (a retry right after a failure) never overwrites the first. The decision log's secret
    (``instinct.key``) goes with ``instinct.db``, private: without it a restored log's keyed hashes match
    nothing."""
    from colonymind.instinct.log import KEY_FILE as INSTINCT_KEY_FILE
    from colonymind.backup import _copy_private, _private_directory, _private_file
    stamp = _utc_stamp()
    destination, number = home / BACKUPS_DIR / stamp, 1
    while destination.exists():
        number += 1
        destination = home / BACKUPS_DIR / f"{stamp}-{number}"
    _private_directory(destination.parent)
    _private_directory(destination)
    for path in sorted(home.iterdir()):
        if path.is_dir():
            continue
        if path.suffix == ".db":
            try:
                _private_file(destination / path.name)
                with closing(sqlite3.connect(path)) as source, \
                        closing(sqlite3.connect(destination / path.name)) as target:
                    source.backup(target)
            except sqlite3.DatabaseError as exc:
                raise InitError(f"{path.name} could not be read to back it up ({exc}); restore it from an earlier "
                                f"backup or move it out of {home}, then run 'colonymind upgrade' again") from None
            continue
        if path.name == INSTINCT_KEY_FILE:
            _copy_private(path, destination / path.name)
            os.chmod(destination / path.name, 0o600)
            continue
        if path.name in {CONFIG_FILE, IDENTITY_FILE, KEY_FILE, LLM_CONFIG_FILE, INSTALL_STATE, ".env", LEGACY_MANIFEST,
                         LEGACY_KEYRING}:
            _copy_private(path, destination / path.name)
    return destination


def _migration_stores() -> tuple[tuple[str, Path], ...]:
    package = Path(__file__).resolve().parent
    return (
        ("contacts.db", package / "contacts" / "migrations"),
        ("colonymind-contacts.db", package / "contacts" / "migrations"),
    )


def pending_store_migrations(home: Path) -> list[str]:
    """The numbered migrations the existing stores have not applied yet."""
    from colonymind.migrations import _discover, applied_versions_sync
    pending: list[str] = []
    for name, migrations in _migration_stores():
        path = home / name
        if not path.is_file():
            continue
        with closing(sqlite3.connect(path)) as connection, connection:
            applied = applied_versions_sync(connection)
        pending.extend(f"{name}:{version}" for version, _ in _discover(migrations) if version not in applied)
    return pending


def run_store_migrations(home: Path) -> list[str]:
    """Apply the numbered SQL migrations to the stores that exist."""
    from colonymind.migrations import run_migrations_sync
    applied: list[str] = []
    for name, migrations in _migration_stores():
        path = home / name
        if not path.is_file():
            continue
        with closing(sqlite3.connect(path)) as connection, connection:
            for version in run_migrations_sync(connection, migrations):
                applied.append(f"{name}:{version}")
    return applied


# State owned by code that no longer exists: the autonomy scheduler, the queue
# approval ledger, standing grants, the proactive delivery bridge, the task
# queue, the project engine, the governed action ledger, the directive store,
# directed tasks, the response guard's ledgers, the agent bridge poller's
# seen-lists and (since the drives milestone) the cognitive workspace, the
# cognition spine, its evidence and drive-governance ledgers, the external
# event inbox and the surprise store, (since the people milestone) the
# second-order theory-of-mind stores, the engagement profiles, the relationship
# briefs the per-contact digest replaced, the P8 shadow stores, the conversation
# presence census (its readers went with P8) and the identity bootstrap report,
# and (since the memory milestone) the belief engine, the chain's identity files,
# keys and manifests and the world model; the Neo4j graph and the continuous
# learner kept nothing local; and (since the self-improvement milestone) the
# toolsmith's registry and library, the P4 experiments and their parameter store,
# skills memory and the escalation miner; and (since the prune) the briefings,
# the synthesis insights, the observations, the agent registry, the repo mirrors,
# the legacy goal records, the patterns, the proposals, the channel registry, the
# shared facts, the situation spine, the corrections ledger, and the leftovers no
# code names (an old skills store, a sandbox tree and two autonomy state files).
# (The competence self-model, its action journal and the benchmark stay while
# callers outside the sidecar still read their routes.) An upgrade moves them into
# the backup instead of leaving orphans behind, and never deletes them. A
# directory entry names a whole tree.
RETIRED_STATE = (
    "approval_authority.db",
    "schedules.db",
    "standing_approvals.json",
    "colonymind-delivery-rate-limit.db",
    "colonymind-governed-gateway-outcomes.db",
    "task_queue.db",
    "colonymind-projects.db",
    "colonymind-workspace.db",
    "colonymind-cognition.db",
    "colonymind-cognition-evidence.db",
    "cognition-drive-governance.db",
    "external-cognition-events.db",
    "colonymind-surprise.db",
    "governed-actions",
    "colonymind-directives.db",
    "colonymind-directed.db",
    "colonymind-guard-audit.db",
    "colonymind-context-provenance.db",
    "colonymind-tom2-taint.db",
    "colonymind-tom2.db",
    "colonymind-tom2-exposure.db",
    "colonymind-engagement.db",
    "colonymind-relationships.db",
    "colonymind-p8-visibility.db",
    "colonymind-p8-arcs.db",
    "colonymind-p8-recipient-audit.db",
    "colonymind-presence.db",
    "bootstrap.db",
    "bridge",
    "colonymind-beliefs.db",
    "chain.db",
    "colonymind-id",
    "node-id",
    "node-cert.json",
    "genesis.json",
    "colonymind-manifest.json",
    "colonymind_world_model.db",
    "colonymind-keys",
    "node-keys",
    "colonymind-toolsmith.db",
    "toolsmith_library",
    "colonymind-experiments.db",
    "colonymind-params.db",
    "colonymind-skills.db",
    "colonymind-mining.db",
    "briefings.db",
    "insights.db",
    "observations.db",
    "agents.db",
    "agents.db.backup",
    "repo-mirrors",
    "goals.db",
    "colonymind-goals.db",
    "colonymind-patterns.db",
    "colonymind-proposals.db",
    "colonymind-channels.db",
    "colonymind-facts.db",
    "colonymind-situation.db",
    "colonymind-learning-feedback.db",
    "skills.db",
    "sandbox",
    "autonomy_checkin.json",
    "autonomy_synthesis.json",
)
# Tables inside surviving stores whose code was deleted: the legacy perspective tables
# (the automatic opinion revisions and the attention snapshot), and (since the people
# milestone) the contact store's owner-provisioning receipts and its old merge-proposal,
# merge-audit and confirmed-distinct tables (merges are identity-link receipts now), and (since
# the self-improvement milestone) the trust ladder's stages and notices in the self-model
# store, whose competence tables stay. The goal store, whose subtask and DAG tables were
# listed here, retires whole since the prune (RETIRED_STATE). The backup taken before the
# migrations keeps their rows; the upgrade drops them from the live store.
RETIRED_TABLES: dict[str, tuple[str, ...]] = {
    "turn-idempotency.db": ("self_opinion_revisions", "self_attention", "self_judgment_runs",
                            "execution_observations", "execution_runtime_observations"),
    "colonymind-contacts.db": ("contact_provision_operations", "contact_merge_proposals", "contact_merge_audit",
                              "contact_confirmed_distinct"),
    "colonymind-self-model.db": ("trust_stage", "trust_notices"),
}
INITIATIVES_DB = "initiatives.db"
COMMS_DB = "colonymind-comms.db"


def retired_state_present(home: Path) -> list[str]:
    return [name for name in RETIRED_STATE if (home / name).exists()]


def retire_state(home: Path, backup_dir: Path) -> list[str]:
    """Move the retired stores (and SQLite side files) into ``backup_dir/retired``.

    The chain's ``colonymind-id`` is adopted as ``instance-id`` before it moves, so
    the agent registry rows that name this instance keep matching.
    """
    notes: list[str] = []
    destination = backup_dir / "retired"
    if (home / "colonymind-id").exists():
        from colonymind.instance import instance_id
        instance_id(home)
    for name in retired_state_present(home):
        from colonymind.backup import _make_tree_private, _private_directory
        _private_directory(destination)
        if (home / name).is_dir():
            shutil.move(str(home / name), str(destination / name))
        else:
            for suffix in ("", "-wal", "-shm", "-journal"):
                source = home / (name + suffix)
                if source.exists():
                    shutil.move(str(source), str(destination / (name + suffix)))
        _make_tree_private(destination)
        notes.append(f"retired {name} (moved to {destination})")
    return notes


def retired_tables_present(home: Path) -> list[str]:
    """``store:table`` for every retired table that still exists in a surviving store."""
    present: list[str] = []
    for name, tables in RETIRED_TABLES.items():
        path = home / name
        if not path.is_file():
            continue
        try:
            with closing(sqlite3.connect(path)) as connection, connection:
                existing = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'")}
        except sqlite3.DatabaseError:
            continue
        present.extend(f"{name}:{table}" for table in tables if table in existing)
    return present


def retire_tables(home: Path) -> list[str]:
    """Drop the retired tables (the backup taken first keeps their rows)."""
    notes: list[str] = []
    for item in retired_tables_present(home):
        name, table = item.split(":", 1)
        with closing(sqlite3.connect(home / name)) as connection, connection:
            rows = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
            connection.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608
        notes.append(f"retired table {table} in {name} ({rows} rows kept in the backup)")
    return notes


def pending_ingress_adoption(home: Path) -> list[str]:
    """Transport ingress receipts and coverage rows still stamped by a retired producer.

    Earlier lines issued several client principals and stamped durable intake
    rows with the principal's name; this line authenticates one key and stamps
    every caller with its single producer. Rows left under another name can
    never be read back or handed off by the transport that journaled them.
    """
    path = home / COMMS_DB
    if not path.is_file():
        return []
    from colonymind.api.auth import KEY_PRINCIPAL
    from colonymind.contacts.transport_ingress import retired_producer_rows
    try:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            found = retired_producer_rows(connection, KEY_PRINCIPAL)
    except sqlite3.DatabaseError:
        return []
    pending: list[str] = []
    if found["receipts"]:
        pending.append(f"{COMMS_DB}:transport_ingress ({found['receipts']} receipts from "
                       f"{len(found['producers'])} retired producers)")
    if found["coverage"]:
        pending.append(f"{COMMS_DB}:transport_ingress_coverage ({found['coverage']} rows)")
    return pending


def adopt_ingress_producers(home: Path) -> list[str]:
    """Re-scope the retired producers' ingress rows to the instance key (idempotent)."""
    if not pending_ingress_adoption(home):
        return []
    from colonymind.api.auth import KEY_PRINCIPAL
    from colonymind.contacts.transport_ingress import adopt_retired_producers
    with closing(sqlite3.connect(home / COMMS_DB)) as connection, connection:
        connection.row_factory = sqlite3.Row
        result = adopt_retired_producers(connection, KEY_PRINCIPAL)
    notes = [f"migration applied: {COMMS_DB}: {result['receipts']} transport ingress receipts and "
             f"{result['coverage']} coverage rows re-scoped from retired producers "
             f"({', '.join(result['producers'])}) to the instance key"]
    if result["kept"]:
        notes.append(f"{result['kept']} transport ingress receipts kept on their retired producer: "
                     "the same event already exists under the instance key")
    return notes


def pending_initiative_columns(home: Path) -> list[str]:
    """The intention and audit columns an existing initiatives store still lacks."""
    path = home / INITIATIVES_DB
    if not path.is_file():
        return []
    from colonymind.initiatives.store import missing_mind_columns
    with closing(sqlite3.connect(path)) as connection, connection:
        return [f"{INITIATIVES_DB}:{column}" for column in missing_mind_columns(connection)]


def _initiative_rows(path: Path) -> int | None:
    """The initiatives table's row count, read only; None when it cannot be read."""
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(initiatives)")}
            predicate = (" WHERE kind IS NULL OR kind NOT IN "
                         "('owner_goal','owner_subgoal','owner_goal_progress')") if "kind" in columns else ""
            return int(connection.execute("SELECT count(*) FROM initiatives" + predicate).fetchone()[0])
    except sqlite3.DatabaseError:
        return None


def migrate_initiatives(home: Path) -> list[str]:
    """Add the intention and audit columns in place (architecture 5.2).

    The store is opened the way the sidecar opens it, then checked again: the
    migration counts as applied only when every column is there and no row went
    missing. A store that cannot be opened (another process
    holds its lock) is left as it is and the upgrade fails with the reason.
    """
    pending = pending_initiative_columns(home)
    if not pending:
        return []
    from colonymind.initiatives import store as initiative_store
    path = home / INITIATIVES_DB
    before = _initiative_rows(path)
    try:
        store = initiative_store.InitiativeStore(state_dir=home)
        store.close()
    except sqlite3.DatabaseError as exc:
        raise InitError(f"{INITIATIVES_DB} could not be opened to add its columns ({exc}); it was left as it is. "
                        "Stop every process that uses it ('colonymind service stop') and run 'colonymind upgrade' "
                        "again") from None
    still, after = pending_initiative_columns(home), _initiative_rows(path)
    if still:
        raise InitError(f"{INITIATIVES_DB} still lacks {', '.join(item.split(':', 1)[1] for item in still)} after "
                        "the upgrade opened it; run 'colonymind upgrade' again with the service stopped")
    if after is None or (before is not None and after < before):
        raise InitError(f"{INITIATIVES_DB} lost rows while it was opened ({before} rows before, {after} after); "
                        f"restore it from the backup this upgrade took under {BACKUPS_DIR}/")
    return [f"migration applied: {item}" for item in pending]


# ---------------------------------------------------------------------------
# Credential grants (docs/CREDENTIALS.md)
# ---------------------------------------------------------------------------

INSTANCE_KEY_ROLE = "api.key: the Hermes adapter, and any client that shares its key"


def pending_credential_grants(home: Path) -> list[str]:
    """``colonymind.yaml`` names no ``credentials``: the instance ran before credentials were bound."""
    path = home / CONFIG_FILE
    if not path.is_file():
        return []
    try:
        raw = configuration.read_raw(path)
    except ConfigError:
        return []
    return [] if "credentials" in raw else [f"{CONFIG_FILE}:credentials"]


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def owner_handle_platforms(identity: dict[str, Any]) -> list[str]:
    """The platforms the owner has a handle on in ``identity.yaml`` (a ``[{platform, id}]`` list or a
    ``{platform: [ids]}`` mapping), in order."""
    handles = (identity.get("owner") or {}).get("handles") if isinstance(identity, dict) else None
    if isinstance(handles, dict):
        found = [str(name) for name in handles]
    elif isinstance(handles, list):
        found = [str(item.get("platform") or "") for item in handles if isinstance(item, dict)]
    else:
        found = []
    return list(dict.fromkeys(name.strip().lower() for name in found if name.strip()))


def credential_report(cfg: Config, *, roles: dict[str, str] | None = None) -> list[str]:
    """What each credential may write as, one line each, and the owner's handles on platforms the
    instance key may not write the owner from."""
    table = CredentialTable.from_config(cfg.get("credentials"))
    roles = roles or {}
    lines = []
    for grant in table.grants():
        role = roles.get(grant.name) or (INSTANCE_KEY_ROLE if grant.instance else "")
        lines.append(f"{grant.name}{f' ({role})' if role else ''} {describe(grant)}")
    key = table.instance
    missing = [name for name in owner_handle_platforms(load_identity(cfg.home))
               if name not in key.owner_platforms and ANY_PLATFORM not in key.owner_platforms]
    if missing:
        flags = " ".join(f"--owner-platform {name}" for name in [*key.owner, *missing])
        lines.append(f"the owner has a handle on {_join(missing)}, which api.key may not write the owner from: "
                     f"turns from there are kept as {key.identity}, not as the owner's; to grant it: "
                     f"colonymind credentials set {INSTANCE_CREDENTIAL} {flags}")
    return lines


def record_credential_grants(home: Path) -> list[str]:
    """Write the grants the instance's credentials now run under into ``colonymind.yaml`` and say what
    each may write as. Idempotent: an instance that names its credentials is left as it is.

    The instance runs 1.10 or later (it has ``colonymind.yaml``), which authenticated only the instance key:
    the key keeps what 1.10 gave it (the owner on the default platforms, the whole API), and a 1.9.0
    keyring still on disk, which 1.10 no longer honoured, is retired, never carried."""
    if not pending_credential_grants(home):
        return []
    configuration.update_config({"credentials": default_credentials()}, home)
    notes = [f"credential grants recorded in {CONFIG_FILE} (credentials); turns a credential may not write "
             "are refused with 403, never re-attributed:"]
    notes.extend("  " + line for line in credential_report(load_config(home)))
    if (home / LEGACY_KEYRING).is_file():
        notes.append(f"  {LEGACY_KEYRING} is a 1.9.0 keyring this instance no longer honoured (it runs 1.10 or "
                     "later, which accepted only api.key): retired: none of its principals or secrets is carried; "
                     "the file is left as it is, remove it when you no longer need it")
    return notes


# ---------------------------------------------------------------------------
# 1.9.0 migration
# ---------------------------------------------------------------------------

def is_legacy_instance(home: Path) -> bool:
    return (home / LEGACY_MANIFEST).is_file() and not (home / CONFIG_FILE).is_file()


def _keyring_time(value: Any) -> _dt.datetime | None:
    """A 1.9.0 ``accept_until`` in UTC; ``None`` when absent. Raises ValueError when it does not parse."""
    if not value:
        return None
    parsed = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def _keyring_accepts(entry: dict[str, Any], now: _dt.datetime) -> bool:
    """The 1.9.0 rule for a principal or credential: active or retiring, and not past
    ``accept_until``."""
    if str(entry.get("status") or "active").lower() not in {"active", "retiring"}:
        return False
    try:
        until = _keyring_time(entry.get("accept_until"))
    except ValueError:
        return False
    return until is None or now < until


def _earliest(*times: _dt.datetime | None) -> _dt.datetime | None:
    present = [value for value in times if value is not None]
    return min(present) if present else None


@dataclasses.dataclass
class LegacyPrincipal:
    """One 1.9.0 keyring principal, as the migration carries it (docs/CREDENTIALS.md)."""

    principal: str
    reason: str                                    # why it is not carried, when it has no live secret
    accept_until: _dt.datetime | None              # the principal's own expiry
    secrets: list[tuple[str, _dt.datetime | None]]  # every live secret, with its credential's expiry
    authority: Authority                           # what it could do (``legacy_authority``)
    unmapped: list[str]                            # what it could do that has no counterpart here


def legacy_authority(raw: dict[str, Any], owner_id: str, shared_id: str = "shared",
                     global_id: str = "global") -> tuple[Authority, list[str]]:
    """What a 1.9.0 keyring principal could do, as an :class:`~colonymind.api.credentials.Authority`, and what it
    could do that has no counterpart in this release (named for the report).

    Its persons are its static grant (``person_ids``, ``viewer_person_id`` and the owner, shared and global
    persons its ``audiences`` name); a request that names nobody acted as its viewer; ``turns:resolve-sender``
    let a sender resolve to anyone. Its scopes (with ``allow_unscoped_api``) decided the routes it reached,
    by 1.9.0's rules (``colonymind.api.legacy_scopes``): they are carried as they were.
    1.9.0 never limited a principal granted the owner to a platform (its declared platforms only told it
    which senders to attest), so it may write the owner from every platform."""
    mapped = {"owner": owner_id or "owner", "shared": shared_id, "global": global_id}
    persons = {str(item) for item in raw.get("person_ids") or [] if item}
    viewer = str(raw.get("viewer_person_id") or "").strip() or None
    if viewer:
        persons.add(viewer)
    persons |= {mapped[str(item)] for item in raw.get("audiences") or [] if str(item) in mapped}
    scopes = {str(item) for item in raw.get("scopes") or []}
    # 1.9.0 never limited a principal granted the owner to a platform: its declared platforms only told it
    # which senders to attest, so it wrote the owner's turns from every platform.
    owner: frozenset[str] = frozenset({ANY_PLATFORM}) if mapped["owner"] in persons else frozenset()
    unmapped = (["attested_contact_grants (the contacts its senders resolved to, as persons to name)"]
                if raw.get("attested_contact_grants") else [])
    return Authority(persons=frozenset(persons), default=viewer,
                     senders="*" in scopes or "turns:resolve-sender" in scopes, operations=frozenset(),
                     owner=owner, scopes=frozenset(scopes),
                     unscoped_api=raw.get("allow_unscoped_api", True) is not False), unmapped


def legacy_principals(home: Path, env: dict[str, str], owner_id: str = "") -> list[LegacyPrincipal]:
    """Every principal of a 1.9.0 keyring, with what it could do (``legacy_authority``) and each secret the
    1.9.0 sidecar still accepted with its expiry. A malformed keyring carries nothing, and so does the
    keyring of an instance that runs 1.10 or later (``colonymind.yaml``: ``is_legacy_instance``), which 1.10
    no longer honoured: only a 1.9.0 instance's keyring is migrated."""
    keyring = home / LEGACY_KEYRING
    if not keyring.is_file() or not is_legacy_instance(home):
        return []
    now = now_utc()
    owner = env.get("COLONYMIND_OWNER_PERSON_ID") or owner_id or "owner"
    lanes = (env.get("COLONYMIND_SHARED_PERSON_ID") or "shared", env.get("COLONYMIND_GLOBAL_PERSON_ID") or "global")
    found: list[LegacyPrincipal] = []
    try:
        document = json.loads(keyring.read_text(encoding="utf-8"))
        for raw in document.get("principals", []):
            if not isinstance(raw, dict):
                continue
            live = _keyring_accepts(raw, now)
            status = str(raw.get("status") or "active").lower()
            # Why the principal is not carried, should it have no live secret: a live principal only for
            # want of one; any other for its status, or its time being up.
            if live:
                reason = "no live secret"
            else:
                reason = "expired" if status in {"active", "retiring"} else status
            try:
                until = _keyring_time(raw.get("accept_until"))
            except ValueError:
                until = None
            secrets_ = []
            for credential in raw.get("credentials") or [] if live else []:
                if isinstance(credential, dict) and credential.get("secret") and _keyring_accepts(credential, now):
                    secrets_.append((str(credential["secret"]), _keyring_time(credential.get("accept_until"))))
            authority, unmapped = legacy_authority(raw, owner, *lanes)
            found.append(LegacyPrincipal(principal=str(raw.get("principal") or ""), reason=reason,
                                         accept_until=until, secrets=secrets_, authority=authority,
                                         unmapped=unmapped))
    except (OSError, ValueError, AttributeError, TypeError):
        return []
    return found


def _credential_name(principal: str, taken: set[str]) -> str:
    """A credential name for a 1.9.0 principal id: lowercase, the allowed characters, unique."""
    base = re.sub(r"[^a-z0-9_.-]+", "-", principal.strip().lower()).strip("-_.")[:56] or "principal"
    name, suffix = base, 2
    while name in taken or name in {INSTANCE_CREDENTIAL, DEVELOPMENT_PRINCIPAL}:
        name, suffix = f"{base}-{suffix}", suffix + 1
    return name


def _secret_entries(secrets_: list[tuple[str, _dt.datetime | None]]) -> list[dict[str, str]]:
    entries = []
    for secret, until in secrets_:
        entry = {"sha256": secret_digest(secret)}
        if until is not None:
            entry["accept_until"] = until.isoformat()
        entries.append(entry)
    return entries


def _exact_entry(item: LegacyPrincipal, owner_id: str) -> dict[str, Any]:
    """The grants that give a credential exactly ``item``'s authority: its persons by name (the owner through
    ``owner``), its viewer, its senders and its 1.9.0 scopes."""
    authority = item.authority
    entry: dict[str, Any] = {
        "owner": ANY_OWNER if authority.owner else [],
        "contacts": sorted(authority.persons - {owner_id}),
        "scopes": sorted(authority.scopes or ()),
    }
    if not authority.unscoped_api:
        entry["unscoped_api"] = False
    if authority.default:
        entry["viewer"] = authority.default
    if authority.senders:
        entry["senders"] = "all"
    return entry


@dataclasses.dataclass
class LegacyMigration:
    """What a 1.9.0 keyring becomes: the ``api.key`` secret (empty: generate one), the credentials and the
    lines of the upgrade report, one for every principal and every secret it moved."""

    key: str
    credentials: dict[str, Any]
    roles: dict[str, str]
    notes: list[str]


def migrate_legacy_credentials(home: Path, env: dict[str, str], owner_id: str = "") -> LegacyMigration:
    """The 1.9.0 principals as credentials, each with exactly the authority it had (docs/CREDENTIALS.md).

    Every live principal but the adapter's becomes a credential with the same persons (named in a list; the
    owner through its ``owner`` platforms), the same default person, the same senders, the same 1.9.0 scopes
    (which reach exactly the routes 1.9.0 gave them) and the same owner authority; what has no counterpart
    (its attested contact grants) is named in the report. The
    principal whose secret the adapter used (``COLONYMIND_CLIENT_API_KEY``, else the first live one) becomes
    the instance key, the Hermes adapter's, which writes and reads as anyone the sidecar resolves and reaches
    the whole API; it writes the owner from every platform, as the principal could, or from the default ones
    when the principal was not granted the owner. ``api.key`` is
    one of its secrets that never expires (a new one when every secret it has retires); its other live
    secrets, and a live ``COLONYMIND_API_KEY``, stay accepted as the instance key, each until its expiry.
    Every live secret is kept with its own expiry, bounded by the principal's. The report says, for every
    principal, what it could do before and what its credential may do after."""
    owner = env.get("COLONYMIND_OWNER_PERSON_ID") or owner_id or "owner"
    lanes = (env.get("COLONYMIND_SHARED_PERSON_ID") or "shared", env.get("COLONYMIND_GLOBAL_PERSON_ID") or "global")
    principals = legacy_principals(home, env, owner_id)
    client = env.get("COLONYMIND_CLIENT_API_KEY") or ""
    legacy_bearer = env.get("COLONYMIND_API_KEY") or ""
    carried = [item for item in principals if item.secrets]
    chosen = next((item for item in carried if any(secret == client for secret, _ in item.secrets)),
                  carried[0] if carried else None)
    grants: dict[str, Any] = default_credentials()
    roles: dict[str, str] = {}
    notes: list[str] = []
    lines: dict[int, str] = {}
    candidates: list[tuple[str, _dt.datetime | None, str]] = []   # the instance key's secrets, and whose
    if chosen is not None:
        ordered = sorted(chosen.secrets, key=lambda item: item[0] != client)     # the adapter's own first
        candidates += [(secret, _earliest(until, chosen.accept_until), "") for secret, until in ordered]
        roles[INSTANCE_CREDENTIAL] = f"api.key, the Hermes adapter's, from 1.9.0 principal {chosen.principal}"
        if chosen.authority.owner:
            grants[INSTANCE_CREDENTIAL]["owner"] = ANY_OWNER     # 1.9.0 limited it to no platform
    elif not principals and client:
        candidates.append((client, None, ""))            # no keyring: the configured key, as before
    if legacy_bearer and all(legacy_bearer != secret for secret, _, _ in candidates):
        candidates.append((legacy_bearer, None, "the 1.9.0 COLONYMIND_API_KEY"))
    key = next((secret for secret, until, _ in candidates if until is None), "")
    extra = [(secret, until) for secret, until, _ in candidates if secret != key]
    if extra:
        grants[INSTANCE_CREDENTIAL]["secrets"] = _secret_entries(extra)
    if candidates and not key:
        notes.append("api.key generated: every 1.9.0 secret of the adapter's principal retires, and each stays "
                     "accepted as the instance key until its time")
    notes += [f"{role} is accepted as the instance key (api-key)" for secret, _, role in candidates
              if role and secret != key]
    digests = {secret_digest(secret) for secret, _, _ in candidates}
    for index, item in enumerate(principals):
        if item is chosen:
            continue
        if not item.secrets:
            lines[index] = (f"1.9.0 principal {item.principal} ({item.reason}) is not carried: none of its "
                            "secrets is accepted")
            continue
        kept = [(secret, until) for secret, until in item.secrets if secret_digest(secret) not in digests]
        digests.update(secret_digest(secret) for secret, _ in kept)
        if not kept:
            lines[index] = (f"1.9.0 principal {item.principal} shares its secrets with the instance key and is "
                            "carried as api-key")
            continue
        name = _credential_name(item.principal, set(grants))
        entry: dict[str, Any] = ({"sha256": secret_digest(kept[0][0])} if len(kept) == 1 and kept[0][1] is None
                                 else {"secrets": _secret_entries(kept)})
        if item.accept_until is not None:
            entry["accept_until"] = item.accept_until.isoformat()
        entry.update(_exact_entry(item, owner))
        grants[name] = entry
        roles[name] = f"1.9.0 principal {item.principal}"
        lines[index] = name
    table = CredentialTable.from_config(grants)
    now = now_utc()

    def before_after(name: str, item: LegacyPrincipal, suffix: str = "") -> str:
        grant = table.get(name)
        after = authority_text(authority_of(grant, owner, *lanes), owner) + suffix + lifetime_note(grant, now)
        untouched = (f"; its 1.9.0 scopes {', '.join(item.unmapped)} have no counterpart and are not carried"
                     if item.unmapped and "manage" not in grant.access else "")
        return (f"{name} ({roles[name]}): before: {authority_text(item.authority, owner)}; after: {after}"
                f"{untouched}")
    report = []
    for index, item in enumerate(principals):
        if item is chosen:
            report.append(before_after(INSTANCE_CREDENTIAL, item, " (the instance key: the Hermes adapter's)"))
        elif lines.get(index) in grants:
            report.append(before_after(lines[index], item))
        elif index in lines:
            report.append(lines[index])
    return LegacyMigration(key=key, credentials=grants, roles=roles, notes=report + notes)


def prepared_runtime(python: str | Path | None) -> bool:
    """True when *python* belongs to a 1.9.0 prepared (patched) Hermes runtime."""
    if not python:
        return False
    path = Path(python).expanduser()
    for parent in (path, *path.parents):
        if (parent / ".colonymind-runtime.json").is_file() or (parent / ".colonymind-patch-receipt.json").is_file():
            return True
    return "/.local/share/colonymind/hermes/" in str(path)


def archive_legacy_forwarders(home: Path, hermes_home: Path, backup_dir: Path) -> list[str]:
    """Move the 1.9.0 private-directory forwarders out of ``<hermes_home>/plugins``.

    Stock Hermes takes a memory provider from that directory before the installed
    entry point, so a forwarder left behind keeps loading the 1.9.0 adapter copy.
    """
    markers = {str(home / "adapter"), str(home.resolve() / "adapter")}
    moved: list[str] = []
    for name in (PLUGIN_NAME, MEMORY_PROVIDER):
        directory = hermes_home / "plugins" / name
        try:
            source = (directory / "__init__.py").read_text(encoding="utf-8")
        except OSError:
            continue
        if not any(marker in source for marker in markers):
            continue
        destination = backup_dir / "hermes-plugins" / name
        from colonymind.backup import _make_tree_private, _private_directory
        _private_directory(destination.parent)
        shutil.move(str(directory), str(destination))
        _make_tree_private(destination)
        moved.append(f"{hermes_home / 'plugins' / name} moved to {destination} (a 1.9.0 forwarder)")
    return moved


def migrate_legacy_instance(home: Path, *, backup_dir: Path | None = None) -> list[str]:
    """Convert a 1.9.0 instance in place; returns the notes to print."""
    notes: list[str] = []
    manifest = json.loads((home / LEGACY_MANIFEST).read_text(encoding="utf-8"))
    env = _read_env_file(home / ".env")
    # Every principal's grants are resolved here, before any file is written or the sidecar starts.
    carried = migrate_legacy_credentials(home, env, env.get("COLONYMIND_OWNER_CONTACT_ID")
                                         or str(manifest.get("owner_id") or ""))
    if carried.key:
        write_api_key(carried.key, home)
        notes.append("api.key written from the 1.9.0 keyring")
    else:
        write_api_key(secrets.token_urlsafe(32), home)
        if not carried.notes or not carried.notes[0].startswith("api.key generated"):
            notes.append("api.key generated (no live 1.9.0 credential)")
    data = copy.deepcopy(configuration.DEFAULTS)
    data["sidecar"]["port"] = int(env.get("COLONYMIND_SIDECAR_PORT") or 7777)
    data["sidecar"]["host"] = env.get("COLONYMIND_SIDECAR_HOST") or "127.0.0.1"
    data["hermes"]["home"] = str(manifest.get("hermes_home") or "~/.hermes")
    python = str(manifest.get("hermes_python") or "")
    if prepared_runtime(python):
        data["hermes"]["python"] = ""
        stock = find_hermes_executable()
        stock_python = python_of_executable(stock) if stock else None
        target = str(stock_python) if stock_python else "<python of your stock hermes>"
        notes.append(
            "this instance used a prepared (patched) Hermes runtime; install into stock Hermes with:\n"
            f"    colonymind upgrade --hermes-python {target}"
        )
    else:
        data["hermes"]["python"] = python
    data["owner"]["contact_id"] = env.get("COLONYMIND_OWNER_CONTACT_ID") or str(manifest.get("owner_id") or "")
    data["mind"]["autonomy"] = "suggest"
    llm_path = home / LLM_CONFIG_FILE
    if llm_path.is_file():
        try:
            llm = json.loads(llm_path.read_text(encoding="utf-8"))
            data["router"]["base_url"] = str(llm.get("baseUrl") or "")
            models = llm.get("models") or {}
            data["router"]["model"] = str(models.get("medium") or models.get("large") or manifest.get("model") or "")
        except (OSError, ValueError, AttributeError):
            pass
    if not data["router"]["base_url"]:
        data["router"]["base_url"] = str(manifest.get("endpoint") or "")
        data["router"]["model"] = str(manifest.get("model") or "")
    data["credentials"] = carried.credentials
    save_config(data, home)
    notes.append("colonymind.yaml written; autonomy starts at 'suggest' (edit mind.autonomy to choose "
                 "'standard' or 'trusted')")
    preferences = manifest.get("agent_preferences") or {}
    try:
        values = json.loads(env.get("COLONYMIND_AGENT_VALUES") or "[]")
    except ValueError:
        values = []
    identity = {
        "owner": {"name": env.get("COLONYMIND_OWNER_NAME") or "Owner", "handles": []},
        "agent": {
            "name": env.get("COLONYMIND_PERSONA_NAME") or str(manifest.get("agent_name") or "Assistant"),
            "values": values or list(preferences.get("values") or []),
            "timezone": env.get("COLONYMIND_AGENT_TIMEZONE") or str(preferences.get("timezone") or ""),
            "quiet_hours": env.get("COLONYMIND_AGENT_QUIET_HOURS") or str(preferences.get("quiet_hours") or ""),
        },
    }
    save_identity(identity, home)
    notes.append("identity.yaml written from the 1.9.0 environment")
    notes.append("credential grants recorded in colonymind.yaml (credentials), one for every 1.9.0 principal, "
                 "each with exactly what it could do before; turns a credential may not write are refused with "
                 "403, never re-attributed:")
    notes.extend("  " + line for line in carried.notes)
    for name in (".env", LEGACY_KEYRING):
        path = home / name
        if path.is_file():
            path.rename(home / f"{name}.1.9.0")
            notes.append(f"{name} kept as {name}.1.9.0 (no longer read)")
    notes.extend(archive_legacy_forwarders(home, Path(data["hermes"]["home"]).expanduser(),
                                           backup_dir or home / BACKUPS_DIR / _utc_stamp()))
    return notes


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str, non_interactive: bool, ask: Callable[[str], str] | None = None) -> str:
    from colonymind.setup import _prompt
    value = _prompt(prompt, default, non_interactive, ask=ask).strip()
    if any(ord(char) < 32 for char in value):
        raise InitError("configuration values must fit on one line")
    return value


def valid_timezone(name: str) -> bool:
    """A named zone this interpreter knows (``Europe/Paris``, ``UTC``)."""
    from zoneinfo import ZoneInfo
    if not name or name != name.strip() or name.startswith(("/", ".")):
        return False
    try:
        ZoneInfo(name)
    except (ValueError, OSError, KeyError):
        return False
    return True


def system_timezone(localtime: Path = Path("/etc/localtime"), timezone_file: Path = Path("/etc/timezone")) -> str:
    """This machine's named zone: ``$TZ``, the zone ``/etc/localtime`` links to, or ``/etc/timezone``; "" when
    none names one (the agent's clock is then UTC)."""
    candidates = [os.environ.get("TZ", "").strip().lstrip(":")]
    try:
        target = str(localtime.resolve())
        if "zoneinfo/" in target and localtime.is_symlink():
            candidates.append(target.split("zoneinfo/", 1)[1])
    except OSError:
        pass
    try:
        candidates.append(timezone_file.read_text(encoding="utf-8").strip())
    except OSError:
        pass
    return next((name for name in candidates if valid_timezone(name)), "")


def _parse_handles(values: list[str] | None) -> list[dict[str, str]]:
    handles = []
    for item in values or []:
        platform, sep, address = item.partition("=")
        if not sep or not platform.strip() or not address.strip():
            raise InitError(f"--owner-handle expects PLATFORM=ID, got {item!r}")
        handles.append({"platform": platform.strip().lower(), "id": address.strip()})
    return handles


def _comma_list(raw: Any) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def current_quiet_hours(mind: str, answered: str, synced: str | None) -> str:
    """The quiet hours a rerun of init keeps when it is given no new answer. *mind* is ``mind.quiet_hours``
    (the policy the mind reads), *answered* identity.yaml's ``agent.quiet_hours``, *synced* the value init
    last wrote to both (``hermes-install.json``; None for an instance from a release that wrote only
    identity.yaml). When the two disagree, the one changed since init wrote them wins, the mind's if both
    were; with no record the mind still holding its default means the earlier answer never reached it."""
    if mind == answered:
        return mind
    if synced is not None:
        return answered if mind == synced else mind
    return answered if answered and mind == configuration.DEFAULTS["mind"]["quiet_hours"] else mind


def _collect_identity(args, existing: dict[str, Any], non_interactive: bool, *,
                      quiet_current: str | None = None) -> dict[str, Any]:
    """The owner's answers for ``identity.yaml``: the owner, and the agent's constitution (name, values,
    boundaries) plus its time zone (by default this machine's) and quiet hours (by default
    ``quiet_current``, the ones in force (``current_quiet_hours``), else the earlier answer). Keys init does
    not ask about (``owner.contact_id``, ``agent.interests``) are kept as they are. The rendered constitution
    must fit ``CONSTITUTION_CHARS``."""
    owner = dict(existing.get("owner") or {})
    agent = dict(existing.get("agent") or {})
    owner_name = _ask("Your name", getattr(args, "owner_name", None) or owner.get("name") or os.environ.get("USER", "Owner"),
                      non_interactive)
    handles = _parse_handles(getattr(args, "owner_handle", None)) or list(owner.get("handles") or [])
    if not non_interactive and not handles:
        raw = _ask("Your messaging handles, PLATFORM=ID separated by commas (optional)", "", non_interactive)
        handles = _parse_handles([part.strip() for part in raw.split(",") if part.strip()])
    agent_name = _ask("Agent name", getattr(args, "agent_name", None) or agent.get("name") or "Assistant", non_interactive)
    values_raw = getattr(args, "agent_values", None)
    if values_raw is None:
        values_raw = _ask("Guiding values, comma separated (optional)", ", ".join(agent.get("values") or []),
                          non_interactive)
    boundaries_raw = getattr(args, "agent_boundaries", None)
    if boundaries_raw is None:
        boundaries_raw = _ask("Boundaries the agent never crosses, comma separated (optional)",
                              ", ".join(agent.get("boundaries") or []), non_interactive)
    timezone = _ask("Time zone", getattr(args, "timezone", None) or agent.get("timezone") or system_timezone(),
                    non_interactive)
    if timezone and not valid_timezone(timezone):
        raise InitError(f"the time zone must be a zone name like Europe/Paris or UTC, not {timezone!r}")
    quiet = _ask("Quiet hours HH:MM-HH:MM (empty for none)",
                 getattr(args, "quiet_hours", None)
                 or (agent.get("quiet_hours") or "" if quiet_current is None else quiet_current), non_interactive)
    if quiet and not re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", quiet):
        raise InitError("quiet hours must look like 22:00-07:00")
    identity = {
        "owner": {**owner, "name": owner_name, "handles": handles},
        "agent": {**agent, "name": agent_name, "values": _comma_list(values_raw),
                  "boundaries": _comma_list(boundaries_raw), "timezone": timezone, "quiet_hours": quiet},
    }
    length = constitution_length(identity)
    if length > CONSTITUTION_CHARS:
        sizes = {name: len(render_constitution({"agent": {name: identity["agent"][name]}}))
                 for name in ("values", "boundaries")}
        longest = max(sizes, key=sizes.get)
        raise InitError(f"the constitution renders to {length} characters and must fit in {CONSTITUTION_CHARS}; "
                        f"shorten the {longest} ({sizes[longest]} characters rendered)")
    return identity


#: Hermes' name for its own terminal (the platform the provider is initialized on there).
CLI_PLATFORM = "cli"
# Run under the Hermes interpreter: the platform names its gateway knows, bundled plugin platforms included.
_PLATFORMS_PROBE = """
import json
from gateway.config import Platform
names = {member.value for member in Platform}
try:
    discovered = Platform._scan_bundled_plugin_platforms()
    if isinstance(discovered, tuple):
        discovered = discovered[0]
    names |= set(discovered)
except Exception:
    pass
print(json.dumps(sorted(names)))
"""


def hermes_platform_names(python: Path) -> list[str]:
    """The platform names this Hermes knows (its terminal, ``cli``, first); [] when it cannot say."""
    try:
        result = subprocess.run([str(python), "-I", "-c", _PLATFORMS_PROBE], capture_output=True, text=True,
                                timeout=120, check=False)
        names = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return []
    if not isinstance(names, list):
        return []
    return [CLI_PLATFORM, *sorted(str(name) for name in names if str(name) != CLI_PLATFORM)]


def derived_owner_platforms(identity: dict[str, Any]) -> list[str]:
    """The platforms the owner's own turns come from: this terminal, and each platform the owner gave a
    handle on."""
    return list(dict.fromkeys([CLI_PLATFORM, *owner_handle_platforms(identity)]))


def _collect_owner_platforms(args, identity: dict[str, Any], recorded: list[str], non_interactive: bool, *,
                             fresh: bool, python: Path) -> tuple[list[str], list[str]]:
    """The platforms the instance key may write the owner's turns from, and a note per name Hermes does not
    know. A flag gives the list in full; else a fresh instance takes the owner's handles' platforms and the
    terminal, and an existing one keeps its list; interactive init asks, with that as the answer offered."""
    explicit = getattr(args, "owner_platform", None)
    known = hermes_platform_names(python)
    if explicit:
        chosen = [str(item).strip().lower() for item in explicit if str(item).strip()]
    else:
        chosen = derived_owner_platforms(identity) if fresh else list(recorded)
        if not non_interactive:
            raw = _ask("Platforms your own messages reach the agent from, comma separated"
                       + (f" (Hermes knows {', '.join(known)})" if known else ""), ", ".join(chosen), non_interactive)
            chosen = [part.strip().lower() for part in raw.split(",") if part.strip()]
    chosen = list(dict.fromkeys(chosen))
    notes = [f"Hermes has no platform named {name}: the owner's turns from it arrive only through a plugin that "
             "registers that name" for name in chosen if known and name not in known]
    return chosen, notes


# ---------------------------------------------------------------------------
# Day-one history: the owner's own direct sessions in Hermes
# ---------------------------------------------------------------------------

HISTORY_BATCH = 10000
_HISTORY_BINDINGS = 500       # the reviewed importer's most bindings per mapping


def owner_history_sessions(state_db: Path, handles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hermes' direct sessions whose recorded origin is one of the owner's handles (platform and sender id
    exactly, a direct chat, not an automation source, no recorded origin saying otherwise), with their message
    counts. Read-only; [] when Hermes keeps no such database."""
    from colonymind.turns.hermes_history import AUTOMATION
    pairs = {(str(item.get("platform") or "").strip().lower(), str(item.get("id") or "").strip())
             for item in handles if isinstance(item, dict)}
    if not state_db.is_file() or not pairs:
        return []
    try:
        with closing(sqlite3.connect(state_db.as_uri() + "?mode=ro", uri=True)) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            if not {"id", "source", "user_id", "chat_id", "chat_type"} <= columns:
                return []
            origin = "s.origin_json" if "origin_json" in columns else "NULL"
            rows = conn.execute(f"""SELECT s.id, s.source, s.user_id, s.chat_id, s.chat_type, {origin},
                                           count(m.id) FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                                    GROUP BY s.id ORDER BY s.id""").fetchall()
    except sqlite3.Error:
        return []
    found = []
    for session_id, source, actor, chat, kind, recorded, messages in rows:
        owner = (str(source or "").lower(), str(actor or "")) in pairs
        if kind != "dm" or not chat or source in AUTOMATION or not owner:
            continue
        try:
            said = json.loads(recorded) if recorded else {}
        except ValueError:
            continue
        expected = {"platform": source, "user_id": actor, "chat_id": chat, "chat_type": kind}
        if not isinstance(said, dict) or any(said.get(key) and said[key] != value for key, value in expected.items()):
            continue
        found.append({"session_id": session_id, "platform": source, "actor_id": actor, "chat_id": chat,
                      "chat_type": "dm", "messages": messages})
    return found


def history_namespace(hermes_home: Path) -> str:
    """The importer's stable namespace for one Hermes installation."""
    return "hermes-" + hashlib.sha256(str(hermes_home.expanduser().resolve()).encode()).hexdigest()[:16]


def import_owner_history(home: Path, hermes_home: Path, sessions: list[dict[str, Any]],
                         contact_id: str) -> dict[str, int]:
    """The reviewed importer over a consistent copy of Hermes' database: the owner's own direct sessions,
    bound to the owner by the handles the owner gave init, as quotations only (no claims, no learning, no
    model call). The copy is removed afterwards. Returns the importer's counts."""
    import tempfile
    from collections import Counter
    from colonymind.turns.hermes_history import import_history
    totals: Counter = Counter()
    with tempfile.TemporaryDirectory(dir=home, prefix=".history-") as scratch:
        snapshot = Path(scratch) / "state.db"
        with closing(sqlite3.connect((hermes_home / "state.db").as_uri() + "?mode=ro", uri=True)) as source, \
                closing(sqlite3.connect(snapshot)) as target:
            source.backup(target)
        for start in range(0, len(sessions), _HISTORY_BINDINGS):
            mapping = {"version": 1, "namespace": history_namespace(hermes_home), "reviewed": True, "bindings": [
                {**{key: item[key] for key in ("session_id", "platform", "actor_id", "chat_id", "chat_type")},
                 "contact_id": contact_id,
                 "review_evidence": {"kind": "owner_handle_at_init",
                                     "reference": f"the owner's {item['platform']} handle given to colonymind init"}}
                for item in sessions[start:start + _HISTORY_BINDINGS]]}
            while True:
                result = import_history(snapshot, mapping, state_dir=home, apply=True, limit=HISTORY_BATCH)
                if result["complete"]:
                    break
            totals.update(result["counts"])
    return dict(totals)


def _history_step(args, home: Path, hermes_home: Path, identity: dict[str, Any], contact_id: str, *,
                  non_interactive: bool) -> list[str]:
    """Offer the owner's own history: a yes in interactive init, or ``--import-history``, imports it."""
    sessions = owner_history_sessions(hermes_home / "state.db", list(identity.get("owner", {}).get("handles") or []))
    if not sessions:
        return []
    per_handle: dict[str, int] = {}
    for item in sessions:
        handle = f"{item['platform']} {item['actor_id']}"
        per_handle[handle] = per_handle.get(handle, 0) + int(item["messages"])
    counted = "; ".join(f"{handle}: {count} message{'' if count == 1 else 's'}" for handle, count in per_handle.items())
    summary = f"{len(sessions)} direct session{'' if len(sessions) == 1 else 's'} from your own handles ({counted})"
    wanted = bool(getattr(args, "import_history", False))
    if not wanted and not non_interactive:
        _say(f"  Hermes holds {summary}.")
        wanted = _ask("Import their messages into memory, as quotations only: no claims, no learning, no model "
                      "call (y/n)", "n", non_interactive).lower().startswith("y")
    if not wanted:
        return [f"history: Hermes holds {summary}; 'colonymind init --import-history' imports them as quotations"]
    counts = import_owner_history(home, hermes_home, sessions, contact_id)
    kept = counts.get("retained_new", 0) + counts.get("retained_existing", 0)
    left = sum(value for key, value in counts.items() if not key.startswith("retained"))
    return [f"history: {summary} imported as quotations: {kept} kept ({counts.get('retained_new', 0)} new), {left} "
            "left in Hermes (tool calls, summaries and the like)"]


def _collect_autonomy(args, current: str, non_interactive: bool, *, fresh: bool) -> str:
    explicit = getattr(args, "autonomy", None)
    if explicit:
        level = str(explicit).strip().lower()
    elif fresh and not non_interactive:
        level = _ask("Autonomy level (off, suggest, standard, trusted)", current or "suggest", non_interactive).lower()
    else:
        level = current or "standard"
    if level not in AUTONOMY_LEVELS:
        raise InitError(f"autonomy must be one of {', '.join(AUTONOMY_LEVELS)}")
    return level


def _reconcile(cfg: Config, hermes_home: Path, *, backup_dir: Path) -> list[str]:
    """Steps 5 and 6: the Hermes keys and the worker profile. Returns the changes.

    Selecting ColonyMind's memory provider replaces whatever Hermes used: the replaced name (or "" for
    none) and the backup of that config are recorded in ``hermes-install.json``, the change says so, and
    ``colonymind init --uninstall`` puts it back."""
    config_path = hermes_home / "config.yaml"
    current = read_hermes_config(config_path)
    replaced = selected_memory_provider(current)
    skills_dir = cfg.home / SKILLS_DIR
    skills_dir.mkdir(parents=True, exist_ok=True)
    key_file = cfg.home / KEY_FILE
    updated, changes = reconcile_hermes_config(current, sidecar_url=cfg.sidecar_url, key_file=key_file,
                                               skills_dir=skills_dir)
    if changes:
        backup = write_hermes_config(config_path, updated, backup_dir=backup_dir)
        switch = f"memory.provider: {MEMORY_PROVIDER}"
        if switch in changes:
            write_install_state(cfg.home, replaced_memory_provider=replaced,
                                hermes_config_backup=str(backup.relative_to(cfg.home)) if backup is not None
                                and backup.is_relative_to(cfg.home) else str(backup or ""))
            changes[changes.index(switch)] = (
                f"memory.provider: {MEMORY_PROVIDER} (replaces memory provider {replaced}; 'colonymind init "
                "--uninstall' restores it)" if replaced else
                f"memory.provider: {MEMORY_PROVIDER} (none was set; 'colonymind init --uninstall' removes it)")
    profile = worker_profile_config(updated, cfg, sidecar_url=cfg.sidecar_url, key_file=key_file)
    kept = merged_worker_profile(profiles_root(hermes_home), profile)[1]
    if write_worker_profile(profiles_root(hermes_home), profile):
        changes.append(f"profiles/{WORKER_PROFILE} written"
                       + (f"; kept the keys ColonyMind does not write: {', '.join(kept)}" if kept else ""))
    return changes


def _adapter_step(python: Path, *, source: str | None, home: Path) -> list[str]:
    """Install the matching adapter, record both versions in ``hermes-install.json`` (the memory provider
    compares them without the network) and run ``pip check``."""
    notes: list[str] = []
    if install_adapter(python, __version__, source=source):
        notes.append(f"{ADAPTER_DISTRIBUTION} {installed_adapter_version(python)} installed into {python}")
    write_install_state(home, sidecar_version=__version__, adapter_version=installed_adapter_version(python) or "")
    ok, output = pip_check(python)
    if ok:
        notes.append(f"pip check clean in {python}")
    else:
        # An incompatibility that predates the adapter is Hermes' to fix; say so and go on.
        detail = " | ".join(line for line in output.splitlines() if line.strip())
        notes.append(f"pip check reports an incompatibility in {python} ({detail}); "
                     "'colonymind doctor' keeps flagging it until Hermes' environment is repaired")
    return notes


#: ``colonymind init``'s answers from the environment, for a run with no terminal (a container's entrypoint): each
#: variable answers its flag when the flag is not given. Lists are comma separated. The model endpoint's key is
#: ``COLONYMIND_MODEL_API_KEY`` and ``COLONYMIND_INIT_NO_SERVICE`` skips the service, as before.
INIT_ENVIRONMENT: dict[str, str] = {
    "owner_name": "COLONYMIND_INIT_OWNER_NAME",
    "owner_handle": "COLONYMIND_INIT_OWNER_HANDLES",
    "owner_platform": "COLONYMIND_INIT_OWNER_PLATFORMS",
    "agent_name": "COLONYMIND_INIT_AGENT_NAME",
    "agent_values": "COLONYMIND_INIT_AGENT_VALUES",
    "agent_boundaries": "COLONYMIND_INIT_AGENT_BOUNDARIES",
    "timezone": "COLONYMIND_INIT_TIMEZONE",
    "quiet_hours": "COLONYMIND_INIT_QUIET_HOURS",
    "autonomy": "COLONYMIND_INIT_AUTONOMY",
    "hermes_python": "COLONYMIND_INIT_HERMES_PYTHON",
    "host": "COLONYMIND_INIT_HOST",
    "port": "COLONYMIND_INIT_PORT",
    "model_url": "COLONYMIND_INIT_MODEL_URL",
    "model": "COLONYMIND_INIT_MODEL",
    "embed_url": "COLONYMIND_INIT_EMBED_URL",
    "embed_model": "COLONYMIND_INIT_EMBED_MODEL",
    "embed_dims": "COLONYMIND_INIT_EMBED_DIMS",
    "rerank_model": "COLONYMIND_INIT_RERANK_MODEL",
    "adapter_source": "COLONYMIND_INIT_ADAPTER_SOURCE",
    "import_history": "COLONYMIND_INIT_IMPORT_HISTORY",
}
INIT_NON_INTERACTIVE = "COLONYMIND_INIT_NON_INTERACTIVE"
_INIT_LISTS = frozenset({"owner_handle", "owner_platform"})
_INIT_INTEGERS = frozenset({"port", "embed_dims"})
_INIT_SWITCHES = frozenset({"import_history"})


def with_environment(args, environ: dict[str, str] | None = None):
    """``args`` with every answer it leaves out taken from ``INIT_ENVIRONMENT``; a flag always wins."""
    environ = os.environ if environ is None else environ
    values = dict(vars(args))
    for name, variable in INIT_ENVIRONMENT.items():
        raw = str(environ.get(variable) or "").strip()
        if not raw or values.get(name) not in (None, "", [], False):
            continue
        if name in _INIT_LISTS:
            values[name] = _comma_list(raw)
        elif name in _INIT_SWITCHES:
            values[name] = raw.lower() in {"1", "true", "yes", "on"}
        elif name in _INIT_INTEGERS:
            try:
                values[name] = int(raw)
            except ValueError:
                raise InitError(f"{variable} must be a whole number, not {raw!r}") from None
        else:
            values[name] = raw
    if str(environ.get(INIT_NON_INTERACTIVE) or "").strip().lower() in {"1", "true", "yes", "on"}:
        values["non_interactive"] = True
    return type(args)(**values)


def run_init(args) -> int:
    """``colonymind init``. Returns the exit code."""
    try:
        args = with_environment(args)
    except InitError as exc:
        _say(f"colonymind init failed: {exc}")
        return 1
    if getattr(args, "uninstall", False):
        return run_uninstall(args)
    non_interactive = bool(getattr(args, "non_interactive", False))
    try:
        if not vector_store_available():
            raise InitError(VECTOR_STORE_REMEDY)
        home = configuration.instance_home(getattr(args, "home", None))
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        if is_legacy_instance(home):
            for note in migrate_legacy_instance(home):
                _say("  " + note)
        cfg = load_config(home)
        fresh = not cfg.exists
        data = copy.deepcopy(cfg.data)
        seeded = str(cfg.get("owner.contact_id") or "").strip()
        if seeded and (problem := seeded_owner_problem(home, seeded)):
            raise InitError(problem)

        # 1. Identity and autonomy. The mind reads the quiet hours from mind.quiet_hours; identity.yaml keeps
        # the same answer for the follow-up scheduler. Without a new answer the ones in force are kept.
        existing_identity = load_identity(home)
        synced = read_install_state(home).get("quiet_hours")
        identity = _collect_identity(args, existing_identity, non_interactive, quiet_current=current_quiet_hours(
            str(data["mind"].get("quiet_hours") or ""),
            str((existing_identity.get("agent") or {}).get("quiet_hours") or ""),
            None if synced is None else str(synced)))
        data["mind"]["quiet_hours"] = identity["agent"]["quiet_hours"]
        data["mind"]["autonomy"] = _collect_autonomy(args, data["mind"].get("autonomy", "standard"),
                                                     non_interactive, fresh=fresh)
        hermes_home = resolve_hermes_home(getattr(args, "hermes_home", None), data["hermes"].get("home"))
        python, hermes_version = resolve_hermes_python(getattr(args, "hermes_python", None),
                                                       data["hermes"].get("python"), hermes_home)
        _say(f"  Hermes home {hermes_home}: adapter environment {python}")
        owner_platforms, platform_notes = _collect_owner_platforms(
            args, identity, list(data["credentials"][INSTANCE_CREDENTIAL].get("owner") or []), non_interactive,
            fresh=fresh, python=python)
        data["hermes"]["home"] = str(hermes_home)
        data["hermes"]["python"] = str(python)
        if getattr(args, "host", None):
            data["sidecar"]["host"] = str(args.host)
        if getattr(args, "port", None):
            data["sidecar"]["port"] = int(args.port)

        # 3. Router: the endpoint Hermes uses, plus an embedding endpoint when one exists.
        main_config = read_hermes_config(hermes_home / "config.yaml")
        endpoint = hermes_model_endpoint(hermes_home, main_config, python=python)
        base_url = getattr(args, "model_url", None) or data["router"].get("base_url") or ""
        if not base_url:
            if problem := endpoint_problem(endpoint):
                raise InitError(problem)
            base_url = endpoint["base_url"]
        model = getattr(args, "model", None) or data["router"].get("model") or endpoint["model"]
        api_key = getattr(args, "model_key", None) or os.environ.get("COLONYMIND_MODEL_API_KEY") or endpoint["api_key"]
        data["router"]["base_url"] = base_url
        data["router"]["model"] = model
        recall_notes = _recall_routes(args, data, home=home, base_url=base_url, model=model, api_key=api_key,
                                      non_interactive=non_interactive)
        # The platforms the instance key (the adapter's) may write the owner's turns from, in full.
        data["credentials"][INSTANCE_CREDENTIAL]["owner"] = owner_platforms

        # 2. colonymind.yaml, identity.yaml and api.key.
        save_identity(identity, home)
        _say(f"  identity.yaml written; the constitution the agent is given: {render_constitution(identity)}")
        if read_api_key(home, environ={}) is None:
            write_api_key(secrets.token_urlsafe(32), home)
            _say(f"  api.key written ({home / KEY_FILE}, mode 600)")
        save_config(data, home)
        write_install_state(home, quiet_hours=identity["agent"]["quiet_hours"])
        cfg = load_config(home)
        contact_id, created = ensure_owner_contact(home, cfg, identity)
        if created or cfg.get("owner.contact_id") != contact_id:
            cfg.data["owner"]["contact_id"] = contact_id
            save_config(cfg.data, home)
            cfg = load_config(home)
            _say(f"  owner contact {contact_id} recorded")
        for line in credential_report(cfg):
            _say("  credentials: " + line)
        for note in platform_notes:
            _say("  credentials: " + note)
        if write_llm_config(home, base_url=base_url, model=model, api_key=api_key):
            _say(f"  router pointed at {base_url} ({model or 'model chosen by Hermes'})")
        for note in recall_notes:
            _say("  " + note)
        _say("  " + semantic_recall_line(cfg))
        for note in _history_step(args, home, hermes_home, identity, contact_id, non_interactive=non_interactive):
            _say("  " + note)

        # 4. The adapter in Hermes' environment.
        _say(f"  Hermes {hermes_version} at {python}")
        for note in _adapter_step(python, source=getattr(args, "adapter_source", None), home=home):
            _say("  " + note)

        # 5 and 6. Hermes keys and the worker profile.
        for change in _reconcile(cfg, hermes_home, backup_dir=home / BACKUPS_DIR):
            _say("  " + change)
        _say("  plugins.hook_callback_timeout: 0 is Hermes-wide: every plugin's callbacks run inline, with no "
             "timeout, so overlapping capture and guard calls are never dropped")
        _say("  protected_instruction_extra_patterns match by basename in any directory: writes to any "
             "colonymind.yaml, identity.yaml or api.key need a human")

        # 7. The sidecar user service.
        if getattr(args, "no_service", False):
            _say("  sidecar service skipped (--no-service)")
        else:
            _say("  " + install_service(cfg))
    except (InitError, ConfigError) as exc:
        _say(f"colonymind init failed: {exc}")
        return 1
    except (OSError, subprocess.SubprocessError) as exc:
        _say(f"colonymind init failed: {type(exc).__name__}: {exc}")
        return 1
    _say(f"ColonyMind {__version__} is configured in {home}.")
    _say("Restart the gateway to load the adapter: hermes gateway restart")
    return 0


def semantic_recall_line(cfg: Config) -> str:
    """Which recall the configuration gives: by meaning (an endpoint, or the local embedder) or by keywords."""
    local = cfg.get("router.embed_provider") == EMBED_LOCAL
    embedder = (f"the local embedder {cfg.get('router.embed_model')}" if local
                else f"the embedding endpoint {cfg.get('router.embed_url')}")
    if (local or cfg.get("router.embed_url")) and cfg.get("mind.faculties.semantic_recall") is False:
        return ("semantic recall off (mind.faculties.semantic_recall is false; set it to true, or pass "
                f"--embed-url, to use {embedder})")
    if local:
        return f"semantic recall on (local embedder {cfg.get('router.embed_model')}, run by the sidecar)"
    if cfg.get("router.embed_url"):
        return f"semantic recall on (embedding endpoint {cfg.get('router.embed_url')})"
    return ("semantic recall off (no embedder recorded): keyword recall only; 'colonymind init --embed-url auto' "
            "asks the model endpoint for one, and 'colonymind init --embed-url local' runs one in the sidecar")


def semantic_recall_note(cfg: Config) -> str | None:
    """Releases before the switch was live wrote ``semantic_recall: false`` whenever init found no
    endpoint; with an embedder recorded since, recall stays keyword-only until the owner says otherwise."""
    local = cfg.get("router.embed_provider") == EMBED_LOCAL
    if (local or cfg.get("router.embed_url")) and cfg.get("mind.faculties.semantic_recall") is False:
        setting = "router.embed_provider is local" if local else "router.embed_url is set"
        return (f"semantic recall is off although {setting}: set mind.faculties.semantic_recall: "
                "true in colonymind.yaml (or re-run 'colonymind init --embed-url ...') to use the embedder")
    return None


def decisions_pending(home: Path) -> bool:
    """Whether ``colonymind.yaml`` still holds the 1.11 ``decisions`` section (``upgrade_decisions`` carries it)."""
    path = home / CONFIG_FILE
    if not path.is_file():
        return False
    try:
        return "decisions" in configuration.read_raw(path)
    except ConfigError:
        return False


def _carried_decisions(home: Path) -> tuple[dict[str, Any], Any, list[str]] | None:
    """``colonymind.yaml`` as it reads once its 1.11 ``decisions`` section is carried into ``instinct``
    (``colonymind.instinct.settings.carry_decisions``), with the states, the local records and the notes; None when
    there is no such section. Nothing is written. An ``instinct`` section already there is kept."""
    from colonymind.instinct import settings as instinct_settings
    path = home / CONFIG_FILE
    raw = configuration.read_raw(path) if path.is_file() else {}
    if "decisions" not in raw:
        return None
    try:
        carried = instinct_settings.carry_decisions(raw.get("decisions"))
    except instinct_settings.SettingsError as error:
        raise ConfigError(f"{error}; correct it in {path} and run 'colonymind upgrade' again") from None
    rewritten = {key: value for key, value in raw.items() if key != "decisions"}
    notes = list(carried.notes)
    if isinstance(rewritten.get("instinct"), dict):
        notes = ["the decisions section was dropped: colonymind.yaml already has an instinct section, which is kept"]
    else:
        rewritten["instinct"] = carried.section
    from colonymind.removed_settings import ignore_removed_settings
    configuration.validate(configuration._merge(configuration.DEFAULTS, ignore_removed_settings(copy.deepcopy(rewritten))))
    return rewritten, carried, notes


def carried_config(home: Path) -> Config:
    """The configuration as ``load_config`` reads it once the decisions section is carried, read without writing
    anything: ``colonymind upgrade`` needs it before it stops the service, and writes the carry only after."""
    prepared = _carried_decisions(home)
    if prepared is None:
        return load_config(home, required=True)
    from colonymind.removed_settings import ignore_removed_settings
    data = configuration._apply_env_overrides(configuration._merge(configuration.DEFAULTS, ignore_removed_settings(copy.deepcopy(prepared[0]))),
                                              dict(os.environ))
    return Config(home=home, data=configuration.validate(data), exists=True)


def upgrade_decisions(home: Path) -> list[str]:
    """Carry a 1.11 ``decisions`` section into ``instinct``, once (``_carried_decisions``): the states and local
    calibration records go into ``instinct.db`` first, then the file is rewritten without ``decisions`` (the rest
    of it as it was). ``colonymind upgrade`` calls it only with the service stopped and the backup taken. Returns
    the notes."""
    from colonymind.instinct.log import DB_FILE, Store
    prepared = _carried_decisions(home)
    if prepared is None:
        return []
    rewritten, carried, notes = prepared
    if carried.states or carried.records:
        store = Store(home / DB_FILE)
        for (point, tier), state in carried.states.items():
            store.set_state(point, tier, state, source="the 1.11 decisions section",
                            evidence={"setting": f"decisions.points.{point}.enabled"})
        for record in carried.records:
            store.put_record(record)
    configuration._atomic_write(home / CONFIG_FILE, configuration.dump_config(rewritten).encode("utf-8"), mode=0o600)
    return notes


OLD_INIT_HANDS_LIMITS = {
    "hands.cascade.max_per_task": 2,
    "hands.cascade.max_per_hour": 6,
    "hands.retention.events_days": 30,
    "hands.retention.rows_days": 180,
    "hands.obligations.overdue_seconds": 86400,
}


def hands_settings_upgrade(home: Path, *, write: bool = False) -> list[str]:
    """Drop inert keys and exact old-init limits; the caller backs up before writing."""
    from colonymind.removed_settings import removed_settings, remove_settings
    path = home / CONFIG_FILE
    raw = configuration.read_raw(path)
    retired = removed_settings(raw)
    pinned = []
    for key, old in OLD_INIT_HANDS_LIMITS.items():
        value = raw
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            if type(value) is int and value == old:
                pinned.append(key)
    if write and (retired or pinned):
        remove_settings(raw, retired + pinned)
        configuration._atomic_write(path, configuration.dump_config(raw).encode("utf-8"), mode=0o600)
    return ([f"removed retired setting {key}" for key in retired]
            + [f"removed old init default {key}={OLD_INIT_HANDS_LIMITS[key]}; staging default applies"
               for key in pinned])


def run_upgrade(args) -> int:
    """``colonymind upgrade``: stop the service, backup, migrations, adapter, config reconcile, retirement, start.

    The stores of removed code move aside last, once everything else succeeded, and only with this
    instance's writers stopped: a running user service is stopped (and waited for) before the backup,
    and a sidecar running outside it blocks the retirement. A failure leaves every store where the
    sidecar opens it, and a service the upgrade stopped runs again.
    """
    stopped, cfg, home = False, None, None
    conversion = None
    notes: list[str] = []
    plan: list[str] = []
    retiring: list[str] = []
    step = "reading the instance and Hermes' configuration"
    try:
        if not vector_store_available():
            raise InitError(VECTOR_STORE_REMEDY)
        home = configuration.instance_home(getattr(args, "home", None))
        if is_legacy_instance(home):
            step = "converting the 1.9.0 instance"
            backup = backup_instance(home)
            notes.append(f"backup taken in {backup}")
            notes.extend(migrate_legacy_instance(home, backup_dir=backup))
        # The loader refuses a leftover decisions section: until the service is stopped and the backup taken, the
        # configuration is read as it will be once the section is carried, and nothing is written.
        carrying = decisions_pending(home)
        hands_pending = hands_settings_upgrade(home)
        step = "reading the instance and Hermes' configuration"
        cfg = carried_config(home) if carrying else load_config(home, required=True)
        hermes_home = resolve_hermes_home(getattr(args, "hermes_home", None), cfg.get("hermes.home"))
        python, hermes_version = resolve_hermes_python(getattr(args, "hermes_python", None),
                                                       cfg.get("hermes.python"), hermes_home)
        binding_changed = (str(python) != cfg.get("hermes.python")) or (str(hermes_home) != cfg.get("hermes.home"))
        adapter_pending = installed_adapter_version(python) != __version__ or getattr(args, "adapter_source", None)
        current = read_hermes_config(hermes_home / "config.yaml")
        skills_dir = cfg.home / SKILLS_DIR
        updated, config_changes = reconcile_hermes_config(current, sidecar_url=cfg.sidecar_url,
                                                          key_file=cfg.home / KEY_FILE, skills_dir=skills_dir)
        profile_pending = worker_profile_pending(
            profiles_root(hermes_home),
            worker_profile_config(updated, cfg, sidecar_url=cfg.sidecar_url, key_file=cfg.home / KEY_FILE))
        migrations_pending = (pending_store_migrations(home) + pending_initiative_columns(home)
                              + retired_state_present(home)
                              + retired_tables_present(home) + pending_ingress_adoption(home)
                              + pending_credential_grants(home))
        lexical = semantic_recall_note(cfg)
        skills_pending, skills_edited = diagnostic_skills()
        reports = skills_edited + ([lexical] if lexical else [])
        # The conversion is an explicit upgrade step, never a service-command fallback.
        # A service-only upgrade must not stop/reinstall an existing instance service.
        global_service = _service(cfg).global_launchd_path(reports) if sys.platform == "darwin" else None
        if not (notes or carrying or hands_pending or binding_changed or adapter_pending or config_changes or profile_pending
                or migrations_pending or skills_pending):
            if global_service:
                step = "converting the old global service"
                conversion = _service(cfg).prepare_global_launchd_conversion(global_service)
                stopped = conversion.start_required  # Before bootout: failures and interrupts recover too.
                conversion.perform()
                _say("  old global service converted to the per-instance manager")
                if stopped:
                    _say("  " + conversion.start())
                    stopped = False
            else:
                _say(f"ColonyMind {__version__}: nothing to do.")
            for note in retired_mcp_left(hermes_home / "config.yaml") + reports:
                _say("  " + note)
            return 0
        plan = [*(["converting the old global service"] if global_service else []),
                "stopping the sidecar service", "checking that no other sidecar writes the stores",
                "taking the backup", *(["carrying the decisions section into instinct"] if carrying else []),
                *(["removing retired hands settings and old init limits"] if hands_pending else []),
                "applying the store migrations",
                *(["recording the Hermes binding"] if binding_changed else []), "installing the adapter",
                "reconciling the Hermes config and the worker profile", "moving aside the retired stores"]
        if global_service:
            step = "converting the old global service"
            conversion = _service(cfg).prepare_global_launchd_conversion(global_service)
            stopped = conversion.start_required
            conversion.perform()
            notes.append("old global service converted to the per-instance manager")
        step = "stopping the sidecar service"
        # Set before the stop is issued: an interrupt while it runs still starts the service again.
        stopped = stopped or service_running_for_upgrade(cfg)
        if stopped:
            stop_service_for_upgrade(cfg)
        step = "checking that no other sidecar writes the stores"
        retiring = retired_state_present(home)
        retiring_tables = retired_tables_present(home)
        if (retiring or retiring_tables) and (running := running_outside_service(cfg)):
            raise InitError(f"{running}, outside the sidecar service, and this upgrade moves aside stores it may "
                            f"still write ({', '.join(retiring + retiring_tables)}); stop it ('colonymind stop') and "
                            "run 'colonymind upgrade' again")
        step = "taking the backup"
        if not any(note.startswith("backup taken") for note in notes):
            backup = backup_instance(home)
            notes.insert(0, f"backup taken in {backup}")
        if hands_pending:
            step = "removing retired hands settings and old init limits"
            notes.extend(hands_settings_upgrade(home, write=True))
        if carrying:
            step = "carrying the decisions section into instinct"
            notes.extend(upgrade_decisions(home))
        step = "applying the store migrations"
        notes.extend("migration applied: " + item for item in run_store_migrations(home))
        notes.extend(migrate_initiatives(home))
        notes.extend(adopt_ingress_producers(home))
        notes.extend(record_credential_grants(home))
        cfg = load_config(home, required=True)
        if binding_changed:
            step = "recording the Hermes binding"
            cfg.data["hermes"]["python"] = str(python)
            cfg.data["hermes"]["home"] = str(hermes_home)
            save_config(cfg.data, home)
            cfg = load_config(home)
            notes.append(f"Hermes binding recorded: {python} ({hermes_version}) in {hermes_home}")
        step = "installing the adapter"
        notes.extend(_adapter_step(python, source=getattr(args, "adapter_source", None), home=home))
        step = "reconciling the Hermes config and the worker profile"
        notes.extend(_reconcile(cfg, hermes_home, backup_dir=home / BACKUPS_DIR))
        if colonymind_in_hermes_environment(python):
            notes.append(
                "the 'colonymind' sidecar package is installed inside Hermes' environment; move it out with:\n"
                f"    {python} -m pip uninstall colonymind\n    uv tool install --python 3.12 colonymind"
            )
        # Last, once everything else succeeded: a failure above leaves every store in place.
        step = "moving aside the retired stores"
        notes.extend(retire_state(home, backup))
        notes.extend(retire_tables(home))
        notes.extend(retire_diagnostic_skills(backup))
        step, plan = "starting the sidecar service", []
        if stopped:
            # Cleared only once the start returned: an interrupt before that still starts it in the finally.
            notes.append(conversion.start() if conversion else start_service_after_upgrade(cfg))
            stopped = False
        else:
            notes.append(restart_service(cfg))
        notes.extend(retired_mcp_left(hermes_home / "config.yaml") + reports)
    except BaseException as exc:
        # Whatever failed, and however: say what changed and what did not; the finally below starts the
        # service this upgrade stopped. An interrupt still stops the command once that is done.
        for line in _upgrade_failure(exc, step=step, notes=notes, plan=plan, home=home, retiring=retiring):
            _say(line)
        if not isinstance(exc, Exception):
            raise
        return 1
    finally:
        if stopped:
            _say("  " + (conversion.start() if conversion else start_service_after_upgrade(cfg)))
    for note in notes:
        _say("  " + note)
    _say(f"ColonyMind {__version__} is up to date in {home}.")
    _say("Restart the gateway to load the adapter: hermes gateway restart")
    return 0


def _upgrade_failure(exc: BaseException, *, step: str, notes: list[str], plan: list[str], home: Path | None,
                     retiring: list[str]) -> list[str]:
    """The failure report: the step that failed and why, what the upgrade changed before it, the steps it did
    not do, and which retired stores moved aside (the others are where the sidecar opens them)."""
    reason = str(exc) if isinstance(exc, (InitError, ConfigError)) else \
        f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    lines = [f"colonymind upgrade failed while {step}: {reason}"]
    lines.append("  changed before the failure:" if notes else "  nothing was changed")
    lines.extend("    " + note.replace("\n", "\n    ") for note in notes)
    if step in plan:
        lines.append("  not done: " + ", ".join(plan[plan.index(step):]))
    moved = [name for name in retiring if home is not None and not (home / name).exists()]
    if moved:
        kept = [name for name in retiring if name not in moved]
        lines.append("  moved aside before the failure: " + ", ".join(moved)
                     + ("; left in place: " + ", ".join(kept) if kept else ""))
    else:
        lines.append("  no store was moved aside: every store is where the sidecar opens it")
    return lines


def _mind_off(cfg: Config) -> None:
    """Best effort off-switch cleanup through the sidecar; the route arrives with the mind."""
    key = read_api_key(cfg.home)
    if not key:
        return
    try:
        import httpx
        httpx.post(cfg.sidecar_url + "/v1/mind/off", headers={"Authorization": "Bearer " + key},
                   timeout=5, trust_env=False)
    except Exception:
        pass


def run_uninstall(args) -> int:
    """``colonymind init --uninstall``: leave stock Hermes behind and keep the instance data."""
    try:
        home = configuration.instance_home(getattr(args, "home", None))
        cfg = load_config(home, required=True)
        hermes_home = resolve_hermes_home(getattr(args, "hermes_home", None), cfg.get("hermes.home"))
        _mind_off(cfg)
        _say("  " + uninstall_service(cfg))
        config_path = hermes_home / "config.yaml"
        current = read_hermes_config(config_path)
        replaced = str(read_install_state(home).get("replaced_memory_provider") or "")
        updated, changes = strip_hermes_config(current, skills_dir=home / SKILLS_DIR, restore_provider=replaced)
        if changes:
            write_hermes_config(config_path, updated, backup_dir=home / BACKUPS_DIR)
        for change in changes:
            _say("  " + change)
        chosen = selected_memory_provider(current)
        if chosen and chosen != MEMORY_PROVIDER:
            _say(f"  memory.provider is {chosen}, chosen after install: left as it is"
                 + (f" (init replaced {replaced})" if replaced and replaced != chosen else ""))
        if (home / INSTALL_STATE).is_file():
            write_install_state(home, replaced_memory_provider=None)
        backup_dir = home / BACKUPS_DIR / _utc_stamp()
        if remove_worker_profile(profiles_root(hermes_home), backup_dir=backup_dir):
            _say(f"  profiles/{WORKER_PROFILE} moved to {backup_dir / 'profiles' / WORKER_PROFILE}")
        # Uninstall reads what Hermes recorded and goes on without it: it never has Hermes build an environment,
        # and a record it cannot read does not stop it after the steps above.
        try:
            candidates = hermes_python_candidates(getattr(args, "hermes_python", None),
                                                   cfg.get("hermes.python"), hermes_home)
        except InitError as exc:
            candidates = []
            _say(f"  {exc}")
        python_value = next((p for p in candidates if p.exists()), None)
        users = adapter_users(hermes_home)
        if python_value is None:
            _say(f"  no Hermes interpreter found for {hermes_home}: {ADAPTER_DISTRIBUTION} is left where it is")
        elif users:
            _say(f"  {ADAPTER_DISTRIBUTION} kept in {python_value}: still enabled by {', '.join(users)}")
        elif uninstall_adapter(Path(python_value)):
            _say(f"  {ADAPTER_DISTRIBUTION} removed from {python_value}")
    except (InitError, ConfigError) as exc:
        _say(f"colonymind init --uninstall failed: {exc}")
        return 1
    except (OSError, subprocess.SubprocessError) as exc:
        _say(f"colonymind init --uninstall failed: {type(exc).__name__}: {exc}")
        return 1
    if cfg.get("router.embed_provider") == EMBED_LOCAL:
        from colonymind.vector import local_embedder
        _say(f"  the local embedder's files stay in {local_embedder.models_dir(home)} with the instance data")
    _say(f"Hermes in {hermes_home} is stock again; the instance data in {home} is kept.")
    _say("Restart the gateway to drop the adapter: hermes gateway restart")
    return 0


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def add_parsers(sub) -> None:
    init_p = sub.add_parser("init", help="Configure this instance and attach it to stock Hermes")
    init_p.add_argument("--home", default=None, help="Instance directory (default: $COLONYMIND_HOME or ~/.colonymind)")
    init_p.add_argument("--non-interactive", "-n", action="store_true", help="Run without prompts")
    init_p.add_argument("--uninstall", action="store_true",
                        help="Remove the adapter, its Hermes keys and the worker profile; keep the instance data")
    init_p.add_argument("--owner-name", help="Your name")
    init_p.add_argument("--owner-handle", action="append", metavar="PLATFORM=ID",
                        help="One of your messaging handles; repeat per account")
    init_p.add_argument("--agent-name", help="The agent's name")
    init_p.add_argument("--agent-values", help="Comma-separated guiding values")
    init_p.add_argument("--agent-boundaries", help="Comma-separated boundaries the agent never crosses")
    init_p.add_argument("--timezone", help="Named time zone, for example Europe/Paris")
    init_p.add_argument("--quiet-hours", help="Local quiet window, HH:MM-HH:MM")
    init_p.add_argument("--autonomy", choices=AUTONOMY_LEVELS, help="Autonomy level (default: suggest)")
    init_p.add_argument("--owner-platform", action="append", metavar="PLATFORM",
                        help="A platform the adapter may write the owner's turns from; repeat per platform "
                             "(default: whatsapp, sms, rcs, voice; the CLI is 'cli')")
    init_p.add_argument("--hermes-home", help="Hermes home (default: $HERMES_HOME or ~/.hermes)")
    init_p.add_argument("--hermes-python", help="Interpreter that runs 'hermes' (default: read from the executable)")
    init_p.add_argument("--host", help="Sidecar bind address (default 127.0.0.1)")
    init_p.add_argument("--port", type=int, help="Sidecar port (default 7777)")
    init_p.add_argument("--model-url", help="OpenAI-compatible API root (default: the one Hermes uses)")
    init_p.add_argument("--model", help="Model identifier at that endpoint")
    init_p.add_argument("--model-key", help="API key for that endpoint")
    init_p.add_argument("--embed-url", help="OpenAI-compatible embeddings root, which turns semantic recall on; "
                                            "'auto' asks the model endpoint's embeddings and rerank routes (never "
                                            "a chat request) and records what answers, else offers the local "
                                            "embedder; 'local' runs a pinned embedder in the sidecar (needs "
                                            "colonymind[vectors]), its files in the instance")
    init_p.add_argument("--embed-model", help="Embedding model at that endpoint (with --embed-url local: a pinned "
                                              "local model other than the one chosen for this machine)")
    init_p.add_argument("--embed-dims", type=int,
                        help="Vector width of that model (default: learned from its first embedding)")
    init_p.add_argument("--rerank-model", help="With --embed-url auto: the rerank model to ask the endpoint for")
    init_p.add_argument("--adapter-source", help="Install the adapter from this wheel, directory or requirement")
    init_p.add_argument("--import-history", action="store_true",
                        help="Import your own direct sessions in Hermes (those from the handles you give) as "
                             "quotations, without asking")
    init_p.add_argument("--no-service", action="store_true", help="Do not install the sidecar user service")

    upgrade_p = sub.add_parser("upgrade", help="Backup, migrate, upgrade the adapter, reconcile config, restart")
    upgrade_p.add_argument("--home", default=None, help="Instance directory (default: $COLONYMIND_HOME or ~/.colonymind)")
    upgrade_p.add_argument("--hermes-home", help="Hermes home when it moved")
    upgrade_p.add_argument("--hermes-python", help="Interpreter that runs 'hermes' when it moved")
    upgrade_p.add_argument("--adapter-source", help="Install the adapter from this wheel, directory or requirement")
