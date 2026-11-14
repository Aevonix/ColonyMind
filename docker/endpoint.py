"""What the container's one model endpoint replaces, for the setup step (``docker/init.sh``).

Installed as ``/opt/colonymind/docker/endpoint.py`` and run by the setup step as the hermes user, with the sidecar's
interpreter:

``hermes-key HERMES_HOME URL [FIELD]``
    what becomes of a key Hermes holds in ``model.FIELD`` (``api_key``, the default, or the legacy ``api``, which
    Hermes reads when ``model.api_key`` is empty): ``moved`` when it was given for another endpoint than ``URL``
    (Hermes would send it there once its model moves), ``removed`` when it is the container's own reference to
    ``COLONYMIND_CONTAINER_MODEL_KEY`` (the key was taken away), else ``keep``. Either of the first two is removed
    from Hermes' config. The setup step asks it for ``api_key`` when the environment gives no key (one it gives
    replaces that field) and for ``api`` always (a key given only shadows it).

``init-key HERMES_HOME URL``
    ``keyless`` when the key ``colonymind init`` would select on its own for the endpoint ``URL`` is not the one given
    for it (``bound_key``): ``OPENAI_API_KEY`` in Hermes' ``.env``, or Hermes' key when ColonyMind is told another
    endpoint than Hermes'. The setup step then tells init there is none, so init neither writes that key into a
    router file it creates nor sends it to ``URL`` with its recall probe. ``same`` otherwise.

``router ROUTER_FILE HERMES_HOME URL MODEL``
    points the router's own file, when there is one, at the endpoint ``URL`` serving ``MODEL``: its base URL, its
    key (``bound_key``) and every tier's model; everything else the file says stays (a tier's completion and
    context limits, request extras, the model pool, the roles), and each binding of the model pool keeps its
    endpoint with the key it was sent (``pool_pointed_at``). It is rewritten only when that changes something,
    with the version it replaces copied into ``backups/``, and prints what it did. A missing file is init's to
    write.

When it cannot tell (Hermes' config or the router's file does not parse, for example) it exits non-zero with the
reason and changes nothing, and the setup step stops rather than guess.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from colonymind.init import _MODEL_ROLES, InitError, hermes_model_endpoint, read_hermes_config
from colonymind.setup import OPENAI_COMPAT_PROVIDERS, normalize_llm_base_url

#: A value of Hermes' config that names an environment variable (``${NAME}`` or ``${env:NAME}``): Hermes reads the
#: variable.
_REFERENCE = re.compile(r"^\$\{\s*(?:env:)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}$")
_DEFAULT_PORTS = {"http": 80, "https": 443}


def expand(value: str) -> str:
    """*value* as Hermes reads it: a variable's value for a reference, else *value* itself."""
    match = _REFERENCE.match(value.strip())
    return os.environ.get(match.group(1), "") if match else value


def origin(url: str) -> tuple[str, str, int | None] | None:
    """Where a credential sent to *url* goes: its scheme, host and port. None when *url* names no host."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    return scheme, parts.hostname.lower(), port or _DEFAULT_PORTS.get(scheme)


def hermes_model(hermes_home: Path) -> dict:
    """Hermes' ``model`` block ({} when it has none, or names only a model)."""
    model = read_hermes_config(hermes_home / "config.yaml").get("model")
    return model if isinstance(model, dict) else {}


#: The variable the container's key is in; Hermes' config names it, never the value.
CONTAINER_KEY = "COLONYMIND_CONTAINER_MODEL_KEY"
#: Where Hermes' model block holds its key, in the order Hermes reads them: ``api_key``, else the legacy ``api``
#: (Hermes clears both when it clears a credential).
KEY_FIELDS = ("api_key", "api")


def hermes_key(model: dict) -> str:
    """The key Hermes' model block holds, as Hermes picks it: the first of ``KEY_FIELDS`` that is set."""
    return next((value.strip() for value in (model.get(field) for field in KEY_FIELDS)
                 if isinstance(value, str) and value.strip()), "")


def hermes_key_verdict(hermes_home: Path, url: str, field: str = "api_key") -> str:
    """What becomes of the key Hermes holds in ``model.<field>`` (``KEY_FIELDS``) for the endpoint *url*.

    ``removed``: it is the container's own reference to ``COLONYMIND_CONTAINER_MODEL_KEY``, which is now empty or
    gone, so Hermes would send an empty key, or the reference itself. ``moved``: it was given for another endpoint.
    A key belongs to the endpoint Hermes' model had when it was given: its ``base_url``, compared by where a request
    goes (scheme, host and port; a path of its own on the same server is the same endpoint). A named provider
    without a base URL is its own endpoint, never *url*. ``keep`` otherwise: no key, or the owner's own for *url*."""
    if field not in KEY_FIELDS:
        raise ValueError(f"Hermes holds its model's key in {' or '.join(KEY_FIELDS)}, not {field}")
    model = hermes_model(hermes_home)
    key = str(model.get(field) or "").strip()
    if not key:
        return "keep"
    reference = _REFERENCE.match(key)
    if reference and reference.group(1) == CONTAINER_KEY:
        return "removed"
    given_for = origin(expand(str(model.get("base_url") or "")))
    return "moved" if given_for is None or given_for != origin(url) else "keep"


#: What init records for an endpoint without a key (``colonymind.setup.ensure_api_key``).
KEYLESS = "local-no-key"
#: A tier's own endpoint and key. With the container's one endpoint every tier is served there, with its key.
TIER_ENDPOINT = ("baseUrl", "base_url", "apiKey", "api_key")


class RouterFileProblem(Exception):
    """The router's file cannot be read as a router config: nothing can be compared, so nothing is changed."""


def bound_key(hermes_home: Path, url: str) -> str:
    """The key given for the endpoint *url*, "" for none: the router's own override (``COLONYMIND_MODEL_API_KEY``)
    first, else Hermes' key as Hermes reads it (``hermes_key``: ``model.api_key``, else the legacy ``model.api``),
    while Hermes' model is on *url*'s server (scheme, host and port). By then the setup step has removed a key Hermes held for another endpoint, so that is the
    container's own (``COLONYMIND_CONTAINER_MODEL_KEY``) or one the owner gave Hermes for this server. A key in
    Hermes' ``.env`` is never taken: in the container the endpoint comes from the variables, and ``.env`` names no
    key for it (Hermes itself sends ``OPENAI_API_KEY`` only to OpenAI's own hosts)."""
    override = os.environ.get("COLONYMIND_MODEL_API_KEY")
    if override:
        return override
    model = hermes_model(hermes_home)
    key = hermes_key(model)
    if not key:
        return ""
    given_for = origin(expand(str(model.get("base_url") or "")))
    return expand(key) if given_for is not None and given_for == origin(url) else ""


def init_key_verdict(hermes_home: Path, url: str) -> str:
    """``keyless`` when the key ``colonymind init`` would select for *url* on its own (its override, else the one
    Hermes' config resolves to, with init's own function) is not the one given for it (``bound_key``): init is then
    told there is none. ``same`` otherwise: it selects that key, or none."""
    selected = (os.environ.get("COLONYMIND_MODEL_API_KEY")
                or hermes_model_endpoint(hermes_home, read_hermes_config(hermes_home / "config.yaml"))["api_key"])
    return "same" if selected in ("", bound_key(hermes_home, url)) else "keyless"


def model_id(entry: object) -> str | None:
    """The model a tier's entry names, as the router reads it: the entry itself, or an object's ``model`` (else its
    ``model_id``). None for an entry the router cannot read."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        found = entry["model"] if "model" in entry else entry.get("model_id")
        return found if isinstance(found, str) else None
    return None


def same_model(entry: object, model: str) -> bool:
    """True when *entry* names *model*. The router calls an OpenAI-compatible endpoint with ``openai/`` in front of
    a bare name, so ``openai/NAME`` and ``NAME`` are the same model there."""
    found = model_id(entry)
    return found is not None and found.removeprefix("openai/") == model.removeprefix("openai/")


def pool_pointed_at(pool: dict, before: tuple[object, object], url: str, key: str) -> None:
    """The model pool's bindings, changed in place so that none pairs a server with a key given for another, once
    the file's endpoint and key move from *before* (its base URL and key as they were) to *url* and *key*. A binding
    names an endpoint and a key of its own, or takes the file's (the routing snapshot's rule). One on a server of its
    own, other than *url*'s, keeps that server with the key it was sent there: the file's former key becomes its
    own when it had none. One the file's endpoint serves (it names none, or one on *url*'s server) follows it with
    *key*, or with a key of its own given for that same server; its own key for the former endpoint is dropped."""
    destination = origin(url)
    for binding in pool.values():
        if not isinstance(binding, dict):
            continue
        served = binding["baseUrl"] if "baseUrl" in binding else before[0]
        if "baseUrl" in binding and origin(str(binding["baseUrl"])) != destination:
            if "apiKey" not in binding and before[1] != key:
                binding["apiKey"] = before[1]
        elif "apiKey" in binding and origin(str(served)) != destination:
            del binding["apiKey"]


def pointed_at(current: dict, url: str, model: str, key: str) -> dict:
    """*current* with the container's endpoint: an OpenAI-compatible provider, *url* (normalized as init writes it),
    *key* (or the keyless placeholder) and *model* in every tier init writes. A tier that is an object keeps its
    other settings and loses only its own endpoint and key; the model pool's bindings keep each endpoint with its
    key (``pool_pointed_at``); everything else stays as it is."""
    document = copy.deepcopy(current)
    # What a binding without an endpoint or key of its own took until now (``build_snapshot``'s defaults).
    before = (current.get("baseUrl", ""), current.get("apiKey", ""))
    provider = str(document.get("provider") or "").strip().lower()
    if provider not in OPENAI_COMPAT_PROVIDERS:
        document["provider"] = provider = "local"
    wanted, _ = normalize_llm_base_url(url, provider)
    base = str(document.get("baseUrl") or "")
    if normalize_llm_base_url(base, provider)[0].rstrip("/") != wanted.rstrip("/"):
        document["baseUrl"] = wanted
    document["apiKey"] = key or KEYLESS
    if isinstance(document.get("modelPool"), dict):
        pool_pointed_at(document["modelPool"], before, url, document["apiKey"])
    models = document.get("models")
    if models is None:
        models = document["models"] = {}
    for role in _MODEL_ROLES:
        entry = models.get(role)
        if isinstance(entry, dict):
            entry = {name: value for name, value in entry.items() if name not in TIER_ENDPOINT}
            if not same_model(entry, model):
                entry["model_id" if "model" not in entry and "model_id" in entry else "model"] = model
            models[role] = entry
        elif not same_model(entry, model):
            models[role] = model
    return document


def _write_private(path: Path, text: str) -> None:
    """*text* into *path*, readable by its owner only, replacing it in one step."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _keep(path: Path, text: str) -> Path:
    """A copy of the router's file as it was, in ``backups/`` beside it, never over an earlier one."""
    backups = path.parent / "backups"
    backups.mkdir(mode=0o700, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for attempt in range(1000):
        kept = backups / (f"router-{stamp}.json" if not attempt else f"router-{stamp}-{attempt}.json")
        try:
            descriptor = os.open(kept, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        return kept
    raise FileExistsError(f"no free name for a copy of {path.name} in {backups}")


def reconcile_router(path: Path, hermes_home: Path, url: str, model: str) -> str:
    """Point the router's file at the container's endpoint (``pointed_at``). Returns what changed, "" for nothing.
    Raises ``RouterFileProblem`` for a file it cannot read as a router config, before changing anything."""
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RouterFileProblem(f"it cannot be read ({exc.strerror or exc})") from None
    try:
        current = json.loads(text)
    except ValueError as exc:
        raise RouterFileProblem(f"it is not valid JSON ({exc})") from None
    if not isinstance(current, dict):
        raise RouterFileProblem("it is not a JSON object")
    if current.get("models") is not None and not isinstance(current["models"], dict):
        raise RouterFileProblem("its models is not an object")
    updated = pointed_at(current, url, model, bound_key(hermes_home, url))
    if updated == current:
        return ""
    kept = _keep(path, text)
    _write_private(path, json.dumps(updated, indent=2) + "\n")
    return f"the router's file now names {model} at {updated['baseUrl']}; the version it replaces is {kept}"


def main(argv: list[str]) -> int:
    command, *args = argv or [""]
    try:
        if command == "hermes-key" and len(args) in (2, 3) and (len(args) == 2 or args[2] in KEY_FIELDS):
            print(hermes_key_verdict(Path(args[0]), args[1], *args[2:]))
            return 0
        if command == "init-key" and len(args) == 2:
            print(init_key_verdict(Path(args[0]), args[1]))
            return 0
        if command == "router" and len(args) == 4:
            note = reconcile_router(Path(args[0]), Path(args[1]), args[2], args[3])
            if note:
                print(note)
            return 0
    except InitError as exc:
        print(f"[colonymind] {exc}", file=sys.stderr)
        return 1
    except RouterFileProblem as exc:
        print(f"[colonymind] the router's file {args[0]} is left as it is: {exc}. Fix it, or move it away so init "
              "writes a new one, then restart", file=sys.stderr)
        return 1
    print(f"usage: {Path(__file__).name} hermes-key HERMES_HOME URL [api_key|api] | init-key HERMES_HOME URL"
          " | router ROUTER_FILE HERMES_HOME URL MODEL", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
