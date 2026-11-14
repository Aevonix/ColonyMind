"""Secrets in the Hermes body: sealing at the gateway, fills and masks on tool calls, and the owner's tool.

The sidecar keeps secrets that arrive in conversation sealed (docs/SECRETS.md). This module is the Hermes half:

* ``pre_gateway_dispatch``: a gateway message with a detected value is sealed through ``POST /v1/secrets/seal``
  and rewritten before Hermes sees it, so Hermes' own transcript, the model provider and session search never
  receive the value (``secrets.capture.ingress``). The words that reach the model with it without being the
  sender's (the message it quotes, a channel's backfill) are sealed in place, in their author's scope. When the
  sidecar cannot seal, each value is withheld; a value it returns as it came passes only where it says the owner
  released it. A message with no detected value never reaches the sidecar here.
* ``tool_request`` middleware, in the owner's own session: a value is used when the reference was sealed from the
  owner's own words in their latest message in that same chat (their words stand behind the call; a quoted message, a
  backfill or an expanded file never are); any other becomes a one-tap ask bound to the digest of the complete
  call and put to the owner directly with its destination in full, and the same call after their yes is served
  once. A value goes as text only into a send's or a write's words (text a person reads or a file holds) and into
  a waiting process's input as the whole of it. A value is never put into a command's or code's text: a terminal
  command or Python code names ``$COLONYMIND_SECRET_<id>`` (the value) or ``$COLONYMIND_SECRET_<id>_FILE`` (a
  private file holding it), and the plugin sets both in that call's environment and nowhere else (``bind_shell``,
  ``bind_python``: the command or code is never read or changed); a
  reference written into a command or code is refused, naming the variable to use. ``write_file`` and ``patch``
  are served with no ask there. In a mind run a message to anyone but the owner has every detected value and
  every reference hidden, as the outbox hides them in what the mind composes.
* ``pre_tool_call``: a call in the owner's session that still carries a reference, or names a variable the
  plugin did not set for it (its ask is open, it was refused, it cannot be served there), is held with the
  reason; a reference never leaves as text in a send. ``post_tool_call`` removes the private files of a call that
  is over, unless the program that took one still runs (a background command removes its own when it ends).
* ``colonymind_secret``: the owner's tool: list, use (to Hermes' credentials, a file, an autofill login), reveal
  (sent to the owner's own chat by the plugin, or put in a private file where there is no chat, never through the
  model), release, forget. Refused for guests, cron runs and workers.
* ``words_to_send``: the body's outbox words with the references the sidecar allows resolved just before the
  send; one it cannot settle for now holds the message for a later tick.

Every value used is registered with Hermes' redactor for model-facing output, and the vendor token patterns are
registered with it too (``register_redaction_patterns``). The detector is shared with the sidecar and packaged
with the standalone adapter.
"""

from __future__ import annotations

import re

from colonymind_shared.secret_detection import (
    HIDDEN, Hit, REFERENCE, RULES, VENDOR_PATTERNS, WITHHELD, clean_label, find_secrets,
    hide_secrets, replace_secrets,
)



import json
import logging
import os
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .capture import INTERNAL_PLATFORMS, SessionMap, session_env
from .client import ColonyMindClient, Settings, SidecarUnavailable, final_answer, read_yaml
from .guard import MESSAGING_TOOLS, mind_run, parse_deliver
from .busy_ingress import seen

logger = logging.getLogger(__name__)

SEAL_ROUTE = "/v1/secrets/seal"
RESOLVE_ROUTE = "/v1/secrets/resolve"
LIST_ROUTE = "/v1/secrets"
SEAL_TIMEOUT = 3.0
# The most references one resolve request carries (the route's own limit); more are asked in turns.
RESOLVE_BATCH = 32
AUTOMATED_PLATFORMS = frozenset({"cron"})
# Calls that run a program they carry: a value reaches one only through that call's environment (``bind_*``).
PROGRAM_FIELDS = {"terminal": "command", "execute_code": "code"}
# A process waiting for input: a value reaches it only as the whole of that input.
INPUT_FIELDS = {"process": "data", "process_manage": "data"}
EXECUTOR_TOOLS = frozenset(PROGRAM_FIELDS) | frozenset(INPUT_FIELDS)
# Calls that send or run what they carry: a reference in one is served only on the owner's words.
FILL_TOOLS = frozenset(MESSAGING_TOOLS | EXECUTOR_TOOLS)
# Local writes in the owner's own session: served without an ask.
LOCAL_TOOLS = frozenset({"write_file", "patch"})
# The arguments of a send or a write that are its words: the only arguments a value goes into as text.
WORD_FIELDS = frozenset({"message", "text", "content", "body", "caption", "comment", "new_string", "patch"})
# What a command or code names to use a sealed value: the variable holds the value, ``_FILE`` a private file with it.
VARIABLE_PREFIX = "COLONYMIND_SECRET_"
VARIABLE = re.compile(VARIABLE_PREFIX + r"([A-Z2-7]{8})(?![A-Z2-7])")
# How long the references sealed from the owner's latest message stand behind a use without an ask, how long a
# value's file waits for the program that is to take it, and how long a revealed value's file is kept.
FRESH_SECONDS = 1800
UNCLAIMED_SECONDS = 3600
REVEAL_SECONDS = 1800
# What Hermes' redactor takes from the plugin (each must start with two literal characters).
REDACTION_PATTERNS = VENDOR_PATTERNS + (r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{16,}",
                                        r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]{20,}",
                                        r"https://discord\.com/api/webhooks/[A-Za-z0-9/_-]{20,}")
SECRET_OPERATIONS = ("list", "use", "reveal", "release", "forget")
USE_TARGETS = ("to_env", "to_file", "to_login")
SECRET_SCHEMA = {
    "name": "colonymind_secret",
    "description": "Owner only. Values the owner gave you are sealed; [secret S-XXXXXXXX: label] stands for each. "
                   "list; use <ref> to_env <NAME> | to_file <path> | to_login <origin> (+identifier); reveal <ref> "
                   "(sent to the owner, not to you); release <ref>; forget <ref>. In a message's words put the "
                   "reference where the value goes. A terminal command or Python code never contains a reference: "
                   "it reads $COLONYMIND_SECRET_XXXXXXXX (os.environ in Python; $COLONYMIND_SECRET_XXXXXXXX_FILE is a "
                   "private file holding it), set for that one call; keep it a variable (never splice it into code "
                   "text). A value is used if the owner's latest message gave it, else the owner is asked about "
                   "that exact call.",
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": list(SECRET_OPERATIONS)}, "ref": {"type": "string"},
        "target": {"type": "string", "enum": list(USE_TARGETS)}, "name": {"type": "string"},
        "path": {"type": "string"}, "origin": {"type": "string"}, "identifier": {"type": "string"},
        "identifier_type": {"type": "string", "enum": ["email", "username", "phone"]}},
        "required": ["operation"]}}
REVEAL_WARNING = ("the value now sits in this chat's history; delete the message when you have it, and prefer use "
                  "over reveal next time")


class Unplaceable(ValueError):
    """A reference sits where its value cannot go (outside a call's words, mixed into a process's input, or in a
    command's or code's text): the call waits and says how to use the value instead."""


# -- configuration ---------------------------------------------------------------------------------------------

def _flag(value: Any, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"secrets.{name} must be true or false")


def _capture(section: Any) -> dict[str, Any]:
    """The capture part of a ``secrets:`` section, validated exactly as ``colonymind.secrets.config`` validates the
    whole section (a test keeps the two in step); raises ``ValueError`` on a malformed one."""
    if section is None:
        return {"enabled": True, "ingress": True, "rules": RULES}
    if not isinstance(section, Mapping):
        raise ValueError("secrets must be a mapping")
    parts = {name: section.get(name) or {} for name in ("capture", "vault", "migrate")}
    for name, part in parts.items():
        if not isinstance(part, Mapping):
            raise ValueError(f"secrets.{name} must be a mapping")
    capture, vault = parts["capture"], parts["vault"]
    if str(vault.get("key_store") or "file").strip().lower() not in {"file", "keyring"}:
        raise ValueError("secrets.vault.key_store must be one of file, keyring")
    _flag(vault.get("key_in_backups"), False, "vault.key_in_backups")
    _flag(parts["migrate"].get("on_start"), True, "migrate.on_start")
    rules = capture.get("rules")
    if rules is None:
        enabled = RULES
    elif isinstance(rules, (Mapping, list, tuple)):
        named = [str(name).strip() for name in rules]
        unknown = sorted(set(named) - set(RULES))
        if unknown:
            raise ValueError(f"secrets.capture.rules names unknown rules: {', '.join(unknown)}")
        if isinstance(rules, Mapping):
            off = {str(name) for name, on in rules.items() if not _flag(on, True, f"capture.rules.{name}")}
            enabled = tuple(rule for rule in RULES if rule not in off)
        else:
            enabled = tuple(rule for rule in RULES if rule in named)
    else:
        raise ValueError("secrets.capture.rules must be a list of rule names or a mapping of rule: on/off")
    return {"enabled": _flag(capture.get("enabled"), True, "capture.enabled"),
            "ingress": _flag(capture.get("ingress"), True, "capture.ingress"), "rules": enabled}


def capture_settings(settings: Settings) -> dict[str, Any]:
    """``secrets.capture`` from ``colonymind.yaml``: enabled and ingress (both on unless turned off), the rules. A
    malformed section never weakens capture here, as in the sidecar: the defaults stand (every rule, ingress on)
    and the problem is logged."""
    try:
        return _capture(read_yaml(settings.home / "colonymind.yaml").get("secrets"))
    except ValueError as error:
        logger.warning("secrets settings malformed (%s); capture keeps its defaults", error)
        return {"enabled": True, "ingress": True, "rules": RULES}


# -- references in tool calls ----------------------------------------------------------------------------------

def _leaves(value: Any, change) -> Any:
    if isinstance(value, str):
        return change(value)
    if isinstance(value, list):
        return [_leaves(item, change) for item in value]
    if isinstance(value, dict):
        return {key: _leaves(item, change) for key, item in value.items()}
    return value


def _strings(value: Any) -> list[str]:
    found: list[str] = []
    _leaves(value, lambda text: found.append(text) or text)
    return found


def _ref_of(match: Any) -> str:
    return match.group(0)[len("[secret "):].split(":", 1)[0]


def reference_ids(value: Any) -> list[str]:
    """The references a tool call's arguments carry, in order."""
    ids: list[str] = []
    for text in _strings(value):
        for match in REFERENCE.finditer(text):
            if _ref_of(match) not in ids:
                ids.append(_ref_of(match))
    return ids


def variable(ref: str) -> str:
    """The environment variable a command or code reads ``ref``'s value from (``S-ABCDEFGH``:
    ``COLONYMIND_SECRET_ABCDEFGH``; with ``_FILE``, the path of a private file holding it)."""
    return VARIABLE_PREFIX + ref[len("S-"):]


def named_refs(tool: str, args: Mapping[str, Any]) -> list[str]:
    """The references whose variables a command or code names, in order (none for any other call)."""
    field = PROGRAM_FIELDS.get(tool)
    text = args.get(field) if field else None
    if not isinstance(text, str):
        return []
    return list(dict.fromkeys("S-" + match.group(1) for match in VARIABLE.finditer(text)))


def program_refusal(refs: list[str]) -> str:
    """What a command or code with a reference written into it is told instead."""
    names = ", ".join("$" + variable(ref) for ref in refs)
    return (f"a private value never goes into a command or code as text: write {names} where it goes "
            f"(os.environ['{variable(refs[0])}'] in Python; ${variable(refs[0])}_FILE names a private file holding "
            f"it), set for this one call")


def fill(value: Any, values: Mapping[str, str]) -> Any:
    """``value`` with each reference whose id is in ``values`` replaced by the value, as text: only for words
    (a message, a file's content), never for anything a program runs."""
    return _leaves(value, lambda text: REFERENCE.sub(lambda match: values.get(_ref_of(match), match.group(0)), text))


def place(tool: str, args: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
    """``args`` with every reference's value placed where the call's words take it: as text in a send's or a
    write's words, as the whole input of a process waiting for it. Raises ``Unplaceable`` for a reference anywhere
    else, and for any reference in a command or code (those read a variable: ``bind_shell``, ``bind_python``)."""
    if tool in PROGRAM_FIELDS:
        raise Unplaceable(program_refusal(reference_ids(args) or ["S-XXXXXXXX"]))
    field = INPUT_FIELDS.get(tool)
    allowed = {field} if field else WORD_FIELDS
    for key, value in args.items():
        if reference_ids(value) and key not in allowed:
            raise Unplaceable(f"{key} is not where a private value goes in {tool}")
    if field:
        match = REFERENCE.fullmatch(str(args.get(field) or ""))
        if args.get("action") not in {"write", "submit"} or match is None:
            raise Unplaceable("a process takes a private value only as its whole input: data is the reference alone")
        return {**args, field: values[_ref_of(match)]}
    return {key: fill(value, values) if key in WORD_FIELDS else value for key, value in args.items()}


def bind_shell(command: str, files: Mapping[str, str]) -> str:
    """``command`` run in a subshell whose environment alone holds the values: lines set before it read each
    ``$COLONYMIND_SECRET_<id>`` from its private file as data (its exact bytes, a last newline included) and let
    ``..._FILE`` name that file. The command follows as it was written: never read or changed here, so no value
    is ever part of what the shell parses, and Hermes still reads the command's own words (a ``sudo`` it serves).
    Nothing it sets outlives the subshell: the session's own shell, and the environment Hermes keeps between
    commands, never hold a value (a command that closes the subshell early runs the rest without it). The lines
    use the shell's own builtins and ``command`` (no alias or function of the session can stand in for them),
    record who took the files and remove them when the subshell ends; a value that cannot be read stops the call
    before it runs."""
    paths = [shlex.quote(path) for path in files.values()]
    pids = [shlex.quote(path + ".pid") for path in files.values()]
    lines = ["(",
             "__colonymind_done() { command rm -f -- " + " ".join(f"{path} {pid}" for path, pid in zip(paths, pids))
             + "; }",
             "builtin trap __colonymind_done EXIT"]
    for ref, path, pid in zip(files, paths, pids):
        name = variable(ref)
        lines += [f"builtin printf '%s' \"$$\" > {pid} || builtin exit 125",
                  f"{name}=$(command cat -- {path} && builtin printf x) || builtin exit 125",
                  f"{name}=${{{name}%x}}",
                  f"{name}_FILE={path}",
                  f"builtin export {name} {name}_FILE"]
    return "\n".join(lines + [command, ")"])


def bind_python(code: str, files: Mapping[str, str]) -> str:
    """``code`` run with the values in the interpreter's environment for that run only: each
    ``COLONYMIND_SECRET_<id>`` read from its private file (its exact bytes) and ``..._FILE`` naming it, removed
    again when the code ends, however it ends (the session's interpreter keeps nothing of them). The code is
    handed to ``exec`` whole, as one string literal: never read or changed here."""
    bound = {variable(ref): path for ref, path in files.items()}
    return "\n".join([
        f"__colonymind_bound = {bound!r}",
        "try:",
        "    for __colonymind_name, __colonymind_path in __colonymind_bound.items():",
        "        with open(__colonymind_path + '.pid', 'w') as __colonymind_file:",
        "            __colonymind_file.write(str(__import__('os').getpid()))",
        "        with open(__colonymind_path, encoding='utf-8', newline='') as __colonymind_file:",
        "            __import__('os').environ[__colonymind_name] = __colonymind_file.read()",
        "        __import__('os').environ[__colonymind_name + '_FILE'] = __colonymind_path",
        f"    exec(compile({code!r}, '<cell>', 'exec'), globals())",
        "finally:",
        "    for __colonymind_name, __colonymind_path in __colonymind_bound.items():",
        "        __import__('os').environ.pop(__colonymind_name, None)",
        "        __import__('os').environ.pop(__colonymind_name + '_FILE', None)",
        "        for __colonymind_end in ('', '.pid'):",
        "            try:",
        "                __import__('os').unlink(__colonymind_path + __colonymind_end)",
        "            except OSError:",
        "                pass",
        "    for __colonymind_left in ('__colonymind_bound', '__colonymind_name', '__colonymind_path', '__colonymind_end',",
        "                             '__colonymind_file'):",
        "        globals().pop(__colonymind_left, None)",
        "    globals().pop('__colonymind_left', None)",
    ])


def destination(tool: str, args: Mapping[str, Any]) -> str:
    """Where a call takes a value, in full, as the owner is asked about it (a command or code as it was written)."""
    if tool in PROGRAM_FIELDS:
        return {"terminal": "run the command: ", "execute_code": "run the code: "}[tool] + str(
            args.get(PROGRAM_FIELDS[tool]) or "")
    if tool in INPUT_FIELDS:
        return f"the input of process {args.get('session_id') or ''}: " + str(args.get(INPUT_FIELDS[tool]) or "")
    target = args.get("target") or args.get("chat_id") or args.get("to") or args.get("channel_id") or args.get("path")
    return str(target or tool)


def _local_backend() -> bool:
    """Whether commands and code run on this host (a value's private file is readable there only)."""
    try:
        from tools.terminal_scope import terminal_env
        backend = terminal_env("TERMINAL_ENV", "local")
    except ImportError:
        backend = os.environ.get("TERMINAL_ENV", "local")
    except Exception:
        return False
    return str(backend or "local").strip().lower() == "local"


class Bindings:
    """Private files (0600, in a 0700 directory) that hold one value each for the program of one call, which
    reads it when it runs. The program records its process id next to the file when it takes it and removes both
    when it ends. The plugin removes a file when its call is over and no program holds it, when the program that
    took it is gone (killed before it could), or when no program took it within ``UNCLAIMED_SECONDS``; never
    while the program that took it runs, however long that is."""

    def __init__(self, root: Path):
        self.root = root
        self._calls: dict[str, list[Path]] = {}

    def make(self, call_id: str, value: str) -> str:
        self.sweep()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        path = self.root / f"{uuid.uuid4().hex}.value"
        with os.fdopen(os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8",
                       newline="") as out:
            out.write(value)
        if call_id:
            self._calls.setdefault(call_id, []).append(path)
        return str(path)

    @staticmethod
    def held(path: Path) -> bool | None:
        """True while the program that took ``path`` runs, False once it is gone, None when none took it."""
        try:
            pid = int(Path(str(path) + ".pid").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _remove(path: Path) -> None:
        for name in (str(path), str(path) + ".pid"):
            try:
                os.unlink(name)
            except OSError:
                pass

    def done(self, call_id: str) -> None:
        """A call is over: its files go unless the program that took one still runs (a command Hermes moved to
        the background removes its own when it ends)."""
        for path in self._calls.pop(call_id, []):
            if self.held(path) is not True:
                self._remove(path)

    def sweep(self) -> None:
        if not self.root.is_dir():
            return
        cutoff = time.time() - UNCLAIMED_SECONDS
        for path in self.root.glob("*.value"):
            try:
                held = self.held(path)
                if held is False or (held is None and path.stat().st_mtime < cutoff):
                    self._remove(path)
            except OSError:
                pass
        for pid in self.root.glob("*.value.pid"):
            if not os.path.exists(str(pid)[:-len(".pid")]):
                self._remove(Path(str(pid)[:-len(".pid")]))


def register_for_redaction(value: str) -> None:
    """Hermes scrubs this exact value from later model-facing output (its vault redaction registry)."""
    try:
        from agent.redact import register_vault_redaction_value
    except ImportError:
        return
    try:
        register_vault_redaction_value(value)
    except Exception:
        logger.debug("redaction registration failed", exc_info=True)


def _batched(refs: list[str], ask) -> dict[str, Any] | None:
    """``ask(chunk)`` for each chunk of at most ``RESOLVE_BATCH`` references, the answers merged; None when one
    chunk had no readable answer."""
    merged: dict[str, Any] = {"values": {}, "refused": {}, "asks": {}, "unavailable": {}}
    for start in range(0, len(refs), RESOLVE_BATCH):
        answer = ask(refs[start:start + RESOLVE_BATCH])
        if not isinstance(answer, dict):
            return None
        for key, found in merged.items():
            found.update(answer.get(key) or {})
    return merged


def words_to_send(client: ColonyMindClient, message_id: str, text: str, *, to_owner: bool) -> str | None:
    """An outbox message's words as they leave the body: each sealed reference resolved just before the send when
    the sidecar allows it for this message (the owner's own reference, to the owner or in their dictated words;
    ``/v1/secrets/resolve`` purpose ``send``) and never stored here. One it refuses stays a reference for the owner
    and is hidden for anyone else. None when any reference is not settled now (the sidecar unreachable or without
    an answer for it, the vault's key unavailable): nothing leaves, and the body tries again later."""
    refs = reference_ids(text)
    if not refs:
        return text

    def ask(chunk: list[str]) -> dict[str, Any] | None:
        try:
            response = client.post(RESOLVE_ROUTE, timeout=SEAL_TIMEOUT,
                                   json={"refs": chunk, "purpose": "send", "intention_id": message_id})
            return response.json() if response.is_success else None
        except (SidecarUnavailable, ValueError, AttributeError):
            return None
    answer = _batched(refs, ask)
    if answer is None or answer["unavailable"] or set(refs) - set(answer["values"]) - set(answer["refused"]):
        return None
    values = {str(key): str(value) for key, value in answer["values"].items()}
    for value in values.values():
        register_for_redaction(value)
    text = fill(text, values)
    return text if to_owner else REFERENCE.sub(HIDDEN, text)


def hide_stored_send_errors(connection: Any) -> None:
    """Once per process, on the body's send ledger: an error that echoed a private value keeps the error, not
    the value."""
    connection.execute("PRAGMA secure_delete=ON")
    for row in connection.execute("SELECT id, error FROM mind_sends WHERE error != ''").fetchall():
        hidden = hide_secrets(row["error"])
        if hidden != row["error"]:
            connection.execute("UPDATE mind_sends SET error = ? WHERE id = ?", (hidden, row["id"]))


class Secrets:
    """The Hermes half of the sealed store."""

    def __init__(self, client: ColonyMindClient, sessions: SessionMap, settings: Settings):
        self.client, self.sessions, self.settings = client, sessions, settings
        self._asks: dict[tuple[str, str], str] = {}   # (session, ref) -> why a call with it waits
        self._told: set[str] = set()                   # the ask codes already put to the owner directly
        # (platform, chat, sender) -> the references sealed from that sender's own words in their latest message
        # in that chat
        self._fresh: dict[tuple[str, str, str], tuple[frozenset[str], float]] = {}
        # (session, program text) -> when: the commands and code this plugin set values for (the blocking hook
        # lets exactly those through)
        self._bound: dict[tuple[str, str], float] = {}
        # (session, reference) the sidecar refused to resolve (another person's value, a deleted one): no value,
        # so a local write keeps it as the reference it is
        self._refused: set[tuple[str, str]] = set()
        self.bindings = Bindings(Path(settings.home) / "private" / "secret-bindings")
        self.bindings.sweep()
        self.revealed = Path(settings.home) / "private" / "revealed"
        self._sweep_revealed()

    # -- who ----------------------------------------------------------------------
    def owner_session(self, session_id: str) -> bool:
        """The owner's own interactive session: never a kanban worker, never a cron run."""
        if os.environ.get("HERMES_KANBAN_TASK"):
            return False
        info = self.sessions.get(session_id)
        if info is not None and info.platform.strip().lower() in AUTOMATED_PLATFORMS:
            return False
        return self.sessions.is_owner(session_id) is True

    def _scope(self, platform: str, user_id: str) -> dict[str, Any]:
        """Whose value it is: the owner (the sidecar names its own owner's scope), else the sender's contact, else
        the sender's handle (never created here)."""
        if user_id and self.sessions.sender_is_owner(platform, user_id):
            return {"owner": True, "contact_id": self.settings.owner_contact_id()}
        try:
            found = self.client.resolve_contact(platform, user_id, timeout=SEAL_TIMEOUT) if user_id else None
        except Exception:
            found = None
        if found and found.get("contact_id"):
            return {"contact_id": str(found["contact_id"])}
        return {"contact_id": f"handle:{platform.lower()}:{user_id}"}

    def _own_refs(self, session_id: str) -> set[str]:
        """The references sealed from the owner's own words in their latest message in this session's chat (what
        their words stand behind); never a quoted message's, a channel's backfill or an expanded file's, and never
        their words in another chat."""
        info = self.sessions.get(session_id)
        if info is None:
            return set()
        chat, _ = self.sessions.chat(session_id)
        record = self._fresh.get((info.platform.strip().lower(), str(chat or ""), info.sender_id))
        if record is None or time.time() - record[1] > FRESH_SECONDS:
            return set()
        return set(record[0])

    # -- ingress ----------------------------------------------------------------------

    def _seal(self, text: str, scope: Mapping[str, Any], rules: tuple) -> str:
        """``text`` sealed by the sidecar in ``scope``. A value the sidecar returns as it came is let through only
        where it says the owner released it; any other detected value, and every value when the sidecar cannot
        seal, is withheld."""
        sealed, released = None, set()
        try:
            response = self.client.post(SEAL_ROUTE, timeout=SEAL_TIMEOUT, json={"text": text, **scope})
            if response.is_success:
                answer = response.json()
                sealed = answer.get("text")
                released = {tuple(span) for span in answer.get("released") or [] if isinstance(span, list)}
        except (SidecarUnavailable, ValueError, AttributeError) as error:
            logger.warning("a secret arrived and the sidecar could not seal it (%s); withheld", type(error).__name__)
        if not isinstance(sealed, str):
            return replace_secrets(text, lambda hit: WITHHELD, rules=rules)[0]
        return replace_secrets(sealed, lambda hit: hit.value if (hit.start, hit.end) in released else WITHHELD,
                               rules=rules)[0]

    def pre_gateway_dispatch(self, event: Any = None, **_: Any) -> dict[str, str] | None:
        if seen(event):   # sealed on the busy path already (busy_ingress): its text is the sealed text
            return None
        source = getattr(event, "source", None)
        platform = getattr(source, "platform", "")
        platform = str(getattr(platform, "value", platform) or "")
        user_id = str(getattr(source, "user_id", "") or "")
        fresh_key = (platform.strip().lower(), str(getattr(source, "chat_id", "") or ""), user_id)
        # A new message from this sender in this chat: the words of the one before no longer stand behind a use.
        self._fresh.pop(fresh_key, None)
        config = capture_settings(self.settings)
        if not (config["enabled"] and config["ingress"]):
            return None
        rules = config["rules"]
        # Words the sender did not write now reach the model with this message: the message they quote (in its
        # author's scope) and a channel's backfill (nobody's). Each is sealed in place and never counts as theirs.
        author = str(getattr(event, "reply_to_author_id", "") or "")
        for name, scope in (("reply_to_text", lambda: self._scope(platform, author) if author else
                             {"contact_id": f"handle:{platform.lower()}:quoted"}),
                            ("channel_context", lambda: {"contact_id": f"handle:{platform.lower()}:channel:"
                                                         f"{getattr(source, 'chat_id', '') or ''}"})):
            value = getattr(event, name, None)
            if isinstance(value, str) and value and find_secrets(value, rules=rules):
                setattr(event, name, self._seal(value, scope(), rules))   # Hermes copies the event as it is now
        text = getattr(event, "text", None)
        if not isinstance(text, str) or not text or not find_secrets(text, rules=rules):
            return None
        scope = self._scope(platform, user_id)
        from colonymind_shared.midturn import seal_rewrite
        sealed = seal_rewrite(event, lambda original: self._seal(original, scope, rules))
        if scope.get("owner"):
            own = frozenset(reference_ids(sealed)) - frozenset(reference_ids(text))
            self._fresh[fresh_key] = (own, time.time())
        return {"action": "rewrite", "text": sealed}

    # -- tool calls ------------------------------------------------------------------------
    def tool_request(self, tool_name: str = "", args: Any = None, session_id: str = "", tool_call_id: str = "",
                     **_: Any) -> dict | None:
        """``tool_request`` middleware: values used in the owner's session, masks in a mind run."""
        try:
            return self._tool_request(str(tool_name or ""), args if isinstance(args, dict) else {},
                                      str(session_id or ""), str(tool_call_id or ""))
        except Exception as error:   # middleware fails open in Hermes: the guard hook below holds the call
            logger.warning("secret use failed for %s (%s)", tool_name, type(error).__name__)
            return None

    def _tool_request(self, tool: str, args: dict, session_id: str, call_id: str) -> dict | None:
        if mind_run(self.settings):
            if tool not in MESSAGING_TOOLS or self._to_owner(args):
                return None
            masked = _leaves(args, hide_secrets)
            if masked == args:
                return None
            return {"args": masked, "source": "colonymind", "reason": "private values hidden"}
        if tool not in FILL_TOOLS | LOCAL_TOOLS or not self.owner_session(session_id):
            return None
        written = reference_ids(args)
        if tool in PROGRAM_FIELDS:
            if written:   # never placed: the call waits and says which variable to name instead
                for ref in written:
                    self._asks[(session_id, ref)] = program_refusal(written)
                return None
            refs = named_refs(tool, args)
            if not refs:
                return None
            if not _local_backend():
                for ref in refs:
                    self._asks[(session_id, ref)] = (
                        f"{ref} cannot be set for a command on this terminal backend (the value would have to "
                        "travel as command text); store it with colonymind_secret use to_env NAME and forward NAME "
                        "with the backend's environment passthrough")
                return None
        else:
            refs = written
            if not refs:
                return None
            try:   # where each value would go, before anyone is asked about it
                place(tool, args, dict.fromkeys(refs, ""))
            except Unplaceable as why:
                for ref in refs:
                    self._asks[(session_id, ref)] = f"{ref} cannot be placed there: {why}"
                return None
        fills = tool not in LOCAL_TOOLS
        answer = self._resolve(refs, "fill" if fills else "use", session_id, tool=tool, **({
            "own_refs": sorted(self._own_refs(session_id) & set(refs)), "action": {"tool": tool, "args": args},
            "destination": destination(tool, args)} if fills else {}))
        values = answer.get("values") or {}
        for ref, ask in (answer.get("asks") or {}).items():
            self._asks[(session_id, ref)] = str(ask.get("text") or "")
            if ask.get("code") and ask["code"] not in self._told and self._tell_owner(
                    session_id, f"ColonyMind asks before a private value is used: {ask.get('text')}"):
                self._told.add(ask["code"])
        for ref, why in {**(answer.get("refused") or {}), **(answer.get("unavailable") or {})}.items():
            self._asks[(session_id, ref)] = str(why)
        for ref in refs:
            (self._refused.add if ref in (answer.get("refused") or {}) else self._refused.discard)((session_id, ref))
        if tool in LOCAL_TOOLS and values and not answer.get("unavailable"):
            for value in values.values():
                register_for_redaction(value)
            return {"args": place(tool, args, values), "source": "colonymind", "reason": "owner's private value placed"}
        if set(values) != set(refs):
            return None
        for value in values.values():
            register_for_redaction(value)
        if tool in PROGRAM_FIELDS:
            field = PROGRAM_FIELDS[tool]
            # A background command reads its files after Hermes reports the call over: it removes them itself.
            owner = "" if tool == "terminal" and bool(args.get("background")) else call_id
            files = {ref: self.bindings.make(owner, values[ref]) for ref in refs}
            program = (bind_shell if tool == "terminal" else bind_python)(str(args[field]), files)
            now = time.time()
            self._bound = {key: at for key, at in self._bound.items() if now - at < UNCLAIMED_SECONDS}
            self._bound[(session_id, program)] = now
            return {"args": {**args, field: program}, "source": "colonymind",
                    "reason": "owner's private value set for this call"}
        return {"args": place(tool, args, values), "source": "colonymind", "reason": "owner's private value placed"}

    def post_tool_call(self, tool_call_id: str = "", **_: Any) -> None:
        """A call is over: the private files that held its values go, unless the program that took one still runs
        (it removes its own when it ends)."""
        if tool_call_id:
            self.bindings.done(str(tool_call_id))
        self.bindings.sweep()

    def pre_tool_call(self, tool_name: str = "", args: Any = None, session_id: str = "", **_: Any) -> dict | None:
        try:
            return self._pre_tool_call(tool_name, args, session_id)
        except Exception as error:
            logger.warning("private value authorization failed (%s)", type(error).__name__)
            return {"action": "block", "message": "ColonyMind: private value authorization is unavailable. "
                    "The call was not authorized."}

    def _pre_tool_call(self, tool_name: str = "", args: Any = None, session_id: str = "") -> dict | None:
        """A call in the owner's session that still carries a reference, or a command or code naming a variable
        this plugin did not set for it, is held: its ask is open, it was refused, or the value cannot be used
        there. A reference never runs as text; a program needs its named value."""
        tool, session = str(tool_name or ""), str(session_id or "")
        if tool not in FILL_TOOLS | LOCAL_TOOLS or mind_run(self.settings) or not isinstance(args, Mapping):
            return None
        refs = reference_ids(dict(args))
        if tool in PROGRAM_FIELDS and not refs:
            program = args.get(PROGRAM_FIELDS[tool])
            if (session, program) in self._bound:
                return None
            refs = named_refs(tool, args)
        if tool in LOCAL_TOOLS and refs:
            try:   # a reference where no value goes waits; one whose value is never used is written as it is
                place(tool, dict(args), dict.fromkeys(refs, ""))
                refs = [ref for ref in refs if (session, ref) not in self._refused]
            except Unplaceable:
                pass
        if not refs or not self.owner_session(session):
            return None
        reasons = list(dict.fromkeys(self._asks.get((session, ref)) or f"{ref} needs the owner's word" for ref in refs))
        return {"action": "block", "message": "ColonyMind: a private value waits for the owner: " + "; ".join(reasons)
                + ". Tell the owner; after their yes the same call is served."}

    def _to_owner(self, args: Mapping[str, Any]) -> bool:
        target = str(args.get("target") or args.get("chat_id") or args.get("to") or "")
        platform, address = (parse_deliver(f"{args['platform']}:{target}" if args.get("platform") else target)
                             or [("", "")])[0]
        return bool(address) and self.sessions.sender_is_owner(platform, address)

    def _tell_owner(self, session_id: str, text: str) -> bool:
        """``text`` sent by the plugin itself, never through the model, to the owner's own direct chat of this
        session: what a reveal shows, and what an ask is about. False where the session has no such chat."""
        info = self.sessions.get(session_id)
        chat, chat_type = self.sessions.chat(session_id)
        if (info is None or not chat or chat_type not in {"", "dm"} or info.platform.strip().lower()
                in INTERNAL_PLATFORMS or not self.sessions.sender_is_owner(info.platform, info.sender_id)):
            return False
        from .body import send_message
        return send_message(f"{info.platform.strip().lower()}:{chat}", text)["result"] == "sent"

    def _resolve(self, refs: list[str], purpose: str, session_id: str, **extra: Any) -> dict[str, Any]:
        body = {"purpose": purpose, "lane": "owner",
                "contact_id": self.sessions.contact_id(session_id) or self.settings.owner_contact_id() or None,
                "session_id": session_id, **{key: value for key, value in extra.items() if value is not None}}

        def ask(chunk: list[str]) -> dict[str, Any]:
            try:
                response = self.client.post(RESOLVE_ROUTE, timeout=SEAL_TIMEOUT, json={"refs": chunk, **body})
            except SidecarUnavailable:
                return {"unavailable": {ref: "the sidecar is unreachable" for ref in chunk}}
            if not response.is_success:
                return {"refused": {ref: f"the sidecar answered HTTP {response.status_code}" for ref in chunk}}
            try:
                value = response.json()
            except ValueError:
                return {"refused": {ref: "the sidecar's answer was unreadable" for ref in chunk}}
            return value if isinstance(value, dict) else {}
        return _batched(refs, ask) or {}

    # -- the owner's tool ---------------------------------------------------------------------

    def _sweep_revealed(self) -> None:
        cutoff = time.time() - REVEAL_SECONDS
        for path in self.revealed.glob("*.txt") if self.revealed.is_dir() else ():
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    def _reveal_to_file(self, ref: str, value: str) -> Path:
        """A revealed value where there is no chat to send it to (a local session): a private file the owner opens
        themselves, kept ``REVEAL_SECONDS`` at most. The model is told where it is, never what it holds."""
        self._sweep_revealed()
        self.revealed.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.revealed, 0o700)
        path = self.revealed / f"{ref}-{uuid.uuid4().hex[:8]}.txt"
        with os.fdopen(os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8",
                       newline="") as out:
            out.write(value)
        # Removed on time while the process runs; a file a restart left behind goes at the next start.
        timer = threading.Timer(REVEAL_SECONDS, lambda: path.unlink(missing_ok=True))
        timer.daemon = True
        timer.start()
        return path

    def secret_tool(self, args: Any = None, *, session_id: str = "", **_: Any) -> str:
        args = args if isinstance(args, dict) else {}
        operation = str(args.get("operation") or "")
        if operation not in SECRET_OPERATIONS:
            return json.dumps({"error": f"operation is one of {', '.join(SECRET_OPERATIONS)}"})
        if not self.owner_session(str(session_id or "")) or session_env()[3].lower() not in {"", "dm"}:
            return final_answer("private values are the owner's and are used only in the owner's own chat")
        if operation == "list":
            try:
                response = self.client.get(LIST_ROUTE, timeout=SEAL_TIMEOUT)
            except SidecarUnavailable:
                return final_answer("the sidecar is unreachable")
            if not response.is_success:
                return final_answer(f"the sidecar answered HTTP {response.status_code}")
            owner = self.settings.owner_contact_id()
            entries = [{"ref": item.get("ref"), "label": item.get("label"), "state": item.get("state"),
                        "created_at": item.get("created_at"), "uses": item.get("uses"),
                        "whose": ("the owner's" if item.get("scope_contact") == owner
                                  else "another person's (never used)")}
                       for item in (response.json().get("entries") or []) if isinstance(item, dict)]
            return json.dumps({"references": entries}, ensure_ascii=False)
        ref = str(args.get("ref") or "").strip()
        if not REFERENCE.fullmatch(f"[secret {ref}: x]"):
            return json.dumps({"error": "ref is the S-XXXXXXXX part of a reference"})
        if operation in {"release", "forget"}:
            try:
                response = self.client.post(f"{LIST_ROUTE}/{ref}/{operation}", timeout=30)
            except SidecarUnavailable:
                return final_answer("the sidecar is unreachable")
            if response.status_code in {403, 404, 409}:
                return final_answer(f"not done: {response.text[:200]}")
            return response.text if response.is_success else final_answer(f"sidecar HTTP {response.status_code}")
        answer = self._resolve([ref], "reveal" if operation == "reveal" else "use", str(session_id or ""),
                               tool=f"colonymind_secret {operation} {args.get('target') or ''}".strip())
        value = (answer.get("values") or {}).get(ref)
        if value is None:
            reason = (answer.get("refused") or {}).get(ref) or (answer.get("unavailable") or {}).get(ref)
            return final_answer(reason or "the value is unavailable")
        register_for_redaction(value)   # model-facing output never carries it from here on
        if operation == "reveal":
            # Shown to the owner by the plugin itself, never through the model (whose reply Hermes would mask):
            # to their own chat, or, in a session without one, in a private file they open.
            if self._tell_owner(str(session_id or ""), f"{ref}: {value}\n({REVEAL_WARNING})"):
                return json.dumps({"ref": ref, "shown_to_owner": True, "note": "the value went to the owner's chat "
                                   "directly; it is not in your context, and you need not repeat it"})
            path = self._reveal_to_file(ref, value)
            return json.dumps({"ref": ref, "shown_to_owner": False, "file": str(path),
                               "note": "the value is in this private file, readable by the owner's account only; "
                                       "tell the owner to open it (it is removed within half an hour); it is not "
                                       "in your context, and you need not read it"}, ensure_ascii=False)
        return self._use(ref, value, args)

    @staticmethod
    def _use(ref: str, value: str, args: Mapping[str, Any]) -> str:
        target = str(args.get("target") or "")
        try:
            if target == "to_env":
                name = str(args.get("name") or "").strip()
                if not name:
                    return json.dumps({"error": "name is the variable to store it as"})
                from hermes_cli.config import save_env_value
                save_env_value(name, value)
                return json.dumps({"ok": True, "stored_as": name, "where": "Hermes' credentials"})
            if target == "to_file":
                path = Path(str(args.get("path") or "")).expanduser()
                if not path.is_file():
                    return json.dumps({"error": "path is an existing file that contains the reference"})
                text = path.read_text(encoding="utf-8")
                marker = [match for match in REFERENCE.finditer(text) if match.group(0).startswith(f"[secret {ref}:")]
                if not marker:
                    return json.dumps({"error": f"{path} does not contain {ref}"})
                for match in reversed(marker):
                    text = text[:match.start()] + value + text[match.end():]
                temporary = path.with_name(f".{path.name}.colonymind-tmp")
                fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, path.stat().st_mode & 0o777)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
                os.replace(temporary, path)
                return json.dumps({"ok": True, "filled": len(marker), "path": str(path)})
            if target == "to_login":
                origin, identifier = str(args.get("origin") or ""), str(args.get("identifier") or "")
                if not origin or not identifier:
                    return json.dumps({"error": "origin and identifier are required for a login"})
                from agent.vault_store import get_vault_store
                item = get_vault_store().add_item("login", f"{ref} login", {
                    "identifier_type": str(args.get("identifier_type") or "email"), "identifier": identifier,
                    "password": value}, origin=origin)
                return json.dumps({"ok": True, "login": getattr(item, "id", None), "origin": origin})
        except ImportError:
            return final_answer("this Hermes has no such credential store")
        except Exception as error:
            return final_answer(f"not stored ({type(error).__name__})")
        return json.dumps({"error": f"target is one of {', '.join(USE_TARGETS)}"})


def register(ctx: Any, client: ColonyMindClient, sessions: SessionMap, settings: Settings, toolset: str) -> Secrets:
    """The hooks, the middleware, the tool and the redaction patterns (all stock seams)."""
    secrets = Secrets(client, sessions, settings)
    ctx.register_hook("pre_gateway_dispatch", secrets.pre_gateway_dispatch)
    ctx.register_hook("pre_tool_call", secrets.pre_tool_call)
    ctx.register_hook("post_tool_call", secrets.post_tool_call)
    register_middleware = getattr(ctx, "register_middleware", None)
    if callable(register_middleware):
        register_middleware("tool_request", secrets.tool_request)
    register_patterns = getattr(ctx, "register_redaction_patterns", None)
    if callable(register_patterns):
        register_patterns(list(REDACTION_PATTERNS))
    ctx.register_tool(name=SECRET_SCHEMA["name"], toolset=toolset, schema=SECRET_SCHEMA, handler=secrets.secret_tool)
    return secrets


__all__ = ["EXECUTOR_TOOLS", "FILL_TOOLS", "HIDDEN", "LOCAL_TOOLS", "PROGRAM_FIELDS", "REDACTION_PATTERNS",
           "REFERENCE", "RULES", "SECRET_SCHEMA", "Secrets", "Unplaceable", "VARIABLE", "VARIABLE_PREFIX", "WITHHELD",
           "bind_python", "bind_shell", "capture_settings", "destination", "fill", "find_secrets", "hide_secrets",
           "hide_stored_send_errors", "named_refs", "place", "reference_ids", "register", "replace_secrets",
           "variable", "words_to_send"]
