"""``pre_tool_call`` rules for mind-originated and non-owner runs."""

from __future__ import annotations

from colonymind_shared.effects import (
    ACTION_KEYS, ACTION_NAME, ACTION_TIMEOUT, ACTION_TOOL, ARG_KEYS, ARG_NAME, DECLARATION_KEYS,
    EFFECT_FREE_TOOLS, EXECUTOR_TOOLS, Effect, FLOOR_PATTERNS, MESSAGE_TOOLS, MONEY_HOSTS, RESERVED_TOOLS,
    action_argv, action_problem, action_values, canonical_call, declaration_problem, effect_of,
    executor_text, finite_amount, floor_class, floor_match, money_host, read_as, table_reads,
)

import fnmatch
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import contextvars
from typing import Any, Mapping

from .capture import DM_CHAT_TYPES, SessionMap
from .capture import GROUP_WITHHELD, group_chat
from .client import MIND_STATE_ROUTE, ColonyMindClient, Settings, SidecarUnavailable, final_answer
from .standing import STANDING_FILE, StandingStore
from .authority_diagnostics import ancestry_unreadable
from .write_origin import WriteOrigin, call_origin, untrusted_addresses, request_owner_action

logger = logging.getLogger(__name__)

READ_ONLY_TOOLS = frozenset({
    "read_file", "search_files", "web_search", "web_extract", "x_search", "vision_analyze",
    "session_search", "kanban_get", "kanban_list", "kanban_attachments", "skills_list", "skill_view",
    "todo_list", "tool_search", "tool_describe", "colonymind_memory_search", "colonymind_self",
    "colonymind_people",
})
MESSAGING_TOOLS = frozenset({
    "send_message", "react_to_message", "discord", "discord_admin", "yb_send_dm", "yb_send_sticker",
    "feishu_drive_reply_comment", "feishu_drive_add_comment",
})
WRITE_TOOLS = frozenset({"write_file", "patch"})
# The owner-authored files of an instance. A mind run may read them and never change them, by any tool:
# a write target, a patch header, a shell command or code that names one is blocked before anything else.
PROTECTED_BASENAMES = ("colonymind.yaml", "identity.yaml", "api.key", STANDING_FILE)
GUARD_ROUTE = "/v1/mind/guard"
TRIPWIRE_ROUTE, TRIPWIRE_REPORT_SECONDS = "/v1/mind/tripwire", 600.0   # one report per signal and session
# The tripwire's signals, matched like a protected name: the instance key (its file or variable), an answer route.
KEY_SIGNALS = ("api.key", "colonymind_api_key")
ANSWER_ROUTES = re.compile(r"v1/mind/(answers|decide|asks/[^/]+/(yes|shown)|permit-asks/[^/]+/answer)")
CRON_ORIGIN_ROUTE = "/v1/mind/cron-origin"
LINEAGE_ROUTE = "/v1/mind/lineage"
GUARD_TIMEOUT = 2.0
# Stock cron runs its agent in a session named ``cron_<job_id>_<YYYYmmdd_HHMMSS>`` with the task id
# ``cron:<job_id>:<run>`` (cron/scheduler.py). A compressed session gets a new id and keeps the task id; a
# delegated child has its own session and task id, and its parent session is the run's.
CRON_SESSION = re.compile(r"^cron_(?P<job>.+)_\d{8}_\d{6}$")
CRON_TASK = re.compile(r"^cron:(?P<job>[^:]+):")
# How long the owner's yes to a payment fill stands in for Hermes' own fill consent (one fill of that item).
PAYMENT_CONSENT_SECONDS = 120.0
# The V4A headers stock ``patch`` writes to (tools/file_tools.py checks the same two shapes).
V4A_HEADER = re.compile(r"^\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*(.+)$", re.MULTILINE)
V4A_MOVE = re.compile(r"^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)$", re.MULTILINE)


# In a contact's session the final response is delivered to the session's own chat, and only there: a send to
# that chat repeats it, and anything meant for another recipient does not reach them through it.
REPLY = "your final response is delivered to this conversation as your reply; there is nothing to send"
UNKNOWN_RECIPIENT = ("nothing was sent: no contact is known at that target; your final response goes to this "
                     "conversation only")
NOT_SENT = ("nothing was sent: that recipient cannot be messaged from this conversation; your final response goes to "
            "this conversation only")


# A non-owner session (docs/HERMES-ADAPTER.md, Owner and guest) never runs the agent's machine. It may call only the guest
# tools below; every other tool is refused, as the one final answer and before any sidecar verdict: what runs a program
# or code, drives the desktop or a browser, writes files or the agent's shared memory, or does one of those later in
# the owner's name (a scheduled job, a skill, a board task, a declared action, a delegated child), and every tool this
# table does not know (an MCP server's, a plugin's) unless the operator names it in ``guests.tools``. A file a guest
# tool takes (a read, a search, an image or video to look at, an image to edit and its reference images, a ``MEDIA:``
# attachment or a message's media file) must be one the gateway saved from the sender's own messages (``guest_file``); a
# speech file is written only where Hermes puts it. A tool the operator names is one this table does not know, so every
# string it is given that names a local file counts.
# ``guests.executor_platforms`` in colonymind.yaml is the operator's opt-in for a platform whose non-owner sessions keep
# every tool; the instance key and the answer routes stay out of reach there too.
GUEST_TOOLS = frozenset({
    "web_search", "web_extract", "x_search", "vision_analyze", "video_analyze", "image_generate", "text_to_speech",
    "clarify", "todo", "todo_list", "skills_list", "skill_view", "tool_search", "tool_describe", "session_search",
    "read_file", "search_files"}) | MESSAGING_TOOLS
# What ``guests.tools`` can never add: the tools that run this machine or change what the owner's sessions load.
GUEST_NEVER = EXECUTOR_TOOLS | WRITE_TOOLS | frozenset({
    "cronjob_manage", "skill_manage", "kanban_create", "kanban_comment", "kanban_unblock", "kanban_complete",
    "kanban_block", "memory", "ha_call_service", ACTION_TOOL})
# The arguments through which a guest tool takes a local file (a URL or data is no file); a list or a mapping counts
# by every string in it (an image edit's reference images, a message's media files).
GUEST_FILE_ARGS = {"read_file": ("path",), "search_files": ("path",), "vision_analyze": ("image_url",),
                   "video_analyze": ("video_url",), "image_generate": ("image_url", "reference_image_urls"),
                   "yb_send_dm": ("media_files",)}
MEDIA_TAG = re.compile(r"MEDIA:\s*(\S+)")
GUEST_REFUSAL = "{tool} is not available in this conversation; answer in your final response"
GUEST_FILE_REFUSAL = ("{tool} reaches only the files sent in this conversation; answer from the message and the "
                      "recalled context in your final response")
# The question tool in a group chat's turn of someone who is not the owner: the gateway takes an answer only in the
# asker's own session, so nobody in the chat can answer, and the turn would hold the chat until the question times out.
GROUP_QUESTION = "a question tool call cannot be answered in a group chat; ask in your reply instead"
# ``search_files`` targets that list file names and read no content (stock maps ``find`` to ``files``).
NAME_SEARCHES = frozenset({"files", "find"})


def guest_allowed(tool: str, extra: Any = ()) -> bool:
    """A tool a non-owner session may call on a platform that is not opted in: a guest tool, a ColonyMind tool (each
    answers a guest itself) but the declared actions, or one the operator named in ``guests.tools``."""
    if tool in GUEST_NEVER or tool.startswith("browser_"):
        return False
    return tool in GUEST_TOOLS or (tool.startswith("colonymind_") and tool != ACTION_TOOL) or tool in set(extra or ())


def resolved_path(path: Any) -> Path | None:
    """A path as the file tools reach it: the user's home expanded, a relative one against the terminal's directory,
    every link followed; None when it cannot be told."""
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = Path(os.path.expanduser(path.strip()))
    if not candidate.is_absolute():
        candidate = Path(os.environ.get("TERMINAL_CWD") or os.getcwd()) / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def local_file(value: Any) -> str | None:
    """The local file an argument names (a path, ``~``, ``file://``), or None for a web or data URL or nothing."""
    text = str(value or "").strip()
    if not text or re.match(r"^(https?|data|blob):", text, re.IGNORECASE):
        return None
    return text[len("file://"):] if text.lower().startswith("file://") else text


def visible_paths(path: str) -> set[str]:
    """A saved attachment as the file tools may name it: its host path, links followed, and where the terminal
    backend sees it (Hermes' own translation; a container mounts the cache elsewhere)."""
    names = {path}
    resolved = resolved_path(path)
    if resolved is not None:
        names.add(str(resolved))
    try:
        from tools.credential_files import to_agent_visible_cache_path
        names.add(str(to_agent_visible_cache_path(path)))
    except Exception:
        pass
    return names


def guest_file(value: Any, sent: list[str]) -> bool:
    """Whether a file a guest tool names is one the gateway saved from this conversation's own messages: the same
    file once links are followed (or, on a backend that sees it elsewhere, the path Hermes gives it there). A path
    anyone typed, a directory, another conversation's file and anything the name does not resolve to never are."""
    path = local_file(value)
    if path is None:
        return True
    allowed = set().union(*(visible_paths(item) for item in sent)) if sent else set()
    if path.strip() in allowed:
        target = resolved_path(path)
        return target is None or not target.is_dir()
    target = resolved_path(path)
    return target is not None and not target.is_dir() and str(target) in allowed


def strings_in(value: Any) -> list[str]:
    """Every string in an argument value, nested lists and mappings included (their values, not their keys)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in strings_in(item)]
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in strings_in(item)]
    return []


def names_local_file(value: Any) -> bool:
    """Whether a string a tool is given names a local file or directory: a path from the root or the home, a
    ``file://`` link, a relative path (a separator and no spaces, or a leading ``./`` or ``../``), or a name that exists
    where the tool would look (the terminal's directory or this process's). A web or data link and text do not."""
    text = str(value or "").strip()
    path = local_file(text)
    if path is None:
        return False
    if text.lower().startswith("file:") or path.startswith(("/", "~", "./", "../")):
        return True
    if not any(character.isspace() for character in path) and ("/" in path or "\\" in path):
        return True
    try:
        bases = [os.environ.get("TERMINAL_CWD"), os.getcwd()]
    except OSError:              # this process's directory is gone: a bare name counts as a file
        return True
    for base in bases:
        try:
            if base and (Path(base) / os.path.expanduser(path)).exists():
                return True
        except (OSError, ValueError):
            continue
    return False


def guest_files_ok(tool: str, args: Mapping[str, Any], sent: list[str]) -> bool:
    """Every file a guest's call takes is one its conversation sent, and a speech file goes where Hermes puts it. The
    guest tools' file arguments are known; for a tool the operator names in ``guests.tools`` (which the table does
    not know), every string it is given that names a local file is one."""
    if tool == "text_to_speech" and str(args.get("output_path") or "").strip():
        return False
    names: list[Any] = []
    for name in GUEST_FILE_ARGS.get(tool, ()):
        if name in args:
            names += strings_in(args[name]) if isinstance(args[name], (list, tuple, set, Mapping)) else [args[name]]
        elif tool == "search_files":
            names.append(".")
    for value in args.values():
        if isinstance(value, str):
            names += MEDIA_TAG.findall(value)
    if tool not in GUEST_TOOLS and not tool.startswith("colonymind_"):
        for text in strings_in(dict(args)):
            if names_local_file(text):
                names.append(text)
            names += MEDIA_TAG.findall(text)
    return all(guest_file(name, sent) for name in names if name is not None)


def search_roots(path: Any) -> list[str]:
    """The roots a ``search_files`` call can search: its whole ``path`` and, since stock searches each part of a path
    that names nothing (``_try_multi_path_search``: split at commas, else at whitespace), every part of it too."""
    text = str(path if path is not None else ".")
    parts = [part.strip() for part in text.split(",")] if "," in text else text.split()
    return [text, *(part for part in parts if part and part != text)]


GLOB_ALTERNATIVES = 256   # a glob that spells out more alternatives than this counts as matching


def glob_alternatives(glob: str) -> list[str] | None:
    """A glob's ``{a,b}`` alternatives spelled out, nested ones too, the way ripgrep reads them (a ``\\`` escapes the
    next character); None when its braces do not pair or it spells out more than ``GLOB_ALTERNATIVES``."""
    depth, start, cuts, index = 0, -1, [], 0
    while index < len(glob):
        character = glob[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
            if depth == 1:
                start, cuts = index, []
        elif character == "}":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0:
                head, tail, bounds = glob[:start], glob[index + 1:], [start, *cuts, index]
                spelled: list[str] = []
                for left, right in zip(bounds, bounds[1:]):
                    rest = glob_alternatives(head + glob[left + 1:right] + tail)
                    if rest is None or len(spelled) + len(rest) > GLOB_ALTERNATIVES:
                        return None
                    spelled += rest
                return spelled
        elif character == "," and depth == 1:
            cuts.append(index)
        index += 1
    return None if depth else [glob]


def glob_may_match(glob: Any, names: Any) -> bool:
    """Conservatively match search globs against protected file names, including unreadable patterns."""
    glob = str(glob or "").strip()
    if not glob or glob.startswith("!"):
        return True
    alternatives = glob_alternatives(glob)
    if alternatives is None:
        return True
    names = {str(name).casefold() for name in names}
    for candidate in {*alternatives, *(part.strip() for part in glob.split(",") if part.strip())}:
        pattern = re.sub(r"\\(.)", r"\1", candidate).replace("[^", "[!").lstrip("/").casefold()
        for form in {pattern, pattern.replace("**/", ""), pattern.replace("/**", "")}:
            if any(fnmatch.fnmatchcase(name, form) for name in names):
                return True
    return False


def reaches_key(tool: str, args: Mapping[str, Any], key_file: Any) -> bool:
    """Whether a read reaches the instance key file however it is spelled: a ``read_file`` whose path resolves to it
    (a link, a relative path, another case), or a ``search_files`` of contents with any root (``search_roots``) that is
    the key file (a file named as the root is searched whatever the glob says) or a directory that holds it where its
    ``file_glob``, if any, could match it (``glob_may_match``)."""
    if tool not in ("read_file", "search_files"):
        return False
    key = resolved_path(str(key_file or ""))
    if key is None:
        return False
    if tool == "read_file":
        return resolved_path(str(args.get("path") or "")) == key
    if str(args.get("target") or "content") in NAME_SEARCHES:
        return False
    for root in search_roots(args.get("path")):
        target = resolved_path(root or ".")
        if target is None:
            continue
        if target == key:
            return True
        if key.is_relative_to(target) and glob_may_match(args.get("file_glob"),
                                                          (key.name, str(key.relative_to(target)))):
            return True
    return False


def block(message: str) -> dict[str, str]:
    return {"action": "block", "message": f"BLOCKED by ColonyMind guard: {message}"}


def vault_meta(handle: Any) -> Any:
    """A Hermes vault item's metadata (kind, label, bound origin), or None when it cannot be read."""
    try:
        from agent.vault_backends import backend_for_handle
        backend = backend_for_handle(str(handle or ""))
        return backend.get_meta(str(handle)) if backend is not None else None
    except Exception:
        return None


def vault_kind(handle: Any) -> str:
    """The kind of a Hermes vault item (``login``, ``payment``, ``address``), or '' when it cannot be read (a
    locked backend, no such item): an unknown kind is held as money."""
    return str(getattr(vault_meta(handle), "kind", "") or "")


# Stock ``browser_vault_fill`` asks its own consent (``_confirm_payment_fill``) before a card is written into a
# page, on the surface of the session; a kanban worker has none and the fill is refused. When the sidecar lets a
# mind run's fill of one payment item through on the owner's exact yes (a permit) or their standing grant, that is
# the consent for that fill: the item's handle in that task, once, within ``PAYMENT_CONSENT_SECONDS``, asked from
# inside that fill (the fill is wrapped to say which handle and task it runs for). Another item with the same label
# and site, the same item from another task, and every other fill ask the stock consent as before; a Hermes
# without those seams refuses the fill (nothing is widened). The fill is the same function whether Hermes offers it
# as the tool ``browser_vault_fill`` or as ``browser_vault`` with ``action: "fill"`` (``read_as``). The yes ends
# with the fill it was given for: a Hermes that refuses the fill before it asks (a page on another origin) leaves
# no yes behind for the next one.
_PAYMENT_CONSENTS: dict[tuple[str, str], float] = {}
_PAYMENT_LOCK = threading.Lock()
_FILLING: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("colonymind_fill", default=None)


def _filling(stock: Any) -> Any:
    def fill(handle: Any = "", task_id: Any = None, *args: Any, **kwargs: Any) -> Any:
        filling = (str(handle or ""), str(task_id or ""))
        token = _FILLING.set(filling)
        try:
            return stock(handle, task_id, *args, **kwargs)
        finally:
            _FILLING.reset(token)
            with _PAYMENT_LOCK:     # whatever Hermes answered, and whether or not it asked
                _PAYMENT_CONSENTS.pop(filling, None)
    fill.colonymind_consent = True  # type: ignore[attr-defined]
    return fill


def _consenting(stock: Any) -> Any:
    def confirm(label: str, origin: str) -> bool:
        filling = _FILLING.get()
        with _PAYMENT_LOCK:
            until = _PAYMENT_CONSENTS.pop(filling, None) if filling is not None else None
        if until is not None and time.monotonic() <= until:
            return True
        return stock(label, origin)
    confirm.colonymind_consent = True  # type: ignore[attr-defined]
    return confirm


def consent_payment_fill(handle: Any, task_id: Any = "") -> bool:
    """The owner's yes to a fill of this payment item in this task, given to Hermes' fill consent for that one fill."""
    meta = vault_meta(handle)
    if meta is None or str(getattr(meta, "kind", "") or "") != "payment":
        return False
    try:
        import tools.browser_vault_tool as vault_tool
    except Exception:
        return False
    stock_confirm, stock_fill = (getattr(vault_tool, "_confirm_payment_fill", None),
                                 getattr(vault_tool, "browser_vault_fill", None))
    if not callable(stock_confirm) or not callable(stock_fill):
        return False
    if not getattr(stock_confirm, "colonymind_consent", False):
        vault_tool._confirm_payment_fill = _consenting(stock_confirm)
    if not getattr(stock_fill, "colonymind_consent", False):
        vault_tool.browser_vault_fill = _filling(stock_fill)
    with _PAYMENT_LOCK:
        _PAYMENT_CONSENTS[(str(handle or ""), str(task_id or ""))] = time.monotonic() + PAYMENT_CONSENT_SECONDS
    return True


def terminal_cwd(args: Mapping[str, Any], task_id: str) -> str | None:
    """The directory Hermes runs a terminal call in (``tools/terminal_tool.py``): ``workdir``, else the session's
    recorded cwd (an earlier ``cd`` moved it), else the task's override or the configured one. None when it cannot
    be told (it cannot be read, two records disagree, a relative ``workdir``) or is no host path (a container
    backend): then no relative path counts as inside the workspace."""
    workdir = str(args.get("workdir") or "")
    try:
        from tools.approval import get_current_session_key
        from tools.terminal_tool import _get_env_config, get_session_cwd, resolve_task_overrides
        config = _get_env_config()
        if str(config.get("env_type") or "local") != "local":
            return None
        if workdir:
            return workdir if os.path.isabs(workdir) else None
        keys = dict.fromkeys(key for key in (get_current_session_key(default="") or "", str(task_id or "")) if key)
        recorded = {cwd for cwd in (get_session_cwd(key) for key in keys) if cwd}
        if len(recorded) > 1:
            return None
        if recorded:
            return recorded.pop()
        overrides = resolve_task_overrides(task_id) if task_id else {}
        value = str((overrides or {}).get("cwd") or config.get("cwd") or "")
        return value if os.path.isabs(value) else None
    except Exception:
        return None


def cron_context() -> bool:
    """Whether Hermes runs this call inside a cron run (its ``HERMES_CRON_SESSION`` marker)."""
    try:
        from gateway.session_context import get_session_env
        value = get_session_env("HERMES_CRON_SESSION", "")
    except Exception:
        value = os.environ.get("HERMES_CRON_SESSION", "")
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def mind_run(settings: Settings) -> bool:
    """A worker with positively recorded mind creation authority; a profile grants nothing."""
    task = os.environ.get("HERMES_KANBAN_TASK") or ""
    if not task:
        return False
    home = Path(getattr(settings, "home", settings.outbox_path.parent))
    saved = StandingStore(home / STANDING_FILE).get("kanban", task)
    return saved is not None and saved[0] == "mind"


def workspace() -> Path | None:
    value = os.environ.get("HERMES_KANBAN_WORKSPACE") or ""
    return Path(value).resolve() if value else None


def path_inside(path: Any, root: Path | None) -> bool:
    if root is None or not isinstance(path, str) or not path.strip():
        return False
    resolved = Path(os.path.expanduser(path))
    if not resolved.is_absolute():
        resolved = Path(os.environ.get("TERMINAL_CWD") or os.getcwd()) / resolved
    try:
        return resolved.resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def write_targets(tool: str, args: Mapping[str, Any]) -> list[Any]:
    """Every path a ``write_file`` or ``patch`` call writes."""
    targets: list[Any] = [args["path"]] if args.get("path") else []
    if tool == "patch" and args.get("mode") == "patch":
        patch = str(args.get("patch") or "")
        targets += [match.group(1).strip() for match in V4A_HEADER.finditer(patch)]
        targets += [group.strip() for match in V4A_MOVE.finditer(patch) for group in match.groups()]
    return targets


# What a shell or code spelling can put between the letters of a name without changing the file it
# reaches: quotes, escapes, whitespace, backticks and string concatenation.
SPELLING = re.compile(r"[\s'\"\\`+]")


def names_protected_file(tool: str, args: Mapping[str, Any]) -> str | None:
    """Find protected basenames in write targets or arguments, ignoring literal spelling separators."""
    if tool in WRITE_TOOLS:
        haystack = [str(target) for target in write_targets(tool, args)]
    else:
        haystack = [json.dumps(args, ensure_ascii=False, sort_keys=True)]
    spelled = [SPELLING.sub("", text).casefold() for text in haystack]
    return next((name for name in PROTECTED_BASENAMES for text in spelled if name in text), None)


def forgery_signals(tool: str, args: Mapping[str, Any]) -> list[str]:
    """``api.key`` and ``answer_route`` when a call's arguments name them (spelling ignored)."""
    text = SPELLING.sub("", json.dumps(dict(args), ensure_ascii=False, sort_keys=True, default=str)).casefold()
    return [name for name, hit in (("api.key", any(key in text for key in KEY_SIGNALS)),
                                   ("answer_route", bool(ANSWER_ROUTES.search(text))),
                                   ("standing_store", STANDING_FILE in text)) if hit]


# The core's wrapper for deferred tools: ``tool_call {calls: [{name, arguments}]}``. The core unwraps it before this
# hook, so a call still named ``tool_call`` here is one the core could not resolve (it answers with its own error and
# nothing runs) or one of connector tools only, whose entries the core then brings to this hook one by one under
# their own names. The wrapper is no call of its own: a mind run's wrapper is never judged, held or filed under its
# name.
WRAPPER_TOOL = "tool_call"
WRAPPER_NEVER = frozenset({"tool_search", "tool_describe", WRAPPER_TOOL})   # what the core refuses inside a wrapper
CONNECTOR_PREFIX = "connectors__"
WRAPPER_UNREADABLE = ("this tool_call could not be read and nothing ran: write the call again, either calling the "
                      "tool by its own name or giving calls as a list of {name, arguments}")
WRAPPER_DIRECT = "call {tool} directly, by its own name: nothing ran through tool_call"


def wrapped_calls(args: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]] | None:
    """The calls a ``tool_call`` wrapper carries as (name, arguments), read as the core reads them (stock
    ``normalize_tool_call_entries``): ``calls`` as a list, as one entry or as JSON text, or with no ``calls`` the
    single ``{name, arguments}`` form; an entry's arguments as a mapping, as JSON text or absent. None for a wrapper
    the core refuses to read."""
    calls = args.get("calls")
    if calls is None:
        calls = [{"name": args.get("name"), "arguments": args.get("arguments")}]
    try:
        if isinstance(calls, str):
            calls = json.loads(calls)
        if isinstance(calls, Mapping):
            calls = [calls]
        if not isinstance(calls, list) or not calls:
            return None
        entries: list[tuple[str, dict[str, Any]]] = []
        for entry in calls:
            name = str(entry.get("name") or "").strip() if isinstance(entry, Mapping) else ""
            if not name or name in WRAPPER_NEVER:
                return None
            arguments = entry.get("arguments")
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                arguments = {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, Mapping):
                return None
            entries.append((name, dict(arguments)))
        return entries
    except ValueError:   # JSON text that does not parse
        return None


def parse_deliver(value: Any) -> list[tuple[str, str]]:
    """``deliver`` targets as (platform, chat_id); 'origin' and 'local' are literal."""
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    targets: list[tuple[str, str]] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part in {"local", "origin"}:
            targets.append((part, ""))
        elif part == "all" or part.startswith("bot-chat"):
            targets.append(("*", part))
        else:
            pieces = part.split(":")
            targets.append((pieces[0], pieces[1] if len(pieces) > 1 else ""))
    return targets


_hand_session_check = lambda session, task: False
_hand_session_standing = lambda session, task: "unknown"


def register_hand_sessions(check, standing=None) -> None:
    """The hands plugin supplies its structural registry, including rotated ids."""
    global _hand_session_check, _hand_session_standing
    _hand_session_check = check
    _hand_session_standing = standing or (lambda session, task: "unknown")


class Guard:
    def __init__(self, client: ColonyMindClient, sessions: SessionMap, settings: Settings):
        self.client, self.sessions, self.settings = client, sessions, settings
        self._enabled_cache: tuple[float, bool] | None = None
        # cron run (session and task id) -> (whose job it is: mind | owner, the kanban task that made a mind job)
        self._cron_origins: dict[str, tuple[str, str]] = {}
        # session -> the cron job it runs for ('' for none), read from the task id or the session's ancestry
        self._cron_jobs: dict[str, str] = {}
        self._tripped: dict[tuple[str, str], float] = {}   # (session, signal) -> when the tripwire reported it
        self.sessions.automation_standing = self.automation_standing
        self.write_origin = self.sessions.write_guard = WriteOrigin(self)

    @staticmethod
    def _creator_key(session_id: str, task_id: str, tool: str, tool_call_id: str) -> str:
        return json.dumps([session_id, task_id, tool, tool_call_id], separators=(",", ":")) if tool_call_id else ""

    def _remember_creator(self, tool: str, args: Mapping[str, Any], session_id: str, task_id: str,
                          tool_call_id: str, turn_id: str, standing: tuple[str, str]) -> None:
        # Tool completion can arrive after another group speaker. Retain the
        # initiating turn instead of resolving the group's latest speaker then.
        if tool == "delegate_task" and turn_id:
            self.sessions.standings.put("delegation", json.dumps([session_id, turn_id]),
                                        standing[0], standing[1], session_id, replace=False)
        action = str(args.get("action") or "").strip().lower()
        if tool == "kanban_create" or (tool == "cronjob_manage" and action in {"create", "update"}):
            self.sessions.standings.put("creation", self._creator_key(session_id, task_id, tool, tool_call_id),
                                        standing[0], standing[1], session_id, replace=False)

    def automation_standing(self, session_id: str, task_id: str = "") -> tuple[str, str]:
        """Creation-time authority from the trusted dispatcher or registered cron creator."""
        worker_task = os.environ.get("HERMES_KANBAN_TASK")
        if worker_task:
            inherited = self.sessions.standings.get("kanban", worker_task)
            return (inherited[0], inherited[1] or "worker") if inherited else ("unknown", "worker")
        origin = self._cron_origin(session_id, task_id)
        return origin or "unknown", "cron" if origin else ""

    def session_class(self, session_id: str, task_id: str = "") -> str:
        """Fixed classes only: no session ids, sender ids or arbitrary platform strings in logs."""
        if CRON_SESSION.match(session_id) or CRON_TASK.match(task_id) or cron_context():
            return "cron"
        info = self.sessions.get(session_id)
        platform = info.platform.lower() if info is not None else ""
        return platform if platform in {"subagent", "colonymind_task", "companion", "cli", "system"} else (
            "gateway" if info is not None and info.sender_id else "unregistered")

    def unknown(self, tool: str, session_id: str, task_id: str = "") -> dict[str, str]:
        try:
            session_class = self.session_class(session_id, task_id)
        except Exception:
            session_class = "unregistered"
        logger.warning("standing_unknown tool=%s session_class=%s", tool, session_class)
        return final_answer(GUEST_REFUSAL.format(tool=tool), block=True)

    def standing(self, session_id: str, task_id: str = "") -> tuple[str, str]:
        """Resolve trusted automation origins explicitly; unknown cron ownership grants nothing."""
        info = self.sessions.get(session_id)
        if info is not None and info.sender_id:
            return self.sessions.standing(session_id)
        inherited = self.sessions.creation_standing(session_id)
        if inherited is not None:
            return inherited
        run, platform = self.automation_standing(session_id, task_id)
        if platform:
            return run, platform
        return self.sessions.standing(session_id)

    def subagent_start(self, *, child_session_id: str = "", parent_session_id: str = "",
                       parent_turn_id: str = "", **_: Any) -> None:
        try:
            saved = self.sessions.standings.get("delegation", json.dumps([parent_session_id, parent_turn_id])) \
                if parent_turn_id else None
            info = self.sessions.get(parent_session_id)
            changed = bool(parent_turn_id and (info is None or info.turn_id != parent_turn_id))
            run, platform = saved or (self.standing(parent_session_id) if parent_session_id and not changed else (
                "unknown", ""))
        except Exception as error:
            logger.warning("standing_inheritance_failed exception_type=%s", type(error).__name__)
            run, platform = "unknown", ""
        self.sessions.inherit(child_session_id, parent_session_id, standing=run, platform=platform)

    # -- hook ---------------------------------------------------------------

    def pre_tool_call(self, tool_name: str = "", args: Any = None, session_id: str = "", task_id: str = "",
                      tool_call_id: str = "", **_: Any) -> dict[str, str] | None:
        tool = str(tool_name or "")
        self.write_origin.clear_call()
        try:
            self.write_origin.enter(tool, args if isinstance(args, Mapping) else {}, str(session_id or ""),
                                    str(_.get("turn_id") or ""))
            verdict = self.decide(tool, args if isinstance(args, Mapping) else {}, str(session_id or ""),
                                  str(task_id or ""), str(tool_call_id or ""), str(_.get("turn_id") or ""))
            if verdict is None or verdict.get("action") != "block":
                verdict = self.write_origin.register(tool, args if isinstance(args, Mapping) else {}) or verdict
            return verdict
        except Exception as error:  # a failing guard blocks effects, never permits them
            self.write_origin.clear_call()
            logger.warning("guard error on %s (%s)", tool, type(error).__name__)
            reader = self._error_read(tool, args if isinstance(args, Mapping) else {}, str(session_id or ""))
            return None if reader else block(f"guard error ({type(error).__name__})")

    def _error_read(self, tool: str, args: Mapping[str, Any], session_id: str) -> bool:
        """Whether a read runs on when the guard itself failed: in a mind run (its reads never wait on the guard) but
        its read of the instance key, and in a session positively established as the owner's (its standing, else the
        session map's own reading of its sender). A contact's session the map cannot place (a compressed or bound
        one), a session nothing tells about and anything that cannot be read are refused."""
        if tool not in READ_ONLY_TOOLS:
            return False
        try:
            if _hand_session_check(session_id, "") and _hand_session_standing(session_id, "") in {
                    "owner", "mind", "system"}:
                return not (forgery_signals(tool, args) or self._authority_read(tool, args))
            try:
                run, _ = self.standing(session_id)
            except Exception:
                return self.sessions.known_owner(session_id)
            if run in {"mind", "system"}:
                return not (forgery_signals(tool, args) or self._authority_read(tool, args))
            return run == "owner"
        except Exception:
            return False

    def _authority_read(self, tool: str, args: Mapping[str, Any]) -> bool:
        stored = getattr(getattr(self.sessions, "standings", None), "path", None)
        return reaches_key(tool, args, self.settings.key_file) or reaches_key(tool, args, stored)

    def post_tool_call(self, tool_name: str = "", args: Any = None, result: Any = None, session_id: str = "",
                       task_id: str = "", tool_call_id: str = "", **_: Any) -> None:
        """After a mind run's call returned: a cron job it created or changed (by the id stock cron gave it) and a
        kanban task it created are told to the sidecar, so the job's runs are mind runs and the task counts with
        its maker. It never blocks; a report that does not arrive leaves the sidecar's pending mark, which keeps
        the jobs made around it gated."""
        try:
            self.write_origin.result(result, session_id, str(_.get("turn_id") or ""))
            self._report(str(tool_name or ""), args if isinstance(args, Mapping) else {}, result,
                         str(session_id or ""), str(task_id or ""), str(tool_call_id or ""))
        except Exception as error:
            logger.warning("post-call report on %s failed (%s)", tool_name, type(error).__name__)
        finally:
            self.write_origin.clear_call()
        return None

    def decide(self, tool: str, args: Mapping[str, Any], session_id: str, task_id: str = "",
               tool_call_id: str = "", turn_id: str = "") -> dict[str, str] | None:
        try:
            hand = _hand_session_check(session_id, task_id)
        except Exception as error:
            logger.warning("standing_hand_unavailable tool=%s exception_type=%s", tool, type(error).__name__)
            return self.unknown(tool, session_id, task_id)
        try:
            run, platform = (_hand_session_standing(session_id, task_id), "subagent") if hand else self.standing(
                session_id, task_id)
        except Exception as error:
            # An independent, positively verified current sender may still
            # establish standing. No parent, binding cache or platform default
            # substitutes for that evidence.
            try:
                known = self.sessions.known_owner(session_id)
            except Exception:
                known = False
            run, platform = ("owner" if known else "unknown"), ""
            logger.warning("standing_fallback tool=%s exception_type=%s standing=%s",
                           tool, type(error).__name__, run)
        self.write_origin.resolve(run, platform, session_id, task_id, hand)
        if run not in {"owner", "mind", "system", "guest"}:
            self._log_mind_decision(tool, run, platform, session_id, task_id, hand)
            return self.unknown(tool, session_id, task_id)
        self._log_mind_decision(tool, run, platform, session_id, task_id, hand)
        self._remember_creator(tool, args, session_id, task_id, tool_call_id, turn_id, (run, platform))
        if tool == "session_search" and group_chat():
            # A turn in a group chat replies to everyone in it, the owner's turn included: the search reads other
            # conversations, so the turn and the sessions it delegates to get none.
            return final_answer(GROUP_WITHHELD, block=True)
        if tool == "clarify" and run == "guest" and group_chat():
            return final_answer(GROUP_QUESTION, block=True)
        if hand and run != "guest":
            named = self.tripwire(tool, args, session_id, "hand", strict=True)
            if named:
                return block(f"{tool} names {', '.join(named)}: hand sessions cannot read the instance key or answer asks")
            # Hand allow-lists, budgets and floor permits govern this lane,
            # including hands started by the mind with the assistant's shell.
            return None
        mind = run == "mind"
        guest = run == "guest"
        try:
            named = self.tripwire(tool, args, session_id, run, strict=mind or guest)
        except Exception:  # what a call names could not be read: a guest's or a mind run's call is refused
            named = ["unreadable"] if mind or guest else []
        if named and guest:
            return final_answer(GUEST_REFUSAL.format(tool=tool), block=True)
        if named and mind:
            protected = names_protected_file(tool, args) if tool not in READ_ONLY_TOOLS else None
            return block(f"{protected} is owner-authored; a mind task cannot change it" if protected else
                         f"{tool} names {', '.join(named)}: the instance key and the answer routes are the owner's; "
                         "a mind task never reads or names them")
        if not mind and not guest:
            return None
        if guest:
            try:   # a contact's session never runs this machine: refused before any verdict, with no sidecar at all
                refusal = self._guest_refusal(tool, args, session_id, platform)
            except Exception:  # anything that cannot be read about a guest's call refuses it
                refusal = GUEST_REFUSAL.format(tool=tool)
            if refusal:
                return final_answer(refusal, block=True)
        if guest and tool in MESSAGING_TOOLS:
            answer = self._guest_message(args, session_id)
            if answer is not None:
                return final_answer(answer, block=True)
        if mind and tool == WRAPPER_TOOL:
            return self._wrapper(args, session_id)
        try:
            verdict = self._rules(tool, args, session_id, mind, task_id=task_id, tool_call_id=tool_call_id)
        except Exception as error:  # fails closed, and still answers a contact's turn finally
            if not guest:
                raise
            verdict = block(f"guard error ({type(error).__name__})")
        if guest and verdict is not None and verdict.get("action") == "block":
            # Final for the turn, so a contact's turn does not spend its iterations on other arguments, and
            # worded for the reply: the model repeats a "blocked" to the contact.
            return final_answer("session_search is unavailable to non-owner sessions; answer from the message and "
                                "the recalled context" if tool == "session_search" else NOT_SENT
                                if tool in MESSAGING_TOOLS else
                                f"{tool} is not available in this conversation; answer in your final response",
                                block=True)
        return verdict

    def _wrapper(self, args: Mapping[str, Any], session_id: str) -> dict[str, str] | None:
        """A mind run's ``tool_call`` wrapper gets no verdict of its own and never reaches the sidecar. One the core
        cannot read goes back to be written again. One that reads passes only when no entry waits on a verdict here:
        a connector tool (the core brings each such entry to this hook under its own name, where it is judged) or a
        read that a direct call is let through with. For any other entry the wrapper goes back to be called
        directly, where the call is judged, and held, as that tool."""
        entries = wrapped_calls(args)
        if entries is None:
            return block(WRAPPER_UNREADABLE)
        for name, arguments in entries:
            if name.startswith(CONNECTOR_PREFIX):
                continue
            if name in READ_ONLY_TOOLS and not (name == "session_search" and group_chat()) \
                    and not self.tripwire(name, arguments, session_id, "mind", strict=True):
                continue
            return block(WRAPPER_DIRECT.format(tool=name[:80]))
        return None

    def _rules(self, tool: str, args: Mapping[str, Any], session_id: str, mind: bool, *, task_id: str = "",
               tool_call_id: str = "") -> dict[str, str] | None:
        if tool in READ_ONLY_TOOLS:
            return block("session_search") if not mind and tool == "session_search" else None
        text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        if mind:
            if not self.mind_enabled():
                return block("the mind is off; no effects until it is turned on")
            protected = names_protected_file(tool, args)
            if protected:
                return block(f"{protected} is owner-authored; a mind task cannot change it")
            verdict = self._deny(tool, text) or self._deny(read_as(tool, args), text)
            if verdict is not None:
                return verdict
            # The worker was dispatched for this commitment; its own word that it is done proves
            # nothing. The body's outcome report closes the row once the task itself is done.
            if tool == "colonymind_resolve_commitment" and str(args.get("action") or "").strip().lower() == "fulfilled":
                return block("a mind task cannot mark its own commitment fulfilled; finish the task and report "
                             "what you did, the outcome settles the commitment")
            if tool == "kanban_create":
                if str(args.get("assignee") or "") != self.settings.worker_profile:
                    return block(f"mind tasks may only create tasks assigned to {self.settings.worker_profile}")
                if str(args.get("idempotency_key") or "").startswith("mind:"):
                    return block("mind: idempotency keys belong to the sidecar")
                # The dispatcher hands a child whatever workspace it names; a child
                # of this task gets a fresh scratch workspace or a directory inside this one.
                if args.get("project") or args.get("project_id"):
                    return block("child tasks inherit the task's project")
                if (str(args.get("workspace_kind") or "scratch") != "scratch" or args.get("workspace_path")) \
                        and not path_inside(args.get("workspace_path"), workspace()):
                    return block("child task workspaces must stay inside the task workspace")
            if tool in WRITE_TOOLS:
                targets = write_targets(tool, args)
                if not targets or not all(path_inside(target, workspace()) for target in targets):
                    return block("writes must stay inside the task workspace")
        context = self._effect_context(tool, args) if mind else {}
        cwd = terminal_cwd(args, task_id) if mind and tool == "terminal" else None
        effect = self._effect(tool, args, context, cwd) if mind else None
        if effect is not None and effect.floor and self.client.has_mind_routes() is not True:
            # Only the sidecar holds an effect as the owner's ask; without its mind routes it waits here.
            return block(f"{effect.describe()} waits for the owner's yes, which this sidecar cannot take; "
                         "stop and report that it waits")
        cron_action = str(args.get("action") or "").strip().lower()  # what stock cronjob() runs
        recipients: list[str] = []  # the contacts a delivering cron job reaches later
        if tool in MESSAGING_TOOLS or (tool == "cronjob_manage" and cron_action in {"create", "update"}):
            verdict = self._outbound(tool, args, session_id, cron_action, recipients)
            if verdict is not None:
                return verdict
        return self._sidecar_verdict(tool, args, session_id, "mind" if mind else "guest", recipients,
                                     effect=effect, context=context, cwd=cwd, task_id=task_id,
                                     tool_call_id=tool_call_id)

    def tripwire(self, tool: str, args: Mapping[str, Any], session_id: str, run: str, *,
                 strict: bool = False) -> list[str]:
        """Report protected-key and answer-route signals without arguments; strictly refuse non-owner access."""
        named = forgery_signals(tool, args)
        if strict and "api.key" not in named and reaches_key(tool, args, self.settings.key_file):
            named = ["api.key", *named]
        stored = getattr(getattr(self.sessions, "standings", None), "path", None)
        if strict and "standing_store" not in named and reaches_key(tool, args, stored):
            named = ["standing_store", *named]
        try:
            now, seen = time.monotonic(), self._tripped
            fresh = [item for item in named if now - seen.get((session_id, item), -1e9) >= TRIPWIRE_REPORT_SECONDS]
            if fresh and self.client.has_mind_routes() is True:
                logger.warning("answer_forgery_attempt: %s in a %s session named %s", tool, run, fresh)
                seen.update({(session_id, item): now for item in fresh})
                self.client.post(TRIPWIRE_ROUTE, timeout=GUARD_TIMEOUT, json={
                    "kind": "answer_forgery_attempt", "tool": tool, "named": fresh, "run": run,
                    "session_id": session_id or None, "refused": strict})
        except Exception as error:  # a report that does not arrive changes nothing about the call
            logger.debug("tripwire report failed (%s)", type(error).__name__)
        return named

    # -- guests -------------------------------------------------------------------

    def _guest_refusal(self, tool: str, args: Mapping[str, Any], session_id: str, platform: str) -> str:
        """Why a non-owner session's call is refused ('' when it is not): a tool no guest may call, or a file it takes
        that this conversation did not send. A platform the operator opted in keeps every tool."""
        guests = self.settings.guests()
        if platform and platform in guests["executor_platforms"]:
            return ""
        if not guest_allowed(tool, guests["tools"]):
            return GUEST_REFUSAL.format(tool=tool)
        info = self.sessions.origin(session_id)       # a delegated child's is the chat its origin's turn was bound to
        sent = self.sessions.sent_files(info.platform, info.sender_id, self.sessions.chat(info.session_id)[0]) \
            if info is not None else []
        return "" if guest_files_ok(tool, args, sent) else GUEST_FILE_REFUSAL.format(tool=tool)

    # -- effects ----------------------------------------------------------------

    def _effect_context(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """What only the plugin can read about a call's target: a vault item's kind, and the id of the cron job
        a removal or update names (a job may be named by its name; the sidecar knows the jobs this task made by
        id)."""
        context: dict[str, Any] = {}
        if read_as(tool, args) == "browser_vault_fill":
            context["vault_kind"] = vault_kind(args.get("handle"))
        if tool == "cronjob_manage" and str(args.get("action") or "").strip().lower() in {"remove", "delete", "update"}:
            context["job_id"] = str(self._stored_job(args.get("job_id")).get("id") or "")
        return context

    def _effect(self, tool: str, args: Mapping[str, Any], context: Mapping[str, Any],
                cwd: str | None = None) -> Effect | None:
        """The call's effect by the shared table, with ``mind.floor`` (declared tools, money hosts), ``mind.actions``
        and, for a command, the directory Hermes runs it in (``terminal_cwd``)."""
        mind = self.settings.mind()
        floor = mind.get("floor") if isinstance(mind.get("floor"), Mapping) else {}
        declared, actions = (floor.get("tools") if isinstance(floor.get("tools"), Mapping) else {},
                             mind.get("actions") if isinstance(mind.get("actions"), Mapping) else {})
        return effect_of(tool, dict(args), cwd, context=dict(context), money_hosts=floor.get("money_hosts") or None,
                         declared={str(name): dict(spec) if isinstance(spec, Mapping) else spec
                                   for name, spec in declared.items()},
                         actions={str(name): dict(spec) if isinstance(spec, Mapping) else spec
                                  for name, spec in actions.items()})

    def _mind_cron(self, session_id: str, task_id: str = "") -> bool:
        """A cron run whose job a mind run made is a mind run: its effects are classified and it has no workspace,
        so it writes no file. So is a cron run whose job cannot be told apart yet (the sidecar cannot say, or the
        run names no job): gated until it can. Any other cron run, and every other session, is not."""
        return self._cron_origin(session_id, task_id) == "mind"

    def _cron_origin(self, session_id: str, task_id: str = "") -> str:
        """'' outside cron; otherwise use a saved creation record or positive mind provenance."""
        job = self._cron_job(session_id, task_id)
        if not job and not cron_context():
            return ""
        key = f"{session_id}\n{task_id}"
        if key in self._cron_origins:
            return self._cron_origins[key][0]
        if not job:
            return "unknown"
        inherited = self.sessions.standings.get("cron", job)
        if inherited is not None:
            return inherited[0]
        routes = self.client.has_mind_routes()
        if routes is not True:
            return "unknown"
        stored = self._stored_job(job)
        if not stored:
            return "unknown"
        try:
            response = self.client.get(CRON_ORIGIN_ROUTE, timeout=GUARD_TIMEOUT, params={
                "job_id": str(stored.get("id") or job), "created_at": str(stored.get("created_at") or "")})
            value = response.json() if response.is_success else None
        except (SidecarUnavailable, ValueError):
            return "unknown"
        origin = str(value.get("origin") or "") if isinstance(value, Mapping) else ""
        # The legacy route's owner means only "no mind mark". Only its
        # positively retained mind record is evidence; owner needs our store.
        if origin != "mind":
            return "unknown"
        maker = str(value.get("task_id") or "") if origin == "mind" else ""
        if len(self._cron_origins) >= 256:
            self._cron_origins.pop(next(iter(self._cron_origins)))
        self._cron_origins[key] = (origin, maker)
        return origin

    def _cron_job(self, session_id: str, task_id: str = "") -> str | None:
        """The cron job a call runs for: from its task id (``cron:<job>:<run>``, kept through compression), else
        from its session or the nearest ancestor session named ``cron_<job>_<time>`` (a delegated child). '' for
        none; None when an ancestor cannot be read (not kept: the next call reads it again)."""
        match = CRON_TASK.match(str(task_id or ""))
        if match is not None:
            return match.group("job")
        if session_id in self._cron_jobs:
            return self._cron_jobs[session_id]
        job, current, seen = "", str(session_id or ""), set()
        while current and current not in seen and len(seen) < 8:
            seen.add(current)
            found = CRON_SESSION.match(current)
            if found is not None:
                job = found.group("job")
                break
            current = self._parent_session(current)
            if current is None:
                return None
        if len(self._cron_jobs) >= 1024:
            self._cron_jobs.pop(next(iter(self._cron_jobs)))
        self._cron_jobs[session_id] = job
        return job

    def _parent_session(self, session_id: str) -> str | None:
        """The session a session was delegated or compressed from (``SessionMap.parent``)."""
        try:
            return self.sessions.parent(session_id)
        except Exception as error:
            ancestry_unreadable(logger, error)
            return None

    def _log_mind_decision(self, tool: str, run: str, platform: str, session_id: str,
                           task_id: str, hand: bool) -> None:
        """Log automation authority with fixed sources and reason codes only."""
        if os.environ.get("HERMES_KANBAN_TASK"):
            source = "kanban_worker"
        elif platform == "cron" or CRON_SESSION.match(session_id) or CRON_TASK.match(task_id) or cron_context():
            source = "cron"
        elif run == "mind":
            source = "hand" if hand else "inherited"
        else:
            return
        origin = run if run in {"mind", "owner", "guest"} else "unknown"
        reason = "mind_run" if run == "mind" else "standing_unknown" if origin == "unknown" else "session_standing"
        logger.info("mind_run_decision tool=%s source=%s origin=%s reason=%s", tool, source, origin, reason)

    def _task_id(self, session_id: str, task_id: str = "") -> str:
        """The kanban task of a mind run: the worker's own, or for a mind-origin cron run the task that made it."""
        return os.environ.get("HERMES_KANBAN_TASK") or \
            self._cron_origins.get(f"{session_id}\n{task_id}", ("", ""))[1]

    def _report(self, tool: str, args: Mapping[str, Any], result: Any, session_id: str, task_id: str,
                tool_call_id: str) -> None:
        action = str(args.get("action") or "").strip().lower()
        if not ((tool == "cronjob_manage" and action in {"create", "update"}) or tool == "kanban_create"):
            return
        # Record every successful creator's standing, including owner and guest
        # opt-ins. Delivery origin alone never establishes the creator's standing.
        value = result
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                value = {}
        value = value if isinstance(value, Mapping) else {}
        key = self._creator_key(session_id, task_id, tool, tool_call_id)
        run = "unknown"
        if tool == "kanban_create" and value.get("ok", True) is not False:
            run, _ = self.sessions.standings.complete(key, "kanban", str(value.get("task_id") or ""), session_id)
        elif tool == "cronjob_manage" and value.get("success") is True:
            nested = value.get("job") if isinstance(value.get("job"), Mapping) else {}
            job = str(value.get("job_id") or nested.get("id") or "")
            if action == "update" and not job:
                job = str(self._stored_job(args.get("job_id")).get("id") or "")
            run, _ = self.sessions.standings.complete(key, "cron", job, session_id)
        else:
            creator = self.sessions.standings.pop("creation", key)
            run = creator[0] if creator else "unknown"
        # Reporting provenance is separate from publishing creation authority.
        # Older/missing pre-call hooks still report a positively recorded mind
        # worker's results, while complete() above keeps missing call evidence
        # unknown. Require a positive immutable session snapshot; never infer
        # session admission from the task or consult a later group sender.
        worker_task = os.environ.get("HERMES_KANBAN_TASK")
        report_mind = run == "mind" or (
            run == "unknown" and bool(worker_task) and worker_task == task_id and mind_run(self.settings)
            and (self.sessions.creation_standing(session_id) or ("", ""))[0] in {"owner", "mind"})
        if not report_mind:
            return
        value = result
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                value = {}
        value = value if isinstance(value, Mapping) else {}
        kanban = self._task_id(session_id, task_id)
        if tool == "kanban_create":
            child = str(value.get("task_id") or "")
            if child and kanban and value.get("ok", True) is not False:
                self.client.post(LINEAGE_ROUTE, timeout=GUARD_TIMEOUT, json={"child": child, "parent": kanban})
            return
        ok = value.get("success") is True
        nested = value.get("job") if isinstance(value.get("job"), Mapping) else {}
        job = str(value.get("job_id") or nested.get("id") or "") if ok else ""
        if ok and not job and action == "update":
            job = str(self._stored_job(args.get("job_id")).get("id") or "")
        self.client.post(CRON_ORIGIN_ROUTE, timeout=GUARD_TIMEOUT, json={
            "job_id": job, "task_id": kanban, "tool_call_id": tool_call_id, "action": action, "ok": bool(ok and job)})

    # -- rules ----------------------------------------------------------------

    def _deny(self, tool: str, text: str) -> dict[str, str] | None:
        deny = self.settings.mind().get("deny")
        deny = deny if isinstance(deny, Mapping) else {}
        if tool in {str(name) for name in (deny.get("tools") or [])}:
            return block(f"{tool} is on the deny list")
        for pattern in deny.get("text") or []:
            try:
                if re.search(str(pattern), text, re.IGNORECASE):
                    return block("the arguments match a denied text pattern")
            except re.error:
                logger.warning("ignoring invalid deny pattern %r", pattern)
        return None

    def mind_enabled(self) -> bool:
        if self.settings.mind().get("enabled") is False:
            return False
        cached = self._enabled_cache
        if cached is not None and time.monotonic() - cached[0] < 30:
            return cached[1]
        enabled = True
        try:
            if self.client.has_mind_routes():
                response = self.client.get(MIND_STATE_ROUTE, timeout=GUARD_TIMEOUT)
                if response.is_success and response.json().get("enabled") is False:
                    enabled = False
        except (SidecarUnavailable, ValueError):
            pass  # unreachable: the sidecar verdict below blocks effects anyway
        self._enabled_cache = (time.monotonic(), enabled)
        return enabled

    def _outbound(self, tool: str, args: Mapping[str, Any], session_id: str,
                  cron_action: str = "", recipients: list[str] | None = None) -> dict[str, str] | None:
        """Messaging tools and delivering cron jobs need a permitted recipient."""
        if tool in MESSAGING_TOOLS:
            if self.client.has_mind_routes():
                return None  # the sidecar verdict decides with may_contact and budgets
            return block(f"{tool} needs the sidecar's contact permissions, which this version lacks")
        # The job delivers to ``deliver`` and, on failure, ``failure_deliver``; an update
        # keeps whatever the stored job has for the fields it does not supply.
        stored = self._stored_job(args.get("job_id")) if cron_action == "update" else {}
        fields = {"deliver": [("origin", "")] if cron_action == "create" else [], "failure_deliver": []}
        targets: list[tuple[str, str]] = []
        for field, default in fields.items():
            if field in args:
                targets += parse_deliver(args.get(field))
            elif field in stored:
                targets += parse_deliver(stored.get(field))
            else:
                targets += default
        for platform, chat_id in targets:
            if platform == "local":
                continue
            if platform == "*":
                return block("cron delivery to every channel is not allowed in this run")
            if platform == "origin":
                contact = self._session_contact(session_id)
            else:
                contact = self.client.resolve_contact(platform, chat_id, timeout=GUARD_TIMEOUT)
            if not contact:
                return block("cron delivery to an unknown recipient")
            if contact.get("may_contact") == "never":
                return block("cron delivery to a contact who may not be contacted")
            if recipients is not None and contact.get("contact_id") and contact["contact_id"] not in recipients:
                recipients.append(str(contact["contact_id"]))
        return None

    @staticmethod
    def _stored_job(job_id: Any) -> dict[str, Any]:
        """The stock cron record behind ``job_id`` (id or name), or {} when there is none."""
        if not job_id:
            return {}
        try:
            from cron import jobs
            job = jobs.resolve_job_ref(str(job_id))
        except Exception:  # outside Hermes, or an unreadable store: nothing to merge
            return {}
        return dict(job) if isinstance(job, Mapping) else {}

    def _guest_message(self, args: Mapping[str, Any], session_id: str) -> str | None:
        """A contact's session messaging ``platform:address``: the session's own chat (in a direct chat the
        sender's handle too) is the reply, a target no contact is known at is refused as the verdict would;
        anything else, including a call with no address or a sidecar that cannot answer, is the verdict's."""
        target = str(args.get("target") or args.get("chat_id") or args.get("to") or "")
        platform, address = (parse_deliver(f"{args['platform']}:{target}" if args.get("platform") else target)
                             or [("", "")])[0]
        info = self.sessions.get(session_id)
        if args.get("contact_id") or info is None or not address or platform == "*":
            return None
        chat, chat_type = self.sessions.chat(session_id)
        if platform.lower() == info.platform.lower() and (
                address == chat or ((not chat_type or chat_type in DM_CHAT_TYPES) and address == info.sender_id)):
            return REPLY
        try:
            contact = self.client.resolve_contact(platform, address, timeout=GUARD_TIMEOUT)
        except Exception:
            return None
        return UNKNOWN_RECIPIENT if contact is None else None

    def _session_contact(self, session_id: str) -> dict[str, Any] | None:
        if self.sessions.is_owner(session_id):
            return {"contact_id": self.settings.owner_contact_id() or "owner", "may_contact": "auto"}
        info = self.sessions.get(session_id)
        if info is None or not info.sender_id:
            return None
        return self.client.resolve_contact(info.platform, info.sender_id, timeout=GUARD_TIMEOUT)

    def _sidecar_verdict(self, tool: str, args: Mapping[str, Any], session_id: str,
                         run: str, recipients: list[str] | None = None, *, effect: Effect | None = None,
                         context: Mapping[str, Any] | None = None, cwd: str | None = None, task_id: str = "",
                         tool_call_id: str = "") -> dict[str, str] | None:
        """``POST /v1/mind/guard {tool, args, session, recipients, ...}`` → ``{allow, reason}``; 2 s timeout."""
        info = self.sessions.get(session_id) if run == "guest" else None  # a worker session has no sender
        payload: dict[str, Any] = {
            "tool": tool, "args": dict(args), "session": session_id, "run": run,
            "task_id": self._task_id(session_id, task_id) if run == "mind"
            else os.environ.get("HERMES_KANBAN_TASK") or "",
            "recipients": list(recipients or []),
            "owner": self.sessions.is_owner(session_id) if info is not None else None,
            "contact_id": self.sessions.contact_id(session_id) if info is not None else None,
            "platform": info.platform if info is not None else "", "sender_id": info.sender_id if info is not None else ""}
        if run == "mind":
            payload.update({"cwd": cwd, "tool_call_id": tool_call_id, "context": dict(context or {}),
                            "effect": effect.as_dict() if effect is not None else None})
        response = self.client.post(GUARD_ROUTE, timeout=GUARD_TIMEOUT, json=payload)
        if response.status_code == 404:
            return None
        if not response.is_success:
            return block(f"the sidecar refused the guard check (HTTP {response.status_code})")
        verdict = response.json()
        if not isinstance(verdict, Mapping):
            return block("the sidecar answered the guard check with no verdict")
        reason = str(verdict.get("reason") or verdict.get("message") or "")
        if verdict.get("ask") is True or verdict.get("action") == "ask":
            return block(reason or f"{tool} needs the owner's yes")
        if "allow" in verdict:
            if verdict["allow"] is not True:
                return block(reason or f"{tool} was refused by the mind")
            if run == "mind" and tool == ACTION_TOOL and effect is not None and \
                    (verdict.get("effect") if isinstance(verdict.get("effect"), Mapping) else {}).get("digest") \
                    != effect.digest:
                # The plugin runs the declaration it reads now; the sidecar decided on the one it loaded.
                return block(f"{effect.describe()} waits for the owner: the sidecar read this action's declaration "
                             "otherwise (it changed since the sidecar started); stop and report that it waits")
            if run == "mind" and read_as(tool, args) == "browser_vault_fill" \
                    and (verdict.get("permit") or verdict.get("grant")):
                consent_payment_fill(args.get("handle"), task_id)
            return None
        if verdict.get("action") == "block":
            return block(reason or f"{tool} was refused by the mind")
        return None


__all__ = ["ACTION_TOOL", "Effect", "FLOOR_PATTERNS", "GUEST_NEVER", "GUEST_TOOLS", "Guard", "MESSAGING_TOOLS",
           "PROTECTED_BASENAMES", "READ_ONLY_TOOLS", "WRITE_TOOLS", "action_argv", "action_problem", "action_values",
           "block", "consent_payment_fill", "cron_context", "effect_of", "floor_match", "forgery_signals",
           "guest_allowed", "guest_file", "guest_files_ok", "mind_run", "names_protected_file", "parse_deliver",
           "path_inside", "reaches_key", "terminal_cwd", "vault_kind", "vault_meta", "workspace", "write_targets"]
