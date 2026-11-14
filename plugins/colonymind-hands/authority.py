"""Owner turns run floor effects unless a hand of that session holds the work.

Shell policy is best effort. An unrecognized command remains part of the trusted
lane. An owner message does not transfer a hand's approval to the front.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import uuid

from .budgets import TERMINAL, WAITING, block, is_child_context, refuse, resolve_hand

logger = logging.getLogger(__name__)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def call_digest(call: dict, workdir: str) -> str:
    return hashlib.sha256(canonical({"call": call, "workdir": workdir}).encode()).hexdigest()


def exact_call(tool: str, args: dict, workdir) -> dict:
    """The call an approval binds: a terminal `workdir` argument that only restates the effective working
    directory is left out, so one yes covers the same command whether or not the hand spells the directory out."""
    arguments = dict(args or {})
    if tool == "terminal" and "workdir" in arguments and workdir is not None:
        value = arguments.get("workdir")
        try:
            same = not value or Path(str(value)).expanduser().resolve() == Path(str(workdir)).resolve()
        except (OSError, RuntimeError, ValueError):
            same = False
        if same:
            arguments.pop("workdir")
    return {"tool": tool, "arguments": arguments}


def _guard():
    # The installed adapter carries a byte-identical copy of mind.authority's
    # effect table. The plugin never imports the sidecar into Hermes.
    from colonymind_hermes import guard
    return guard


def _workdir(tool, args, runtime, task_id) -> Path | None:
    reader = getattr(runtime, "workdir_for", None)
    if callable(reader):
        value = reader(tool, args, task_id)
    elif tool == "terminal":
        value = _guard().terminal_cwd(args, task_id)
    elif tool == "execute_code":
        value = None  # An executor with no local directory proof cannot bind local files.
    else:
        value = str(Path.cwd())
    if not value or not Path(value).is_absolute():
        return None
    return Path(value).expanduser().resolve()


def _path(value, workdir) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workdir / path).resolve()


def _words(command):
    try:
        return shlex.split(command)
    except ValueError:
        return []


class UnboundDeletion(ValueError):
    """A recognized deletion whose targets are not the literal operands of a standalone delete command."""


SIMPLE_DELETES = {"rm", "rmdir", "un" + "link"}
UNBOUND_DELETION = ("a deletion can wait for the owner's yes only as its own terminal command: run the delete alone "
                    "with literal paths, with no sudo, ;, &&, |, redirections, echo, variables or globs, and do any "
                    "checks in separate calls")
UNHOLDABLE_DELETION = ("this deletion cannot be held for the owner's yes in this form; do not retry it another way. "
                       "Stop and report exactly what should be deleted and why, so the owner can decide")


def unbound_deletion_message(tool, args):
    """Advice for a refused deletion: rerun it standalone only when that form can be held."""
    words = _words(str(args.get("command") or "")) if tool == "terminal" else []
    if words and Path(words[0]).name in SIMPLE_DELETES:
        return UNBOUND_DELETION
    return UNHOLDABLE_DELETION


def _deletion_paths(tool, args):
    """An exemption only for a simple rm/unlink/rmdir with literal operands."""
    if tool != "terminal":
        return None
    command = str(args.get("command") or "")
    if re.search(r"[;&|\n`$*?\[\]{}<>]", command):
        return None
    words = _words(command)
    if not words or Path(words[0]).name not in {"rm", "unlink", "rmdir"}:
        return None
    result, operands = [], False
    for word in words[1:]:
        if word == "--" and not operands:
            operands = True
        elif operands or not word.startswith("-"):
            result.append(word)
        elif word.startswith("--") and word.split("=", 1)[0] not in {"--force", "--recursive", "--verbose",
                                                                       "--dir", "--interactive", "--one-file-system"}:
            return None
    return result or None


def _destination(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return value
    if parsed.username or parsed.password:
        raise ValueError("credential-bearing destinations cannot enter an ask")
    host = parsed.hostname.lower().rstrip(".")
    port = parsed.port
    if port and (parsed.scheme.lower(), port) not in {("https", 443), ("http", 80)}:
        host += f":{port}"
    return urlunsplit((parsed.scheme.lower(), host, parsed.path or "/", parsed.query, ""))


def _urls(args):
    # Structured URLs and literal command URLs. Values are destinations, never
    # an assertion in a hand's final answer or brief.
    return [_destination(value.rstrip(".,)")) for value in
            re.findall(r"(?:https?|ssh)://[^\s\"'<>]+", canonical(args))]


def _configured_class(args, config):
    urls = _urls(args)
    text = canonical(args)
    for name, values in (("publishing", config.get("publish_targets", [])),
                         ("owner_account", config.get("owner_account_hosts", []))):
        for configured in values:
            configured = str(configured).strip()
            if not configured:
                continue
            if "://" in configured:
                root = _destination(configured).rstrip("/")
                if any(value == root or value.startswith(root + "/") or value.startswith(root + "?") for value in urls):
                    return name
            elif any(urlsplit(value).hostname == configured.lower().rstrip(".") for value in urls):
                return name
            elif re.search(r"(?<![\w./-])" + re.escape(configured) + r"(?![\w./-])", text):
                return name
    return None


def floor_class(tool, args, workdir, hand, runtime):
    configured = runtime.config.get("floor", {})
    cls = _configured_class(args, configured)
    push = _push(tool, args, workdir) if configured.get("publish_targets") or configured.get("owner_account_hosts") else None
    if push:
        repo, remote, _, _ = push
        destination = _remote(repo, remote)
        cls = cls or _configured_class({"destination": destination}, runtime.config.get("floor", {}))
    if cls:
        return cls
    effect = _guard().effect_of(tool, args, str(workdir))
    if effect is None or not effect.floor:
        return None
    if effect.cls == "irreversible_deletion" and hand:
        operands = _deletion_paths(tool, args)
        workspace = hand.plan.get("workspace") or hand.plan.get("workdir") or hand.plan.get("cwd")
        if operands and workspace and workdir is not None:
            root = Path(workspace).expanduser().resolve()
            if all(_path(value, workdir).is_relative_to(root) and _path(value, workdir) != root for value in operands):
                return None
    return effect.cls


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "absent"
    if path.is_file():
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    if not path.is_dir():
        raise ValueError("cannot bind a special file")
    rows = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            rows.append((str(child.relative_to(path)), "symlink", str(child.readlink())))
        elif child.is_file():
            rows.append((str(child.relative_to(path)), "file", _file_hash(child)))
        elif child.is_dir():
            rows.append((str(child.relative_to(path)), "directory"))
        else:
            raise ValueError("cannot bind a directory containing special files")
    return hashlib.sha256(canonical(rows).encode()).hexdigest()


def _git(workdir, *args):
    # Assumption: a local read of git metadata gets five seconds. No shell or
    # configured probe command executes while collecting approval bindings.
    result = subprocess.run(["git", "-C", str(workdir), *args], capture_output=True, text=True, timeout=5,
                            check=True)
    return result.stdout.strip()


def _push(tool, args, workdir):
    words = _words(str(args.get("command") or "")) if tool == "terminal" else []
    if not words or Path(words[0]).name != "git" or "push" not in words:
        return None
    index, repo = 1, workdir
    while index < len(words) and words[index] == "-C":
        repo = _path(words[index + 1], repo)
        index += 2
    if words[index] != "push":
        raise ValueError("cannot bind push with unsupported git options")
    # A push option taking a value changes the destination or remote state;
    # do not guess which argument is its value.
    supported = {"--force", "-f", "--force-with-lease", "--all", "--mirror", "--tags", "--delete",
                 "-u", "--set-upstream", "--atomic", "--follow-tags", "--dry-run", "-n", "--verbose", "-v"}
    if any(word.startswith("-") and word not in supported for word in words[index + 1:]):
        raise ValueError("cannot bind push with unsupported push options")
    operands = [word for word in words[index + 1:] if not word.startswith("-")]
    return repo, operands[0] if operands else "origin", operands[1:] or ["HEAD"], words


def _remote(workdir, remote):
    value = remote if ":" in remote or "/" in remote else _git(workdir, "remote", "get-url", "--push", remote)
    return _destination(value)


def _move_paths(tool, args):
    """Literal mv operands for comparison when the effect table recognizes it.

    This does not broaden either floor recognition or a hand's deletion exemption.
    """
    command = str(args.get("command") or "")
    if tool != "terminal" or re.search(r"[;&|\n`$*?\[\]{}<>]", command):
        return None
    words = _words(command)
    if not words or Path(words[0]).name != "mv":
        return None
    paths, operands = [], False
    for word in words[1:]:
        if word == "--" and not operands:
            operands = True
        elif operands or not word.startswith("-"):
            paths.append(word)
        elif word.startswith("--target-directory="):
            paths.append(word.split("=", 1)[1])
        elif word.startswith("-t") and len(word) > 2:
            paths.append(word[2:])
    return paths or None


def bound_targets(tool, args, workdir, hand, runtime, *, identities_only=False):
    values = {}
    def add(kind, target, value):
        values[(kind, target)] = {"kind": kind, "target": target, "value": value}
    for url in _urls(args):
        add("destination", url, url)
    if _guard().effect_of(tool, args, str(workdir)) is not None:
        effect = _guard().effect_of(tool, args, str(workdir))
        if (effect.cls == "irreversible_deletion" and tool in {"terminal", "execute_code"}
                and not _deletion_paths(tool, args)
                and not (identities_only and _move_paths(tool, args))):
            raise UnboundDeletion("recognized deletion needs literal target paths before it can be approved")
    paths = list(_deletion_paths(tool, args) or [])
    if identities_only:
        paths.extend(_move_paths(tool, args) or [])
    for name in ("path", "file", "file_path", "source", "input_path", "output_path"):
        if isinstance(args.get(name), str):
            paths.append(args[name])
    words = _words(str(args.get("command") or "")) if tool == "terminal" else []
    for index, word in enumerate(words):
        if word.startswith("@") and len(word) > 1:
            paths.append(word[1:])
        if word in {"--upload-file", "-T", "--file", "-F"} and index + 1 < len(words):
            value = words[index + 1]
            paths.append(value.split("=@", 1)[-1] if "=@" in value else value)
    for value in paths:
        path = _path(value, workdir)
        add("file", str(path), "" if identities_only else _file_hash(path))
    push = _push(tool, args, workdir)
    if push:
        repo, remote, refs, words = push
        if any(word.startswith(("--all", "--mirror", "--tags", "--delete")) for word in words):
            # Bind the whole ref set for pushes without a single source ref.
            add("git_ref", str(repo) + "#all", "" if identities_only else
                hashlib.sha256(_git(repo, "show-ref").encode()).hexdigest())
        else:
            for ref in refs:
                source = ref.lstrip("+").split(":", 1)[0] or "HEAD"
                commit = "" if identities_only else _git(
                    repo, "rev-parse", "--verify", "--end-of-options", source + "^{commit}")
                add("git_ref", str(repo) + "#" + source, commit)
        destination = _remote(repo, remote)
        add("destination", destination, destination)
    for spec in (hand.plan.get("floor_targets", []) if hand else []):
        kind, target = spec["kind"], spec["target"]
        if kind == "file":
            path = _path(target, workdir)
            add(kind, str(path), _file_hash(path))
        elif kind in {"probe", "service"}:
            probe = getattr(runtime, "probe_target", None)
            if not callable(probe):
                raise ValueError("target probe is unavailable")
            result = probe(kind, target)
            add(kind, target, hashlib.sha256(result if isinstance(result, bytes) else str(result).encode()).hexdigest())
        elif kind == "destination":
            value = _destination(target)
            add(kind, value, value)
        elif kind == "git_ref":
            add(kind, target, _git(workdir, "rev-parse", "--verify", "--end-of-options", target + "^{commit}"))
    return [values[key] for key in sorted(values)]


def owner_started_turn(runtime, session_id, turn_id="") -> bool:
    """Trust ingress provenance, retaining the owner-session fallback without a ledger."""
    reader = getattr(sys.modules.get("colonymind_hands_front"), "owner_started_turn", None)
    if callable(reader):
        return reader(session_id, turn_id) is True
    reader = getattr(getattr(runtime, "ingress", None), "owner_started_turn", None)
    return reader(session_id) is True if callable(reader) else True


def _approved_call(hand, tool, args, workdir) -> bool:
    """Whether this is the exact call an owner approved for this continuation.

    That call always goes through the permit check, which consumes its one-use
    permit, even where a workspace exemption would otherwise let it run.
    """
    approved = hand.plan.get("approved_call") if hand is not None else None
    if not isinstance(approved, dict) or workdir is None:
        return False
    granted = approved.get("call") if isinstance(approved.get("call"), dict) else {}
    return call_digest(exact_call(tool, args, workdir), str(workdir)) == call_digest(
        exact_call(granted.get("tool"), granted.get("arguments"), approved.get("workdir")), str(approved.get("workdir")))


def _continue_held(hand_id, reason):
    return block(f"{reason}; the runtime resumes holding hand {hand_id} automatically after the owner's yes. "
                 "Do not continue it again or perform the action yourself")


def _owner_front_call(runtime, session_id, tool, args, task_id):
    # Recognize locally first: most owner calls are no floor effect, and those
    # never wait on the controller (each read may take its full timeout).
    recognized, workdir = True, None
    try:
        workdir = _workdir(tool, args, runtime, task_id)
        recognized = bool(floor_class(tool, args, workdir, None, runtime))
    except Exception:
        logger.warning("floor recognition unavailable in an owner turn", exc_info=True)
    if not recognized:
        return None
    # This read never checks or consumes authority. The controller owns durable
    # permit state, including after the gateway loses its in-memory registry.
    waiting = [h.hand_id for h in runtime.registry.all()
               if h.parent_session_id == session_id and h.activity == "awaiting_approval"]
    unavailable = False
    try:
        rows = runtime.client.get("/v1/hands/permits/held", {"session": session_id})["permits"]
        if not isinstance(rows, list):
            raise ValueError("invalid held permit records")
        held = [row for row in rows if row["parent_session"] == session_id
                and row["state"] in {"waiting", "granted"}]
    except Exception:
        logger.warning("held targets unavailable for owner session", exc_info=True)
        held, unavailable = [], True
    if not held and (not unavailable or not waiting):
        # Retain the owner exemption even if workdir or binding collection would
        # fail. No held work means no reason to add an approval boundary.
        return None
    waiting.extend(row["hand_id"] for row in held if row["state"] == "waiting")
    try:
        if unavailable:
            return _continue_held(waiting[0],
                "held targets unavailable while a hand of this session awaits approval")
        digest = call_digest(exact_call(tool, args, workdir), str(workdir))
        for row in held:
            if row.get("call_digest") == digest:
                return _continue_held(row["hand_id"], "this call is held by a hand of this session")
        if workdir is None:
            raise ValueError("local working directory unavailable")
        targets = {(t["kind"], t["target"]) for t in bound_targets(
            tool, args, workdir, None, runtime, identities_only=True)}
        for row in held:
            if targets.intersection((t["kind"], t["target"]) for t in row["targets"]):
                return _continue_held(row["hand_id"], "this target is held by a hand of this session")
    except Exception:
        logger.warning("held target comparison unavailable", exc_info=True)
        if waiting:
            return _continue_held(waiting[0],
                "held target comparison unavailable while a hand of this session awaits approval")
    return None


def pre_tool_call(runtime: Any, tool_name: str = "", args=None, session_id: str = "", **ids):
    hand = None
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        if hand is None and is_child_context(**ids):
            return block("unbound hand session: authority check refused")
        args = args if isinstance(args, dict) else {}
        if hand is not None and ('definition_policy' in hand.plan or 'definition_snapshot' in hand.plan
                                 or hand.plan.get('task', {}).get('hand')):
            from .allow import check_tool
            reason = check_tool(hand, tool_name, args)
            if reason:
                return refuse(hand, reason, ids.get("tool_call_id", ""))
        if hand is None and owner_started_turn(runtime, session_id, ids.get("turn_id", "")):
            return _owner_front_call(runtime, session_id, tool_name, args, ids.get("task_id", ""))
        workdir = _workdir(tool_name, args, runtime, ids.get("task_id") or (hand.subagent_id if hand else ""))
        if not _approved_call(hand, tool_name, args, workdir) and not floor_class(tool_name, args, workdir, hand, runtime):
            return None
        if hand is None:
            # Owner answers grant the held call to its hand's continuation, never
            # to the front or a review borrowing the latest hand in the session.
            return block("recognized floor effect must run in a hand; the runtime resumes a holding hand "
                         "automatically after approval. Do not continue it again or perform the action yourself. "
                         "Delegate a new effect to a hand for its own hold "
                         "and owner approval; do not execute it in the front")
        if (hand.blocked or hand.stop_reason or hand.status in TERMINAL or hand.activity in WAITING
                or ids.get("tool_call_id", "") in hand.refused_calls):
            # Hermes runs every hook before applying a veto. Do not spend a
            # permit on a call an earlier hook has already refused.
            return refuse(hand, "hand cannot execute this call; the runtime resumes it automatically after approval",
                          ids.get("tool_call_id", ""))
        if workdir is None:
            return refuse(hand, "recognized floor effect needs a known local working directory before approval",
                          ids.get("tool_call_id", ""))
        call = exact_call(tool_name, args, workdir)
        try:
            targets = bound_targets(tool_name, args, workdir, hand, runtime)
        except UnboundDeletion:
            # A policy refusal the hand can act on, not an outage: it reruns the deletion alone, which is then held.
            return refuse(hand, unbound_deletion_message(tool_name, args), ids.get("tool_call_id", ""))
        nonce = uuid.uuid4().hex
        # ControllerClient.get has no write-key parameter. Consumption uses the
        # existing lower-level transport so the frozen GET carries its write key.
        checked = runtime.client._request("GET", "/v1/hands/permits/check", key="permit:" + nonce,
            params={"hand": hand.hand_id, "call_digest": call_digest(call, str(workdir)),
                    "targets_now": canonical(targets)})
        if checked.get("decision") == "allow":
            return None
        body = {"epoch": hand.epoch or runtime.epoch, "call": call, "workdir": str(workdir), "targets": targets}
        runtime.emit(f"/v1/hands/{hand.hand_id}/hold", body, "hold:" + nonce)
        with hand.lock:
            hand.activity = "awaiting_approval"
            hand.blocked = True
        return refuse(hand, "held for the owner's yes to this exact call and its targets; stop and report that it waits",
                      ids.get("tool_call_id", ""))
    except Exception:
        logger.exception("hand authority check unavailable")
        return refuse(hand, "authority check unavailable: effect refused", ids.get("tool_call_id", ""))


def register(ctx: Any) -> None:
    from .dispatch import get_runtime
    runtime = get_runtime(ctx)
    if runtime is None:
        return
    def standing(sid, task=""):
        hand = runtime.registry.for_session(sid, task_id=task)
        return hand.plan.get("standing", "unknown") if hand is not None else "unknown"
    _guard().register_hand_sessions(
        lambda sid, task="": runtime.registry.for_session(sid, task_id=task) is not None, standing)
    ctx.register_hook("pre_tool_call", lambda **kwargs: pre_tool_call(runtime, **kwargs))
