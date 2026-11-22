"""Turn and sender capture: the session map and the durable turn outbox.

Capture callbacks enqueue. ``pre_llm_call`` records who is speaking in each
session (the guard and the tools read that map); ``post_llm_call`` writes one
row to a private SQLite outbox. The body thread delivers rows to the sidecar.
``pre_gateway_dispatch`` (``SessionMap.hear``) records, per message, what the
gateway's own event shows of it: when it arrived, its id, what it replies or
reacts to; native settlement synchronously releases interrupted transport bindings.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from .client import ColonyMindClient, Settings, SidecarUnavailable, read_yaml
from .busy_ingress import turn_platform_id
from colonymind_shared import host_owner
from colonymind_shared.midturn import bind_rewrite, composition, matches_rewrite, member_turn_id, valid_ingress
from colonymind_shared.turn_bindings import TurnBindings
from .work import WorkCapture
from .standing import STANDING_FILE, StandingStore

logger = logging.getLogger(__name__)

# Turns without a sender on these platforms belong to the owner's own terminal.
INTERNAL_PLATFORMS = frozenset({"", "cli", "internal", "system", "owner", "api", "worker", "cron"})
# The sidecar's refusals of a turn its credential may not write (docs/CREDENTIALS.md): a failed capture.
ATTRIBUTION_REFUSALS = frozenset({"owner_not_granted", "contact_not_granted", "sender_not_granted"})
REFUSED_PREFIX = "attribution refused: "
# The skills the sidecar writes into its own skills.external_dirs entry (P/mind/skills.py).
SKILL_PREFIX = "colonymind-"
# The chat types of a sender's direct chat with the agent.
DM_CHAT_TYPES = frozenset({"dm", "direct", "private"})


# Host recall observations stay in memory; Body passes drain records and queue alarms.
ORDER = ("no_turn_id", "not_owner", "runtime_unavailable", "circuit_open", "review",
         "binding_refused", "error", "cancelled", "deadline", "no_candidates", "host_drop", "ok")
STAGES = {"embed", "search", "owner_search", "rerank", "assembly_left", "recall"}
DROPS = {"prefetch_wait", "context_fit", "late_callback"}
_lock = threading.RLock()
# Serializes drains so emit order and streak state belong to one drainer; response hooks never take it.
_drain_lock = threading.Lock()
_turns = OrderedDict()
_pending = deque()
_streak = 0
_codes = set()
_alarmed = False
_threshold = 2
_alarm_target = None
_alarm_notice = None
_debug_pending = False
_debug_at = float("-inf")


def safe(call):
    @wraps(call)
    def guarded(*args, **kwargs):
        global _debug_pending
        try:
            return call(*args, **kwargs)
        except BaseException:
            _debug_pending = True
    return guarded


@dataclass
class Observation:
    turn_id: str = ""
    owner: bool = False
    excluded: bool = False
    exit: str = "ok"
    stage: str | None = None
    subcode: str | None = None
    exception: str | None = None
    capture: str = "none"
    channel: str = "other"
    recall_ms: int = 0
    budget_ms: int = 0
    rerank: bool = False
    sections_returned: int = 0
    sections_submitted: int = 0
    chars_submitted: int = 0
    existing_vs_new: str = "none"
    drop: str | None = None
    requested: bool = False
    finished: bool = False
    emitted: bool = False
    prefetch_running: bool = False
    prefetch_deadline: float = 0.0


@safe
def configure_from_settings(settings):
    from .client import read_yaml
    config = read_yaml(settings.home / "colonymind.yaml").get("context") or {}
    configure(config.get("recall_alarm_turns", 2), config.get("recall_alarm_target"))


@safe
def configure(threshold=2, target=None):
    global _threshold, _alarm_target
    with _lock:
        _threshold = max(1, int(threshold))
        _alarm_target = target.strip() if isinstance(target, str) and target.strip() else None


@safe
def begin(session_id, turn_id, *, owner=False, platform="", excluded=False):
    with _lock:
        key = (session_id, turn_id)
        if key not in _turns:
            _turns[key] = Observation(turn_id=turn_id, owner=owner, excluded=excluded,
                channel=platform if platform in {"whatsapp", "rcs", "sms"} else "other")
        # Bounded in-flight metadata; never retain content or a session id in a record.
        while len(_turns) > 256:
            _turns.popitem(last=False)
        return _turns[key]


@safe
def prefetch_start(session_id, cached_turn_id, wait):
    with _lock:
        state = _turns.get((session_id, cached_turn_id))
        if state is None:
            state = next((value for (sid, _), value in reversed(_turns.items()) if sid == session_id), None)
        if state is not None:
            state.prefetch_running = True
            state.prefetch_deadline = time.monotonic() + wait
            if not cached_turn_id:
                update(state, exit="no_turn_id")
        return state


@safe
def prefetch_done(state):
    if state is not None:
        with _lock:
            if not state.emitted:
                state.prefetch_running = False


@safe
def update(state, **facts):
    if state is None:
        return
    with _lock:
        if state.emitted:
            return
        value = facts.get("exit")
        if value in ORDER and ORDER.index(value) < ORDER.index(state.exit):
            state.exit = value
            state.stage = facts.get("stage") if facts.get("stage") in STAGES else None
            state.subcode = "identity" if facts.get("subcode") == "identity" else None
            exception = facts.get("exception")
            # Exception class spelling only; no messages, attrs or arbitrary strings.
            state.exception = exception if isinstance(exception, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}", exception) else None
        for key in ("recall_ms", "budget_ms", "sections_returned"):
            if type(facts.get(key)) is int and facts[key] >= 0:
                setattr(state, key, facts[key])
        if type(facts.get("owner")) is bool:
            state.owner = state.owner or facts["owner"]
        if type(facts.get("rerank")) is bool:
            state.rerank = facts["rerank"]
        if facts.get("capture") in {"none", "pending", "c", "d", "recorded"}:
            state.capture = facts["capture"]
        if (state.requested and not state.sections_submitted and facts.get("sections_returned", 0) > 0):
            state.drop = "late_callback"
        if facts.get("drop") in DROPS and state.drop != "late_callback":
            state.drop = facts["drop"]


# Only provider-owned, fenced memory sections qualify. Request strings are read
# transiently; neither messages, bodies nor hashes are retained in observations.
_SECTION = re.compile(r"(?:^|\n\n)## Relevant Memories\n(.*?)(?=\n\n## |\n\nunchanged since your last turn:|$)", re.S)


def _text_parts(content):
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"} and isinstance(part.get("text"), str):
                yield part["text"]


@safe
def pre_api_request(*, session_id="", turn_id="", request_messages=None, **_):
    with _lock:
        state = _turns.get((session_id, turn_id))
        if state is None or state.excluded or state.emitted:
            return
    messages = request_messages if isinstance(request_messages, list) else []
    last_user = max((i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"), default=-1)
    old = new = submitted = chars = 0
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        for text in _text_parts(message.get("content")):
            offset = 0
            while True:
                start = text.find("<memory-context>", offset)
                if start < 0:
                    break
                start += len("<memory-context>")
                end = text.find("</memory-context>", start)
                if end < 0:
                    break
                offset = end + len("</memory-context>")
                for section in _SECTION.finditer(text[start:end].lstrip()):
                    body = section.group(1)
                    size = len(body[:-1] if body.endswith("\n") else body)
                    if size:
                        submitted += 1
                        chars += size
                        if i == last_user:
                            new += 1
                        else:
                            old += 1
    with _lock:
        if state.emitted:
            return
        # Maxima describe memory actually present in an individual request;
        # retries do not multiply turn counts. A later attempt can supply memory
        # missing from the first one. Keep reuse provenance across attempts.
        state.sections_submitted = max(state.sections_submitted, submitted)
        state.chars_submitted = max(state.chars_submitted, chars)
        old = bool(old) or state.existing_vs_new in {"existing", "both"}
        new = bool(new) or state.existing_vs_new in {"new", "both"}
        state.existing_vs_new = "both" if old and new else "existing" if old else "new" if new else "none"
        state.requested = True


@safe
def post_llm_call(*, session_id="", turn_id="", **_):
    with _lock:
        state = _turns.get((session_id, turn_id))
        if state is None or state.finished:
            return
        state.finished = True
        # A Body pass emits observations; only its loop thread sends alarms.
        _pending.append(state)


@safe
def raise_alarm(streak, codes):
    """Log once per streak and replace the one unsent notice."""
    global _alarm_notice
    logger.info("RECALL_ALARM Owner memory context missing on %d consecutive turns; exits: %s",
                streak, ",".join(sorted(set(codes) & set(ORDER))))
    if _alarm_target is None:
        return
    text = (f"Memory context was missing on {streak} consecutive owner turns - exit codes: "
            + ",".join(sorted(set(codes) & set(ORDER))) + ".")
    with _lock:
        _alarm_notice = text


@safe
def take_alarm_notice():
    """Take the single pending notice for the Body loop, releasing the lock before sending."""
    global _alarm_notice
    with _lock:
        text, _alarm_notice = _alarm_notice, None
        return (_alarm_target, text) if text is not None and _alarm_target is not None else None


def _emit(state):
    global _streak, _alarmed, _alarm_notice
    if not state.owner or state.excluded:
        return
    # Earlier failed gates remain visible even when lexical/history fallback
    # supplies memory. Only turns with no failed gate reach ok/host_drop.
    if state.exit == "ok" and not state.sections_submitted:
        state.exit = "host_drop" if state.sections_returned else "no_candidates"
    optional = "".join(" %s=%s" % (key, value) for key, value in (
        ("stage", state.stage), ("subcode", state.subcode), ("exception", state.exception),
        ("drop", state.drop if state.exit == "host_drop" else None)) if value is not None)
    # JSON-style quoting protects the one-line format while preserving the opaque id.
    import json
    logger.info("RECALL_EXIT exit=%s turn_id=%s channel=%s sections_returned=%d sections_submitted=%d chars_submitted=%d existing_vs_new=%s recall_ms=%d budget_ms=%d rerank=%s capture=%s%s",
        state.exit, json.dumps(state.turn_id, ensure_ascii=True), state.channel, state.sections_returned,
        state.sections_submitted, state.chars_submitted, state.existing_vs_new,
        state.recall_ms, state.budget_ms, "yes" if state.rerank else "no", state.capture, optional)
    # The streak counts turns that reached the model WITHOUT memory; history memory counts as present.
    if state.exit == "ok" or state.sections_submitted:
        _streak, _alarmed = 0, False
        _codes.clear()
        with _lock:
            _alarm_notice = None  # recovered before the loop sent it: the notice is no longer true
    else:
        _streak += 1
        _codes.add(state.exit)
        if _streak >= _threshold and not _alarmed:
            _alarmed = True
            raise_alarm(_streak, _codes)


@safe
def drain():
    """A Body pass emits records and queues alarms; response hooks only enqueue."""
    with _drain_lock:
        _drain()


def _drain():
    global _debug_at, _debug_pending
    while True:
        with _lock:
            if not _pending:
                break
            # A late callback still belongs to this exact turn. Preserve turn
            # order without blocking Body or adding any response-path wait.
            state = _pending[0]
            if state.prefetch_running:
                if time.monotonic() < state.prefetch_deadline:
                    break
                if not state.sections_submitted:  # history memory still reached the model: ok
                    update(state, exit="host_drop", drop="late_callback")
            state = _pending.popleft()
            # Freeze under the lock before a late provider can update it.
            state.emitted = True
        safe(_emit)(state)
    if _debug_pending and time.monotonic() - _debug_at >= 300:
        _debug_pending = False
        _debug_at = time.monotonic()
        logger.debug("RECALL_EXIT_COUNTER_FAILED")


def session_env() -> tuple[str, str, str, str]:
    """The platform, sender, chat and chat type the gateway bound for the current turn, or blanks.

    A turn the host's owner wrote (``colonymind_shared.host_owner``, the decision the memory provider takes) reads
    as blanks. Hermes 0.22 binds the owner's own chat as a local chat of the operating-system user who runs it;
    that is the owner's lane with no sender, which is how Hermes 0.21 runs the terminal."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return "", "", "", ""
    bound = tuple(str(get_session_env(name, "") or "").strip() for name in (
        "HERMES_SESSION_PLATFORM", "HERMES_SESSION_USER_ID", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_CHAT_TYPE"))
    if bound[1] and bound[1] == host_owner.subject() and host_owner.running_turn():
        return "", "", "", ""
    return bound  # type: ignore[return-value]


# The one line a tool's result carries when a turn in a group chat was not given everything the tool reads.
GROUP_WITHHELD = ("this is a group chat: what does not come from this chat's own history was withheld; the owner "
                  "can ask you in a direct chat")


def group_chat() -> dict[str, str]:
    """The turn's chat when the gateway bound a group chat for it (a chat kind that is not a direct one): its kind
    and ``platform:chat id``, as the memory provider names it for the turn's context. Else nothing.

    The turn's recall reads that chat's own history only, and its reply goes to everyone in the chat, so a tool's
    read in the turn names the same chat and is narrowed the same way, or is refused (``GROUP_WITHHELD``)."""
    try:
        from gateway.session_context import get_session_env
        kind, platform, chat = (str(get_session_env(name, "") or "").strip() for name in (
            "HERMES_SESSION_CHAT_TYPE", "HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID"))
    except Exception:
        return {}
    if not kind or kind.lower() in DM_CHAT_TYPES:
        return {}
    return {"chat_type": kind.lower(), **({"chat": f"{platform.lower()}:{chat}"} if chat else {})}


def text_of(content: Any) -> str:
    """Plain text of a message content value (string or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(block.get("text") or "") for block in content
                 if isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}]
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def owner_input_text(session_id: str, turn_id: str, text: str, *, legacy_text: str | None = None,
                     own_words: str | None = None) -> str | None:
    """Use the hands ingress ledger before treating user-role text as owner words.

    With hands absent the adapter keeps its existing attribution. An installed bridge must return verified text;
    a missing callback, an unreadable ledger or an invalid result cannot grant owner provenance. Capture's pre-call hook supplies
    structural turn history first, since memory hooks may run before the front's own hook. ``own_words`` is what the
    adapter heard the owner send for this turn (Hermes' notes and quote removed): once the bridge accepts the turn,
    reads of what he said (the answer route, a branch verdict) take it when the accepted text carries more.
    """
    bridge = sys.modules.get("colonymind_hands_front")
    if bridge is None:
        return text if legacy_text is None else legacy_text
    try:
        clean = bridge.capture_owner_text(str(session_id or ""), str(turn_id or ""), text)
    except Exception as error:
        logger.warning("owner provenance unavailable (%s)", type(error).__name__)
        return None
    if not isinstance(clean, str):
        return None
    return own_words if own_words and own_words.strip() != clean.strip() else clean


@dataclass(frozen=True)
class Heard:
    """What the gateway's own event showed of one message before Hermes added to it (``SessionMap.hear``): where
    it came from, when it arrived (``MessageEvent.timestamp``: a message queued behind a busy session keeps its own
    time), its id there, what it replies or reacts to, whether it is a voice note, an event from the agent's own
    account or one the channel marks forwarded, and a salted digest of the words as sent (never the words: the
    gateway hears a message before a secret in it is sealed). A turn takes the record whose digest matches its own
    words, so two quick messages never swap their evidence."""
    platform: str
    sender: str
    chat_id: str
    chat_type: str
    message_id: str
    arrived_at: str
    reply_to: str = ""
    reaction: str = ""
    reaction_removed: bool = False
    voice: bool = False
    own_account: bool = False
    forwarded: bool = False
    digest: str = ""
    heard_at: float = 0.0
    rewrite: Any = None

    @property
    def reacts(self) -> bool:
        return bool(self.reaction or self.reaction_removed)

    def evidence(self) -> dict[str, Any]:
        """The message's own evidence as the sidecar reads it (``POST /v1/mind/answers``, and the ingest read of a
        captured turn): never words, never what the chat showed."""
        return {"chat_id": self.chat_id, "chat_type": self.chat_type, "message_id": self.message_id,
                "reply_to_message_id": self.reply_to, "reaction": self.reaction,
                "reaction_removed": self.reaction_removed, "forwarded": self.forwarded, "voice": self.voice,
                "own_account": self.own_account, "arrived_at": self.arrived_at}


def genuine(info: SessionInfo | None, words: Any = None) -> bool:
    """Whether a turn is a genuine inbound message, the only kind read as the owner's answer (the ingress ledger):
    ``pre_gateway_dispatch`` recorded it (``SessionMap.hear``: Hermes runs that hook for every message a platform
    delivers and never for an internal event such as a background delegation's completion, a watchdog's redirect or
    an injected turn), the turn took that record by its words (``SessionMap._take_heard``), the record carries the
    id its platform gave the message, and ``words``, when given (the turn's user message as captured), are still the
    words it matched (a redirect joined to the message after the model started changes them)."""
    heard = info.heard if info is not None else None
    if heard is None or not heard.message_id:
        return False
    return words is None or text_of(words).strip() == info.user_message.strip()


def ingress_verified(info: SessionInfo | None, session_id: str, turn_id: str, words: Any = None) -> bool:
    """Whether a turn is the owner's own inbound message. With hands, the front's ingress ledger vouches for a turn
    it bound to the platform's message (``owner_started_turn``; it also strips the redirect lines it added);
    otherwise, or when it does not vouch, the adapter's own record decides (``genuine``)."""
    if info is not None and info.ingress_rejected:
        return False
    bridge = sys.modules.get("colonymind_hands_front")
    try:
        if bridge is not None and bridge.owner_started_turn(str(session_id or ""), str(turn_id or "")) is True:
            return True
    except Exception as error:
        logger.warning("hands ingress verdict unavailable (%s)", type(error).__name__)
    return genuine(info, words)


def arrival_of(value: Any) -> str:
    """An event's arrival time as ISO 8601 UTC (a naive stamp is the gateway's local time); now when unreadable."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return datetime.fromtimestamp(time.time(), timezone.utc).isoformat()


def whatsapp_reaction(metadata: Mapping[str, Any]) -> tuple[str, str] | None:
    """``(emoji, target message id)`` of a stock WhatsApp reaction event (``metadata.whatsapp_native.reaction``; an
    empty emoji is a reaction removed), or None."""
    native = metadata.get("whatsapp_native") if isinstance(metadata, Mapping) else None
    reaction = native.get("reaction") if isinstance(native, Mapping) else None
    if not isinstance(reaction, Mapping) or not str(reaction.get("messageId") or "").strip():
        return None
    return str(reaction.get("text") or "").strip(), str(reaction["messageId"]).strip()


# The stock adapters that deliver a tapback on one of the agent's messages as a message event: its text
# ``reaction:added:<emoji>`` (or ``removed``), replying to its target, which the adapter marks as the agent's own.
TAPBACK_PLATFORMS = frozenset({"photon"})
TAPBACK = re.compile(r"reaction:(added|removed):(.*)", re.DOTALL)


def tapback_reaction(platform: str, event: Any) -> tuple[str, str] | None:
    """``(emoji, target message id)`` of a stock tapback event (an empty emoji: a tapback removed), or None. Only on
    an adapter that delivers tapbacks this way, and only an event the adapter itself built as one: a reply to the
    agent's own message, naming its target. A typed look-alike carries no target and stays a message."""
    if platform not in TAPBACK_PLATFORMS or getattr(event, "reply_to_is_own_message", False) is not True:
        return None
    target = str(getattr(event, "reply_to_message_id", "") or "").strip()
    match = TAPBACK.fullmatch(str(getattr(event, "text", "") or "").strip())
    if not target or match is None:
        return None
    emoji = "" if match.group(1) == "removed" else match.group(2).strip()
    return (emoji, target) if emoji or match.group(1) == "removed" else None


@dataclass
class SessionInfo:
    session_id: str
    platform: str = ""
    sender_id: str = ""
    user_message: str = ""
    parent_session_id: str = ""
    seen_at: float = field(default_factory=time.time)
    owner: bool | None = None
    contact_id: str | None = None
    # The chat the gateway bound for the sender's turn ("" without a gateway: the sender's own direct chat).
    chat_id: str = ""
    chat_type: str = ""
    turn_id: str = ""
    # What the gateway's own event showed of this turn's message (``SessionMap.hear``), matched by its words; None
    # without a gateway or when no record matches (a sealed secret changed the words, a steered message).
    heard: Heard | None = None
    # The owner's own words in this turn as the answer route reads them: what they sent (the gateway's quote and
    # notes removed) when the record matched, a voice note's transcript, else the turn's message.
    said: str = ""
    members: tuple = ()
    # Whether this turn's record was selected by the platform message id the gateway bound for it, rather than by
    # its words (``SessionMap._take_heard``). Only an anchored record attests a message that carries no words of
    # the sender's: with nothing to match, the id is the only thing tying the record to this turn.
    anchored: bool = False
    ingress_rejected: bool = False
    inherited_standing: str = ""
    inherited_platform: str = ""
    # Set when the turn's hook named Hermes' subject for the host's owner as its sender (Hermes 0.22's local
    # sessions): True when ``host_owner`` takes the turn as the owner's, which it does for a turn without a named
    # author on the owner's own surface (the sender is then blank, the owner's own lane), False when it does not (a
    # message there whose named author is a bot or a peer, another surface). Such a turn is nobody's: the subject
    # stays its sender, never becomes a contact, and the turn is not captured.
    host: bool | None = None


# A gateway record waits for its turn with no age limit. Records are bounded by COUNT only: HEARD_PER_SENDER per
# sender, SessionMap.limit senders. The cost of no age limit is disambiguation, not memory: an unmatched message
# stays a candidate for a later turn with the same words, and two candidates with the same words resolve to
# neither (``_take_heard``), so that turn gets no owner evidence.
SENT_FILES_KEPT = 64      # the attachments kept per conversation (sender and chat), newest last
# Stock delegation runs a child agent on this platform with no sender, naming its parent session.
DELEGATED_PLATFORM = "subagent"
LINEAGE_DEPTH = 8
BOUND_STANDING_SECONDS = 600.0   # how long a session's standing read from the gateway's binding is kept
HEARD_PER_SENDER = 16   # at most this many records wait per sender (a burst of messages); the oldest go first

# Assumed local capture ceilings: 128 KiB per span, 1 MiB per branch, before SQLite sees any text.
BRANCH_SPAN_BYTES, BRANCH_BYTES = 128 * 1024, 1024 * 1024


def branches_enabled(settings: Settings, *, consult: bool = False) -> bool:
    faculties = settings.mind().get("faculties")
    if isinstance(faculties, Mapping) and faculties.get("working_memory") is False:
        return False
    options = read_yaml(settings.home / "colonymind.yaml").get("working_memory")
    return not (consult and isinstance(options, Mapping) and options.get("summary_only") is True)


def branch_sender(info: SessionInfo) -> dict[str, str]:
    return {"platform": info.platform.strip().lower(), "user_id": info.sender_id}


def branch_owner(sessions: SessionMap, session_id: str) -> SessionInfo | None:
    """The locally established owner parent, with no network lookup on a capture hook."""
    info = sessions.origin(session_id)
    if info is None or info.chat_type not in {"", "dm", "direct", "private"}:
        return None
    platform = info.platform.strip().lower()
    if platform in {"sms", "rcs"}:
        return info if info.sender_id.strip().lower() in sessions._owner_handles(platform) else None
    if (not info.sender_id and platform == "cli") or (info.sender_id and (
            info.owner is True or info.sender_id.strip().lower() in sessions._owner_handles(platform))):
        return info
    return None


def branch_text(value: Any) -> str:
    """Mask secrets before the durable outbox; an oversized span is refused whole."""
    from .secrets import hide_secrets
    text = text_of(value)
    if len(text.encode("utf-8")) > BRANCH_SPAN_BYTES:
        raise ValueError("branch span exceeds local capture ceiling")
    clean = hide_secrets(text)
    if len(clean.encode("utf-8")) > BRANCH_SPAN_BYTES:
        raise ValueError("masked branch span exceeds local capture ceiling")
    return clean


def branch_structure(value: Any) -> Any:
    """Mask text leaves before JSON escaping can hide credential labels in tool arguments."""
    if isinstance(value, str):
        return branch_text(value)
    if isinstance(value, Mapping):
        return {key: branch_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [branch_structure(item) for item in value]
    return value


class SessionMap:
    """``session_id -> (sender, platform, last user message)``, bounded."""

    def __init__(self, settings: Settings, client: ColonyMindClient, *, limit: int = 2048):
        self.settings, self.client, self.limit = settings, client, limit
        self._sessions: OrderedDict[str, SessionInfo] = OrderedDict()
        self._lock = threading.Lock()
        self._heard: OrderedDict[tuple[str, str], list[Heard]] = OrderedDict()
        # (platform, sender, chat) -> (chat type, the local files the gateway's own events from that sender carried in
        # that chat) (``hear``)
        self._sent: OrderedDict[tuple[str, str, str], tuple[str, list[str]]] = OrderedDict()
        # session -> (standing, platform, when) read from the gateway's binding of a turn (``standing``)
        self._bound: OrderedDict[tuple[str, str, str, str, str], tuple[str, str, float]] = OrderedDict()
        self._salt = os.urandom(16)     # the digests never leave this process, nor outlive it
        self._composed_bindings = TurnBindings(
            Path(getattr(settings, "home", settings.outbox_path.parent)) / "capture-bindings.sqlite3", limit=limit)
        # The authority belongs to the instance, across Hermes profiles. Two
        # instances sharing a Hermes state directory must never share authority.
        self.standings = StandingStore(Path(getattr(settings, "home", settings.outbox_path.parent)) / STANDING_FILE)
        self.automation_standing: Callable[..., tuple[str, str]] | None = None

    def inherit(self, child_session_id: str, parent_session_id: str, *, standing: str = "",
                platform: str = "") -> tuple[str, str]:
        """Snapshot the creating turn, before the child can call tools; never follow a later group speaker."""
        saved = self.standings.get("session", child_session_id)
        if saved is not None:
            return saved
        if not standing:
            standing, platform = self.standing(parent_session_id) if parent_session_id else ("unknown", "")
        self.standings.put("session", child_session_id, standing, platform, parent_session_id, replace=False)
        return self.standings.get("session", child_session_id) or ("unknown", "")

    def creation_standing(self, session_id: str) -> tuple[str, str] | None:
        """Nearest immutable snapshot in Hermes' ancestry, including explicit unknown.

        Compression links the new session to the previous identity in the durable
        session database. It need not repeat the delegate-creation hook. Never
        traverse a saved child to consult a mutable group's later speaker.
        """
        current, seen = str(session_id or ""), set()
        while current and current not in seen and len(seen) < LINEAGE_DEPTH:
            seen.add(current)
            saved = self.standings.get("session", current)
            if saved is not None:
                if current != session_id:
                    return self.inherit(session_id, current, standing=saved[0], platform=saved[1])
                return saved
            info = self.get(current)
            if info is not None and info.sender_id:
                return None
            current = self.parent(current)
        return None

    def digest(self, words: Any) -> str:
        return hashlib.sha256(self._salt + str(words or "").strip().encode("utf-8", "replace")).hexdigest()

    def observe(self, *, session_id: str = "", platform: str = "", sender_id: str = "",
                user_message: Any = None, parent_session_id: str = "", turn_id: str = "",
                task_id: str = "",
                conversation_history: Any = None, **_: Any) -> SessionInfo | None:
        session_id = str(session_id or "")
        if not session_id:
            return None
        sender_id, host = str(sender_id or ""), None
        if sender_id and sender_id == host_owner.subject():
            # Hermes 0.22 names the operating-system user who runs it as the sender of every local lane. A task
            # run and a delegated child keep their lane's own rules, with no sender, as on Hermes 0.21. Elsewhere
            # the memory provider's decision says whether the owner wrote the turn: then it is the owner's own
            # lane, again with no sender; otherwise the subject stays, as a sender that is nobody's contact.
            if os.environ.get("HERMES_KANBAN_TASK") or parent_session_id:
                sender_id = ""
            else:
                host = host_owner.running_turn(session_id, (str(platform or ""), sender_id))
                sender_id = "" if host else sender_id
        # Resolve the database's compression link BEFORE observing the agent's
        # unchanged _parent_session_id (which still names its creating group).
        inherited = self.creation_standing(session_id) if not sender_id else None
        if not inherited and parent_session_id and not sender_id:
            saved = self.creation_standing(str(parent_session_id))
            inherited = self.inherit(session_id, str(parent_session_id),
                                     standing=saved[0] if saved else "unknown", platform=saved[1] if saved else "")
        if not inherited and not sender_id:
            where = str(platform or "").strip().lower()
            run, automated = self.automation_standing(session_id, task_id) if self.automation_standing else (
                "unknown", "")
            if automated:
                where = automated
            elif host or where == "cli":
                run = "owner"
            elif where in {"system", "internal"}:
                run = "system"
            elif where and where not in INTERNAL_PLATFORMS and where not in {DELEGATED_PLATFORM, "colonymind_task"}:
                # Preserve the existing senderless gateway guest lane. A
                # detached task or child still needs an explicit parent.
                run = "guest"
            inherited = self.inherit(session_id, "", standing=run, platform=where)
        with self._lock:
            info = self._sessions.get(session_id)
            if (info is None or info.sender_id != str(sender_id or "") or info.platform != str(platform or "")
                    or info.host != host):
                info = SessionInfo(session_id, str(platform or ""), str(sender_id or ""), host=host)
            info.user_message = text_of(user_message)
            info.parent_session_id = str(parent_session_id or "")
            info.turn_id = str(turn_id or "")
            if inherited and not sender_id:
                info.inherited_standing, info.inherited_platform = inherited
            info.chat_id, info.chat_type = self._bound_chat(info) or (info.chat_id, info.chat_type)
            # The gateway's record of this turn's own message (``hear``), matched by its words, once and fresh.
            info.members = ()
            info.heard, info.said = None, info.user_message
            info.anchored, info.ingress_rejected = False, False
            if not self._take_composed(info, conversation_history):
                info.heard, info.said = self._take_heard(info)
            info.seen_at = time.time()
            self._sessions[session_id] = info
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.limit:
                self._sessions.popitem(last=False)
        return info

    def get(self, session_id: str) -> SessionInfo | None:
        with self._lock:
            return self._sessions.get(str(session_id or ""))

    def hear(self, event: Any) -> Heard | None:
        """``pre_gateway_dispatch``: record, per sender and per message, what the gateway's event shows before
        Hermes adds to it (``Heard``): its chat, its id, when it arrived, what it replies or reacts to (a stock
        WhatsApp reaction or tapback names its target), a voice note, the agent's own account (a WhatsApp owner
        echo) or a forwarded mark, and a salted digest of its words. The words themselves are never kept (a secret
        is sealed only after this hook)."""
        source = getattr(event, "source", None)
        platform = getattr(source, "platform", "")
        platform = str(getattr(platform, "value", platform) or "").strip().lower()
        sender = str(getattr(source, "user_id", "") or getattr(event, "user_id", "") or "").strip()
        if not sender:
            return None
        rewrite = bind_rewrite(event, observer=self)
        kind = getattr(event, "message_type", "")
        metadata = getattr(event, "metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        reaction = whatsapp_reaction(metadata) or tapback_reaction(platform, event)
        heard = Heard(
            platform=platform, sender=sender, chat_id=str(getattr(source, "chat_id", "") or "").strip(),
            chat_type=str(getattr(source, "chat_type", "") or "").strip().lower(),
            message_id=str(getattr(event, "message_id", "") or getattr(source, "message_id", "") or "").strip(),
            arrived_at=arrival_of(getattr(event, "timestamp", None)),
            reply_to=reaction[1] if reaction else str(getattr(event, "reply_to_message_id", "") or "").strip(),
            reaction=reaction[0] if reaction else "", reaction_removed=bool(reaction) and not reaction[0],
            voice=str(getattr(kind, "value", kind) or "").strip().lower() == "voice",
            own_account=metadata.get("whatsapp_from_owner") is True,
            forwarded=metadata.get("forwarded") is True,
            digest=self.digest(getattr(event, "text", "")) if rewrite is not None else "",
            heard_at=time.time(), rewrite=rewrite)
        media = getattr(event, "media_urls", None)
        files = [str(item) for item in media if isinstance(item, str) and os.path.isabs(item)] \
            if isinstance(media, (list, tuple)) else []
        with self._lock:
            queue = self._heard.pop((platform, sender), [])
            self._heard[(platform, sender)] = [*queue, heard][-HEARD_PER_SENDER:]
            while len(self._heard) > self.limit:
                self._heard.popitem(last=False)
            if files and rewrite is not None:
                key = (platform, sender, heard.chat_id)
                kept = [item for item in self._sent.pop(key, ("", []))[1] if item not in files]
                self._sent[key] = (heard.chat_type, [*kept, *files][-SENT_FILES_KEPT:])
                while len(self._sent) > self.limit:
                    self._sent.popitem(last=False)
        return heard if rewrite is not None else None

    def sent_files(self, platform: str, sender: str, chat: str = "") -> list[str]:
        """The local files the gateway's own events from this sender carried in this chat (``hear``: an attachment
        Hermes saved): what that person sent in that conversation, never a path anyone typed, nor what anyone else
        sent, nor what they sent in another chat. With no chat (a session no gateway bound: the sender's direct
        chat), what they sent in a direct chat."""
        platform, sender, chat = str(platform or "").lower(), str(sender or ""), str(chat or "")
        with self._lock:
            if chat:
                return list(self._sent.get((platform, sender, chat), ("", []))[1])
            return [item for (where, who, _), (chat_type, files) in self._sent.items()
                    if (where, who) == (platform, sender) and (not chat_type or chat_type in DM_CHAT_TYPES)
                    for item in files]

    def _take_composed(self, info: SessionInfo, history: Any) -> bool:
        bound, _ = self._bound_message(info)
        members = composition(history, info.user_message, bound)
        if members is None:
            return False
        key = (info.platform.lower(), info.sender_id)
        queue = self._heard.get(key, [])
        taken = []
        for message_id, words in members:
            present = [heard for heard in queue if heard.message_id == message_id
                       and heard.chat_id == info.chat_id]
            matches = [heard for heard in present if self._matches(heard, words)]
            if present and (len(matches) != 1 or len(present) != 1):
                info.ingress_rejected = True
                return True  # invalid PRESENT evidence cannot fall back or recover
            if matches:
                taken.append(replace(info, user_message=words, heard=matches[0], said=words, members=()))
        scope = (info.session_id, info.turn_id, info.platform.lower(), info.sender_id,
                 info.chat_id, info.chat_type, bound)
        consumed = [member.heard for member in taken]
        saved = self._composed_bindings.recover(scope, info.user_message, members)
        if saved is False:
            info.ingress_rejected = True
            return True
        if saved is None and len(taken) == len(members):
            evidence = [{name: value for name, value in vars(member.heard).items()
                         if name not in {"digest", "rewrite"}} for member in taken]
            if not self._composed_bindings.remember(scope, info.user_message, members, evidence,
                                                    [member.heard.rewrite for member in taken]):
                info.ingress_rejected = True
                return True
        elif saved is not None:
            taken = [replace(info, user_message=words, said=words, members=(),
                             heard=Heard(**evidence, digest=self.digest(words)))
                     for (_, words), evidence in zip(members, saved)]
        else:
            return True  # missing evidence preserves generic capture, never owner attribution
        info.members = tuple(taken)
        info.heard = taken[-1].heard
        info.said = "\n\n".join(member.said for member in taken)
        self._heard[key] = [heard for heard in queue if all(heard is not own for own in consumed)]
        return True

    def _matches(self, heard: Heard, words: str) -> bool:
        return valid_ingress(heard.rewrite) and (
            self.digest(words) == heard.digest or matches_rewrite(heard.rewrite, words, strip=True))

    def _wordless(self, heard: Heard) -> bool:
        """Whether this record's message carried no words of the sender's: the gateway's event text was empty.
        An attachment sent with no caption arrives this way, and so does a voice note sent with no caption -
        for voice whether or not the transcription produced anything, because a transcript is the host's
        rendering of the audio and never the event's own text. The turn is then all host text - an attachment
        note, a description, a quoted transcript - with no words of the sender's in it to find.

        Every wordless message digests the same, so words can never tell two of them apart, and a wordless record
        offered to a match by words is a record that matches anything. ``_take_heard`` therefore uses this in both
        directions: a wordless record is taken only where the platform message id already bound it to this turn,
        and is excluded from the branch that has no id to go on.
        """
        return self._matches(heard, "")

    def _take_heard(self, info: SessionInfo) -> tuple[Heard | None, str]:
        """Match this turn to its own fresh gateway message by id, chat and digest; ambiguous matches grant no evidence."""
        message = info.user_message
        if not info.sender_id:
            return None, message
        key = (info.platform.lower(), info.sender_id)
        queue = [item for item in self._heard.get(key, [])
                 if not (info.chat_id and item.chat_id != info.chat_id)]
        others = [item for item in self._heard.get(key, []) if item not in queue]
        bound, known = self._bound_message(info)
        if bound:
            own = [item for item in queue if item.message_id == bound]
            queue = [item for item in queue if item.message_id != bound]      # this turn's, matched or not
            matches = [(item, said) for item in own for said in [self._said(item, message)] if said is not None]
            taken = matches[0] if matches else None
            if taken is None and len(own) == 1 and self._wordless(own[0]):
                # This turn IS that message (its id bound it) and that message carried no words, so there is
                # nothing to match and no words to report: the record is evidence, and ``said`` is empty.
                taken = (own[0], "")
            if taken is not None:
                info.anchored = True          # the gateway's id for this turn chose the record, not its words
            if own and taken is None:
                info.ingress_rejected = True
        else:
            matches = [(item, said) for item in queue
                       if not (known and item.message_id) and not self._wordless(item)
                       for said in [self._said(item, message)] if said is not None]
            taken = matches[0] if len(matches) == 1 else None
            if taken is not None:
                queue.remove(taken[0])
        remaining = [item for item in self._heard.get(key, []) if item in queue or item in others]
        if remaining:
            self._heard[key] = remaining
        else:
            self._heard.pop(key, None)
        return taken if taken is not None else (None, message)

    def _said(self, heard: Heard, message: str) -> str | None:
        """The owner's words in ``message`` when ``heard`` is its record (its digest matches), else None."""
        if not valid_ingress(heard.rewrite):
            return None
        parts = message.split("\n\n")
        for index in range(len(parts)):
            line = parts[index].strip()
            if heard.voice:
                caption = "\n\n".join(parts[index + 1:]).strip()
                if len(line) >= 2 and line[0] == line[-1] == '"' and self.digest(caption) == heard.digest:
                    return (line[1:-1].strip() + ("\n\n" + caption if caption else "")).strip()
            else:
                rest = "\n\n".join(parts[index:]).strip()
                if self._matches(heard, rest):
                    return rest
        return None

    @staticmethod
    def _bound_message(info: SessionInfo) -> tuple[str, bool]:
        """``(id, known)``: the id of the message this turn is for, when the turn the gateway bound is this session's
        sender's. A queued follow-up runs under the session variables of the message that opened the run, so the
        turn's own id comes from the agent running it (``busy_ingress.turn_platform_id``) and ``known`` is True
        (an empty id is then the turn's own: an internal event); without that agent, ``HERMES_SESSION_MESSAGE_ID``.
        ``("", False)`` otherwise (no gateway, or an adapter that names none)."""
        platform, sender, _, _ = session_env()
        if not sender or sender != info.sender_id or platform.lower() != info.platform.lower():
            return "", False
        own = turn_platform_id(info.session_id, info.turn_id)
        if own is not None:
            return own.strip(), True
        try:
            from gateway.session_context import get_session_env
        except Exception:
            return "", False
        return str(get_session_env("HERMES_SESSION_MESSAGE_ID", "") or "").strip(), False

    @staticmethod
    def _bound_chat(info: SessionInfo) -> tuple[str, str] | None:
        """The gateway's ``(chat, chat type)`` when the turn it bound is this session's sender's, else None."""
        platform, sender, chat, chat_type = session_env()
        if (chat or chat_type) and sender == info.sender_id and platform.lower() == info.platform.lower():
            return chat, chat_type.lower()
        return None

    def chat(self, session_id: str) -> tuple[str, str]:
        """Where a session's final response goes: the chat bound now or when its turn began; blanks without a
        gateway (the sender's direct chat)."""
        info = self.get(session_id)
        return ("", "") if info is None else self._bound_chat(info) or (info.chat_id, info.chat_type)

    def _owner_handles(self, platform: str) -> set[str]:
        return {item.lower() for item in self.settings.owner_handles().get(platform.lower(), [])}

    def parent(self, session_id: str) -> str | None:
        """The session a session was delegated or compressed from: what this process saw, else Hermes' store; ''
        for none, None when it cannot be read."""
        info = self.get(session_id)
        if info is not None:
            return str(info.parent_session_id or "")
        try:
            from hermes_state_registry import acquire, release_or_close
            db = acquire()
        except Exception as error:
            from .authority_diagnostics import ancestry_unreadable
            ancestry_unreadable(logger, error)
            return None
        try:
            row = db.get_session(session_id)
            return str((row or {}).get("parent_session_id") or "") if isinstance(row, Mapping) or row is None \
                else None
        except Exception as error:
            from .authority_diagnostics import ancestry_unreadable
            ancestry_unreadable(logger, error)
            return None
        finally:
            try:
                release_or_close(db)
            except Exception as error:
                from .authority_diagnostics import ancestry_unreadable
                ancestry_unreadable(logger, error)

    def origin(self, session_id: str) -> SessionInfo | None:
        """The session a turn's sender spoke in: this one, else the one a delegated child or a compressed session
        comes from (its parents followed here or in Hermes' store); None when none is known."""
        current, seen = str(session_id or ""), set()
        while current and current not in seen and len(seen) < LINEAGE_DEPTH:
            seen.add(current)
            info = self.get(current)
            if info is not None and not (info.platform.strip().lower() == DELEGATED_PLATFORM and not info.sender_id):
                return info
            current = self.parent(current) or ""
        return None

    def standing(self, session_id: str) -> tuple[str, str]:
        """``(owner | mind | system | guest | unknown, platform)``. Detached sessions use their explicit creating-turn
        snapshot. A session the plugin saw a turn
        of is read from its sender (``is_owner``). A delegated child and a session the plugin has not seen a turn of
        (a compression's new id) take the standing of the session they come from (``origin``); with none there, the
        gateway's binding of the current turn says: a bound sender is the owner only once the owner's handle or
        contact is established, and is a guest otherwise, the sidecar unreachable included; a gateway platform with
        no sender is a guest. With nothing at all the session is unknown.
        An unreadable session store with no recorded sender or creating-turn standing yields unknown, never the
        gateway's bound sender."""
        own = self.get(session_id)
        if own is not None and own.inherited_standing and not own.sender_id:
            return own.inherited_standing, own.inherited_platform
        inherited = self.creation_standing(str(session_id or "")) if own is None or not own.sender_id else None
        if inherited and (own is None or not own.sender_id):
            return inherited
        if own is None or not own.sender_id:
            parent_id = self.parent(session_id)
            if parent_id is None or parent_id:
                # ``None`` is "the ancestry could not be read", which is not "there is no parent": only
                # ``""`` attests that this session stands alone. An unreadable store says nothing, and a
                # session whose ancestry is unknown must not borrow the standing of whichever sender the
                # gateway happens to have bound for the current turn.
                return "unknown", ""
        platform, sender, chat, chat_type = session_env()
        info = self.origin(session_id)
        # A group session is a route, not an owner credential. The currently
        # bound sender wins over a previous turn's cached SessionInfo.
        if info is not None and sender and not info.parent_session_id and (
                sender != info.sender_id or platform.lower() != info.platform.lower()):
            info = None
        if info is not None:
            owner = self.is_owner(info.session_id)
            return ("owner" if owner else "guest" if owner is False else "unknown"), info.platform.strip().lower()
        parent_id = self.parent(session_id)
        if parent_id is None or parent_id:
            # The same rule after ``origin`` found nothing: an unreadable ancestry is an explicit unknown,
            # never a fall-through to the bound sender.
            return "unknown", ""
        key = (str(session_id or ""), platform, sender, chat, chat_type)
        platform = platform.strip().lower()
        if platform in {"sms", "rcs"}:
            return ("owner" if self.sender_is_owner(platform, sender, timeout=2.0) else "guest"), platform
        with self._lock:
            cached = self._bound.get(key)
        if cached is not None and time.time() - cached[2] < BOUND_STANDING_SECONDS:
            return cached[0], cached[1]
        if not sender:
            gateway = platform and platform not in INTERNAL_PLATFORMS and platform != DELEGATED_PLATFORM
            if gateway:
                logger.warning("standing_sender_unresolved")
            return ("guest" if gateway else "unknown"), platform
        if sender.strip().lower() in self._owner_handles(platform):
            run, known = "owner", True
        else:
            owner_id = self.settings.owner_contact_id()
            try:
                contact = self.client.resolve_contact(platform, sender, timeout=2.0)
                if not contact:
                    logger.warning("standing_sender_unresolved")
                run, known = ("owner" if contact and owner_id and str(contact.get("contact_id")) == owner_id
                              else "guest"), True
            except Exception as error:  # not established: a guest this once, asked again at the next call
                logger.warning("standing_sender_unresolved exception_type=%s", type(error).__name__)
                run, known = "guest", False
        if known:
            with self._lock:
                self._bound[key] = (run, platform, time.time())
                while len(self._bound) > self.limit:
                    self._bound.popitem(last=False)
        return run, platform

    def is_owner(self, session_id: str, *, timeout: float | None = None) -> bool | None:
        """True for the owner, False for anyone else, None for an unknown session. ``timeout`` bounds the sidecar's
        resolution of the sender (a caller's budget); unresolved in time, the sender is not the owner this once."""
        info = self.get(session_id)
        if info is None:
            return None
        if info.platform.strip().lower() in {"sms", "rcs"}:
            info.owner = self.sender_is_owner(info.platform, info.sender_id, timeout=timeout)
            return info.owner
        if info.owner is not None:
            return info.owner
        if info.host is not None:
            info.owner = info.host   # Hermes' own subject for the host's owner: the shared decision, no lookup
            return info.owner
        if not info.sender_id and info.platform.lower() in INTERNAL_PLATFORMS:
            info.owner = True
            return True
        if info.sender_id.strip().lower() in self._owner_handles(info.platform):
            info.owner = True
            return True
        owner_id = self.settings.owner_contact_id()
        contact = self.contact_id(session_id, timeout=timeout)
        if contact is None:
            logger.warning("standing_sender_unresolved")
            return False  # unresolved senders never gain owner authority
        info.owner = bool(owner_id) and contact == owner_id
        return info.owner

    def known_owner(self, session_id: str) -> bool:
        """Independent fallback for a known direct sender; never reuse another group speaker."""
        info = self.get(session_id)
        if info is None or info.parent_session_id or (not info.sender_id and info.platform != "cli"):
            return False
        if info.inherited_standing and info.inherited_standing != "owner":
            return False
        platform, sender, _, _ = session_env()
        if sender and (sender != info.sender_id or platform.lower() != info.platform.lower()):
            return False
        return self.is_owner(session_id) is True

    def sender_is_owner(self, platform: str, sender_id: str, *, timeout: float | None = None) -> bool:
        """One of the owner's handles, or a handle the sidecar resolves to the owner's contact (within ``timeout``
        when given). Nothing is recorded and no contact is created; an unresolvable sender is not the owner."""
        platform, sender = str(platform or "").strip().lower(), str(sender_id or "").strip()
        if not sender:
            logger.warning("standing_sender_unresolved")
            return False
        if sender.lower() in self._owner_handles(platform):
            return True
        owner_id = self.settings.owner_contact_id()
        if not owner_id:
            logger.warning("standing_owner_unconfigured")
            return False
        try:
            contact = self.client.resolve_contact(platform, sender, **({"timeout": timeout} if timeout else {}))
        except Exception as error:
            logger.warning("standing_sender_unresolved exception_type=%s", type(error).__name__)
            return False
        if not contact:
            logger.warning("standing_sender_unresolved")
        return bool(contact) and str(contact.get("contact_id")) == owner_id

    def owner_only(self, session_info: Mapping[str, Any]) -> bool:
        """Whether the session whose prompt Hermes is rendering is the owner's alone."""
        platform, sender, chat, chat_type = session_env()
        platform = (platform or str(session_info.get("platform") or "")).strip().lower()
        if chat_type.lower() not in {"", "dm"}:
            return False
        if not sender:
            # Blanks are the terminal of Hermes 0.21, or a turn of Hermes 0.22 that the host's owner wrote.
            return not chat and (platform in INTERNAL_PLATFORMS
                                 or host_owner.running_turn(str(session_info.get("session_id") or "")))
        known = self.get(str(session_info.get("session_id") or ""))
        if platform not in {"sms", "rcs"} and known is not None and known.sender_id == sender and known.owner is not None:
            return known.owner
        return self.sender_is_owner(platform, sender)

    def contact_id(self, session_id: str, *, timeout: float | None = None) -> str | None:
        """The sidecar contact for a session's sender; the owner for internal turns."""
        info = self.get(session_id)
        if info is None:
            return None
        if info.platform.strip().lower() in {"sms", "rcs"}:
            owner_id = self.settings.owner_contact_id()
            if info.sender_id.strip().lower() in self._owner_handles(info.platform):
                info.contact_id = owner_id or None
                return info.contact_id
            # Re-resolve every phone sender so a removed deployment binding cannot survive this cache.
            info.contact_id = None
        if info.contact_id:
            return info.contact_id
        if info.host is False:
            return None   # Hermes' subject for the host's owner on a turn someone else wrote: nobody's contact
        if not info.sender_id:
            if info.host or info.platform.lower() in INTERNAL_PLATFORMS:
                info.contact_id = self.settings.owner_contact_id() or "owner"
                return info.contact_id
            return None
        try:
            contact = self.client.resolve_contact(info.platform, info.sender_id, create=True,
                                                  **({"timeout": timeout} if timeout else {}))
        except (SidecarUnavailable, Exception) as error:
            logger.warning("standing_sender_unresolved exception_type=%s", type(error).__name__)
            return None
        if contact:
            info.contact_id = str(contact["contact_id"])
        return info.contact_id


# A hook writes one row before the reply returns, while the body thread reads the same file.
# Five seconds of busy waiting was not enough under a burst of concurrent turns.
BUSY_TIMEOUT_SECONDS = 30
# What a delivered row keeps of its payload: nothing (its turn id and envelope digest stay). A rejected row keeps it.
DELIVERED_PAYLOAD, KEPT = "{}", "(last_error LIKE 'rejected:%')"
ENQUEUE_ATTEMPTS = 4
ENQUEUE_RETRY_SECONDS = 0.05
WORK_EVENT_ATTEMPTS, WORK_EVENT_MAX_AGE = 20, 86400


class TurnOutbox:
    """Durable SQLite ledger of captured turns shared across Hermes processes.

    The row commits with ``synchronous=FULL`` before the hook returns. Delivery
    claims rows under a lease so several processes can drain the same file. A
    delivered row keeps its turn id and envelope digest and drops its payload
    (the sidecar holds the turn, its secrets sealed); a rejected row keeps its
    payload in this private file, the turn's only copy, for repair. A row the
    sidecar refuses because its source was erased drops its payload too.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            if not self._ready:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not self.path.exists():
                    self.path.touch(mode=0o600)
                os.chmod(self.path, 0o600)
        connection = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
        connection.row_factory = sqlite3.Row
        # WAL keeps the body thread's reads from blocking a hook's write: with the rollback
        # journal, a few dozen turns arriving at once made writers wait on readers until the
        # busy timeout expired and the turn was dropped. A filesystem without shared memory
        # refuses the mode and keeps the old one, which still works.
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA secure_delete=ON")  # a dropped payload leaves no bytes in free pages
        with self._lock:
            if not self._ready:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS turn_outbox (
                        turn_id TEXT PRIMARY KEY, envelope_sha256 TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        state TEXT NOT NULL CHECK (state IN ('pending', 'delivered')),
                        attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                        updated_at REAL NOT NULL, last_error TEXT NOT NULL DEFAULT '',
                        lease_id TEXT NOT NULL DEFAULT '', lease_expires_at REAL NOT NULL DEFAULT 0)
                """)
                connection.execute("CREATE TABLE IF NOT EXISTS work_admission (turn_id TEXT PRIMARY KEY,"
                                   " admitted INTEGER NOT NULL)")
                connection.execute(f"UPDATE turn_outbox SET payload_json = '{DELIVERED_PAYLOAD}' WHERE state = "
                                   f"'delivered' AND payload_json != '{DELIVERED_PAYLOAD}' AND {KEPT} = 0")
                self._ready = True
        return connection

    def enqueue(self, turn_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Commit one turn; a repeated turn id keeps the first row."""
        turn_id = str(turn_id or "").strip()
        if not turn_id:
            raise ValueError("turn_id is required")
        encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if len(encoded) > 8 * 1024 * 1024:
            raise ValueError("turn payload exceeds 8 MiB")
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        now = time.time()
        for attempt in range(ENQUEUE_ATTEMPTS):
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute("SELECT state FROM turn_outbox WHERE turn_id = ?", (turn_id,)).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO turn_outbox (turn_id, envelope_sha256, payload_json, state, attempts,"
                        " created_at, updated_at) VALUES (?, ?, ?, 'pending', 0, ?, ?)",
                        (turn_id, digest, encoded, now, now))
                connection.execute("COMMIT")
                return {"turn_id": turn_id, "state": row["state"] if row else "pending", "new": row is None}
            except sqlite3.OperationalError as error:
                # A busy database is the one failure worth retrying here: losing the row loses
                # the turn, and with it anything the person committed to in it.
                if "locked" not in str(error) and "busy" not in str(error):
                    raise
                if attempt == ENQUEUE_ATTEMPTS - 1:
                    raise
                time.sleep(ENQUEUE_RETRY_SECONDS * (attempt + 1))
            finally:
                connection.close()

    def work_admission(self, root: str, admitted: bool | None = None) -> bool | None:
        connection = self._connect()
        try:
            if admitted is not None:
                connection.execute("INSERT OR REPLACE INTO work_admission VALUES (?, ?)", (root, int(admitted)))
                return admitted
            row = connection.execute("SELECT admitted FROM work_admission WHERE turn_id = ?", (root,)).fetchone()
            return bool(row[0]) if row is not None else None
        finally:
            connection.close()

    def pending_count(self) -> int:
        return self._count("state = 'pending'")

    def refused_count(self) -> int:
        """Rows the sidecar refused to attribute as sent (docs/CREDENTIALS.md): failed captures."""
        return self._count("last_error LIKE ?", f"rejected: {REFUSED_PREFIX}%")

    def _count(self, where: str, *args: Any) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM turn_outbox WHERE " + where, args).fetchone()[0])
        finally:
            connection.close()

    def rows(self, state: str | None = None) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            query = "SELECT turn_id, payload_json, state, attempts, last_error FROM turn_outbox"
            rows = connection.execute(query + (" WHERE state = ?" if state else "") + " ORDER BY created_at",
                                      (state,) if state else ()).fetchall()
            return [{"turn_id": r["turn_id"], "payload": json.loads(r["payload_json"]), "state": r["state"],
                     "attempts": int(r["attempts"]), "last_error": r["last_error"]} for r in rows]
        finally:
            connection.close()

    def drain(self, deliver: Callable[[dict[str, Any]], bool], *, limit: int = 32,
              lease_seconds: float = 60.0) -> int:
        """Deliver up to ``limit`` pending rows; returns how many were delivered.

        ``deliver`` returns True when the sidecar accepted the row and False to
        retry later; an :class:`Undeliverable` settles the row with its error
        recorded and its payload kept, so a rejected one never blocks the queue.
        An :class:`Erased` settles the row with a fixed reason and its payload dropped.
        """
        delivered, lease = 0, uuid.uuid4().hex
        for _ in range(max(1, limit)):
            now = time.time()
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT turn_id, payload_json, attempts, created_at FROM turn_outbox WHERE state = 'pending'"
                    " AND lease_expires_at < ? ORDER BY created_at LIMIT 1", (now,)).fetchone()
                if row is not None:
                    connection.execute("UPDATE turn_outbox SET lease_id = ?, lease_expires_at = ? WHERE turn_id = ?",
                                       (lease, now + lease_seconds, row["turn_id"]))
                connection.execute("COMMIT")
            finally:
                connection.close()
            if row is None:
                break
            payload, error, done = json.loads(row["payload_json"]), "", False
            work_event = payload.get("kind") == "work_event"
            expired = work_event and (now - row["created_at"] >= WORK_EVENT_MAX_AGE
                                      or int(row["attempts"]) >= WORK_EVENT_ATTEMPTS)
            try:
                done = expired or bool(deliver(payload))
            except Erased:
                done, error = True, "erased: source_erased"  # fixed, and not "rejected:": the payload is dropped
            except Undeliverable as failure:
                done, error = True, f"rejected: {failure}"
            except Exception as failure:
                error = type(failure).__name__
            attempts = int(row["attempts"]) + 1
            if work_event and not done and attempts >= WORK_EVENT_ATTEMPTS:
                done = expired = True
            if expired:
                error = "expired: work event retry limit"
            backoff = min(300.0, 2.0 ** min(attempts, 8))
            connection = self._connect()
            try:
                if done:
                    connection.execute("UPDATE turn_outbox SET state = 'delivered', attempts = ?, updated_at = ?,"
                                       " last_error = ?, lease_id = '', lease_expires_at = 0, payload_json = ?"
                                       " WHERE turn_id = ?", (attempts, time.time(), error,
                                                              row["payload_json"] if error.startswith("rejected:") else DELIVERED_PAYLOAD,
                                                              row["turn_id"]))
                    delivered += 1
                else:
                    connection.execute("UPDATE turn_outbox SET attempts = ?, updated_at = ?, last_error = ?,"
                                       " lease_id = '', lease_expires_at = ? WHERE turn_id = ?",
                                       (attempts, time.time(), error, time.time() + backoff, row["turn_id"]))
                    if not work_event and (not error or error.startswith("SidecarUnavailable")):
                        break  # the sidecar is down: stop the pass, keep the rest for later
            finally:
                connection.close()
        return delivered

    def prune(self, keep_delivered: int = 4096) -> None:
        connection = self._connect()
        try:
            connection.execute(
                f"DELETE FROM turn_outbox WHERE state = 'delivered' AND {KEPT} = 0 AND turn_id NOT IN (SELECT turn_id"
                " FROM turn_outbox WHERE state = 'delivered' ORDER BY updated_at DESC LIMIT ?)", (keep_delivered,))
            connection.execute("DELETE FROM work_admission WHERE turn_id NOT IN (SELECT turn_id FROM turn_outbox)")
        finally:
            connection.close()


class Undeliverable(Exception):
    """The sidecar rejected the payload; retrying cannot help."""


class Erased(Exception):
    """The sidecar erased the turn's source; the row is settled and keeps nothing of the turn."""


class Capture:
    """Enqueue capture; synchronously release an interrupted transport binding."""

    def __init__(self, sessions: SessionMap, outbox: TurnOutbox, settings: Settings,
                 *, on_enqueue: Callable[[], Any] | None = None):
        self.sessions, self.outbox, self.settings = sessions, outbox, settings
        self.on_enqueue = on_enqueue
        self._hand_refs: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
        self._hand_refs_lock = threading.Lock()
        self.work = WorkCapture(settings, outbox)
        self._recall_owner_handles = self._load_recall_owner_handles()

    @safe
    def _load_recall_owner_handles(self):
        # Load once off the turn path, alongside the plugin's other settings.
        return {platform: {value.lower() for value in handles}
                for platform, handles in self.settings.owner_handles().items()}

    def record_hand_refs(self, *, session_id: str, turn_id: str, digest: Mapping[str, Any]) -> None:
        """The digest renderer records only result lines actually shown in this turn.

        This is a host callback, never a model tool argument. A stable turn id
        binds the snapshot to capture; another turn cannot inherit it. The
        integrator calls it after rendering the controller's digest.
        """
        if not session_id or not turn_id:
            raise ValueError("hand provenance requires session and turn ids")
        shown = set(str(digest.get("text") or "").splitlines())
        refs, seen = [], set()
        for item in digest.get("items", []):
            if not isinstance(item, Mapping):
                continue
            evidence = item.get("evidence")
            ident, line = item.get("hand_id"), item.get("text")
            if (not isinstance(evidence, Mapping) or not isinstance(ident, str) or not ident
                    or ident in seen or not isinstance(line, str) or "\n" in line
                    or "- " + line not in shown):
                continue
            level = evidence.get("level")
            if level not in {"E0", "E1", "E2", "E3", "E4", "E5"}:
                continue
            refs.append({"hand_id": ident, "evidence_level": level,
                         "name": str(item.get("name") or line.split(" (", 1)[0])})
            seen.add(ident)
        with self._hand_refs_lock:
            self._hand_refs[(session_id, turn_id)] = refs
            self._hand_refs.move_to_end((session_id, turn_id))
            # Assumption: retain at most 256 in-flight turn snapshots per host.
            while len(self._hand_refs) > 256:
                self._hand_refs.popitem(last=False)

    def _captured_hand_refs(self, session_id: str, turn_id: str) -> list[dict[str, str]]:
        with self._hand_refs_lock:
            return [dict(item) for item in self._hand_refs.get((session_id, turn_id), [])]

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        info = self.sessions.observe(**kwargs)
        if info and getattr(self.sessions, "write_guard", None):
            self.sessions.write_guard.begin(info.session_id, info.turn_id)
        context = None
        observe = getattr(sys.modules.get("colonymind_hands_front"), "observe_turn", None)
        if callable(observe):
            context = observe(**kwargs)
        self.work.start(info, **kwargs)
        self.branch_reaction(**kwargs)
        self.observe_recall_exit(**kwargs)
        return context if isinstance(context, dict) else None

    @safe
    def observe_recall_exit(self, **kwargs):
        info = self.sessions.get(str(kwargs.get("session_id") or ""))
        if info is not None:
            self._begin_recall_exit(info, kwargs)

    @safe
    def _begin_recall_exit(self, info, kwargs):
        history = kwargs.get("conversation_history")
        current = history[-1] if isinstance(history, (list, tuple)) and history else {}
        internal = (isinstance(current, dict) and current.get("role") == "user"
                    and current.get("display_kind") == "internal_notification")
        local_owner = (info.owner is True or info.host is True
                       or (not info.sender_id and info.platform in {"cli", "tui", "owner"}))
        local_owner = local_owner or info.sender_id.lower() in (self._recall_owner_handles or {}).get(info.platform, set())
        inbound = not info.sender_id or genuine(info)
        front = sys.modules.get("colonymind_hands_front")
        started = getattr(front, "owner_started_turn", None)
        if callable(started):
            inbound = inbound or started(info.session_id, info.turn_id) is True
        begin(info.session_id, info.turn_id, owner=local_owner, platform=info.platform,
            excluded=bool(not inbound or internal or kwargs.get("internal_notification") or kwargs.get("replay") or kwargs.get("rewind")
                      or kwargs.get("is_replay") or kwargs.get("is_rewind")
                      or os.environ.get("HERMES_KANBAN_TASK") or info.platform in {"cron", "subagent", "worker"}
                      # A group chat's turn recalls that chat's own history only; an empty recall there is no fault.
                      or getattr(info, "chat_type", "") not in {"", "dm", "direct", "private"}))

    def branch_reaction(self, *, session_id: str = "", turn_id: str = "", user_message: Any = None,
                        **_: Any) -> None:
        """Only the next owner turn after a successful consult supplies an outcome, never tool arguments."""
        info = self.sessions.get(session_id)
        pending = getattr(info, "branch_consult_turn", None)
        if pending is None or pending == turn_id:
            return
        raw = text_of(user_message)
        words = owner_input_text(session_id, turn_id or info.turn_id, raw,
                                 legacy_text=info.said or raw,
                                 own_words=info.said if info.heard is not None else None)
        if words is None:
            return  # a synthetic turn cannot use up the next owner's opportunity to label the consult
        # A later turn cannot label this consult if the actual next turn was withheld or failed to queue.
        delattr(info, "branch_consult_turn")
        if not branches_enabled(self.settings):
            return
        owner = branch_owner(self.sessions, session_id)
        if owner is None or os.environ.get("HERMES_KANBAN_TASK"):
            return
        try:
            payload = {"kind": "branch_reaction", "session_id": session_id,
                       "requester": self.settings.owner_contact_id(), "sender": branch_sender(owner),
                       "text": branch_text(words), "turn_id": str(turn_id or "")}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            self.outbox.enqueue("branch-reaction:" + digest, payload)
            if self.on_enqueue is not None:
                self.on_enqueue()
        except Exception as error:
            logger.debug("branch reaction not queued (%s)", type(error).__name__)

    def post_llm_call(self, **kwargs: Any) -> None:
        post_llm_call(**kwargs)
        try:
            self.work.finish_child(str(kwargs.get("turn_id") or ""))
            receipt = self._capture(**kwargs)
        except Exception as error:  # capture must never disturb the reply
            logger.warning("turn capture failed for %s (%s: %s)",
                           kwargs.get("turn_id") or "?", type(error).__name__, error)
            return None
        if receipt and receipt.get("new") and self.on_enqueue is not None:
            try:
                self.on_enqueue()  # wake the body thread; delivery stays off the hook
            except Exception as error:
                logger.debug("body wake failed (%s)", type(error).__name__)
        return None

    def on_native_turn_settled(self, *, session_id="", task_id="", turn_id="", outcome=None, **_):
        """Hermes runs this after releasing the native turn, before any injection.

        Interruption can skip capture. The durable transport's batch outlives
        the turn, so its next turn needs an explicit binding release now.
        """
        if getattr(self.sessions, "write_guard", None):
            self.sessions.write_guard.end(session_id, turn_id)
        native = {"session_id": session_id, "task_id": task_id, "turn_id": turn_id}
        if not all(native.values()) or not isinstance(outcome, Mapping) or outcome.get("interrupted") is not True:
            return
        try:
            # Releasing this exact tuple is idempotent, including a committed
            # request whose response timed out. Finish bounded retries before
            # the successor can bind; never retry admission or handoff here.
            for attempt in range(3):
                try:
                    response = self.sessions.client.post("/v1/host/transport/ingress/native-settled",
                        json={"native_turn": native, "interrupted": True}, timeout=2)
                    break
                except SidecarUnavailable:
                    if attempt == 2:
                        raise
            if not response.is_success:
                raise RuntimeError(f"HTTP {response.status_code}")
        except Exception as error:
            logger.warning("interrupted transport binding release failed (%s)", type(error).__name__)

    def native_turn_settled(self, *, turn_id: str = "", **kwargs: Any) -> None:
        """Failed/interrupted native turns have no post_llm_call; freeze once here."""
        activity = getattr(self, "activity_observer", None)
        if activity is not None:
            try:
                activity(turn_id=turn_id, **kwargs)  # memory only
            except Exception:
                pass  # optional observation must never prevent capture cleanup
        info = self.work.info(turn_id, unfinished=True)
        if info is not None:
            self.post_llm_call(**{**kwargs, "turn_id": turn_id, "user_message": info.user_message,
                                  "session_id": info.session_id, "assistant_response": ""})

    def skill_loaded(self, *, action: str = "", skill_name: str = "", session_id: str = "", task_id: str = "",
                     **_: Any) -> None:
        """``on_skill_lifecycle``: a load of one of ColonyMind's own skills (``colonymind-*``) is queued for
        ``POST /v1/mind/skills/used``; every other skill and action is Hermes' business. Enqueue only."""
        name = str(skill_name or "")
        if action != "loaded" or not name.startswith(SKILL_PREFIX):
            return None
        payload = {"kind": "skill_use", "skill": name, "session_id": str(session_id or ""),
                   "task_id": str(task_id or "")}
        try:  # a load is never disturbed by its bookkeeping
            self.outbox.enqueue(f"skill:{uuid.uuid4().hex}", payload)
            if self.on_enqueue is not None:
                self.on_enqueue()
        except Exception as error:
            logger.debug("skill load not queued (%s)", type(error).__name__)
        return None

    def _capture(self, *, session_id: str = "", task_id: str = "", turn_id: str = "",
                 user_message: Any = None, assistant_response: Any = None, model: str = "",
                 platform: str = "", conversation_history: Any = None, **_: Any) -> dict[str, Any] | None:
        """One row per turn. A turn in a gateway chat that is a genuine inbound message (``genuine``: the gateway's
        record of it matched, and the words captured are still the words it matched) names that message's own evidence
        (``answer``): a message the answer route could not read before the model ran is read when the turn is
        captured, by explicit references only. Any other turn names none, so the sidecar never reads it as an
        answer: a background completion, an injected turn, or a turn a redirect line was joined to."""
        session_id = str(session_id or "")
        if not session_id or os.environ.get("HERMES_KANBAN_TASK"):
            return None  # worker runs are task work, reconciled through kanban, not conversation
        info = self.work.info(turn_id) or self.sessions.get(session_id)
        if info is not None and info.ingress_rejected:
            return None  # a real arrival whose dispatch text failed provenance is never an owner capture
        if getattr(info, "host", None) is False:
            return None  # the host's own subject on a turn the owner did not write: it has no contact to file under
        if info is not None and info.parent_session_id and info.platform.lower() == DELEGATED_PLATFORM:
            return self._branch(info, conversation_history, user_message, assistant_response)
        where = str((info.platform if info is not None else platform) or "").lower()
        if not (info is not None and info.sender_id) and where not in INTERNAL_PLATFORMS and group_chat():
            return None  # a gateway turn in a group chat that names no sender is nobody's: never filed as the owner's
        raw = text_of(user_message)
        composed = info is not None and info.members and raw.strip() == info.user_message.strip()
        user = owner_input_text(session_id, turn_id or (info.turn_id if info else ""), raw,
                                legacy_text=info.said if composed else None,
                                own_words=info.said if composed else None)
        if user is None:
            return None
        assistant = text_of(assistant_response)
        if not user.strip() and not assistant.strip():
            return None
        stable = str(turn_id or "").strip() or "hermes:" + hashlib.sha256(json.dumps(
            [session_id, task_id, user, assistant], ensure_ascii=True).encode()).hexdigest()
        payload = {
            "session_id": session_id, "turn_id": stable, "task_id": str(task_id or ""),
            "platform": str(platform or (info.platform if info else "") or ""),
            "sender_id": info.sender_id if info else "", "user_message": user,
            "assistant_message": assistant, "model": str(model or ""),
            "occurred_at": datetime.fromtimestamp(time.time(), timezone.utc).isoformat(),
        }
        hand_refs = self._captured_hand_refs(session_id, stable)
        if hand_refs:
            payload["hand_refs"] = hand_refs
        answer_info = self.sessions.get(session_id)
        if (answer_info is not None and answer_info.sender_id and answer_info.chat_id
                and ingress_verified(answer_info, session_id, stable, user)):
            evidence = answer_info.heard.evidence() if answer_info.heard is not None else {}
            payload["answer"] = {**evidence, "chat_id": answer_info.chat_id, "chat_type": answer_info.chat_type}
        work = self.work.freeze(stable)
        if work is not None:
            payload["work"] = work
        if (info is not None and info.members and user.strip() == info.said.strip()
                and (composed or sys.modules.get("colonymind_hands_front") is not None)):
            receipt = None
            for member in info.members:
                own_id = stable if member is info.members[-1] else member_turn_id(stable, member.heard.message_id)
                piece = {**payload, "turn_id": own_id, "user_message": member.said,
                         "answer": member.heard.evidence()}
                if member is not info.members[-1]:
                    piece.pop("work", None)
                    piece.pop("hand_refs", None)
                receipt = self.outbox.enqueue(own_id, piece)
            return receipt
        return self.outbox.enqueue(stable, payload)

    def _branch(self, info: SessionInfo, history: Any, brief: Any, summary: Any) -> dict[str, Any] | None:
        if info.platform.lower() != DELEGATED_PLATFORM or not branches_enabled(self.settings):
            return None
        owner = branch_owner(self.sessions, info.parent_session_id)
        requester = self.settings.owner_contact_id()
        if owner is None or not requester:
            return None
        messages = []
        for message in history if isinstance(history, list) else []:
            if not isinstance(message, Mapping) or message.get("role") not in {"user", "assistant", "tool"}:
                continue
            # Preserve structured tool calls as text, so their arguments are part of the source span too.
            content = text_of(message.get("content"))
            if message.get("tool_calls"):
                content += "\n" + json.dumps(branch_structure(message["tool_calls"]), ensure_ascii=False)
            messages.append({"role": message["role"], "content": branch_text(content)})
        clean_summary = branch_text(summary)
        if clean_summary and (not messages or messages[-1] != {"role": "assistant", "content": clean_summary}):
            messages.append({"role": "assistant", "content": clean_summary})
        payload = {"kind": "branch", "parent_session": info.parent_session_id, "child_session": info.session_id,
                   "messages": messages, "brief": branch_text(brief or info.user_message),
                   "summary": clean_summary, "requester": requester, "sender": branch_sender(owner),
                   "queued_at": time.time()}
        if len(json.dumps(payload, ensure_ascii=True).encode()) > BRANCH_BYTES:
            raise ValueError("branch exceeds local capture ceiling")
        digest = hashlib.sha256(json.dumps([info.parent_session_id, info.session_id]).encode()).hexdigest()
        return self.outbox.enqueue("branch:" + digest, payload)


def checkpoint(messages: list[dict[str, Any]], *, session_id: str, contact_id: str,
               outbox: TurnOutbox, platform: str = "", sender_id: str = "") -> dict[str, Any]:
    """Enqueue direct user/assistant content before Hermes compresses it away. ``platform`` and ``sender_id``
    say where the participant speaks (docs/CREDENTIALS.md); they are not part of the checkpoint's identity."""
    from .voice_writeback import is_bridged_voice_row
    evidence = [{"role": m["role"], "content": m["content"]} for m in messages
                if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
                and not m.get("_compressed_summary") and not is_bridged_voice_row(m)
                and text_of(m.get("content")).strip()]
    if not evidence:
        return {"state": "empty", "messages": 0}
    if not session_id or not contact_id:
        raise ValueError("checkpoint requires a bound session and participant")
    payload: dict[str, Any] = {"session_id": session_id, "contact_id": contact_id, "checkpoint_messages": evidence}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    payload["turn_id"] = "checkpoint:" + digest
    payload["platform"], payload["sender_id"] = str(platform or ""), str(sender_id or "")
    receipt = outbox.enqueue(payload["turn_id"], payload)
    return {"state": receipt["state"], "messages": len(evidence), "turn_id": payload["turn_id"]}


def deliver(client: ColonyMindClient, sessions: SessionMap, settings: Settings,
            payload: Mapping[str, Any]) -> bool:
    """POST one outbox row to ``/v1/host/turns/sync`` (a skill load to ``/v1/mind/skills/used``); True when
    accepted."""
    if payload.get("kind") == "activity_snapshot":
        # Retire rows queued by the previous adapter without any activity I/O.
        # A legacy status row must never obstruct healthy conversation delivery.
        return True
    if payload.get("kind") == "work_event":
        if TurnOutbox(settings.outbox_path).work_admission(str(payload.get("root_turn_id") or "")) is False:
            return True
        faculties = settings.mind().get("faculties")
        if isinstance(faculties, Mapping) and faculties.get("work_records") is False:
            return True
        response = client.post("/v1/host/work/events", timeout=8,
                               json={key: value for key, value in payload.items() if key != "kind"})
        if response.status_code == 410 and refusal_code(response) in {
                "work_root_ineligible", "work_source_erased", "work_root_erased", "work_parent_expired"}:
            return True
        if response.status_code == 404 and refusal_code(response) in {
                "work_root_not_found", "root_not_found", "work_root_missing", "work_parent_missing",
                "work_lineage_missing"}:
            return False
        if response.status_code in {400, 404, 409, 413, 422}:
            raise Undeliverable(f"HTTP {response.status_code}")
        if not response.is_success:
            return False
        result = response.json()
        return result.get("accepted") is True or result.get("skipped_reason") == "work_records_disabled"
    if payload.get("kind") in {"branch", "branch_session", "branch_reaction"}:
        if not branches_enabled(settings):
            return True
        if payload.get("kind") == "branch":
            options = read_yaml(settings.home / "colonymind.yaml").get("working_memory") or {}
            # The planned retention assumption is 14 days; an expired pending capture is not delivered.
            days = float(options.get("retention_days", 14)) if isinstance(options, Mapping) else 14
            if time.time() - float(payload.get("queued_at") or 0) >= days * 86400:
                return True
        checkpoint_turn = payload.get("checkpoint_turn_id")
        if checkpoint_turn:
            dependency = next((row for row in TurnOutbox(settings.outbox_path).rows()
                               if row["turn_id"] == checkpoint_turn), None)
            if dependency is None or dependency["state"] != "delivered":
                return False
            if dependency["last_error"]:
                raise Undeliverable("session checkpoint was refused")
        path = {"branch": "", "branch_session": "/session", "branch_reaction": "/reaction"}[payload["kind"]]
        response = client.post("/v1/memory/branches" + path, timeout=8,
                               json={key: value for key, value in payload.items()
                                     if key not in {"kind", "checkpoint_turn_id", "queued_at"}})
        if response.status_code in {400, 401, 403, 404, 409, 413, 422}:
            # Unlike ordinary rejected turns, worker material must not remain in the outbox indefinitely.
            logger.warning("branch outbox request refused (HTTP %s)", response.status_code)
            return True
        return response.is_success
    if payload.get("kind") == "skill_use":
        response = client.post("/v1/mind/skills/used", timeout=5, json={
            "skill": payload.get("skill"), "session_id": payload.get("session_id") or None,
            "task_id": payload.get("task_id") or None})
        if response.status_code in {400, 404, 409, 413, 422}:
            raise Undeliverable(f"HTTP {response.status_code}")
        return response.is_success
    contact = str(payload.get("contact_id") or "")
    platform, sender = str(payload.get("platform") or ""), str(payload.get("sender_id") or "")
    if not contact:
        if sender:
            found = client.resolve_contact(platform, sender, create=True)
            if not found:
                raise Undeliverable("sender has no contact")
            contact = str(found["contact_id"])
        else:
            contact = settings.owner_contact_id() or "owner"
    # The platform the turn arrives on is named (the sender's chat, else the lane itself: "cli", "cron"):
    # the sidecar writes the owner's turns only from the platforms the key is granted the owner on.
    channel = f"{platform}:{sender}" if platform and sender else (platform or None)
    metadata: dict[str, Any] = {"occurred_at": payload["occurred_at"]} if payload.get("occurred_at") else {}
    if isinstance(payload.get("answer"), Mapping):
        metadata["answer"] = dict(payload["answer"])   # the message's own evidence, for the ingest read
    body: dict[str, Any] = {
        "identity": {"host_id": "hermes"},
        "context": {"session_id": payload["session_id"], "contact_id": contact,
                    "turn_id": payload.get("turn_id") or None, "channel_id": channel,
                    "metadata": metadata or None},
    }
    if payload.get("checkpoint_messages") is not None:
        body["checkpoint_messages"] = list(payload["checkpoint_messages"])
    else:
        if sender:
            body["sender"] = {"platform": platform or "unknown", "user_id": sender}
        if payload.get("user_message"):
            body["user_message"] = {"role": "user", "content": payload["user_message"]}
        if payload.get("assistant_message"):
            body["assistant_message"] = {"role": "assistant", "content": payload["assistant_message"]}
            if payload.get("hand_refs"):
                body["assistant_message"]["metadata"] = {"hand_refs": payload["hand_refs"]}
        if payload.get("model"):
            body["model"] = payload["model"]
    if isinstance(payload.get("work"), Mapping):
        body["work"] = dict(payload["work"])
    if payload.get("transport_version") == 2:
        from urllib.parse import quote
        response = client.put("/v2/host/turns/" + quote(str(payload["turn_id"]), safe=""), json=body, timeout=8)
    else:
        response = client.post("/v1/host/turns/sync", json=body, timeout=8)
    if response.status_code == 403 and refusal_code(response) in ATTRIBUTION_REFUSALS:
        _refused(body, platform, refusal_code(response))
    if response.status_code in {400, 404, 409, 413, 422}:
        raise Undeliverable(f"HTTP {response.status_code}")
    if not response.is_success:
        return False
    try:
        result = response.json()
    except ValueError:
        if "work" in body:
            raise Undeliverable("work transport unsupported") from None
        return True
    if result.get("accepted") is False and result.get("skipped_reason") in ATTRIBUTION_REFUSALS:
        _refused(body, platform, str(result["skipped_reason"]))    # the sender resolved to someone it may not be
    if result.get("accepted") is False and result.get("skipped_reason") == "source_erased":
        raise Erased()    # final: every later send would get the same answer
    if "work" in body and result.get("accepted") is not False:
        recorded = result.get("work_recorded")
        if isinstance(recorded, bool):
            TurnOutbox(settings.outbox_path).work_admission(str(payload["turn_id"]), recorded)
            return True
        raise Undeliverable("work transport unsupported")
    return result.get("accepted") is not False


def _refused(body: Mapping[str, Any], platform: str, code: str) -> None:
    """A failed capture (docs/CREDENTIALS.md): logged, kept on its row, never sent again as another contact."""
    logger.warning("turn %s not captured: the sidecar refused it (%s) from platform %r; it is not sent again as "
                   "anyone else (grant the platform with 'colonymind credentials set' to capture such turns)",
                   body["context"]["turn_id"], code, platform)
    raise Undeliverable(REFUSED_PREFIX + code)


def refusal_code(response: Any) -> str:
    """The ``detail.code`` of a sidecar refusal, or ""."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return ""
    return str(detail.get("code") or "") if isinstance(detail, dict) else ""


__all__ = ["Capture", "Erased", "Heard", "INTERNAL_PLATFORMS", "SessionInfo", "SessionMap", "TAPBACK_PLATFORMS", "TurnOutbox",
           "Undeliverable", "arrival_of", "checkpoint", "deliver", "owner_input_text", "refusal_code", "tapback_reaction", "text_of",
           "whatsapp_reaction"]
