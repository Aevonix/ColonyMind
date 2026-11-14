"""The owner's answers to asks, on the host's side (docs/MIND.md, Asks; docs/HERMES-ADAPTER.md).

The sidecar decides what an owner message answers (``POST /v1/mind/answers``): which ask it is bound to, by
structure, and what its words say, by fixed tables. The plugin brings what only the host can see, before the model
runs, and nothing else:

- the owner's own message as the gateway received it (``SessionMap.hear``: its arrival time, its id, what it replies
  or reacts to; never its words at that point, since a secret is sealed only after the gateway hears it) and the
  owner's words as the turn carries them;
- what reached the owner's chat before it (``ChatLedger``): the body's sends with their outbox ids, the model's
  final replies, interim messages, the model's own sends to the owner and every cron run's reply (a cron run can
  deliver to any of the owner's chats, so it breaks all of them), and a turn under way in the chat when the message
  arrived: what a turn streams into the chat can pass no hook (a live commentary message on stock Hermes), so a
  turn that had started and not yet replied counts as having said something just before the message.

It asks only in the owner's own direct chat through a gateway, only for a genuine inbound message (the turn matched
the gateway's own record of it, by its words and its platform message id: a background completion, a redirect line or
an injected turn has none and is never read as an answer), and only when an ask is open or the message replies or
reacts to something. The read has ``READ_SECONDS`` in all, one deadline for every call it makes (the sender's
resolution, the open asks' state, the answer route); past that the model runs without it and the sidecar reads the
captured turn at ingest, by explicit references only. A reaction on an ask's own message is read as the gateway
receives it and takes no model turn. The model is told what the message did (``{"context": ...}``) and never
approves anything: ``colonymind_self show`` only asks the sidecar to put an open ask in front of the owner again
(``POST /v1/mind/asks/{id}/show``), as its own message in the sidecar's own words, every term a yes grants in full.
The plugin never words an ask: what it lists for the model is a title, and a title shown to the owner could stand
for terms they never saw. An ask's notice, a shown ask or a clarification waits for any turn under way in its
target chat to reply (``Answers.hold``), so it lands last and apart.

The constants below are heuristic defaults, not measured rules: see each constant.
"""

from __future__ import annotations

from datetime import datetime, timezone
from colonymind_shared.midturn import member_turn_id
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
from urllib.parse import quote

from .capture import Heard, SessionInfo, SessionMap, ingress_verified, owner_input_text
from .client import MIND_STATE_ROUTE, ColonyMindClient, Settings, SidecarUnavailable, acting

logger = logging.getLogger(__name__)

ANSWERS_ROUTE = "/v1/mind/answers"
SHOW_ROUTE = "/v1/mind/asks/{}/show"
READ_SECONDS = 1.5            # the plan's budget for the read before the model runs, every call in it together
CALL_SECONDS = 0.05           # my choice: with less of the budget left than this, no call is made
STATE_SECONDS = 30.0          # the plan's bound on how old the cached count of open asks may be
AFTER_REPLY_SECONDS = 5.0     # the plan's gap between a turn's reply and a message that turn caused
LEDGER_SECONDS = 48 * 3600.0  # my choice: twice the sidecar's longest window (a list's numbers mean it a day)
LEDGER_ENTRIES = 100          # my choice: the latest entries a read sends (the route takes at most 200)
AFTER_ARRIVAL_SECONDS = 10.0  # my choice: entries this soon after the arrival go too (the sidecar orders sends)
TURN_OPEN_SECONDS = 600.0     # my choice: a turn with no reply after this no longer holds what it caused
UNDER_WAY_LEAD = 0.001        # a turn under way at a message's arrival is dated this just before it
TEXT_CHARS = 8000             # the route's bound on the owner's words: a longer message is never cut to fit
# Ask messages go only after any turn under way in the target chat has replied (``hold``).
HELD_TYPES = frozenset({"ask_notice", "ask_show", "ask_clarify"})
# What reached a chat (sent to the sidecar with a read), what a turn started, and a removed invented instruction.
# A turn under way when a message arrived goes to the sidecar as one ``turn`` entry just before it (``ChatLedger.seen``).
OUTBOX, REPLY, INTERIM, MODEL_SEND, CRON = "outbox", "reply", "interim", "model_send", "cron"
SEEN_KINDS = (OUTBOX, REPLY, INTERIM, MODEL_SEND, CRON)
TURN, INVENTED = "turn", "invented_reference"
EVERY_CHAT = "*"
AUTOMATED_PLATFORMS = frozenset({"cron"})
# A line of the model's reply that tells the owner to answer with a code (asks carry none since 1.12): a reply word
# and a yes or no followed by a code-shaped token as asks spell them, or "with the code".
_CODE = r"[A-HJKMNP-Z2-9]{3}"
CODE_LINE = re.compile(rf"(?i:\b(?:reply|answer|say|type|respond|text|send|confirm)\b.*\b(?:yes|no)\b\W{{0,3}})"
                       rf"{_CODE}\b|(?i:\bwith (?:the|its|this) code\b)")
NOT_READ = ("an ask is answered in the owner's own message, read before you reply; no tool call answers one, and "
            "never tell the owner to reply with a code")


def chat_key(platform: Any, chat: Any) -> str:
    """One chat in the ledger: ``platform:chat id`` (a chat id is unique only on its platform)."""
    platform, chat = str(platform or "").strip().lower(), str(chat or "").strip()
    return f"{platform}:{chat}" if platform and chat else ""


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _left(deadline: float) -> float | None:
    """What is left of a read's budget for its next call, or None when too little is left to make one."""
    left = deadline - time.monotonic()
    return left if left >= CALL_SECONDS else None


def _epoch(value: Any) -> float:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return time.time()
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()


class ChatLedger:
    """What reached each owner chat, as the plugin saw it: one SQLite file beside the body's send ledger, shared by
    every Hermes process of the profile (a worker's own send to the owner counts as much as the gateway's). It keeps
    kinds, times and ids only, never words."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()
        self._ready = False
        self._writes = 0

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            if not self._ready:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not self.path.exists():
                    self.path.touch(mode=0o600)
                os.chmod(self.path, 0o600)
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        with self._lock:
            if not self._ready:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS chat_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, chat TEXT NOT NULL, at REAL NOT NULL,
                        kind TEXT NOT NULL, outbox_id TEXT NOT NULL DEFAULT '',
                        message_id TEXT NOT NULL DEFAULT '', ordered INTEGER NOT NULL DEFAULT 0)
                """)
                connection.execute("CREATE INDEX IF NOT EXISTS chat_events_chat ON chat_events (chat, at)")
                self._ready = True
        return connection

    def record(self, chat: str, kind: str, *, at: float | None = None, outbox_id: str = "", message_id: str = "",
               ordered: bool = False) -> None:
        if not chat:
            return
        now = time.time()
        connection = self._connect()
        try:
            connection.execute("INSERT INTO chat_events (chat, at, kind, outbox_id, message_id, ordered) "
                               "VALUES (?, ?, ?, ?, ?, ?)", (chat, now if at is None else float(at), kind,
                                                             str(outbox_id or ""), str(message_id or ""),
                                                             1 if ordered else 0))
            self._writes += 1
            if self._writes % 200 == 0:
                connection.execute("DELETE FROM chat_events WHERE at < ?", (now - LEDGER_SECONDS,))
        finally:
            connection.close()

    @staticmethod
    def _chats(chat: str) -> tuple[str, str, str]:
        """The chat, every chat of its platform (a send named only by its platform) and every chat (a cron run)."""
        return chat, chat.split(":", 1)[0] + ":" + EVERY_CHAT, EVERY_CHAT

    def seen(self, chat: str, *, until: float) -> list[dict[str, Any]]:
        """What reached ``chat`` up to ``until`` (and ``AFTER_ARRIVAL_SECONDS`` past it), oldest first, as the answer
        route reads it: ``{at, kind, outbox_id?, message_id?, ordered}``. When a turn was under way in the chat at
        ``until`` (``under_way``), one ``turn`` entry just before it stands for what that turn may have streamed
        past every hook; a turn that replied before ``until`` is its reply."""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT at, kind, outbox_id, message_id, ordered FROM chat_events WHERE chat IN (?, ?, ?) "
                f"AND kind IN ({', '.join('?' for _ in SEEN_KINDS)}) AND at >= ? AND at <= ? "
                "ORDER BY at DESC, id DESC LIMIT ?",
                (*self._chats(chat), *SEEN_KINDS, until - LEDGER_SECONDS, until + AFTER_ARRIVAL_SECONDS,
                 LEDGER_ENTRIES)).fetchall()
        finally:
            connection.close()
        rows = list(reversed(rows))
        if self.under_way(chat, until):
            rows = sorted([*rows, (until - UNDER_WAY_LEAD, TURN, "", "", 0)], key=lambda row: row[0])
        entries = []
        for at, kind, outbox_id, message_id, ordered in rows:
            entry: dict[str, Any] = {"at": _iso(at), "kind": kind, "ordered": bool(ordered)}
            if outbox_id:
                entry["outbox_id"] = outbox_id
            if message_id:
                entry["message_id"] = message_id
            entries.append(entry)
        return entries

    def _last(self, chat: str, kind: str) -> float:
        connection = self._connect()
        try:
            row = connection.execute("SELECT MAX(at) FROM chat_events WHERE chat = ? AND kind = ?",
                                     (chat, kind)).fetchone()
        finally:
            connection.close()
        return float(row[0]) if row and row[0] is not None else 0.0

    def under_way(self, chat: str, at: float) -> bool:
        """Whether a turn in ``chat`` had started before ``at`` and not replied by then (for at most
        ``TURN_OPEN_SECONDS``: a turn with no reply after that ended without one)."""
        connection = self._connect()
        try:
            started = connection.execute("SELECT MAX(at) FROM chat_events WHERE chat = ? AND kind = ? AND at < ?",
                                         (chat, TURN, at)).fetchone()[0]
            if started is None or at - float(started) > TURN_OPEN_SECONDS:
                return False
            replied = connection.execute("SELECT 1 FROM chat_events WHERE chat = ? AND kind = ? AND at > ? "
                                         "AND at <= ? LIMIT 1", (chat, REPLY, float(started), at)).fetchone()
        finally:
            connection.close()
        return replied is None

    def sent_since(self, chat: str, since: float) -> bool:
        """Whether the body put a message into ``chat`` after ``since``."""
        connection = self._connect()
        try:
            row = connection.execute("SELECT 1 FROM chat_events WHERE chat = ? AND kind = ? AND at > ? "
                                     "LIMIT 1", (chat, OUTBOX, since)).fetchone()
        finally:
            connection.close()
        return row is not None

    def busy(self, chat: str, now: float) -> bool:
        """A turn in ``chat`` has no reply yet (for at most ``TURN_OPEN_SECONDS``), or its reply was handed to the
        gateway less than ``AFTER_REPLY_SECONDS`` ago."""
        started, replied = self._last(chat, TURN), self._last(chat, REPLY)
        return (started > replied and now - started < TURN_OPEN_SECONDS) or now - replied < AFTER_REPLY_SECONDS

    def count(self, kind: str) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM chat_events WHERE kind = ?", (kind,)).fetchone()[0])
        finally:
            connection.close()


class Answers:
    """The plugin's half of the answer route: the read before the model runs, the ledger's hooks, the sidecar's
    shows and the hold on what a turn caused."""

    def __init__(self, client: ColonyMindClient, sessions: SessionMap, settings: Settings, ledger: ChatLedger, *,
                 wake: Callable[[], Any] | None = None, limit: int = 512):
        self.client, self.sessions, self.settings, self.ledger = client, sessions, settings, ledger
        self.wake, self.limit = wake, limit
        self._lock = threading.Lock()
        self._state: tuple[float, int] | None = None
        self._reads: dict[str, dict[str, Any]] = {}      # session -> what its last owner message did
        self._listed: dict[str, list[str]] = {}          # session -> the asks ``asks`` listed, in that order
        self.not_inbound = 0                             # owner turns not read: no inbound message matched them

    # -- where the owner is -------------------------------------------------------------------------------------

    def owner_chat(self, session_id: str, *, deadline: float | None = None) -> tuple[SessionInfo, str] | None:
        """``(session, chat)`` when the session is the owner's own direct chat through a gateway; never a guest's, a
        group, a kanban worker, a cron run, a delegated child or a lane with no chat (the CLI). ``deadline``
        (``time.monotonic``) bounds the sender's resolution for a read; past it the session is not the owner's."""
        if os.environ.get("HERMES_KANBAN_TASK"):
            return None
        info = self.sessions.get(session_id)
        if info is None or info.parent_session_id or not info.sender_id:
            return None
        if info.platform.strip().lower() in AUTOMATED_PLATFORMS:
            return None
        chat_id, chat_type = self.sessions.chat(session_id)
        if chat_type.strip().lower() != "dm" or not chat_id:
            return None
        if deadline is None:
            owner = self.sessions.is_owner(session_id)
        else:
            left = _left(deadline)
            owner = self.sessions.is_owner(session_id, timeout=left) if left is not None else None
        return (info, chat_key(info.platform, chat_id)) if owner is True else None

    def _open(self, chat: str, deadline: float) -> bool:
        """Whether an ask is open: the sidecar's count at most ``STATE_SECONDS`` old, read again (within what is left
        of the read's budget) when a message reached this chat since (an ask sent a moment ago). A sidecar without
        the mind's routes has none, for as long as a count is kept."""
        with self._lock:
            cached = self._state
        if cached is None or time.time() - cached[0] > STATE_SECONDS or self.ledger.sent_since(chat, cached[0]):
            left = _left(deadline)
            if left is None:
                return False
            try:
                response = self.client.get(MIND_STATE_ROUTE, timeout=left)
                if response.status_code == 404:
                    state: Any = {"open_asks": 0}      # a sidecar without the mind's routes: nothing is open
                else:
                    state = response.json() if response.is_success else None
            except (SidecarUnavailable, ValueError) as error:
                logger.info("the open asks were not read in time (%s)", type(error).__name__)
                return False
            if not isinstance(state, dict):
                return False
            asks = state.get("asks")
            count = state.get("open_asks")
            count = int(count) if isinstance(count, int) else len(asks) if isinstance(asks, list) else 0
            cached = (time.time(), count)
            with self._lock:
                self._state = cached
        return cached[1] > 0

    def _post(self, body: Mapping[str, Any], deadline: float) -> dict[str, Any] | None:
        """``POST /v1/mind/answers`` within what is left of the read's budget; None when it did not answer in time
        (the ingest read then takes the message), refused, or the sidecar has no such route."""
        left = _left(deadline)
        if left is None:
            logger.info("owner message left to the ingest read (the read's budget ran out)")
            return None
        try:
            response = self.client.post(ANSWERS_ROUTE, json=dict(body), timeout=left)
        except SidecarUnavailable as error:
            logger.info("owner message left to the ingest read (%s)", error)
            return None
        if not response.is_success:
            logger.info("the answer route answered HTTP %s", response.status_code)
            return None
        try:
            value = response.json()
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def _request(self, info: SessionInfo, chat: str, heard: Heard | None, turn_id: str, text: str) -> dict[str, Any]:
        """The owner's message as the answer route takes it: the platform, the sender (the sidecar resolves it again),
        the chat, the turn, the owner's own words and, when the gateway's record of the message matched, its own
        evidence and the chat's ledger as of its arrival. Without the record: no evidence and no ledger."""
        body: dict[str, Any] = {"platform": info.platform.strip().lower(), "sender_id": info.sender_id,
                                "chat_id": chat.split(":", 1)[1], "chat_type": "dm",
                                "turn_id": str(turn_id or info.turn_id or "") or None,
                                "text": text}
        if heard is None:
            body["arrived_at"] = _iso(time.time())
            return body
        evidence = {key: value for key, value in heard.evidence().items() if key not in {"chat_id", "chat_type"}}
        body.update({key: (value if value != "" else None) for key, value in evidence.items()})
        body["ledger"] = self.ledger.seen(chat, until=_epoch(heard.arrived_at))
        return body

    # -- hooks --------------------------------------------------------------------------------------------------

    def pre_llm_call(self, *, session_id: str = "", turn_id: str = "", **_: Any) -> dict[str, str] | None:
        """In the owner's own direct chat: take the chat's ledger as of the message's arrival, note that this turn is
        under way (what it causes waits for its reply; a later message counts it), then hand the owner's message to
        the answer route when an ask is open or the message replies or reacts to something, and give the model what
        it did. Registered after ``Capture.pre_llm_call``. Never raises."""
        try:
            return acting(session_id, self._pre_llm_call, str(session_id or ""), str(turn_id or ""))
        except Exception as error:  # the read never disturbs the turn
            logger.warning("owner message not read as an answer (%s)", type(error).__name__)
            return None

    def _pre_llm_call(self, session_id: str, turn_id: str) -> dict[str, str] | None:
        with self._lock:
            self._reads.pop(session_id, None)
        # Capture has already given the front this turn's structural history. A verified completion has no
        # owner answer to read. Keep the local ledger's protection against ambiguous streamed replies without
        # resolving a contact or asking the sidecar to read synthetic words.
        review = False
        if session_id and turn_id:
            try:
                bridge = sys.modules.get("colonymind_hands_front")
                review = bridge is not None and bridge.is_review_turn(session_id, turn_id) is True
            except Exception:
                pass
        if review:
            if not self._silent_hold_review(session_id, turn_id):
                self._record_review_turn(session_id, turn_id)
            return None
        deadline = time.monotonic() + READ_SECONDS
        where = self.owner_chat(session_id, deadline=deadline)
        if where is None:
            return None
        info, chat = where
        members = getattr(info, "members", ())
        bodies = []
        try:
            for member in members or (info,):
                body = self._to_read(member, chat, turn_id, deadline, session_id)
                if body is None:
                    continue
                if members and member is not members[-1]:
                    body["turn_id"] = member_turn_id(turn_id or info.turn_id, member.heard.message_id)
                bodies.append(body)
        finally:
            self.ledger.record(chat, TURN)    # all arrival snapshots precede this one model turn
        results = []
        for body in bodies:
            result = self._read_one(body, deadline, session_id)
            if result is not None:
                results.append(result)
        if not results:
            return None
        result = {**results[-1], "context": "\n".join(str(row.get("context") or "") for row in results)}
        with self._lock:
            self._reads[session_id] = result
            while len(self._reads) > self.limit:
                self._reads.pop(next(iter(self._reads)))
        context = str(result.get("context") or "").strip()
        return {"context": context} if context else None

    def _read_one(self, body, deadline, session_id):
        result = self._post(body, deadline)
        if result is None:
            return None
        grants = result.get('granted_hands')
        approved = result.get('approved') or []
        if grants or approved:
            bridge = sys.modules.get('colonymind_hands_front')
            resume = getattr(bridge, 'resume_approvals', None)
            if callable(resume):
                try:
                    line = resume(session_id, grants or [], approved=approved)
                    if line:
                        result = {**result, 'context': (str(result.get('context') or '') + '\n' + line).strip()}
                except Exception:
                    logger.exception('Hand resumption will retry from the stored grant')
        return result

    def _record_review_turn(self, session_id: str, turn_id: str) -> None:
        """Record only the owner DM that capture already bound to this exact review, with no sidecar work."""
        info = self.sessions.get(session_id)
        if (info is None or info.turn_id != turn_id or info.owner is not True or info.parent_session_id
                or not info.sender_id or info.platform.strip().lower() in AUTOMATED_PLATFORMS
                or info.chat_type.strip().lower() != "dm" or not info.chat_id
                or os.environ.get("HERMES_KANBAN_TASK")):
            return
        self.ledger.record(chat_key(info.platform, info.chat_id), TURN)

    @staticmethod
    def _silent_hold_review(session_id: str, turn_id: str) -> bool:
        """Only the front's exact, structurally verified local settlement is silent."""
        try:
            bridge = sys.modules.get("colonymind_hands_front")
            check = getattr(bridge, "is_silent_hold_review", None)
            return callable(check) and check(session_id, turn_id) is True
        except Exception:
            return False

    def _to_read(self, info: SessionInfo, chat: str, turn_id: str, deadline: float,
                 session_id: str = "") -> dict[str, Any] | None:
        """The answer route's request for this turn's message, or None when it is not read before the model."""
        words = owner_input_text(session_id or info.session_id, turn_id or info.turn_id, info.user_message,
                                 legacy_text=info.said or info.user_message,
                                 own_words=info.said if info.heard is not None else None)
        if words is None:
            return None
        heard = info.heard
        if not ingress_verified(info, session_id or info.session_id, turn_id or info.turn_id):
            # Ingress: only a message the gateway received from the owner answers. A background delegation's
            # completion, a redirect or steer line and an injected or synthetic turn reach the session as user-role
            # text that no inbound event carried, so no record of the gateway's matches them (``capture.ingress_verified``).
            with self._lock:
                self.not_inbound += 1
            logger.info("owner turn not read as an answer: no inbound message of the gateway matches it")
            return None
        if len(words) > TEXT_CHARS:
            # Never read from part of the owner's words: a refusal past the cut would be lost. The ingest read has
            # the whole message, with its own evidence.
            return None
        if not (heard is not None and (heard.reply_to or heard.reacts)) and not self._open(chat, deadline):
            return None
        return self._request(info, chat, heard, turn_id, words)

    def post_llm_call(self, *, session_id: str = "", platform: str = "", turn_id: str = "",
                      assistant_response: str = "", **_: Any) -> None:
        """The turn's reply was handed to the gateway: note it in the owner's chat (a cron run's reply in every
        owner chat), and wake the body ``AFTER_REPLY_SECONDS`` later for an ask's notice, a show or a
        clarification. Never raises."""
        try:
            if (assistant_response == "[SILENT]" and self._silent_hold_review(session_id, turn_id)):
                return None
            self._post_llm_call(str(session_id or ""), str(platform or ""))
        except Exception as error:
            logger.warning("the chat ledger missed a reply (%s)", type(error).__name__)
        return None

    def _post_llm_call(self, session_id: str, platform: str) -> None:
        if os.environ.get("HERMES_KANBAN_TASK"):
            return
        info = self.sessions.get(session_id)
        if platform.strip().lower() in AUTOMATED_PLATFORMS or (
                info is not None and info.platform.strip().lower() in AUTOMATED_PLATFORMS):
            self.ledger.record(EVERY_CHAT, CRON)       # where a cron run delivers is not visible here
            return
        where = self.owner_chat(session_id)
        if where is None:
            return
        self.ledger.record(where[1], REPLY)
        if self.wake is not None:
            self._later(self.wake, extra=0.2)        # what the turn queued goes as soon as it may

    @staticmethod
    def _later(action: Callable[[], Any], *, extra: float = 0.0) -> None:
        timer = threading.Timer(AFTER_REPLY_SECONDS + extra, action)
        timer.daemon = True
        timer.start()

    def on_interim_message(self, *, session_id: str = "", **_: Any) -> None:
        """An interim message reached the chat mid-turn: it breaks the follow-up like any other."""
        try:
            where = self.owner_chat(str(session_id or ""))
            if where is not None:
                self.ledger.record(where[1], INTERIM)
        except Exception as error:
            logger.debug("the chat ledger missed an interim message (%s)", type(error).__name__)
        return None

    def post_tool_call(self, *, tool_name: str = "", args: Any = None, result: Any = None, **_: Any) -> None:
        """The model's own ``send_message`` to a chat (from any session, a worker's included): it breaks the
        follow-up there. A send to the owner breaks every owner chat on that platform; one named only by its
        platform (a home channel) every chat on it."""
        if tool_name != "send_message" or not isinstance(args, Mapping):
            return None
        try:
            if str(args.get("action") or "send") == "list" or not self._succeeded(result):
                return None
            platform, _, chat = str(args.get("target") or "").strip().partition(":")
            platform = platform.strip().lower()
            if not platform:
                return None
            if not chat.strip() or self.sessions.sender_is_owner(platform, chat.strip()):
                self.ledger.record(f"{platform}:{EVERY_CHAT}", MODEL_SEND)
            else:
                self.ledger.record(chat_key(platform, chat), MODEL_SEND)
        except Exception as error:
            logger.debug("the chat ledger missed a model send (%s)", type(error).__name__)
        return None

    @staticmethod
    def _succeeded(result: Any) -> bool:
        """Whether a stock send reported success; an answer it cannot read counts (it may have reached the chat)."""
        try:
            value = json.loads(result) if isinstance(result, str) else result
        except ValueError:
            return True
        if isinstance(value, Mapping) and ("success" in value or "partial_success" in value or "error" in value):
            return bool(value.get("success") or value.get("partial_success"))
        return True

    def transform_llm_output(self, *, response_text: Any = None, session_id: str = "", **_: Any) -> str | None:
        """In the owner's own chat, remove a line of the reply that tells the owner to answer with a code (asks
        carry none; the model invented it), and count it. Best effort: Hermes takes the first transform that
        returns text, so another plugin's can pre-empt this one, and nothing else depends on it."""
        text = response_text if isinstance(response_text, str) else ""
        try:
            if not text or not CODE_LINE.search(text) or self.owner_chat(str(session_id or "")) is None:
                return None
            lines = text.split("\n")
            kept = [line for line in lines if not CODE_LINE.search(line)]
            removed = len(lines) - len(kept)
            chat = self.owner_chat(str(session_id or ""))[1]
            for _ in range(removed):
                self.ledger.record(chat, INVENTED)
            logger.warning("invented_ask_reference: %d line(s) telling the owner to reply with a code removed",
                           removed)
            return "\n".join(kept).strip() or None
        except Exception as error:
            logger.debug("reply not checked for code instructions (%s)", type(error).__name__)
            return None

    # -- reactions ----------------------------------------------------------------------------------------------

    def read_reaction(self, heard: Heard | None) -> bool:
        """A reaction the owner put on a message in their own direct chat, read as the gateway receives it (a tap
        takes no model turn when it answered an ask), within ``READ_SECONDS`` in all. True when the sidecar bound it
        to an ask or it settled or undid one: the caller then drops the event."""
        deadline = time.monotonic() + READ_SECONDS
        if heard is None or not heard.reacts or heard.chat_type != "dm" or heard.own_account:
            return False
        left = _left(deadline)
        if left is None or not self.sessions.sender_is_owner(heard.platform, heard.sender, timeout=left):
            return False
        chat = chat_key(heard.platform, heard.chat_id)
        body: dict[str, Any] = {"platform": heard.platform, "sender_id": heard.sender, "chat_id": heard.chat_id,
                                "chat_type": "dm", "text": ""}
        evidence = {key: value for key, value in heard.evidence().items() if key not in {"chat_id", "chat_type"}}
        body.update({key: (value if value != "" else None) for key, value in evidence.items()})
        body["ledger"] = self.ledger.seen(chat, until=_epoch(heard.arrived_at))
        result = self._post(body, deadline) or {}
        return bool(result.get("path") == "reaction" or result.get("settled") or result.get("undone"))

    def gateway_platform_event(self, *, platform: str = "", event_type: str = "", payload: Any = None,
                               **_: Any) -> None:
        """Stock ``gateway_platform_event``: a Telegram reaction. The event names only its chat and the message; a
        direct chat's id is the person's own (a group's or a channel's is negative), so the reactor of a reaction in
        a direct chat is that chat's person."""
        try:
            if str(platform or "") != "telegram" or event_type != "reaction" or not isinstance(payload, Mapping):
                return None
            chat, target = str(payload.get("chat_id") or "").strip(), str(payload.get("message_id") or "").strip()
            if not chat or not target or chat.startswith("-"):
                return None
            emojis = [item for item in payload.get("emojis") or [] if isinstance(item, str) and item]
            emoji = emojis[0] if emojis else ""
            self.read_reaction(Heard(
                platform="telegram", sender=chat, chat_id=chat, chat_type="dm",
                message_id=f"reaction:{target}:{emoji or 'removed'}:{time.time_ns()}", arrived_at=_iso(time.time()),
                reply_to=target, reaction=emoji, reaction_removed=not emoji, heard_at=time.time()))
        except Exception as error:
            logger.debug("reaction not read (%s)", type(error).__name__)
        return None

    # -- the body -----------------------------------------------------------------------------------------------

    def hold(self, message: Mapping[str, Any], target: str) -> bool:
        """``Body.send_outbox``: an ask's notice, a shown ask or a clarification waits while a turn in the
        target chat has no reply (for at most ``TURN_OPEN_SECONDS``), and
        ``AFTER_REPLY_SECONDS`` after it, so it lands last and apart from the reply.
        A hand notice first waits for its native parent review and gap, bounded
        by the controller's fixed deadline. A later show keeps its own reply gap."""
        if str(message.get("type") or "") not in HELD_TYPES:
            return False
        platform, _, chat = str(target or "").partition(":")
        key = chat_key(platform, chat)
        try:
            now = time.time()
            waits = message.get("hand_approval_waits") or []
            if waits:
                bridge = sys.modules.get("colonymind_hands_front")
                ended = getattr(bridge, "approval_review_ended", None)
                for wait in waits:
                    if now >= _epoch(wait["deadline"]):
                        continue
                    reviewed = ended(wait["hand_id"]) if callable(ended) else None
                    if reviewed is None or reviewed < _epoch(wait["held_at"]) or now - reviewed < AFTER_REPLY_SECONDS:
                        return True
                if message.get("type") == "ask_notice":
                    # The native settlement and gap, or the fixed cap, finish
                    # this wait even when a silent review left no chat reply.
                    return False
            return bool(key) and self.ledger.busy(key, now)
        except Exception as error:     # an unreadable ledger never keeps a message from the owner
            logger.warning("the chat ledger could not be read (%s); the message goes", type(error).__name__)
            return False

    def sent(self, message: Mapping[str, Any], target: str, delivery_id: str | None) -> None:
        """``Body.send_outbox``: a message the body delivered, in the order it sent them, into the ledger of the
        chat it reached (the sidecar reads which asks it showed from its own row)."""
        platform, _, chat = str(target or "").partition(":")
        self.ledger.record(chat_key(platform, chat), OUTBOX, outbox_id=str(message.get("id") or ""),
                           message_id=str(delivery_id or ""), ordered=True)

    # -- the model's view: open asks in words, and showing one ----------------------------------------------------

    def open_asks(self) -> list[dict[str, Any]] | None:
        """The open asks from the sidecar's state, in its order, with no code: ``{id, ask, message?}``; None when
        the sidecar is unreachable. An owner question (answered in words, no code) is not listed. This is the
        model's view, a title: it is never what the owner is shown (``show``)."""
        state = self.client.mind_state()
        if state is None:
            return None
        asks = []
        for item in state.get("asks") or []:
            if not isinstance(item, Mapping) or not (item.get("code") or item.get("ask_code")) or not item.get("id"):
                continue
            entry = {"id": str(item["id"]), "ask": " ".join(str(item.get("title") or "an ask").split())}
            if item.get("message"):
                entry["message"] = str(item["message"])
            asks.append(entry)
        return asks

    def listed(self, session_id: str, asks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The asks numbered for the model (``n``), the numbers meaning this listing for ``show``."""
        with self._lock:
            self._listed[session_id] = [item["id"] for item in asks]
            while len(self._listed) > self.limit:
                self._listed.pop(next(iter(self._listed)))
        return [{"n": number, **{key: value for key, value in item.items() if key != "id"}}
                for number, item in enumerate(asks, 1)]

    def pick(self, session_id: str, number: int, asks: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Ask ``number`` of the last listing in this session (else of the current order), when still open."""
        with self._lock:
            order = list(self._listed.get(session_id) or [item["id"] for item in asks])
        if not 1 <= number <= len(order):
            return None
        return next((item for item in asks if item["id"] == order[number - 1]), None)

    def show(self, ask_id: str) -> str:
        """Ask the sidecar to put one open ask in front of the owner again (``POST /v1/mind/asks/{id}/show``): its
        own ``ask_show`` message in the owner's chat, in the sidecar's own words with every term a yes grants in
        full, which binds like the ask's own message. The body sends it after this turn's reply (``hold``).
        ``shown``; ``closed`` when the ask is no longer open; ``missing`` when this sidecar cannot show an ask
        again (the plugin then words nothing in its place: the ask's own message stays the one the owner answers);
        ``failed`` otherwise."""
        try:
            response = self.client.post(SHOW_ROUTE.format(quote(str(ask_id), safe="")), timeout=5)
        except SidecarUnavailable as error:
            logger.info("an ask was not shown (%s)", error)
            return "failed"
        if response.status_code in {404, 405}:
            return "closed" if _names_an_ask(response) else "missing"
        if not response.is_success:
            logger.info("the sidecar did not show an ask (HTTP %s)", response.status_code)
            return "failed"
        return "shown"

    def last_read(self, session_id: str) -> str:
        """What the answer route made of this turn's owner message, for a model that still calls ``yes`` or
        ``no``."""
        with self._lock:
            result = self._reads.get(session_id)
        if not result:
            return NOT_READ
        context = str(result.get("context") or "").strip()
        return f"{NOT_READ}. This turn's message was read already: {context}" if context else \
            f"{NOT_READ}. This turn's message answered no open ask."


def _names_an_ask(response: Any) -> bool:
    """Whether a 404 is the sidecar's own "no open ask" (the route exists) rather than a route it does not serve."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return False
    return isinstance(detail, Mapping) and detail.get("code") == "unknown_ask"


def ledger_path(settings: Settings) -> Path:
    """The chat ledger beside the body's send ledger."""
    base = settings.body_ledger_path or settings.outbox_path.with_name("colonymind-body.sqlite3")
    return Path(base).with_name("colonymind-chats.sqlite3")


__all__ = ["AFTER_REPLY_SECONDS", "ANSWERS_ROUTE", "Answers", "ChatLedger", "CODE_LINE", "HELD_TYPES", "NOT_READ",
           "READ_SECONDS", "chat_key", "ledger_path"]
