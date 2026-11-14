"""Inbound triage (``pre_gateway_dispatch``): does a non-owner's message need an answer from the assistant?

Shadow only in this release. The hook asks about a message and returns None at once, whatever the answer: it never
returns ``skip`` (a skipped message never reaches capture, so the person's words would be lost from memory) and
never ``rewrite`` or ``allow`` (Hermes reads the callbacks' answers in order and stops at the first rewrite or
allow, and ColonyMind's own callback must be free to seal a secret before Hermes sees it). The sidecar logs the
answer for measurement.

The owner is never asked about: one of the owner's handles is known at once, and a sender the sidecar resolves to
the owner is dropped before any request. Hermes calls the hook before its own admission check, so the triage asks
that check first (``admitted``): a message Hermes would refuse, a bot's message, or one whose gateway cannot say is
never queued. Internal events, the owner's platforms without a sender, chats outside ``chats`` (when set) and text
longer than the route reads are not asked about either. Asking happens on one
background thread with a bounded queue (a full queue drops the question), so dispatch never waits for it; a
detected secret is hidden first from every text sent: the message, the chat's title and the sender's name.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from . import config

logger = logging.getLogger(__name__)

POINT = "inbound_triage"
QUEUE_SIZE = 64
MAX_TEXT = 8000
INTERNAL_PLATFORMS = frozenset({"", "cli", "internal", "system", "owner", "api", "worker", "cron"})


def _platform(source: Any) -> str:
    platform = getattr(source, "platform", "")
    return str(getattr(platform, "value", platform) or "").strip().lower()


def admitted(gateway: Any, source: Any) -> bool:
    """Whether Hermes admits this message's sender, by Hermes' own check on the gateway the hook is handed (the hook
    runs before it). False when it cannot say (no gateway, no check, an error) and for a bot's message, which Hermes'
    loop guard counts on admission and could still refuse: then the triage abstains."""
    check = getattr(gateway, "_is_user_authorized_for_source", None)
    if not callable(check) or getattr(source, "is_bot", False) is True:
        return False
    try:
        return check(source) is True
    except Exception as error:
        logger.debug("triage could not ask Hermes' admission (%s)", type(error).__name__)
        return False


def session_key(gateway: Any, source: Any) -> str:
    """The key of the Hermes session this message lands in, as the gateway derives it; "" when it cannot say."""
    derive = getattr(gateway, "_session_key_for_source", None)
    try:
        key = derive(source) if callable(derive) else ""
    except Exception:
        return ""
    return key if isinstance(key, str) else ""


class Triage:
    def __init__(self, settings: Any, client: Any, sessions: Any, gate: config.TriageConfig):
        self.settings, self.client, self.sessions, self.gate = settings, client, sessions, gate
        self.queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.dropped = 0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def pre_gateway_dispatch(self, event: Any = None, gateway: Any = None, **_: Any) -> None:
        try:
            self._consider(event, gateway)
        except Exception as error:          # triage never disturbs dispatch
            logger.debug("triage skipped a message (%s)", type(error).__name__)
        return None

    def _consider(self, event: Any, gateway: Any = None) -> None:
        # ColonyMind's adapter runs this hook for a message parked behind a busy session and marks it; a late arrival
        # handed to the idle path afterwards is not triaged twice.
        if event is None or getattr(event, "internal", False) or getattr(event, "_colonymind_ingress_done", False) is True:
            return
        text = getattr(event, "text", None)
        source = getattr(event, "source", None)
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT or source is None:
            return
        platform, sender = _platform(source), str(getattr(source, "user_id", "") or "").strip()
        if not sender or platform in INTERNAL_PLATFORMS:
            return
        chat_id = str(getattr(source, "chat_id", "") or "")
        if self.gate.chats and chat_id not in self.gate.chats:
            return
        owner = {handle.lower() for handle in self.settings.owner_handles().get(platform, [])}
        if sender.lower() in owner or not admitted(gateway, source):
            return
        # Every text sent is read for secrets: the chat's title and the sender's name as well as the message.
        from colonymind_hermes.secrets import hide_secrets
        chat_type = str(getattr(source, "chat_type", "") or "dm").lower()
        name = hide_secrets(" ".join(str(getattr(source, "chat_name", "") or "").split()))
        chat = "Direct" if chat_type == "dm" else (f"Group '{name}'" if name else "Group")
        message = str(getattr(event, "message_id", "") or getattr(source, "message_id", "") or "")
        sender_name = hide_secrets(" ".join(str(getattr(source, "user_name", "") or "someone").split()))
        fields = {"chat": chat, "sender": sender_name, "text": hide_secrets(text)}
        # A message id is unique only within its chat: the references name the chat (and thread) and the Hermes
        # session the message lands in, so each decision points at one message and one Hermes record.
        thread = str(getattr(source, "thread_id", "") or "")
        refs = {key: value for key, value in (("platform", platform), ("chat", chat_id), ("thread", thread),
                                              ("message", message), ("session", session_key(gateway, source)))
                if value}
        self._submit((platform, sender, fields, refs))

    def _submit(self, item: tuple) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="colonymind-gates-triage", daemon=True)
                self._thread.start()
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            platform, sender, fields, refs = self.queue.get()
            try:
                if not self.sessions.sender_is_owner(platform, sender):
                    self.client.decide({"point": POINT, "event": "inbound", "fields": fields, "refs": refs},
                                       deadline_s=config.TRIAGE_DEADLINE_S)
            except Exception as error:
                logger.debug("triage ask failed (%s)", type(error).__name__)
            finally:
                self.queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait up to ``timeout`` seconds for the questions queued so far (tests, a clean stop)."""
        end = time.monotonic() + timeout
        while self.queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self.queue.unfinished_tasks


__all__ = ["INTERNAL_PLATFORMS", "POINT", "Triage", "admitted", "session_key"]
