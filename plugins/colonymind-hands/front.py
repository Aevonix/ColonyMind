"""Front deadlines, ingress provenance and small completion reviews.

W1-1: the watchdog reads pending queues. It never wraps or drains Hermes.
All durations and collection limits below are design assumptions.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import logging
import os
import tempfile
import re
import sys
import threading
import time
from types import ModuleType
from typing import Any

from colonymind_shared.midturn import bind_rewrite, composition, matches_rewrite, valid_ingress
from colonymind_shared.turn_bindings import TurnBindings

from .settlement import Settlement

logger = logging.getLogger(__name__)
# Review guidance asks for the reply Owner asked for, built from finished results.
# Only unfinished work goes to a hand.
YIELD_LINE = ('[runtime deadline, not from Owner] Time to reply: answer Owner now, in the form Owner asked for, '
              'using the results you already have, and hand only unfinished work to a hand.')
OWNER_WAITING_LINE = '[runtime deadline, not from Owner] An owner message is waiting. End this review now with [SILENT].'
SILENT = '[SILENT]'
# Assumption: scheduled retries of one dropped review. A retry that never starts a classified review must not
# repeat after every owner turn; the obligation then stays open with no review receipt.
REVIEW_RETRIES = 3
# Native per-tool timeout names verified at the pinned Hermes revision.
TIMEOUTS = {'terminal': 'timeout', 'process': 'timeout', 'browser_cdp': 'timeout', 'connections': 'timeout_seconds'}
# Pure lookups, which count as reads rather than writes, including tool discovery.
# Deliberately NOT here: terminal, because a shell command can do anything; skill_manage, which mutates;
# and colonymind_people, which reads like a lookup but carries set_permission and propose_link.
READ_TOOLS = {'read_file', 'file_read', 'search_files', 'list_files', 'glob', 'grep', 'memory_search',
              'colonymind_recall', 'colonymind_memory_search', 'web_search', 'web_extract',
              'tool_search', 'tool_describe', 'skill_view'}
UNBOUNDED_TOOLS = {'execute_code', 'web_search', 'web_extract'}
# How the front hands work off. A hand-off is never counted against a tool-call cap.
# Once the owner is waiting, a few hand-offs still go through, then the front must reply.
HAND_OFF_TOOLS = {'delegate_task'}
HAND_OFFS_AFTER_REPLY_NOW = 3
_BRIDGE = 'colonymind_hands_front'


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(str(_get(part, 'text', '')) for part in value if _get(part, 'type') in {'text', 'input_text', 'output_text'})
    return ''


def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _env(key):
    try:
        from gateway.session_context import get_session_env
        return str(get_session_env('HERMES_SESSION_' + key) or '')
    except (ImportError, AttributeError):
        return ''


def visible_text(content, agent=None):
    """Use Hermes' complete-string visibility rule, including unclosed blocks."""
    content = _text(content)
    native = getattr(agent, '_strip_think_blocks', None)
    if callable(native):
        return native(content).strip()
    try:
        from agent.agent_runtime_helpers import strip_think_blocks
        return strip_think_blocks(None, content).strip()
    except ImportError:
        pass
    try:
        from agent.think_scrubber import THINK_TAG_NAMES
    except ImportError:
        THINK_TAG_NAMES = ('think', 'thinking', 'reasoning', 'thought', 'REASONING_SCRATCHPAD')
    tags = '|'.join(re.escape(tag) for tag in THINK_TAG_NAMES)
    content = re.sub(r'<(' + tags + r')>.*?</\1>\s*', '', content, flags=re.I | re.S)
    content = re.sub(r'(?:^|\n)[ \t]*<(?:' + tags + r')\b[^>]*>.*$', '', content, flags=re.I | re.S)
    return re.sub(r'</?(?:' + tags + r')>\s*', '', content, flags=re.I).strip()


_UNSET = object()
# How long a message the busy path heard may wait in Hermes' queue-mode text debounce before the adapter's slot has it
# (HERMES_GATEWAY_BUSY_TEXT_DEBOUNCE_SECONDS, 0.35 s by default, and its bounded flush); an assumption, generous on
# purpose: a heard message that never reaches the slot (a hook dropped it) stops counting after this.
HEARD_WAIT_SECONDS = 3.0


@dataclass
class Inbound:
    platform: str
    sender: str
    chat: str
    message_id: str
    text_hash: str
    arrived: float
    session: str = ''
    session_key: str = ''
    consumed: tuple[str, str] | None = None
    stripped_hash: str = ''
    queued: bool = False   # heard on the busy path (busy_ingress): it runs later, as a queued follow-up
    mixed: bool = False    # another sender's message arrived behind it while it waited: its turn may carry both

    rewrite: Any = None

    def reference(self):
        if all((self.platform, self.chat, self.message_id)):
            return {'platform': self.platform, 'chat': self.chat, 'message_id': self.message_id}
        return None


@dataclass
class Turn:
    session: str
    turn: str
    started: float
    platform: str = ''
    sender: str = ''
    chat: str = ''
    session_key: str = ''
    generation: int | None = None
    agent: Any = None
    pending_binding: bool = False
    inbound: Inbound | None = None
    bound: bool = False
    composed: tuple = ()
    admissions: tuple = ()
    envelope: str = ""
    mode: str = 'answer_now'
    synthetic: list[str] = field(default_factory=list)
    visible: bool = False
    soft_sent: bool = False
    owner_waiting: bool = False
    owner_redirected: bool = False
    request_dropped: bool = False
    hold_only: bool = False
    review_message: str = ''
    finished: bool = False
    armed: bool = False
    duplicate: bool = False
    rendered: list[dict] = field(default_factory=list)
    completions: dict[str, str] = field(default_factory=dict)
    trigger_hands: set[str] = field(default_factory=set)
    reference: dict | None = None
    saved_callbacks: dict = field(default_factory=dict)
    calls: dict = field(default_factory=lambda: {'read': 0, 'write': 0})
    tool_ids: set = field(default_factory=set)
    context: str = ''
    final_response: str = ''
    outcome_recorded: bool = False
    outcome_current: bool = False
    early_exit_current: bool = False
    stopped: threading.Event = field(default_factory=threading.Event)
    thread: Any = None
    order: int = 0
    work_seconds: float = 0.0
    tool_started: float | None = None
    active_tools: set = field(default_factory=set)
    delivered_tool_ids: set = field(default_factory=set)
    deadlines_disabled: bool = False
    request_gate: Any = field(default_factory=threading.Lock)
    model_waiting: bool = False


@dataclass
class DeferredReview:
    message: str
    generation: int
    scheduled_after: str = ''
    attempts: int = 0
    after: int = 0
    scheduled: bool = False


class FrontGuard:
    def __init__(self, runtime, ctx, *, clock=time.monotonic, start_threads=True):
        self.runtime, self.ctx, self.clock = runtime, ctx, clock
        self.start_threads = start_threads
        self.lock = threading.RLock()
        self.turns = OrderedDict()
        self.inbound = OrderedDict()
        self.rendered = set()
        self.approval_reviews = {}
        settings = getattr(getattr(runtime, 'client', None), 'settings', None)
        home = (getattr(runtime, 'spool_dir', None) or getattr(settings, 'home', None)
                or getattr(settings, 'hermes_home', None))
        self._composed_bindings = TurnBindings(Path(home) / 'front-bindings.sqlite3' if home else None)
        self.settlement = Settlement(self.on_gateway_turn_settled, getattr(settings, 'hermes_home', None),
                                     prepare=self.before_gateway_turn_clear)
        self.deferred_reviews = {}
        self._order = itertools.count(1)
        self.metrics = {'capture_refused': 0, 'redirect_accepted': 0, 'redirect_refused': 0,
                        'injection_failed': 0, 'binding_missing': 0, 'review_request_dropped': 0,
                        'review_retry_abandoned': 0, 'deadline_yield': 0}
        self.gateway = None
        self.closed = False
        self._jobs = set()
        self.bridge = None
        self.timeout_inventory = dict(TIMEOUTS)
        self.unbounded_tools = set(UNBOUNDED_TOOLS)
        self._load_rendered()
        self.config = getattr(runtime, 'config', {}).get('front', {})

    def _receipt_path(self):
        directory = getattr(self.runtime, 'spool_dir', None)
        return Path(directory) / 'front-reviewed.json' if directory else None

    def _load_rendered(self):
        path = self._receipt_path()
        if path is None:
            return
        try:
            rows = json.loads(path.read_text())
            if isinstance(rows, list):
                self.rendered.update(tuple(row) for row in rows if isinstance(row, list) and len(row) == 3
                                     and isinstance(row[0], str) and isinstance(row[1], int)
                                     and isinstance(row[2], (str, type(None))))
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception('front review receipts unreadable; reviews remain eligible')

    def _save_rendered(self):
        path = self._receipt_path()
        if path is None:
            return
        temporary = None
        try:
            with self.lock:
                with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
                    temporary = stream.name
                    json.dump(sorted(self.rendered, key=str), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                temporary = None
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            if temporary:
                os.unlink(temporary)

    def install_bridge(self):
        bridge = self.bridge or ModuleType(_BRIDGE)
        for name in ('capture_owner_text', 'owner_started_turn', 'origin_ref', 'request_caps', 'observe_turn',
                     'is_review_turn', 'is_silent_hold_review', 'approval_review_ended',
                     'defer_review_request'):
            setattr(bridge, name, getattr(self, name))
        self.bridge = bridge
        resume = getattr(self.runtime, 'resume_approvals', None)
        if callable(resume):
            bridge.resume_approvals = resume
        sys.modules[_BRIDGE] = bridge
        self.runtime.ingress = self
        self.settlement.install(self._gateway())

    def _gateway(self):
        if self.gateway is not None:
            return self.gateway
        ref = getattr(self.runtime, 'gateway', None)
        return ref() if callable(ref) else ref

    def _inactive(self):
        closed = getattr(self.runtime, '_closed', None)
        if self.closed or (closed is not None and closed.is_set()):
            self.close()
            return True
        return False

    def _session_info(self, sid):
        reader = getattr(getattr(self.runtime, 'sessions', None), 'get', None)
        return reader(sid) if callable(reader) else None

    def _messaging(self, sid, platform=''):
        info = self._session_info(sid)
        platform = platform or _env('PLATFORM') or _get(info, 'platform', '')
        platform = str(_get(platform, 'value', platform) or '').lower()
        return platform not in {'cli', 'api', 'api_server', 'voice', 'cron', 'kanban', 'subagent', 'delegation'}

    def _managed(self, sid, platform=''):
        return bool(not self._inactive() and sid and self._messaging(sid, platform) and self.runtime.authorized_owner(sid))

    def pre_gateway_dispatch(self, event=None, gateway=None, session_store=None, **_):
        if gateway is not None:
            self.gateway = gateway
            self.settlement.install(gateway)
        # The adapter runs this chain for a message parked behind a busy session and marks it (busy_ingress); a late
        # arrival the idle path takes over afterwards is the same message, already recorded.
        if event is None or _get(event, 'internal', False) or _get(event, '_colonymind_ingress_done', False) is True:
            return None
        source = _get(event, 'source')
        platform = _get(source, 'platform', '')
        platform = str(_get(platform, 'value', platform)).lower()
        sender, chat = str(_get(source, 'user_id', '') or ''), str(_get(source, 'chat_id', '') or '')
        message_id = str(_get(event, 'message_id', '') or '')
        if not sender or not platform:
            return None
        text = _text(_get(event, 'text', ''))
        session_key = str(_get(event, 'session_key', '') or '')
        if not session_key and gateway is not None:
            derive = getattr(gateway, '_session_key_for_source', None)
            if callable(derive):
                try:
                    session_key = derive(source)
                except Exception:
                    logger.debug('ingress route unavailable', exc_info=True)
        key = (platform, chat, sender, message_id or 'anonymous:' + _hash(text))
        rewrite = bind_rewrite(event, observer=self)
        if rewrite is None:
            # Keep a refusal for this exact arrival; absence must not hide an
            # invalid present receipt behind a retained admission.
            with self.lock:
                self.inbound[key] = Inbound(platform, sender, chat, message_id, '', self.clock(),
                    str(_get(event, 'session_id', '') or ''), session_key, rewrite=False)
            return None
        queued = _get(event, '_colonymind_busy_ingress', False) is True
        with self.lock:
            # Re-delivery must not grant a second turn owner authority.
            if key not in self.inbound:
                self.inbound[key] = Inbound(platform, sender, chat, message_id, _hash(text), self.clock(),
                    str(_get(event, 'session_id', '') or ''), session_key, stripped_hash=_hash(text.strip()),
                    queued=queued, rewrite=rewrite)
                while len(self.inbound) > 2048:
                    self.inbound.popitem(last=False)
            if queued:
                self._mark_mixed(gateway or self._gateway(), session_key, sender)
        return None

    def _mark_mixed(self, gateway, session_key, sender):
        """Hermes merges a message parked behind a busy session into the one already waiting in the adapter's slot
        (a photo burst, media on either side, or text joined to text) and keeps the waiting message's sender and id,
        whoever sent the later one. When another sender's message arrives behind it, the waiting message's turn can
        carry both people's words, so it never binds as anyone's (a merge that did not happen only costs the
        attribution of that turn). Called under the lock, before the adapter merges."""
        if gateway is None or not session_key:
            return
        for adapter in list((getattr(gateway, 'adapters', None) or {}).values()):
            waiting = (getattr(adapter, '_pending_messages', None) or {}).get(session_key)
            if waiting is None or _get(waiting, 'internal', False):
                continue
            source = _get(waiting, 'source')
            owner = str(_get(source, 'user_id', '') or '')
            if owner == sender:
                continue
            platform = _get(source, 'platform', '')
            platform = str(_get(platform, 'value', platform)).lower()
            message_id = str(_get(waiting, 'message_id', '') or '')
            row = self.inbound.get((platform, str(_get(source, 'chat_id', '') or ''), owner,
                                    message_id or 'anonymous:' + _hash(_text(_get(waiting, 'text', '')))))
            if row is not None:
                row.mixed = True

    def _scope(self, sid, tid, values):
        info = self._session_info(sid)
        platform = values.get('platform') or _env('PLATFORM') or _get(info, 'platform', '')
        session_key = str(values.get('session_key') or _env('KEY') or '')
        own = None if values.get('message_id') else self._platform_message_id(sid, tid, session_key)
        return (str(_get(platform, 'value', platform) or '').lower(),
                str(values.get('sender_id') or _env('USER_ID') or _get(info, 'sender_id', '') or ''),
                str(values.get('chat_id') or _env('CHAT_ID') or _get(info, 'chat_id', '') or ''),
                str(values.get('message_id') or (own if own is not None else _env('MESSAGE_ID')) or ''),
                session_key, own is not None)

    def _platform_message_id(self, sid, tid, session_key):
        """This turn's own inbound platform id, read from the agent running it; None when no agent is found.

        Hermes binds the HERMES_SESSION_* variables once per dispatched message, and a message parked behind a busy
        session runs in-band as that run's queued follow-up under the first message's variables. The agent carries each
        turn's own raw inbound id (``_persist_user_message_platform_id``, set with ``_current_turn_id`` before
        ``pre_llm_call``), so a follow-up binds to its own message and never to the one that opened the run."""
        if not tid:
            return None
        gateway = self._gateway()
        candidates = [agent for _, agent in self._agent_candidates(gateway, session_key)] if gateway is not None else []
        try:
            # A follow-up that rebuilt its agent runs bound here before the gateway's tracker promotes it.
            from agent.subagent_lifecycle import get_active_subagent_parent
            candidates.append(get_active_subagent_parent())
        except ImportError:
            pass
        for agent in candidates:
            if (agent is not None and getattr(agent, 'session_id', None) == sid
                    and getattr(agent, '_current_turn_id', None) == tid):
                # A Hermes without the seam leaves the id unknown (the session variables decide, as before).
                own = getattr(agent, '_persist_user_message_platform_id', _UNSET)
                return None if own is _UNSET else str(own or '')
        return None

    def _agent_candidates(self, gateway, session_key):
        agents = getattr(gateway, '_running_agents', {})
        candidates = list(agents.items())
        # Hermes binds the agent in this ContextVar before pre_llm_call. The
        # asynchronous runner tracker can still hold its pending sentinel.
        if session_key and self._pending_agent(agents.get(session_key)):
            try:
                from agent.subagent_lifecycle import get_active_subagent_parent
                candidates.append((session_key, get_active_subagent_parent()))
            except ImportError:
                pass
        return candidates

    def _turn(self, sid, tid, text='', **values):
        key = (sid, tid)
        with self.lock:
            platform, sender, chat, message_id, session_key, id_known = self._scope(sid, tid, values)
            if key in self.turns:
                old = self.turns[key]
                coordinates = (platform, sender, chat, session_key)
                previous = (old.platform, old.sender, old.chat, old.session_key)
                same = (all(not value or value == saved for value, saved in zip(coordinates, previous))
                        and (not (id_known or message_id) or old.inbound is None or old.inbound.message_id == message_id))
                if old.composed:
                    same = coordinates == previous and old.inbound is not None and old.inbound.message_id == message_id
                if same:
                    return old
                old.stopped.set()
                self._restore_callbacks(old)
            state = Turn(sid, tid, self.clock(), platform, sender, chat, session_key)
            state.order = next(self._order)
            history = values.get('conversation_history') or []
            current = history[-1] if history else {}
            internal = (_get(current, 'role') == 'user' and _get(current, 'display_kind') == 'internal_notification')
            eligible = [row for row in self.inbound.values() if row.consumed in {None, key}
                        and (not row.session or row.session == sid)
                        and (not platform or row.platform == platform)
                        and (not sender or row.sender == sender)
                        and (not chat or row.chat == chat)
                        and (not row.session_key or not session_key or row.session_key == session_key)
                        and (row.session == sid or (platform and sender))]
            # A turn whose own id is known to be empty (an internal event) never takes, by its words alone, a message
            # that has an id: identical words are not the same message.
            matches = [] if internal else [row for row in eligible if not row.mixed and
                (row.message_id == message_id if message_id else
                 row.text_hash == _hash(text) and not (id_known and row.message_id))]
            members = composition(history, text, message_id)
            if members is not None and not internal:
                admitted = []
                invalid = False
                for own_id, words in members:
                    present = [row for row in eligible if row.message_id == own_id]
                    own = [row for row in present if not row.mixed and valid_ingress(row.rewrite)
                           and (row.text_hash == _hash(words) or matches_rewrite(row.rewrite, words))]
                    if present and (len(own) != 1 or len(present) != 1):
                        invalid = True
                    if len(own) == 1:
                        admitted.append(own[0])
                scope = (sid, tid, platform, sender, chat, message_id, session_key)
                if not invalid and len(admitted) == len(members):
                    evidence = [{name: getattr(row, name) for name in
                                 ('platform', 'sender', 'chat', 'message_id', 'arrived', 'queued')}
                                for row in admitted]
                    invalid = not self._composed_bindings.remember(scope, text, members, evidence,
                                                                  [row.rewrite for row in admitted])
                elif not invalid:
                    saved = self._composed_bindings.recover(scope, text, members)
                    if saved is not None and saved is not False:
                        admitted = [Inbound(**evidence, text_hash=_hash(words), consumed=key)
                                    for (_, words), evidence in zip(members, saved)]
                    else:
                        admitted = []
                if invalid:
                    admitted = []
                if len(admitted) == len(members):
                    state.composed, state.envelope = members, text
                    state.admissions = tuple(admitted)
                    for row in admitted:
                        row.consumed = key
                    matches = [admitted[-1]]
                else:
                    matches = []
            if len(matches) == 1:
                row = state.inbound = matches[0]
                row.consumed = key
                state.bound = bool(message_id)
                # Preserve arrival provenance; deadlines use only work_seconds.
                if not row.queued:
                    state.started = row.arrived
                state.platform, state.sender, state.chat = row.platform, row.sender, row.chat
                state.session_key = session_key or row.session_key
                state.reference = row.reference()
            self._classify_internal(state, internal)
            self.turns[key] = state
            while len(self.turns) > 2048:
                _, old = self.turns.popitem(last=False)
                old.stopped.set()
                self._restore_callbacks(old)
            return state

    def _classify_internal(self, state, internal=False):
        # Owner provenance wins, independently of agent binding.
        if state.inbound is None and internal and state.mode == 'answer_now':
            state.mode = 'system'

    def _state(self, sid, tid=''):
        if tid:
            return self.turns.get((sid, tid))
        return next((row for (session, _), row in reversed(self.turns.items()) if session == sid), None)

    def observe_turn(self, session_id='', turn_id='', user_message='', **values):
        # Memory observes the sender before invoking this bridge, independently
        # of plugin registration order. Context is emitted once by either hook.
        return self.pre_llm_call(session_id=session_id, turn_id=turn_id, user_message=user_message, **values)

    def capture_owner_text(self, session_id, turn_id, text):
        if self._inactive() or not self._messaging(session_id):
            return text
        if not self._managed(session_id):
            info = self._session_info(session_id)
            # A known guest stays outside owner capture; unresolved messaging fails closed.
            return text if _get(info, 'owner') is False else None
        state = self._state(session_id, turn_id)
        if state is None:
            self.metrics['capture_refused'] += 1
            return None
        row = state.inbound
        if row is not None and not valid_ingress(row.rewrite):
            self.metrics['capture_refused'] += 1
            return None
        cleaned = _text(text)
        if state.composed:
            current = [row for row in self.inbound.values() if row.platform == state.platform
                       and row.sender == state.sender and row.chat == state.chat
                       and any(row.message_id == ident for ident, _ in state.composed)]
            if any(not valid_ingress(row.rewrite) or row.mixed for row in (*state.admissions, *current)):
                self.metrics['capture_refused'] += 1
                return None
            if cleaned == state.envelope:
                return '\n\n'.join(words for _, words in state.composed)
            if any(cleaned == words for _, words in state.composed):
                return cleaned
            self.metrics['capture_refused'] += 1
            return None
        if row is not None and (_hash(cleaned) == row.text_hash or matches_rewrite(row.rewrite, cleaned)):
            return cleaned
        if row is not None and state.bound and not any(line in cleaned for line in state.synthetic):
            # Hermes puts its own notes ahead of what was sent (a reply's quote, a voice transcript, a media
            # note), joined by a blank line. The platform message id already binds this turn to the owner's real
            # message, so those notes keep it his; anything after his words, or a recorded line left inside,
            # cannot be split exactly and is refused.
            parts = cleaned.split('\n\n')
            for index in range(1, len(parts) + 1):
                rest = '\n\n'.join(parts[index:])
                if (_hash(rest) == row.text_hash or (row.stripped_hash and _hash(rest.strip()) == row.stripped_hash)
                        or matches_rewrite(row.rewrite, rest)):
                    return cleaned
        self.metrics['capture_refused'] += 1
        return None

    def owner_started_turn(self, session_id, turn_id=''):
        if self._inactive():
            return False
        state = self._state(session_id, turn_id)
        return bool(state is not None and state.inbound is not None)

    def origin_ref(self, session_id, turn_id=''):
        if self._inactive():
            return None
        state = self._state(session_id, turn_id)
        return dict(state.reference) if state is not None and state.reference else None

    def request_caps(self, session_id, turn_id=''):
        state = self._state(session_id, turn_id)
        if state is None or not self._managed(session_id):
            return None
        # Review and system turns lower caller-supplied bounds to at most 512; none is added.
        # Other managed turns bypass the middleware's bound rewriting entirely.
        return {'mode': state.mode, 'max_tokens': 512 if state.mode in {'review_result', 'system'} else 0,
                'duplicate': state.duplicate}

    def is_review_turn(self, session_id, turn_id=''):
        """Local, exact-turn classification for pre-request work in sibling plugins."""
        with self.lock:
            state = self.turns.get((session_id, turn_id)) if session_id and turn_id else None
            return bool(not self.closed and state is not None and not state.finished
                        and state.mode == 'review_result')

    def is_silent_hold_review(self, session_id, turn_id=''):
        """A local completion settlement has no chat output, even after its turn ends."""
        with self.lock:
            state = self.turns.get((session_id, turn_id)) if session_id and turn_id else None
            return bool(not self.closed and state is not None and state.mode == 'review_result' and state.hold_only)

    def defer_review_request(self, session_id, turn_id=''):
        """Settle a hold-only completion locally, or defer for a queued owner.

        Request middleware decides; execution middleware consumes the same decision
        without calling its provider callback. Recheck at execution for arrivals
        between the two hooks. The native queues remain untouched.
        """
        with self.lock:
            if not self.is_review_turn(session_id, turn_id):
                return False
            state = self.turns[(session_id, turn_id)]
            if state.hold_only:
                return True
            if self._inactive():
                return False
            if not self._bind(state):
                return False
            if not self._deadlines_on(state):
                return False
            if state.request_dropped:
                return True
            if not (state.owner_waiting or self._owner_waits(state)):
                return False
            state.owner_waiting = True
            state.request_dropped = True
            state.stopped.set()
            self._silence_callbacks(state)
            self.deferred_reviews[(state.session, state.session_key)] = DeferredReview(
                state.review_message, state.generation, after=next(self._order))
            self.metrics['review_request_dropped'] += 1
            return True

    @staticmethod
    def _pending_agent(value):
        try:
            from gateway.run import _AGENT_PENDING_SENTINEL
            return value is _AGENT_PENDING_SENTINEL
        except ImportError:
            return False

    def _bind(self, state):
        gateway = self._gateway()
        if gateway is None:
            return False
        agents = getattr(gateway, '_running_agents', {})
        if state.agent is not None:
            return self._current(state)
        for key, agent in self._agent_candidates(gateway, state.session_key):
            if (getattr(agent, 'session_id', None) == state.session
                    and getattr(agent, '_current_turn_id', None) == state.turn):
                native = gateway._peek_session_state(key)
                generation = native.persistent.run_generation if native is not None else None
                if state.generation is not None and state.generation != generation:
                    return False
                state.agent, state.session_key, state.generation = agent, key, generation
                state.pending_binding = self._pending_agent(agents.get(key))
                if not state.chat:
                    state.chat = _env('CHAT_ID')
                return self._current(state)
        if state.session_key and state.generation is None:
            native = gateway._peek_session_state(state.session_key)
            if native is not None:
                state.generation = native.persistent.run_generation
        return False

    def _current(self, state):
        gateway = self._gateway()
        if gateway is None:
            return False
        live = getattr(gateway, '_running_agents', {}).get(state.session_key)
        same = live is state.agent or (state.pending_binding and self._pending_agent(live))
        return bool(not state.finished and state.agent is not None and same
                    and getattr(state.agent, '_current_turn_id', None) == state.turn
                    and getattr(state.agent, 'session_id', None) == state.session
                    and state.generation is not None
                    and gateway._is_session_run_current(state.session_key, state.generation))

    def _adapters(self):
        gateway = self._gateway()
        adapters = getattr(gateway, 'adapters', {}) if gateway is not None else {}
        return list(adapters.values())

    def _owner_waits(self, state):
        gateway = self._gateway()
        events = [getattr(adapter, '_pending_messages', {}).get(state.session_key) for adapter in self._adapters()]
        native = gateway._peek_session_state(state.session_key)
        if native is not None:
            events.extend(list(native.conversation.queued_events))
        for event in events:
            if event is None or _get(event, 'internal', False):
                continue
            source = _get(event, 'source')
            sender = str(_get(source, 'user_id', '') or '')
            chat = str(_get(source, 'chat_id', '') or '')
            # The queue is session-bound, but a sender check still excludes shared-chat guests.
            if sender and state.sender and sender == state.sender and (not state.chat or state.chat == chat):
                return True
        # In queue mode Hermes holds busy text in its debounce buffer (0.35 s by default) before the adapter's slot above
        # has it; the busy path recorded it in this ledger on arrival (busy_ingress), so the owner waits from then on.
        now = self.clock()
        return any(row.queued and row.consumed is None and row.session_key == state.session_key
                   and state.sender and row.sender == state.sender and (not state.chat or row.chat == state.chat)
                   and now - row.arrived <= HEARD_WAIT_SECONDS for row in self.inbound.values())

    def _redirect(self, state, line):
        """Legacy metric names, but delivery uses non-cancelling native steer."""
        if not self._current(state):
            return False
        first = line not in state.synthetic
        if first:
            state.synthetic.append(line)
        try:
            accepted = state.agent.steer(line) is True
        except Exception:
            accepted = False
            logger.exception('front boundary guidance failed')
        if accepted or first:
            self.metrics['redirect_accepted' if accepted else 'redirect_refused'] += 1
        return accepted

    def _deadlines_on(self, state):
        if not self._watchdog_on():
            # Latch off for this turn. Re-enabling cannot replay old deadlines.
            state.deadlines_disabled = True
            state.owner_waiting = False
            state.request_dropped = False
            self._restore_callbacks(state)
        return not state.deadlines_disabled

    def tick(self, session_id, turn_id):
        with self.lock:
            state = self._state(session_id, turn_id)
            if state is None or state.finished or self._inactive():
                return
            # No asynchronous watchdog action may cancel an owner's own model request. In
            # particular, a final first response stays untouched however late.
            self._deadlines_on(state)
            if state.agent is not None and not self._current(state):
                state.stopped.set()
                self._restore_callbacks(state)
                return
            self._drop_review_for_owner(state)

    def _drop_review_for_owner(self, state):
        """A hand's result review yields to a waiting owner message even mid-request.

        Only review turns, at most once per turn, and only while this review's own provider call is in flight:
        llm_execution sets model_waiting under self.lock and clears it under self.lock, so while we hold the lock the
        in-flight request cannot be the owner's next turn. The review is deferred and retried after the owner turn
        (as defer_review_request does before a request starts); the owner's own turns are never cut."""
        if (state.mode != 'review_result' or state.request_dropped or not state.model_waiting
                or not self._watchdog_on() or not (state.owner_waiting or self._owner_waits(state))
                or not self._current(state)):
            return False
        state.owner_waiting = True
        state.request_dropped = True
        state.stopped.set()
        self._silence_callbacks(state)
        self.deferred_reviews[(state.session, state.session_key)] = DeferredReview(
            state.review_message, state.generation, after=next(self._order))
        self.metrics['review_request_dropped'] += 1
        logger.info('hands front review dropped for a waiting owner session=%s turn=%s', state.session, state.turn)
        state.agent.hard_interrupt()
        return True

    def _tool_boundary(self, state):
        if (state.finished or state.stopped.is_set() or self._inactive()
                or not self._deadlines_on(state) or not self._bind(state)):
            return
        # Hermes may abandon a timed-out hook and advance without it. Execution
        # middleware owns this gate throughout the actual provider call, so a
        # late boundary cannot overlap that call. Native interrupt locks are
        # deliberately untouched: hard_interrupt itself takes a plain Lock.
        if not state.request_gate.acquire(blocking=False):
            return
        try:
            active = getattr(state.agent, '_model_request_active', None)
            if active is not None and active.is_set():
                return
            if self._current(state):
                self._apply_tool_boundary(state)
        finally:
            state.request_gate.release()

    def llm_execution(self, request, next_call, session_id='', turn_id='', **values):
        state = self._state(session_id, turn_id)
        if state is None:
            return next_call(request)
        with state.request_gate:
            with self.lock:
                # Native execution may abandon a timed-out tool or hook. Close
                # its interval before entering a provider, and reject late tool
                # notifications during that wait so they cannot spend it later.
                if state.active_tools:
                    state.work_seconds += max(0, self.clock() - state.tool_started)
                    state.active_tools.clear()
                    state.tool_started = None
                state.model_waiting = True
            try:
                return next_call(request)
            finally:
                with self.lock:
                    state.model_waiting = False

    def _apply_tool_boundary(self, state):
        line = None
        owner_waits = state.mode == 'review_result' and self._owner_waits(state)
        if owner_waits and not state.owner_redirected:
            line = OWNER_WAITING_LINE
        elif (state.mode == 'review_result' and state.work_seconds >= 10
              and not state.visible and not state.soft_sent and not state.owner_waiting):
            # Owner turns reach no time limit at all: no 30 s yield line and no hard interrupt.
            # A long job runs to its end instead of being told to answer early.
            # A review without visible text yields after ten seconds of tool work.
            line = YIELD_LINE
        if line and self._redirect(state, line):
            self._guidance_sent(state, line)

    def _guidance_sent(self, state, line):
        if line not in state.synthetic:
            state.synthetic.append(line)
        if line == OWNER_WAITING_LINE:
            state.owner_waiting = state.owner_redirected = True
        else:
            state.soft_sent = True
            self.metrics['deadline_yield'] += 1
            logger.info('hands front yield session=%s turn=%s boundary=post_tool_call', state.session, state.turn)

    def _watchdog_on(self):
        """Read the runtime's live faculty mapping each time; no extra cache."""
        loader = getattr(self.runtime, 'config_loader', None)
        try:
            config = loader() if callable(loader) else getattr(self.runtime, 'config', {})
        except Exception:
            config = getattr(self.runtime, 'config', {})
        if not isinstance(config, dict):
            return True
        front = config.get('front')
        faculty = config.get('mind', {}).get('faculties', {}).get('hands', config.get('faculty'))
        return faculty is not False and not (isinstance(front, dict) and front.get('watchdog') is False)

    def _watch(self, state):
        # Assumption: 100 ms queue polling, independent of model/tool execution.
        while not state.stopped.wait(0.1):
            try:
                self.tick(state.session, state.turn)
            except Exception:
                logger.exception('front watchdog tick failed')

    def _silence_callbacks(self, state):
        if not self._current(state) or state.saved_callbacks:
            return
        for name in ('_disable_streaming', 'stream_delta_callback', '_stream_callback',
                     'interim_assistant_callback', 'reasoning_callback'):
            if hasattr(state.agent, name):
                state.saved_callbacks[name] = getattr(state.agent, name)
                setattr(state.agent, name, True if name == '_disable_streaming' else None)
        if '_disable_streaming' not in state.saved_callbacks:
            state.saved_callbacks['_disable_streaming'] = None
            state.agent._disable_streaming = True

    def _restore_callbacks(self, state):
        if state.agent is not None and getattr(state.agent, '_current_turn_id', '') == state.turn:
            for name, value in state.saved_callbacks.items():
                setattr(state.agent, name, value)
        state.saved_callbacks.clear()

    def _completion_texts(self, state):
        """Read only owned native receipts and render them with Hermes' own formatter.

        The public ledger omits event_json. A read-only connection avoids the
        native writer's schema initialization. The 50 ms / 256 row bounds are
        assumptions; a miss keeps the turn system-attributed and never silent.
        """
        from tools.process_registry_notifications import format_process_notification
        module = self.runtime.native.async_module
        owned = {hand.delegation_id: hand.plan.get('controller_hand_id', hand.hand_id)
                 for hand in self.runtime.registry.all()
                 if hand.parent_session_id == state.session and hand.delegation_id}
        if not owned or not state.session_key:
            return {}
        path = Path(module._db_path()).resolve()
        query = ("SELECT delegation_id,event_json FROM async_delegations "
                 "WHERE parent_session_id=? AND origin_session=? "
                 "AND state NOT IN ('running','finalizing') AND event_json IS NOT NULL "
                 "ORDER BY completed_at DESC LIMIT 256")
        started = time.monotonic()
        db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0.05)
        try:
            rows = db.execute(query, (state.session, state.session_key)).fetchall()
        finally:
            db.close()
        rendered = {}
        for ident, raw in rows:
            if time.monotonic() - started > 0.05:
                return {}
            if ident not in owned:
                continue
            event = json.loads(raw)
            if (isinstance(event, dict) and event.get('type') == 'async_delegation'
                    and event.get('delegation_id') == ident
                    and event.get('parent_session_id') == state.session
                    and event.get('session_key') == state.session_key
                    and event.get('completed_at') and not event.get('task_failure_notice')):
                text = format_process_notification(event)
                if isinstance(text, str) and text:
                    rendered[owned[ident]] = text
        return rendered

    def _classify_completion(self, state, text):
        if state.mode != 'system':
            # A plugin injection can race a new owner turn and enter Hermes' queued follow-up path, which
            # omits the internal-notification display kind. Only our accepted replay can use this fallback:
            # an actual inbound owner message always wins, and native receipts still have to match below.
            with self.lock:
                deferred = self.deferred_reviews.get((state.session, state.session_key))
                if (state.mode != 'answer_now' or state.inbound is not None or deferred is None
                        or not deferred.scheduled or deferred.message != text or not self._current(state)
                        or state.generation <= deferred.generation):
                    return
        try:
            trusted = self._completion_texts(state)
        except Exception:
            logger.debug('native completion receipts unavailable', exc_info=True)
            return
        matches = {ident for ident, block in trusted.items() if _hash(block) == _hash(text) and block == text}
        if not matches:
            found = sorted((text.find(block), ident, block) for ident, block in trusted.items() if block in text)
            if 2 <= len(found) <= 256 and len({pos for pos, _, _ in found}) == len(found):
                header = (f'[IMPORTANT: {len(found)} background subagent delegations '
                          'completed for this session. Treat these results as one '
                          'completion batch and send at most one consolidated user-facing '
                          'response. If a result does not change the current conclusion, absorb it silently.]')
                expected = '\n\n'.join([header, *(block for _, _, block in found)])
                if _hash(expected) == _hash(text) and expected == text:
                    matches = {ident for _, ident, _ in found}
        if matches:
            state.completions, state.trigger_hands = trusted, matches
            state.mode = 'review_result'
            state.review_message = text
            with self.lock:
                key = (state.session, state.session_key)
                deferred = self.deferred_reviews.get(key)
                if (deferred is not None and deferred.message == text and state.generation is not None
                        and state.generation > deferred.generation):
                    self.deferred_reviews.pop(key, None)

    def _result_key(self, row):
        return (row.get('hand_id') or row.get('id'), row.get('intent_version', 1),
                row.get('result_hash') or row.get('ended_at') or row.get('finished_at') or row.get('outcome') or row.get('state'))

    def _only_hold(self, row):
        """Require executor hold provenance and the controller's blocked result.

        Result prose never decides silence. Validated work still gets a review;
        an unconfirmed or failed completion keeps its ordinary error path.
        """
        if (row.get('state') != 'stopped' or row.get('outcome') != 'blocked'
                or (row.get('evidence') or {}).get('level') not in {'E0', 'E1'}):
            return False
        hand = self.runtime.registry.get(row.get('hand_id') or row.get('id'))
        if not (hand is not None and hand.parent_session_id == row.get('parent_session')
                and hand.blocked and hand.activity == 'awaiting_approval'
                and hand.status != 'unknown_effect' and not hand.plan.get('observed_artifacts')
                and not hand.plan.get('fetched_texts')):
            return False
        effects = getattr(self.runtime, 'effects', None)
        return effects is not None and effects.has_result_to_review(hand.hand_id) is False

    def _held_triggers(self, state):
        """Check locally before yielding to a queued owner, without controller I/O."""
        lookup = getattr(self.runtime.registry, 'get', None)
        if state.mode != 'review_result' or not state.trigger_hands or not callable(lookup):
            return False
        hands = [lookup(ident) for ident in state.trigger_hands]
        return all(hand is not None and hand.blocked and hand.activity == 'awaiting_approval' for hand in hands)

    def _shown(self, state, text, items):
        lines = set(text.splitlines())
        shown = [item for item in items if item.get('evidence') and item.get('hand_id')
                 and isinstance(item.get('text'), str) and '- ' + item['text'] in lines]
        snapshot = {'text': text, 'items': shown}
        try:
            capture = sys.modules.get('colonymind_capture')
            if capture is not None:
                capture.record_hand_refs(session_id=state.session, turn_id=state.turn, digest=snapshot)
        except Exception:
            logger.exception('Hands relay capture unavailable')
        if shown:
            try:
                self._emit('/v1/hands/relay', {'session': state.session, 'turn': state.turn,
                           'hand_ids': [item['hand_id'] for item in shown]}, 'relay:' + state.session + ':' + state.turn)
            except Exception:
                logger.exception('Hands relay delivery unavailable')
        return text

    def _digest(self, state):
        digest = self.runtime.client.get('/v1/hands/digest', {'session': state.session})
        text = str(digest.get('text') or '')
        if state.mode != 'review_result':
            return self._shown(state, text[:900], digest.get('items', []))
        rows = self.runtime.client.get('/v1/hands/', {'session': state.session}).get('hands', [])
        ready = [row for row in rows if row.get('parent_session', state.session) == state.session
                 and (row.get('hand_id') or row.get('id')) in state.completions
                 and row.get('state') in {'done', 'failed', 'below_minimum', 'stopped', 'lost', 'unknown_effect'}]
        fresh = [row for row in ready if self._result_key(row) not in self.rendered]
        # The native event is already matched byte for byte to owned receipts.
        # The ask is the hold's message. Absorb its completion before a provider
        # or any streaming callback can append another message after that ask.
        news = [row for row in ready if row in fresh or (row.get('hand_id') or row.get('id')) in state.trigger_hands]
        if (news and state.trigger_hands <= {row.get('hand_id') or row.get('id') for row in ready}
                and all(self._only_hold(row) for row in news) and self._bind(state)):
            state.hold_only = True
            self._silence_callbacks(state)
            return 'This hand stopped at an owner approval hold. Its ask is the message; this completion is silent.'
        if ready and not fresh:
            # A missing agent cannot enforce outbound silence. Do not claim a duplicate then.
            if self._bind(state):
                state.duplicate = True
                self._silence_callbacks(state)
                return 'These results were already reviewed. Reply with exactly [SILENT].'
            self.metrics['binding_missing'] += 1
        if fresh:
            items = {item.get('hand_id'): item for item in digest.get('items', [])}
            # Assumption: show at most twelve compact result lines; omitted rows
            # remain eligible for the next review and are never marked rendered.
            state.rendered = [{**row, **items.get(row.get('hand_id') or row.get('id'), {})} for row in fresh[:12]]
            refs = [row.get('origin_ref') for row in state.rendered if isinstance(row.get('origin_ref'), dict)]
            matching = [ref for ref in refs if ref.get('platform') == state.platform and ref.get('chat') == state.chat]
            if matching and all(ref == matching[0] for ref in matching):
                state.reference = dict(matching[0])
            # The full native trigger remains in the turn. Compact structural
            # lines expose every other ready result inside the 900-character cap.
            prefix = text[:300] + '\nReview all ready results together:\n'
            width = (900 - len(prefix) - 3 * len(state.rendered)) // len(state.rendered)
            lines, shown = [], []
            for row in state.rendered:
                evidence = row.get('evidence') or {}
                fallback = (str(row.get('name') or row.get('id')) + ': '
                            + str(row.get('outcome') or row.get('state')) + '; '
                            + str(evidence.get('level') or 'evidence not recorded'))
                if evidence.get('scope'):
                    fallback += ' ' + str(evidence['scope'].get('kind', ''))
                line = str(row.get('text') or fallback)
                line = line if len(line) <= width else line[:max(0, width - 3)] + '...'
                lines.append('- ' + line)
                shown.append({**row, 'text': line})
            state.rendered = shown
            return self._shown(state, prefix + '\n'.join(lines), shown)
        return self._shown(state, text[:900], digest.get('items', []))

    def _emit(self, path, body, key):
        return self.runtime.emit(path, body, 'front:' + key)

    def pre_llm_call(self, session_id='', turn_id='', user_message='', **values):
        if not self._managed(session_id, values.get('platform', '')) or values.get('parent_session_id'):
            return None
        state = self._turn(session_id, turn_id, _text(user_message), **values)
        # Capture may have established the turn before this hook received structural history. Hermes calls this
        # hook and the capture bridge (observe_turn) for the same turn: only a turn still unclassified takes the
        # structure, so a second call never sets a classified review back to 'system'.
        current = (values.get('conversation_history') or [{}])[-1]
        self._classify_internal(state, _get(current, 'display_kind') == 'internal_notification')
        if state.armed:
            self._classify_completion(state, _text(user_message))
        if state.armed:
            return None
        state.armed = True
        self._bind(state)
        self._classify_completion(state, _text(user_message))
        if self._held_triggers(state):
            try:
                state.context = self._digest(state)
            except Exception:
                logger.exception('held completion digest unavailable; keeping the normal review')
        if self.defer_review_request(session_id, turn_id):
            return {'context': state.context} if state.context else None
        if self.start_threads and self._watchdog_on():
            state.thread = threading.Thread(target=self._watch, args=(state,), name='hands-front-watchdog', daemon=True)
            state.thread.start()
        if state.inbound and state.reference:
            self._emit('/v1/hands/owner_message', state.reference, 'owner:' + _hash(str(state.reference)))
        try:
            if not state.context:
                state.context = self._digest(state)
        except Exception:
            logger.exception('front digest unavailable; keeping the normal reply')
        return {'context': state.context} if state.context else None

    def post_api_request(self, session_id='', turn_id='', assistant_message=None, response=None, **_):
        state = self._state(session_id, turn_id)
        if state is None:
            return
        content = _get(assistant_message, 'content', '')
        if not content and isinstance(response, dict):
            content = _get(response.get('assistant_message'), 'content', '')
        if visible_text(content, state.agent):
            state.visible = True

    def finalize_response(self, response_text='', session_id='', turn_id='', **_):
        state = self._state(session_id, turn_id)
        if state is not None and (state.duplicate or state.owner_waiting or state.hold_only):
            return SILENT
        return None

    def post_llm_call(self, session_id='', turn_id='', assistant_response='', **_):
        state = self._state(session_id, turn_id)
        if state is None:
            return
        state.final_response = _text(assistant_response)
        state.stopped.set()
        self._restore_callbacks(state)

    def approval_review_ended(self, hand_id):
        """Wall time when this hand's parent review released its gateway lease.

        A restart without this receipt waits for the controller's fixed deadline.
        Merely receiving a hand result or drafting a reply does not release it.
        """
        with self.lock:
            return self.approval_reviews.get(hand_id)

    def on_session_end(self, session_id='', turn_id='', task_id='', completed=None, failed=None,
                       interrupted=None, turn_exit_reason=None, **_):
        self.on_native_turn_settled(session_id=session_id, turn_id=turn_id, task_id=task_id,
            outcome={'completed': completed, 'failed': failed, 'interrupted': interrupted,
                     'turn_exit_reason': turn_exit_reason})

    def on_native_turn_settled(self, session_id='', turn_id='', outcome=None, **_):
        state = self._state(session_id, turn_id)
        if state is None or state.outcome_recorded:
            return
        current = self._current(state)
        state.finished = True
        state.outcome_recorded = True
        state.outcome_current = current
        state.stopped.set()
        self._restore_callbacks(state)
        successful = (current and isinstance(outcome, dict) and outcome.get('completed') is True
                      and outcome.get('failed') is False and outcome.get('interrupted') is False
                      and not state.owner_waiting)
        text = visible_text(state.final_response, state.agent)
        if not successful or not text:
            return
        if self.start_threads:
            self._background(lambda: self._record_review(state, text))
        else:
            self._record_review(state, text)

    def _background(self, callback):
        def run():
            try:
                callback()
            except Exception:
                logger.exception('front controller bookkeeping failed')
            finally:
                with self.lock:
                    self._jobs.discard(threading.current_thread())
        thread = threading.Thread(target=run, name='hands-front-bookkeeping', daemon=True)
        with self.lock:
            self._jobs.add(thread)
        thread.start()

    def _record_review(self, state, text):
        # The outcome hook may enqueue this work from a callback worker. A newer
        # run must cancel remaining receipts, including after an HTTP response.
        gateway = self._gateway()
        runtime_closed = getattr(self.runtime, '_closed', None)
        def current():
            return (not self.closed and not (runtime_closed is not None and runtime_closed.is_set())
                    and gateway is not None and state.generation is not None
                    and gateway._is_session_run_current(state.session_key, state.generation))
        if not current():
            return
        reviews, communicated = {}, set()
        for row in state.rendered:
            if not current():
                break
            oid = row.get('obligation_id')
            if oid:
                if oid not in reviews:
                    reviews[oid] = self._emit(f'/v1/hands/obligations/{oid}/review',
                        {'turn_ref': state.turn, 'successful': True, 'rendered': True}, f'review:{state.turn}:{oid}')
                if not current():
                    break
                receipt = reviews[oid]
                if not isinstance(receipt, dict) or receipt.get('error') or not (receipt.get('state') == 'reviewed' or receipt.get('ok') is True):
                    continue
                origin = row.get('origin_ref')
                if (oid not in communicated and text != SILENT and isinstance(origin, dict) and origin.get('platform') == state.platform
                        and origin.get('chat') == state.chat):
                    self._emit(f'/v1/hands/obligations/{oid}/communicate',
                        {'turn_ref': state.turn, 'silent': False,
                         'destination': {'platform': state.platform, 'chat': state.chat}},
                        f'communicate:{state.turn}:{oid}')
                    communicated.add(oid)
            with self.lock:
                if not current():
                    break
                self.rendered.add(self._result_key(row))
        self._save_rendered()

    def before_gateway_turn_clear(self, session_id='', session_key='', run_generation=None, **_):
        """Retain exact live identity when an early exit skipped on_session_end."""
        with self.lock:
            if self._inactive():
                return
            for state in self.turns.values():
                if (state.session == session_id and state.session_key == session_key
                        and state.generation == run_generation and state.armed and state.agent is not None
                        and state.mode == 'review_result' and not state.owner_waiting
                        and not state.outcome_recorded and self._current(state)):
                    state.early_exit_current = True

    def on_gateway_turn_settled(self, session_id='', session_key='', run_generation=None, **_):
        with self.lock:
            gateway = self._gateway()
            current = (not self._inactive() and gateway is not None and run_generation is not None
                       and gateway._is_session_run_current(session_key, run_generation))
            ended = time.time()
            owners = []
            for state in self.turns.values():
                if (state.session == session_id and state.session_key == session_key
                        and state.generation == run_generation):
                    if state.inbound is not None:
                        owners.append(state)
                    # on_session_end runs before native cleanup. Only successful
                    # owned lease release starts an approval notice's reply gap.
                    # Omitted and deferred results still await their own review.
                    early_exit = (state.early_exit_current and not state.outcome_recorded and not state.finished
                        and getattr(state.agent, '_current_turn_id', None) == state.turn
                        and getattr(state.agent, 'session_id', None) == state.session
                        and getattr(gateway, '_running_agents', {}).get(session_key) in (None, state.agent))
                    if (current and (state.outcome_current or early_exit) and state.mode == 'review_result'
                            and not state.owner_waiting):
                        reviewed = state.trigger_hands if state.hold_only or state.duplicate else {
                            row.get('hand_id') or row.get('id') for row in state.rendered}
                        for hand_id in reviewed:
                            self.approval_reviews[hand_id] = ended
                    state.finished = True
                    state.stopped.set()
                    self._restore_callbacks(state)
            self.approval_reviews = {key: at for key, at in self.approval_reviews.items() if ended - at < 600}
            # Hermes runs a queued owner message as the same run's follow-up, so the owner's turn usually settles
            # with the dropped review's generation. Only an owner turn that began after the drop schedules the retry.
            owner = max(owners, key=lambda state: state.order) if owners else None
            key = (session_id, session_key)
            deferred = self.deferred_reviews.get(key)
            if (owner is None or deferred is None or self._inactive() or not self._deadlines_on(owner)
                    or run_generation is None or run_generation < deferred.generation
                    or owner.order < deferred.after
                    or deferred.scheduled_after == owner.turn
                    or not self._gateway()._is_session_run_current(session_key, run_generation)
                    or self._owner_waits(owner)):
                return
            if deferred.attempts >= REVIEW_RETRIES:
                self.deferred_reviews.pop(key, None)
                self.metrics['review_retry_abandoned'] += 1
                logger.warning('deferred review abandoned after %d scheduled retries', deferred.attempts)
                return
            deferred.scheduled_after = owner.turn
            deferred.attempts += 1
            try:
                # Gateway injection is internal even with role=user. Retain the
                # exact native completion body so its receipt can classify again.
                accepted = self.ctx.inject_message(deferred.message, session_key=session_key, role='user')
            except Exception:
                accepted = False
                logger.exception('deferred review was not scheduled')
            # A scheduled injection can still fail in the asynchronous native
            # route. Classification of its actual review turn clears the retry;
            # until then a later owner settlement may schedule it again.
            if not accepted:
                self.metrics['injection_failed'] += 1
            else:
                deferred.scheduled = True

    def agent_loop_stopped(self, session_id='', turn_id='', session_key='', **_):
        states = [self._state(session_id, turn_id)] if session_id else [
            state for state in self.turns.values() if state.session_key == session_key]
        for state in states:
            if state is not None:
                state.finished = True
                state.stopped.set()
                self._restore_callbacks(state)

    def inventory_tools(self, schemas):
        """Only rewrite timeout fields explicitly present in native tool schemas."""
        for schema in schemas:
            function = schema.get('function', schema)
            name = function.get('name', '')
            props = function.get('parameters', {}).get('properties', {})
            for key in ('timeout', 'timeout_seconds'):
                if props.get(key, {}).get('type') in {'number', 'integer'}:
                    self.timeout_inventory[name] = key
                    break
            else:
                self.unbounded_tools.add(name)
        return {'timeouts': dict(self.timeout_inventory), 'without_timeout': sorted(self.unbounded_tools)}

    def pre_tool_call(self, session_id='', turn_id='', tool_name='', args=None, tool_input=None, tool_call_id='', **_):
        # A union of tool intervals is the work clock: parallel tools count once,
        # and every intervening model request contributes zero seconds.
        active = False
        with self.lock:
            state = self._state(session_id, turn_id)
            if state is not None and not state.finished and not state.stopped.is_set() and not state.model_waiting:
                if not state.active_tools:
                    state.tool_started = self.clock()
                state.active_tools.add(tool_call_id or threading.get_ident())
                active = True
        if active:
            self._deadlines_on(state)
        with self.lock:
            return self._pre_tool_call(session_id, turn_id, tool_name, args, tool_input, tool_call_id)

    def post_tool_call(self, session_id='', turn_id='', tool_call_id='', tool_name='', args=None, result=None, **_):
        with self.lock:
            state = self._state(session_id, turn_id)
            # send_message bypasses the native final-reply ledger. Its own
            # success receipt marks the turn visible, but aliases, home-channel
            # defaults, reactions and ambiguous results cannot attest this chat.
            if state is not None and tool_name == 'send_message' and tool_call_id and isinstance(args, dict):
                try:
                    receipt = json.loads(result) if isinstance(result, str) else result
                except (ValueError, TypeError):
                    receipt = None
                words = visible_text(args.get('message', ''), state.agent)
                delivered = state.delivered_tool_ids
                if (isinstance(receipt, dict) and receipt.get('success') is True
                        and not receipt.get('skipped') and not receipt.get('error')
                        and str(args.get('action') or 'send') == 'send'
                        and state.platform and state.chat
                        and str(args.get('target') or '').strip() == state.platform + ':' + state.chat
                        and words and words != SILENT and tool_call_id not in delivered):
                    delivered.add(tool_call_id)
                    state.visible = True
            key = tool_call_id or threading.get_ident()
            if state is None or key not in state.active_tools:
                return
            state.active_tools.remove(key)
            if not state.active_tools:
                state.work_seconds += max(0, self.clock() - state.tool_started)
                state.tool_started = None
                self._tool_boundary(state)

    def _pre_tool_call(self, session_id='', turn_id='', tool_name='', args=None, tool_input=None, tool_call_id=''):
        state = self._state(session_id, turn_id)
        if state is None:
            return None
        if state.duplicate:
            return {'action': 'block', 'message': 'Reply now and hand off remaining work.'}
        reply_now = not state.deadlines_disabled and state.owner_waiting
        if tool_name in HAND_OFF_TOOLS:
            if not reply_now:
                return {}
            if not tool_call_id or tool_call_id not in state.tool_ids:
                if state.calls.get('hand_off', 0) >= HAND_OFFS_AFTER_REPLY_NOW:
                    return {'action': 'block', 'message': 'Reply now.'}
                state.calls['hand_off'] = state.calls.get('hand_off', 0) + 1
                if tool_call_id:
                    state.tool_ids.add(tool_call_id)
            return {}
        if reply_now:
            return {'action': 'block', 'message': 'Reply now and hand off remaining work with delegate_task.'}
        read = tool_name in READ_TOOLS
        # Counted, never capped. Every path in this class is owner-turns-only already -- state exists only for turns
        # _turn() created, and pre_llm_call creates none unless _managed() with no parent_session_id --
        # so there is no non-owner caller left to cap and no owner conditional to add. The counters stay:
        # the hand-off path below reads state.calls['hand_off'], and a per-turn tool census is worth
        # having in the log without being a budget.
        bucket = 'read' if read else 'write'
        if not tool_call_id or tool_call_id not in state.tool_ids:
            state.calls[bucket] += 1
            if tool_call_id:
                state.tool_ids.add(tool_call_id)
        # No per-tool timeout ceiling. This used to clamp terminal/process/browser_cdp/connections to
        # min(30, ...), which silently overrode what the caller asked for: a deliberate
        # terminal(timeout=180) became 30. It existed to stop one hung call eating the 90 s turn budget
        # that no longer exists. The tool's own default applies when the caller passes nothing.
        return {}

    def close(self):
        self.closed = True
        self.settlement.close()
        for state in list(self.turns.values()):
            state.stopped.set()
            self._restore_callbacks(state)
        for state in list(self.turns.values()):
            if state.thread is not None and state.thread is not threading.current_thread():
                state.thread.join(timeout=0.2)
        for thread in list(self._jobs):
            thread.join(timeout=0.2)
        if sys.modules.get(_BRIDGE) is self.bridge:
            sys.modules.pop(_BRIDGE, None)


def register(ctx):
    from .dispatch import get_runtime
    if getattr(ctx, '_load_abandoned', False):
        return
    guard = FrontGuard(get_runtime(ctx), ctx)
    if guard._inactive():
        return
    ctx.on_unload(guard.close)
    # Stock Hermes ignores cleanup registration after a load is abandoned.
    # A late loader must not replace the active bridge or settlement observer.
    if getattr(ctx, '_load_abandoned', False):
        guard.close()
        return
    if guard._inactive():
        return
    guard.install_bridge()
    try:
        from tools.registry import registry
        guard.inventory_tools([registry.get_schema(name) for name in registry.get_all_tool_names() if registry.get_schema(name)])
    except (ImportError, AttributeError):
        logger.info('front timeout inventory uses built-in schemas until tool schemas are available')
    for name in ('pre_gateway_dispatch', 'pre_llm_call', 'post_api_request', 'post_llm_call',
                 'pre_tool_call', 'post_tool_call', 'agent_loop_stopped', 'on_session_end'):
        callback = getattr(guard, name)
        def safe(_callback=callback, **kwargs):
            try:
                return _callback(**kwargs)
            except Exception:
                logger.exception('front hook failed: %s', _callback.__name__)
                if _callback.__name__ == 'pre_tool_call' and guard._managed(kwargs.get('session_id', '')):
                    return {'action': 'block', 'message': 'Front guard unavailable. Reply and hand off.'}
                return None
        ctx.register_hook(name, safe)
    ctx.register_hook('transform_llm_output', guard.finalize_response)
    ctx.register_middleware('llm_execution', guard.llm_execution)
