"""A message that arrives while its session is busy runs ``pre_gateway_dispatch`` too.

Hermes runs that hook for a message reaching an idle session (``GatewayRunner._hm_admit_event``). A message that
arrives while the session runs is parked by the adapter (``BasePlatformAdapter._handle_message_while_active``) and
later runs in-band as the run's queued follow-up, so it never reaches the hook (Hermes says so in
``gateway/run_busy.py``: "A steered or queued follow-up never reaches _hm_admit_event"). Every hook that hears,
seals or gates a message there then misses it: the secret seal, the heard evidence an answer to an ask needs, the
hands ingress ledger and instinct triage.

``install`` wraps each adapter's busy-session handler through the stock seam (``set_busy_session_handler``), so a
message that is not internal runs the same chain through the gateway's own runner (``_hm_pre_gateway_dispatch_hook``)
before the stock busy handling. ``skip`` drops it as the idle path would; ``rewrite`` replaces its text in place, since
the adapter queues the event object it holds. Every resolvable slash command (``/stop``, ``/approve``, a plugin's) is
dispatched before the busy handler runs and through the gateway's own message handler, which runs the chain itself
(``should_bypass_active_session``, ``_dispatch_inline_reply``), so what reaches the wrap is words for the model, a
slash-prefixed path among them. A late arrival the adapter later hands to the idle path runs the chain there again,
so the event is marked once the chain ran and every ColonyMind ``pre_gateway_dispatch`` hook skips a marked event.
Profile multiplexing scopes the busy path per profile, so there the wrap stays off and says so once. Adapters can
appear after this plugin loads and again on a reconnect: the two stock hooks below and a slow keeper thread apply the
wrap again (it is idempotent). Unloading restores each handler this load wrapped; a wrapper left by an earlier load
of this module is replaced rather than kept.

``turn_platform_id`` is the turn's own message id for the hooks that bind a turn to what was heard: a queued
follow-up runs under the session variables of the message that opened the run, so ``HERMES_SESSION_MESSAGE_ID`` is
not its own.
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
from typing import Any
import weakref

logger = logging.getLogger(__name__)

BUSY = "_colonymind_busy_ingress"   # set while the chain runs for a parked message; the hands ledger marks it queued
DONE = "_colonymind_ingress_done"   # set once the chain ran for it; ColonyMind's dispatch hooks skip it afterwards
_WRAPPED = "_colonymind_busy_wrapped"
_TOKEN = object()                  # this load of the module: a wrapper carrying another load's token is replaced
_UNSET = object()
_lock = threading.Lock()
_warned: set[str] = set()
_installed: list[tuple[Any, Any]] = []   # (adapter reference, wrapper) for each handler this load wrapped
_closed = False


def seen(event: Any) -> bool:
    """Whether the chain already ran for this event on the busy path."""
    return getattr(event, DONE, False) is True


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def _wrap(runner: Any, original: Any):
    async def busy_session_handler(event: Any, session_key: str) -> bool:
        if not getattr(event, "internal", False) and not seen(event):
            from colonymind_shared.midturn import begin_rewrite, finish_ingress
            receipt = begin_rewrite(event)
            setattr(event, BUSY, True)
            try:
                result = runner._hm_pre_gateway_dispatch_hook(event, event.source)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as error:
                logger.warning("busy ingress hooks failed (%s); the message is queued as Hermes queues it",
                               type(error).__name__)
                result = event
            setattr(event, DONE, True)
            if result is None:
                return True
            if result is not event and isinstance(getattr(result, "text", None), str):
                event.text = result.text
            finish_ingress(receipt, event)
        return bool(await original(event, session_key)) if original is not None else False

    setattr(busy_session_handler, _WRAPPED, _TOKEN)
    busy_session_handler.__wrapped__ = original
    return busy_session_handler


def _reference(adapter: Any) -> Any:
    try:
        return weakref.ref(adapter)
    except TypeError:
        return lambda: adapter


def install(gateway: Any = None) -> int:
    """Wrap every adapter's busy-session handler not wrapped yet; the number newly wrapped."""
    runner = gateway
    if runner is None:
        try:
            from gateway.run import _gateway_runner_ref
            runner = _gateway_runner_ref()
        except Exception:
            return 0
    if runner is None:
        return 0
    if not callable(getattr(runner, "_hm_pre_gateway_dispatch_hook", None)):
        _warn_once("hook", "busy ingress parity unavailable: this gateway has no pre_gateway_dispatch runner")
        return 0
    try:
        multiplex = getattr(runner, "_multiplex_on", None)
        if callable(multiplex) and multiplex():
            _warn_once("multiplex", "busy ingress parity off: profile multiplexing scopes the busy path per profile")
            return 0
    except Exception:
        return 0
    wrapped = 0
    with _lock:
        if _closed:
            return 0
        for adapter in list((getattr(runner, "adapters", None) or {}).values()):
            setter = getattr(adapter, "set_busy_session_handler", None)
            current = getattr(adapter, "_busy_session_handler", None)
            mark = getattr(current, _WRAPPED, None)
            if not callable(setter) or mark is _TOKEN:
                continue
            if mark is not None:   # an earlier load's wrapper: wrap the stock handler it holds, not the wrapper
                current = getattr(current, "__wrapped__", None)
            wrapper = _wrap(runner, current)
            setter(wrapper)
            _installed.append((_reference(adapter), wrapper))
            wrapped += 1
    return wrapped


def uninstall() -> int:
    """Give each adapter this load wrapped its own handler back, when it still holds the wrapper; no later wrap."""
    global _closed
    restored = 0
    with _lock:
        _closed = True
        for reference, wrapper in _installed:
            adapter = reference()
            if adapter is not None and getattr(adapter, "_busy_session_handler", None) is wrapper:
                adapter.set_busy_session_handler(wrapper.__wrapped__)
                restored += 1
        _installed.clear()
    return restored


def turn_platform_id(session_id: str, turn_id: str) -> str | None:
    """This turn's own raw inbound platform id ("" when it has none), read from the agent running it; None when no
    running agent is found. Hermes sets ``_persist_user_message_platform_id`` with ``_current_turn_id`` before
    ``pre_llm_call`` (``_bind_turn_identity``). The agent bound in the subagent-parent ContextVar counts too: a
    follow-up that rebuilt its agent runs there before the gateway's tracker promotes it."""
    if not session_id or not turn_id:
        return None
    candidates: list[Any] = []
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
        candidates.extend(list((getattr(runner, "_running_agents", None) or {}).values()))
    except Exception:
        pass
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent
        candidates.append(get_active_subagent_parent())
    except Exception:
        pass
    for agent in candidates:
        if (agent is not None and getattr(agent, "session_id", None) == session_id
                and getattr(agent, "_current_turn_id", None) == turn_id):
            own = getattr(agent, "_persist_user_message_platform_id", _UNSET)   # a Hermes without the seam: unknown
            return None if own is _UNSET else str(own or "")
    return None


class Keeper:
    """Applies the wrap every two seconds for two minutes after load, then every thirty seconds."""

    def __init__(self, fast: float = 2.0, settle: float = 120.0, slow: float = 30.0):
        self.fast, self.settle, self.slow = fast, settle, slow
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="colonymind-busy-ingress", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        waited = 0.0
        while True:
            delay = self.fast if waited < self.settle else self.slow
            if self.stopped.wait(delay):
                return
            waited += delay
            try:
                install()
            except Exception:
                logger.debug("busy ingress keeper skipped a pass", exc_info=True)

    def stop(self) -> None:
        self.stopped.set()


def register(ctx: Any) -> None:
    """The stock hooks that see the gateway early apply the wrap; a keeper covers adapters made later."""
    global _closed
    with _lock:
        _closed = False

    def ensure(gateway: Any = None, **_: Any) -> None:
        try:
            install(gateway)
        except Exception:
            logger.debug("busy ingress install skipped", exc_info=True)
        return None

    for name in ("pre_gateway_dispatch", "pre_llm_call"):
        ctx.register_hook(name, ensure)
    keeper = None if os.environ.get("HERMES_KANBAN_TASK") else Keeper()
    if keeper is not None:
        keeper.start()

    def unload() -> None:
        if keeper is not None:
            keeper.stop()
        uninstall()

    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        on_unload(unload)
