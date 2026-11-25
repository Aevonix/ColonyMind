"""ColonyMind adapter for Hermes: stock seams only.

Registration wires the hooks (``pre_llm_call``, ``post_llm_call``,
``pre_tool_call``, ``post_tool_call``, ``on_kanban_dispatch_tick``, ``on_skill_lifecycle``), the answer route's
own (``answers.py``: the owner's message read before the model runs, and the chat ledger fed by
``on_interim_message``, ``post_tool_call``, ``transform_llm_output`` and ``gateway_platform_event``), the
gateway gates (``pre_gateway_dispatch``: ``/mind`` is the owner's, each message is heard with its own evidence and
an owner's reaction on an ask is read at once, and a secret is sealed before Hermes sees it), the
``tool_request`` middleware that fills or hides sealed values and ``post_tool_call`` that clears their files
(``secrets.py``), one command (``/mind``), the
model tools and one system prompt section (the constitution from
``identity.yaml``, the owner, the self-narrative the sidecar keeps and two tool
notes), then starts the body thread, which delivers captured turns and runs the
mind loop (dispatch, outbox, reconciliation, observations) against
``/v1/mind``. Registration performs no network I/O and never changes the
Hermes configuration.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Mapping

__version__ = "1.1.0"

from .answers import Answers, ChatLedger, ledger_path
from .archive import ArchiveWorker
from .body import Activity, Body, BodyLedger
from . import busy_ingress, native_settlement
from .capture import Capture, SessionMap, TurnOutbox
from .client import ColonyMindClient, Settings, SidecarUnavailable, hermes_home, load_settings
from . import capture as capture_module, commands
from .guard import Guard
from .ideas import GoogleIdeas
from .registration import RegistrationEvidence
from .reminders import SCHEMA as REMINDER_SCHEMA, Reminders
from . import secrets as secrets_module
from .tools import Tools

logger = logging.getLogger(__name__)

HOOKS = ("pre_llm_call", "post_llm_call", "pre_tool_call", "post_tool_call", "on_kanban_dispatch_tick",
         "on_skill_lifecycle", "on_interim_message", "transform_llm_output", "gateway_platform_event",
         "subagent_start", "on_native_turn_settled", "pre_api_request", "post_api_request", "api_request_error",
         "pre_approval_request", "post_approval_response")
TOOLSET = "colonymind"
# The one prompt section, frozen per session by Hermes (architecture 4.2, seam 2): the constitution
# (<= 1,500 characters), the owner, the self-narrative (<= 800, the sidecar's own cap) and the two tool
# notes, <= 4,000 in all. Every model request of an owner session carries it (tests/hermes_adapter/
# test_overhead_budget.py pins its largest render).
SECTION_CHARS, CONSTITUTION_CHARS, NARRATIVE_CHARS = 4000, 1500, 800
NARRATIVE_LEAD = ("What you know about yourself, from your own record. Plain ids are your own actions "
                  "(colonymind_self why explains them); prefixed ids are record references:")
HELD = f"[narrative lines held back: over the {NARRATIVE_CHARS}-character limit]"
NOTES = ("ColonyMind keeps your long-term memory: recalled evidence arrives with each message and "
         "colonymind_memory_search finds more. The owner answers asks in their own words, read before you reply: "
         "never approve one or ask for a code (colonymind_self show sends one again).")


def prompt_section(settings: Settings, client: ColonyMindClient | None = None,
                   sessions: SessionMap | None = None) -> Callable[[Mapping[str, Any]], str]:
    """The block the plugin adds to every session prompt: who the agent is (the owner-authored constitution
    read from ``identity.yaml``), who the owner is, what the agent knows about itself from its own record
    (``GET /v1/mind/narrative``: 2 s, a failure cached 60 s, fail open to the constitution alone) and the two things
    the tool schemas cannot say. Hermes renders it once per session, so a nightly narrative change reaches
    the next session and the prompt cache holds within one. Recall guidance is the memory provider's block.

    The narrative is the owner's record (what the agent did for the owner, its working stances), so it is
    rendered only in a session that is the owner's alone (``SessionMap.owner_only``); every other session
    gets the constitution and the notes, and the sidecar is not asked on its behalf."""
    audience = sessions if sessions is not None or client is None else SessionMap(settings, client)

    def render(session_info: Mapping[str, Any]) -> str:
        identity = settings.identity()
        parts = [settings.constitution()[:CONSTITUTION_CHARS]]
        owner = identity.get("owner") if isinstance(identity.get("owner"), Mapping) else {}
        if owner.get("name"):
            parts.append(f"Your owner is {owner['name']}.")
        narrative: Any = None
        if client is not None and audience is not None and owner_only(audience, session_info):
            try:
                narrative = client.narrative()
            except Exception:  # the record is optional; the constitution is not
                logger.debug("narrative unavailable for the prompt section", exc_info=True)
        if isinstance(narrative, Mapping) and narrative.get("enabled") is True and narrative.get("text"):
            text = str(narrative["text"])
            if len(text) > NARRATIVE_CHARS:  # a sidecar that fitted nothing (1.10): whole lines, and it is said
                text = text[:NARRATIVE_CHARS - len(HELD) - 1].rsplit("\n", 1)[0] + "\n" + HELD
            parts.append(NARRATIVE_LEAD + "\n" + text)
        parts.append(NOTES)
        return "\n\n".join(part for part in parts if part)[:SECTION_CHARS]
    return render


def owner_only(sessions: SessionMap, session_info: Mapping[str, Any]) -> bool:
    """``SessionMap.owner_only`` for the session being rendered; never raises (a failure withholds)."""
    try:
        return sessions.owner_only(session_info or {})
    except Exception as error:
        logger.warning("standing_audience_unknown exception_type=%s", type(error).__name__)
        return False


_BODY: Body | None = None
_ACTIVITY = None
_REGISTRATION_LOCK = threading.RLock()
_REGISTRATION_STATE_LOCK = threading.Lock()
_REGISTERED_TOOLS: tuple[dict[str, Any], ...] | None = None


class _Registration(RegistrationEvidence):
    """Keep the tool declarations, including those registered by helpers."""

    def __init__(self, context):
        super().__init__(context)
        self.tools: list[dict[str, Any]] = []
        self.cleanups: list[Callable[[], None]] = []

    def register_tool(self, **registration):
        result = self.context.register_tool(**registration)
        if result is not None:
            self.tools.append(registration)
        return result

    def on_unload(self, callback):
        # The same cleanup may run on local rollback and later Hermes teardown.
        done = False
        def once():
            nonlocal done
            with _REGISTRATION_STATE_LOCK:
                if done:
                    return
                done = True
            # Cleanup may stop workers; never hold the state lock across it.
            callback()
        self.cleanups.append(once)
        if callable(getattr(self.context, "on_unload", None)):
            return self.context.on_unload(once)

    def rollback(self):
        for cleanup in reversed(self.cleanups):
            try:
                cleanup()
            except Exception as error:
                logger.warning("registration_cleanup_unavailable category=%s", type(error).__name__)


def _other_profile_tool(*args, **kwargs):
    return json.dumps({"error": "ColonyMind is active for another profile in this process"})


def activity_capabilities():
    """Local producer health; an optional activity failure never disables capture."""
    return {"available": bool(_ACTIVITY is not None and _ACTIVITY.available)}


def flush() -> dict[str, Any]:
    """Deliver the turn outbox now, on the caller's thread.

    Capture callbacks only enqueue; the body thread delivers within moments of a
    turn. A host that ends right after a turn (a one-shot CLI run, a benchmark
    episode) can call this to hand every pending row to the sidecar first.
    """
    if _BODY is None:
        raise RuntimeError("the ColonyMind adapter is not registered")
    return _BODY.run_once()


def tick() -> dict[str, Any]:
    """One mind tick, then one body pass, on the caller's thread.

    ``POST /v1/mind/tick`` makes the sidecar form intentions now (its own
    60 s timer does the same in a gateway), then :meth:`Body.run_once` drains
    the turn outbox and runs dispatch, outbox, reconciliation and
    observations. A benchmark body tick calls this before Hermes cron and
    kanban dispatch; without the mind routes only the body pass runs.
    """
    if _BODY is None:
        raise RuntimeError("the ColonyMind adapter is not registered")
    result: dict[str, Any] = {"mind_tick": None}
    try:
        if _BODY.client.has_mind_routes() is True:
            # A forced tick waits up to 300 s for a nightly consolidation it found due.
            response = _BODY.client.post("/v1/mind/tick", timeout=420)
            result["mind_tick"] = response.json() if response.is_success else {"http": response.status_code}
    except (SidecarUnavailable, ValueError) as error:
        result["mind_tick"] = {"error": type(error).__name__}
    result.update(_BODY.run_once(mind=True))  # the caller is the dispatcher here
    return result


def register(ctx: Any) -> None:
    # Hermes discovers entry-point plugins again for each profile's manager in
    # the same gateway. Only the first owns capture, hooks and background work.
    # Later managers still need the tool declarations: kanban resolves their
    # plugin toolset keys when constructing a child's explicit --toolsets pin.
    with _REGISTRATION_LOCK:
        with _REGISTRATION_STATE_LOCK:
            registered_tools = _REGISTERED_TOOLS
        if registered_tools is not None:
            logger.warning("ColonyMind home=%s gets no ColonyMind in this process: active for another profile",
                           hermes_home().resolve())
            for registration in registered_tools:
                ctx.register_tool(**{**registration, "handler": _other_profile_tool, "check_fn": None})
            return
        evidence = _Registration(ctx)
        try:
            _register(evidence)
        except BaseException:
            evidence.rollback()
            raise


def _register(ctx: Any) -> None:
    global _BODY, _ACTIVITY, _REGISTERED_TOOLS
    settings = load_settings()
    # Observe successful public registrations, including hooks registered by
    # helpers. The facade does not alter the context or Hermes' hook dispatcher.
    evidence = ctx
    client = ColonyMindClient(settings)
    sessions = SessionMap(settings, client)
    outbox = TurnOutbox(settings.outbox_path)
    answers = Answers(client, sessions, settings, ChatLedger(ledger_path(settings)))
    body = Body(client, outbox, sessions, settings,
                ledger=BodyLedger(settings.body_ledger_path or settings.outbox_path.with_name("colonymind-body.sqlite3")),
                sends=answers)
    activity = None
    owned_tools = None
    released = False
    def release_owner():
        nonlocal released
        global _BODY, _ACTIVITY, _REGISTERED_TOOLS
        with _REGISTRATION_STATE_LOCK:
            released = True
            if _REGISTERED_TOOLS is owned_tools:
                _REGISTERED_TOOLS = None
            if _BODY is body:
                _BODY = None
            if _ACTIVITY is activity:
                _ACTIVITY = None
    # Registered first, released last, after this attempt's workers and bridge.
    ctx.on_unload(release_owner)
    answers.wake = body.wake
    capture = Capture(sessions, outbox, settings, on_enqueue=body.wake)
    body.work_capture = capture.work
    if callable(getattr(ctx, "call_mcp", None)):
        body.ideas = GoogleIdeas(client, ctx.call_mcp)
    # Host-only callback: the digest renderer reaches this exact Capture instance.
    import sys
    from types import ModuleType
    bridge = ModuleType("colonymind_capture")
    bridge.record_hand_refs = capture.record_hand_refs
    bridge.recall_exit = capture_module
    sys.modules[bridge.__name__] = bridge
    def remove_capture_bridge():
        if sys.modules.get(bridge.__name__) is bridge:
            sys.modules.pop(bridge.__name__, None)
    if callable(getattr(ctx, "on_unload", None)):
        ctx.on_unload(remove_capture_bridge)
    guard = Guard(client, sessions, settings)
    with _REGISTRATION_STATE_LOCK:
        if not released:
            _BODY = body

    # Capture first: the session map it fills is what the answer route reads, before the model runs.
    ctx.register_hook("pre_llm_call", capture.pre_llm_call)
    ctx.register_hook("pre_llm_call", answers.pre_llm_call)
    capture_module.configure_from_settings(settings)
    ctx.register_hook("post_llm_call", capture.post_llm_call)
    ctx.register_hook("post_llm_call", answers.post_llm_call)
    ctx.register_hook("pre_tool_call", guard.pre_tool_call)
    ctx.register_hook("post_tool_call", guard.post_tool_call)   # what a mind run made: a cron job, a kanban task
    ctx.register_hook("post_tool_call", answers.post_tool_call)  # the model's own send reached a chat
    ctx.register_hook("post_tool_call", capture.work.post_tool_call)
    ctx.register_hook("pre_api_request", capture.work.pre_api_request)
    ctx.register_hook("post_api_request", capture.work.post_api_request)
    ctx.register_hook("api_request_error", capture.work.api_request_error)
    ctx.register_hook("subagent_start", capture.work.subagent_start)
    ctx.register_hook("subagent_start", guard.subagent_start)
    native_settlement.register(ctx, capture)
    ctx.register_hook("on_interim_message", answers.on_interim_message)
    ctx.register_hook("transform_llm_output", answers.transform_llm_output)
    ctx.register_hook("gateway_platform_event", answers.gateway_platform_event)
    ctx.register_hook("on_kanban_dispatch_tick", body.on_dispatch_tick)
    ctx.register_hook("on_skill_lifecycle", capture.skill_loaded)
    # A message parked behind a busy session runs these dispatch hooks too (busy_ingress), once per message.
    busy_ingress.register(ctx)
    ctx.register_hook("pre_gateway_dispatch", commands.gate(sessions, answers))
    # After the /mind gate: a secret in a gateway message is sealed before Hermes sees it; the owner's tool,
    # the fills and masks on tool calls and Hermes' redaction patterns come with it (docs/SECRETS.md).
    secrets_module.register(ctx, client, sessions, settings, TOOLSET)

    ctx.register_command("mind", commands.handler(client, settings, outbox, body),
                         description="ColonyMind mind: status, log, why <id>, asks, off",
                         args_hint="[status|log|why <id>|asks|off]")

    tools = Tools(client, sessions, settings, answers)
    for schema, tool_handler in tools.handlers():
        availability = {"check_fn": tools.consult_available} if schema["name"] == "colonymind_consult" else {}
        ctx.register_tool(name=schema["name"], toolset=TOOLSET, schema=schema, handler=tool_handler, **availability)
    ctx.register_tool(name=REMINDER_SCHEMA["name"], toolset=TOOLSET, schema=REMINDER_SCHEMA,
                      handler=Reminders(client, sessions).handle)

    ctx.register_system_prompt_section("colonymind", prompt_section(settings, client, sessions))

    # All existing seams register before optional activity setup. No activity
    # storage or network work runs on this thread, even for Kanban workers.
    try:
        activity = Activity(settings.outbox_path, ColonyMindClient(settings), profile=settings.hermes_home.resolve())
        leases = [ctx.register_hook(name, getattr(activity, name)) for name in (
            "pre_llm_call", "post_llm_call", "pre_tool_call", "post_tool_call",
            "pre_approval_request", "post_approval_response")]
        if any(lease is None for lease in leases):
            # A rejected/abandoned public registration must not spawn a worker.
            raise RuntimeError("activity_hooks_unregistered")
        capture.activity_observer = activity.native_turn_settled
        if callable(getattr(ctx, "on_unload", None)):
            ctx.on_unload(activity.close)
        activity.start()
        with _REGISTRATION_STATE_LOCK:
            if not released:
                _ACTIVITY = activity
    except Exception as error:
        logger.warning("activity_initialization_unavailable category=%s", type(error).__name__)

    # Workers never run the body; a host that drives ticks itself (the paired benchmark) sets
    # COLONYMIND_BODY_THREAD=0 and calls tick() or flush() instead.
    if not os.environ.get("HERMES_KANBAN_TASK") and os.environ.get("COLONYMIND_BODY_THREAD", "1") != "0":
        on_unload = getattr(ctx, "on_unload", None)
        if callable(on_unload):
            on_unload(body.stop)
        body.start()
    evidence.write(settings.home)
    ArchiveWorker(settings, ColonyMindClient(settings), sessions).start(ctx)
    # A timed-out loader can finish after Hermes stopped accepting registrations.
    with _REGISTRATION_STATE_LOCK:
        accepted = "pre_tool_call" in evidence.hooks and not released
        if accepted:
            owned_tools = _REGISTERED_TOOLS = tuple(evidence.tools)
    if not accepted:
        evidence.rollback()
        return
    logger.info("ColonyMind adapter %s registered", __version__)


__all__ = ["HOOKS", "NOTES", "TOOLSET", "__version__", "flush", "prompt_section", "register", "tick"]
