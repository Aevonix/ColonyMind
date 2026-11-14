"""ColonyMind memory provider plugin for Hermes."""
# Hermes also loads installed providers under a synthetic source namespace.
# Reuse the canonical class only when it is this exact source. A selected
# standalone profile must not silently import a different installed version.
from importlib.util import find_spec
from pathlib import Path
from time import monotonic

_canonical = find_spec("colonymind_memory")
if _canonical is not None and _canonical.origin and Path(_canonical.origin).resolve() == Path(__file__).resolve():
    from colonymind_memory.provider import ColonyMindMemoryProvider, is_trivial_prompt
else:
    from .provider import ColonyMindMemoryProvider, is_trivial_prompt

__all__ = ["ColonyMindMemoryProvider"]


def _clock_message(message):
    """Use the host's recall query, including text parts only where it recalls list turns."""
    try:
        from agent.turn_context import _memory_query_text
    except ImportError:
        return message if isinstance(message, str) else ""
    return _memory_query_text(message)


def register(ctx):
    """Register the ColonyMind memory provider + a pre_llm_call lifecycle hook.

    The hook lives HERE (a Hermes plugin), not in Hermes core, so it survives
    Hermes updates — Hermes core lives under hermes-agent/ and is replaced on
    update, while ~/.hermes/plugins/ is not.

    Hermes injects returned ``str`` / ``{"context": str}`` into the user
    message's API content, which it also retains for later history replay.
    ``pre_llm_call`` is the only lifecycle hook that carries the message sender
    (``sender_id``) and ``platform``, so BOTH contact resolution and current-time
    injection happen here (the old ``agent:start`` hook is not a valid hook name
    in this Hermes build and was silently dropped).

    - resolve the real contact from the sender → per-contact memory and affect
      engage instead of 'default'
    - attach a clock and owner continuity to trivial turns, where Hermes skips recall
    """
    ctx.register_memory_provider(ColonyMindMemoryProvider())

    def _pre_llm_call(**kwargs):
        # Collection replaces this hook whenever another agent (or the gateway's
        # identity cache) loads a provider. Resolve the running agent's instance,
        # not the instance created by that last registration. Hermes binds this
        # execution context around run_conversation, including hook workers.
        from agent.subagent_lifecycle import get_active_subagent_parent

        agent = get_active_subagent_parent()
        if agent is None or agent.session_id != kwargs.get("session_id"):
            return None
        manager = getattr(agent, "_memory_manager", None)
        provider = next((p for p in manager.providers
                         if isinstance(p, ColonyMindMemoryProvider)), None) if manager else None
        if provider is None:
            return None
        deadline = monotonic() + 1.5
        message = _clock_message(kwargs.get("user_message"))
        trivial = is_trivial_prompt(message)
        # Memory hooks may precede capture and the front. Defer an internal notification's contact lookup until
        # prefetch, when every hook has run and the front can attest whether it is a completion review.
        history = kwargs.get("conversation_history")
        current = history[-1] if isinstance(history, (list, tuple)) and history else {}
        internal = (isinstance(current, dict) and current.get("role") == "user"
                    and current.get("display_kind") == "internal_notification")
        # Hermes 0.22 sets the turn's author on the agent before this hook and hands it to the provider's
        # on_turn_start only after it: a message a bot placed in the owner's chat must not get the owner's clock.
        try:
            provider.turn_author(getattr(agent, "_turn_author", None))
        except Exception:
            pass
        # 1) Resolve the real ColonyMind contact from the message sender (cached per
        #    sender inside the provider) so per-contact memory engages.
        try:
            provider.resolve_contact(
                platform=str(kwargs.get("platform", "") or ""),
                user_id=str(kwargs.get("sender_id", "") or ""),
                session_id=str(kwargs.get("session_id", "") or ""),
                turn_id=str(kwargs.get("turn_id", "") or ""),
                **({"warm": False} if internal or trivial else {}),
            )
        except Exception:
            pass
        # 2) A clock only for the turns whose recalled context will not carry one (a trivial
        #    prompt, where Hermes skips recall, or a turn with no bound participant). Hermes
        #    replays this text as history on every later turn. Owner greetings also carry
        #    the small recent-conversation section from the sidecar's temporal read.
        try:
            context = provider.turn_clock(
                session_id=str(kwargs.get("session_id", "") or ""),
                message=message,
                **({"deadline": deadline} if trivial else {}),
                **({"lookup_contact": False} if internal else {}),
            )
            if context:
                return {"context": context}
        except Exception:
            pass
        return None

    if hasattr(ctx, "register_hook"):
        try:
            from hermes_cli.plugins import VALID_HOOKS as _VALID
        except Exception:
            _VALID = None
        if _VALID is None or "pre_llm_call" in _VALID:
            try:
                ctx.register_hook("pre_llm_call", _pre_llm_call)
            except Exception:
                pass
