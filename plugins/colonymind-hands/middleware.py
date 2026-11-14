"""Per-request identity, modality and actual-producer accounting for hands."""
from __future__ import annotations

from copy import deepcopy
import logging
import sys
import time
from types import SimpleNamespace
from typing import Any

from colonymind_hand_context import render_identity

from .budgets import (FINAL_REPORT_TOKENS, WAITING, TERMINAL, is_child_context, limits, resolve_hand, settings,
                      stop, stopped_text)

logger = logging.getLogger(__name__)
IDENTITY_ANCHOR = "You are a focused subagent working on a specific delegated task."
_AUDIO_NOTE = {"type": "text", "text": "[audio omitted: use the transcript]"}


def contract_for(runtime: Any, hand: Any = None, provider: str = "", model: str = "",
                 base_url: str = "", **_: Any) -> tuple[str, dict[str, Any]]:
    """Match the actual route; never substitute the planned primary after fallback."""
    contracts = settings(runtime).get("contracts", {})
    contract = contracts.get(provider)
    if isinstance(contract, dict):
        return provider, contract
    if hand is not None:
        routes = hand.plan.get("routes", [])
        route_matches = []
        for route in routes:
            actual = {"provider": provider, "model": model, "base_url": base_url}
            exact = route.get("model") and all(not route.get(key) or route[key] == actual[key] for key in actual)
            endpoint = model and base_url and route.get("model") == model and route.get("base_url") == base_url
            if exact or endpoint:
                reference = route.get("contract", {})
                name = reference.get("provider") if isinstance(reference, dict) else reference
                if isinstance(contracts.get(name), dict):
                    route_matches.append((name, contracts[name]))
        if len(route_matches) == 1:
            return route_matches[0]
    matches = []
    for name, candidate in contracts.items():
        if not isinstance(candidate, dict):
            continue
        identity = {key: candidate[key] for key in ("provider", "model", "base_url") if candidate.get(key)}
        if identity and all({"provider": provider, "model": model, "base_url": base_url}[key] == value for key, value in identity.items()):
            matches.append((name, candidate))
    return matches[0] if len(matches) == 1 else (provider, {})


def release_request(runtime: Any, hand: Any, api_request_id: str = "") -> bool:
    registry = runtime.registry
    with registry.requests:
        if api_request_id and hand.request_id and api_request_id != hand.request_id:
            return False
        registry.inflight.pop(hand.hand_id, None)
        hand.request_family = ""
        registry.requests.notify_all()
        return True


def _admit_request(runtime: Any, hand: Any, family: str, api_request_id: str = "", final: bool = False) -> bool:
    """Wait for a provider slot. A stopped hand gets one only for its claimed final report."""
    registry = runtime.registry
    health = settings(runtime).get("health", {})
    cap = int(health.get("family_caps", {}).get(family, health.get("default_family_cap", 1)))
    priority = hand.plan.get("priority", "owner")
    # hands.queue.deadline_seconds; the fallbacks repeat its defaults (assumptions).
    deadlines = {"owner": 900, "own": 1800, "auto": 3600, "upkeep": 3600}
    deadlines.update(settings(runtime).get("queue", {}).get("deadline_seconds", {}))
    timeout = float(deadlines.get(priority, deadlines["own"]))
    deadline = time.monotonic() + timeout
    with registry.requests:
        # Native retries rebuild their request; the old attempt no longer owns a slot.
        registry.inflight.pop(hand.hand_id, None)
        while True:
            closed = getattr(runtime, "_closed", None)
            ended = (not _final_call(hand, api_request_id) if final else
                     (hand.blocked and hand.activity not in WAITING) or hand.stop_reason or hand.status in TERMINAL)
            if ended or (closed and closed.is_set()):
                hand.request_refused = True
                return False
            if sum(value == family for value in registry.inflight.values()) < cap:
                registry.inflight[hand.hand_id] = family
                hand.request_family = family
                hand.request_id = api_request_id
                hand.request_refused = False
                return True
            if time.monotonic() >= deadline:
                hand.request_refused = True
                break
            if hand.activity not in WAITING:
                hand.activity = "queued"
                hand.activity_since = time.monotonic()
            registry.requests.wait(timeout=0.1)
    stop(runtime, hand, "queue_timeout")
    return False


def _strip_audio(value: Any) -> tuple[Any, int]:
    if isinstance(value, list):
        items = [_strip_audio(item) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    if isinstance(value, dict):
        if value.get("type") in {"input_audio", "audio", "audio_url"}:
            return dict(_AUDIO_NOTE), 1
        if value.get("type") == "file" and str(value.get("mime_type", "")).startswith("audio/"):
            return dict(_AUDIO_NOTE), 1
        result, count = {}, 0
        for key, item in value.items():
            result[key], stripped = _strip_audio(item)
            count += stripped
        return result, count
    return value, 0


def _identity(runtime: Any, hand: Any) -> str:
    brief = hand.plan.get("brief", {})
    supplied = hand.plan.get("identity") or (brief.get("identity") if isinstance(brief, dict) else None)
    config = settings(runtime)
    supplied = supplied or config.get("persona_addendum", "")
    name = str(hand.plan.get("name") or hand.hand_id)
    return render_identity(name=name, persona_addendum=supplied, style=config.get("style", ""))


def _replace_anchor(request: dict, identity: str) -> bool:
    def replace_content(content: Any) -> tuple[Any, bool]:
        if isinstance(content, str) and IDENTITY_ANCHOR in content:
            return content.replace(IDENTITY_ANCHOR, identity, 1), True
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and IDENTITY_ANCHOR in part["text"]:
                    part["text"] = part["text"].replace(IDENTITY_ANCHOR, identity, 1)
                    return content, True
        return content, False

    if "system" in request:
        request["system"], changed = replace_content(request["system"])
        if changed:
            return True
    if "instructions" in request:
        request["instructions"], changed = replace_content(request["instructions"])
        if changed:
            return True
    for key in ("messages", "input"):
        messages = request.get(key, [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                message["content"], changed = replace_content(message.get("content"))
                if changed:
                    return True
    return False


def _front_request(payload: dict[str, Any], session_id: str, hand: Any = None, **ids: Any) -> dict[str, Any]:
    """Apply ledger-selected front caps in the hands request callback.

    This runs after the hand guard even on its conservative error path. A separate
    callback would lose either these caps or the hand guard's rewrite in native middleware.
    """
    try:
        bridge = sys.modules.get("colonymind_hands_front")
        if hand is not None or bridge is None or is_child_context(**ids):
            return payload
        caps = bridge.request_caps(session_id, str(ids.get("turn_id") or ""))
        if not isinstance(caps, dict):
            return payload
        cap = int(caps["max_tokens"])
        if cap <= 0:
            return payload
        token_fields = ("max_tokens", "max_completion_tokens", "max_output_tokens")
        extra = payload.get("extra_body")
        if not isinstance(extra, dict):
            extra = {}
        # SDK extra_body values may replace top-level fields on the wire.
        # Preserve the strictest existing bound in either representation.
        cap = min([cap] + [body[key] for body in (payload, extra) for key in token_fields
                           if type(body.get(key)) is int and body[key] > 0])
        for body in (payload, extra):
            for key in token_fields:
                if key not in body:
                    continue
                body[key] = cap
        # A caller that sent no bound keeps none. Inserting one here would turn a restriction into a
        # permission and break the "never raise a bound" rule above, and an invented ceiling can 400 a
        # route whose own limit is lower. The route's configured default applies instead.
        # Thinking is not forced off: chat_template_kwargs.enable_thinking is left unchanged.
        # Nothing is written in its place: the route's own template
        # default decides, so a caller that wants thinking off can still ask for it.
        if caps.get("duplicate") is True:
            for body in (payload, extra):
                for key in ("tools", "tool_choice", "parallel_tool_calls"):
                    body.pop(key, None)
    except Exception:
        logger.exception("front request caps unavailable")
    return payload


def _defer_review(session_id: str, hand: Any = None, **ids: Any) -> bool:
    if hand is not None or is_child_context(**ids):
        return False
    bridge = sys.modules.get("colonymind_hands_front")
    defer = getattr(bridge, "defer_review_request", None)
    try:
        return callable(defer) and defer(session_id, str(ids.get("turn_id") or "")) is True
    except Exception:
        logger.exception("review request deferral unavailable")
        return False


def _terminal_response(text: str, api_mode: str = "", model: str = "", *, response_id: str) -> Any:
    """A local terminal response in the native transport's raw response shape.

    No usage is reported: no provider ran. Returning None or raising from an
    execution callback would trigger native retries or fail-open execution.
    """
    ns = SimpleNamespace
    fields = {"id": response_id, "model": model, "usage": None}
    if api_mode == "anthropic_messages":
        return ns(**fields, content=[ns(type="text", text=text)], stop_reason="end_turn")
    if api_mode == "codex_responses":
        return ns(**fields, status="completed", output=[ns(type="message", role="assistant",
            status="completed", content=[ns(type="output_text", text=text)])])
    return ns(**fields, choices=[ns(message=ns(content=text, tool_calls=None), finish_reason="stop")])


def _silent_review_response(api_mode: str = "", model: str = "") -> Any:
    return _terminal_response("[SILENT]", api_mode, model, response_id="colonymind-review-deferred")


def _held_child(hand: Any) -> bool:
    return (hand is not None and hand.blocked and hand.activity == "awaiting_approval"
            and not hand.stop_reason and hand.status not in TERMINAL)


def _claim_final_report(hand: Any, request_id: str) -> bool:
    """Give this request the stopped hand's one final report call, if that call is still open.

    A claim that produced no response (a provider error before any reply) passes to Hermes'
    own bounded retry. Once a response arrives the claim is used and later calls are refused.
    """
    with hand.lock:
        if hand.final_report not in {"allowed", "claimed"}:
            return False
        hand.final_report = "claimed"
        hand.final_report_request = request_id
        return True


def _final_call(hand: Any, request_id: str) -> bool:
    return hand.final_report == "claimed" and hand.final_report_request == request_id


def _cap_report(payload: dict[str, Any], cap: int) -> None:
    """Bound the final report's output in every field the request uses; never raise a bound."""
    fields = ("max_tokens", "max_completion_tokens", "max_output_tokens")
    extra = payload.get("extra_body")
    bodies = [payload] + ([extra] if isinstance(extra, dict) else [])
    present = False
    for body in bodies:
        for key in fields:
            if key in body:
                present = True
                current = body[key]
                body[key] = min(current, cap) if type(current) is int and current > 0 else cap
    config = payload.get("inferenceConfig")
    if isinstance(config, dict):
        present = True
        current = config.get("maxTokens")
        config["maxTokens"] = min(current, cap) if type(current) is int and current > 0 else cap
    if not present and "modelId" not in payload:
        payload["max_output_tokens" if "input" in payload and "messages" not in payload else "max_tokens"] = cap


def llm_request(runtime: Any, request: dict[str, Any], session_id: str = "", **ids: Any) -> dict[str, Any]:
    """Hermes middleware must return a request envelope and must never raise."""
    hand = None
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        try:
            _, contract = contract_for(runtime, hand, **ids)
        except Exception:
            logger.exception("hand route contract unavailable; stripping audio")
            contract = {}
        payload = deepcopy(request)
        if "audio" not in contract.get("modalities", []):
            payload, count = _strip_audio(payload)
            if hand is not None and count:
                with hand.lock:
                    hand.plan["audio_strips"] = hand.plan.get("audio_strips", 0) + count
        if hand is not None:
            # A token budget stop leaves one bounded call for the hand's final report.
            final = bool(hand.stop_reason) and _claim_final_report(hand, str(ids.get("api_request_id") or ""))
            if hand.activity in WAITING or final:
                for key in ("tools", "tool_choice", "parallel_tool_calls"):
                    payload.pop(key, None)
            if final:
                _cap_report(payload, int(settings(runtime).get("budgets", {}).get(
                    "final_report_tokens", FINAL_REPORT_TOKENS)))
            # A held child finishes through the native loop with a local terminal
            # response. It needs neither another model report nor a provider slot.
            admitted = False if _held_child(hand) else _admit_request(
                runtime, hand, str(contract.get("family") or "unknown:" + ids.get("provider", "")),
                str(ids.get("api_request_id") or ""), final=final)
            with hand.lock:
                if not _replace_anchor(payload, _identity(runtime, hand)) and not hand.anchor_warned:
                    hand.anchor_warned = True
                    logger.warning("hand %s: identity anchor absent from system content", hand.hand_id)
                if admitted and hand.activity not in WAITING:
                    hand.activity = "generating"
                    hand.activity_since = time.monotonic()
                    hand.stall_steered_at = None
                    output = payload.get("max_tokens", payload.get("max_output_tokens", contract.get("max_output_tokens", 0)))
                    speed = float(contract.get("decode_tok_s") or 0)
                    hand.expected_seconds = float(output or 0) / speed if speed > 0 else 0
    except Exception:
        logger.exception("hand request middleware failed; stripping audio conservatively")
        if hand is not None:
            hand.request_refused = True
        # Keep modality enforcement even when identity or accounting cannot run.
        try:
            payload, _ = _strip_audio(request)
        except Exception:
            payload = {"messages": [{"role": "user", "content": "[request omitted: hand guard unavailable]"}]}
    deferred = _defer_review(session_id, hand, **ids)
    envelope = {"request": _front_request(payload, session_id, hand, **ids), "source": "colonymind-hands"}
    if deferred:
        envelope["reason"] = "review_deferred_for_owner"
    return envelope


def _count(value: Any) -> int:
    # A malformed count raises: the caller stops the hand as accounting_unavailable.
    return 0 if value is None else max(0, int(value))


# Prompt tokens a provider served from its prefix cache, in the order Hermes reads them
# (agent/usage_pricing.py): its canonical summary, OpenAI chat, Responses, Anthropic, DeepSeek, Kimi.
_CACHED = (("cache_read_tokens",), ("prompt_tokens_details", "cached_tokens"),
           ("input_tokens_details", "cached_tokens"), ("cache_read_input_tokens",),
           ("prompt_cache_hit_tokens",), ("cached_tokens",))


def _usage_tokens(usage: Any) -> tuple[int, int]:
    """Charged and raw tokens for one response.

    raw is every prompt token plus every output token: what the budget counted before.
    charged leaves out prompt tokens the provider read from its prefix cache, so a long
    cached context is not charged again on every call. Cache writes were prefilled and stay
    charged. Without cache fields, charged equals raw.
    """
    if not isinstance(usage, dict):
        return 0, 0
    output = _count(usage.get("completion_tokens", usage.get("output_tokens")))
    if usage.get("prompt_tokens") is not None:
        prompt = _count(usage["prompt_tokens"])  # Hermes' summary and OpenAI chat include the cache.
    elif "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        # The Anthropic wire reports uncached input_tokens beside its cache buckets.
        prompt = sum(_count(usage.get(key)) for key in
                     ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    else:
        prompt = sum(_count(usage.get(key)) for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
    raw = _count(usage["total_tokens"]) if usage.get("total_tokens") is not None else prompt + output
    cached = 0
    for path in _CACHED:
        value: Any = usage
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        cached = _count(value)
        if cached:
            break
    return max(0, raw - min(cached, prompt)), raw


def post_api_request(runtime: Any, session_id: str = "", usage: Any = None,
                     api_request_id: str = "", **ids: Any) -> None:
    hand = None
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        if hand is None:
            return
        released = release_request(runtime, hand, api_request_id)
        with hand.lock:
            if api_request_id in hand.local_requests:
                return
            if api_request_id and api_request_id in hand.counted_requests:
                return
            if api_request_id:
                hand.counted_requests.add(api_request_id)
            provider, contract = contract_for(runtime, hand, **ids)
            family = contract.get("family") or "unknown:" + provider
            version = str(contract.get("version", ""))
            hand.families_used.add(family)
            hand.contract_versions.add((provider, version))
            contracts_used = hand.plan.setdefault("contracts_used", [])
            served = {"provider": provider, "version": version, "family": family,
                      "model": str(ids.get("response_model") or ids.get("model") or "unknown")}
            if version and served not in contracts_used:
                contracts_used.append(served)
            charged, raw = _usage_tokens(usage)
            hand.tokens += charged
            hand.raw_tokens += raw
            if hand.final_report in {"claimed", "used"} and api_request_id == hand.final_report_request:
                hand.final_report = "used"
                content = getattr(ids.get("assistant_message"), "content", None)
                if isinstance(content, str) and content.strip():
                    hand.final_report_text = content
            if released and hand.activity not in WAITING:
                hand.activity = "queued"
                hand.activity_since = time.monotonic()
                hand.stall_steered_at = None
            if hand.tokens >= limits(runtime, hand)["tokens"]:
                stop(runtime, hand, "token_budget")
    except Exception:
        logger.exception("hand API accounting failed")
        if hand is not None:
            try:
                stop(runtime, hand, "accounting_unavailable")
            except Exception:
                logger.exception("cannot interrupt hand after accounting failure")


def llm_execution(runtime: Any, request: dict, next_call: Any, session_id: str = "", **ids: Any) -> Any:
    """A cancelled wait never reaches the native non-streaming transport."""
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        if _held_child(hand):
            request_id = str(ids.get("api_request_id") or "")
            release_request(runtime, hand, request_id)
            with hand.lock:
                hand.local_requests.add(request_id)
            return _terminal_response("Waiting for owner approval. The held action has not run.",
                str(ids.get("api_mode") or ""), str(ids.get("model") or ""), response_id="colonymind-hand-held")
        request_id = str(ids.get("api_request_id") or "")
        final = hand is not None and not hand.request_refused and _final_call(hand, request_id)
        if hand is not None and not final and (hand.request_refused or hand.stop_reason or
                                (hand.blocked and hand.activity not in WAITING) or hand.status in TERMINAL):
            release_request(runtime, hand, request_id)
            if not hand.stop_reason:
                return None
            # A stopped hand ends through the native loop with a local reply: its final report,
            # or why it ended. None would make Hermes retry and then fail the hand.
            with hand.lock:
                hand.local_requests.add(request_id)
                text = stopped_text(hand)
            return _terminal_response(text, str(ids.get("api_mode") or ""), str(ids.get("model") or ""),
                                      response_id="colonymind-hand-stopped")
    except Exception:
        logger.exception("hand request execution guard unavailable")
        if is_child_context(**ids):
            return None
        hand, final = None, False
    if _defer_review(session_id, hand, **ids):
        return _silent_review_response(str(ids.get("api_mode") or ""), str(ids.get("model") or ""))
    # Keep downstream exceptions outside the guard's exception handler.
    try:
        response = next_call(request)
        if final:
            with hand.lock:
                if _final_call(hand, request_id):
                    hand.final_report = "used"
        return response
    finally:
        if hand is not None:
            release_request(runtime, hand, ids.get("api_request_id", ""))


def api_request_error(runtime: Any, session_id: str = "", **ids: Any) -> None:
    try:
        hand = resolve_hand(runtime, session_id, **ids)
        if hand is not None:
            released = release_request(runtime, hand, ids.get("api_request_id", ""))
            if released and hand.activity not in WAITING:
                hand.activity = "queued"
    except Exception:
        logger.exception("hand request slot release failed")


def register(ctx: Any) -> None:
    from .dispatch import get_runtime

    def request_hook(request: dict[str, Any], **kwargs: Any) -> dict:
        try:
            return llm_request(get_runtime(ctx), request=request, **kwargs)
        except Exception:
            logger.exception("hand middleware runtime unavailable")
            return llm_request(None, request=request, **kwargs)

    def response_hook(**kwargs: Any) -> None:
        try:
            post_api_request(get_runtime(ctx), **kwargs)
        except Exception:
            logger.exception("hand middleware runtime unavailable")

    def execution_hook(request: dict, next_call: Any, **kwargs: Any) -> Any:
        try:
            runtime = get_runtime(ctx)
        except Exception:
            logger.exception("hand execution runtime unavailable")
            if is_child_context(**kwargs):
                return None
            runtime = None
        return llm_execution(runtime, request=request, next_call=next_call, **kwargs)

    def error_hook(**kwargs: Any) -> None:
        try:
            api_request_error(get_runtime(ctx), **kwargs)
        except Exception:
            logger.exception("hand middleware runtime unavailable")

    ctx.register_middleware("llm_request", request_hook)
    ctx.register_middleware("llm_execution", execution_hook)
    ctx.register_hook("post_api_request", response_hook)
    ctx.register_hook("api_request_error", error_hook)
