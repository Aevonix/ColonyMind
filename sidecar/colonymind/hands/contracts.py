"""Validate every provider in a hand's route before it can be reserved."""

from math import isfinite

from pydantic import TypeAdapter, ValidationError

from .types import RouteContract


def validate_contract(value):
    contract = TypeAdapter(RouteContract).validate_python(value)
    if (not contract.version or not contract.family or "text" not in contract.modalities
            or contract.context_tokens <= 0 or contract.max_output_tokens <= 0
            or contract.max_output_tokens > contract.context_tokens
            or not isfinite(contract.decode_tok_s) or contract.decode_tok_s <= 0
            or not 0 <= contract.tool_calls.success_share <= 1
            or contract.admission.family != contract.family):
        raise ValueError("invalid route contract")
    return contract


def validate_chain(chain, configured, *, needs, kind, input_tokens, growth_tokens=0,
                   excluded_families=(), effort=2):
    """Return eligible providers and a reason for each dropped provider.

    Assumption: all kinds except deep, plan and synthesize require qualified
    tool calls. An explicit code, shell or web need always requires tools.
    The caller supplies context growth; no unmeasured provider limit is inferred.
    """
    valid, dropped = [], {}
    for provider in dict.fromkeys(chain):
        try:
            contract = validate_contract(configured[provider])
            required = {"text"}
            if "vision" in needs:
                required.add("image")
            if "audio" in needs:
                required.add("audio")
            tools = kind not in {"deep", "plan", "synthesize"} or bool(set(needs) & {"code", "shell", "web"})
            if not required.issubset(contract.modalities):
                reason = "modality"
            elif tools and not contract.tool_calls.qualified:
                reason = "tools"
            elif input_tokens + growth_tokens + contract.max_output_tokens > contract.context_tokens:
                reason = "context"
            elif contract.family in excluded_families:
                reason = "producer_family"
            elif effort not in contract.effort:
                reason = "effort"
            else:
                valid.append(provider)
                continue
        except (KeyError, TypeError, ValueError, ValidationError):
            reason = "contract"
        dropped[provider] = reason
    return valid, dropped
