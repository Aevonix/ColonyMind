"""Section 9.3 action sequences and safety assertions, shared with clone replay.

Counts in the stimuli are fixture sizes, not operating limits. The owner-first
bound is the measured no-review turn-start baseline median plus the design's
1.5-second margin. Stage 1 has one hand and two native slots, including cron headroom.
"""

from dataclasses import dataclass
import math
from statistics import median


class SafetyViolation(AssertionError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class MissingObservation(SafetyViolation):
    pass


@dataclass(frozen=True)
class Scenario:
    number: int
    title: str
    guard: str
    assertion: str
    steps: tuple[str, ...]


SCENARIOS = (
    Scenario(1, "One-slot contention", "dispatch admission", "slot_limit", (
        "Set one hand slot and two Hermes slots; retain owner parents in two sessions.",
        "Race owner calls, a queued escalation, a continuation and manual cron.",
        "Observe peak hand count, native inline calls and every admission answer.")),
    Scenario(2, "Cancel during a tool", "cancel propagation", "cancel_release", (
        "Start a hand with a running local subprocess and a model stream.",
        "Record process identity, engine running count and stream identity.",
        "Cancel through the owner control; observe process exit, stream closure and engine release.")),
    Scenario(3, "Restart with an open effect", "unreconciled effect gate", "unknown_effect", (
        "Record a tool intent without its outcome, then restart the gateway epoch.",
        "Reconcile against the durable Hermes ledger, inspect the task probe and digest.",
        "Try an unreconciled continuation; count any repeated executor action.")),
    Scenario(4, "Duplicate completion", "completion idempotency", "completion_once", (
        "Finish an artifact against its predeclared acceptance check at E3.",
        "Deliver the same completion twice, including after reopening durable state.",
        "Count result transitions, autobiography entries and completion notices for that hand.")),
    Scenario(5, "Uncertain notice delivery", "stable notice delivery id", "notice_once", (
        "Deliver a home-channel notice but lose the platform response.",
        "Reopen the notice ledger and retry the same delivery.",
        "Observe one platform line, the same delivery id and a real send receipt.")),
    Scenario(6, "Revised goal", "intent version comparison", "stale_result", (
        "Start version 1, then revise its goal while the hand runs.",
        "Complete version 1 and render the current digest.",
        "Observe superseded status and that the current task is not relayed as done.")),
    Scenario(7, "Exclusive repository writer", "resource lock", "exclusive_writer", (
        "Plan two coding hands touching the same repository with otherwise free capacity.",
        "Try both writers; then release the first and sweep the second reservation.",
        "Observe queued lock reason and ordered, noninterleaved writes.")),
    Scenario(8, "Engine fallback", "actual-route modality guard", "fallback_audio", (
        "Start on a contracted audio-capable provider, fail the engine mid-hand.",
        "Use a contract-valid text fallback or park when none exists.",
        "Observe the actual providers, families_used and payloads at every request.")),
    Scenario(9, "Sidecar outage", "durable spool acknowledgement", "replay_once", (
        "Fail two controller calls and exercise owner-only, one-hand degraded admission.",
        "Complete work locally with its transitions spooled; restore the sidecar.",
        "Replay twice and observe one controller write per key and no repeated execution.")),
    Scenario(10, "Owner preempts review", "owner-first review redirect", "owner_first", (
        "Measure at least three no-review owner-message arrival-to-turn-start samples "
        "in the same run on the same clone, using the same owner message and pre_llm_call instrument.",
        "Start a completion review in a messaging session using busy mode queue.",
        "Deliver a real owner message through the busy-session path, which queues it before "
        "pre_gateway_dispatch runs.",
        "Observe that the review is redirected or its pending request is dropped, and the owner's turn runs next, "
        "with no intervening turn, "
        "starting within the no-review baseline median plus 1.5 seconds of arrival.")),
    Scenario(11, "Blocking question", "awaiting-input tool gate", "question_stops_tools", (
        "Run colonymind_ask with blocking true inside a hand.",
        "Try another tool through the child pre_tool_call hook.",
        "Observe awaiting_input and zero subsequent tool executions.")),
    Scenario(12, "Changed approved file", "permit target comparison", "stale_permit", (
        "Hold a structured floor call and record a fixture owner yes bound to its file hash.",
        "Change the file before the pre-call permit check recomputes its targets.",
        "Observe stale old permit, a distinct waiting ask, and no effect execution.")),
    Scenario(13, "Multi-turn recovery", "latest-intent acceptance", "latest_intent", (
        "Start one owner task, correct it twice and keep every intent version.",
        "Return conflicting results from older attempts across an engine outage and recovery.",
        "Complete the latest intent, communicate it, and cancel any abandoned obligations.",
        "Observe every obligation in creation order; a revision closes one as superseded and opens its successor.")),
    Scenario(14, "Unsupported platform", "platform allow-list", "platform_refusal", (
        "With only launch platforms admitted, request hands on RCS, API, voice and an unknown platform.",
        "Record each refusal reason and continue the owner conversation.",
        "Observe zero native dispatches and zero inline delegation.")),
)


def check(number, observed):
    """Assert observations, never accept a driver-supplied pass boolean."""
    def need(*names):
        if not isinstance(observed, dict) or any(name not in observed for name in names):
            raise MissingObservation("missing observation: " + ", ".join(names))
        counts = {"peak_hands", "inline", "answers", "engine_before", "engine_after", "retries",
                  "results", "entries", "notices", "lines", "attempts", "version", "audio_parts",
                  "pending", "duplicates", "executions", "executed", "final_version", "dispatches"}
        for name in names:
            value = observed[name]
            if name in counts and (type(value) is not int or value < 0):
                raise MissingObservation("invalid observation: " + name)
            if name == "elapsed" and (type(value) not in (int, float) or not math.isfinite(value)):
                raise MissingObservation("invalid observation: elapsed")

    def require(condition, code):
        if not condition:
            raise SafetyViolation(code)

    if number == 1:
        need("peak_hands", "inline", "answers", "paths")
        require(observed["peak_hands"] <= 1 and observed["inline"] == 0, "slot_limit")
        require(observed["answers"] >= 4 and set(observed["paths"]) ==
                {"owner-a", "owner-b", "escalation", "continuation"}, "contention_coverage")
    elif number == 2:
        need("process_gone", "stream_closed", "engine_before", "engine_after", "state")
        require(observed["process_gone"] is True and observed["stream_closed"] is True
                and observed["state"] == "stopped", "cancel_release")
        require(observed["engine_before"] > observed["engine_after"] >= 0, "engine_release")
    elif number == 3:
        need("state", "retries", "continue_refused", "probe_shown")
        require(observed["state"] == "unknown_effect" and observed["retries"] == 0
                and observed["continue_refused"] is True, "unknown_effect")
        require(observed["probe_shown"] is True, "reconciliation_probe")
    elif number == 4:
        need("results", "entries", "notices", "evidence", "entry_hand", "same_result")
        require(observed["results"] == observed["entries"] == observed["notices"] == 1
                and observed["same_result"] is True,
                "completion_once")
        require(observed["evidence"] == "E3" and bool(observed["entry_hand"]), "completion_evidence")
    elif number == 5:
        need("lines", "delivery_ids", "receipt", "attempts")
        require(observed["lines"] == 1 and len(set(observed["delivery_ids"])) == 1
                and observed["attempts"] >= 2, "notice_once")
        require(isinstance(observed["receipt"], str) and bool(observed["receipt"]), "notice_receipt")
    elif number == 6:
        need("version", "outcome", "task_state", "relay_done")
        require(observed["version"] == 2 and observed["outcome"] == "superseded"
                and observed["task_state"] != "done" and observed["relay_done"] is False, "stale_result")
    elif number == 7:
        need("second_admit", "reason", "writes", "promoted")
        require(observed["second_admit"] == "queued" and observed["reason"] == "lock"
                and observed["writes"] == ["first:start", "first:end", "second:start", "second:end"]
                and observed["promoted"] is True, "exclusive_writer")
    elif number == 8:
        need("fallback_valid", "families", "audio_parts", "fallback_called")
        require(observed["fallback_valid"] is True and observed["fallback_called"] is True,
                "fallback_contract")
        require(set(observed["families"]) == {"audio-family", "text-family"}, "fallback_families")
        require(observed["audio_parts"] == 0, "fallback_audio")
    elif number == 9:
        need("degraded", "peak_hands", "own_refused", "pending", "duplicates", "executions")
        require(observed["degraded"] is True and observed["peak_hands"] == 1
                and observed["own_refused"] is True, "degraded_admission")
        require(observed["pending"] == 0 and observed["duplicates"] == 0
                and observed["executions"] == 1, "replay_once")
    elif number == 10:
        need("next_turn", "elapsed", "redirected", "baseline")
        samples = observed["baseline"]
        if (not isinstance(samples, list) or len(samples) < 3
                or any(type(sample) not in (int, float) or not math.isfinite(sample) or sample < 0
                       for sample in samples)):
            raise MissingObservation("invalid observation: baseline")
        if observed["next_turn"] not in ("owner", "review") or type(observed["redirected"]) is not bool:
            raise MissingObservation("invalid observation: next_turn, redirected")
        request_dropped = observed.get("request_dropped", False)
        if type(request_dropped) is not bool:
            raise MissingObservation("invalid observation: request_dropped")
        # The clone gate records seconds at millisecond resolution.
        bound = round(median(samples) + 1.5, 3)
        require(observed["next_turn"] == "owner" and 0 <= observed["elapsed"] <= bound
                and (observed["redirected"] is True or request_dropped is True), "owner_first")
    elif number == 11:
        need("activity", "executed", "blocked")
        require(observed["activity"] == "awaiting_input" and observed["executed"] == 0
                and observed["blocked"] is True, "question_stops_tools")
    elif number == 12:
        need("old_state", "ask_ids", "new_state", "executed", "old_hash", "new_hash")
        require(observed["old_state"] == "stale" and len(set(observed["ask_ids"])) == 2
                and observed["new_state"] == "waiting" and observed["executed"] == 0
                and observed["old_hash"] != observed["new_hash"], "stale_permit")
    elif number == 13:
        need("versions", "conflicts", "outage", "recovered", "final_version", "final_goal",
             "expected_goal", "obligations")
        require(observed["versions"] == [1, 2, 3] and observed["conflicts"] == ["superseded"] * 2
                and observed["outage"] is True and observed["recovered"] is True
                and observed["final_version"] == 3 and observed["final_goal"] == observed["expected_goal"],
                "latest_intent")
        # Section 4.4: a revision closes the task's obligation as superseded and
        # opens a successor. Every obligation therefore ends communicated or
        # cancelled, or superseded by a later one; the chain's last must end
        # communicated or cancelled (the integrator's reading of 9.3 with 4.4).
        chain = observed["obligations"]
        require(isinstance(chain, list) and bool(chain) and chain[-1] in {"communicated", "cancelled"}
                and set(chain[:-1]) <= {"communicated", "cancelled", "superseded"}, "obligations_closed")
    elif number == 14:
        need("refused", "reasons", "dispatches", "inline", "front_continued")
        launched = observed.get("launched_platforms", [])
        require(isinstance(launched, list) and all(isinstance(p, str) for p in launched)
                and set(launched) <= {"whatsapp", "sms", "rcs"}, "launch_platforms")
        unsupported = {"rcs", "api", "voice", "unknown"} - set(launched)
        require(set(observed["refused"]) == unsupported
                and observed["reasons"] == ["platform"] * len(unsupported) and observed["dispatches"] == 0
                and observed["inline"] == 0 and observed["front_continued"] is True, "platform_refusal")
    else:
        raise ValueError("unknown scenario")
