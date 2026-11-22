"""``colonymind recall``: what memory recalls for a question, and with ``--explain`` why (docs/RECALL-HYBRID.md,
Tracing a recall).

It asks the running sidecar's memory search as the instance key, which is the owner's credential: the owner's
diagnostic, for any person the key may search. Without ``--explain`` it prints the packet the search returns; with
it, the trace, one line per candidate: where it was found, its place in the fused order, what the reranker made of it
and whether it is in the packet or why not.

``colonymind recall-questions`` scores recall against the instance's own question set and prints one result line
(docs/RECALL-HYBRID.md, The instance's own question set).
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from numbers import Real
import sys
from typing import Any

LANES = {"lexical": "keyword", "stemmed": "stemmed keyword", "vector": "vector", "window": "date window",
         "vector_media": "vector (images)", "lexical_media": "keyword (images)"}
CHECKS = {"configuration_verified": "configuration verified",
          "configuration_verified_weights_unverified": "configuration verified, weights unverified",
          "mismatch": "the configuration does not match", "unverified": "unverified"}
EXITS = {
    "no_lane": "left out before ranking (no lane carried it there: a repeat of words already ranked, outside the "
               "question's time, or a reply shown only beside its input)",
    "below_submission_bound": "left out: below the submission bound",
    "over_session_share": "left out: the reranker's batch was spread across sessions, and other sessions' "
                          "candidates took its place",
    "below_cutoff": "left out: below the calibrated cutoff",
    "over_packet_limit": "left out: the packet was full",
    "over_char_budget": "left out: over the character budget",
    "over_host_allowance": "left out: over what the host puts inline, less the items kept whole beside the memory",
    "removed_by_recheck": "left out: its source or its corrections changed while it was ranked",
    "erased": "its source was erased during the recall (its ids are not shown)",
}


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("recall", help="What memory recalls for a question; --explain says why, stage by stage")
    parser.add_argument("question", help="The question, as a turn would ask it")
    parser.add_argument("--as", dest="person", required=True, help="Whose memory: a contact id")
    parser.add_argument("--session", default=None, help="Also admit this session's session-scoped evidence")
    parser.add_argument("--limit", type=int, default=5,
                        help="The most records in the packet (default 5, an explicit search's; a turn's packet has "
                             "no count limit behind a verified cutoff)")
    parser.add_argument("--explain", action="store_true",
                        help="For each candidate: where it was found, how it ranked, why it is in the packet or not")
    parser.add_argument("--json", action="store_true", help="Print the sidecar's answer as JSON")
    questions = sub.add_parser("recall-questions", help="Score recall against the instance's own question set, "
                                                        "built from its history when there is none: one result line")
    questions.add_argument("--new", action="store_true", help="Build the set again from the history first")


def run_questions(args: argparse.Namespace) -> int:
    """``colonymind recall-questions``: the running sidecar scores recall against the instance's question set
    (``memory.question_set``), and this prints the one result line."""
    from colonymind.mind.cli import Sidecar
    try:
        value = Sidecar(getattr(args, "instance", None)).call(
            "POST", "/v1/mind/recall-questions", json_body={"new": bool(args.new)}, timeout=960)
    except SystemExit as exc:
        if exc.code and not isinstance(exc.code, int):
            print(f"colonymind recall-questions: {exc.code}", file=sys.stderr)
            return 1
        raise
    if not value.get("questions"):
        print("recall questions: none (the history gave no question to ask)")
        return 1
    built = f" (set built: {value['built']} questions)" if value.get("built") else ""
    print(f"recall questions: {value.get('recalled', 0)} of {value['questions']} recalled{built}")
    return 0


def run(args: argparse.Namespace) -> int:
    from colonymind.mind.cli import Sidecar
    body: dict[str, Any] = {"identity": {"host_id": "colonymind-cli"}, "person_id": args.person,
                            "query": args.question, "limit": args.limit}
    if args.session:
        body["session_id"] = args.session
    if args.explain:
        body["trace"] = True
    try:
        value = Sidecar(getattr(args, "instance", None)).call("POST", "/v1/host/memory/search", json_body=body)
    except SystemExit as exc:
        if exc.code and not isinstance(exc.code, int):
            print(f"colonymind recall: {exc.code}", file=sys.stderr)
            return 1
        raise
    if args.json:
        print(json.dumps(value, indent=2, sort_keys=True))
    elif args.explain:
        print("\n".join(explain(value, person=args.person)))
    else:
        print(value.get("content") or "(nothing recalled)")
    return 0


def explain(value: dict, *, person: str) -> list[str]:
    """A traced search in plain words: a few lines about the recall, then one line per candidate."""
    trace = value.get("trace") or {}
    packet = trace.get("packet") or {}
    count = value.get("count", 0)
    held = (f"{count} record{'' if count == 1 else 's'} in the packet (no count limit)"
            if "limit" in packet and packet["limit"] is None
            else f"{count} of at most {packet.get('limit', '?')} records in the packet")
    lines = [f"Recall for {person}: {held}, within {packet.get('max_chars', '?')} characters."]
    lanes = trace.get("lanes") or {}
    lines.append("Lanes: " + (", ".join(f"{_lane(name)} found {count}" for name, count in lanes.items())
                              or "none answered") + ".")
    lines.append("Reranker: " + _reranker(trace) + ".")
    rows = trace.get("rows") or []
    if not rows:
        lines.append("No candidate was collected.")
    for number, row in enumerate(rows, 1):
        lines.append(f"{number:>3}. {_name(row)}: {_stages(row, trace)}")
    if trace.get("rows_omitted"):
        lines.append(f"({trace['rows_omitted']} more candidates are not listed.)")
    return lines


def _reranker(trace: dict) -> str:
    rerank = trace.get("rerank") or {}
    calibration = trace.get("calibration") or {}
    outcome = rerank.get("outcome")
    if outcome in (None, "off"):
        text = "off"
    elif outcome == "not_needed":
        text = "not called (every candidate fit the packet and no cutoff applied)"
    elif outcome == "missed":
        sent = rerank.get("submitted") or 0
        text = (f"missed ({rerank.get('missed')}) "
                + (f"after it was sent {sent} candidates (submission bound {rerank.get('submission_bound')})"
                   if sent else "before any candidate was sent")
                + "; the candidates kept their fused order")
    else:
        text = (f"read {rerank.get('submitted')} candidates (submission bound {rerank.get('submission_bound')})"
                + ("; shadow mode, so the order and the cutoff were not used" if outcome == "shadow" else ""))
    status = calibration.get("status") or "none"
    if status == "none":
        return text + "; no calibrated cutoff configured"
    if calibration.get("cutoff") is None:
        return text + f"; the configured cutoff was not applied (its calibration check: {CHECKS.get(status, status)})"
    applied = "" if outcome == "scored" else ", not applied"
    return text + f"; calibrated cutoff {calibration['cutoff']:.2f} ({CHECKS.get(status, status)}{applied})"


def _name(row: dict) -> str:
    if not row.get("turn_id"):
        return "(erased)"
    digest = row.get("message_hash")
    return f"{row.get('form') or 'message'} {row['turn_id']}" + (f" #{digest[:12]}" if digest else "")


def _stages(row: dict, trace: dict) -> str:
    if row.get("exit") == "erased":
        return EXITS["erased"]
    parts = [_found(row)]
    if row.get("candidate") is not None:
        parts.append(f"fused {row['fused_rank']}" if row.get("fused_rank") else
                     "not fused (found by name)" if row.get("form") == "media_name" else "not fused")
        parts.append(_reranked(row, trace))
    if row.get("exit") is None:
        parts.append("in the packet" if not row.get("shown_by") else
                     f"in the packet, through {row['shown_by'].split(':', 1)[0]} {row['shown_by'][-64:][:12]}")
    else:
        detail = EXITS.get(row["exit"], row["exit"])
        packet = trace.get("packet") or {}
        if row["exit"] == "over_packet_limit":
            detail += f" ({packet.get('limit')} records)"
        elif row["exit"] == "over_char_budget":
            detail += f" ({packet.get('max_chars')} characters)"
        elif row["exit"] in ("below_submission_bound", "over_session_share"):
            detail += f" ({(trace.get('rerank') or {}).get('submission_bound')})"
        parts.append(detail)
    return "; ".join(parts)


def _found(row: dict) -> str:
    lanes = row.get("lanes") or {}
    if not lanes:
        return ("found by name" if row.get("form") == "media_name"
                else "carried in with the evidence found (no lane found it itself)")
    return "found by " + " and ".join(f"{_lane(name)} (rank {rank})" for name, rank in lanes.items())


def _lane(name: str) -> str:
    """A lane in words: its own words when it has them, else its name read as words (a lane added later)."""
    return LANES.get(name) or name.replace("_", " ")


def _reranked(row: dict, trace: dict) -> str:
    rerank = trace.get("rerank") or {}
    outcome = rerank.get("outcome")
    if outcome in (None, "off"):
        return "not reranked (reranking is off)"
    if outcome == "not_needed":
        return "not reranked (every candidate fit)"
    if outcome == "missed":
        if row.get("submitted"):
            return f"sent to the reranker, which missed ({rerank.get('missed')}): not reranked"
        return f"not reranked (the reranker missed: {rerank.get('missed')})"
    if not row.get("submitted"):
        return "not read by the reranker"
    score = row.get("rerank_score")
    text = f"reranked {score:.2f}" if score is not None else "reranked"
    if outcome == "shadow":
        return text + " in shadow (order unchanged)"
    cutoff = (trace.get("calibration") or {}).get("cutoff")
    if cutoff is None or score is None:
        return text + " (no calibrated cutoff)"
    return text + f" (calibrated cutoff {cutoff:.2f}, {'passed' if score >= cutoff else 'below'})"


def refit_query_adapter_cutoff(
    metadata: Mapping[str, Any], query_transform: str,
    development: Iterable[Mapping[str, Any]], held_out: Iterable[Mapping[str, Any]], *,
    max_false_injection_rate: float = 0.0, min_answer_recall: float = 1.0,
    reference_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """Refit from captured reranker scores without loading an endpoint or embedding anything.

    Each row has a unique ``query_id``, aligned ``scores`` and boolean ``relevant`` lists, a boolean
    ``no_answer``, and the ``calibration_fingerprint`` of the configuration that produced its candidates
    and scores. The capture must use this query transform, including for candidate retrieval. Merely
    relabelling old captures does not requalify them. Both splits need answer and no-answer queries;
    the held-out split must contain at least 150 distinct no-answer queries.

    Only development scores choose the cutoff. One held-out check then either qualifies the record or
    returns None. Invalid evidence raises ValueError. The caller freezes the split before collecting
    scores and consumes each revision's held-out check once, including failures; this pure callable
    cannot enforce consumption across invocations. It does not publish or change a serving closure.

    Full answer recall and zero false injection are conservative builder assumptions. Relaxed bars
    require the previous closure's metrics on the same frozen holdout, and cannot regress either metric.
    Answer recall counts answerable queries with at least one relevant candidate above the cutoff;
    false injection counts no-answer queries with any candidate above it, not individual distractors.
    """
    from colonymind.memory.recall import calibration_fingerprint
    from colonymind.vector.indexes import query_transform_identity

    query_transform_identity(query_transform)
    maximum = _cutoff_rate(max_false_injection_rate, "max_false_injection_rate")
    minimum = _cutoff_rate(min_answer_recall, "min_answer_recall", positive=True)
    reference = None
    if reference_metrics is not None:
        reference = {
            "answer_recall": _cutoff_rate(reference_metrics.get("answer_recall"), "reference answer_recall"),
            "false_injection_rate": _cutoff_rate(reference_metrics.get("false_injection_rate"),
                                                 "reference false_injection_rate"),
        }
        maximum = min(maximum, reference["false_injection_rate"])
        minimum = max(minimum, reference["answer_recall"])
    elif maximum > 0 or minimum < 1:
        raise ValueError("relaxed calibration bars require reference metrics from the same held-out queries")
    configuration = json.loads(json.dumps({**metadata, "query_transform": query_transform}, allow_nan=False))
    fingerprint = calibration_fingerprint(configuration)
    dev = _cutoff_rows(development, fingerprint)
    if not dev or not any(row["no_answer"] for row in dev) or all(row["no_answer"] for row in dev):
        return None

    # A query's highest relevant score is the highest threshold that can still recall it. Between these
    # values recall cannot improve, so testing other scores would only choose a less conservative tie.
    thresholds = sorted({max((score for score, relevant in zip(row["scores"], row["relevant"]) if relevant),
                             default=-math.inf) for row in dev if not row["no_answer"]} - {-math.inf})
    qualified = []
    for cutoff in thresholds:
        metrics = _cutoff_metrics(dev, cutoff)
        if metrics["answer_recall"] >= minimum and metrics["false_injection_rate"] <= maximum:
            qualified.append((metrics["answer_recall"], -metrics["false_injection_rate"], cutoff, metrics))
    if not qualified:
        return None
    _, _, cutoff, development_metrics = max(qualified, key=lambda entry: entry[:3])

    held = _cutoff_rows(held_out, fingerprint)
    if {row["query_id"] for row in dev} & {row["query_id"] for row in held}:
        raise ValueError("development and held-out query IDs must be disjoint")
    if sum(row["no_answer"] for row in held) < 150 or all(row["no_answer"] for row in held):
        return None
    held_metrics = _cutoff_metrics(held, cutoff)
    if held_metrics["answer_recall"] < minimum or held_metrics["false_injection_rate"] > maximum:
        return None
    record = {
        "version": "recall-cutoff-1", "query_transform": query_transform, "metadata": configuration,
        "fingerprint": fingerprint, "cutoff": cutoff, "development": development_metrics,
        "held_out": held_metrics, "held_out_checks": 1,
        "development_digest": _cutoff_digest(dev), "held_out_digest": _cutoff_digest(held),
        "quality_bars": {"min_answer_recall": minimum, "max_false_injection_rate": maximum},
        "reference_metrics": reference,
    }
    return {"id": "recall-cutoff-1:" + _cutoff_digest(record), **record}


def _cutoff_rate(value: Any, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or not 0 <= value <= 1 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}rate between zero and one")
    return float(value)


def _cutoff_rows(examples: Iterable[Mapping[str, Any]], fingerprint: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("calibration rows must be mappings")
        if example.get("calibration_fingerprint") != fingerprint:
            raise ValueError("captured scores must match this query transform's calibration fingerprint")
        query_id = example.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen:
            raise ValueError("calibration query IDs must be nonempty and unique within each split")
        seen.add(query_id)
        no_answer = example.get("no_answer")
        scores, relevant = example.get("scores"), example.get("relevant")
        if not isinstance(scores, (list, tuple)) or not isinstance(relevant, (list, tuple)) or len(scores) != len(relevant):
            raise ValueError("candidate scores and relevant labels must be aligned lists")
        if any(isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) for score in scores):
            raise ValueError("candidate scores must be finite numbers")
        if type(no_answer) is not bool or any(type(label) is not bool for label in relevant):
            raise ValueError("no_answer and relevant labels must be booleans")
        if no_answer and any(relevant):
            raise ValueError("a no-answer query cannot label a candidate relevant")
        rows.append({"query_id": query_id, "scores": [float(score) for score in scores],
                     "relevant": list(relevant), "no_answer": no_answer, "calibration_fingerprint": fingerprint})
    return rows


def _cutoff_metrics(rows: list[dict[str, Any]], cutoff: float) -> dict[str, Any]:
    no_answer = [row for row in rows if row["no_answer"]]
    answerable = [row for row in rows if not row["no_answer"]]
    injected = sum(any(score >= cutoff for score in row["scores"]) for row in no_answer)
    recalled = sum(any(label and score >= cutoff for score, label in zip(row["scores"], row["relevant"]))
                   for row in answerable)
    return {"queries": len(rows), "answer_queries": len(answerable), "no_answer_queries": len(no_answer),
            "answers_recalled": recalled, "false_injections": injected,
            "answer_recall": recalled / len(answerable), "false_injection_rate": injected / len(no_answer)}


def _cutoff_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


__all__ = ["add_parser", "explain", "refit_query_adapter_cutoff", "run", "run_questions"]
