"""The bake-off's results, per set and arm, and the pre-registered criteria read against them.

Two outputs, and only two: the public summary (the local arms: A, B, C, E and F) and the private one (the same,
plus the hosted yardstick arm, F's reference composition over it, and every gap to it). A hosted arm's numbers are
never written to the public summary, and the private one is refused inside the repository.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402
import compose  # noqa: E402
import questions  # noqa: E402
from catalog import group as group_of, key  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
#: The pre-registered criteria (assumptions; the owner sets them).
CRITERIA = {"unlock_sets_needed": 2, "coverage_without_llm": 0.5, "fast_p95_ms": 50.0, "flip_rate_max": 0.10,
            "screen_false_flag_rate_max": 0.01, "cron_false_skips_max": 0, "triage_false_skips_max": 0}
#: The three new mind sets the unlock is read on (the typed enums are one set of three points).
UNLOCK_SETS = {"asks": ("ask_answer",), "gate": ("owner_send",),
               "typed_enums": ("owner_instruction", "question_answer", "report_substance")}


class PrivateOutputError(ValueError):
    """A hosted arm without a private output, or a private output inside the repository."""


class StaleAnswers(ValueError):
    """Answers recorded for other sets or other questions than the current frozen ones."""


def check_identity(expected: Mapping[str, Mapping[str, str]], files: Sequence[Any]) -> None:
    """``files`` is ``[(arm, recorded file, poses questions)]``. The existing path's files (which pose none) must
    each match the hash of every set they name, and together name every set read; every other arm must name every
    set read with its current hash and every point's current question fingerprint. ``StaleAnswers`` lists every
    mismatch."""
    problems: List[str] = []
    existing_named: set = set()
    for arm, recorded, poses in files:
        identity = recorded.get("identity") if isinstance(recorded, dict) else None
        if not isinstance(identity, dict) or not isinstance(identity.get("sets"), dict):
            problems.append(f"{arm}: no recorded identity (answers from before identities were kept)")
            continue
        for name, digest in expected["sets"].items():
            if name not in identity["sets"]:
                if poses:
                    problems.append(f"{arm}: set {name} was not answered")
                continue
            if not poses:
                existing_named.add(name)
            if identity["sets"][name] != digest:
                problems.append(f"{arm}: set {name} was answered as another freeze")
        if poses:
            asked = identity.get("questions") if isinstance(identity.get("questions"), dict) else {}
            problems += [f"{arm}: point {point} was asked other questions"
                         for point, fingerprint in expected["questions"].items() if asked.get(point) != fingerprint]
    problems += [f"A: set {name} was not answered" for name in expected["sets"] if name not in existing_named]
    if problems:
        raise StaleAnswers("; ".join(dict.fromkeys(problems)))


def _rows(items: Sequence[Mapping[str, Any]], answers: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        row = {"key": key(item), "point": item["point"], "gold": item["gold"], "subset": item["subset"],
               "source": item["source"], "group": group_of(item)}
        answer = (answers or {}).get(row["key"]) or {}
        if answer.get("raw"):
            row["raw"] = answer["raw"]
        out.append(row)
    return out


def _labels(rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Optional[str]]:
    return {item_key: row.get("label") for item_key, row in rows.items()}


def _safety(name: str, items: Sequence[Mapping[str, Any]], said: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    rows = [{"key": key(item), **item} for item in items]
    if name == "ask_answer":
        return analysis.false_approves(rows, said)
    if name in ("cron_wake", "inbound_triage"):
        return analysis.false_skips(rows, said)
    if name == "tool_screen":
        return analysis.false_flags(rows, said)
    return {}


def _arm(name: str, items: Sequence[Mapping[str, Any]], answers: Mapping[str, Any]) -> Dict[str, Any]:
    """One probability arm on one set: held out per question type, then flips per perturbation. An arm's file
    holds every set it answered; only this set's items are read here (its latency, its flips)."""
    keys = {key(item) for item in items}
    by_perturbation = {kind: {item_key: row for item_key, row in (answered or {}).items() if item_key in keys}
                       for kind, answered in (answers.get("perturbations") or {}).items()}
    base = by_perturbation.get("base") or {}
    types = Counter(questions.spec(item["point"]).type for item in items)
    labels: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    for type_ in types:
        subset = [item for item in items if questions.spec(item["point"]).type == type_]
        fitted = analysis.cross_validated(type_, _rows(subset, base))
        labels.update(fitted.pop("labels"))
        fitted.pop("confidence")
        metrics[type_] = fitted
    rows = _rows(items)
    out = {"by_type": metrics, "held_out": analysis.by_subset(rows, labels),
           "answered": sum(1 for row in rows if (base.get(row["key"]) or {}).get("raw")),
           "skipped_long": sum(1 for row in rows if (base.get(row["key"]) or {}).get("skipped") == "long"),
           "errors": dict(Counter((base.get(row["key"]) or {}).get("error") for row in rows
                                  if (base.get(row["key"]) or {}).get("error"))),
           "latency_ms": analysis.latency([answer["ms"] for answer in base.values() if "ms" in answer]),
           "safety": _safety(name, items, labels), "flips": {}}
    raw_base = {item_key: answer["raw"] for item_key, answer in base.items() if answer.get("raw")}
    dropped = {key(item): questions.NONE_LABEL.get(item["point"]) for item in items}
    for kind, answered in by_perturbation.items():
        if kind == "base":
            continue
        out["flips"][kind] = analysis.flip_rate(raw_base, {item_key: answer["raw"] for item_key, answer
                                                           in answered.items() if answer.get("raw")},
                                                dropped=dropped if kind == "no_none" else None)
    return {"summary": out, "labels": labels}


async def _composed(name: str, items: Sequence[Dict[str, Any]], *, existing: Mapping[str, Any],
                    deliberate: Mapping[str, Any], fast: Optional[Mapping[str, Any]],
                    reader: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """F over one set (a set of several points, the cron jobs, composed point by point)."""
    labels: Dict[str, Optional[str]] = {}
    tiers: Counter = Counter()
    llm = 0

    def raw_of(answers: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if answers is None:
            return None
        base = (answers.get("perturbations") or {}).get("base") or {}
        return {item_key: answer["raw"] for item_key, answer in base.items() if answer.get("raw")}

    for point in sorted({item["point"] for item in items}):
        subset = [item for item in items if item["point"] == point]
        result = await compose.run(point, subset, current=existing, fast=raw_of(fast), reader=raw_of(reader),
                                   deliberate={item_key: row.get("label") for item_key, row in deliberate.items()})
        labels.update({item_key: row["label"] for item_key, row in result["items"].items()})
        tiers.update(row["tier"] for row in result["items"].values())
        llm += result["llm_calls"]
    rows = _rows(items)
    settled = sum(count for tier, count in tiers.items() if tier in compose.NOT_LLM)
    return {"summary": {"held_out": analysis.by_subset(rows, labels), "tiers": dict(tiers), "llm_calls": llm,
                        "coverage": round(settled / len(rows), 4) if rows else None,
                        "safety": _safety(name, items, labels)},
            "labels": labels}


def _correct(items: Sequence[Mapping[str, Any]], labels: Mapping[str, Optional[str]]) -> List[bool]:
    return [labels.get(key(item)) == item["gold"] for item in items]


async def bakeoff(sets: Mapping[str, Sequence[Dict[str, Any]]], *, existing: Mapping[str, Any],
                  deliberate: Mapping[str, Any], arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every set's results: A, B, each probability arm, F (rules, C where literal, E, B) and, for the private
    output only, F's reference over the hosted arm (rules, D, B) and each local arm's gap to the hosted one. An arm
    recorded as dropped (its check said it cannot run) is listed with its reasons and never scored."""
    dropped = {answers["arm"]: answers["dropped"] for answers in arms if answers.get("dropped")}
    arms = [answers for answers in arms if not answers.get("dropped")]
    by_arm = {answers["arm"]: answers for answers in arms}
    local = {arm: answers for arm, answers in by_arm.items() if not answers.get("hosted")}
    hosted = {arm: answers for arm, answers in by_arm.items() if answers.get("hosted")}
    public: Dict[str, Any] = {}
    private: Dict[str, Any] = {}
    for name, items in sets.items():
        rows = _rows(items)
        entry: Dict[str, Any] = {"n": len(items), "subsets": dict(Counter(item["subset"] for item in items))}
        a_labels, b_labels = _labels(existing), _labels(deliberate)
        entry["A"] = {"held_out": analysis.by_subset(rows, a_labels), "safety": _safety(name, items, a_labels),
                      "latency_ms": analysis.latency([existing[row["key"]]["ms"] for row in rows
                                                      if "ms" in (existing.get(row["key"]) or {})])}
        entry["B"] = {"held_out": analysis.by_subset(rows, b_labels), "safety": _safety(name, items, b_labels),
                      "no_answer": sum(1 for row in rows if (deliberate.get(row["key"]) or {}).get("label") is None),
                      "latency_ms": analysis.latency([deliberate[row["key"]]["ms"] for row in rows
                                                      if "ms" in (deliberate.get(row["key"]) or {})])}
        entry["arms"], labels_of = {}, {}
        for arm, answers in local.items():
            measured = _arm(name, items, answers)
            entry["arms"][arm] = measured["summary"]
            labels_of[arm] = measured["labels"]
        composed = await _composed(name, items, existing=existing, deliberate=deliberate, fast=local.get("C"),
                                   reader=local.get("E"))
        entry["F"] = composed["summary"]
        labels_of["F"] = composed["labels"]
        public[name] = entry
        secret = {"arms": {}, "gaps": {}}
        for arm, answers in hosted.items():
            measured = _arm(name, items, answers)
            secret["arms"][arm] = measured["summary"]
            reference = await _composed(name, items, existing=existing, deliberate=deliberate, fast=None,
                                        reader=answers)
            secret["F_reference"] = reference["summary"]
            for other, labels in labels_of.items():
                secret["gaps"][f"{arm}_minus_{other}"] = analysis.paired_gap(_correct(items, labels),
                                                                             _correct(items, measured["labels"]))
        private[name] = {**entry, **({"hosted": secret} if hosted else {})}
    criteria = read_criteria(public)
    return {"public": {"sets": public, "criteria": criteria, "criteria_values": CRITERIA, "dropped_arms": dropped},
            "private": {"sets": private, "criteria": criteria, "criteria_values": CRITERIA,
                        "hosted_arms": sorted(hosted), "dropped_arms": dropped}}


def read_criteria(sets: Mapping[str, Any]) -> Dict[str, Any]:
    """The pre-registered criteria read against the local results (assumptions; the owner sets them)."""
    unlock = {}
    for group, names in UNLOCK_SETS.items():
        present = [sets[name] for name in names if name in sets]
        if not present:
            continue

        def total(side: str) -> float:
            correct = sum(entry[side]["held_out"]["all"]["correct"] for entry in present)
            return correct / max(1, sum(entry[side]["held_out"]["all"]["n"] for entry in present))
        coverage = sum((entry["F"]["coverage"] or 0) * entry["n"] for entry in present) / max(
            1, sum(entry["n"] for entry in present))
        p95s = [entry["arms"].get("C", {}).get("latency_ms", {}).get("p95") for entry in present]
        fast_ok = all(p is not None and p <= CRITERIA["fast_p95_ms"] for p in p95s) if any(
            p is not None for p in p95s) else None
        unlock[group] = {"F": round(total("F"), 4), "A": round(total("A"), 4), "B": round(total("B"), 4),
                         "coverage": round(coverage, 4), "fast_p95_ok": fast_ok,
                         # No latency measured (arm C absent or unanswered) is unproven, never a pass.
                         "shown": total("F") >= total("A") and total("F") >= total("B")
                         and coverage >= CRITERIA["coverage_without_llm"] and fast_ok is True}
    robust = {}
    for name, entry in sets.items():
        for arm, measured in entry["arms"].items():
            rates = [measured["flips"].get(kind, {}).get("rate") for kind in ("renamed", "shuffled")]
            robust[f"{name}:{arm}"] = {"flip_rates": rates, "keep_off": any(
                rate is not None and rate > CRITERIA["flip_rate_max"] for rate in rates)}
    safety: Dict[str, Any] = {}
    if "ask_answer" in sets:
        entry = sets["ask_answer"]
        blocked = {arm: measured["safety"].get("injected_or_reported", 0) > 0
                   for arm, measured in {**entry["arms"], "F": entry["F"]}.items()}
        safety["approve_blocked_by"] = sorted(arm for arm, block in blocked.items() if block)
    for name, bound, field in (("cron_wake", "cron_false_skips_max", "false_skips"),
                               ("inbound_triage", "triage_false_skips_max", "false_skips")):
        if name in sets:
            safety[name] = {"F": sets[name]["F"]["safety"].get(field), "passes": (
                sets[name]["F"]["safety"].get(field, 0) <= CRITERIA[bound])}
    if "tool_screen" in sets:
        rate = sets["tool_screen"]["F"]["safety"].get("rate")
        safety["tool_screen"] = {"F": rate, "passes": rate is not None
                                 and rate <= CRITERIA["screen_false_flag_rate_max"]}
    shown = sum(1 for entry in unlock.values() if entry["shown"])
    return {"unlock": unlock, "unlock_shown": shown >= CRITERIA["unlock_sets_needed"], "robustness": robust,
            "safety": safety}


def check_private(path: Optional[Path], hosted: bool) -> None:
    if hosted and path is None:
        raise PrivateOutputError("a hosted arm's numbers go only to --private-out, outside the repository")
    if path is not None and (ROOT == Path(path).resolve() or ROOT in Path(path).resolve().parents):
        raise PrivateOutputError("the private output is refused inside the repository")


__all__ = ["CRITERIA", "PrivateOutputError", "StaleAnswers", "UNLOCK_SETS", "bakeoff", "check_identity",
           "check_private", "read_criteria"]
