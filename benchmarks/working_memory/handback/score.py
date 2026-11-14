"""The scorer: per arm accuracy (overall and per question type), paired bootstrap intervals between arms, cost per
question, and WM3's verdict under the pre-registered bars.

    python -m handback.score --set-seeds 1301 1302 1303 --answers <dir>/set-*/answers.jsonl --out report.json

An answer is right when it names the gold code and no distractor (the stale value of a contradiction, the next link
of a relation, the other entities' values of a question that spans children); ``unknown`` is wrong. Intervals are
95% percentile bootstraps of the paired difference in accuracy, resampling questions (``RESAMPLES`` resamples, my
assumption, under a fixed seed).

The bars are D 6.4's (D assumptions): the consult passes within ``BAR_QUESTIONS`` question of the full text and at
least ``BAR_POINTS`` points above the summary; if the excerpts alone land within ``BAR_QUESTIONS`` question of the
full text, branches are not built and excerpts go into the result contract (``excerpts_suffice``, which takes
precedence); otherwise a passing consult is ``branches_earned`` and anything else is ``neither``.

A verdict needs complete coverage: every arm answered every question of the scored sets exactly once, and no row
names a question outside them. Otherwise the outcome is ``incomplete`` (the arms and pairs are still reported, and
the report's ``coverage`` says what is missing), because pairs over different partial cohorts can pass bars the
full sets would not.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from .arms import ARMS
except ImportError:  # run as a script
    from arms import ARMS  # type: ignore[no-redef]

RESAMPLES = 10000
SEED = 1
BAR_QUESTIONS = 1
BAR_POINTS = 10.0
PAIRS = (("full", "summary"), ("full", "excerpts"), ("full", "consult"), ("consult", "summary"),
         ("consult", "excerpts"), ("excerpts", "summary"))


def _mentions(text: str, value: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(value.lower()) + r"(?![a-z0-9])", text) is not None


def correct(answer: Any, gold: str, distractors: Sequence[str]) -> bool:
    text = str(answer or "").lower()
    return _mentions(text, gold) and not any(_mentions(text, d) for d in distractors)


def paired_bootstrap(a: Sequence[int], b: Sequence[int], *, resamples: int = RESAMPLES,
                     seed: int = SEED) -> Tuple[float, float, float]:
    """The mean paired difference ``a - b`` and its 95% percentile bootstrap interval."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return 0.0, 0.0, 0.0
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    low = means[math.floor(0.025 * (resamples - 1))]
    high = means[math.ceil(0.975 * (resamples - 1))]
    return round(sum(diffs) / n, 4), round(low, 4), round(high, 4)


def _mean(values: List[float]) -> Any:
    return round(statistics.fmean(values), 3) if values else None


def _percentile(values: List[float], share: float) -> Any:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))], 3)


def _cost(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = [float(r.get("prompt_tokens") or 0) for r in rows]
    cached = [float(r.get("cached_tokens") or 0) for r in rows]
    walls = [float(r.get("wall_s") or 0) for r in rows]
    out = {"prompt_tokens_mean": _mean(prompt), "cached_tokens_mean": _mean(cached),
           "incremental_tokens_mean": _mean([p - c for p, c in zip(prompt, cached)]),
           "completion_tokens_mean": _mean([float(r.get("completion_tokens") or 0) for r in rows]),
           "wall_s_mean": _mean(walls), "wall_s_p50": _percentile(walls, 0.5), "wall_s_p95": _percentile(walls, 0.95),
           "cache_hit_share": round(sum(1 for c in cached if c > 0) / len(rows), 4) if rows else None}
    calls = [call for r in rows for call in r.get("consult_calls") or []]
    if calls:
        c_prompt = [float(c.get("prompt_tokens") or 0) for c in calls]
        c_cached = [float(c.get("cached_tokens") or 0) for c in calls]
        c_walls = [float(c.get("wall_s") or 0) for c in calls]
        out["consult_calls"] = {
            "calls": len(calls), "prompt_tokens_mean": _mean(c_prompt), "cached_tokens_mean": _mean(c_cached),
            "incremental_tokens_mean": _mean([p - c for p, c in zip(c_prompt, c_cached)]),
            "wall_s_mean": _mean(c_walls), "wall_s_p95": _percentile(c_walls, 0.95),
            "cache_hit_share": round(sum(1 for c in c_cached if c > 0) / len(calls), 4)}
    return out


def score(questions: Iterable[Dict[str, Any]], rows: Iterable[Dict[str, Any]], *, resamples: int = RESAMPLES,
          seed: int = SEED) -> Dict[str, Any]:
    by_id = {q["id"]: q for q in questions}
    marks: Dict[str, Dict[str, int]] = {arm: {} for arm in ARMS}
    kept: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARMS}
    duplicates = {arm: 0 for arm in ARMS}
    unmatched = 0
    for row in rows:
        question = by_id.get(row.get("question_id"))
        arm = row.get("arm")
        if question is None or arm not in marks:
            unmatched += 1
            continue
        if row["question_id"] in marks[arm]:
            duplicates[arm] += 1
            continue
        marks[arm][row["question_id"]] = int(correct(row.get("answer"), question["gold"], question["distractors"]))
        kept[arm].append(row)
    arms: Dict[str, Any] = {}
    for arm in ARMS:
        if not marks[arm]:
            continue
        right = marks[arm]
        by_type: Dict[str, Dict[str, Any]] = {}
        for qid, mark in right.items():
            entry = by_type.setdefault(by_id[qid]["type"], {"n": 0, "correct": 0})
            entry["n"] += 1
            entry["correct"] += mark
        for entry in by_type.values():
            entry["accuracy"] = round(entry["correct"] / entry["n"], 4)
        arms[arm] = {"n": len(right), "correct": sum(right.values()),
                     "accuracy": round(sum(right.values()) / len(right), 4),
                     "by_type": dict(sorted(by_type.items())), "cost": _cost(kept[arm])}
    pairs: Dict[str, Any] = {}
    for a, b in PAIRS:
        if a not in arms or b not in arms:
            continue
        common = sorted(set(marks[a]) & set(marks[b]))
        first, second = [marks[a][q] for q in common], [marks[b][q] for q in common]
        difference, low, high = paired_bootstrap(first, second, resamples=resamples, seed=seed)
        pairs[f"{a}-{b}"] = {"n": len(common), "a_correct": sum(first), "b_correct": sum(second),
                             "difference": difference, "low": low, "high": high}
    coverage: Dict[str, Any] = {
        "questions": len(by_id), "unmatched_rows": unmatched,
        "arms": {arm: {"answered": len(marks[arm]), "missing": len(set(by_id) - set(marks[arm])),
                       "duplicates": duplicates[arm]} for arm in ARMS}}
    coverage["complete"] = bool(by_id) and unmatched == 0 and all(
        entry["missing"] == 0 and entry["duplicates"] == 0 for entry in coverage["arms"].values())
    return {"arms": arms, "pairs": pairs, "coverage": coverage,
            "verdict": verdict(pairs, complete=coverage["complete"]),
            "method": {"interval": "95% percentile bootstrap of the paired difference, resampling questions",
                       "resamples": resamples, "seed": seed, "bar_questions": BAR_QUESTIONS, "bar_points": BAR_POINTS}}


def verdict(pairs: Dict[str, Any], *, complete: bool) -> Dict[str, Any]:
    """The pre-registered outcome, only when ``complete`` (every arm answered every question once); otherwise
    ``incomplete`` and no bar is read."""
    if not complete:
        return {"outcome": "incomplete", "complete": False}
    out: Dict[str, Any] = {"complete": True}
    full_consult, consult_summary = pairs.get("full-consult"), pairs.get("consult-summary")
    full_excerpts = pairs.get("full-excerpts")
    if full_consult:
        gap = full_consult["a_correct"] - full_consult["b_correct"]
        out["consult_within_one_question_of_full"] = gap <= BAR_QUESTIONS
    if consult_summary:
        out["consult_points_above_summary"] = round(100.0 * consult_summary["difference"], 2)
    if full_excerpts:
        out["excerpts_within_one_question_of_full"] = \
            full_excerpts["a_correct"] - full_excerpts["b_correct"] <= BAR_QUESTIONS
    if full_consult and consult_summary:
        out["consult_passes"] = bool(out["consult_within_one_question_of_full"]
                                     and out["consult_points_above_summary"] >= BAR_POINTS)
    if out.get("excerpts_within_one_question_of_full"):
        out["outcome"] = "excerpts_suffice"
    elif out.get("consult_passes"):
        out["outcome"] = "branches_earned"
    elif "consult_passes" in out:
        out["outcome"] = "neither"
    return out


def main() -> int:
    import argparse
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from handback.sets import load_frozen_set
    parser = argparse.ArgumentParser(description="Score a handback run.")
    parser.add_argument("--set-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    questions = [q for seed in args.set_seeds for q in load_frozen_set(seed)["questions"]]
    rows = [json.loads(line) for path in args.answers for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = score(questions, rows, resamples=args.resamples)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
