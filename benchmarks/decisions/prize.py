"""The prize count: how often each typed call's ambiguous band fires, and how often the opinion, appraisal and
capture calls return nothing, read from the synthetic paired-benchmark records with no model.

    python benchmarks/decisions/prize.py <results root> --out prize.json

A paired run keeps a private trace per attempt (``<root>/<campaign>/runs/<run>/attempts/<case>/container.log``,
lines marked ``colonymind.qualification.paired_trace.MARKER``): every model request, its response schema's name (the
call's kind) and its response. A typed call is made only where its reader leaves the case open, so its count is
how often that band fired. Only development records are read, and a campaign's own plan says which it is: its
``paired.json`` must declare the dataset's split as development before any of its traces is opened. A campaign on
another split, one with no readable plan, and any path with a held-out component are skipped and counted.

Every answer is read by the reader production uses for its call before it counts: the typed calls' reader and
their enum, the gate's choices, the appraisal's and the capture's own parsers, and the opinion call's action with
the schema it declares (the checks that need the call's own packet, its premise and stance ids, cannot be read
from a trace). An answer production would reject is no answer, never an empty result.

The output is counts only, never text: per call kind the calls, the answers, the calls with no final answer (cut
at their token limit, empty, not the JSON asked for, or one production rejects), the errors, the requests whose response the trace does not
hold (a trace is capped), a typed call's answers by value (``unsure`` included), and for the opinion, appraisal and
capture calls how many returned nothing; per point the band's calls per 100 owner turns; per campaign directory the
owner turns and the typed calls answered (so a count can be read for the builds that have a call).
A point whose band fires less than ``RARE_PER_100_TURNS`` times per 100 owner turns is marked ``rare``: it leaves
the bake-off with that number as the reason.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from colonymind.qualification.paired_trace import MARKER, PROTOCOL

#: Below this many band calls per 100 owner turns a point's band "almost never fires" (my choice).
RARE_PER_100_TURNS = 1.0
HELD_OUT = ("heldout", "held-out", "held_out")
#: The splits a campaign's plan declares for development data: the frozen fixtures' and the generated families'.
DEVELOPMENT = ("development", "dev")
#: The typed calls of the mind: (the decision point, the answer's field).
TYPED: Dict[str, Tuple[str, str]] = {
    "mind_owner_instruction": ("owner_instruction", "instruction"),
    "mind_report_substance": ("report_substance", "verdict"),
    "mind_question_answer": ("question_answer", "answer"),
    "mind_claim_relation": ("claim_relation", "relation"),
    "mind_wish_scope": ("mind_wish_scope", "scope"),
    "owner_gate_review": ("owner_send", "choice"),
}
AGENT = "agent"


def _nothing_opinion(answer: Dict[str, Any]) -> bool:
    return answer.get("action") == "none"


def _nothing_appraisal(answer: Dict[str, Any]) -> bool:
    contact = answer.get("contact") if isinstance(answer.get("contact"), dict) else {}
    decisions = answer.get("incident_decisions") or []
    return (not answer.get("observations") and not answer.get("outcomes")
            and not any(isinstance(item, dict) and item.get("outcome") == "resolved" for item in decisions)
            and not contact.get("opt_out") and contact.get("their_valence") is None)


def _nothing_capture(answer: Dict[str, Any]) -> bool:
    return not answer.get("items")


#: The calls a pre-screen could skip: (the screen's name, what "nothing" is for its answer).
SCREENS = {"opinion_pass": ("opinion", _nothing_opinion), "source_appraisal": ("appraisal", _nothing_appraisal),
           "commitment_extract": ("capture", _nothing_capture)}


def split_of(campaign: Path) -> Optional[str]:
    """The split a campaign's plan (``paired.json``) declares for its dataset; None when it has no readable plan."""
    try:
        plan = json.loads((Path(campaign) / "paired.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    dataset = plan.get("dataset") if isinstance(plan, dict) else None
    split = dataset.get("split") if isinstance(dataset, dict) else None
    return split if isinstance(split, str) else None


def attempts(root: Path) -> Iterator[Tuple[Path, Optional[str]]]:
    """Every attempt's trace under ``root``, with why it is skipped (``held_out``, ``not_development``,
    ``no_plan``) or None when it may be read. The campaign's plan is read first; a trace is never opened here."""
    root = Path(root)
    splits: Dict[str, Optional[str]] = {}
    for path in sorted(root.glob("*/runs/*/attempts/*/container.log")):
        parts = path.relative_to(root).parts
        if parts[0] not in splits:
            splits[parts[0]] = split_of(root / parts[0])
        split = splits[parts[0]]
        named = any(marker in part.lower() for part in parts for marker in HELD_OUT)
        if named or (split is not None and any(marker in split.lower() for marker in HELD_OUT)):
            yield path, "held_out"
        elif split is None:
            yield path, "no_plan"
        elif split not in DEVELOPMENT:
            yield path, "not_development"
        else:
            yield path, None


def rows(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """The trace rows of one attempt, and how many marked lines did not parse."""
    out, bad = [], 0
    with open(path, errors="replace") as handle:
        for line in handle:
            if not line.startswith(MARKER):
                continue
            try:
                row = json.loads(line[len(MARKER):])
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict) and row.get("protocol") == PROTOCOL and isinstance(row.get("data"), dict):
                out.append(row)
            else:
                bad += 1
    return out, bad


def _kind(payload: Any) -> str:
    fmt = payload.get("response_format") if isinstance(payload, dict) else None
    schema = fmt.get("json_schema") if isinstance(fmt, dict) else None
    name = schema.get("name") if isinstance(schema, dict) else None
    return str(name) if isinstance(name, str) and name else AGENT


def final(data: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """A response's final content and its finish reason: a whole completion, or streamed chunks joined."""
    content: List[str] = []
    finish = ""
    whole: Optional[str] = None
    for event in data.get("events") or []:
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            finish = str(choice.get("finish_reason") or finish or "")
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                whole = message["content"]
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                content.append(delta["content"])
    text = whole if whole is not None else ("".join(content) if content else None)
    return (text if text and text.strip() else None), finish


def _answer(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1]
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _typed_schemas() -> Dict[str, Dict[str, Any]]:
    from colonymind.instinct import deliberate
    from colonymind.mind import consolidate, gate
    from colonymind.mind import questions as ledger
    return {"mind_owner_instruction": deliberate.INSTRUCTION_SCHEMA, "mind_report_substance": deliberate.SUBSTANCE_SCHEMA,
            "mind_question_answer": ledger.ANSWER_SCHEMA, "mind_claim_relation": consolidate.RELATION_SCHEMA,
            "mind_wish_scope": deliberate.SCOPE_SCHEMA, "owner_gate_review": gate.JUDGE_SCHEMA}


def accepted(kind: str, text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The answer to a call of ``kind`` as production reads it, or None where production would reject it (then
    it is no answer: never counted as an answer, nor as one that returned nothing)."""
    if text is None:
        return None
    if kind == AGENT:
        return {}
    try:
        if kind in TYPED:
            field = TYPED[kind][1]
            allowed = _typed_schemas()[kind]["schema"]["properties"][field]["enum"]
            if kind == "owner_gate_review":
                value = json.loads(text)
            else:
                from colonymind.instinct.deliberate import _object
                value = _object(text)
            return value if isinstance(value, dict) and value.get(field) in allowed else None
        if kind == "source_appraisal":
            from colonymind.self_model.appraisals import AppraisalStore
            return AppraisalStore._parse(text)
        if kind == "commitment_extract":
            from colonymind.commitments.extract import parse_items
            return {"items": parse_items(text)}
        if kind == "opinion_pass":
            import jsonschema
            from colonymind.mind import opinions
            value = json.loads(text)
            if not isinstance(value, dict) or value.get("action") not in ("none", "form", "revise"):
                return None
            if value["action"] != "none":
                try:
                    jsonschema.validate(value, opinions.RESPONSE_SCHEMA["schema"])
                except jsonschema.ValidationError:
                    return None
            return value
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return _answer(text)


def count(root: Path) -> Dict[str, Any]:
    root = Path(root)
    totals = Counter()
    calls: Dict[str, Counter] = defaultdict(Counter)
    values: Dict[str, Counter] = defaultdict(Counter)
    campaigns: Dict[str, Counter] = defaultdict(Counter)
    for path, skipped in attempts(root):
        if skipped:
            totals[f"skipped_{skipped}"] += 1
            continue
        totals["attempts"] += 1
        trace, bad = rows(path)
        totals["unreadable_rows"] += bad
        sidecar = any(row["kind"] == "episode" and (row["data"].get("profile") or {}).get("plugin") is True
                      for row in trace)
        if not sidecar:
            continue
        totals["with_sidecar"] += 1
        owner_turns = sum(1 for row in trace if row["kind"] == "native_turn" and row["data"].get("kind") == "user")
        totals["owner_turns"] += owner_turns
        campaigns[path.relative_to(root).parts[0]]["owner_turns"] += owner_turns
        totals["ticks"] += sum(1 for row in trace if row["kind"] == "body_tick")
        kinds: Dict[Any, str] = {}
        for row in trace:
            data = row["data"]
            if row["kind"] == "model_request":
                kinds[data.get("request_id")] = _kind(data.get("payload"))
            elif row["kind"] == "model_response" and data.get("request_id") in kinds:
                kind = kinds.pop(data.get("request_id"))
                tally = calls[kind]
                tally["calls"] += 1
                if data.get("status") != 200:
                    tally["errors"] += 1
                    continue
                text, finish = final(data)
                answer = accepted(kind, text)
                if answer is None:
                    tally["no_answer"] += 1
                    if finish == "length":
                        tally["cut_at_limit"] += 1
                    continue
                tally["answered"] += 1
                if kind in TYPED:
                    campaigns[path.relative_to(root).parts[0]][TYPED[kind][0]] += 1
                if kind in TYPED:
                    value = answer.get(TYPED[kind][1])
                    values[kind][value if isinstance(value, str) else "other"] += 1
                if kind in SCREENS and SCREENS[kind][1](answer):
                    tally["nothing"] += 1
        for kind in kinds.values():                   # a request whose response the trace does not hold
            calls[kind]["calls"] += 1
            calls[kind]["unrecorded"] += 1
    turns = totals["owner_turns"]

    def per_turns(n: int) -> Optional[float]:
        return round(100.0 * n / turns, 3) if turns else None

    out_calls = {}
    for kind in sorted(set(calls) | set(TYPED) | set(SCREENS)):
        tally = calls.get(kind, Counter())
        entry = {name: int(tally[name]) for name in ("calls", "answered", "no_answer", "cut_at_limit", "errors",
                                                      "unrecorded")}
        entry["per_100_owner_turns"] = per_turns(tally["calls"])
        if kind in TYPED:
            entry["values"] = dict(sorted(values[kind].items()))
        if kind in SCREENS:
            entry["nothing"] = int(tally["nothing"])
        out_calls[kind] = entry
    points = {}
    for kind, (point, _) in TYPED.items():
        entry = out_calls[kind]
        rate = entry["per_100_owner_turns"]
        points[point] = {"band_calls": entry["calls"], "per_100_owner_turns": rate, "answered": entry["answered"],
                         "unsure": entry["values"].get("unsure", 0), "no_answer": entry["no_answer"],
                         "rare": rate is None or rate < RARE_PER_100_TURNS}
    screens = {}
    for kind, (name, _) in SCREENS.items():
        entry = out_calls[kind]
        screens[name] = {"calls": entry["calls"], "answered": entry["answered"], "nothing": entry["nothing"],
                         "share": round(entry["nothing"] / entry["answered"], 4) if entry["answered"] else None}
    return {"protocol": "instinct-prize-count-1", "rare_below_per_100_owner_turns": RARE_PER_100_TURNS,
            **{name: int(totals[name]) for name in ("attempts", "with_sidecar", "skipped_held_out",
                                                     "skipped_not_development", "skipped_no_plan", "owner_turns",
                                                     "ticks", "unreadable_rows")},
            "calls": out_calls, "points": points, "screens": screens,
            "by_campaign": {name: dict(sorted(tally.items())) for name, tally in sorted(campaigns.items())}}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", help="the directory holding the paired runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    counted = count(Path(args.root))
    Path(args.out).write_text(json.dumps(counted, indent=1, sort_keys=True) + "\n")
    for point, entry in counted["points"].items():
        print(f"{point:18s} band calls {entry['band_calls']:5d}  per 100 owner turns {entry['per_100_owner_turns']}"
              f"  answered {entry['answered']}  unsure {entry['unsure']}  no answer {entry['no_answer']}"
              f"{'  rare' if entry['rare'] else ''}")
    for name, entry in counted["screens"].items():
        print(f"{name:18s} calls {entry['calls']:5d}  answered {entry['answered']}  nothing {entry['nothing']}"
              f"  share {entry['share']}")


if __name__ == "__main__":
    main()
