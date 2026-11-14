"""The prefix prize count: how much of ColonyMind's prompt work a prefix cache could serve, today and under WM1.

    python benchmarks/working_memory/prize.py --traces <results root> --tokenizer <dir> [--routing table.json]
        [--out count.json] [--aggregate-only]
    python benchmarks/working_memory/prize.py --events events.jsonl --tokenizer <dir> ...

No model runs; a tokenizer does (CPU). For each owner turn, every model call ColonyMind makes for it is counted,
grouped by the model it reaches under a routing table given as input:

- the per-turn passes (claims, the claim review, appraisal, commitments, opinions: ``PER_TURN``);
- within the turn's window, the other calls that fire (deliberation, the gate's second opinion, the typed calls,
  Instinct's reader tier, the composer): every ColonyMind call that is neither a per-turn pass nor a night stage;
- the night's stages (``NIGHT``), counted per night, not per turn.

A call belongs to the latest owner turn whose text it carries; a call that carries no turn's text belongs to the
window of the latest owner turn recorded before it; a night stage belongs to its night (a run of night-stage calls
with no owner turn between them). ColonyMind calls before the first owner turn are counted as unattributed.

Per turn (and per night) and per group:

- ``prompt_tokens``: the prompt as the model's chat template renders it (thinking off, generation prompt added);
- ``served_tokens``: tokens in the longest prefix shared with an earlier call of the same group in the same
  instance, which is what a stock prefix cache with no eviction serves today (an upper bound: engines recompute at
  least the last token and evict under pressure); each call counts its one longest match once;
- ``served_tokens_block``: the same, rounded down to whole blocks of ``BLOCK`` tokens, as a paged cache matches;
- ``turn_located_calls`` and ``turn_tokens``: the calls whose prompt carries the turn's own text (verbatim or in its
  JSON string rendering), and that text's tokens summed over them (the prompt's tokens that hold any of its
  characters), which is shareable if placed first;
- ``wm1_bound``: WM1's upper bound on what a cache serves once the turn is placed first: the turn's tokens in every
  located call of the group after the first, which computes the block.

The bound and today's count overlap where today's cache already serves part of the turn's text, so the located
calls also split what is served today around the turn. The turn's place is read from one tokenization of the whole
rendered prompt, with each token's character span: its first token is the first that holds any of its characters
(a token that joins the text before the turn with its first word belongs to the turn), never a count of the text
before it tokenized on its own, whose last token can differ.

- ``before_turn_tokens``: the tokens before the turn's text;
- ``served_before_turn``: today's served tokens that lie before it (what a turn-first order would have to keep
  ahead of the turn to go on serving);
- ``turn_served_today``: the turn's tokens today's cache already serves;
- ``wm1_new_tokens``: the bound less the turn tokens today's cache already serves in the same calls, which is what
  the bound adds to today's count, never counted twice.

Every call to the same group fills the cache, the host's own calls included when the table routes ``host`` there
(the default table puts every role on one model); the host's calls are never counted as ColonyMind's.

Sources. ``--traces`` reads the private request traces of paired runs (the harness's ``container.log`` lines marked
``COLONYMIND_PAIRED_DIAGNOSTIC:``), development campaigns only: a campaign's ``paired.json`` must declare a
development split before any of its traces is opened, and a path with a held-out component is never opened: the
root, each campaign, its plan and each attempt's log are checked, before they are opened or listed, in every
component of the absolute path as written and of the path it resolves to (links followed), and a held-out root is
refused outright. A
request the trace truncated is counted and skipped, and so is an attempt of the host alone (its episode's profile
has no plugin). Owner turns are the harness's ``user`` steps; ColonyMind's calls are the ones its source worker and
its server thread made (``colonymind_thread``), every other call is the host's. A call's kind is its response
schema's name (``untyped`` without one). ``--events`` reads the same shape as JSON
lines: ``{"type": "turn", "instance": ..., "text": ...}`` and ``{"type": "call", "instance": ..., "kind": ...,
"messages": [...], "tools": [...], "colonymind": true}``, each instance's events together and in order (a replay
through the scripted model server writes this shape; its replies shape later prompts, so such a count is an
estimate).

The routing table is JSON: ``{"default": "<group>", "kinds": {"<kind>": "<group>"}}``; ``host`` names the host's
calls. A deployment's own table stays in its own files and is passed at run time; its counts are private.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import statistics
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

PROTOCOL = "wm-prize-1"
TRACE_MARKER = "COLONYMIND_PAIRED_DIAGNOSTIC:"
TRACE_PROTOCOL = "paired-private-trace-1"
DEFAULT_MODEL = "one-model"
HOST = "host"
UNTYPED = "untyped"
#: A paged prefix cache matches whole blocks; 16 tokens is vLLM's default block size (my assumption as the unit).
BLOCK = 16
#: The per-turn passes: each reads the owner's turn on its own today.
PER_TURN = frozenset({"source_claims", "source_claim_review", "source_appraisal", "commitment_extract",
                      "opinion_pass"})
#: The night's stages as the mind's tick runs them: consolidation (episodes, narrative, digests, claim relations),
#: its lesson and wish stages, and the identity check (my classification, from the tick's order).
NIGHT = frozenset({"mind_consolidate_episode", "mind_consolidate_narrative", "mind_consolidate_digest",
                   "mind_claim_relation", "mind_lessons", "mind_wish", "mind_drift_check"})
HELD_OUT = ("heldout", "held-out", "held_out")
DEVELOPMENT = ("development", "dev")
#: The paired harness's thread names for ColonyMind's own calls: its source worker, and the sidecar's server
#: thread (Python names an unnamed thread after its target, ``Thread-N (run)``).
SOURCE_WORKER_THREAD = "paired-source-worker"
SERVER_THREAD_SUFFIX = " (run)"
FIELDS = ("calls", "prompt_tokens", "served_tokens", "served_tokens_block", "turn_located_calls", "turn_tokens",
          "wm1_bound", "before_turn_tokens", "served_before_turn", "turn_served_today", "wm1_new_tokens")


def colonymind_thread(thread: str) -> bool:
    return thread == SOURCE_WORKER_THREAD or thread.endswith(SERVER_THREAD_SUFFIX)


def phase_of(kind: str) -> str:
    if kind == HOST:
        return HOST
    if kind in PER_TURN:
        return "turn"
    if kind in NIGHT:
        return "night"
    return "window"


class Routing:
    """Which model (group) each call kind reaches: ``{"default": group, "kinds": {kind: group}}``."""

    def __init__(self, table: Optional[Dict[str, Any]] = None):
        table = {"default": DEFAULT_MODEL} if table is None else table
        if (not isinstance(table, dict) or set(table) - {"default", "kinds"}
                or not isinstance(table.get("default"), str)):
            raise ValueError("a routing table is {\"default\": group, \"kinds\": {kind: group}}")
        kinds = table.get("kinds") or {}
        if not isinstance(kinds, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in kinds.items()):
            raise ValueError("a routing table's kinds map a kind to a group name")
        self.default, self.kinds = table["default"], dict(kinds)

    def group(self, kind: str) -> str:
        return self.kinds.get(kind, self.default)

    def as_dict(self) -> Dict[str, Any]:
        return {"default": self.default, "kinds": dict(sorted(self.kinds.items()))}


class HFTokenizer:
    """A local tokenizer directory with its chat template (``transformers``, read offline)."""

    def __init__(self, path: str):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.name = Path(path).name
        self.template_failures = 0

    def _text(self, messages: List[Dict[str, Any]], tools: Optional[list]) -> Tuple[str, bool]:
        clean = [_clean_message(m) for m in messages]
        try:
            return str(self._tok.apply_chat_template(clean, tools=tools or None, add_generation_prompt=True,
                                                     tokenize=False, enable_thinking=False)), True
        except Exception:
            # A message shape the template refuses: count its plain text instead.
            return _plain(clean), False

    def render(self, messages: List[Dict[str, Any]],
               tools: Optional[list] = None) -> Tuple[str, List[int], List[Tuple[int, int]]]:
        """The prompt as the chat template renders it (thinking off, generation prompt added), its token ids and
        each token's character span in it, from one tokenization of the whole prompt."""
        text, rendered = self._text(messages, tools)
        if not rendered:
            self.template_failures += 1
        encoded = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return text, list(encoded["input_ids"]), [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


def _plain(messages: List[Dict[str, Any]]) -> str:
    return "".join(f"{m['role']}\n{m['content']}\n" for m in messages)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)
                       and isinstance(part.get("text"), str))
    return "" if content is None else json.dumps(content)


def _tool_call(call: Any) -> Any:
    """A tool call as chat templates read it: its arguments an object, not the JSON string the wire carries."""
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    try:
        arguments = json.loads(function["arguments"])
    except ValueError:
        return call
    if not isinstance(arguments, dict):
        return call
    return {**call, "function": {**function, "arguments": arguments}}


def _clean_message(message: Dict[str, Any]) -> Dict[str, Any]:
    out = {"role": str(message.get("role") or "user"), "content": _text(message.get("content"))}
    for key in ("tool_calls", "tool_call_id", "name"):
        if message.get(key) is not None:
            out[key] = message[key]
    if isinstance(out.get("tool_calls"), list):
        out["tool_calls"] = [_tool_call(call) for call in out["tool_calls"]]
    return out


def _prompt_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(_text(m.get("content")) for m in messages if isinstance(m, dict))


def renderings(text: str) -> List[str]:
    """The ways a turn's text appears in a prompt: verbatim, and as the inside of a JSON string."""
    out: List[str] = []
    for candidate in (text, json.dumps(text)[1:-1], json.dumps(text, ensure_ascii=False)[1:-1]):
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def locate(text: str, prompt: str) -> Optional[str]:
    """The rendering of ``text`` found in ``prompt``, or None."""
    if not text.strip():
        return None
    for candidate in renderings(text):
        if candidate in prompt:
            return candidate
    return None


def token_range(spans: List[Tuple[int, int]], start: int, stop: int) -> Tuple[int, int]:
    """The tokens that hold any character of ``text[start:stop]``, as ``(first, end)``: ``first`` is the first token
    whose span ends past ``start``, ``end`` the first after it whose span starts at or past ``stop``."""
    first = next((index for index, (_, b) in enumerate(spans) if b > start), len(spans))
    end = next((index for index in range(first, len(spans)) if spans[index][0] >= stop), len(spans))
    return first, end


def _key(ids: List[int]) -> bytes:
    # Big-endian words keep byte order equal to token order, so sorted neighbours hold the longest match.
    return struct.pack(f">{len(ids)}I", *ids)


def _lcp(a: bytes, b: bytes) -> int:
    hi = min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi // 4
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo // 4


class PrefixSet:
    """The prompts one model has seen; the longest prefix any of them shares with a new prompt."""

    def __init__(self):
        self._keys: List[bytes] = []

    def longest(self, key: bytes) -> int:
        at = bisect.bisect_left(self._keys, key)
        best = 0
        for index in (at - 1, at):
            if 0 <= index < len(self._keys):
                best = max(best, _lcp(self._keys[index], key))
        return best

    def add(self, key: bytes) -> None:
        bisect.insort(self._keys, key)


def _zero() -> Dict[str, int]:
    return {field: 0 for field in FIELDS}


def _add(into: Dict[str, int], row: Dict[str, int]) -> None:
    for field in FIELDS:
        into[field] += row.get(field, 0)


def _instances(events: Iterable[Dict[str, Any]]) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    current, batch, seen = None, [], set()
    for event in events:
        instance = str(event.get("instance"))
        if instance != current:
            if current is not None:
                yield current, batch
            if instance in seen:
                raise ValueError(f"instance {instance!r} reappears: each instance's events must be together")
            seen.add(instance)
            current, batch = instance, []
        batch.append(event)
    if current is not None:
        yield current, batch


def count(events: Iterable[Dict[str, Any]], tokenizer: Any, routing: Routing, *, block: int = BLOCK) -> Dict[str, Any]:
    turns_out: List[Dict[str, Any]] = []
    nights_out: List[Dict[str, Any]] = []
    unattributed: Dict[str, Dict[str, int]] = defaultdict(_zero)
    by_group: Dict[str, Dict[str, int]] = defaultdict(_zero)
    by_kind: Dict[str, Dict[str, int]] = defaultdict(_zero)
    by_phase: Dict[str, Dict[str, int]] = defaultdict(_zero)
    totals = {"owner_turns": 0, "nights": 0, "colonymind_calls": 0, "host_calls": 0, "turn_not_in_rendering": 0}
    for instance, batch in _instances(events):
        turns = [(position, event) for position, event in enumerate(batch) if event.get("type") == "turn"]
        rows = [{"instance": instance, "turn": index, "groups": defaultdict(_zero)} for index in range(len(turns))]
        texts = [str(event.get("text") or "") for _, event in turns]
        totals["owner_turns"] += len(turns)
        caches: Dict[str, PrefixSet] = defaultdict(PrefixSet)
        first_located: set = set()
        nights: List[Dict[str, Any]] = []
        night_mark: Optional[int] = None
        turns_seen = 0
        for position, event in enumerate(batch):
            if event.get("type") == "turn":
                turns_seen += 1
                continue
            if event.get("type") != "call":
                continue
            kind = str(event.get("kind") or UNTYPED)
            colonymind = bool(event.get("colonymind", kind != HOST))
            kind = kind if colonymind else HOST
            group = routing.group(kind)
            messages = event.get("messages") or []
            rendered, ids, spans = tokenizer.render(messages, event.get("tools"))
            key = _key(ids)
            served = caches[group].longest(key)
            caches[group].add(key)
            if not colonymind:
                totals["host_calls"] += 1
                continue
            totals["colonymind_calls"] += 1
            row = {"calls": 1, "prompt_tokens": len(ids), "served_tokens": served,
                   "served_tokens_block": (served // block) * block}
            phase = phase_of(kind)
            if phase == "night":
                if night_mark != turns_seen or not nights:
                    nights.append({"instance": instance, "night": len(nights), "groups": defaultdict(_zero)})
                    night_mark = turns_seen
                target = nights[-1]["groups"][group]
            else:
                prompt = _prompt_text(messages)
                located = [(index, locate(texts[index], prompt)) for index in range(len(turns))]
                located = [(index, found) for index, found in located if found]
                before = [item for item in located if turns[item[0]][0] < position]
                after = [item for item in located if turns[item[0]][0] > position]
                chosen = before[-1] if before else (after[0] if after else None)
                if chosen is not None:
                    index, found = chosen
                    at = rendered.find(found)
                    already = 0
                    if at >= 0:
                        first, end = token_range(spans, at, at + len(found))
                        tokens = end - first
                        already = max(0, min(served, end) - first)
                        row.update(before_turn_tokens=first, served_before_turn=min(served, first),
                                   turn_served_today=already)
                    else:
                        tokens = tokenizer.count(found)
                        totals["turn_not_in_rendering"] += 1
                    row.update(turn_located_calls=1, turn_tokens=tokens)
                    if (index, group) in first_located:
                        row["wm1_bound"] = tokens
                        row["wm1_new_tokens"] = tokens - already
                    first_located.add((index, group))
                    target = rows[index]["groups"][group]
                elif turns_seen:
                    target = rows[turns_seen - 1]["groups"][group]
                else:
                    target = unattributed[group]
            _add(target, row)
            _add(by_group[group], row)
            _add(by_kind[kind], row)
            _add(by_phase[phase], row)
        for row in rows:
            row["groups"] = {name: dict(values) for name, values in sorted(row["groups"].items())}
            turns_out.append(row)
        for night in nights:
            night["groups"] = {name: dict(values) for name, values in sorted(night["groups"].items())}
            nights_out.append(night)
        totals["nights"] += len(nights)
    totals.update(by_group=_shares(by_group), by_kind=_shares(by_kind), by_phase=_shares(by_phase))
    return {"protocol": PROTOCOL, "block": block, "routing": routing.as_dict(),
            "tokenizer": getattr(tokenizer, "name", type(tokenizer).__name__),
            "turns": turns_out, "nights": nights_out,
            "unattributed": {name: dict(values) for name, values in sorted(unattributed.items())},
            "totals": totals, "per_turn": _per_turn(turns_out)}


def _shares(table: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name, values in sorted(table.items()):
        row: Dict[str, Any] = dict(values)
        prompt = values["prompt_tokens"]
        row["served_share"] = round(values["served_tokens"] / prompt, 4) if prompt else None
        row["wm1_share"] = round(values["wm1_bound"] / prompt, 4) if prompt else None
        row["wm1_new_share"] = round(values["wm1_new_tokens"] / prompt, 4) if prompt else None
        out[name] = row
    return out


def _distribution(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {"n": len(ordered), "mean": round(statistics.fmean(ordered), 1), "median": statistics.median(ordered),
            "p90": p90, "max": ordered[-1]}


def _per_turn(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for field in ("calls", "prompt_tokens", "served_tokens", "turn_tokens", "wm1_bound", "served_before_turn",
                  "wm1_new_tokens"):
        out[field] = _distribution([sum(g[field] for g in t["groups"].values()) for t in turns])
    return out


def aggregate(result: Dict[str, Any]) -> Dict[str, Any]:
    """The count without per-instance rows (instance names are private paths)."""
    return {key: value for key, value in result.items() if key not in ("turns", "nights", "unattributed")} | {
        "unattributed_calls": sum(g["calls"] for g in result["unattributed"].values())}


def held_out_path(path: Path) -> bool:
    """Whether a held-out name appears in any component of ``path``, as written (made absolute, ancestors
    included) or as it resolves (links followed)."""
    path = Path(path)
    for candidate in (Path(os.path.abspath(path)), path.resolve()):
        if any(marker in part.lower() for part in candidate.parts for marker in HELD_OUT):
            return True
    return False


def split_of(campaign: Path, opener: Callable = open) -> Optional[str]:
    try:
        with opener(Path(campaign) / "paired.json", encoding="utf-8") as handle:
            plan = json.load(handle)
    except (OSError, ValueError):
        return None
    dataset = plan.get("dataset") if isinstance(plan, dict) else None
    split = dataset.get("split") if isinstance(dataset, dict) else None
    return split if isinstance(split, str) else None


def _schema_name(payload: Dict[str, Any]) -> str:
    fmt = payload.get("response_format")
    schema = fmt.get("json_schema") if isinstance(fmt, dict) else None
    name = schema.get("name") if isinstance(schema, dict) else None
    return name if isinstance(name, str) and name else UNTYPED


def _owner_text(data: Dict[str, Any]) -> str:
    for message in data.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return _text(message.get("content"))
    return ""


def trace_events(root: Path, opener: Callable = open) -> Tuple[Iterator[Dict[str, Any]], Dict[str, Any]]:
    """The events of every development attempt under ``root``, and counts of what was read and skipped (filled as
    the events are read). A campaign's plan is read before any of its traces is opened."""
    root = Path(root)
    if held_out_path(root):
        raise ValueError(f"refusing a held-out traces root (a held-out name in its path or its target): {root}")
    stats: Dict[str, Any] = defaultdict(int)
    stats["threads"] = defaultdict(lambda: {"colonymind": False, "calls": 0})

    def generate() -> Iterator[Dict[str, Any]]:
        for campaign in sorted(root.iterdir()):
            if held_out_path(campaign) or held_out_path(campaign / "paired.json"):
                stats["skipped_held_out_campaigns"] += 1
                continue
            if not campaign.is_dir():
                continue
            split = split_of(campaign, opener)
            if split is not None and any(marker in split.lower() for marker in HELD_OUT):
                stats["skipped_held_out_campaigns"] += 1
                continue
            for path in sorted(campaign.glob("runs/*/attempts/*/container.log")):
                if held_out_path(path):
                    stats["skipped_held_out_attempts"] += 1
                    continue
                if split is None:
                    stats["skipped_no_plan"] += 1
                    continue
                if split not in DEVELOPMENT:
                    stats["skipped_not_development"] += 1
                    continue
                yield from _attempt(campaign, path)

    def _attempt(campaign: Path, path: Path) -> Iterator[Dict[str, Any]]:
        # The instance is the attempt's place under the root (a private path, left out of the aggregate).
        parts = (campaign.name,) + path.relative_to(campaign).parts
        instance = "/".join(parts[:-1])
        batch: List[Dict[str, Any]] = []
        plugin = False
        with opener(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(TRACE_MARKER):
                    continue
                try:
                    row = json.loads(line[len(TRACE_MARKER):])
                except ValueError:
                    stats["unreadable_rows"] += 1
                    continue
                if not isinstance(row, dict) or row.get("protocol") != TRACE_PROTOCOL:
                    stats["unreadable_rows"] += 1
                    continue
                data = row.get("data") if isinstance(row.get("data"), dict) else {}
                if row.get("kind") == "episode":
                    profile = data.get("profile") if isinstance(data.get("profile"), dict) else {}
                    plugin = plugin or profile.get("plugin") is True
                elif row.get("kind") == "native_turn" and data.get("kind") == "user":
                    batch.append({"type": "turn", "instance": instance, "text": _owner_text(data)})
                elif row.get("kind") == "model_request":
                    payload = data.get("payload")
                    if data.get("truncated") or not isinstance(payload, dict) or \
                            not isinstance(payload.get("messages"), list):
                        stats["truncated_requests"] += 1
                        continue
                    thread = str(row.get("thread") or "")
                    colonymind = colonymind_thread(thread)
                    label = "".join("N" if ch.isdigit() else ch for ch in thread)
                    stats["threads"][label]["colonymind"] = colonymind
                    stats["threads"][label]["calls"] += 1
                    batch.append({"type": "call", "instance": instance, "colonymind": colonymind,
                                  "kind": _schema_name(payload) if colonymind else HOST,
                                  "messages": payload["messages"], "tools": payload.get("tools")})
        if not plugin:
            stats["skipped_no_colonymind"] += 1
            return
        stats["attempts"] += 1
        yield from batch

    return generate(), stats


def jsonl_events(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", type=Path, help="a root of paired campaign directories")
    source.add_argument("--events", type=Path, help="a JSON-lines file of turn and call events")
    parser.add_argument("--tokenizer", required=True, help="a local tokenizer directory (with its chat template)")
    parser.add_argument("--routing", type=Path, help="a routing table (JSON); default: every role on one model")
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--aggregate-only", action="store_true", help="leave out the per-turn and per-night rows")
    args = parser.parse_args(argv)
    routing = Routing(json.loads(args.routing.read_text(encoding="utf-8")) if args.routing else None)
    tokenizer = HFTokenizer(args.tokenizer)
    if args.traces:
        events, stats = trace_events(args.traces)
    else:
        events, stats = jsonl_events(args.events), {}
    result = count(events, tokenizer, routing, block=args.block)
    result["sources"] = {key: (dict(value) if isinstance(value, dict) else value) for key, value in stats.items()}
    result["template_failures"] = tokenizer.template_failures
    if args.aggregate_only:
        result = aggregate(result)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
