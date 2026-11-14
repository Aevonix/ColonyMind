"""The working-memory prize count (``benchmarks/working_memory/prize.py``): for each owner turn, every model call
ColonyMind makes for it, grouped by the model a routing table sends it to; per turn and group the prompt tokens, the
tokens a stock prefix cache serves today (the longest prefix shared with an earlier call of the same group), and
WM1's upper bound (the turn's own text, located in each prompt, counted as shareable if placed first).

The stub tokenizer makes every number checkable by hand: one token per character, a prompt rendered as
``role:content|`` per message. The real tokenizer is checked on a tiny fixture when
``COLONYMIND_WM_TOKENIZER`` names a local copy of it."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "working_memory"


def _prize():
    spec = importlib.util.spec_from_file_location("wm_prize", BENCH / "prize.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CharTokenizer:
    """One token per character; a prompt is ``role:content|`` per message (tools ignored)."""

    name = "char-stub"

    def prompt_text(self, messages, tools=None):
        return "".join(f"{m['role']}:{m['content']}|" for m in messages)

    def render(self, messages, tools=None):
        text = self.prompt_text(messages, tools)
        return text, [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]

    def prompt_ids(self, messages, tools=None):
        return self.render(messages, tools)[1]

    def count(self, text):
        return len(text)


class WordTokenizer(CharTokenizer):
    """A space joins the word after it (`` Hello`` is one token, as in byte-level BPE), so a text's tokens on its
    own need not line up with its tokens inside a longer prompt; a prompt is ``role:content|`` per message."""

    name = "word-stub"
    PATTERN = re.compile(r" ?[^ ]+| +")

    def __init__(self):
        self.vocab = {}

    def _spans(self, text):
        return [match.span() for match in self.PATTERN.finditer(text)]

    def render(self, messages, tools=None):
        text = self.prompt_text(messages, tools)
        spans = self._spans(text)
        return text, [self.vocab.setdefault(text[a:b], len(self.vocab)) for a, b in spans], spans

    def count(self, text):
        return len(self._spans(text))


def _call(kind, system, user, instance="a"):
    return {"type": "call", "instance": instance, "kind": kind, "colonymind": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def _turn(text, instance="a"):
    return {"type": "turn", "instance": instance, "text": text}


T0, T1 = "Pickup at 09:15", "Pickup moved to 10:45"
C1 = ("CLAIMS", "msg: " + T0)
C2 = ("APPRAISE", "ev: " + T0)
C3 = ("CLAIMS", "msg: " + T1)
C4 = ("APPRAISE", "ev: " + T1)


def _events():
    return [_turn(T0), _call("source_claims", *C1), _call("source_appraisal", *C2),
            _turn(T1), _call("source_claims", *C3), _call("source_appraisal", *C4)]


def _plen(system, user):
    return len(f"system:{system}|user:{user}|")


def test_the_count_is_exact_per_turn_and_group_on_a_fixture_with_known_shared_prefixes():
    prize = _prize()
    result = prize.count(_events(), CharTokenizer(), prize.Routing())
    turns = result["turns"]
    assert [t["turn"] for t in turns] == [0, 1]
    g0, g1 = turns[0]["groups"][prize.DEFAULT_MODEL], turns[1]["groups"][prize.DEFAULT_MODEL]
    assert g0["calls"] == 2 and g1["calls"] == 2
    assert g0["prompt_tokens"] == _plen(*C1) + _plen(*C2)
    assert g1["prompt_tokens"] == _plen(*C3) + _plen(*C4)
    # C2 shares "system:" (7) with C1; C3 shares "system:CLAIMS|user:msg: Pickup " (31) with C1;
    # C4 shares "system:APPRAISE|user:ev: Pickup " (32) with C2.
    assert g0["served_tokens"] == 0 + 7
    assert g1["served_tokens"] == 31 + 32
    # Block-aligned at 16 tokens: 7 -> 0, 31 -> 16, 32 -> 32.
    assert g0["served_tokens_block"] == 0 and g1["served_tokens_block"] == 16 + 32
    totals = result["totals"]["by_group"][prize.DEFAULT_MODEL]
    assert totals["prompt_tokens"] == g0["prompt_tokens"] + g1["prompt_tokens"]
    assert totals["served_tokens"] == 70
    assert result["totals"]["owner_turns"] == 2 and result["totals"]["colonymind_calls"] == 4


def test_a_shared_prefix_is_counted_once_per_call_never_against_every_earlier_call():
    prize = _prize()
    # Three identical calls: each later one is served its whole prompt once, not once per earlier match.
    events = [_turn(T0)] + [_call("source_claims", *C1) for _ in range(3)]
    group = prize.count(events, CharTokenizer(), prize.Routing())["turns"][0]["groups"][prize.DEFAULT_MODEL]
    assert group["served_tokens"] == 2 * _plen(*C1)
    assert group["served_tokens"] <= group["prompt_tokens"]


def test_a_call_routed_to_another_model_never_counts_as_shared():
    prize = _prize()
    routing = prize.Routing({"default": "m", "kinds": {"source_appraisal": "n"}})
    result = prize.count(_events(), CharTokenizer(), routing)
    t0, t1 = result["turns"][0]["groups"], result["turns"][1]["groups"]
    assert set(t0) == {"m", "n"}
    # Under one model C2 was served 7 tokens by C1; on its own model nothing came before it.
    assert t0["n"]["served_tokens"] == 0 and t0["m"]["served_tokens"] == 0
    assert t1["m"]["served_tokens"] == 31 and t1["n"]["served_tokens"] == 32
    assert result["routing"] == {"default": "m", "kinds": {"source_appraisal": "n"}}


def test_changing_the_routing_table_moves_the_counts_as_expected():
    prize = _prize()
    one = prize.count(_events(), CharTokenizer(), prize.Routing())["totals"]["by_group"]
    split = prize.count(_events(), CharTokenizer(),
                        prize.Routing({"default": "m", "kinds": {"source_appraisal": "n"}}))["totals"]["by_group"]
    assert one[prize.DEFAULT_MODEL]["served_tokens"] == 70
    assert split["m"]["served_tokens"] + split["n"]["served_tokens"] == 70 - 7
    assert split["m"]["prompt_tokens"] + split["n"]["prompt_tokens"] == one[prize.DEFAULT_MODEL]["prompt_tokens"]


def test_the_wm1_bound_locates_the_turns_text_in_each_prompt():
    prize = _prize()
    result = prize.count(_events(), CharTokenizer(), prize.Routing())
    g0, g1 = (t["groups"][prize.DEFAULT_MODEL] for t in result["turns"])
    assert g0["turn_located_calls"] == 2 and g0["turn_tokens"] == 2 * len(T0)
    assert g1["turn_located_calls"] == 2 and g1["turn_tokens"] == 2 * len(T1)
    # Placed first, the first call of the group computes the turn's block and every later call is served it.
    assert g0["wm1_bound"] == len(T0) and g1["wm1_bound"] == len(T1)


def test_the_bound_is_split_from_what_the_cache_serves_today_so_nothing_is_counted_twice():
    prize = _prize()
    result = prize.count(_events(), CharTokenizer(), prize.Routing())
    g0, g1 = (t["groups"][prize.DEFAULT_MODEL] for t in result["turns"])
    # The turn's text starts after "system:CLAIMS|user:msg: " (24) in C1 and C3, and after
    # "system:APPRAISE|user:ev: " (25) in C2 and C4.
    assert g0["before_turn_tokens"] == 24 + 25 and g1["before_turn_tokens"] == 24 + 25
    # Served today: C1 0, C2 7 ("system:"), C3 31, C4 32; the part before the turn's text is at most its offset.
    assert g0["served_before_turn"] == 0 + 7 and g1["served_before_turn"] == 24 + 25
    # C3 and C4 are already served "Pickup " (7) of the second turn's text by C1 and C2; the first turn's is not.
    assert g0["turn_served_today"] == 0 and g1["turn_served_today"] == 7 + 7
    # WM1's bound counts the turn's text in the later call of each turn (C2, C4); what today's cache does not
    # already serve of it is the bound less the turn tokens served there.
    assert g0["wm1_bound"] == len(T0) and g0["wm1_new_tokens"] == len(T0)
    assert g1["wm1_bound"] == len(T1) and g1["wm1_new_tokens"] == len(T1) - 7
    for group in (g0, g1):
        assert group["served_before_turn"] + group["turn_served_today"] <= group["served_tokens"]
        assert group["wm1_new_tokens"] <= group["wm1_bound"]
    by_group = result["totals"]["by_group"][prize.DEFAULT_MODEL]
    assert by_group["wm1_new_tokens"] == len(T0) + len(T1) - 7
    assert by_group["wm1_new_share"] == round(by_group["wm1_new_tokens"] / by_group["prompt_tokens"], 4)


def test_the_turns_place_is_its_token_offset_in_the_whole_prompt_where_a_space_joins_the_turns_first_word():
    prize = _prize()
    turn = "Hello world"
    events = [_turn(turn), _call("source_claims", "S", "Say: Hello there"),
              _call("source_claims", "S", "Say: " + turn), _call("source_appraisal", "S", "Say: " + turn)]
    tokenizer = WordTokenizer()
    # Whole prompts: "system:S|user:Say:", " Hello", " there|" and twice "system:S|user:Say:", " Hello", " world|".
    # On its own, the text before the turn is two tokens ("system:S|user:Say:", " "); in the prompt the space
    # belongs to " Hello", the turn's first token, so the turn starts at token 1.
    text = tokenizer.prompt_text(events[2]["messages"])
    assert tokenizer.count(text[:text.find(turn)]) == 2
    group = prize.count(events, tokenizer, prize.Routing())["turns"][0]["groups"][prize.DEFAULT_MODEL]
    # Served today: 0, then 2 (" Hello" is cached), then 3 (the whole prompt).
    assert group["served_tokens"] == 0 + 2 + 3
    assert group["turn_located_calls"] == 2 and group["turn_tokens"] == 2 + 2
    assert group["before_turn_tokens"] == 1 + 1 and group["served_before_turn"] == 1 + 1
    # The second call already has the turn's first token (" Hello") served, the third both of its tokens.
    assert group["turn_served_today"] == 1 + 2
    assert group["wm1_bound"] == 2 and group["wm1_new_tokens"] == 0
    assert group["served_before_turn"] + group["turn_served_today"] <= group["served_tokens"]


def test_a_call_without_the_turns_text_adds_nothing_to_the_split():
    prize = _prize()
    events = [_turn(T0), _call("mind_deliberate", "DELIB", "state only"), _call("mind_deliberate", "DELIB", "state")]
    group = prize.count(events, CharTokenizer(), prize.Routing())["turns"][0]["groups"][prize.DEFAULT_MODEL]
    assert group["served_tokens"] > 0
    assert group["before_turn_tokens"] == group["served_before_turn"] == group["turn_served_today"] == 0
    assert group["wm1_new_tokens"] == 0


def test_a_prompt_without_the_turns_text_counts_zero_toward_the_bound():
    prize = _prize()
    events = _events() + [_call("mind_deliberate", "DELIB", "state only")]
    result = prize.count(events, CharTokenizer(), prize.Routing())
    g1 = result["turns"][1]["groups"][prize.DEFAULT_MODEL]
    assert g1["calls"] == 3 and g1["turn_located_calls"] == 2
    assert g1["turn_tokens"] == 2 * len(T1) and g1["wm1_bound"] == len(T1)
    kinds = result["totals"]["by_kind"]
    assert kinds["mind_deliberate"]["turn_tokens"] == 0 and kinds["mind_deliberate"]["calls"] == 1


def test_the_turns_text_is_located_in_its_json_rendering():
    prize = _prize()
    text = 'Owner said "call Sam at 555-0100"\nthen left'
    body = json.dumps({"message": text})
    assert text not in body
    events = [_turn(text), _call("source_claims", "CLAIMS", body), _call("source_appraisal", "APPRAISE", body)]
    group = prize.count(events, CharTokenizer(), prize.Routing())["turns"][0]["groups"][prize.DEFAULT_MODEL]
    rendering = json.dumps(text)[1:-1]
    assert group["turn_located_calls"] == 2 and group["turn_tokens"] == 2 * len(rendering)


def test_a_pass_goes_to_the_latest_turn_whose_text_it_carries_and_other_calls_to_the_turns_window():
    prize = _prize()
    events = [_turn(T0), _turn(T1),
              # A late pass for the first turn (its text, not the second's) still belongs to it.
              _call("source_claims", "CLAIMS", "msg: " + T0),
              # A pass that carries both turns (the conversation before it) belongs to the later one.
              _call("commitment_extract", "COMMIT", T0 + " / " + T1),
              # Deliberation carries no turn text: it belongs to the window of the latest turn.
              _call("mind_deliberate", "DELIB", "state only")]
    result = prize.count(events, CharTokenizer(), prize.Routing())
    t0, t1 = (t["groups"][prize.DEFAULT_MODEL] for t in result["turns"])
    assert t0["calls"] == 1 and t1["calls"] == 2
    assert result["totals"]["by_phase"]["turn"]["calls"] == 2
    assert result["totals"]["by_phase"]["window"]["calls"] == 1


def test_the_nights_stages_are_counted_per_night_not_per_turn():
    prize = _prize()
    events = [_turn(T0), _call("source_claims", *C1),
              _call("mind_consolidate_episode", "EPISODE", "day one"),
              _call("mind_consolidate_narrative", "NARRATIVE", "day one"),
              _turn(T1), _call("source_claims", *C3),
              _call("mind_consolidate_episode", "EPISODE", "day two")]
    result = prize.count(events, CharTokenizer(), prize.Routing())
    assert [t["groups"][prize.DEFAULT_MODEL]["calls"] for t in result["turns"]] == [1, 1]
    nights = result["nights"]
    assert [n["night"] for n in nights] == [0, 1]
    assert [n["groups"][prize.DEFAULT_MODEL]["calls"] for n in nights] == [2, 1]
    # The second night's episode call is served its shared prefix by the first night's.
    assert nights[1]["groups"][prize.DEFAULT_MODEL]["served_tokens"] == len("system:EPISODE|user:day ")
    assert result["totals"]["nights"] == 2


def test_host_calls_fill_the_cache_but_are_never_counted_as_colonyminds():
    prize = _prize()
    host = {"type": "call", "instance": "a", "kind": prize.HOST, "colonymind": False,
            "messages": [{"role": "system", "content": "CLAIMS"}, {"role": "user", "content": "other"}]}
    events = [host, _turn(T0), _call("source_claims", *C1)]
    result = prize.count(events, CharTokenizer(), prize.Routing())
    group = result["turns"][0]["groups"][prize.DEFAULT_MODEL]
    assert group["calls"] == 1 and group["served_tokens"] == len("system:CLAIMS|user:")
    assert result["totals"]["colonymind_calls"] == 1 and result["totals"]["host_calls"] == 1
    # Routed elsewhere, the host's calls serve nothing.
    elsewhere = prize.count(events, CharTokenizer(), prize.Routing({"default": "m", "kinds": {prize.HOST: "h"}}))
    assert elsewhere["turns"][0]["groups"]["m"]["served_tokens"] == 0


def test_instances_never_share_a_cache():
    prize = _prize()
    events = [_turn(T0, "a"), _call("source_claims", *C1, instance="a"),
              _turn(T0, "b"), _call("source_claims", *C1, instance="b")]
    result = prize.count(events, CharTokenizer(), prize.Routing())
    assert [t["groups"][prize.DEFAULT_MODEL]["served_tokens"] for t in result["turns"]] == [0, 0]


def _write_campaign(root, name, split, lines):
    campaign = root / name
    attempt = campaign / "runs" / "001-full" / "attempts" / "case.01"
    attempt.mkdir(parents=True)
    (campaign / "paired.json").write_text(json.dumps({"dataset": {"split": split}}), encoding="utf-8")
    (attempt / "container.log").write_text("".join(lines), encoding="utf-8")
    return attempt / "container.log"


def _row(prize, sequence, kind, data, thread="MainThread"):
    event = {"protocol": prize.TRACE_PROTOCOL, "sequence": sequence, "thread": thread, "kind": kind, "data": data}
    return prize.TRACE_MARKER + json.dumps(event) + "\n"


def _request(prize, sequence, rid, schema, system, user, thread="paired-source-worker", tools=None):
    payload = {"model": "m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": schema, "schema": {}}}
    if tools:
        payload["tools"] = tools
    return _row(prize, sequence, "model_request", {"request_id": rid, "payload": payload}, thread)


def test_paired_traces_become_events_and_held_out_or_unplanned_campaigns_are_never_opened(tmp_path):
    prize = _prize()
    turn = {"index": 0, "kind": "user", "session_id": "owner-1",
            "messages": [{"role": "user", "content": T0}, {"role": "assistant", "content": "Noted."}]}
    episode = _row(prize, 0, "episode", {"arm": "full", "profile": {"name": "full", "plugin": True}})
    lines = [episode,
             _request(prize, 1, 1, None, "HOST", T0, thread="Thread-5 (<lambda>)", tools=[{"type": "function"}]),
             _row(prize, 2, "native_turn", turn),
             _request(prize, 3, 2, "source_claims", *C1),
             _request(prize, 4, 3, "source_appraisal", *C2),
             _request(prize, 5, 4, "mind_deliberate", "DELIB", "state", thread="Thread-9 (run)"),
             _row(prize, 6, "native_turn", {**turn, "kind": "inbound", "session_id": "contact-1"}),
             "a line the harness wrote\n"]
    _write_campaign(tmp_path, "dev-campaign", "dev", lines)
    held = _write_campaign(tmp_path, "some-heldout", "dev", lines)
    unplanned = _write_campaign(tmp_path, "unplanned", "dev", lines)
    (unplanned.parents[4] / "paired.json").unlink()
    other = _write_campaign(tmp_path, "public-eval", "frozen_public_evaluation", lines)
    # An attempt of the host alone (no ColonyMind) holds owner turns but no ColonyMind calls: it is left out.
    bare = [_row(prize, 0, "episode", {"arm": "base_hermes", "profile": {"name": "base_hermes", "plugin": False}})]
    _write_campaign(tmp_path, "dev-bare", "dev", bare + lines[1:])
    opened = []
    real_open = open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    events, stats = prize.trace_events(tmp_path, opener=spy)
    events = list(events)
    assert not any(str(p) in opened for p in (held, unplanned, other))
    assert not any("heldout" in p for p in opened)
    assert stats["attempts"] == 1 and stats["skipped_no_colonymind"] == 1
    assert stats["skipped_held_out_campaigns"] == 1 and stats["skipped_no_plan"] == 1
    assert stats["skipped_not_development"] == 1
    kinds = [(e["type"], e.get("kind"), e.get("colonymind")) for e in events]
    assert kinds == [("call", prize.HOST, False), ("turn", None, None), ("call", "source_claims", True),
                     ("call", "source_appraisal", True), ("call", "mind_deliberate", True)]
    assert events[1]["text"] == T0
    result = prize.count(events, CharTokenizer(), prize.Routing())
    assert result["turns"][0]["groups"][prize.DEFAULT_MODEL]["calls"] == 3


def _campaign_lines(prize):
    turn = {"index": 0, "kind": "user", "messages": [{"role": "user", "content": T0}]}
    return [_row(prize, 0, "episode", {"arm": "full", "profile": {"plugin": True}}),
            _row(prize, 1, "native_turn", turn), _request(prize, 2, 1, "source_claims", *C1)]


def _spy(opened):
    def opener(path, *args, **kwargs):
        opened.append(str(path))
        return open(path, *args, **kwargs)
    return opener


def _reserved(path):
    return any(marker in part.lower() for part in Path(path).resolve().parts for marker in ("heldout", "held-out",
                                                                                              "held_out"))


def test_a_root_beneath_a_reserved_ancestor_is_refused_before_anything_is_opened(tmp_path):
    prize = _prize()
    for number, name in enumerate(("held-out", "HeldOut", "held_out")):
        root = tmp_path / "results" / name / "traces"
        _write_campaign(root, "dev-campaign", "dev", _campaign_lines(prize))
        opened = []
        with pytest.raises(ValueError, match="held-out"):
            prize.trace_events(root, opener=_spy(opened))
        assert opened == []
        # The same root reached through a link with a plain name.
        link = tmp_path / f"plain-{number}"
        link.symlink_to(root)
        with pytest.raises(ValueError, match="held-out"):
            prize.trace_events(link, opener=_spy(opened))
        assert opened == []


def test_a_link_whose_target_is_reserved_is_never_opened(tmp_path):
    prize = _prize()
    lines = _campaign_lines(prize)
    root = tmp_path / "root"
    root.mkdir()
    store = tmp_path / "heldout-store"
    # A campaign linked under a plain name whose target lies in a reserved directory.
    _write_campaign(store, "campaign-a", "dev", lines)
    (root / "campaign-a").symlink_to(store / "campaign-a")
    # A campaign whose plan is a link into a reserved directory.
    plan_link = _write_campaign(root, "campaign-b", "dev", lines).parents[4]
    (plan_link / "paired.json").unlink()
    (store / "plans").mkdir()
    (store / "plans" / "paired.json").write_text(json.dumps({"dataset": {"split": "dev"}}))
    (plan_link / "paired.json").symlink_to(store / "plans" / "paired.json")
    # A plain campaign with one real attempt and one attempt directory linked into a reserved directory.
    real = _write_campaign(root, "campaign-c", "dev", lines)
    reserved_attempt = tmp_path / "held_out" / "case.02"
    reserved_attempt.mkdir(parents=True)
    (reserved_attempt / "container.log").write_text("".join(lines))
    (real.parents[1] / "case.02").symlink_to(reserved_attempt)
    # And an attempt whose log alone is a link.
    (real.parents[1] / "case.03").mkdir()
    (real.parents[1] / "case.03" / "container.log").symlink_to(reserved_attempt / "container.log")
    opened = []
    events, stats = prize.trace_events(root, opener=_spy(opened))
    events = list(events)
    assert opened and not any(_reserved(path) for path in opened)
    assert stats["attempts"] == 1 and sum(1 for e in events if e["type"] == "turn") == 1
    assert stats["skipped_held_out_campaigns"] == 2 and stats["skipped_held_out_attempts"] == 2


def test_a_truncated_request_is_counted_as_unreadable_never_as_a_call(tmp_path):
    prize = _prize()
    turn = {"index": 0, "kind": "user", "messages": [{"role": "user", "content": T0}]}
    lines = [_row(prize, 0, "episode", {"arm": "full", "profile": {"plugin": True}}),
             _row(prize, 1, "native_turn", turn),
             _row(prize, 2, "model_request", {"truncated": True, "original_bytes": 900000, "prefix": "{"},
                  "paired-source-worker"),
             _request(prize, 3, 2, "source_claims", *C1)]
    _write_campaign(tmp_path, "dev-campaign", "development", lines)
    events, stats = prize.trace_events(tmp_path)
    events = list(events)
    assert stats["truncated_requests"] == 1
    assert sum(1 for e in events if e["type"] == "call") == 1


def test_the_aggregate_carries_no_instance_names():
    prize = _prize()
    result = prize.count(_events(), CharTokenizer(), prize.Routing())
    aggregate = prize.aggregate(result)
    assert "turns" not in aggregate and "nights" not in aggregate
    assert aggregate["totals"] == result["totals"]
    assert set(aggregate["per_turn"]) >= {"prompt_tokens", "served_tokens", "wm1_bound"}


@pytest.mark.skipif(not os.environ.get("COLONYMIND_WM_TOKENIZER"),
                    reason="COLONYMIND_WM_TOKENIZER does not name a local tokenizer directory")
def test_on_a_tiny_fixture_the_real_tokenizer_gives_the_naive_longest_shared_prefix():
    pytest.importorskip("transformers")
    prize = _prize()
    tokenizer = prize.HFTokenizer(os.environ["COLONYMIND_WM_TOKENIZER"])
    events = _events()
    result = prize.count(events, tokenizer, prize.Routing())
    prompts = [tokenizer.render(e["messages"])[1] for e in events if e["type"] == "call"]

    def naive(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    served = [0] + [max(naive(prompts[i], prompts[j]) for j in range(i)) for i in range(1, len(prompts))]
    g0, g1 = (t["groups"][prize.DEFAULT_MODEL] for t in result["turns"])
    assert g0["prompt_tokens"] == len(prompts[0]) + len(prompts[1])
    assert g0["served_tokens"] == served[0] + served[1]
    assert g1["served_tokens"] == served[2] + served[3]
    assert 0 < g1["served_tokens"] < g1["prompt_tokens"]
    # The turn's tokens are counted inside each whole prompt (the exact split is checked in the next test).
    assert 0 < g0["turn_tokens"] < g0["prompt_tokens"]
    for group in (g0, g1):
        assert 0 < group["before_turn_tokens"] < group["prompt_tokens"] - group["turn_tokens"]
        assert group["served_before_turn"] + group["turn_served_today"] <= group["served_tokens"]
        assert group["turn_served_today"] <= group["turn_tokens"]
        assert 0 <= group["wm1_new_tokens"] <= group["wm1_bound"]
    # A host's tool history (arguments as a JSON string, as the wire carries them) renders through the template.
    call = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\": \"pickup\"}"}}
    history = [{"role": "system", "content": "HOST"}, {"role": "user", "content": T0},
               {"role": "assistant", "content": "", "tool_calls": [call]},
               {"role": "tool", "tool_call_id": "c1", "content": "{\"found\": 1}"}]
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up.",
                                               "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    assert len(tokenizer.render(history, tools)[1]) > 0 and tokenizer.template_failures == 0


@pytest.mark.skipif(not os.environ.get("COLONYMIND_WM_TOKENIZER"),
                    reason="COLONYMIND_WM_TOKENIZER does not name a local tokenizer directory")
def test_with_the_real_tokenizer_the_split_matches_token_boundaries_read_back_by_decoding():
    pytest.importorskip("transformers")
    prize = _prize()
    tokenizer = prize.HFTokenizer(os.environ["COLONYMIND_WM_TOKENIZER"])
    turn = "Hello world"
    events = [_turn(turn), _call("source_claims", "S", "Say: Hello there"),
              _call("source_claims", "S", "Say: " + turn), _call("source_appraisal", "S", "Say: " + turn)]
    group = prize.count(events, tokenizer, prize.Routing())["turns"][0]["groups"][prize.DEFAULT_MODEL]
    prompts = [e["messages"] for e in events if e["type"] == "call"]
    ids = [tokenizer.render(m)[1] for m in prompts]

    def naive(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    def decoded(prompt, k):
        return len(tokenizer._tok.decode(prompt[:k], skip_special_tokens=False, clean_up_tokenization_spaces=False))

    expected = dict.fromkeys(("turn_tokens", "before_turn_tokens", "served_before_turn", "turn_served_today",
                              "wm1_bound", "wm1_new_tokens"), 0)
    for index in (1, 2):
        served = max(naive(ids[index], ids[j]) for j in range(index))
        text = tokenizer.render(prompts[index])[0]
        at = text.find(turn)
        # The turn's first token is the first whose decoded prefix reaches past the turn's first character; its
        # last is the first whose decoded prefix covers the turn's last one.
        first = next(k for k in range(len(ids[index])) if decoded(ids[index], k + 1) > at)
        end = next(k for k in range(first, len(ids[index]) + 1) if decoded(ids[index], k) >= at + len(turn))
        already = max(0, min(served, end) - first)
        expected["turn_tokens"] += end - first
        expected["before_turn_tokens"] += first
        expected["served_before_turn"] += min(served, first)
        expected["turn_served_today"] += already
        if index == 2:
            expected["wm1_bound"] += end - first
            expected["wm1_new_tokens"] += end - first - already
    assert {key: group[key] for key in expected} == expected
    # The cached " Hello" of the first call is the turn's first token in the second.
    assert expected["turn_served_today"] > expected["wm1_bound"]
