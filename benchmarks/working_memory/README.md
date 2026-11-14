# Working memory: the prefix prize count and the handback experiment

Two measurements that decide what shared working memory is worth building. Neither changes ColonyMind's behaviour;
nothing here is imported by the sidecar.

- **The prefix prize count** (`prize.py`) asks how much of ColonyMind's own prompt work a stock prefix cache serves
  today, and how much more it could serve if every same-model call put the owner's turn first (WM1, the mind prefix
  and turn block).
- **The handback experiment** (`handback/`) asks what a parent agent needs to keep from a finished worker to answer
  detailed questions about the worker's material later: the worker's summary (today's contract), verbatim excerpts
  with handles, a consult of the worker's exact context (WM3's branches), or the full text.

Every number in this directory that is not a measurement is a choice made when the benchmark was built (labelled
"my assumption" in the code), not a measured bound. The bars the verdict applies were set by the design before any
run and are its assumptions too.

## The prefix prize count

```sh
python benchmarks/working_memory/prize.py --traces RESULTS_ROOT --tokenizer MODEL_DIR \
    [--routing TABLE.json] [--block 16] [--aggregate-only] --out count.json
python benchmarks/working_memory/prize.py --events events.jsonl --tokenizer MODEL_DIR ...
```

No model runs, only a tokenizer (CPU). For each owner turn, every model call ColonyMind makes for it is counted and
grouped by the model it reaches under a routing table:

- the per-turn passes (claims, the claim review, appraisal, commitments, opinions);
- the other calls within the turn's window, where they fire (deliberation, the gate's second opinion, the typed
  calls, Instinct's reader tier, the composer);
- the night's stages, per night rather than per turn.

Per turn (and per night) and per group it reports the prompt tokens as the model's chat template renders them; the
tokens a stock prefix cache serves today (the longest prefix shared with an earlier call of the same group, each
call counted once, also rounded down to whole cache blocks); and WM1's upper bound (the turn's own text, found in
each prompt verbatim or as a JSON string, served to every call of the group after the first once it is placed
first). The host's own calls fill the cache of the model they reach, and are never counted as ColonyMind's.

The bound and today's count overlap where today's cache already serves part of the turn's text, so in every call
that carries the turn the count also splits what is served today around the turn's place in the rendered prompt.
The place is read from one tokenization of the whole prompt with each token's character span (the turn's tokens are
those that hold any of its characters, so a token that joins a space before the turn with its first word is the
turn's), never from the text before the turn tokenized on its own. The split gives the tokens before the turn,
today's served tokens that lie before it (what a turn-first order would have to keep ahead of the turn to go on
serving), the turn's tokens already served today, and `wm1_new_tokens`, the bound less those, which is all the bound
adds to today's count.

**Sources.** `--traces` reads the request traces the paired harness writes (`COLONYMIND_PAIRED_DIAGNOSTIC:` lines in
each attempt's `container.log`). A campaign's `paired.json` must declare a development split before any of its
traces is opened, and an attempt of the host alone is left out. A path with a held-out component is never opened or
listed: the root, each campaign, its plan and each attempt's log are checked in every component of the absolute
path as written (ancestors included) and of the path it resolves to (links followed), and a held-out root is refused.
`--events` reads turn and call events as JSON lines (the shape a replay through the scripted model server can be
turned into); a replay is an estimate, since scripted replies shape later prompts.

**Routing.** `{"default": "<group>", "kinds": {"<kind>": "<group>"}}`, where a kind is a call's response schema name
(`untyped` without one) and `host` names the host's calls. The default puts every role and the host on one model. A
deployment's own table stays in its own files; counts under it are private and never committed.

**Limits.** The served count is an upper bound: it assumes no eviction and a cache shared by every call of the group
within an attempt, while an engine recomputes at least the last token and evicts under pressure. A paged cache
matches whole blocks; a hybrid linear-attention model's block can be far larger than the default 16 tokens (vLLM
0.30.0 picks 528 for Qwen3.5-4B), so the count is given at both. The traces' prompts were produced by the campaigns'
own models; only the token counts use the tokenizer named here.

## The handback experiment

```sh
cd benchmarks/working_memory
python -m handback.sets 1301 1302 1303          # the hashes frozen in handback/frozen.json
python -m handback.run --base-url http://127.0.0.1:PORT/v1 --model NAME --tokenizer MODEL_DIR \
    --out OUT --model-revision REV [--seeds 1301 1302 1303]
python -m handback.score --set-seeds 1301 1302 1303 --answers OUT/set-*/answers.jsonl --out report.json
```

**Sets** (`handback/sets.py`). A set is a pure function of its seed: 8 children, each a finished worker task with
8,000 to 24,000 tokens of material (document passages, tool results and messages, each with a handle such as
`d2.p3`, `t4` or `m7`) and the worker's own work notes. Each child has 40 keyed questions in equal shares:
contradictions (a value stated and later corrected; the gold is the later value, the stale one a distractor),
relations (a person's direct manager, the task another depends on directly; the next link of the chain is a
distractor, and a test checks from the material that each relation question admits its gold and nothing else),
questions across children (a value in this child's material for an entity another child's material assigns to a
person), and single facts. Gold answers are unique codes, so scoring is an exact match. Only generic names appear
(Owner, Sam, Robin, coded people such as `p-214`). The runner and the scorer refuse a set whose hash is not frozen.
The sets are version `wm-handback-2`: version 1 asked which task must be finished before another can start, which
the dependency's own dependency answers too; version 2 asks for the direct prerequisite, and draws everything else
from version 1's random stream, so each seed's material is version 1's byte for byte.

**Arms** (`handback/arms.py`), each answered through the same parent prompt, which carries the handbacks of the
children a question involves and then the question:

- `summary`: the child's own summary, written by the model from the child's exact context under a stock host's
  subagent completion instruction (today's contract);
- `excerpts`: the child names handles; the runner copies those passages verbatim with their handles, in the child's
  order, and stops before the first that would pass 2,000 tokens per child;
- `consult`: the question appended to the child's exact context, byte for byte, with no tools offered; the child's
  answer, with handles, is what the parent reads;
- `full`: the parent reads the child's task, full material and work, cold.

**Runner** (`handback/run.py`). One model for every arm and role, greedy decoding, thinking off. It refuses any base
URL that is not loopback, ignores proxy settings and follows no redirect. A child's summary, excerpt selection and
consults share one cache salt and run back to back, so a prefix cache can serve the child's context to each consult;
every parent call has a salt of its own (a cold read), drawn fresh on every try, a retry or a resumed request
included. Every salt carries a nonce drawn when the run starts, so a second run of the same sets against the same
server shares no cached prefix with the first. The server must report cached prompt tokens (vLLM:
`--enable-prompt-tokens-details`) for the consult's cost line. Each set's directory holds the run's identity in
`run.json`, written before the first request: the served model and its revision, a hash of the tokenizer's files,
the set's version, seed and hash, a fingerprint of every prompt the run sends, the decoding settings, the output caps
and the excerpt budget. A stopped run resumes only under the same identity, and outputs with no identity beside them
are never resumed, so one run's rows are never mixed with or relabelled as another's.

**Scorer** (`handback/score.py`). An answer is right when it names the gold code and no distractor; `unknown` is
wrong. Accuracy per arm, overall and per question type; 95% percentile bootstrap intervals of the paired accuracy
difference between arms, resampling questions; cost per question (prompt, cached and incremental tokens, wall-clock,
the share of calls where the cache hit at all), and for the consult per consult call. The verdict applies the
design's bars: the consult passes when it lands within 1 question of the full text and at least 10 points above the
summary; if the excerpts alone land within 1 question of the full text, excerpts suffice and take precedence. A
verdict needs complete coverage (every arm answered every question of the scored sets exactly once, and no row names
a question outside them); otherwise the outcome is `incomplete` and the report's `coverage` says what is missing.

## Results

`results/2026-09-29.json` holds the aggregate numbers below, the pooled score report (every arm, type, pair, cost
line and the coverage) and each set's accuracy. Every size, cap, budget and seed in it is a choice made when the
benchmark was built, not a measured bound; the bars are the design's, set before the run.

### The prefix prize count, 2026-09-29

The request traces of 1,550 development attempts with ColonyMind (695 attempts of the host alone left out, no
held-out campaign read): 3,777 owner turns, 17,900 ColonyMind calls and 17,106 host calls, counted with Qwen3.5-4B's
chat template under the default table (every role and the host on one model). No template refused a prompt, and
every located turn's text was found in its rendered prompt.

| | Tokens | Share of ColonyMind's prompt tokens |
|---|---|---|
| Prompt tokens | 39,787,099 (per turn: mean 10,168, median 8,621, p90 16,268) | 100% |
| Served by a stock prefix cache today (no eviction) | 21,756,909 | 54.7% |
| the same in whole 16-token blocks | 21,635,824 | 54.4% |
| the same in whole 528-token blocks (the hybrid model's block in vLLM 0.30.0) | 19,407,168 | 48.8% |
| Tokens before the turn's text, in the 13,136 calls that carry it | 28,407,029 | 71.4% |
| of which served today | 16,039,916 | 40.3% (73.7% of all served tokens) |
| The turn's own text, summed over those calls | 850,169 | 2.1% (per turn: mean 225, median 192) |
| WM1's bound: the turn served to every call after the first | 596,095 | 1.5% |
| What the bound adds to today's count (`wm1_new_tokens`) | 501,977 | 1.3% |

By pass, the commitment extraction holds 46% of the prompt tokens, claims 23%, appraisal 17%, the claim review 10%
and opinions 3%; the night's stages and the other calls in a turn's window hold under 1% between them. Each per-turn
pass carries a long, stable instruction block before a short turn, so what today's cache serves is mostly those
instructions, shared with earlier calls.

What this says for WM1. Today's order already serves 54.7% of the same-model prompt tokens in these traces (48.8% at
the hybrid model's block size), against WM1's bar of 50% (the design's assumption). Sharing the turn itself can add
at most 1.3 points, because an owner's turn is short beside the passes' instructions. Almost three quarters of what
is served today lies before the turn, so an order that put the turn first, with each pass's instructions after it,
would lose far more than it gains; a WM1 order has to keep the stable instructions ahead of the turn (one canonical
prefix that holds them, then the turn, then a short tail per pass) to add anything. These are upper bounds (no
eviction; one cache per attempt, shared by every call of the group), the traces' prompts came from the campaigns'
own models, and a paired campaign's day is not a deployment's day.

### The handback experiment, 2026-09-29

Three frozen sets of version `wm-handback-2` (seeds 1301 to 1303; 24 children, 960 questions, every arm on every
question), Qwen/Qwen3.5-4B (revision 851bf6e8) served by vLLM 0.30.0 on one 24 GB GPU, bf16, prefix caching on,
greedy decoding, thinking off, one model for every arm and role, in one start with no resume. Child contexts were
9,561 to 24,169 tokens (mean 17,864) with the chat template; summaries 132 to 416 tokens (mean 310); excerpts 12 to
16 passages (mean 14.1, 1,939 tokens) of the 33 to 40 handles each child named. No request failed; 43 of the 1,200
consult answers reached their 256-token cap. Every arm answered every question once (complete coverage).

| Arm | Correct of 960 | Accuracy | Contradiction | Relation | Across children | Single fact |
|---|---|---|---|---|---|---|
| Summary (today's contract) | 10 | 1.0% | 4.2% | 0.0% | 0.0% | 0.0% |
| Excerpts with handles (2,000 tokens) | 91 | 9.5% | 0.4% | 16.7% | 0.8% | 20.0% |
| Consult | 699 | 72.8% | 93.3% | 96.7% | 1.3% | 100.0% |
| Full text | 677 | 70.5% | 89.2% | 87.1% | 5.8% | 100.0% |

Paired differences (95% bootstrap intervals, resampling questions): full text minus consult -2.3 points (-3.9 to
-0.7); consult minus summary +71.8 (+68.8 to +74.6); full text minus excerpts +61.0 (+57.9 to +64.3); excerpts minus
summary +8.4 (+6.5 to +10.4).

**Verdict under the pre-registered bars: `branches_earned`.** The consult landed within 1 question of the full text
(22 more right) and 71.8 points above the summary; the excerpts landed 586 questions below the full text, so they do
not suffice. Per set (secondary, never used for the verdict), the consult passed on all three sets (232 against 222,
235 against 233, 232 against 222).

This run replaces the first run of the same day on version 1 of the sets (same seeds and material; consult 677,
full text 676, excerpts 90, summary 10; also `branches_earned`). The rewording moved the dependency questions most:
the consult got all 120 of them right against 98 before, the full text 120 against 119; no answer in either run
named only the dependency's own dependency. The summaries and excerpt selections were the same in both runs.

Cost per question. The full text read 22,207 prompt tokens cold (mean 1.84 s, p95 3.84 s). The consult's 1,200
calls each continued a child's cached context: 17,792 prompt tokens, 17,510 served from the prefix cache (98.4%, a
hit on every call), 282 computed (mean 0.64 s, p95 1.96 s); with the parent's read, 524 computed tokens per question
(mean 0.89 s). The summary's parent read 495 tokens and the excerpts' 2,533.

Limits. The consults ran right after their child finished, while its context was still cached; a consult after
eviction pays the full text's cold read. A question that spans two children defeats every arm on a 4B model (the
full text reached 5.8%), and a consult cannot join two children's material by construction (each child answers from
its own); across children the full text did better (14 against 3). "Within 1 question" is a bar far narrower than
the paired interval (about 1.6 points either way). A 4B stand-in writes worse summaries and chooses worse excerpts
than a larger model would, which moves every arm; the comparison between arms is the result.
