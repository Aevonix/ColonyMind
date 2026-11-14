# Public memory datasets

Adapters for the public datasets the memory benchmark reads. Each one downloads its upstream files once into a
local cache, uses a file only when its SHA-256 matches the pin, and turns it into the same records
(`records.py`): conversations of dated sessions in time order, and questions with the time they are asked, their
gold sessions, their gold turns and whether they are abstention questions. The data is never committed.

## Sources

| Dataset | Upstream | Pinned at | File | SHA-256 | Licence |
| --- | --- | --- | --- | --- | --- |
| LongMemEval-S, cleaned | `huggingface.co/datasets/xiaowu0162/longmemeval-cleaned` | revision `98d7416c24c7` | `longmemeval_s_cleaned.json` (277,383,467 bytes) | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` | MIT |
| LongMemEval judge prompts | `github.com/xiaowu0162/LongMemEval` | commit `9e0b455f4ef0` | `src/evaluation/evaluate_qa.py` | `ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251` | MIT |
| BEAM, 100K split | `huggingface.co/datasets/Mohammadta/BEAM` | revision `3205395e897e` | `data/100K-00000-of-00001.parquet` | `c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace` | CC BY-SA 4.0 |
| PrecisionMemBench | `github.com/tenurehq/precisionMemBench` | commit `61a712a1754e` | `fixtures/beliefs.seed.json` | `35659b3b9a581b644fc6746e1b4e902c3e7b56c0973b546bf7b0816a69e3de98` | MIT |
| | | | `fixtures/retrieval.cases.json` | `09e99f0852280538abe34b03570f65583fdb10df3c554b7150f0f665d5db709c` | |
| | | | `fixtures/session-retrieval.cases.json` | `f24a359cdb2bf50237e1ab463af938cd54a81917f64c8d97304de318aed4cc1c` | |

The full URLs are in each adapter's pins. Papers: Wu et al., "LongMemEval: Benchmarking Chat Assistants on
Long-Term Interactive Memory" (ICLR 2025); Tavakoli et al., "Beyond a Million Tokens: Benchmarking and Enhancing
Long-Term Memory in LLMs" (arXiv 2510.27246); PrecisionMemBench (arXiv 2605.11325).

BEAM's licence is CC BY-SA 4.0, as its dataset card states at the pinned revision: anything published that is
derived from its data (per-question rows, quoted questions or answers) carries the attribution and the same licence.

## Use

From the repository root, with the sidecar installed (the BEAM adapter reads parquet with `pyarrow`):

```
python -m benchmarks.public_memory.datasets fetch                 # every pinned file
python -m benchmarks.public_memory.datasets verify longmemeval_s  # parse, check the counts, list the findings
python -m benchmarks.public_memory.datasets subset longmemeval_s --seed 1 --size 60 --out subset.json
python -m benchmarks.public_memory.datasets subset longmemeval_s --seed 1 --quota knowledge-update=4 \
    --quota single-session-user=4 --exclude subset.json --out dev.json
python -m benchmarks.public_memory.datasets check longmemeval_s subset.json
```

Files go to `--data-dir`, else `COLONYMIND_BENCH_DATA_DIR`, else `~/.cache/colonymind/public-datasets`. A cached
file that does not match its pin is refused and left in place. `--offline` never downloads. The sidecar test suite
reads the pinned files only when `COLONYMIND_BENCH_DATA_DIR` holds them, and skips those tests otherwise.

A subset is stratified by question type (or by type and abstention). Each stratum gets its share of the size by
largest remainder, or an explicit quota; within a stratum questions are taken in the order of SHA-256 of
`"<seed>:<question id>"`. The record names the dataset's SHA-256, the seed, the strata, the ids and the ids left out,
and `check` draws it again from the data: it refuses a record whose ids are not the ones drawn, or whose size is
not the number of its ids.

## Times

Every time is UTC. A session's `at` is when it happened, and sessions replay in time order (a tie keeps the file's
order). A question's `asked_at` is never before a session of its conversation; `date_text` is the question date
exactly as the dataset gives it to a reader, where it gives one.

- LongMemEval stamps are `YYYY/MM/DD (Day) HH:MM` local wall times with no zone, read as UTC; a stamp whose weekday is
  not its date's is refused. Where a question is stamped earlier than a session of its own day, it is asked one
  minute after that session, still on that day (23:59:59 at the latest). A session on a later day than its question
  would be refused; the pinned file has none.
- BEAM gives each session a day (`March-15-2024`) and no time of day; the session is placed at 12:00 UTC, which keeps
  the calendar day in any zone from UTC-11 to UTC+11. BEAM gives no question date: its questions are asked about the
  whole conversation, at 12:00 UTC the day after the last session.
- Two sessions can share a start (95 LongMemEval questions have a tie), so a replay that spaces a session's turns
  a second apart must not assume distinct session starts. BEAM sessions hold up to 132 turns.

## Gold turns

- LongMemEval: the gold sessions are `answer_session_ids`; the gold turns are the turns marked `has_answer` (896 in
  the pinned file). 62 questions have a gold session that marks no turn, among them all 30 abstention questions;
  21 questions, all abstention questions, mark no turn at all. Reach there can be checked per session only.
- BEAM: every ability except abstention names its evidence by chat id (`source_chat_ids`), so reach can be checked
  per turn. 355 of the 400 questions name evidence turns; the other 45 are the 40 abstention questions, 4
  summarization questions and 1 preference-following question. A knowledge-update question labels the turn of the
  original value and of the updated one, a contradiction question its two statements, a temporal question its two
  events.

## Counts checked in the pinned files

| Dataset | Checked |
| --- | --- |
| LongMemEval-S | 500 questions: 133 multi-session, 133 temporal reasoning, 78 knowledge update, 70 single-session user, 56 single-session assistant, 30 single-session preference; 30 abstention (`_abs`); 23,867 sessions (about 48 per question); 122,416 user turns (about 245 per question) |
| BEAM 100K | 20 conversations, 90 sessions, 2,866 user turns; 400 questions, 40 for each of 10 abilities, 40 of them abstention |
| PrecisionMemBench | 35 seed beliefs (one of another user; four superseded, among them a chain of three versions); 77 single-turn cases in 13 categories; 2 session scenarios with 12 scored turns |

What the LongMemEval file needed, by finding: 211 questions list their sessions out of time order; 95 have two
sessions at the same minute; 76 are stamped before a session of their own day (4 of them asked at 23:59:59); 13 list
one session twice (each occurrence is kept, the second under `<id>#2`).

The PrecisionMemBench README's category table lists 8 fuzzy-matching cases; the pinned fixture holds 7, which matches
the README's own total of 89 scored cases. One session turn states a condition inside another tier; it is recorded
under `unread_conditions` and its raw form is kept.

## LongMemEval's abstention judge prompt

It is in `src/evaluation/evaluate_qa.py` of the LongMemEval repository at the pinned commit: `get_anscheck_prompt`
holds one prompt per question type and, in its `abstention=True` branch (line 41), the abstention prompt, which the
evaluator chooses when the question id contains `_abs` (line 101). The file is pinned by hash (`JUDGE_PIN`).

## PrecisionMemBench and turn capture

PrecisionMemBench scores which stored beliefs surface for a query, in four output tiers. Its seed beliefs have no
source turns, so a replay through turn capture would state each belief as an owner message and measure claim
formation as well as retrieval; `pinned`, `epistemic_status` and `superseded_by` reach memory only as far as the words
carry them. The tiers have no one-to-one counterpart in a recall packet: only "this belief surfaces, that one does
not" maps, through the turn each belief was replayed from. The two session scenarios are real turns and map directly;
their mid-session belief creation and alias update are harness actions with no turn behind them. The one belief of
another user maps onto person scope.
