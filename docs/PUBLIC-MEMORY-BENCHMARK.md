# Public memory benchmark

How well ColonyMind's memory serves a conversation, measured on public datasets at matched memory budgets, and the
published results. The method below is written by hand; every pin, prompt and number on this page is generated from
the harness's code and the committed results files (`python benchmarks/public_memory/export.py page`), and CI checks
that the page is current, so it never states what the harness does not do or what no run measured.

## What is measured

For each question, each arm puts a memory section in front of one fixed reader model at a memory budget (a number of
characters), and the reader answers from it. Three things are scored apart and never averaged into one number:

- **Evidence reached**: whether what the arm showed cites a message that holds the answer. For ColonyMind's arms this
  is structural: the identifiers of the gold evidence messages against the identifiers each shown item cites, with no
  text matched. A baseline that shows whole sessions reaches when it shows a gold session.
- **Accuracy**: whether the answer is right, judged by the dataset's own protocol.
- **Traps**: an old value asserted as current after it changed, one side of a contradiction chosen when nothing
  settles it, or a specific answer given to a question nothing answers. Asking which value holds is recorded apart.

A right answer from evidence that never reached the reader is a guess; a wrong answer from evidence that did is a
reading failure: the two-by-two of reached and right keeps them apart. For knowledge updates and contradictions read
from a ColonyMind store, the results also say how the store held the value (see [Scoring](#scoring)).

## Datasets, budgets and prompts

Public datasets only, each file pinned by its SHA-256; the data is never committed. The adapters are described in
[the datasets README](../benchmarks/public_memory/datasets/README.md).

<!-- generated: method -->
| Dataset | Pinned upstream file | SHA-256 | Licence |
| --- | --- | --- | --- |
| BEAM, 100K split (`beam`) | [Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM/resolve/3205395e897e7318c7b094ef4e6047b9b82dbb03/data/100K-00000-of-00001.parquet) | `c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace` | CC BY-SA 4.0 |
| LongMemEval-S (`longmemeval_s`) | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json) | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` | MIT |

Budgets: 4,000 and 24,000 characters by default; 10,000 is the host's default inline cap.

**The reader** reads every arm alike, at temperature 0 with at most 1,024 output tokens and a strict JSON schema (`answer`, `answerable`). Its system prompt:

```text
You are an assistant answering a question from your user. The memory below is what your memory system recalled from your earlier conversations with this user; it may be incomplete, and it may hold nothing that answers the question. Answer from what the memory holds and the date the question is asked. Reply with one JSON object and nothing else: {"answer": "<your answer>", "answerable": <true or false>}. Set answerable to false when the question depends on something the memory does not hold, and then say in answer what is missing.
```

Its user message is the question date, the question and the memory the arm provided, as the host renders it (`(no memory was provided)` when there is none). Reading SHA-256 (the fixed prompt, format and settings): `08224b93cfeefd53293fe3187f319f6c982a2ee26e6b7a32e1b9cc2378b3aced`.

**The judge** is never the reader's model. LongMemEval answers are checked with LongMemEval's own per-type prompts and its abstention prompt, and BEAM answers with BEAM's own rubric prompt, copied verbatim from the upstream code with their citations and licence notices ([judge_prompts.py](../benchmarks/public_memory/judge_prompts.py)). One call at temperature 0, or three votes at 0.7 whose majority decides. Traps are judged as structures. The trap prompt (a knowledge update or a contradiction, shown the frozen values in an order that says nothing about which is which):

```text
You read a question, an answer to it, and a numbered list of values. Say which of the listed values the answer states as the answer to the question: for something that changed, the value it gives as holding now; for statements that disagree, the side it takes. A value the answer mentions only to say that it no longer holds, was replaced or is wrong is not stated as the answer. Say also whether the answer points out that the information changed or disagrees and asks which value holds. Reply with one JSON object and nothing else: {"asserted": [<the numbers of the values stated as the answer>], "asks_which_holds": <true or false>}.
```

The abstention prompt (for a question nothing answers, when the reader claimed an answer):

```text
You read a question and an answer to it. Say whether the answer states a specific answer to the question: a particular value, name, date, number, place, event or description given as what holds, even with a hedge. An answer that says the information is missing, unknown or was never mentioned, or that declines or asks for more, states no specific answer, even when it names what is missing. Reply with one JSON object and nothing else: {"states_answer": <true or false>}.
```

Judge prompts SHA-256: `fc34f8fcea5e028c1a31cb83395a4908c7c3f86173736ea3e93bf1db288246b9`.
<!-- end generated: method -->

## Arms

Every ColonyMind arm is one change from `all-on` (every faculty on, the configuration a release ships) and reads what
`POST /v1/host/context/assemble` would inject into a host's turn for the question at the budget. The baselines read
the raw history, packed into one section of the same budget by the same packer a host's turn uses.

| Arm | What it shows the reader, and what it isolates |
| --- | --- |
| `all-on` | The store formed by replaying the history through the sidecar, with its night: the product as it ships. |
| `lexical-only` | The formed store with no embedder: what vectors add. |
| `reranker-off` | The formed store with no reranker or calibrated cutoff: what reranking adds. |
| `no-night` | The store before its night: what consolidation adds. |
| `memory-core` | The formed store with the mind switched off: what the mind adds to memory. |
| `import-only` | The history imported as recorded messages, with no claims and no model call: what claims add. |
| `bm25-sessions` | Whole sessions ranked by BM25: plain lexical retrieval. |
| `oracle` | The gold sessions only: how well the reader reads at the budget when retrieval is perfect. |
| `full-context` | The whole history, where it fits the reader's context: the ceiling. |
| `session-search` | Stock Hermes' session search, called once with the question as asked: the host alone. |
| `session-search-or` | Stock Hermes' session search with the question's words joined by OR. |

A replay writes no host notes, so the stock Hermes arms stand for the host's session search, not all of its memory.

## Scoring

- **Verdict**: `right`, `wrong`, `reader_refused` (the reader's reply was not the asked-for structure: a wrong answer),
  `undecided` (the judge's votes gave no majority) or `skipped` (the host failed to serve the capture, or the full
  history does not fit the reader). Accuracy counts the answers with a verdict; the others are listed as failures.
- **Traps** come from structured judgments of what an answer states, never from words found in it. The trap values
  (the old values of a knowledge update, the two sides of a contradiction) are labelled by hand, frozen and pinned by
  SHA-256 before any arm runs.
- **Storage outcome**, for a knowledge update or a contradiction read from a ColonyMind store: from the store's own
  links (the claims the evidence messages yielded, whether one was replaced or withdrawn, whether the night decided
  that two cannot both stand), the first of `superseded`, `conflict` (both kept, and the owner asked which holds),
  `both_current` (two different values, each from its own evidence message, both held as current), `single_value`
  (one value, however often it was stated) and `not_stored`.
- **Statistics**: a Wilson interval per arm and per question type; paired tests over the same questions in two arms
  at one budget (the exact sign test, and a seeded bootstrap interval resampling whole conversations, since BEAM's
  questions share one store per conversation). A test with a question unscored in either arm is withheld, never run on
  part of the questions. The primary contrasts are `all-on` against stock Hermes' session search, `all-on` against
  `import-only` on knowledge updates and temporal questions, and the traps against stock Hermes.

## Reproducing a run

From the repository root, with the sidecar installed (details in
[the pipeline README](../benchmarks/public_memory/README.md)):

```sh
python -m benchmarks.public_memory.datasets fetch
python benchmarks/public_memory/runner.py replay --dataset longmemeval_s --subset subset.json \
    --llm-config model.json --endpoints endpoints.json --out stores
python benchmarks/public_memory/runner.py read --dataset longmemeval_s --subset subset.json --stores stores \
    --arm all-on --capture capture.jsonl --work work
python benchmarks/public_memory/baselines.py --dataset longmemeval_s --subset subset.json \
    --arm session-search --work work --capture baselines.jsonl
python benchmarks/public_memory/storage.py --dataset longmemeval_s --stores stores --output storage.jsonl
python benchmarks/public_memory/reader.py --dataset longmemeval_s --capture capture.jsonl \
    --capture baselines.jsonl --answers answers.jsonl
python benchmarks/public_memory/judge.py judge --dataset longmemeval_s --answers answers.jsonl \
    --verdicts verdicts.jsonl --labels labels.json --labels-sha256 HASH
python benchmarks/public_memory/export.py results --declaration publication.json --answers answers.jsonl \
    --verdicts verdicts.jsonl --storage storage.jsonl --agreement agreement.json --labels labels.json \
    --private-terms private-terms.txt --output benchmarks/public_memory/results/longmemeval_s.json
python scripts/render_evidence_chart.py benchmarks/public_memory/results/longmemeval_s.json
python benchmarks/public_memory/export.py page
```

A results file and its chart are published by committing them in `benchmarks/public_memory/results/`; the page then
renders their section.

The exporter publishes public datasets only and only the fields it allowlists: the reader's answers, the judge's
replies and anything a server reported are never published, and models are named by the publisher's declaration
(public name and pinned weights revision).

## Results

<!-- generated: results -->
No run has been published yet. Until one is, every memory benchmark result is not demonstrated: nothing on this page is a measurement.
<!-- end generated: results -->
