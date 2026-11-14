# Public memory benchmark: the pipeline

Each step writes a file the next one reads, so every step can be re-run, resumed or re-scored offline. A resumed step
keeps a row already written only when it was written from the same inputs (the same capture or answer row, the
same question and prompts, the same frozen label); a changed input is refused before any call, never mixed into
one file. Every module's docstring has the details; this page is the order.

| Step | Command | Writes |
| --- | --- | --- |
| Form the stores | `runner.py replay` (with claims) and `runner.py import` (import-only) | one store per conversation |
| Read what the host injects | `runner.py read --arm <arm> --budget <chars>` | capture rows |
| How each store holds a changed value | `storage.py --stores <dir>` | one storage row per question and store |
| Baselines | `baselines.py --arm <arm>` | baseline rows |
| Label the traps | `labels.py draft`, review, `labels.py freeze` | frozen old values and their SHA-256 |
| Read | `reader.py --capture <file>` | answer rows |
| Judge | `judge.py judge --answers <file> --labels <file> --labels-sha256 <hash>` | verdict rows |
| Report | `judge.py report --verdicts <file>` | per-arm, per-budget tables |
| Calibrate the judge | `calibration.py sample`, label by hand, `calibration.py agreement` | agreement and kappa |
| Publish | `export.py results --declaration <file> --answers <file> --verdicts <file> --storage <file> --agreement <file> --labels <file>` | the results file: allowlisted rows, tables, paired tests, failures |
| Chart and page | `scripts/render_evidence_chart.py results/<dataset>.json`, then `export.py page` | the evidence chart beside the results; the page's generated blocks |

## What each arm is shown

- **ColonyMind arms** (`runner.py`, `ARMS`): what `POST /v1/host/context/assemble` returns for the question at the
  budget, the budget being both the host's inline allowance and recall's own ceiling. That is the packing a host's
  turn receives: recall with no count limit behind a verified reranker cutoff (five records otherwise), packed again
  within the allowance less the items kept whole beside it.
- **Baselines** (`baselines.py`): BM25 over whole sessions, the oracle (gold sessions only), full context, and stock
  Hermes' `session_search` called once with the question (as asked, and with its words joined by OR). Their items
  are packed in each arm's own order by the same packer, with no count limit, into one section that costs no more
  than the budget. Full context is read only where the reader's whole request fits its context: the history, the
  reader's own text, the question and the response format, with the answer's room reserved.
- The source recall comparison (`benchmarks/source_recall`) measures the same turn packet over its frozen corpus.

The reader (`reader.py`) is shown each arm's sections as the host renders them, with one fixed prompt: the
question, its date and the memory. It never selects or packs anything itself.

## Models

The reader and the judge are named by public name and pinned revision
(`COLONYMIND_BENCH_READER_MODEL`, `_REVISION`; `COLONYMIND_BENCH_JUDGE_MODEL`, `_REVISION`) and served at
`COLONYMIND_BENCH_READER_BASE_URL` and `COLONYMIND_BENCH_JUDGE_BASE_URL`, with keys in `COLONYMIND_BENCH_READER_API_KEY`
and `COLONYMIND_BENCH_JUDGE_API_KEY`. No address or key is written to any file. The judge is refused when it is the
reader's model.

## Judging

- LongMemEval: its own per-type answer-check prompts and its abstention prompt, copied from the MIT-licensed
  upstream (`judge_prompts.py`, with the licence notice); a reply is read as `yes` or `no` alone.
- BEAM: its own judge prompt once per rubric item, scored 0, 0.5 or 1 and averaged over the items as BEAM computes
  it; an answer counts as right when every item is fully met (this harness's threshold), and the mean is reported
  beside it. A reply the reader failed to structure is wrong and meets no item (0 on each); a skipped or undecided
  row has no score and is counted apart. BEAM's additional event-order measure is not computed.
- One call per item at temperature 0, or three votes for every item. A reply that is not the asked-for structure is
  recorded as refused and counts as no vote; nothing is asked again because it failed. Each decision (a verdict, a
  rubric item's score, whether a value is asserted or the answer asks) needs a majority of the votes for one value:
  two replies that disagree beside a refusal leave the item undecided, counted apart.
- Traps are structured judgments: for abstention questions, the reader's own `answerable` field and, when it claims
  an answer, whether the answer states a specific one (a specific answer given as answerable is the trap); for
  knowledge updates and contradictions, which of the frozen values (and the current answer) the answer states as the
  answer, and whether it asks which holds.

## How a store holds a changed value

For a knowledge update or a contradiction, `storage.py` reads how each store holds what the question's evidence
said, from the store's own links and never from text: the claims its evidence messages yielded (anchored or
witnessed), whether one was replaced or withdrawn (`superseded`), whether the night decided two live ones cannot both
stand (`conflict`: kept, and the owner asked), two different live values from different evidence messages with
nothing settling them (`both_current`), one value however often it was stated (`single_value`), or none
(`not_stored`). Each arm reads one store, so each row is joined to the arm that read it.

## Publishing

`export.py` is the only way a run's results leave the private files, and it publishes public datasets only (the
pinned LongMemEval-S and BEAM files). Every field is built from an allowlist, models are named only by the
publisher's declaration (public name, pinned revision, an optional public page), every arm is one of the harness's
own, and a private list of terms (never committed) refuses a results file that holds one: configured once
(`COLONYMIND_PRIVATE_TERMS`, else `~/.config/colonymind/private-terms`), it applies to every export, beside any
`--private-terms` file, and a configured list that cannot be read stops the export. The
reader's answers, the judge's replies and anything a server reported are never published. The benchmark page
(`docs/PUBLIC-MEMORY-BENCHMARK.md`) and the evidence chart are generated from the results file.

## Reach

ColonyMind arms: the gold turns' identifiers against the identifiers each shown item cites. Baselines: whether a
shown session is a gold session (and, beside it, which gold turns are shown whole). Scoring and statistics are in
`scoring.py`.
