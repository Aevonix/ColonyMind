# KU20: paraphrased knowledge updates

KU20 measures whether a change stated in new words reaches the record it changes. It was registered with
extraction version 18, which offers the extractor prior records found by meaning as well as by keyword
([SOURCE-CLAIMS.md](../../docs/SOURCE-CLAIMS.md)), and before any run of it.

## Why

The extractor can link a correction or change only to a prior record it is offered. The keyword lane restricts
those to the claims of the turns a keyword search of the new message finds. For a person with many claims nearly
every message finds something, so "We relocated to Munich last week, and I work there now" after "My office is in
Berlin" is never offered the office record, and becomes an independent assertion under a new key.

## The set

`ku20.json`, frozen by its SHA-256 in `run.py` (`SET_SHA256`); the runner refuses any other bytes. One person,
one time zone:

- 60 filler messages: that person's other facts.
- 20 change pairs: a first message stating a property, and a second stating a change of it. The second shares no
  searchable word (longer than two characters, not a stop word of the keyword search) and no word root with its
  first, carries a change cue the validator recognizes, and shares a word with at least one filler, so a keyword
  search of it finds other records and never the first.
- 10 controls, built the same way, whose second is a change of a different, nearby property (a relative's, a
  colleague's, another object's): it changes nothing in the store.

Each item's `gold` fields (property and values) are for scoring and the offline tests only; they never reach
extraction. The sizes are choices made when the set was built, not measured bounds.

## Procedure

1. A new empty state directory. The person's memory is formed for real: every filler and every first message is
   recorded in time order, extracted and reviewed by the model under test, and projected into the source vector
   index as the sidecar projects it.
2. The store must hold at least 50 other live claims of the person for every item; the run reports the smallest
   count and is invalid below it.
3. Each item's target is found: the live claim of its first message that holds the value the item is about (its
   gold first value). A first message can state more than one property ("I train at the Riverside gym on
   Tuesdays" is a gym and a day), and only the claim of the property the item changes counts. A claim holds the
   value when their normalized words are equal, or else when one contains the other as whole words in order; an
   exact match wins over a containing one (a rule chosen when the runner was built, before any run). A change
   pair whose first message has no such claim, or more than one, is not scored and is listed with the reason.
4. Each second message goes through the claim dry run (`SourceClaimProjection.rehearse`, the route
   `POST /v1/host/claims/dry-run`), against the same formed store: nothing one item would write reaches the next.
5. A build without the dry run (version 17 had neither it nor the vector lane) runs with `--path commit`: each
   second message is committed for real into its own copy of the formed store, and the offered records are read
   from the extraction call. `--path commit` on a later build measures it without its vector lane.

## Outcomes

For the scored change pairs: the share whose target is **offered** to the extractor, the share whose target the
change would **supersede** (retract or supersede), and the number that would retire any other claim, the first
message's other claims included (a wrong target). The change pairs not scored, with the reason. For the
controls: the share that would retire any claim (**over-linking**). Over all items: the review rejection rate, any
preference change held for the owner, items that did not complete, and the calls and tokens used. Builds are
compared with each other on the same model and embedder; no threshold is set here.

## Running

Only on endpoints no live instance uses:

```
COLONYMIND_BENCH_CHAT_BASE_URL=... COLONYMIND_BENCH_CHAT_MODEL=... \
COLONYMIND_BENCH_EMBED_BASE_URL=... COLONYMIND_BENCH_EMBED_MODEL=... COLONYMIND_BENCH_EMBED_DIMS=... \
python benchmarks/knowledge_update/run.py --state-dir /path/to/new-empty-dir --out ku20-results.json
```

`COLONYMIND_BENCH_CHAT_API_KEY`, `COLONYMIND_BENCH_EMBED_API_KEY` and `COLONYMIND_BENCH_EMBED_QUERY_INSTRUCTION` are
optional; `--path commit` needs no embedder. Run it with the interpreter of the build under test. The offline
tests (`sidecar/tests/test_knowledge_update_set.py`) check the set's rules and run the whole procedure with a
scripted extractor and a controlled embedding: every change is offered and superseded with the vector lane, and
none without it.
