# Fidelity suites

Three small, frozen suites that ask whether memory keeps what a person said faithful to that person: a value stays with
the subject it belongs to, within one message and across the messages of a conversation, and what the night writes
about a person stays in that person's language. Each runs the
build under test on its production path, repeats every question, and is graded by a judge that is never the model
under test.

Every question is graded twice, **trap first**. The trap is the failure the suite exists to find; an answer that falls
into it is never also counted correct. A judge answer that does not fit the schema it was given (every field present
and of its type, every claim and fact named exactly once), or an output production did not accept, leaves that
attempt unscored with its reason (`judge`, `no_output`, or `dry_run` for a dry run that did not complete): it is
counted apart, never as a pass, and never stops the run. The labels of an item (who holds which
value; the language) reach only the judge, after the model under test has answered.

## Borrowed value (`borrowed_value.json`)

Sixteen messages, each stating values for more than one subject: "My sister moved to Denver last spring, but I still
live in Boston." Each message goes through the claim dry run (`SourceClaimProjection.rehearse`, the route
`POST /v1/host/claims/dry-run`): extraction, every validation step and review, with nothing written.

- **Trap**: an accepted claim whose value, in the message, belongs to another subject than the claim's own (the
  sister's Denver stored as the speaker's city).
- **Correct**: no trap, and each fact the item expects is asserted for its own subject.

Two items are controls where both subjects hold the same value ("My sister and I both live in Leeds now"), so a
correct claim for the speaker is never read as borrowed.

## Borrowed value across messages (`borrowed_value_context.json`)

Sixteen short conversations of one speaker. The earlier messages name one or more subjects ("My sister moved last
month."), and the last message names its subject only by a pronoun or leaves it implied ("She lives in Denver now.").
Each conversation is its own person and session: its earlier messages are recorded for real and their claims formed
by the model under test on the production path (the claim worker), once per run; the last message then goes through
the claim dry run, which offers the conversation's earlier messages as the worker would, with nothing written.

- **Trap**: an accepted claim whose value, in the conversation, belongs to another subject than the claim's own (the
  brother's new job stored as the sister's), or that names one subject where the conversation leaves open which of
  several holds the value (two people either of whom "she" could be). A claim whose subject is the pronoun itself is
  no trap.
- **Correct**: no trap, and each fact the item expects is asserted for its own subject.

Four items are controls: two ambiguous antecedents and a pronoun with no antecedent expect no claim at all, and one
last message names its subject itself, so it needs no earlier message. The judge reads the whole conversation.

## Pronoun subjects in formed stores (`subject_bases.py`)

A secondary measurement, beside the suites: in stores the public memory benchmark formed (for example the dev slice's
single-session-user questions), every claim whose subject an earlier message named, with the words that named it,
its message and the conversation before it, and optionally a judge's verdict on each (the context suite's judge:
correct when the value is its own subject's; another subject's value, or an open one, is a trap). The judge reads the
conversation, the assistant's replies too, from the first earlier message the extraction is offered to the claim's
own message, so a message between them naming another subject is seen, never the naming message alone. The stores are
read, never written. Run it once per build to compare how many claims each forms and how many of the named ones are
right.

## Language (`language.json`)

One person per language (Spanish, German, French, Portuguese, Italian, Japanese, and English as the control), each
with three statements, a conversation of three turns and one long statement. In a new state per attempt, a real night
(a mind, the contact store and the ledger) writes the person's digest and the conversation's episode summary, and the
plain summary shortens the long statement the way an owner question quotes a statement that does not fit. Every
other call of the night is answered with nothing, so only those three outputs spend the model.

What is graded is what production accepted, never the model's raw answer: the digest the night stored in the
person's record, the episode summary it stored in the ledger, and the summary the plain summary returned. An answer
production parses (a fenced JSON answer, say) is graded as stored; an answer it refuses (a digest that cites none of
the person's claims, a summary that fails the no-cut check) reaches no one and leaves the attempt unscored, with the
row's `answered` count showing the model did answer.

- **Trap**: an output in another language than its source's (a digest in English about a person who wrote in
  Spanish is a quiet translation of their words).
- **Correct**: no trap, and the output says only what its source says, names kept as written.

The statements are committed as claims with the item's own labels (extraction is not what this suite measures). Each
claim's subject is a phrase of its statement ("Mi hija"): a claim about the speaker needs an English first-person
word ("I", "my", "mine") in its quotation to be grounded, so a statement in another language forms no claim about the
speaker at all.

## Run

```sh
export COLONYMIND_BENCH_CHAT_BASE_URL=... COLONYMIND_BENCH_CHAT_MODEL=...      # the model under test
export COLONYMIND_BENCH_JUDGE_BASE_URL=... COLONYMIND_BENCH_JUDGE_MODEL=...    # the judge, a different model
python benchmarks/fidelity/run.py --suite borrowed_value --state-dir /tmp/fidelity-bv --out bv.json
python benchmarks/fidelity/run.py --suite borrowed_value_context --state-dir /tmp/fidelity-bvc --out bvc.json
python benchmarks/fidelity/run.py --suite language --state-dir /tmp/fidelity-lang --out lang.json
python benchmarks/fidelity/subject_bases.py --stores <formed stores> --out named.json --judge   # the judge variables only
```

Run each with the interpreter of the build under test, so the same suite measures the build before a change and the
build after it.

`--repeats` sets the attempts per question (three by default, a builder's choice: a trap that appears one time in
three is seen). `COLONYMIND_BENCH_CHAT_TEMPERATURE` pins the model under test's temperature; the judge runs at
temperature 0, one call per output. API keys are optional (`COLONYMIND_BENCH_CHAT_API_KEY`,
`COLONYMIND_BENCH_JUDGE_API_KEY`) and never written to the results. Never point either endpoint at a model a live
instance uses.

The results hold every attempt (its accepted outputs, how many answers the model gave for each, the judge's grade
and the repeat it belongs to) and a summary: the traps
over every scored attempt with a Wilson interval, the correct rate, the questions that fell into a trap at least once,
and the unscored attempts. The language suite also summarises each output kind. A zero trap count over the repeats is
the gate proposed for pronoun subjects (both borrowed-value suites: the one-message suite guards what a change must
not break, the conversation suite exercises the change) and the source-language rule (language); whether it gates a
release is the owner's call.

The sets are frozen by their SHA-256 (`suites.py`); a changed set is a new pre-registration.
