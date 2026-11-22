# The recorder

One line for everything the agent decides and does: an inbound event, a card (a piece of the agent's own work, or
an ask to the owner), a run of a check, an error. A line says where it came from, what was chosen and by whom, what
came of it and how long it took. The lines show what is broken or not working, and they are the labelled examples a
later Instinct tier can learn from ([DECISIONS.md](DECISIONS.md)).

A line holds code labels and timings only. It never holds message text, a name, a phone number, an address, a chat
id, a handle, a subject line, a URL or the path of someone's content. A value that is not shaped like a label is
recorded as `other`.

The recorder (`colonymind.recorder`) is one module to call, one table, and one small module that reads the table
for the owner (`recorder/watch.py`). Recording never raises and never makes its caller wait: it is never the reason
a message, a turn, a task or a check fails.

## A line

| field | values |
|---|---|
| `id` | 32 hex characters, given by `begin` or `once` |
| `kind` | `event`, `card`, `check`, `error`, `alert` |
| `source` | a channel, a job or a module: `whatsapp`, `check.mail`, `mind.task` |
| `ref` | the key of a row in one of this instance's own stores: `ingress:17`, `initiative:42`, `ask:7` |
| `choice` | `ignore`, `remember`, `react`, `reply`, `act`, `tell_owner`, `ask_owner` |
| `chooser` | `rule`, `instinct`, `model`, `owner` |
| `result` | `sent`, `reacted`, `silent`, `failed`, `done`, `blocked`, `pending`; empty while the line is open |
| `detail` | one short code label: an error class, a rule name |
| `marks` | labels added later, separated by spaces: `owner_acted`, `owner_declined`, `owner_said_should_have`, `owner_said_no_need` |
| `began_at` | when the line began, in seconds since the epoch |
| `took_ms` | milliseconds from `begin` to `finish`, or what `once` was told |

`kind`, `choice`, `chooser` and `result` are closed sets (the tuples `KINDS`, `CHOICES`, `CHOOSERS` and `RESULTS`);
a value outside its set is recorded as `other`, and a new label is one entry in its tuple. `source`, `ref`, `detail`
and a mark are labels of one shape: a lowercase letter, then lowercase letters, digits and `_ . : -`, 64 characters
at most (a `detail` may hold capitals too, so an error class is kept as it is written).

Where a fact already lives in another store (the row of an inbound message, a decision in the decision log, an
entry of the event journal), the line carries its `ref` and copies nothing.

## Recording

In the engine's own process:

```python
from colonymind import recorder

line = recorder.begin("event", "whatsapp", ref="ingress:17")
recorder.finish(line, result="silent", choice="ignore", chooser="model")
recorder.once("check", "check.mail", result="done", choice="ignore", chooser="rule")
recorder.mark("initiative:42", "owner_acted")
recorder.recent(20, kind="card")
```

`begin` and `finish` bracket work that has a start and an end. `once` records something settled in one step. `mark`
adds a later label to every line that carries a `ref` and returns how many. `recent` reads lines back, newest first.

From another process (a gateway plugin, a check script) the four writes are routes of the engine's API:

| route | body | answer |
|---|---|---|
| `POST /v1/recorder/begin` | `{"kind", "source", "ref"?}` | `{"id": "<hex>"}` |
| `POST /v1/recorder/finish` | `{"id", "result", "choice"?, "chooser"?, "detail"?}` | `{"ok": true}` |
| `POST /v1/recorder/once` | `{"kind", "source", "result", "ref"?, "choice"?, "chooser"?, "detail"?, "took_ms"?}` | `{"id": "<hex>"}` |
| `POST /v1/recorder/mark` | `{"ref", "label"}` | `{"count": <n>}` |

They are served at the engine's own address and take the instance key (`Authorization: Bearer <key>`, `api.key`),
as the engine's other routes for the local plugins do. A caller sends with a short timeout (about 300 ms), in the
background where it can, and ignores every failure. Bad input is never an error status: a field that is missing or
is not a string counts as empty, a body that is not a JSON object counts as an empty one, and the answer is 200.

## The store

One table, `lines`, in `recorder.db` in the instance's state directory, created on first use (mode 600). The nightly
backup copies it with the instance's other stores.

```sql
CREATE TABLE lines (
    id       TEXT PRIMARY KEY,
    kind     TEXT NOT NULL,
    source   TEXT NOT NULL,
    ref      TEXT NOT NULL DEFAULT '',
    choice   TEXT NOT NULL DEFAULT '',
    chooser  TEXT NOT NULL DEFAULT '',
    result   TEXT NOT NULL DEFAULT '',
    detail   TEXT NOT NULL DEFAULT '',
    marks    TEXT NOT NULL DEFAULT '',
    began_at REAL NOT NULL,
    took_ms  INTEGER
);
```

Each call opens the file for itself and waits a quarter of a second at most for another writer. A line that cannot
be stored (the file is missing, locked or unwritable) is dropped, and the failure itself becomes an `error` line
(source `recorder.store`, the error class in `detail`) with the next write that works. Lines older than
`recorder.keep_days` are dropped once a day, on the mind's tick.

## Alerts

On each tick of the mind (also with the mind switched off) the recorder checks three conditions and tells the owner
when one starts to hold:

- **A source has gone quiet.** A source that recorded on `quiet_active_days` of the 7 days before its last line has
  recorded nothing for `quiet_hours`. The alert clears when the source records again, or after one more week.
- **A check keeps failing.** The last `failing_runs` runs of a check, within 7 days, all failed. The alert clears
  with the first run that does not fail.
- **A line was begun and never finished.** A line of a kind named in `open_minutes` has had no result for that many
  minutes. The alert is per kind and source, and clears when no line of that source went late for a day.

Each condition that starts to hold becomes one line of kind `alert` (source `recorder.quiet`, `recorder.failing` or
`recorder.unfinished`, the subject in `detail`) and one notice to the owner. The line stays open while the condition
holds, so the owner is told once per condition until it clears; the line is closed when it clears, and a condition
that returns later is a new alert. The notice is a `health_notice` on the mind's outbox, the path the engine's other
health notices take to the owner ([MIND.md](MIND.md#outreach-to-the-owner)). It names the source and gives counts
and durations, nothing else. A notice that could not be queued closes its line as `failed`, and the next tick tries
again.

A new alert is one function in `recorder/watch.py` that yields what holds now, and one entry in `ALERTS`.

## The daily line

The daily digest ([MIND.md](MIND.md#the-daily-digest)) carries one line from the recorder for the time the digest
covers: the lines by kind with their choices and results, the failures by source, and the lines still open.

```text
Recorded: 8 events (3 ignore, 1 react, 3 reply; 2 sent, 1 reacted, 3 silent, 1 failed, 1 open), 1 card (1 ask
owner; 1 open), 3 checks (2 failed, 1 done). Failed: 3 (2 check.mail, 1 whatsapp). Still open: 1 event, 2 cards.
```

A day with recorded lines has a digest, with this line in it. An open `alert` in the line is a condition that
still holds.

## The owner's corrections

When the owner says a decision was wrong, the decision's line gets a mark: `owner_said_should_have` ("you should
have answered that") or `owner_said_no_need` ("no need to answer that"). It is stored with
`recorder.mark(ref, label)`, or `POST /v1/recorder/mark` from another process, against the `ref` of the decision.
The same call records whether the owner acted on what the agent started: `owner_acted`, `owner_declined`.

## Configuration

```yaml
recorder:
  quiet_hours: 36          # a source is quiet when its last line is this old
  quiet_active_days: 7     # and it recorded on this many of the 7 days before that line
  failing_runs: 3          # a check keeps failing after this many failed runs in a row
  open_minutes:            # a line of this kind with no result after this many minutes is unfinished
    event: 30
    check: 60
    card: 10080
  keep_days: 180           # older lines are dropped
```

The section is optional and is read again on every tick, so a change needs no restart. A value that is not a
positive number keeps its default. `open_minutes` names the kinds that are watched: a kind set to 0 is not watched,
and another kind is watched once it has an entry.

## Lines as labelled examples

Every settled `event` or `card` line is one example. The input is the row its `ref` names, read from that row's own
store by whoever trains; the recorder holds none of it. The label is the `choice`, with who made it (`chooser`) and
what came of it (`result`). The marks say what the owner later made of it:

```sql
SELECT ref, kind, source, chooser, choice, result,
       (' ' || marks || ' ') LIKE '% owner_said_should_have %' AS should_have,
       (' ' || marks || ' ') LIKE '% owner_said_no_need %'     AS no_need,
       (' ' || marks || ' ') LIKE '% owner_acted %'            AS owner_acted,
       (' ' || marks || ' ') LIKE '% owner_declined %'         AS owner_declined
FROM lines
WHERE kind IN ('event', 'card') AND result != '' AND ref != ''
ORDER BY began_at;
```

`should_have` on a line whose choice was `ignore` says the right label was to answer; `no_need` on a `reply` or a
`react` says it was to stay silent; `owner_acted` and `owner_declined` on a card say whether the owner took up what
the agent started. A line with no mark is a choice that stood.

## Limits

- With its defaults the quiet alert watches the sources that record every day; one that records less often is not
  watched. The thresholds are for all sources alike.
- A notice to the owner goes out as the mind's other notices do: it needs an owner contact, passes the owner gate
  and waits while the mind is off.
- The recorder has no page, no command and no metrics endpoint. `recent()` and the table are the way to read it.
