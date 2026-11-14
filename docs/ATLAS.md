# The atlas

`colonymind atlas build` writes a static HTML site of what people said, from the source ledger's
cited claims, so the owner can walk the same memory the agent recalls from. It is owner-only and
offline: it reads the instance's databases directly (read-only, so it works with the sidecar
stopped), writes one directory, and every page opens from disk with nothing fetched.

```text
colonymind atlas build [--out DIR] [--timezone ZONE] [--max-bytes N] [--max-lines N] [--page-lines N] [--json]
colonymind atlas check [--out DIR] [--json]
```

The default directory is `<instance>/atlas`; open its `index.html`.

The [owner console](CONSOLE.md) (`colonymind console`) shows the mind, the reports, the recall check and the
memory inventory as local pages behind a one-time link; the atlas stays a static site of its own.

## Pages

| Page | Holds |
| --- | --- |
| a person | only that person's own words: the claims from their person-scoped sources, grouped by fact, current value first, then its history |
| a month | everything said that month (by when it was said, in `--timezone`, default the agent's), newest first, by day |
| a topic | every claim about one subject, by whoever said it; a quoted episode is not a topic |
| a source | each cited message, with the quoted spans marked, and what was taken from it |
| index, People, Months, Topics | counts, the newest lines, and the directories |

Every line carries its exact quotation, the date it was said (or "recorded ... (time said unknown)"
when the source has no time of its own, never the build's date), and a link to the message it came
from. A changed value (`superseded`), an explicitly corrected one (`corrected`), one whose
validity ended (`no longer valid`) and one whose stated start is still ahead (`not yet valid`) stay
on the page, marked, and history links to what replaced it; a value whose correction was later
erased is `withdrawn`, never current again. An attributed correction on a message
([source annotations](SOURCE-ANNOTATIONS.md)) makes its lines `disputed`, never current (recall
never takes an annotated message as current either), and is shown, with its date, on the
message's source page; once the correction itself is erased the page says only that one was
on record. Names are
cross-linked by exact, whole-word, case-sensitive match: a unique person's name goes to their page,
a subject's spelling to its topic, and an ambiguous name is not linked.

## What is left out

The same eligibility recall uses, read in one snapshot: a source that was erased, a message removed
from its source, a derived claim whose grounding was erased, and a source whose attribution was
withdrawn are absent, as are the mind's own entries, session-scoped checkpoints and contacts the
owner deleted. A person's page never quotes anyone else, so a statement the owner made about a
contact is on the owner's page and on that subject's topic page, never on the contact's.

## Rebuilds, erasure and size

A rebuild renders the site in memory and writes only the pages whose bytes changed; an unchanged
ledger writes nothing. A page the new snapshot does not produce, such as an erased source's, is
deleted, and every other page quoting it is rewritten without it. The directory's
`atlas-manifest.json` lists every page the build may leave and is updated before any page is, and
every atlas-named page on disk that the new snapshot does not produce is deleted whether the
manifest lists it or not, so neither an interrupted build nor a stale or edited manifest leaves a
page a later build cannot find. One build runs at a time per directory; a second waits. Nothing
but atlas-named pages is ever deleted, and a directory that holds other files and no manifest is
refused. An erasure that lands
while a build runs is picked up before the command returns; if erasures keep landing through every
attempt, the build fails with a non-zero exit and marks the atlas stale, and `atlas check` fails
until a build completes. Between an erasure and the next build
the pages on disk still hold the erased text: `colonymind atlas check` fails until the atlas is
rebuilt.

The site is bounded by `--max-bytes` (default 256 MiB; every file it writes, pages and manifest
together) and `--max-lines` (default 50,000). Over the bound it keeps the newest lines by when they were said and says on the index how many older lines
it left out. `--page-lines` (default 400) splits a long page into numbered pages.

## Checking it

`colonymind atlas check` audits the directory from its files: every link resolves to a page and an
anchor in it; no page has a script, image, stylesheet link, frame, form, remote URL or absolute
path, and each carries a `default-src 'none'` content security policy with its one inline
stylesheet pinned by hash; every page matches the hash it was written with and is readable only by
its owner (directory `0700`, pages `0600`); and no source was erased or reattributed in the ledger
after the build. It exits non-zero on any problem.

`--json` on `build` prints the report: pages, lines, lines left out, people, months, topics,
sources, name and citation links, bytes, pages written, unchanged and deleted, and the time taken.
The report and the manifest hold counts and hashes, never content.
