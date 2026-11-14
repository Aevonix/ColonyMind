# The owner console

`colonymind console` shows what the terminal commands show, as local web pages: the mind's state, concerns, goals,
log and asks, health, the reports, the recall check and the memory inventory. It is read-only, with one control:
the mind's off switch. It never answers an ask and never changes the autonomy level, a grant or an allow.

```text
colonymind console [--port N] [--bench DIR]
```

It listens on `127.0.0.1` only (there is no option for another address), on a free port unless `--port` names
one, and prints a link:

```text
Owner console for /path/to/instance: read-only, with one control, the mind's off switch.
Open this link in a browser on this machine. It works once and opens one session:
  http://127.0.0.1:40123/start?token=...
Ctrl-C stops the console.
```

The link works once: it opens a session in the browser that follows it, and every page needs that session. The
session's pages live under a random path of its own, where the link sends the browser; the page names below are
relative to it. To open the console in another browser, stop it and start it again for a new link.

`--bench DIR` names a paired benchmark results directory: the bench page reads it, and the faculties page puts its
evidence beside each faculty (as `colonymind report faculties --bench DIR` does).

## Pages

| Page | Shows | The terminal's equivalent |
| --- | --- | --- |
| `/` | the mind on or off, its level, queues, last tick and breaker; health in one line; the off switch | `colonymind mind status` |
| `/concerns` | what is on the mind (the broadcast set starred) and the drives | `colonymind mind concerns` |
| `/goals` | the goals the mind adopted that are open | `colonymind mind goals` |
| `/log` | the audit log, newest first; each id opens its page on `/why` | `colonymind mind log` |
| `/why` | one intention: drive, evidence, decision, outcome and its history | `colonymind mind why <id>` |
| `/asks` | the open asks, read only | `colonymind mind asks` |
| `/health` | the sidecar's health: its status, its problems in words, its notes | `colonymind status` |
| `/faculties` | what each faculty did over a window (`since`, default 7 days) | `colonymind report faculties` |
| `/instinct` | what each decision point did per tier over a window | `colonymind report instinct` |
| `/usage` | model calls by task, role and endpoint, in counts only | `colonymind report usage` |
| `/bench` | the paired benchmark's results in plain words (with `--bench`) | `colonymind report bench DIR` |
| `/recall` | what memory recalls for a question, and why each candidate is in the packet or not | `colonymind recall --explain` |
| `/memory` | what memory holds about a person: each claim's state, its exact words and its source | `colonymind memory inventory` |
| `/memory/history` | one claim's own chain, first statement first | `colonymind memory history` |

The mind, health, recall and memory pages ask the running sidecar with the instance key, the owner's credential,
as the commands do; with the sidecar down they say so. The reports read the instance's stores read-only, so they
work with the sidecar stopped.

A claim's state is `current`, `in conflict`, `upcoming` (its stated start is still ahead), `expired`, `superseded`
or `retracted` ([the memory inventory](SOURCE-ANNOTATIONS.md#the-memory-inventory)). A history shows what each
correction was recorded against, and says when the chain may go on: an erasure took a link with it, or the walk
reached its bound. Corrections still go through annotate and removals through forget: the console changes no
memory.

The recall check runs the search `colonymind recall --explain` runs, with no session: person-scoped evidence only,
and nothing supplied to any turn. It may call the embedder and the reranker, as that command does. Its question is
posted from the page's form, so it never sits in an address or the browser's history (a question in the address is
not read), and a value the capture detector would seal is withheld from it before the search, so neither the sidecar
nor an embedder or reranker behind it receives the value; the page says when it did.

## The off switch

The overview's button turns the mind off as `colonymind mind off` does ([the off switch](MIND.md#the-off-switch)):
no further effects, at once, and unsent messages are cancelled. The record names the console. When the sidecar
does not take it (it is stopped, or does not answer), the console writes the off marker instead, and the mind is off
from its next start. A running mind reads the marker only when it starts, so the console never calls the mind off on
the marker's word: while the sidecar does not answer, the overview says the marker is set and that a running mind
has not confirmed it stopped, and keeps the switch, whose next press asks the sidecar again (the marker is left as
it is). A mind the sidecar reports off is left as it is, so a press then changes nothing. The console cannot turn
the mind back on: `colonymind mind on` does, in a terminal.

## Who can open it

A loopback port is reachable by every program and user on the machine, so the link's token is the lock. Each
request is checked in this order, and the first failure ends it:

1. **The address.** The `Host` header must name `127.0.0.1:<port>`. A page elsewhere that points a name of its own
   at this machine (DNS rebinding) is refused before anything else, and a refused request spends nothing.
2. **The session.** The printed link's one-time token opens one session: its pages are served under a random
   path of their own, and the browser receives a random cookie scoped to that path (`Path=/<session path>/`,
   `HttpOnly`, `SameSite=Strict`). Every page requires both. Browsers do not keep cookies apart by port, so a
   cookie for all of `127.0.0.1` would also go to any other local service the browser opens; scoped to a path only
   the session knows, it goes nowhere else. A second use of the link opens nothing. Tokens are compared in constant
   time, and no request line is logged.
3. **A posted form** (the off switch, and the recall check's question) also requires a `POST` from the console's
   own page: `Origin` exactly `http://127.0.0.1:<port>` (and `Sec-Fetch-Site: same-origin` when the browser sends
   it), a form body, and the session's form token, which only the console's own page holds. Every other page is
   read with `GET` only.

Every response, refusals included, carries a content security policy that loads nothing but the page's own pinned
stylesheet and runs no script (`default-src 'none'`, forms only to the console itself, no framing), with
`nosniff`, `no-store` (a page is never cached) and a same-origin referrer policy. No page holds a script, an image,
a stylesheet link or a frame.

On another machine, forward the port with the same number on both ends (the `Host` header must still read
`127.0.0.1:<port>`): `ssh -L 8765:127.0.0.1:8765 <host>`, then `colonymind console --port 8765` there.

## Private values

A value the capture detector would seal ([the sealed store](SECRETS.md)) is withheld on every page, wherever it
appears, form fields included, and from the recall check's question before its search. A reference to a sealed value is shown as the stored words hold it: it names the value and never holds it.
A claim is also read as its card shows it, its predicate beside its value: a claim whose subject or predicate names a
credential however it is spelled (`wifi password`, `wifi_password`, `access_token`) has its whole value withheld
though neither field alone looks like one, a weak password included, and the value is withheld wherever the memory
page or the claim's history holds it, its quotation and corrections included. The value is read whole, however many
words it has, so a descriptive value under such a predicate is withheld too. The terminal's
`colonymind memory inventory` and `history` read claims the same way.
The console never calls the sealed store's routes. The instance key never appears on a page.

## What it never does

- Answer an ask, or change the autonomy level, a grant or an allow: the asks page shows the open asks, and the
  owner answers them in their own chat.
- Turn the mind on, or change any setting.
- Write anything itself, except the off marker when the sidecar does not take the off switch. (The sidecar counts
  the recall check's model calls in its usage, as it does for any search.)
- Fetch anything from outside the machine, or send anything but its own pages.

Its sidecar calls are fixed in its code (`colonymind.console.source.ROUTES`): the reads above, the recall search and
the off switch. It refuses to call any other route.
