# Changelog

## v1.1.0 - pages, checks, the recorder and a wider recall

The agent now keeps a page on each person and on itself, watches sources as scheduled checks, and records what it
decides and does in one place that tells the owner when something stops working. A turn recalls more of what it
knows, and a turn in a group chat recalls only that chat's own history. Ideas for the owner can start as tasks on
their own, and a task reported done settles by its check.

### What changes

- Pages. The agent keeps a page on each person and one on itself: short lines, each citing the turns it rests on.
  The nightly pass writes a page again where there was new conversation, and a line that cites no turn it was shown
  is not written. The agent sets its own boundaries with a person (`colonymind_people` `boundary`) and its own likes
  and dislikes (`colonymind_self` `like`, `dislike`) with its tools. A turn with a person carries the page on that
  person and the lines of the agent's own page that fit the message. The owner reads every page
  (`GET /v1/mind/pages/{subject}`, and in the view of a person); erasing a source reaches the lines that cite it, and
  a merge of two contact records moves the dropped record's page to the kept one.
  [SOCIAL-STATE.md](docs/SOCIAL-STATE.md#pages).
- A wider recall. The default context budget (`context.budget_chars`) is 16,000 characters, up from 9,000. A turn's
  recall takes what that budget, or a smaller host allowance, has past the 6,000 characters the other sections keep,
  and never less than 4,000. Once the other sections are assembled, the turn's recall is packed again within the room
  they really leave, and the wider packet is taken only when it keeps every record the first one shows. A packet
  holds up to `context.recall_items` records; a missed rerank still falls back to five. A widened turn carries more
  memory in its message, which the host sends again with every later request of the session.
- Group chats. A turn in a group chat recalls only what was said in that chat, and is told so in one line. Its
  memory search and its page reads are narrowed the same way, for every sender, the owner included. In a group turn
  `colonymind_people` answers only who one named person is; opinions, `why` and `rate` of `colonymind_self`, the
  memory provider's `colonymind_resolve_commitment` and `session_search` are refused, with a line saying the owner
  can ask in a direct chat. A direct chat is unchanged.
- The agent's own work. An idea for the owner is decided by the mind's authority like any other work: where the
  autonomy level lets owner work act, it starts as a task on the board by itself; otherwise it stays a card that the
  owner's Build starts. The success check now decides how work reported done settles: a report that passes its check
  settles done, one that does not settles failed (`check_not_passed`, outside the breaker count). A task whose check
  reads its report is told the line the report must carry, unless its own text already shows that form. A failed
  task stopped at its runtime limit carries the label `elapsed`, and what a task came to is kept in memory as an
  entry of that task.
- Contradictions settle without a question. When the nightly pass reads a newer statement as replacing an older one,
  it keeps the newer one and retracts the older itself (nothing is deleted); it no longer asks the owner which to
  keep. The digest has no question part any more.
- The recorder. One line for each card, check run and error, with code labels and timings only, never message text
  or names, in `recorder.db` in the state directory. The owner gets one notice when a source goes quiet, a check keeps
  failing or a line never finishes, and the daily digest carries one line of counts. The optional `recorder` section
  of `colonymind.yaml` sets the thresholds and how long lines are kept. [RECORDER.md](docs/RECORDER.md).
- Checks. The repository's `checks/` folder holds sources watched as scheduled jobs of the Hermes scheduler: the
  agent mailbox, repositories, and two forms for a source without a connector. A small script decides whether the
  model wakes, so a quiet check costs no model turn. The mailbox gains `GET /v1/agent-mail/new`.
  [CHECKS.md](docs/CHECKS.md) says how to set them up.
- `colonymind recall-questions` scores recall against a question set built from the instance's own history and
  prints one result line. The set stays private in the state directory; `--new` builds it again.

### Fixes

- A turn could be answered without its memory section, with no warning, when an unrelated write to the memory
  store landed while recall read it.
- A task's report could take the place of its verified outcome in a turn's recall when the packet had room for all
  but one record. The outcome now comes first.
- A group chat turn that names no sender was filed under the owner, so a member's words could come back as the
  owner's own. Such a turn is no longer captured.
- In a group chat, a guest's turn could raise a question nobody there can answer and hold the chat until it timed
  out. That call is refused, and the model is told to ask in its reply.
- A mind run's call through the deferred-tool wrapper (`tool_call`) became an ask the owner could not read. A
  wrapper whose entries are reads passes; for any other entry the worker is told to call the tool by its own name,
  so the call is judged, and held, as that tool. Such an ask still waiting is withdrawn at the next tick.
- A tool call whose arguments hold a URL with a port that is not a number in range was refused in a hand ("authority
  check unavailable") and logged a warning in an owner turn. Such a port now reads as no port: the call is judged
  like any other, and a publish target configured without a port, or an owner-account host, on that host is still
  recognized.
- The engine logged an error with a traceback each time the Hermes adapter hung up while it sent an activity
  snapshot. That hang-up is answered quietly now; the adapter sends the snapshot again later, as before.
- The repository's first-loop check could fail when it restarted the gateway while a task's worker started its own
  profile's gateway: the new gateway exited 75 and the check waited for it. It now starts the gateway again after
  an exit 75, as Hermes' service units do, and fails at once on any other exit.
- The mind's background tick listed commitments on the engine's event loop and checked each row's source over a
  connection of its own, which could hold up requests on a large instance. A listing now checks its sources over
  one connection and runs in a worker thread, and learned ranking and reply timing read their sources once a pass.

### Upgrading

    uv tool upgrade colonymind
    colonymind upgrade
    hermes gateway restart

- No configuration to change and no migration by hand: the recorder's store and the tables for pages and group
  chats are created on first use.
- An instance that sets `context.budget_chars` in `colonymind.yaml` keeps its own value, and the recall that value
  leaves room for.
- A group chat's sessions are noted as its turns arrive, from this release on. A group turn recalls its current
  session and the noted ones, so an earlier session of that chat that no turn has named since is not recalled there.
- A question about two contradicting statements that an earlier release raised comes to an end without the owner:
  one not sent yet is withdrawn, one sent closes as moot, and the night settles each pair itself. A late reply to
  one is an ordinary message. What the owner answered before stays as it is.
- Checks are not set up by any command: copy the files of `checks/` into the Hermes home's `scripts/` folder and
  create the jobs, as [CHECKS.md](docs/CHECKS.md#setting-up) says.
- The hands plugin's `authority.py` changed: a host that runs the plugin from a copied directory copies that file.

## v1.0.1 - a turn a follow-up redirected is recorded once

A message that arrives while the model writes a reply can redirect that turn: Hermes cancels the reply and writes it
again with the new message included. ColonyMind now records such a turn once, with every message it answers.

### Fixes

- A redirected turn was not recorded at all: the hands front refused any text after the owner's message, and Hermes
  adds each follow-up after the line `User correction during the turn:`. The front now accepts the turn when every
  follow-up is a message it heard from the same sender in the same chat while the turn ran, and capture records the
  messages together, each after a blank line, as it does for merged queued messages. A follow-up the front did not
  hear, or one that carries Hermes' message-origin note, is still refused.
- The job watch's launchd template ran `watch --once` in the `Background` process class, which macOS throttles: a
  notice command that starts a heavier program could run several times slower, pass the default 30-second notice
  limit (`notice_timeout_s`) and wait an hour for its retry. The template no longer sets `ProcessType`, so the job
  runs in launchd's standard class.

### Upgrading

    uv tool upgrade colonymind
    colonymind upgrade
    hermes gateway restart

- Nothing else to do. The hands plugin's `front.py` changed: a host that runs the plugin from a copied directory
  copies that file. A job watch loaded from a copy of the launchd template keeps the old class: delete its
  `ProcessType` line, run `launchctl bootout "gui/$(id -u)/com.colonymind.watch"`, then load it again as in
  [INSTALL.md](docs/INSTALL.md#optional-job-watch-timer).

## v1.0.0 - the first release under the name ColonyMind

This is the first release under the name ColonyMind. ColonyMind is a mind for Hermes agents: persistent memory,
commitments, initiative and measured autonomy. It ships as two packages with one version: the sidecar
(`colonymind`) and the adapter that runs inside Hermes (`colonymind-hermes`).

### Upgrading

A new install has nothing to upgrade: this is the first release. To install, follow [INSTALL.md](docs/INSTALL.md);
[COMPATIBILITY.md](docs/COMPATIBILITY.md) names the Hermes and Python versions it supports.

The version numbers 1.9 to 1.13 that upgrade notes, documents and messages name are development versions from
before this first release. The steps that name them concern only an instance that one of those development
versions made.
