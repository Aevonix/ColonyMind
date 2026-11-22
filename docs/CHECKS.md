# Checks: the world as scheduled jobs

Mail, repositories, projects, a calendar: each source the agent watches is one job of the Hermes
scheduler. A small script runs before the model and decides whether the model wakes at all, so a
quiet check costs no model turn. What a check finds becomes a card on the agent's kanban board or
one message to the owner. A new check is one small file plus one job.

There is no separate list of checks and no second scheduler. The scheduler's job list is the list;
the files live in [`checks/`](../checks/).

```
scheduler fires the job
  -> the job's script looks at what is new since its last look
       nothing new        -> the run ends: no model turn, nothing delivered
       something new      -> the model wakes with a count, reads what was found, and
                               needs work          -> one card on the board
                               owner should know   -> one message to the owner
                               neither             -> silent
```

Nothing is reported because a timer fired.

## The two forms

| | the script | the model |
|---|---|---|
| **The script reads the source** (`check_mail.py`, `check_github.py`) | fetches what is new since its marker and counts it | wakes only when the count is not zero |
| **No connector** (`check_projects.py`, `check_calendar.py`) | wakes the model and names a notes file | looks with its own tools, compares with its notes of the last state, writes the notes again, reports only what changed |

Both forms use the scheduler's own wake gate: a job's script runs first, and when its last output line
is `{"wakeAgent": false}` the run ends without a model call. Otherwise the script's output is put in
front of the job's prompt.

The script's output is one line: `{"wakeAgent": true, "check": "<name>", "new": <count>, "read":
"<command>"}`, or with `"notes": "<file>"` for a check without a connector. It never holds anything
read from the source: no sender, subject, title or text. The woken model runs the `read` command to
get what was found, so that content reaches it as a tool result and never as part of its prompt.

## The shipped checks

| job | script | what is new | settings |
|---|---|---|---|
| `check-mail` | `check_mail.py` | mail accepted by the [agent mailbox](agent-mailbox.md) since the last look, through `GET /v1/agent-mail/new` of the local engine; at most 20 messages a run, the rest on the next | `engine_url` |
| `check-github` | `check_github.py` | events of each listed repository newer than the last one seen, read with GET requests only | `github.repositories`, `github.token_env`, `github.api_url` |
| `check-projects` | `check_projects.py` | whatever changed since the model's notes | the "Where to look" line of the job's prompt |
| `check-calendar` | `check_calendar.py` | whatever changed since the model's notes | the "Where to look" line of the job's prompt |

The first look of a script that reads a source only notes where the source stands: mail already in
the mailbox and events already listed are not reported.

## Settings

`checks.json` beside the scripts, read on every run. [`checks.example.json`](../checks/checks.example.json)
shows every entry.

| entry | default | meaning |
|---|---|---|
| `engine_url` | `COLONYMIND_URL`, else `http://127.0.0.1:7777` | the local engine: the mailbox route and the recorder |
| `github.repositories` | none | the repositories to watch, as `owner/name` |
| `github.token_env` | `COLONYMIND_CHECK_GITHUB_TOKEN` | the name of the environment variable that holds a read-only token; unset means anonymous requests |
| `github.api_url` | `https://api.github.com` | the API to read |

The engine's key is found the way the engine's other local callers find it: `COLONYMIND_API_KEY`, else
the first line of `api.key` in the instance home (`COLONYMIND_HOME`, else `~/.colonymind`). No check
stores a credential.

## Setting up

1. Copy the `.py` files of `checks/` into the scheduler's scripts folder (`scripts/` in the Hermes
   home). A job's script must sit inside that folder.
2. Put `checks.json` beside them when a default does not fit, and list the repositories to watch.
3. For `check-github`, set a read-only token in the environment of the process that runs the
   scheduler, under the name `github.token_env` gives. The scheduler removes some names from a
   script's environment, `GITHUB_TOKEN` and `GH_TOKEN` among them, so the token needs a name of its
   own.
4. Create the jobs from [`jobs.yaml`](../checks/jobs.yaml) with the scheduling tool
   (`cronjob_manage`), each with its `name`, `schedule`, `script` and `prompt`. For a check without a
   connector, first put the places to look into the prompt's "Where to look" line.
5. Create them in the owner's chat and leave the delivery target out. A job delivers to the chat it
   was created from, so that is where the owner gets a finding, and the job's runs carry the standing
   of who created it ([whose job it is](#whose-job-it-is)).

A woken run needs the terminal (to run the `read` command), the file tools (for a notes file) and
the board. They come from the job's `enabled_toolsets`, else from the scheduler's toolset setting.

## What the agent changes itself

The schedule and the prompt of a check, pausing and resuming it, dropping it, the repositories in
`checks.json`, and new checks of either form: [Adding, changing or dropping a
check](../checks/ADDING-A-CHECK.md) is the page to point the agent at. A change the agent makes on
its own reaches the owner in the daily digest, as the result of the task it was done in.

An operator sets what the agent does not hold: the token in the scheduler's environment and the
toolsets of scheduled runs.

## Marker, failure and the recorder

A script keeps `check-state/<name>.json` beside the scripts: the marker of its last look and the
marker before its last waking run. The marker moves when the script reports, so one finding wakes
the model once. `read` prints what the last waking run found, fetched again from the source.

A look that fails (the source does not answer, answers with an error, or the settings cannot be
read) leaves the marker where it was, wakes nobody and is recorded as failed; the next run looks
again. When one repository cannot be read, the others are still reported.

Every run writes one line to the recorder (`POST /v1/recorder/once`, sent with a 0.3 second timeout
after the script's own output; a recorder that is down changes nothing):

| run | `source` | `result` | `choice` | `chooser` | `detail` |
|---|---|---|---|---|---|
| nothing new | `check.<name>` | `silent` | `ignore` | `rule` | |
| the model was woken | `check.<name>` | `pending` | `act` | `rule` | |
| the look failed | `check.<name>` | `failed` | | | the error class, in lowercase |

A line holds these labels and the run's duration, never anything read from a source.

## Whose job it is

The engine's guard gives a scheduled run the standing of whoever created or last changed its job
([the mind](MIND.md), [the Hermes adapter](HERMES-ADAPTER.md)):

- created with the scheduling tool in the owner's chat: the owner's job; its runs are not gated;
- created or changed by the agent in its own work: the run counts as the agent's own work, so its
  effects are classified, it writes no file, and a command waits unless the owner approved the task;
- no recorded creator (a job made with the scheduler's command line): every tool call of the woken
  run is refused.

The script itself is run by the scheduler and is not a tool call, so the gate works the same in all
three cases.

## Python

The scripts use the standard library only and need Python 3.9 or newer. The scheduler runs a job's
`.py` script with the job's `interpreter` when one is set, else with the Python that runs the
scheduler itself (on a managed install, the Python of its dependency environment). A shebang line
is not read.
