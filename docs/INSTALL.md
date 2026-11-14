# Install and update

ColonyMind runs beside stock Hermes: the sidecar and its CLI live in their own
Python environment, one adapter package goes into the Python that runs
`hermes`, and one directory holds everything the instance owns. No patched
Hermes, no prepared runtime, no keyring.

Hermes 0.21.3 is the oldest supported host for install, the first loop and the
container, qualified on Python 3.11 to 3.13. Session archives need Hermes 0.21.7
or newer. From 0.21.7 on, a complete stock Hermes install needs Python 3.14:
Hermes and its adapter live in that environment, while the sidecar lives in its
own Python 3.11 to 3.13 environment (3.12 recommended). See
[compatibility](COMPATIBILITY.md) for the installer range and test qualifications.
The installer accepts `>=0.21.3,<0.24` from package metadata, or from Hermes'
`install-stamp.json` `baseVersion` when metadata is `0.0.0`. For PM-managed
environments it reads the owning checkout recorded in `inputs/.project-root`,
not the runtime workspace snapshot. `init`, `upgrade` and `doctor` share this
read; a missing, unreadable or unversioned stamp is refused.
You also need `hermes` on `PATH` (pipx, uv or a venv all work) and one
OpenAI-compatible chat endpoint, which Hermes already has.

## Install

ColonyMind needs a supported Hermes first. Three ways to one are checked.

**Hermes' installer as it comes (Hermes 0.23.1 today).** Put `uv` on `PATH` as
well: the environment this installer builds has no pip, and `colonymind init`
installs the adapter there with `uv`.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

The hosted checks set Hermes up this way on every change, pinned with
`--branch v0.23.1`, then run the install checks (install, upgrade twice, doctor,
uninstall), recall across two sessions, the first closed loop and the demo
against that one Hermes home. The same four steps were run in a container for
this release, with Hermes on Python 3.14.9 and the sidecar on Python 3.12.16.
All four passed, the first closed loop with 28 of its 28 checks.

**Hermes 0.22.7 with Hermes' installer**, pinned with `--branch v0.22.7`, and
`uv` on `PATH` as above:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --branch v0.22.7
```

This way is not part of the hosted checks. It was run once in a container for
this release, in the same way and with the same result: the install checks,
recall across two sessions, the first closed loop (28 of its 28 checks) and the
demo passed.

**Hermes 0.21.3 from source with pip.** The hosted checks install Hermes this way
on every change, then run the same four steps against it:

```bash
git clone https://github.com/NousResearch/hermes-agent
git -C hermes-agent checkout 345cd2b057a452236de401d3534b8502a7465e8d  # Hermes 0.21.3
python3 -m venv hermes-venv
hermes-venv/bin/python -m pip install -e ./hermes-agent
export PATH="$PWD/hermes-venv/bin:$PATH"
```

Hermes 0.24 is outside the supported range. `colonymind init` refuses it before
anything is written, in these words:
`Hermes 0.24.0 from Hermes install-stamp.json baseVersion (hermes-agent metadata
0.0.0) in <python> is outside the supported range >=0.21.3,<0.24`.

Hermes' installer gives every Hermes home its own environment, so the adapter is
installed once per Hermes home: `colonymind init --hermes-home <home>` installs it
for the home it configures, and `colonymind doctor` names a home that is
configured without it. A dependency conflict among Hermes' own packages is a
warning in `colonymind doctor`; one that involves the adapter fails the check.
See [compatibility](COMPATIBILITY.md).

Then ColonyMind itself:

```bash
uv tool install --python 3.12 colonymind
colonymind init
hermes gateway restart
```

The base package includes the vector store (LanceDB), so these three commands
produce a sidecar with semantic recall once an embedding endpoint is
configured; `colonymind init` refuses to run in an environment that lacks it.
Only the local embedder, for a model endpoint with no embeddings route, needs
the `vectors` extra (`uv tool install --force --python 3.12 'colonymind[vectors]'`,
which brings PyTorch, several GB on Linux where its wheels carry the CUDA
libraries, and the transformers 4.51 or later its models need); a remote
OpenAI-compatible embeddings endpoint needs nothing more.

`uv tool install` gives the sidecar an environment of its own on the
interpreter it names, so a package manager's newer `python3` never decides it,
and `uv tool upgrade` keeps that interpreter. pipx works too:
`pipx install --python python3.12 colonymind`.

Or start from Hermes: `hermes memory setup`, then pick `colonymind-memory`. Its
setup runs the same `colonymind init` (installing the sidecar as a tool first
when it is missing, with your yes, and never into Hermes' environment) and ends
with the same restart.

`colonymind init` asks for your name and messaging handles, the agent's name,
values and boundaries, its time zone (by default this machine's), the quiet
hours, the platforms your own messages come from (by default this terminal,
`cli`, and the platforms of your handles, in Hermes' names) and the autonomy
level (`off`, `suggest`, `standard` or `trusted`; default `suggest`). Then it:

1. writes `colonymind.yaml`, `identity.yaml` and `api.key` (mode 600) to the
   instance directory (`$COLONYMIND_HOME`, default `~/.colonymind`). The quiet
   hours go to `mind.quiet_hours`, which the mind reads (`identity.yaml`
   keeps the same answer). Run again without a new answer, init keeps the
   quiet hours in force: ones you set in `mind.quiet_hours` since are kept
   and copied to `identity.yaml`, not overwritten by the earlier answer;
2. creates the owner contact, or keeps the one `owner.contact_id` already
   names. A recorded id that does not resolve to a live contact (a typo, a
   deleted row, no contacts store) is refused before anything is written;
   init never creates a second owner contact. Correct the id, or remove it
   to create a new owner contact;
3. points the router at the model endpoint Hermes uses: `model.base_url`,
   else the provider Hermes names (an entry under `providers:` or
   `custom_providers:`, or one of Hermes' own API-key providers, looked up
   the way Hermes does), else `OPENAI_BASE_URL` in its `.env`. When none
   resolves, or Hermes reaches it over a transport other than the
   OpenAI-compatible chat completions route the router calls (a native
   Messages endpoint, for one), it stops before writing anything and asks
   for `--model-url`.
   Recall: `--embed-url <root>` records that embedding endpoint (semantic
   recall is on only with one). `--embed-url auto`, or your yes in
   interactive init, asks the model endpoint's embeddings route (and its
   rerank route when you name a rerank model, `--rerank-model`) with one small
   request each, never a chat request, and records what answers (the model,
   its vector width, and the key it was asked with under `environment`; a
   route that moves to another endpoint takes that endpoint's key, or none,
   never the earlier one's). When no embeddings route answers, or you say no
   to asking, and the `vectors` extra is installed, init offers the local
   embedder (a non-interactive init with `--embed-url auto` takes it): a
   model from the tier table for this machine's memory, pinned to one
   revision and downloaded once into the instance directory, which the
   sidecar then runs itself (`router.embed_provider: local`).
   `--embed-url local` takes it at once, and `--embed-model` names another
   pinned one. The default is `Qwen/Qwen3-Embedding-0.6B` (1024 dimensions,
   a download of about 1.2 GB into `<instance>/models/embeddings/`, and about
   as much memory while the sidecar runs, twice that where the libraries load
   full precision); `Qwen/Qwen3-Embedding-4B` (about
   8 GB) and `Qwen/Qwen3-Embedding-8B` (about 15 GB) are taken only by name.
   A machine under 8 GB of memory is offered none, since every pinned model
   reads a whole memory chunk and the tier table's smaller one does not: init
   names the command that takes the default anyway (`--embed-url local
   --embed-model Qwen/Qwen3-Embedding-0.6B`). The models need transformers
   4.51 or later: where the sidecar's environment has an older one, init
   neither offers nor downloads a model and names the install that adds it,
   and doctor's `recall-mode` fails with the same remedy. The files stay in
   `<instance>/models` after a switch to an endpoint (init names them) and
   after `colonymind init --uninstall` (it keeps them with the instance data),
   until you delete them. Without any of these it asks nothing, and recall is
   keyword-only, as init and `colonymind doctor` say;
4. offers your own history: Hermes' direct sessions whose recorded origin is
   one of the handles you gave (that exact platform and sender, a direct
   chat, no automation) can be imported as quotations, with no claims, no
   learning and no model call, through the reviewed importer
   ([HERMES-HISTORY-IMPORT.md](HERMES-HISTORY-IMPORT.md)). Your yes, or
   `--import-history`, imports them; groups, other people and other handles
   stay in Hermes;
5. installs `colonymind-hermes==<same version>` into the Python of the `hermes`
   executable, runs `pip check` there and records both versions in
   `<instance>/hermes-install.json`;
6. writes these Hermes keys: `plugins.enabled += colonymind`,
   `memory.provider: colonymind-memory`, `plugins.hook_callback_timeout: 0`,
   `kanban.dispatch_in_gateway: true`, `skills.external_dirs += <instance>/skills`,
   `security.protected_instruction_extra_patterns += colonymind.yaml, identity.yaml, api.key`,
   and `plugins.colonymind.{sidecar_url, key_file}`. Selecting ColonyMind's
   memory provider replaces the one Hermes used, and init says which
   ("replaces memory provider X; 'colonymind init --uninstall' restores
   it"): the previous name and the backup of that config are recorded in
   `hermes-install.json`;
7. creates the `colonymind-act` worker profile: the main `model` with the
   `providers`, `custom_providers` and fallback entries it can name, each
   custom endpoint's `extra_body` carrying `mind.worker_request` (an output cap
   of 8,192 tokens and `top_p` 0.95 on every worker request; a model given as a
   bare `base_url` gets a `custom_providers` entry for it), the mind's
   toolsets as `platform_toolsets.cli` (what the dispatcher pins),
   `approvals.deny` from `mind.deny.commands` and `gateway.standalone: true`
   (Hermes 0.21.7+ multiplexes every named profile into the host gateway
   unless the profile opts out). With `mind.worker_model` set to the name of
   one `providers` (or `custom_providers`) entry of the main config, the
   profile's `model` is that entry (its default model, `base_url` and
   `context_length`) with no fallbacks, instead of the main model and its
   fallbacks; a name the main config lacks stops the upgrade. A rewrite keeps
   every key that only the profile on disk has (one an operator added by
   hand) and prints the dotted paths it kept, never their values; where both
   have a key, the generated value wins. A key later removed from the main
   config therefore stays in the profile until it is removed there by hand.
   `colonymind doctor` asks stock
   Hermes what a dispatched worker resolves and fails on a stale profile, an
   unresolvable model or any toolset beyond the mind's;
8. installs and starts the sidecar user service (systemd `--user` on Linux,
   launchd on macOS; skipped with a message elsewhere, then start it with
   `colonymind start`). The unit asks for 16,384 open files (the vector store
   holds a descriptor per data file; macOS starts a user process at 256), and
   the server raises its own soft limit to the same figure at startup where
   the hard limit allows, so `colonymind start` from a shell is covered too.
   `colonymind doctor` warns when the running sidecar has less. The sidecar
   writes its rotating log to `<instance>/service/sidecar.log`; launchd and
   systemd append the process's raw output (a native crash, a traceback from
   before logging starts) to a file of their own, `service/launchd.log` or
   `service/systemd.log`, and restart a crashed sidecar after 30 seconds. An
   install from an earlier release keeps its old unit (one shared log, a
   restart every 5 seconds) until `colonymind service stop`, `colonymind service
   install` and `colonymind service start` write the new one; `colonymind
   doctor` warns until then.

It writes no Hermes admin lists and never restarts a running gateway. Every
step is idempotent: run it again to change an answer, or pass the flags
(`colonymind init --help`). `--non-interactive` uses the flags and defaults.

Without a terminal (a container's entrypoint), every answer can come from the
environment instead: `COLONYMIND_INIT_OWNER_NAME`, `COLONYMIND_INIT_OWNER_HANDLES`
(comma separated `PLATFORM=ID`), `COLONYMIND_INIT_OWNER_PLATFORMS`,
`COLONYMIND_INIT_AGENT_NAME`, `COLONYMIND_INIT_AGENT_VALUES`,
`COLONYMIND_INIT_AGENT_BOUNDARIES`, `COLONYMIND_INIT_TIMEZONE`,
`COLONYMIND_INIT_QUIET_HOURS`, `COLONYMIND_INIT_AUTONOMY`,
`COLONYMIND_INIT_HERMES_PYTHON`, `COLONYMIND_INIT_HOST`, `COLONYMIND_INIT_PORT`,
`COLONYMIND_INIT_MODEL_URL`, `COLONYMIND_INIT_MODEL`, `COLONYMIND_INIT_EMBED_URL`
(`auto` included), `COLONYMIND_INIT_EMBED_MODEL`, `COLONYMIND_INIT_EMBED_DIMS`,
`COLONYMIND_INIT_RERANK_MODEL`, `COLONYMIND_INIT_ADAPTER_SOURCE` and
`COLONYMIND_INIT_IMPORT_HISTORY`. `COLONYMIND_INIT_NON_INTERACTIVE=1` runs
without prompts, `COLONYMIND_MODEL_API_KEY` carries the model endpoint's key
and `COLONYMIND_INIT_NO_SERVICE=1` skips the service. A flag wins over its
variable, and the same answers give the same instance either way.

Two keys deserve a word. `plugins.hook_callback_timeout: 0` makes Hermes run
plugin callbacks inline, with no timeout, so overlapping capture and guard
calls are never skipped. It is Hermes-wide: every plugin's callbacks run that
way, and init says so. The protected patterns match by basename in any
directory: writes to any `colonymind.yaml`, `identity.yaml` or `api.key` need a
human, even under yolo, and fail closed inside a worker.

Check the result:

```bash
colonymind doctor
```

It checks the Hermes version, the instance files (the instance's own secret,
`instinct.key`, included: `instance-secret` fails a file that holds no whole
key, or a link in its place whose target is gone, either of which keys nothing
for decisions and model usage), the adapter version, `pip
check`, the Hermes keys, the worker profile, the plugin registration, the
sidecar service, the sidecar and, when an embedding endpoint is configured,
that its embedder is serving. Two warnings say what recall can do, each with
its fix: `recall-mode` ("keyword recall only", with no embedding endpoint or
with `mind.faculties.semantic_recall` off) and `recall-cutoff` ("no calibrated
cutoff: recall never abstains", until a reranker's cutoff is calibrated, see
[RECALL-HYBRID.md](RECALL-HYBRID.md)). `recall-cutoff` passes only when the
cutoff is a finite number and its fingerprint matches the reranker
`colonymind.yaml` names, the checks the sidecar applies before it uses the
cutoff; it warns when either fails, and when the sidecar picks the reranker
itself, so the fingerprint cannot be checked. The `service` check passes when this instance's
user service is installed, running and written by this version. It warns when
no service is installed (a sidecar started by hand with `colonymind start
--detach` serves, but nothing restarts it after a crash or starts it at
login) or when the unit is an earlier release's, fails when the service is
installed but not running, and is skipped where there is no user manager.

`/v1/host/health` answers `ok` or `degraded`, and a `problems` list says in
words why it is not `ok`. Degraded means something the sidecar itself runs is
not working: the source ledger is unreadable, a configured embedder or its
vector store did not come up, the mind's tick has not run (ten of its
intervals, at least a quarter hour; `COLONYMIND_STALE_TICK_HOURS` pins it), or
capture jobs have waited over an hour without landing
(`COLONYMIND_STALE_CAPTURE_HOURS`). Quiet is not degradation: how long since the last
`turns/sync` or `context/assemble` is reported under `temporal.silence_hours`
and never changes the status, so a fresh install is `ok` before anyone has
talked to it. `colonymind service start` and `colonymind service status` report
`ready` as soon as the sidecar answers, with the served `health` and its
`problems` beside it; `colonymind doctor` warns on `degraded` and fails the
check that names the cause.

## Optional job-watch timer

Job watch runs separately from the sidecar, so it can report a frozen or stopped sidecar as well as stuck
jobs and broken endpoints. Neither init nor upgrade installs its timer. Configure `watch:` in
`<instance>/colonymind.yaml` using [Job watch](JOB-WATCH.md); set `watch.hermes_home` for the Hermes profile
to inspect. Set `watch.notice_command` to an argv list for an independent delivery command that reads the
notice from stdin. Without that command, findings are recorded but no notices are queued; notice text
is never logged.

First inspect one pass without writing state or sending notices:

```bash
colonymind --instance /absolute/path/to/instance watch --once --dry-run
```

The templates below are in the source checkout. Replace every `/absolute/path/to/venv/bin/colonymind` with
the installed executable's absolute path (`command -v colonymind`), and every `/absolute/path/to/instance`
with your instance directory, including the launchd log paths. Timers do not inherit your interactive
shell's environment; use absolute paths in `notice_command` too.

On macOS, copy the launchd template, edit those paths, then load it as your user:

```bash
mkdir -p "$HOME/Library/LaunchAgents"
cp scripts/launchd/com.colonymind.watch.plist "$HOME/Library/LaunchAgents/"
# Edit $HOME/Library/LaunchAgents/com.colonymind.watch.plist before loading it.
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.colonymind.watch.plist"
```

On Linux, copy both systemd user units, edit the service's `ExecStart`, then enable the timer:

```bash
mkdir -p "$HOME/.config/systemd/user"
cp scripts/systemd/colonymind-watch.service scripts/systemd/colonymind-watch.timer "$HOME/.config/systemd/user/"
# Edit $HOME/.config/systemd/user/colonymind-watch.service before enabling it.
systemctl --user daemon-reload
systemctl --user enable --now colonymind-watch.timer
```

Both templates run `watch --once` every five minutes. Check with
`launchctl print "gui/$(id -u)/com.colonymind.watch"` or `systemctl --user status colonymind-watch.timer`.
`colonymind --instance /absolute/path/to/instance watch --list` reads open findings and delivery failures
without the sidecar. Watch never restarts or stops jobs. Notices are delivered at least once; failed
deliveries wait for the next pass and a crash after delivery can repeat a notice.

To remove this optional schedule, use `launchctl bootout "gui/$(id -u)/com.colonymind.watch"` or
`systemctl --user disable --now colonymind-watch.timer`. The sidecar's off switch does not stop this
independent timer; `watch.enabled: false` disables its checks.

## Container

One container with stock Hermes and ColonyMind. The image starts from Hermes'
own published image of the supported release, pinned by digest, and patches
nothing of it: the adapter is one more package in Hermes' environment, the
sidecar has an environment of its own, and the base image's supervisor starts
ColonyMind's setup and then the sidecar beside Hermes. Each release publishes
it as `ghcr.io/aevonix/colonymind:v<version>` and `:latest` (linux/amd64).

[`docker/compose.yml`](../docker/compose.yml) runs it with one model endpoint:

```bash
export OWNER_HANDLES=telegram=123456789 \
  MODEL_URL=http://host.docker.internal:11434/v1 MODEL=qwen3:8b
docker compose -f docker/compose.yml up -d
docker compose -f docker/compose.yml exec agent colonymind doctor
```

Compose checks the variables the example requires for every command, `exec`
and `down` as much as `up`, so export them in the shell you run it from, or
put them in a `.env` file next to `compose.yml` to keep them. At every start,
before the sidecar runs, the setup step:

1. gives the instance directory (`/var/lib/colonymind`, a volume of its own)
   to the user Hermes runs as;
2. with `COLONYMIND_CONTAINER_MODEL_URL` (and `COLONYMIND_CONTAINER_MODEL`, the
   model it serves), points Hermes at that OpenAI-compatible endpoint with
   Hermes' own `hermes config set` (`model.provider: custom`, `model.base_url`,
   `model.default`, and `model.api_mode: chat_completions`, so a transport
   kept from an earlier model is not used with it). A key, when the endpoint
   needs one, stays in
   `COLONYMIND_CONTAINER_MODEL_KEY`: Hermes' config names the variable
   (`model.api_key: ${COLONYMIND_CONTAINER_MODEL_KEY}`), never the value. A
   key Hermes already holds belongs to the endpoint it was given for: when the
   variables move the model to another server (another scheme, host or port)
   and give no key, the setup removes Hermes' `model.api_key` instead of
   sending it there, and it removes a key held for another server under the
   legacy `model.api` (which Hermes reads when `model.api_key` is empty)
   whether or not a key is given; empty `COLONYMIND_CONTAINER_MODEL_KEY` after
   it held a key, and the key is removed from Hermes and the router alike. The router
   uses the key given for that endpoint (or `COLONYMIND_MODEL_API_KEY`, its
   own override) and never one from Hermes' `.env`: with no key given, it
   calls the endpoint without one. Init is
   given the same endpoint, so there is one. Change a variable and restart,
   and both follow. The router's own file
   (`/var/lib/colonymind/.colonymind-llm-config.json`, which init writes once)
   is pointed at the new endpoint in place: its URL, its key and every tier's
   model change, anything else you set in it (a tier's completion or context
   limit, for example) stays, and the version it replaces is copied into
   `backups/`. A binding of its `modelPool` on a server of its own keeps that
   server with the key it was sent there; one the file's endpoint serves
   follows it, with the new key unless it has one of its own for that
   server. A router file the setup cannot read stops it, with the reason
   in `docker logs`, and is left as it is.
   Without these variables, a new instance waits until Hermes has a model of
   its own (`docker compose -f docker/compose.yml exec agent hermes model`,
   then restart): the config Hermes seeds names a hosted placeholder no one
   chose, and the sidecar is never pointed at it;
3. runs `colonymind upgrade` on an existing instance, then `colonymind init
   --non-interactive` with the answers in the `COLONYMIND_INIT_*` variables
   (listed under [Install](#install)).

A restart with the same variables changes nothing. When the setup fails, the
sidecar stays down and `docker logs` says why; the image's health check (the
sidecar's `/v1/host/health` at the address the instance records, the
container's loopback and port 7777 unless `COLONYMIND_INIT_PORT` set another)
reports it too.

`COLONYMIND_INIT_OWNER_HANDLES` is required by the example for a reason: your
handles are how your messages are told from everyone else's. Without them only
the terminal inside the container counts as you, and your messages from a chat
platform are refused as the owner's, as on any install.

Hermes' data is in one volume (`/opt/data`, Hermes' standard mount) and the
instance in another (`/var/lib/colonymind`); back up both. `HERMES_UID` and
`HERMES_GID` set the user that owns them; after a change, the next start hands
the instance over to the new ids. The sidecar listens on the container's
loopback only and nothing is published. `docker exec` runs `colonymind` and
`hermes` as that same user, so nothing they write ends up owned by root.

Inside the container `colonymind doctor` reads as on any machine, with two
differences. The `service` check warns that no user service is installed: here
the image's supervisor runs the sidecar and restarts it when it exits.
`pip-check` can fail on a dependency conflict Hermes' own image already
carries; building the image stops if the adapter adds one or changes any
package Hermes installed. The owner-lane warnings apply as they do elsewhere:
Hermes' tools run as the user that holds the instance key.

To update, pull the new image and start it (`docker compose -f
docker/compose.yml pull`, then `up -d`): the setup step's `colonymind upgrade`
takes its backup (`/var/lib/colonymind/backups/<stamp>/`) and migrates the
stores before the sidecar starts. `docker compose -f docker/compose.yml down`
stops the container and keeps both volumes; `down --volumes` deletes them.

To build the image yourself, from the repository root:

```bash
python -m build --outdir dist . && python -m build --outdir dist ./sidecar
docker build -f docker/Dockerfile -t colonymind .
```

## Coming from another memory provider

Hermes runs its built-in memory plus at most one external provider, so
ColonyMind replaces the provider you use now rather than running beside it.

- `colonymind init` sets `memory.provider: colonymind-memory`, says which
  provider it replaces, and records that name and a backup of the Hermes
  config in `<instance>/hermes-install.json`.
- Hermes' built-in memory stays on. Editing its memory files never creates a
  statement in ColonyMind, and removing a line there is not a reliable way to
  forget: it forgets a source only when the removed text is one whole message
  of exactly one source of the current session, so a fact told in an earlier
  session, or kept as the model's own summary, stays recallable. To forget,
  name the sources: the owner's `colonymind_memory_forget` tool, or
  `POST /v1/host/memory/curated/sources/forget` with their `source_ids`, whose answer
  is a curated tombstone receipt; historical data is retained
  ([the memory provider](../plugins/colonymind-memory/README.md#each-turn)).
- What the previous provider stored stays where it is: ColonyMind neither reads
  nor deletes it. Your own direct Hermes sessions can come along as
  quotations: init offers them, `--import-history` takes them without asking,
  and the [reviewed importer](HERMES-HISTORY-IMPORT.md) brings in other
  sessions, each bound to its person.
- To go back, run `colonymind init --uninstall` and `hermes gateway restart`
  ([Uninstall](#uninstall)): the previous provider is restored while
  `memory.provider` still names ColonyMind's, and ColonyMind's own data stays in
  the instance directory.

## Update

```bash
uv tool upgrade colonymind        # or: pipx upgrade colonymind
colonymind upgrade
hermes gateway restart
```

`colonymind upgrade` stops the sidecar service and waits for its process to
exit, takes a backup (`<instance>/backups/<stamp>/`: the stores, the
configuration files and `hermes-install.json`, so an instance restored from it
still puts back the memory provider init replaced), applies the SQLite
migrations, upgrades the adapter in Hermes' environment (and records both
versions in `hermes-install.json`), reconciles the Hermes
keys and the worker profile, moves aside the stores of removed code, and starts
the service again. When nothing changed it says so and stops. Run it as often
as you like.

Stores move aside last, and only with their writers stopped. A failure before
that (an adapter install that fails, for example) leaves every store where the
sidecar opens it. Whatever fails, even an interrupt, a service the upgrade
stopped runs again, and the upgrade says which step failed, what it changed
before it, the steps it did not do and which stores, if any, moved aside. A sidecar
started by hand (`colonymind start`), found by the process it recorded or by a
listener on the instance's port, is not the service's to stop: while stores
are due to move aside the upgrade refuses before taking a backup, changes
nothing and names it; stop it with `colonymind stop` and run the upgrade again.

An instance whose `colonymind.yaml` names no `credentials` gets them written
explicitly (the instance key writes the owner's turns from WhatsApp, SMS, RCS
and voice), and the upgrade prints what each credential may write as. If the
owner talks to the agent on another platform (the CLI, another messaging
platform), grant it: `colonymind credentials set api-key --owner-platform ...`
(see [CREDENTIALS.md](CREDENTIALS.md)); until then the sidecar refuses those
turns as the owner's and the adapter counts each as a failed capture
(`/mind status`).

A migration counts as applied only when it is. If another process holds a
store's write lock (a sidecar started by hand from an earlier install, for
example), the initiatives store is left as it is and the upgrade fails with
the reason; stop that process (`colonymind stop`) and run it again.
After adding the intention columns the upgrade checks them and the row count
again. Only a file SQLite reports as damaged is recovered: it is kept as
`initiatives.db.corrupt-<stamp>` beside the store, never deleted, and the
store's own backup (`initiatives.db.backup`) is restored when there is one.

Stores of subsystems a release removed are moved into
`<instance>/backups/<stamp>/retired/`, and their tables in surviving stores are
dropped after the backup copied them. For this release: the briefings, the
synthesis insights, the observations, the agent registry, the repo mirrors, the
legacy goal records, the patterns, the proposals, the channel registry, the
shared facts, the situation spine and the corrections ledger (`briefings.db`,
`insights.db`, `observations.db`, `agents.db`, `agents.db.backup`,
`repo-mirrors/`, `goals.db`, `colonymind-goals.db`, `colonymind-patterns.db`,
`colonymind-proposals.db`, `colonymind-channels.db`, `colonymind-facts.db`,
`colonymind-situation.db`, `colonymind-learning-feedback.db`), and leftovers no
code names (`skills.db`, `sandbox/`, `autonomy_checkin.json`,
`autonomy_synthesis.json`). Corpus exports under
`<instance>/exports/` and `SKILL.md` files an earlier release exported under
Hermes' own skills directory (`<hermes_home>/skills/colonymind/`) are left
where they are; ColonyMind no longer manages them. A second upgrade finds
nothing to move.

The execution registry's `execution_observations` and `execution_runtime_observations` tables are also
retired after backup. Its observe, assessments and model-calls routes, commitment local work and drafts,
tool-observation, task-instruction and transport-media capture routes, the self-model and benchmark reads,
and the hands park and triage stubs are removed (HTTP 404 or 405). Update any private callers. The journal
and public document capture route stay. Upgrade removes unread hands settings with notices through
`removed_settings`; do not put them back into configuration.
[Known gaps](KNOWN-GAPS.md#retained-code-outside-the-baseline) lists what was removed.

The MCP server and its harness setup are retired: `colonymind mcp` no longer
exists. The upgrade removes the `mcp_servers.colonymind` entry `colonymind mcp
setup` wrote into the Hermes config (the previous config stays in
`<instance>/backups/`), and moves the `colonymind-diagnose` skills it installed
under `~/.codex/skills/`, `~/.config/crush/skills/` and
`~/.config/opencode/skills/` into `<instance>/backups/<stamp>/harness-skills/`.
An entry or a skill edited since, and the entries in other harnesses' configs
(`~/.claude.json`, `~/.codex/config.toml`, `~/.crush.json`,
`~/.config/opencode/opencode.json`), are left as they are and named in every
upgrade's report until you remove them.

To roll back, stop the sidecar, install the previous version, and copy the
files in `<instance>/backups/<stamp>/` back into the instance directory. Delete
the `-wal` and `-shm` files beside each store you replace: they belong to the
store being replaced, and SQLite would apply them to the copy. Then move back
what `<instance>/backups/<stamp>/retired/` holds that is not there yet, except
SQLite's `-wal` and `-shm` files (each store in it also has a consistent copy
beside `retired/`), and start the sidecar. Backup files and copies are private (0600), in private
directories (0700).

## Upgrading from 1.9.0

This part concerns only an instance that a development version from before
the first release made: the [changelog](../CHANGELOG.md) says under
"Upgrading" in v1.0.0 what the version numbers 1.9 to 1.13 are.

Run `colonymind upgrade` with `COLONYMIND_HOME` pointing at the 1.9.0 instance
directory (the one holding `instance.json`). It converts the keyring to
`api.key` (a credential the 1.9.0 sidecar still accepted, else a fresh key),
the `.env` values to `colonymind.yaml` and `identity.yaml`, keeps the old files
as `.env.1.9.0` and `api-keyring.json.1.9.0`, moves the private-directory
forwarders out of `<hermes_home>/plugins/` into the backup (stock Hermes would
load them ahead of the installed adapter), and starts the mind at
`autonomy: suggest`; edit `mind.autonomy` to choose `standard` or `trusted`.
Reminders scheduled by 1.9.0 keep firing: their launchers run the new adapter.
Durable transport intake rows (`transport_ingress` receipts and coverage) that
1.9.0 stamped with one of its client principals are re-scoped to the single
instance producer, so a messaging transport that journaled those receipts can
still read, hand off and settle them with the one key; the backup keeps the
rows as they were. Every keyring principal keeps exactly the authority it had
(docs/CREDENTIALS.md): the same persons, the same 1.9.0 scopes and routes, and
the owner from every platform for one granted the owner; the one whose secret
the adapter used becomes the instance key; every secret still accepted is kept
with its expiry. The upgrade states each principal's before and after. An
instance that already runs 1.10 migrates no keyring: a 1.9.0 keyring left on
disk is named as retired and not carried.

If the instance used a prepared (patched) Hermes runtime, the upgrade prints
the one command that binds it to stock Hermes instead:

```bash
colonymind upgrade --hermes-python <python of your stock hermes>
```

If the `colonymind` sidecar package was installed inside Hermes' environment,
the upgrade prints how to move it out. Nothing restarts the running gateway;
do that yourself once you are ready.

## Configure

`<instance>/colonymind.yaml`:

```yaml
sidecar: {host: 127.0.0.1, port: 7777}
hermes: {home: ~/.hermes, python: /path/to/hermes/python}
router: {base_url: http://127.0.0.1:8000/v1, model: my-model, embed_provider: endpoint,
         embed_url: "", embed_model: "", embed_revision: "", embed_dims: 0, rerank_url: "", rerank_model: ""}
owner: {contact_id: "<created by init>", sms_handle: "", rcs_handle: ""}
environment: {}                      # COLONYMIND_* settings no key above covers (see below)
mind:
  enabled: true                      # the off switch
  autonomy: suggest                  # off | suggest | standard | trusted
  deny:
    commands: []                     # globs written to the worker's approvals.deny
    tools: []                        # exact tool names blocked in mind-originated runs
    text: []                         # regexes over intention, message and tool-argument text
  worker_toolsets: [web, file, session_search, memory, todo]
  budgets: {tasks_per_hour: 4, concurrent_tasks: 2, owner_messages_per_day: 3,
            contact_messages_per_day: 5, per_contact_cooldown_hours: 24,
            llm_tokens_per_day: 200000, task_max_runtime_s: 600, task_max_retries: 1,
            task_types: {research: {max_runtime_s: 1800, max_retries: 2}, ...}}   # docs/MIND.md
  worker_request: {max_tokens: 8192, top_p: 0.95}   # extra_body on every mind worker request
  worker_model: null                 # a main-config providers entry the worker runs on; null follows the main model
  quiet_hours: "22:00-07:00"         # owner notices wait; tasks and the digest do not
  ask_expires_hours: 72              # silence = no
  breaker: {failures: 3, window_hours: 24, demotion_hours: 72}
  act_threshold: 0.6                 # the ranker's effective-score floor
  digest_hour: 8                     # local hour after which the daily digest goes out
  stale_task_hours: 72               # open board tasks idle longer are reported to the mind
  faculties: {initiative: true, people: true, affect: true, opinions: true, broadcast: true,
              semantic_recall: true, consolidation: true, self_narrative: true, lessons: true,
              skills: false}
projections:
  backlog_per_hour: 12               # model projections queued a day before this release started
context:
  status: true                       # the status block in the owner's turn (docs/MIND.md)
  recall_alarm_turns: 2               # consecutive owner turns missing memory context before one alarm
  # recall_alarm_target: "<platform>:<chat_id>"  # optional stock Hermes target; unset: log only
  # Only the Body loop sends recall alarms; flush()/tick() never send them.
  # Body thread disabled: log-only. One pending slot; newer streaks replace unsent notices.
instinct:                            # the mind's quick typed decisions (docs/DECISIONS.md)
  providers:
    local: {kind: local-decision.v1, url: "", timeout_ms: 250}   # your decision server; empty: no fast tier
  fast: local                        # the provider the fast tier asks
  reader: ""                         # empty: no reader tier
  points: {}                         # per point: {bars: {label: p}}, which can only raise the code floors
  external: {}
credentials:                         # who each API credential may write as (docs/CREDENTIALS.md)
  api-key: {owner: [cli, telegram], contacts: all}   # api.key: the Hermes adapter; init writes cli and your handles' platforms
guests:                              # non-owner sessions (docs/HERMES-ADAPTER.md, Owner and guest)
  executor_platforms: []             # platforms whose non-owner sessions keep every tool
  tools: []                          # extra tools a non-owner session may call (never a shell, code or file write)
```

`projections.backlog_per_hour` bounds the model work an upgrade inherits. Claim
extraction, appraisal, capture and image descriptions each keep a queue of
jobs, one per source. A job queued more than a day before the running release
first opened the store is backlog: each queue takes its new jobs first and at
once, and backlog jobs start at most this many times an hour across all the
queues, evenly spaced (0 leaves them pending). The sidecar logs the backlog it
found at startup.

`router.embed_url` (an OpenAI-compatible embeddings endpoint, with
`embed_model`) turns semantic recall on. `router.embed_provider: local` runs
the model `embed_model` at the commit `embed_revision` (width `embed_dims`)
inside the sidecar instead, from the copy init downloaded into the instance
directory, with `embed_url` left empty; init records all four, and a start
needs no network. Either way, `mind.faculties.semantic_recall:
false` keeps it off with the embedder still recorded. Releases before this
switch was live wrote `semantic_recall: false` whenever init found no
endpoint; `colonymind init --embed-url ...` sets it back to true, and `colonymind upgrade` and
`colonymind doctor` warn while an endpoint is recorded with the switch off. `router.embed_dims` is the model's
vector width; left at 0, the sidecar learns it from the endpoint's first
embedding, and a declared width is validated against every vector (a mismatch
is a startup failure named after the setting, never a silent switch to
keyword recall). `router.rerank_url` (an OpenAI/Jina style `/v1/rerank`
endpoint) with `router.rerank_model` (the model it serves; required with the
endpoint) makes recall rerank its candidates there. All are exported to the
sidecar process when it starts; a value already in that process environment
wins, so `COLONYMIND_RECALL_RERANK=shadow` can still be pinned to measure a
reranker before it changes what recall returns.

When an embedding endpoint is configured and the embedder does not come up
(the endpoint is down, the model name is wrong, the width differs), the
sidecar still serves, but `/v1/host/health` reports `degraded` with the reason
in `problems` ("semantic recall is off: ..."), and `colonymind doctor` fails its
`semantic-recall` check with that reason. Recall does not quietly fall back to
keywords.

`instinct.providers.local.url` names a non-generative decision model on your
own network (`POST /v1/decide`, typed choice or yes/no answers with
probabilities) that Instinct's fast tier asks at some short typed decisions:
the owner's reply to an outreach, a contact's opt-out, a first mention the
person wants no reminders about, a settled interest, whether an owner message
is a verdict on the agent's work, and whether a message to the owner can wait.
A point's model tier acts only once it is live (it ships in shadow, asked and
logged, unless it measured at least as accurate as the path it falls back to),
and any failure, timeout or unsure answer keeps that path; see
[DECISIONS.md](DECISIONS.md) for the points, the states, the measurements and
the log. With no url nothing is asked. `colonymind upgrade` carries a 1.11
`decisions:` section into `instinct:` once; the sidecar refuses to start with
one left, and nothing reads the old `COLONYMIND_DECISIONS_*` variables
(`colonymind doctor` names one still set).

The mind itself (the tick, authority, asks, the audit log, the outbox and the
off switch) and the `colonymind mind` command are described in
[docs/MIND.md](MIND.md), and its nightly consolidation (digests, contradictions,
episode summaries and the self-narrative) in
[docs/CONSOLIDATION.md](CONSOLIDATION.md).

The sidecar reads a number of tuning settings from its process environment
that have no key of their own: a reranker prompt style, recall thresholds and
an endpoint credential. `environment` carries any of them in the
one configuration file, so a calibration survives an upgrade without a
hand-maintained service unit:

```yaml
environment:
  COLONYMIND_RERANKER_PROMPT_STYLE: qwen3
  COLONYMIND_RECALL_RERANK_MIN_SCORE: "0.74"
  COLONYMIND_EMBED_API_KEY: "<the embedding endpoint's key, if it needs one>"
```

Names must be `COLONYMIND_` followed by capitals, digits and underscores. A
name that another key already defines (the instance directory, `sidecar.*`,
`api.key`, `mind.enabled`, `mind.autonomy`, `owner.contact_id`, `owner.sms_handle`, `owner.rcs_handle`, the identity
fields, `router.embed_*` and `router.rerank_*`) is refused with the key to use
instead. Values are strings or numbers; quote words YAML would read as
booleans (`"on"`, `"off"`, `"yes"`). Entries are exported when the sidecar
starts, after the values the keys above derive (so `COLONYMIND_RECALL_RERANK:
shadow` here measures a configured reranker before it changes what recall
returns), and a value already in the process environment still wins. The
sidecar log names the entries it exported; a credential's value never reaches
a log, and neither does its name.

A few environment variables override the file for one process:
`COLONYMIND_HOME` (the instance directory), `COLONYMIND_SIDECAR_HOST`,
`COLONYMIND_SIDECAR_PORT`, `HERMES_HOME`, `COLONYMIND_MIND_ENABLED`,
`COLONYMIND_AUTONOMY`, `COLONYMIND_OWNER_SMS_HANDLE`, `COLONYMIND_OWNER_RCS_HANDLE` and `COLONYMIND_API_KEY` (in place of `api.key`).

For owner SMS or RCS, set `owner.sms_handle` or `owner.rcs_handle` locally, or give
the corresponding `COLONYMIND_OWNER_SMS_HANDLE` / `COLONYMIND_OWNER_RCS_HANDLE` override
to the sidecar and gateway. Each binding is optional, independent and empty by default.
It identifies one authenticated sender;
it does not enable a phone transport or add a credential grant. See
[Owner SMS identity](CREDENTIALS.md#owner-sms-identity) for the configuration
and transport requirements.

## Security

- The sidecar binds `127.0.0.1` and refuses a non-loopback bind without a key.
- Credentials bound to who they may write as ([CREDENTIALS.md](CREDENTIALS.md)).
  Every request from the plugin sends `Authorization: Bearer <key>` (`api.key`);
  the person a request acts for comes from the contact identity in the body,
  and the key writes a turn as the owner only from the platforms
  `credentials.api-key.owner` lists (init writes this terminal, `cli`, and the
  platforms of your handles; `colonymind init --owner-platform` sets them; an
  instance whose file names no credentials keeps the earlier default,
  WhatsApp, SMS, RCS and voice). Every other client gets its own
  credential (`colonymind credentials add`), which by default writes only under
  its own identity, never as the owner. A turn a credential may not write is
  refused with 403, never re-attributed, and a credential may carry an expiry
  (`accept_until`) that the sidecar enforces on every request.
- Without a key the API serves loopback callers only (a development
  convenience), and the credential-handling routes stay closed. Its caller is
  never the owner and changes nothing outside the turn and context routes.
- A contact's session in Hermes may call only the guest tools (no shell,
  code, file-writing, browser or desktop tool, and none the plugin does not
  know), and the files they take are only those the sender's own messages
  carried, unless `guests` in `colonymind.yaml` opts a platform or a tool in;
  `colonymind doctor` names the lanes that could run this machine
  (`guest-lane-tools`).

## Uninstall

```bash
colonymind init --uninstall
hermes gateway restart
```

This removes the adapter and its keys from Hermes' config and puts back the
memory provider init replaced (while `memory.provider` still names
ColonyMind's; a provider you chose since is left as it is), moves the
`colonymind-act` profile (its sessions, memories and logs) into
`<instance>/backups/<stamp>/profiles/`, removes the sidecar service and
uninstalls the adapter from Hermes' environment, unless another profile of
that Hermes home still enables it. What is left is stock Hermes. The instance
directory and its data are kept; delete it yourself when you no longer want
them.
