# Compatibility

Which Hermes and Python versions each ColonyMind release qualifies. ColonyMind attaches to stock, unpatched Hermes
through its plugin and memory-provider interfaces; the sidecar (`colonymind`) and the adapter (`colonymind-hermes`)
are released together with the same version.

| ColonyMind | Installer's Hermes range (`hermes-agent`) | Python environments | Qualification |
|---|---|---|---|
| 1.0.x | `>=0.21.3, <0.24` | Sidecar: 3.11, 3.12, 3.13; Hermes: 3.14 from 0.21.7 | Install and first loop: Hermes 0.21.3 from source, Hermes 0.23.1 from Hermes' installer. Container: Hermes 0.21.3. Adapter suite: Hermes 0.21.7 and Hermes 0.23.1. |

3.12 is the recommended sidecar Python: the install guide picks it, and the release artifacts are built and checked on it.

Hermes 0.21.3 is the oldest supported host for installation, the first loop and the container, qualified on
Python 3.11 to 3.13. The session archive requires Hermes 0.21.7 or newer; its second-interpreter tests use Python
3.14. With Hermes' installer, installation and the first loop are also qualified on Hermes 0.23.1, the release
that installer brings today.

The adapter's own test suite is qualified on Hermes 0.21.7 and on Hermes 0.23.1. On Hermes 0.23.1 one run of
it, made as the hosted job makes it (Python 3.14, without the sidecar), passed 1714 of its tests, failed none and
skipped 160; the job's skip accounting found each skip to be a test marked for something that job leaves out.
Hermes 0.23.1 has one vault tool, `browser_vault`, with the fill as one of its actions; Hermes 0.21.7 has five,
`browser_vault_fill` among them. The mind reads the one tool as the older tool of its action, so a vault call has
the same effect, the same deny entry and the same grant name on both, and the owner's yes to a payment fill
reaches Hermes' own consent on both.

From Hermes 0.21.7 on, a stock install is complete only on Python 3.14 because its dependencies carry interpreter
markers. Hermes and its adapter then live in a Python 3.14 environment, and the sidecar lives in a separate
Python 3.11 to 3.13 environment.
The hosted adapter checks also install that Hermes on Python 3.11 to 3.13 with
those dependency markers lifted. That is a test setup, not the way a user installs Hermes.

## Installing a supported Hermes

Three ways are checked; the [install guide](INSTALL.md#install) has the commands.

- **Hermes' installer as it comes** (Hermes 0.23.1 today) and `uv` on `PATH`. The environment this installer builds
  has no pip; without `uv` `colonymind init` stops with `<Hermes python> has no pip and 'uv' is not on PATH;
  install one of them to manage Hermes' environment`. The hosted checks set Hermes up this way on every change,
  pinned with `--branch v0.23.1`, and run the install checks (install, upgrade twice, doctor, uninstall), recall
  across two sessions, the first closed loop and the demo against that one Hermes home. The same four steps were
  run in a container for this release, with Hermes on Python 3.14.9 and the sidecar on Python 3.12.16: all four
  passed, the first closed loop with 28 of its 28 checks.
- **Hermes 0.22.7 with Hermes' installer** (`--branch v0.22.7`) and `uv` on `PATH`. This way is not part of the
  hosted checks: it was run once in a container for this release, in the same way and with the same result (all
  four steps passed, the first closed loop with 28 of its 28 checks).
- **Hermes 0.21.3 from source with pip**, in a virtual environment. The hosted fresh-install, first-loop and
  container checks install Hermes this way on every change.

Hermes 0.24 is outside the range and is refused.

Hermes' installer gives every Hermes home its own environment. The adapter is installed once per Hermes home, by
`colonymind init --hermes-home <home>`; `colonymind doctor` names a home that is configured without it.

## What the ranges mean

- **Hermes.** `colonymind init` refuses a Hermes outside the release's range, and `colonymind doctor` reports it as a
  failed check. This installer range is not evidence that every version or feature inside it was qualified;
  the qualifications and the session archive's higher floor are stated above.
  The version comes from `hermes-agent` package metadata. If metadata is exactly `0.0.0`,
  the installer reads the owning Hermes checkout's `install-stamp.json` `baseVersion`.
  For a PM-managed environment, `installs/<id>/inputs/.project-root` locates that checkout;
  the runtime's separate `workspace` snapshot does not carry the stamp.
  One source is read, the stamp only when metadata is `0.0.0`, and the version it gives must be a
  release in `>=0.21.3,<0.24`. A missing, unreadable or unversioned stamp is refused. `init`, `upgrade` and `doctor` share this version read.
  Local versions such as `0.22.2+local.example` use the base release for this check;
  unversioned builds (`0.0.0`, including local versions) are refused.
- **Python.** The sidecar accepts Python 3.11 to 3.13 and the adapter accepts Python 3.11 to 3.14
  (`requires-python`). Install the adapter into Hermes' own environment.

## Checking your install

The supported range ends below Hermes 0.24. `colonymind init` reads the version, refuses a Hermes outside the
range before it writes anything, and says which: `Hermes 0.24.0 from Hermes install-stamp.json baseVersion
(hermes-agent metadata 0.0.0) in <python> is outside the supported range >=0.21.3,<0.24`.

    colonymind doctor

It reports the Hermes version it found against the supported range, whether the adapter in Hermes' environment
matches the sidecar's version, and `pip check` there: clean, a warning for a conflict among Hermes' own packages,
or a failure for one that involves the adapter.

## How this is kept true

- Every change runs the sidecar suite and a fresh install of Hermes 0.21.3 (install, upgrade twice, uninstall) on Python 3.11 to
  3.13. Adapter tests use Hermes 0.21.7, including the marker adjustment described above on those interpreters.
- Every change also sets Hermes 0.23.1 up with Hermes' own installer and runs the install checks, recall across
  two sessions, the first closed loop and the demo against it, with the sidecar on Python 3.12.
- Every change also runs the adapter tests against Hermes 0.23.1 on Python 3.14, without the sidecar, and
  accounts for every skip.
- A daily job runs the adapter tests against the latest published Hermes release.
- A second daily job runs the adapter tests against Hermes' development branch. It only warns: it shows a change
  coming in a future Hermes release early, and blocks nothing.
