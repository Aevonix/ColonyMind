# Changelog

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
