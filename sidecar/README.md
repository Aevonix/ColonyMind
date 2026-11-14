# ColonyMind

**A mind for Hermes agents: persistent memory, commitments, initiative and measured autonomy.**

Not just remember: keep promises and take the next step, within limits the owner sets. ColonyMind gives a personal
[Hermes](https://github.com/NousResearch/hermes-agent) agent memory across sessions and channels in people's own
words, commitments it keeps, people it knows, and a small mind that acts on its own inside an autonomy level, budgets
and quiet hours the owner sets. Asks reach the owner as plain messages, answered with a plain yes, a reply or a
thumbs-up; the model can never approve anything.

This package is the ColonyMind sidecar and its `colonymind` command. It attaches to stock Hermes through the companion
`colonymind-hermes` adapter, which `colonymind init` installs into Hermes' own environment; the sidecar runs in an
environment of its own. No fork, no patches, no graph database, no extra key.

## Install

You need Python 3.11 to 3.13 (3.12 recommended), a stock Hermes install (`hermes-agent >=0.21.3,<0.24`) whose
`hermes` executable is on `PATH`, and one OpenAI-compatible chat endpoint, which Hermes already has.

```bash
uv tool install --python 3.12 colonymind
colonymind init
hermes gateway restart
```

pipx works too (`pipx install --python python3.12 colonymind`). ColonyMind becomes Hermes' memory provider: `init`
names the provider it replaces, and `colonymind init --uninstall` puts it back. `colonymind doctor` checks the install,
and `colonymind demo` shows the loop with no model, no keys and no network: a fact recalled across sessions, a
promise followed up on its own, and an ask the person answers in plain words.

One container runs stock Hermes and ColonyMind together: `ghcr.io/aevonix/colonymind`, stock Hermes' own published
image pinned by digest and patched nowhere
([the container guide](https://github.com/Aevonix/ColonyMind/blob/main/docs/INSTALL.md#container)).

The vector store (LanceDB) is part of every install: semantic recall needs only an OpenAI-compatible embeddings
endpoint (`colonymind init --embed-url auto` looks for one on the model endpoint). Where the endpoint has none, the
`vectors` extra lets the sidecar run a pinned embedding model itself
(`uv tool install --force --python 3.12 'colonymind[vectors]'`, which brings PyTorch, then
`colonymind init --embed-url local`, which downloads the model once into the instance directory).

To update: `uv tool upgrade colonymind` (or `pipx upgrade colonymind`), `colonymind upgrade`, `hermes gateway restart`.

## Learn more

- [Project README](https://github.com/Aevonix/ColonyMind/blob/main/README.md): what works, the evidence against stock
  Hermes, and what is not yet demonstrated.
- What memory holds about a person, with each claim's history:
  [the memory inventory](https://github.com/Aevonix/ColonyMind/blob/main/docs/SOURCE-ANNOTATIONS.md#the-memory-inventory);
  what the model calls cost:
  [usage by lane](https://github.com/Aevonix/ColonyMind/blob/main/docs/FUNCTION-ROUTING.md#usage-model-calls-by-lane).
- [Install and update](https://github.com/Aevonix/ColonyMind/blob/main/docs/INSTALL.md): what `colonymind init`
  changes, configuration, the sidecar service, updating and uninstalling.
- [Documentation index](https://github.com/Aevonix/ColonyMind/blob/main/docs/README.md),
  [troubleshooting](https://github.com/Aevonix/ColonyMind/blob/main/docs/TROUBLESHOOTING.md) and the
  [changelog](https://github.com/Aevonix/ColonyMind/blob/main/CHANGELOG.md).

Licensed under the included MIT license. Optional dependencies retain their own licenses.
