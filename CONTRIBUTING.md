# Contributing to ColonyMind

Develop against the current ColonyMind baseline and its declared Hermes target.

## Development setup

ColonyMind is Python-only.

```bash
git clone https://github.com/Aevonix/ColonyMind.git
cd ColonyMind/sidecar
pip install -e ".[dev]"
```

Python 3.11, 3.12 and 3.13 are supported, and CI runs the suites on each; 3.12
is recommended. The lightweight profile
uses SQLite and one OpenAI-compatible model endpoint. Tests use local fixtures
unless a selected qualification explicitly requires a real model or service.

## Repository layout

| Path | Purpose |
|---|---|
| `sidecar/colonymind/` | The Python package: FastAPI sidecar, CLI, all subsystems |
| `sidecar/colonymind/api/` | Pydantic schemas and routers: the single source of truth for the HTTP contract |
| `sidecar/colonymind/intelligence/` | Cognition components, synthesis, owner corrections |
| `sidecar/colonymind/mind/` | The mind: authority, the tick, the ranker, outcomes, the outbox and the audit log |
| `sidecar/tests/` | Sidecar test suite, kept out of the installed product package |
| `plugins/` | Host integration plugins: `hermes-plugin` (general adapter), `colonymind-memory` (memory provider), `feeds-manage` |
| `docs/` | User guides, indexed by task in `docs/README.md`; design notes and historical studies in `docs/internal/` |

## Making changes

1. **Fork and branch**: create a feature branch from `main`
2. **Write code**: follow existing patterns in the codebase
3. **Test**: the full suite must pass before submitting
4. **Commit**: see commit conventions below
5. **PR**: open a pull request against `main`

### Running the tests

```bash
cd sidecar
python -m pytest tests/ -q                  # sidecar suite
python -m pytest tests/test_doctor.py -q     # one file
python -m pytest -x                          # stop on first failure
```

From the repo root with the qualified Hermes and sidecar installed, run
the native adapter and source-recall harness contracts separately:

```bash
python -m pytest tests/hermes_adapter -q
python -m pytest benchmarks/source_recall/test_harness.py -q
```

The compatibility tests in `sidecar/tests/test_host_identity_parity.py` and
`sidecar/tests/test_longmemeval_anchor_release_a.py` use frozen source fixtures
under `sidecar/tests/fixtures/compatibility_sources`, so shallow checkouts can
run them. The full LongMemEval comparisons also require the pinned dataset cache.

Their distinct `conftest.py` modules require separate collection. The harness
uses a local synthetic HTTP server; it is a storage/transport check, not a
measurement of model quality or a production smoke test.

### Commit conventions

Use a conventional-commit-ish style for new commit subjects:

```
feat(autonomy): COLONYMIND_AUTONOMY_PRESET - one knob for the agency posture
fix(trust): durable graduation/demotion notices
docs(prompts): record adoption status, eval harness, version attribution
refactor(plugins): share the native adapter client
chore(generic): genericize remaining identity strings in two test fixtures
```

Use `feat` / `fix` / `docs` / `refactor` / `chore` / `test` with an optional
scope, and make the body explain the *why*.

## The genericity rule

**This repository is deployment-agnostic.** It must never contain personal,
persona, or deployment specifics: no real names, contact ids, phone numbers,
email addresses, hostnames, addresses, channel ids, API keys, model endpoints
or serving aliases, or measurements taken on a private store. Deployment values
come exclusively from configuration: the instance's `colonymind.yaml`, including
its `environment:` section for sidecar settings (see
[configure](docs/INSTALL.md#configure)). Test fixtures use neutral names,
TEST-NET addresses and 555 numbers; a result measured on a real store is
described in general terms, and a model is named only as a public example.

CI scans every tracked file with `scripts/genericity_scan.py`, which finds the
generic shapes of such details: a user's home directory, an address on a
private network outside the example subnets, a tailnet host, an email address
outside the reserved example domains, and a real-looking phone number. Your
deployment's own words (its people, hosts, endpoints and anything else it never
publishes) belong in a private denylist on your own machine, one regular
expression per line, which the scan reads when it exists:

```bash
# ~/.config/colonymind/genericity-denylist, or the file COLONYMIND_GENERICITY_DENYLIST names
python scripts/genericity_scan.py
```

A finding names the file, line, column and rule, never the text it matched. A
deliberate example that only looks like a detail can carry
`genericity: example` on its line; a real detail is replaced, never marked.

If a feature needs a deployment-specific value to work, read it from
configuration and document the setting in the guide for that feature instead
of hardcoding it.

## Versioning

ColonyMind uses **Semantic Versioning** (`MAJOR.MINOR.PATCH`): MINOR for compatible
features, PATCH for fixes and documentation, and MAJOR for incompatible public
contracts. The `colonymind` sidecar and `colonymind-hermes` integration are published together
on PyPI with the same version.

Version 1.0.0 establishes the first supported baseline. Earlier development
versions and migration paths are not supported public
contracts. Remove unused compatibility code when changing an area, while
accounting for current callers and retained state. Do not add compatibility
layers solely to preserve historical names.

## Release flow

1. Set the version with the one command, then check it:
   `python scripts/set_version.py NEW_VERSION`, then `python scripts/set_version.py --check`.
   The first writes the version into every tracked place that holds it. The second prints it,
   or names what each place holds when they differ.
2. Add an entry at the top of `CHANGELOG.md` from [the release notes template](.github/RELEASE_NOTES_TEMPLATE.md):
   a user-facing title, what changes, and an "Upgrading" block
3. Commit and tag. The tag must be `v` plus that version:
   `git tag vNEW_VERSION && git push --tags`
4. CI (`.github/workflows/release.yml`) publishes to PyPI, pushes the Docker
   image to GHCR (`ghcr.io/aevonix/colonymind`), and creates the GitHub release
   from the changelog entry: all automatically on the tag push

The build that makes the distributions (the `artifacts` job of `.github/workflows/ci.yml`)
runs the check again and fails when the tag is not `v` plus the version. It also writes the
build stamp, with `python scripts/write_build_stamp.py`: the commit, the branch and whether
the tree had uncommitted changes. The sidecar's wheel and source archive carry the stamp,
and `colonymind doctor` shows it. A tag whose tree has uncommitted changes is not built.
Nothing is published when the build fails.

## Architecture notes

The target ownership boundaries and remaining implementation work are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Keep deployment-specific hardware
and identity in private integrations. The notes below describe the existing
implementation, not a claim that every planned boundary is complete.

- **SQLite owns canonical source memory.**
  Other domains retain separate SQLite stores. Lance is an optional search
  index; there is no graph database. Hermes owns its transcripts and native work
  execution. Adapters own delivery outboxes.
- **Native adapters prepare context and capture turns.** They share the
  sidecar client and durable ingestion path rather than duplicating cognition.
- **Models are configured per instance.** Credentials stay in private
  configuration; the host configuration API is `POST /v1/host/configure`.
- **Enable new autonomy through measured behavior.** Start an unqualified
  loop in `off` or `shadow`, use its existing mode resolver, and require the
  applicable authority before consequential effects.

## Reporting issues

- **Bugs:** open an issue with reproduction steps, logs, and environment info
- **Features:** open an issue describing the use case and proposed approach
- **Security:** report privately as [SECURITY.md](SECURITY.md) describes; do not open a public issue

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
