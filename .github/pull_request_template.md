## What changes for users

<!-- In plain words: what someone running ColonyMind will notice. "Nothing" is a fine answer for internal work. -->

## Why

<!-- The problem this solves, or the issue it closes. -->

## Checks

- [ ] A test that failed before this change and passes with it
- [ ] The sidecar suite: `cd sidecar && python -m pytest tests -q`
- [ ] For adapter, provider or plugin changes: `python -m pytest tests/hermes_adapter -q` against stock Hermes
- [ ] `ruff check` over the paths CI checks
- [ ] Code, tests, docs and commit messages follow "The genericity rule" in CONTRIBUTING.md: no names, hosts,
      addresses or other details of a real deployment (`python scripts/genericity_scan.py`)
