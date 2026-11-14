"""The documentation link checker (``scripts/check_doc_links.py``): a relative link reaches a file in the
repository, a fragment reaches a heading, and this repository's pages have no broken link."""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_doc_links.py"


def _tree(base: Path, files: dict[str, str]) -> Path:
    repo = base / "repo"
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return repo


def _check(repo: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(repo), *extra],
                          capture_output=True, text=True, timeout=120)


def _module():
    spec = importlib.util.spec_from_file_location("colonymind_check_doc_links", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GOOD = {
    "README.md": (
        "# A project\n\n"
        "[install](docs/INSTALL.md), [configure](docs/INSTALL.md#configure-it), [here](#a-project),\n"
        "[the docs folder](docs/), [by root](/docs/INSTALL.md), [spaced](<docs/two words.md>),\n"
        '[titled](docs/INSTALL.md "Install"), ![logo](docs/logo.svg) [![badge](docs/logo.svg)](LICENSE)\n'
        '<picture><source srcset="docs/logo.svg 1x, docs/logo.svg 2x"><img src="docs/logo.svg"></picture>\n'
        '<a href="docs/INSTALL.md#back-to-the-start">back</a>\n'
        "[elsewhere](https://example.invalid/nowhere.md), [mail](mailto:someone@example.invalid)\n\n"
        "[ref]: docs/INSTALL.md#configure-it\n"
    ),
    "LICENSE": "MIT\n",
    "docs/logo.svg": "<svg/>\n",
    "docs/two words.md": "# Two words\n",
    "docs/INSTALL.md": (
        "# Install\n\n## Configure it\n\n[the project](../README.md#a-project)\n\n"
        "Back to the start\n-----------------\n"
    ),
}


def test_a_tree_whose_links_resolve_passes(tmp_path):
    done = _check(_tree(tmp_path, GOOD))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "0 broken" in done.stdout


def test_a_missing_file_and_a_missing_heading_are_reported_with_their_line(tmp_path):
    files = dict(GOOD)
    files["docs/GUIDE.md"] = ("# Guide\n\n[gone](MISSING.md)\n\n"
                              "text [not a section](INSTALL.md#nope) and [not here](#nowhere)\n")
    done = _check(_tree(tmp_path, files))
    assert done.returncode == 1, done.stdout + done.stderr
    assert "docs/GUIDE.md:3: MISSING.md (no such file)" in done.stdout
    assert "docs/GUIDE.md:5: INSTALL.md#nope (no such heading)" in done.stdout
    assert "docs/GUIDE.md:5: #nowhere (no such heading)" in done.stdout
    assert "3 broken" in done.stdout


def test_links_inside_code_are_not_links(tmp_path):
    files = dict(GOOD)
    files["docs/CODE.md"] = ("# Code\n\n```python\nx = a[0](missing.md)\n[b](missing.md)\n```\n\n"
                             "~~~\n[c](missing.md)\n~~~\n\n    [indented](missing.md)\n\n"
                             "Inline `[d](missing.md)` stays code.\n")
    done = _check(_tree(tmp_path, files))
    assert done.returncode == 0, done.stdout


def test_a_destination_wrapped_onto_another_line_is_checked_on_the_line_it_is_on(tmp_path):
    """A link, in a quote too, and a reference definition may break the line before the destination or the
    title: the page renders a link, so it is checked, and reported where the destination is. A line break inside
    ``<...>``, a blank line or the end of a heading leaves no link."""
    files = dict(GOOD)
    files["docs/WRAPPED.md"] = (
        "# Wrapped\n\n"
        "See [the guide](\nMISSING.md\n) for more, and [a titled one](INSTALL.md\n\"Install\") that resolves.\n\n"
        "> A quoted [link](\n> GONE.md).\n\n"
        "[ref]:\n  ALSO-MISSING.md\n  'A title'\n\n"
        "Not links: [x](<\nNOPE.md>), [y](\n\nNOPE.md).\n\n"
        "## A heading [z](\nNOPE.md) is not a link.\n\n[ref] used.\n"
    )
    done = _check(_tree(tmp_path, files))
    assert done.returncode == 1, done.stdout
    assert "docs/WRAPPED.md:4: MISSING.md (no such file)" in done.stdout
    assert "docs/WRAPPED.md:9: GONE.md (no such file)" in done.stdout
    assert "docs/WRAPPED.md:12: ALSO-MISSING.md (no such file)" in done.stdout
    assert "NOPE" not in done.stdout
    assert "3 broken" in done.stdout


def test_a_paragraph_under_a_list_item_is_prose_and_only_code_under_it_is_code(tmp_path):
    """A list item's later paragraphs are indented to its content, four columns or more under a number or a nested
    bullet: they render as prose, so their links are checked. Indented code in an item starts four columns past
    its content, a fence in an item is code at the item's indent, and a heading ends the list."""
    files = dict(GOOD)
    files["docs/LIST.md"] = (
        "# List\n\n"
        "- Read more:\n\n    [guide](MISSING.md)\n\n"
        "1. Then:\n\n    [another](GONE.md)\n\n"
        "   - nested\n\n       [deep](DEEP.md)\n\n"
        "- Code under an item:\n\n      [code](NOPE.md)\n\n"
        "10. A fence in an item:\n\n    ```\n    [fenced](NOPE.md)\n    ```\n\n"
        "    [after the fence](LATE.md)\n\n"
        "- One more item\n# A heading ends the list\n\n    [indented code](NOPE.md)\n"
    )
    done = _check(_tree(tmp_path, files))
    assert done.returncode == 1, done.stdout
    for line, target in ((5, "MISSING.md"), (9, "GONE.md"), (13, "DEEP.md"), (25, "LATE.md")):
        assert f"docs/LIST.md:{line}: {target} (no such file)" in done.stdout
    assert "NOPE" not in done.stdout
    assert "4 broken" in done.stdout


def test_heading_fragments_follow_the_rendered_heading_ids(tmp_path):
    files = dict(GOOD)
    files["docs/MIND.md"] = (
        "# The mind\n\n## The status block (off by default)\n\n## Asks\n\n## Asks\n\n"
        "### `colonymind mind` commands\n\n## Wishes: a typed grant, words a checked label\n\n"
        "## Step 4. Check it #\n\nOwner's view\n============\n\n<a id=\"custom-anchor\"></a>\n\n"
        "```\n## Not a heading\n```\n"
    )
    files["docs/LINKS.md"] = (
        "# Links\n\n[a](MIND.md#the-status-block-off-by-default) [b](MIND.md#asks) [c](MIND.md#asks-1)\n"
        "[d](MIND.md#colonymind-mind-commands) [e](MIND.md#wishes-a-typed-grant-words-a-checked-label)\n"
        "[f](MIND.md#step-4-check-it) [g](MIND.md#owners-view) [h](MIND.md#custom-anchor)\n"
        "[i](MIND.md#The-Mind)\n"
    )
    assert _check(_tree(tmp_path, files)).returncode == 0
    files["docs/LINKS.md"] += "[j](MIND.md#asks-2) [k](MIND.md#not-a-heading)\n"
    done = _check(_tree(tmp_path / "second", files))
    assert done.returncode == 1
    assert "MIND.md#asks-2 (no such heading)" in done.stdout
    assert "MIND.md#not-a-heading (no such heading)" in done.stdout


def test_a_link_that_leaves_the_repository_is_broken(tmp_path):
    (tmp_path / "outside.md").write_text("# Outside\n", encoding="utf-8")
    files = dict(GOOD)
    files["docs/OUT.md"] = "# Out\n\n[outside](../../outside.md)\n"
    done = _check(_tree(tmp_path, files))
    assert done.returncode == 1
    assert "docs/OUT.md:3: ../../outside.md (outside the repository)" in done.stdout


def test_links_to_this_repository_on_its_main_branch_are_checked_against_the_tree(tmp_path):
    """A package page (PyPI renders no relative link) links the repository by URL; a moved doc breaks it."""
    files = dict(GOOD)
    files["pyproject.toml"] = '[project]\nname = "x"\n\n[project.urls]\nRepository = "https://github.com/Example/Project"\n'
    files["pkg/README.md"] = (
        "# Package\n\n[install](https://github.com/Example/Project/blob/main/docs/INSTALL.md#configure-it)\n"
        "[old tag](https://github.com/Example/Project/blob/v0.1.0/docs/RETIRED.md)\n"
        "[another project](https://github.com/Other/Project/blob/main/docs/NOTHING.md)\n"
        "[folder](https://github.com/Example/Project/tree/main/docs)\n"
    )
    assert _check(_tree(tmp_path, files)).returncode == 0
    files["pkg/README.md"] += "[moved](https://github.com/Example/Project/blob/main/docs/MOVED.md)\n"
    done = _check(_tree(tmp_path / "second", files))
    assert done.returncode == 1
    assert "pkg/README.md:7: https://github.com/Example/Project/blob/main/docs/MOVED.md (no such file)" in done.stdout


def test_in_a_checkout_a_link_must_reach_a_tracked_file(tmp_path):
    """An untracked or ignored file exists here but not in the published repository."""
    repo = _tree(tmp_path, {"README.md": "# R\n\n[new](docs/NEW.md)\n", "docs/NEW.md": "# New\n",
                            ".gitignore": "build/\n", "build/OUT.md": "# Out\n"})
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "README.md", ".gitignore"], check=True)
    done = _check(repo)
    assert done.returncode == 1
    assert "README.md:3: docs/NEW.md (not tracked)" in done.stdout
    subprocess.run(["git", "-C", str(repo), "add", "docs/NEW.md"], check=True)
    (repo / "README.md").write_text("# R\n\n[new](docs/NEW.md) [built](build/OUT.md)\n", encoding="utf-8")
    done = _check(repo)
    assert "README.md:3: build/OUT.md (not tracked)" in done.stdout
    assert "docs/NEW.md" not in done.stdout


def test_a_changelog_is_checked_in_its_newest_release_only(tmp_path):
    """Older release notes describe the tree of their release: a doc a later release retired is history there."""
    files = dict(GOOD)
    files["CHANGELOG.md"] = ("# Changelog\n\n## v2.0.0 - newest\n\nSee [install](docs/INSTALL.md).\n\n"
                             "## v1.0.0 - older\n\nSee [the retired guide](docs/RETIRED.md).\n")
    assert _check(_tree(tmp_path, files)).returncode == 0
    files["CHANGELOG.md"] = files["CHANGELOG.md"].replace("docs/INSTALL.md", "docs/GONE.md")
    done = _check(_tree(tmp_path / "second", files))
    assert done.returncode == 1
    assert "CHANGELOG.md:5: docs/GONE.md (no such file)" in done.stdout
    assert "RETIRED" not in done.stdout


def test_external_links_are_never_fetched(tmp_path, monkeypatch, capsys):
    module = _module()

    def refuse(*args, **kwargs):
        raise AssertionError("the link checker opened a connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    repo = _tree(tmp_path, {"README.md": "# R\n\n[x](https://example.invalid/a) [y](http://example.invalid/b)\n"})
    assert module.main(["--root", str(repo)]) == 0
    assert "0 broken" in capsys.readouterr().out


def test_this_repositorys_pages_have_no_broken_link():
    if not (ROOT / ".git").exists():
        pytest.skip("not a checkout")
    done = _check(ROOT)
    assert done.returncode == 0, done.stdout[-4000:]
