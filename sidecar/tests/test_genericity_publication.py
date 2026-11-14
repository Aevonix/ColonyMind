"""Publication scanning must report examples without exposing matched values."""

import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "genericity_scan.py"
spec = importlib.util.spec_from_file_location("publication_genericity_scan", SCRIPT)
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


@pytest.mark.parametrize("address", [
    "10." + "0.0.7", "172.16." + "0.7", "192.168." + "0.7", "192.168." + "1.7",
    "10." + "0.0.0/8", "172.16." + "0.0/12", "192.168." + "0.0/16", "100.64." + "0.0/10",
])
def test_private_network_examples_and_prefixes_are_findings(address):
    assert scanner.scan_text(address) == []
    hits = scanner.scan_text(address, strict_publication=True)
    assert len(hits) == 1
    assert hits[0][:2] == (1, 1)
    assert hits[0][2].startswith("private-address")


@pytest.mark.parametrize("value, rule", [
    ("192.168." + "77.5", "private-address"),
    ("/ho" + "me/person-id/notes", "home-path"),
    ("desk.tail" + "123.ts" + ".net", "tailnet-host"),
    ("person" + "@" + "mail-provider.com", "email"),
    ("+1 " + "212 867 5309", "phone"),
])
def test_marked_lines_report_every_rule_as_an_example(value, rule):
    text = value + " # genericity: example\n"
    assert scanner.scan_text(text) == []
    assert scanner.scan_text(text, strict_publication=True) == [(1, 1, rule + " (example)")]


def test_markers_cannot_hide_private_denylist_matches():
    private = [(7, re.compile("synthetic-campaign"))]
    text = "synthetic-campaign # genericity: example"
    assert scanner.scan_text(text, private) == []
    assert scanner.scan_text(text, private, strict_publication=True) == [(1, 1, "private rule 7 (example)")]


def test_cli_strict_publication_fails_without_echoing_examples(tmp_path):
    address = "192.168." + "1.7"
    note = tmp_path / "notes.md"
    note.write_text(address + " # genericity: example\n")
    denylist = tmp_path / "denylist"
    denylist.write_text("synthetic-campaign\n")
    with note.open("a") as stream:
        stream.write("synthetic-campaign # genericity: example\n")
    args = [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--denylist", str(denylist), "notes.md"]
    default = subprocess.run(args, capture_output=True, text=True, check=False)
    assert default.returncode == 0
    strict = subprocess.run(args + ["--strict-publication"], capture_output=True, text=True, check=False)
    assert strict.returncode == 1
    assert "private-address (example)" in strict.stdout
    assert "private rule 1 (example)" in strict.stdout
    assert "2 finding(s)" in strict.stdout
    assert address not in strict.stdout + strict.stderr
    assert "synthetic-campaign" not in strict.stdout + strict.stderr


def test_strict_mode_keeps_reserved_synthetic_identifiers():
    assert scanner.scan_text("sam@example.test /home/user/notes +15555550101", strict_publication=True) == []
