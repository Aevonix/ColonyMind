"""The genericity scan: the tracked files name no particular deployment (CONTRIBUTING.md, The genericity rule).

The repository's own rules find the generic shapes of a deployment's details: a real user's home directory, an
address on a private network outside the conventional example subnets, a tailnet host, an email address outside the
project's and the reserved example domains, and a North American phone number outside the fictional 555 exchange. A
deployment's own words (its people, hosts and endpoints) live in a private denylist outside the repository, which the
scan reads when present and never echoes. The planted values below are assembled at run time, so this file carries
none of them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "scripts/genericity_scan.py"

HOME_PATH = "/ho" + "me/jdoe/notes.txt"
MAC_HOME = "/Us" + "ers/jdoe/Library"
PRIVATE_IP = "10." + "20.30.40"
LAN_IP = "192.168." + "77.5"
TAILNET_IP = "100." + "88.12.34"
TAILNET_HOST = "desk.tail" + "0a1b2c.ts" + ".net"
EMAIL = "jdoe" + "@" + "mail-provider.com"
PHONE = "+1 " + "212 867 5309"
DASHED_PHONE = "212-" + "867-5309"
JID = "1212" + "8675309@s.whatsapp.net"


def _repo(repo: Path, files: dict[str, str]) -> Path:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    return repo


def _run(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCAN), "--root", str(repo), *args], capture_output=True, text=True,
                          timeout=120, env={**os.environ, **(env or {})})


def _scan(tmp_path: Path, files: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    """The scan of a new repository holding ``files``; with no ``args``, under an explicit empty denylist, so no
    denylist on this machine takes part."""
    empty = tmp_path / "empty-denylist"
    empty.write_text("", encoding="utf-8")
    return _run(_repo(tmp_path / "repo", files), *(args or ("--denylist", str(empty))))


@pytest.mark.parametrize("value, rule", [
    (HOME_PATH, "home-path"), (MAC_HOME, "home-path"),
    (PRIVATE_IP, "private-address"), (LAN_IP, "private-address"), (TAILNET_IP, "private-address"),
    (TAILNET_HOST, "tailnet-host"), (EMAIL, "email"), ("(mailto:" + EMAIL + ")", "email"),
    (PHONE, "phone"), (DASHED_PHONE, "phone"), (JID, "phone"),
])
def test_a_planted_deployment_detail_fails_the_scan_and_is_never_printed(tmp_path, value, rule):
    done = _scan(tmp_path, {"notes.md": f"line one\nsee {value} for it\n"})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "notes.md:2:" in done.stdout and f": {rule}" in done.stdout
    assert value not in done.stdout + done.stderr


@pytest.mark.parametrize("value", [
    "/home/user/colonymind-twin", "/home/me/.colonymind/api.key", '"$work/home/.hermes"', "~/.hermes/config.yaml",
    "127.0.0.1:8080", "0.0.0.0", "192.0.2.1", "192.0.2.1", "192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12",
    "192.0.2.10", "198.51.100.7", "203.0.113.9", "8.8.8.8",
    "https://github.com/Aevonix/ColonyMind/issues", "sam@example.com", "casey@example.test", "one@example.invalid",
    "https://secret@host.example/model", "http://user:REDACTED_CREDENTIAL@8.8.8.8/x.png", "git@github.com:owner/repo.git",
    "12125550101@s.whatsapp.net", "+1 (555) 000-1234", "+14155552671", "+15559999999",
    "+1 659 555 0117", 'width="527.8722" 909 258.3965', "at 1790386376123 ms", "limit 2147483647",
])
def test_an_example_that_only_looks_like_a_detail_passes(tmp_path, value):
    done = _scan(tmp_path, {"notes.md": f"{value}\n"})
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_marked_example_line_passes(tmp_path):
    done = _scan(tmp_path, {"notes.md": f"route via {LAN_IP}  (genericity: example)\n"})
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_private_denylist_is_read_from_the_environment_and_never_echoed(tmp_path):
    denylist = tmp_path / "private"
    denylist.write_text("# our own words\n\nzebra-?cluster\n\\bquokka\\b\n", encoding="utf-8")
    repo = _repo(tmp_path / "repo", {"a.md": "the Zebra-Cluster node\n", "b.py": "x = 'quokka'\ny = 'quokkas'\n"})
    done = _run(repo, env={"COLONYMIND_GENERICITY_DENYLIST": str(denylist)})
    assert done.returncode == 1, done.stdout + done.stderr
    assert "a.md:1:5: private rule 3" in done.stdout and "b.py:1:6: private rule 4" in done.stdout
    assert "b.py:2" not in done.stdout
    output = (done.stdout + done.stderr).lower()
    assert "zebra" not in output and "quokka" not in output
    empty = tmp_path / "empty"
    empty.write_text("", encoding="utf-8")
    done = _run(repo, "--denylist", str(empty), env={"COLONYMIND_GENERICITY_DENYLIST": str(denylist)})
    assert done.returncode == 0, "an explicit denylist is read in place of the one the environment names"


def test_a_named_denylist_that_cannot_be_read_is_an_error(tmp_path):
    done = _scan(tmp_path, {"a.md": "nothing\n"}, "--denylist", str(tmp_path / "missing"))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "missing" in done.stderr


def test_a_denylist_pattern_that_does_not_compile_is_an_error_that_names_its_line_only(tmp_path):
    denylist = tmp_path / "private"
    denylist.write_text("fine\nsecret-word(\n", encoding="utf-8")
    done = _scan(tmp_path, {"a.md": "nothing\n"}, "--denylist", str(denylist))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "line 2" in done.stderr and "secret-word" not in done.stdout + done.stderr


def _private_denylist() -> list[str]:
    """The deployment's own denylist, when this machine has one: the file ``COLONYMIND_GENERICITY_DENYLIST`` names,
    else the default in the real user's home (the suite runs with a temporary ``HOME``)."""
    named = os.environ.get("COLONYMIND_GENERICITY_DENYLIST")
    if named:
        return ["--denylist", named]
    try:
        import pwd
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        return []
    default = home / ".config" / "colonymind" / "genericity-denylist"
    return ["--denylist", str(default)] if default.is_file() else []


def test_this_repository_names_no_deployment():
    if not (ROOT / ".git").exists():
        pytest.skip("not a checkout")
    done = subprocess.run([sys.executable, str(SCAN), "--root", str(ROOT), *_private_denylist()],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
