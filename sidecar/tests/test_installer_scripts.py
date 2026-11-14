"""The install proofs preserve verifiable hashes for editable snapshots."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import check_hermes_record as record


@pytest.mark.parametrize("modified", [False, True])
def test_snapshot_still_checks_installed_record(tmp_path, capsys, modified):
    checkout = tmp_path / "workspace"
    checkout.mkdir()
    installed = tmp_path / "site-packages"
    installed.mkdir()
    original = b"editable loader"
    digest = base64.urlsafe_b64encode(hashlib.sha256(original).digest()).decode().rstrip("=")
    (installed / "loader.pth").write_bytes(b"changed" if modified else original)
    files = {"direct_url.json": json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}}),
             "RECORD": f"loader.pth,sha256={digest},15\n"}
    dist = SimpleNamespace(version="0.0.0", read_text=files.get, locate_file=lambda _: installed)
    assert record.check_editable_checkout(dist) == int(modified)
    output = capsys.readouterr().out
    if modified:
        assert "modified: loader.pth" in output
    else:
        assert "1 installed files match RECORD" in output
        # A pass that could not look at the checkout ends with a line that says so.
        assert output.splitlines()[-1].startswith("WARN: Source integrity cannot be verified")


def test_proof_scripts_parse():
    root = Path(__file__).resolve().parents[2]
    for name in ("ci_fresh_install.sh", "ci_recall_probe.sh", "ci_mind_loop.sh"):
        subprocess.run(["bash", "-n", str(root / "scripts" / name)], check=True)
    ast.parse((root / "scripts/check_hermes_record.py").read_text())
