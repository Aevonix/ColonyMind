"""Relocation creates new files, including launchers, and leaves the seed intact."""
import hashlib
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/copy_suite_hermes.py"


def snapshot(root):
    return {str(path.relative_to(root)): (
        ("link", str(path.readlink())) if path.is_symlink() else
        ("file", hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mode)
    ) for path in root.rglob("*") if path.is_symlink() or path.is_file()}


def test_copy_relocates_both_launchers_without_overwriting(tmp_path):
    source, target = tmp_path / "seed", tmp_path / "copy"
    subprocess.run([sys._base_executable, "-m", "venv", "--without-pip", str(source)], check=True)
    site = Path(subprocess.check_output([str(source / "bin/python"), "-I", "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    metadata = site / "hermes_agent-0.21.7.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: hermes-agent\nVersion: 0.21.7\n")
    (source / "TEMPLATE-DO-NOT-USE-DIRECTLY").write_text("template\n")
    for name, header in (
        ("direct", f"#!{source}/bin/python\n"),
        ("shell", f"#!/bin/sh\n'''exec' '{source}/bin/python' \"$0\" \"$@\"\n' '''\n"),
    ):
        launcher = source / "bin" / name
        launcher.write_text(header + "import sys; print(sys.prefix)\n")
        launcher.chmod(0o700)
    before = snapshot(source)
    command = [sys.executable, "-I", str(SCRIPT), str(source), str(target), "0.21.7"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert snapshot(source) == before
    assert not (target / "TEMPLATE-DO-NOT-USE-DIRECTLY").exists()
    assert (target / "bin/python").is_symlink()
    assert target.stat().st_mode & 0o777 == 0o700
    for name in ("direct", "shell"):
        assert subprocess.check_output([str(target / "bin" / name)], text=True).strip() == str(target)
    copied = snapshot(target)
    refused = subprocess.run(command, text=True, capture_output=True)
    assert refused.returncode != 0 and "FileExistsError" in refused.stderr
    assert snapshot(target) == copied
    assert snapshot(source) == before
