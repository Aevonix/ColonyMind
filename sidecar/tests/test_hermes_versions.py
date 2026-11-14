"""Hermes attachment and doctor share a release gate, including local builds."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest

from colonymind import doctor, init


@pytest.mark.parametrize("version,release,supported", [
    ("0.21.2", (0, 21, 2), False),
    ("0.21.3", (0, 21, 3), True),
    ("0.21.7", (0, 21, 7), True),
    ("0.22.0", (0, 22, 0), True),
    ("0.22.2", (0, 22, 2), True),
    ("0.22.2+local.example", (0, 22, 2), True),
    ("0.22.99+local.example", (0, 22, 99), True),
    ("0.23.0", (0, 23, 0), True),
    ("0.23.1", (0, 23, 1), True),
    ("0.23.1+local.example", (0, 23, 1), True),
    ("0.23.99+local.example", (0, 23, 99), True),
    ("0.24.0", (0, 24, 0), False),
    ("0.24.0+local.example", (0, 24, 0), False),
    ("0.0.0", (0, 0, 0), False),
    ("0.0.0+local.example", (0, 0, 0), False),
    ("", (0, 0, 0), False),
    ("unknown 0.22.2", (0, 0, 0), False),
    ("0.22.2+", (0, 0, 0), False),
])
def test_attachment_and_doctor_agree_on_hermes_release(monkeypatch, version, release, supported):
    python = Path(__file__)
    monkeypatch.setattr(init, "hermes_python_version", lambda _: version)
    monkeypatch.setattr(doctor, "_hermes_python", lambda: python)
    assert init.parse_version(version) == release
    assert init.hermes_version_supported(version) is supported
    if supported:
        assert init.resolve_hermes_python(str(python), None) == (python, version)
    else:
        with pytest.raises(init.InitError):
            init.resolve_hermes_python(str(python), None)
    assert doctor.check_hermes_version().status == (doctor.PASS if supported else doctor.FAIL)


@pytest.mark.parametrize("metadata,stamp,expected", [
    ("0.21.3", None, "0.21.3"),
    ("0.21.3", "0.24.0", "0.21.3"),
    ("0.0.0", "0.21.7", "0.21.7"),
    ("0.0.0", "0.22.2+local.example", "0.22.2+local.example"),
    ("0.0.0", "0.21.2", None),
    ("0.0.0", "0.23.0", "0.23.0"),
    ("0.0.0", "0.23.1", "0.23.1"),
    ("0.0.0", "0.24.0", None),
    ("0.0.0", "0.24.1", None),
    ("0.0.0", None, None),
    ("0.0.0", "unreadable", None),
    ("0.0.0", "0.0.0", None),
])
@pytest.mark.parametrize("layout", ["checkout", "installer", "borrowed-root", "missing-root", "outside-root"])
def test_placeholder_metadata_uses_hermes_stamp(tmp_path, monkeypatch, metadata, stamp, expected, layout):
    home = tmp_path / "home" / ".hermes"
    checkout = home / "hermes-agent"
    if layout == "borrowed-root":
        checkout = tmp_path / "shared-checkout"
    checkout.mkdir(parents=True)
    prefix = checkout / "venv"
    workspace = checkout
    if layout != "checkout":
        # Stock install: the venv imports a snapshot without install-stamp.json.
        # PM records the owning checkout in this exact sibling inputs file.
        state = home / "installs" / hashlib.sha256(str(checkout.resolve()).encode()).hexdigest()[:16]
        generation = state / "environments" / "0123456789abcdef0123456789abcdef"
        prefix = generation / "venv"
        workspace = generation / "workspace"
        workspace.mkdir(parents=True)
        (state / "inputs").mkdir()
        if layout != "missing-root":
            root = tmp_path if layout == "outside-root" else checkout
            (state / "inputs" / ".project-root").write_text(str(root))
        if layout not in {"installer", "borrowed-root"} and metadata == "0.0.0":
            expected = None
    # Unrelated stamps must never rescue a missing/unreadable install stamp.
    for unrelated in (tmp_path, home):
        unrelated.mkdir(parents=True, exist_ok=True)
        (unrelated / "install-stamp.json").write_text('{"baseVersion": "0.21.3"}')
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.touch()
    distribution = workspace / "hermes_agent.egg-info"
    distribution.mkdir()
    (distribution / "PKG-INFO").write_text(f"Name: hermes-agent\nVersion: {metadata}\n")
    package = workspace / "hermes_cli"
    package.mkdir()
    # The old lazy-attribute probe can only see the unstamped workspace.
    (package / "__init__.py").write_text('''
import json
from pathlib import Path
def __getattr__(name):
    if name != "__version__":
        raise AttributeError(name)
    path = Path(__file__).resolve().parents[1] / "install-stamp.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8-sig")).get("baseVersion") or "0.0.0")
    except (OSError, ValueError, AttributeError):
        return "0.0.0"
''')
    if stamp is not None:
        path = checkout / "install-stamp.json"
        if stamp == "unreadable":
            path.mkdir()  # read_text raises OSError regardless of the test user's permissions.
        else:
            path.write_text(json.dumps({"baseVersion": stamp, "commit": "a" * 40,
                                        "updateMechanism": "self"}), encoding="utf-8-sig")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        code = (f"import sys; sys.path.insert(0, {str(workspace)!r}); "
                f"sys.prefix = {str(prefix)!r}; " + command[-1])
        return subprocess.run([sys.executable, "-I", "-B", "-c", code],
                              capture_output=True, text=True, **kwargs)

    monkeypatch.setattr(init, "_run", run)
    monkeypatch.setattr(doctor, "_hermes_python", lambda: python)
    if expected is not None:
        assert init.resolve_hermes_python(str(python), None) == (python, expected)
        result = doctor.check_hermes_version()
        assert result.status == doctor.PASS and expected in result.detail
    else:
        with pytest.raises(init.InitError) as failure:
            init.resolve_hermes_python(str(python), None)
        assert init.SUPPORTED_HERMES_RANGE in str(failure.value)
        result = doctor.check_hermes_version()
        assert result.status == doctor.FAIL and result.detail == str(failure.value)
        if stamp in {"0.21.2", "0.24.0", "0.24.1"} and layout in {"checkout", "installer", "borrowed-root"}:
            assert stamp in result.detail and init.SUPPORTED_HERMES_RANGE in result.detail
            assert f"Hermes {stamp} from Hermes install-stamp.json baseVersion" in result.detail
            assert "is outside the supported range" in result.detail
    if metadata != "0.0.0":
        assert all("hermes_cli" not in command[-1] for command in commands)


@pytest.mark.parametrize("defect", ["commit", "updateMechanism", "zero", "mechanism", "light", "stale-git"])
def test_stamp_provenance_required(tmp_path, monkeypatch, defect):
    root = tmp_path / "hermes"
    (root / "hermes_cli").mkdir(parents=True)
    (root / "hermes_cli/__init__.py").touch()
    stamp = {"baseVersion": "0.22.7", "commit": "a" * 40, "updateMechanism": "self"}
    if defect in {"commit", "updateMechanism"}:
        stamp.pop(defect)
    elif defect == "zero":
        stamp["commit"] = "0" * 40
    elif defect == "mechanism":
        stamp["updateMechanism"] = "invented"
    elif defect == "light":
        stamp["payload"] = "light"
    else:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                        "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)
        stamp["source"] = "git"
    (root / "install-stamp.json").write_text(json.dumps(stamp))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "install-stamp.json").write_text(json.dumps({"baseVersion": "0.22.7", "commit": "b" * 40,
                                                            "updateMechanism": "self"}))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(foreign))

    def run(command, **kwargs):
        if "m.version(" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "0.0.0\n", "")
        code = f"import sys; sys.path.insert(0, {str(root)!r}); " + command[-1]
        return subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, **kwargs)

    monkeypatch.setattr(init, "_run", run)
    with pytest.raises(init.InitError, match="no version could be read from Hermes' install stamp through") as exc:
        init.hermes_python_version(Path(sys.executable))
    assert init.SUPPORTED_HERMES_RANGE in str(exc.value)
