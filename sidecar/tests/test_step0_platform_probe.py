"""Both Hermes discovery shapes retain owner lanes and doctor wildcard expansion."""
from contextlib import redirect_stdout
from enum import Enum
from io import StringIO
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from colonymind import doctor, init


@pytest.fixture(params=["set", "names_aliases"])
def platform_probe(monkeypatch, request):
    class Platform(Enum):
        TELEGRAM = "telegram"

        @classmethod
        def _scan_bundled_plugin_platforms(cls):
            names = {"plugin_lane"}
            return names if request.param == "set" else (names, {"alias_lane": "plugin_lane"})

    config = ModuleType("gateway.config")
    config.Platform = Platform
    monkeypatch.setitem(sys.modules, "gateway.config", config)

    def run(argv, **kwargs):
        output = StringIO()
        with redirect_stdout(output):
            exec(argv[-1], {})
        return SimpleNamespace(stdout=output.getvalue(), returncode=0)

    monkeypatch.setattr(init.subprocess, "run", run)
    return Path("/unused/hermes-python")


def test_platform_probe_accepts_set_and_names_aliases_tuple(platform_probe):
    assert init.hermes_platform_names(platform_probe) == ["cli", "plugin_lane", "telegram"]


@pytest.mark.parametrize("explicit", [None, ["plugin_lane", "selected_lane"]])
def test_init_retains_selected_owner_lanes(platform_probe, explicit):
    selected, notes = init._collect_owner_platforms(
        SimpleNamespace(owner_platform=explicit), {}, ["plugin_lane", "recorded_lane"], True,
        fresh=False, python=platform_probe)
    assert selected == (explicit or ["plugin_lane", "recorded_lane"])
    assert len(notes) == 1
    assert ("selected_lane" if explicit else "recorded_lane") in notes[0]


@pytest.mark.parametrize("granted,expected", [(["*"], ["cli", "plugin_lane", "telegram"]),
                                             (["selected_lane"], ["selected_lane"])])
def test_doctor_wildcard_includes_discovered_platforms(platform_probe, tmp_path, monkeypatch, granted, expected):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({
        "hermes": {"home": str(tmp_path / "hermes")},
        "credentials": {"api-key": {"owner": granted, "contacts": "all"}},
    }))
    monkeypatch.setattr(doctor, "_hermes_python", lambda: platform_probe)
    calls = []

    def facts(python, home, platforms):
        calls.append(platforms)
        return {"backend": "docker", "lanes": {platform: [] for platform in platforms}}

    monkeypatch.setattr(doctor, "owner_lane_facts", facts)
    doctor.check_owner_lanes()
    assert calls == [expected]
