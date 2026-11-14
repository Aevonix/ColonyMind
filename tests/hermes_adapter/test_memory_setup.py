"""``hermes memory setup`` with ColonyMind: stock setup runs the provider's ``post_setup``, which runs the same
``colonymind init`` as the command line, as a subprocess, and never installs anything into Hermes' environment."""

import json
from pathlib import Path
import sys

import pytest
import yaml
from conftest import ROOT, probe

SETUP = '''
import builtins, json, os, subprocess, sys
def emit(**values):
    print("@@RESULT@@" + json.dumps(values, default=str), flush=True)
ANSWERS = list(json.loads(os.environ["FAKE_ANSWERS"]))
builtins.input = lambda prompt="": ANSWERS.pop(0) if ANSWERS else ""
installs = []
import tools.lazy_deps as lazy_deps
lazy_deps.install_specs = lambda specs, **kw: installs.append(list(specs))
spawned = []
real_run = subprocess.run
def run(command, *args, **kwargs):
    spawned.append([str(part) for part in command])
    return real_run(command, *args, **kwargs)
subprocess.run = run
import contextlib, io
from hermes_cli.memory_setup import cmd_setup_provider
said = io.StringIO()
with contextlib.redirect_stdout(said):
    cmd_setup_provider("colonymind-memory")
log = os.environ["FAKE_LOG"]
calls = open(log).read().splitlines() if os.path.exists(log) else []
emit(installs=installs, spawned=spawned, calls=calls, python=sys.executable, prefix=sys.prefix, said=said.getvalue())
'''

FAKE = """#!/bin/sh
echo "$(basename "$0") $*" >> "$FAKE_LOG"
"""
FAKE_UV = FAKE + """if [ "$1" = tool ] && [ "$2" = install ]; then
  mkdir -p "$HOME/.local/bin"
  cp "$FAKE_COLONYMIND" "$HOME/.local/bin/colonymind"
  chmod +x "$HOME/.local/bin/colonymind"
fi
"""


def _tools(home, *names):
    """A bin directory holding fake ``colonymind``, ``uv`` or ``pipx`` executables that record their arguments."""
    bin_dir = home.root / "bin"
    bin_dir.mkdir(exist_ok=True)
    spare = home.root / "fake-colonymind"
    spare.write_text(FAKE)
    spare.chmod(0o755)
    for name in names:
        path = bin_dir / name
        path.write_text(FAKE_UV if name == "uv" else FAKE)
        path.chmod(0o755)
    return bin_dir, spare


def _setup(home, bin_dir, spare, *, answers=(), extra_path=(), env=None):
    # A private HOME and no tool-dir overrides: setup also looks in ~/.local/bin, where a developer's own
    # sidecar would otherwise stand in for the fakes.
    env = {"PATH": ":".join([str(bin_dir), *map(str, extra_path), "/usr/bin", "/bin"]),
           "HOME": str(home.root), "UV_TOOL_BIN_DIR": "", "PIPX_BIN_DIR": "", "XDG_BIN_HOME": "",
           "FAKE_LOG": str(home.root / "calls.log"), "FAKE_COLONYMIND": str(spare),
           "FAKE_ANSWERS": json.dumps(list(answers)), **(env or {})}
    return probe(SETUP, home, env=env, prelude="")


def _adapter_version():
    return yaml.safe_load((ROOT / "plugins/colonymind-memory/plugin.yaml").read_text())["version"]


def _no_install_into_hermes(result):
    assert result["installs"] == []                           # the manifest declares no pip dependency
    for command in result["spawned"]:
        assert command[0] != result["python"] and not Path(command[0]).is_relative_to(result["prefix"]), command
        assert "pip" not in command, command


def _init_call(home, result, instance=None):
    return (f"colonymind init --home {instance or home.instance} --hermes-home {home.hermes} "
            f"--hermes-python {result['python']}")


def test_setup_runs_colonymind_init_with_the_answers_hermes_knows(home, sidecar):
    bin_dir, spare = _tools(home, "colonymind", "uv")
    result = _setup(home, bin_dir, spare)
    assert result["calls"] == [_init_call(home, result)]
    assert "hermes gateway restart" in result["said"] and "loads ahead of" not in result["said"]
    _no_install_into_hermes(result)


def test_setup_keeps_the_instance_the_profile_is_attached_to(home, sidecar):
    """The instance is the configured key file's directory, as the provider itself reads it: a rerun without
    COLONYMIND_HOME (or with it naming another instance) must not attach the profile to the default one."""
    bin_dir, spare = _tools(home, "colonymind")
    assert home.instance != home.root / ".colonymind"
    result = _setup(home, bin_dir, spare)
    assert result["calls"] == [_init_call(home, result)]
    (home.root / "calls.log").unlink()
    result = _setup(home, bin_dir, spare, env={"COLONYMIND_HOME": str(home.root / "other")})
    assert result["calls"] == [_init_call(home, result)]
    # With no key file configured the instance is COLONYMIND_HOME's, else ~/.colonymind, as for the provider.
    home.write_config(plugins={"enabled": ["colonymind"]})
    (home.root / "calls.log").unlink()
    result = _setup(home, bin_dir, spare, env={"COLONYMIND_HOME": str(home.root / "other")})
    assert result["calls"] == [_init_call(home, result, home.root / "other")]
    (home.root / "calls.log").unlink()
    # No COLONYMIND_HOME at all (the probe sets one by default, so clear it): the default instance under HOME.
    result = _setup(home, bin_dir, spare, env={"COLONYMIND_HOME": ""})
    assert result["calls"] == [_init_call(home, result, home.root / ".colonymind")]


def test_setup_names_a_provider_directory_that_shadows_the_installed_adapter(home, sidecar):
    """Init installs the adapter into Hermes' interpreter (its entry points); a provider directory under the
    profile's plugins is loaded ahead of it, so setup names that directory instead of copying anything there."""
    import shutil
    shutil.copytree(ROOT / "plugins/colonymind-memory", home.hermes / "plugins" / "colonymind-memory")
    bin_dir, spare = _tools(home, "colonymind")
    result = _setup(home, bin_dir, spare)
    assert f"this provider runs from {home.hermes / 'plugins' / 'colonymind-memory'}" in result["said"]
    assert result["calls"] == [_init_call(home, result)]


def test_setup_installs_the_sidecar_as_a_tool_only_when_it_is_missing(home, sidecar):
    bin_dir, spare = _tools(home, "uv")
    result = _setup(home, bin_dir, spare, answers=["y"])
    assert result["calls"] == [
        f"uv tool install --python 3.12 colonymind=={_adapter_version()}",
        _init_call(home, result)]
    _no_install_into_hermes(result)


@pytest.mark.requires_sidecar
def test_setup_never_uses_a_sidecar_inside_hermes_environment(home, sidecar):
    """This environment has the ``colonymind`` command (the test interpreter holds both): it is Hermes' own, so
    setup installs the sidecar as a tool instead of running it."""
    bin_dir, spare = _tools(home, "uv")
    assert (Path(sys.executable).parent / "colonymind").exists()
    result = _setup(home, bin_dir, spare, answers=["y"], extra_path=[Path(sys.executable).parent])
    assert result["calls"][0].startswith("uv tool install") and result["calls"][1].startswith("colonymind init")
    _no_install_into_hermes(result)


def test_setup_without_an_installer_says_how_and_changes_nothing(home, sidecar):
    bin_dir, spare = _tools(home)
    before = (home.hermes / "config.yaml").read_text()
    result = _setup(home, bin_dir, spare)
    assert result["calls"] == []
    assert (home.hermes / "config.yaml").read_text() == before
    _no_install_into_hermes(result)


def test_setup_declined_installs_nothing(home, sidecar):
    bin_dir, spare = _tools(home, "uv", "pipx")
    result = _setup(home, bin_dir, spare, answers=["n"])
    assert result["calls"] == []


def test_the_config_schema_offers_only_the_connection(home, sidecar):
    """No ``contact_id``: who the owner is comes from the instance, never from a setup field."""
    result = probe('''
from plugins.memory import load_memory_provider
provider = load_memory_provider("colonymind-memory")
provider.save_config({"url": "http://127.0.0.1:9", "key_file": "/elsewhere/api.key", "contact_id": "p-99"},
                     os.environ["HERMES_HOME"])
saved = json.loads(open(os.path.join(os.environ["HERMES_HOME"], "colonymind-memory.json")).read())
emit(schema=provider.get_config_schema(), saved=saved)
''', home)
    assert [field["key"] for field in result["schema"]] == ["url", "key_file"]
    assert not any(field.get("secret") for field in result["schema"])
    assert result["saved"] == {"url": "http://127.0.0.1:9", "key_file": "/elsewhere/api.key"}


def test_the_manifest_declares_no_pip_dependency():
    manifest = yaml.safe_load((ROOT / "plugins/colonymind-memory/plugin.yaml").read_text())
    assert "pip_dependencies" not in manifest and "external_dependencies" not in manifest
