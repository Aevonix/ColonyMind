"""The worker profile runs on the model ``mind.worker_model`` names, and never joins the host gateway.

``colonymind upgrade`` rewrites ``profiles/colonymind-act/config.yaml`` from the main config. On a host whose worker
ran on another model than the main one, that moved the worker onto the main model, and it dropped
``gateway.standalone``:
Hermes 0.21.7+ multiplexes every named profile into the host gateway unless the profile opts out.
So the profile always opts out, and ``mind.worker_model`` names the one main-config provider entry the
worker runs on; unset, it follows the main model and its fallbacks as before.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess

import pytest
import yaml

from colonymind import init
from colonymind.config import ConfigError, load_config
from test_init import HERMES_PYTHON, STOCK_CONFIG

CAP = {"max_tokens": 8192, "top_p": 0.95}
MAIN = {**STOCK_CONFIG,
        "model": {"provider": "head", "default": "head-model", "base_url": "http://127.0.0.1:9/v1",
                  "context_length": 1048576},
        "fallback_providers": [{"provider": "custom:spare", "model": "spare-model",
                                "base_url": "http://127.0.0.1:10/v1"}],
        "fallback_model": {"provider": "head", "model": "head-small"},
        "providers": {
            "head": {"base_url": "http://127.0.0.1:9/v1", "default_model": "head-model", "api_key": "head-secret"},
            "flash": {"base_url": "http://127.0.0.1:11/v1", "default_model": "flash-model", "context_length": 262144,
                      "api_mode": "chat_completions", "models": {"flash-model": {"context_length": 262144}}},
            "listed-only": {"base_url": "http://127.0.0.1:12/v1", "models": {"listed-model": {}}},
        },
        "custom_providers": [{"name": "spare", "base_url": "http://127.0.0.1:10/v1", "model": "spare-model"}]}

PROBE = """
import json
from hermes_cli.runtime_provider import resolve_runtime_provider
runtime = resolve_runtime_provider()
print(json.dumps({"base_url": runtime.get("base_url"), "model": runtime.get("model")}))
"""


def _cfg(tmp_path, mind=None):
    home = tmp_path / "colonymind"
    home.mkdir(parents=True, exist_ok=True)
    if mind is not None:
        (home / "colonymind.yaml").write_text(yaml.safe_dump({"mind": mind}))
    return load_config(home, environ={})


def _profile(tmp_path, main, **mind):
    cfg = _cfg(tmp_path, mind or None)
    return init.worker_profile_config(main, cfg, sidecar_url="http://127.0.0.1:7901",
                                      key_file=tmp_path / "colonymind" / "api.key")


def test_the_profile_always_opts_out_of_the_host_gateway(tmp_path):
    for name, mind in (("mirror", {}), ("named", {"worker_model": "flash"})):
        profile = _profile(tmp_path / name, copy.deepcopy(MAIN), **mind)
        assert profile["gateway"] == {"standalone": True}, name
        assert yaml.safe_load(init.render_worker_profile(profile))["gateway"] == {"standalone": True}


def test_unset_follows_the_main_model_and_its_fallbacks(tmp_path):
    main = copy.deepcopy(MAIN)
    profile = _profile(tmp_path, main)
    assert profile["model"] == MAIN["model"]
    assert profile["fallback_providers"] == MAIN["fallback_providers"]
    assert profile["fallback_model"] == MAIN["fallback_model"]
    assert main == MAIN                                                     # the main config is never touched


def test_the_named_entry_replaces_the_main_model_and_its_fallbacks(tmp_path):
    main = copy.deepcopy(MAIN)
    profile = _profile(tmp_path, main, worker_model="flash")
    assert profile["model"] == {"provider": "flash", "default": "flash-model",
                                "base_url": "http://127.0.0.1:11/v1", "context_length": 262144}
    assert profile["fallback_providers"] == [] and profile["fallback_model"] == []
    # Every provider entry still comes along, capped; the worker's own address gets its capped entry.
    assert set(profile["providers"]) == set(MAIN["providers"])
    assert profile["providers"]["flash"]["extra_body"] == CAP
    assert {"name": init.WORKER_PROFILE, "base_url": "http://127.0.0.1:11/v1", "extra_body": CAP} \
        in profile["custom_providers"]
    assert all(entry["base_url"] != "http://127.0.0.1:9/v1" for entry in profile["custom_providers"])
    assert main == MAIN


def test_a_custom_providers_entry_or_a_models_list_names_the_model(tmp_path):
    spare = _profile(tmp_path / "spare", copy.deepcopy(MAIN), worker_model="spare")["model"]
    assert spare == {"provider": "spare", "default": "spare-model", "base_url": "http://127.0.0.1:10/v1"}
    listed = _profile(tmp_path / "listed", copy.deepcopy(MAIN), worker_model="listed-only")["model"]
    assert listed == {"provider": "listed-only", "default": "listed-model", "base_url": "http://127.0.0.1:12/v1"}


def test_a_name_the_main_config_lacks_is_refused(tmp_path):
    with pytest.raises(init.InitError, match="not a providers or custom_providers entry"):
        _profile(tmp_path, copy.deepcopy(MAIN), worker_model="flash-b")
    bare = copy.deepcopy(MAIN)
    bare["providers"]["flash"] = {"base_url": "http://127.0.0.1:11/v1"}
    with pytest.raises(init.InitError, match="no default model"):
        _profile(tmp_path / "bare", bare, worker_model="flash")


@pytest.mark.parametrize("value", ["", "   ", 7, ["flash"], "x" * 129])
def test_the_setting_is_one_name_or_null(tmp_path, value):
    with pytest.raises(ConfigError, match="mind.worker_model"):
        _cfg(tmp_path, {"worker_model": value})
    assert _cfg(tmp_path / "null", {"worker_model": None}).get("mind.worker_model") is None
    assert _cfg(tmp_path / "unset").get("mind.worker_model") is None


def test_stock_hermes_runs_the_worker_on_the_named_entry(tmp_path):
    root = tmp_path / "profiles"
    init.write_worker_profile(root, _profile(tmp_path, copy.deepcopy(MAIN), worker_model="flash"))
    result = subprocess.run([HERMES_PYTHON, "-c", PROBE], capture_output=True, text=True, timeout=180,
                            env={**os.environ, "HERMES_HOME": str(root / init.WORKER_PROFILE)})
    assert result.returncode == 0, result.stdout + result.stderr
    sent = json.loads(result.stdout.strip().splitlines()[-1])
    assert sent["base_url"] == "http://127.0.0.1:11/v1"


def test_an_earlier_profile_is_pending_and_the_rewrite_carries_standalone(tmp_path):
    """What ``colonymind upgrade`` does with it: a pending profile is rewritten whole (``write_worker_profile``)."""
    root = tmp_path / "profiles"
    expected = _profile(tmp_path, copy.deepcopy(MAIN), worker_model="flash")
    earlier = copy.deepcopy(expected)
    earlier.pop("gateway")                                              # the profile an earlier release wrote
    init.write_worker_profile(root, earlier)
    assert init.worker_profile_pending(root, expected)
    assert init.write_worker_profile(root, expected)
    after = yaml.safe_load((root / init.WORKER_PROFILE / "config.yaml").read_text())
    assert after["gateway"] == {"standalone": True} and after["model"]["provider"] == "flash"
    assert not init.worker_profile_pending(root, expected)


def _on_disk(root, profile):
    path = root / init.WORKER_PROFILE / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    return path


def test_a_rewrite_keeps_the_keys_only_the_profile_has_and_generated_keys_win(tmp_path):
    """A hand-tuned profile: an earlier release's profile with ``gateway.standalone``, ``model.max_tokens`` and an
    onboarding flag added by hand. All three survive the rewrite; where the file and the generated profile both
    have a key, the generated value wins, and a list is the generated one whole."""
    root = tmp_path / "profiles"
    expected = _profile(tmp_path, copy.deepcopy(MAIN), worker_model="flash")
    earlier = _profile(tmp_path / "earlier", copy.deepcopy(MAIN))       # the main model and its fallbacks
    earlier.pop("gateway")
    earlier["custom_providers"] = [{"name": "retired", "base_url": "http://127.0.0.1:13/v1"}]
    earlier["model"]["max_tokens"] = 32000                                # added by hand
    earlier["gateway"] = {"standalone": True}                             # added by hand
    earlier["onboarding"] = {"seen": {"tool_progress_prompt": True}}      # added by hand
    path = _on_disk(root, earlier)
    merged, kept = init.merged_worker_profile(root, expected)
    assert kept == ["model.max_tokens", "onboarding.seen.tool_progress_prompt"]
    assert init.worker_profile_pending(root, expected)
    assert init.write_worker_profile(root, expected)
    after = yaml.safe_load(path.read_text())
    assert after == merged
    assert after["gateway"] == {"standalone": True}
    assert after["model"] == {**expected["model"], "max_tokens": 32000}  # provider, default, address: generated
    assert after["onboarding"] == {"seen": {"tool_progress_prompt": True}}
    assert after["custom_providers"] == expected["custom_providers"]     # the retired entry is gone with its list
    assert after["fallback_providers"] == [] and after["fallback_model"] == []   # no fallback of the main model
    assert {key: after[key] for key in expected if key != "model"} == {key: expected[key] for key in expected
                                                                          if key != "model"}
    assert not init.worker_profile_pending(root, expected)
    assert not init.write_worker_profile(root, expected)                 # the next rewrite changes nothing


def test_unset_keeps_the_mirror_and_the_hand_added_keys(tmp_path):
    root = tmp_path / "profiles"
    expected = _profile(tmp_path, copy.deepcopy(MAIN))
    _on_disk(root, {**copy.deepcopy(expected), "onboarding": {"seen": {"tool_progress_prompt": True}}})
    assert not init.worker_profile_pending(root, expected)               # nothing to rewrite, nothing dropped
    merged, kept = init.merged_worker_profile(root, expected)
    assert kept == ["onboarding.seen.tool_progress_prompt"]
    assert {key: merged[key] for key in expected} == expected


def test_the_rewrite_names_the_kept_paths_and_never_their_values(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(MAIN))
    cfg = _cfg(tmp_path, {"worker_model": "flash"})
    _on_disk(init.profiles_root(hermes_home), {"model": {"max_tokens": 31999},
                                               "onboarding": {"seen": {"tool_progress_prompt": "hand-set"}}})
    changes = init._reconcile(cfg, hermes_home, backup_dir=tmp_path / "backups")
    assert f"profiles/{init.WORKER_PROFILE} written; kept the keys ColonyMind does not write: model.max_tokens, " \
           "onboarding.seen.tool_progress_prompt" in changes
    assert "31999" not in str(changes) and "hand-set" not in str(changes)
