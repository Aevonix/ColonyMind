"""The upgrade removes retired controls and old init limits after a real backup."""

import pytest
import yaml

from colonymind import init
from colonymind.config import DEFAULTS, load_config
from colonymind.removed_settings import REMOVED_SETTINGS, removed_settings
import test_upgrade_retirement as retirement


@pytest.fixture
def instance(tmp_path, monkeypatch):
    # This fixture never opens a listener or contacts a service.
    monkeypatch.setattr(retirement, "free_port", lambda: DEFAULTS["sidecar"]["port"])
    monkeypatch.setattr(init, "running_outside_service", lambda cfg: None)
    return retirement.instance.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("operator", [False, True])
def test_upgrade_hands_settings_removes_only_old_init_values(instance, operator, capsys):
    path = instance.home / "colonymind.yaml"
    data = yaml.safe_load(path.read_text())
    for key in REMOVED_SETTINGS:
        _, group, field = key.split(".")
        data["hands"].setdefault(group, {})[field] = 1
    for key, value in init.OLD_INIT_HANDS_LIMITS.items():
        _, group, field = key.split(".")
        data["hands"][group][field] = value + int(operator)
    # An unrelated operator setting survives unchanged.
    data["hands"]["max_running"] = 7
    path.write_text(yaml.safe_dump(data))
    original = path.read_bytes()
    assert init.hands_settings_upgrade(instance.home)
    assert path.read_bytes() == original
    assert init.run_upgrade(instance.args) == 0
    output = capsys.readouterr().out
    backups = list(instance.home.glob("backups/*/colonymind.yaml"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert backups[0].parent.name[:1].isdigit()
    rewritten = yaml.safe_load(path.read_text())
    assert not removed_settings(rewritten)
    cfg = load_config(instance.home, environ={})
    assert cfg.get("hands.max_running") == 7
    for key in REMOVED_SETTINGS:
        assert f"removed retired setting {key}" in output
    for key, value in init.OLD_INIT_HANDS_LIMITS.items():
        _, group, field = key.split(".")
        if operator:
            assert rewritten["hands"][group][field] == value + 1
            assert f"removed old init default {key}=" not in output
        else:
            assert field not in rewritten["hands"][group]
            assert cfg.get(key) == DEFAULTS["hands"][group][field]
            assert f"removed old init default {key}={value}" in output
    assert init.hands_settings_upgrade(instance.home) == []


def test_backup_failure_leaves_old_init_settings_untouched(instance, monkeypatch):
    path = instance.home / "colonymind.yaml"
    data = yaml.safe_load(path.read_text())
    data["hands"]["brief"]["recall_tokens"] = 2500
    data["hands"]["cascade"]["max_per_task"] = 2
    path.write_text(yaml.safe_dump(data))
    original = path.read_bytes()

    def failed_backup(home):
        raise OSError("backup unavailable")

    monkeypatch.setattr(init, "backup_instance", failed_backup)
    assert init.run_upgrade(instance.args) == 1
    assert path.read_bytes() == original
