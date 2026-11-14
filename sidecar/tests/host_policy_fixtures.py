"""Explicit host policy used to gate the move out of framework defaults."""
from pathlib import Path

import pytest
import yaml

from colonymind.config import load_config


HOST_SETTINGS = yaml.safe_load((Path(__file__).parent / "fixtures/host-language-policy.yaml").read_text())
HOST_VALIDATORS = HOST_SETTINGS["mind"]["drift"]["validators"]
STATUS_REPLACEMENTS = HOST_SETTINGS["mind"]["status_replacements"]


@pytest.fixture
def host_config(tmp_path, monkeypatch):
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump(HOST_SETTINGS))
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    return load_config(tmp_path, environ={})
