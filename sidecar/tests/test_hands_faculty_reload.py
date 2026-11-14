"""Design 8.2: the hands faculty turns off and on through the instance config, with no restart."""

import os

import pytest
import yaml

from colonymind.hands import controller

from test_hands_controller import c, plan  # noqa: F401  (fixture and helper)


def write(home, value, extra=""):
    path = home / "colonymind.yaml"
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    data.setdefault("mind", {}).setdefault("faculties", {})["hands"] = value
    path.write_text(yaml.safe_dump(data) + extra)
    stat = path.stat()
    # Same-second rewrites of equal size must still count as a change.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


def test_reader_follows_the_file_and_keeps_the_last_value_when_unreadable(tmp_path):
    read = controller.faculty_reader(tmp_path)
    assert read() is True                      # no file: the default
    write(tmp_path, False)
    assert read() is False
    write(tmp_path, True)
    assert read() is True
    (tmp_path / "colonymind.yaml").write_text("mind: [unclosed\n")
    assert read() is True                      # unreadable: last good value, never a flip
    write(tmp_path, False)
    assert read() is False
    (tmp_path / "colonymind.yaml").write_text(": : :\n")
    assert read() is False


@pytest.mark.asyncio
async def test_plan_admission_follows_the_faculty_without_a_new_controller(c, tmp_path):  # noqa: F811
    home = tmp_path / "instance"
    home.mkdir()
    write(home, True)
    c._faculty = controller.faculty_reader(home)
    first = await plan(c, "faculty-on")
    assert first["admit"] == "start", first
    write(home, False)
    off = await plan(c, "faculty-off", goal="A second artifact")
    assert off["admit"] != "start" and off["reason"] == "faculty_off", off
    write(home, True)
    again = await plan(c, "faculty-on-again", goal="A third artifact")
    assert again.get("reason") != "faculty_off", again


def test_setting_enabled_directly_still_wins(tmp_path):
    instance = controller.Controller(tmp_path / "hands.db", faculty=lambda: True)
    instance.enabled = False
    assert instance.enabled is False
