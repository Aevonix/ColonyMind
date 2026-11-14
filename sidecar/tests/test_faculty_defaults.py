"""The faculties' shipped defaults follow the standing rule: a faculty is on unless it has shown harm.

Measured without harm, or not measurable yet (which is never a reason for off): the status block, the owner gate,
the learning rules, wishes and the identity check ship on with the others. Two flags ship off, and neither is a
faculty held back: skills wait on lessons proving themselves (their own measured question), and the feeling rules
are the affect mechanism's alternative arm, which would replace the running feelings if it were on.
"""
from datetime import datetime, timezone

from colonymind.config import DEFAULTS, Config, _validate_context
from colonymind.mind.tick import DEFAULT_FACULTIES, faculties_of
from colonymind.reports import faculties as report
from colonymind.reports.registry import FACULTIES

SHIPPED_OFF = {"skills", "affect_rules"}
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def test_a_fresh_configuration_has_every_faculty_on_but_skills_and_the_feeling_rules():
    shipped = DEFAULTS["mind"]["faculties"]
    assert {name for name, on in shipped.items() if not on} == SHIPPED_OFF
    for name in ("owner_gate", "learning", "working_memory", "wishes", "wishes_named", "drift_check"):
        assert shipped[name] is True, name
    assert DEFAULTS["context"]["status"] is True and "blocks" not in DEFAULTS["context"]
    # The mind reads the same defaults for a flag its configuration leaves out.
    assert DEFAULT_FACULTIES == shipped
    assert faculties_of({}) == shipped
    # A context section that names no status switch keeps the status block on.
    assert _validate_context({})["status"] is True


def test_the_install_exports_the_status_block_and_the_learning_rules_on(tmp_path):
    """The two switches the sidecar reads from its environment follow the defaults on a fresh install."""
    from colonymind.config import apply_environment, load_config
    (tmp_path / "colonymind.yaml").write_text("owner: {contact_id: owner-1}\n")
    exported = apply_environment(load_config(tmp_path), environ={})
    assert exported["COLONYMIND_CONTEXT_STATUS"] == "on" and exported["COLONYMIND_LEARNING"] == "1"
    assert isinstance(load_config(tmp_path), Config)


def test_the_faculty_report_of_a_fresh_instance_says_which_are_on(tmp_path):
    home = tmp_path / "fresh"
    home.mkdir()
    (home / "colonymind.yaml").write_text("owner:\n  contact_id: owner-1\n")
    document = report.document(report.collect(home, now=NOW))
    headings = [block.text for block in document.blocks if block.kind == "heading" and block.level == 2]
    assert headings == [f"{faculty.title}: {'off' if faculty.key in SHIPPED_OFF else 'on'}" for faculty in FACULTIES]
    on = len(FACULTIES) - len(SHIPPED_OFF)
    assert f"{on} faculties are on and 2 are off: skills and the feeling rules." in document.text()
