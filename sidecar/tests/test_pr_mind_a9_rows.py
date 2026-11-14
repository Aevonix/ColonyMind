"""Drive levels stay in memory without redundant persistent rows."""
from colonymind.mind.drives import DriveInputs
from test_mind_consolidate import fx, OWNER


def test_drive_levels_do_not_write_mind_state_rows(fx):
    inputs = DriveInputs(now=fx.now, owner_id=OWNER)
    fx.mind._raise_concerns(inputs, fx.now)
    assert fx.mind.drive_levels
    assert fx.mind.mind_state.items('drive.') == []
