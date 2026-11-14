"""The two home-channel notice budgets are independent and durable."""

from datetime import timedelta

import pytest

from colonymind.hands import wire
from test_hands_controller import c, running


async def test_both_notice_caps_exact_and_duplicate_id_dropped(c):
    p = await running(c)
    await c.execute("DELETE FROM notices")
    await c.store.db.commit()
    for budget, cap in (("routine", 30), ("actionable", 20)):
        accepted = []
        for n in range(cap + 2):
            ident = f"ntc:{p['hand_id']}:{budget}-{n}"
            result = await c.notice(ident, p["hand_id"], budget, f"Update {n}")
            accepted.append(result)
            if result:
                assert await c.notice(ident, p["hand_id"], budget, "Changed retry") == result
            c.test_time[0] += timedelta(seconds=61)
        assert sum(bool(row) for row in accepted) == cap
    rows = await c.notices(wire.NoticesRequest())
    assert len(rows["notices"]) == 50
    first = rows["notices"][0]
    await c.sent(first["id"], wire.NoticeSentRequest(platform_message_id="receipt-1"), "sent")
    assert len((await c.notices(wire.NoticesRequest()))["notices"]) == 49
    c.test_time[0] += timedelta(days=1)
    assert await c.notice("ntc:next-day", p["hand_id"], "routine", "Next day")


async def test_notice_composer_refuses_long_lines_and_dashes(c):
    c.config["notices"]["forbidden_patterns"] = [r"[\u2013\u2014]"]
    p = await running(c)
    for text in ("x" * 201, "bad" + chr(0x2014) + "line", "bad" + chr(0x2013) + "line"):
        with pytest.raises(ValueError):
            await c.notice("ntc:bad", p["hand_id"], "routine", text)
