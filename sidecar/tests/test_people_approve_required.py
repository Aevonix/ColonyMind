"""A change that widens who gets messaged is never the console's (docs/MIND.md, Asks: console and API answers).

The operator's own call (a credential granted the owner that names no viewer: the CLI, a script with the key) cites
no owner turn. It may narrow at once, and set what widens nothing, but a raised permission, a new or faster cadence
and a merge that gives a record's handles more reach are refused ``approve_required``: the owner states them in
their own chat, where the change is their ask and their own answer applies it. An agent with a shell holds the same
key, so the credential cannot stand in for the owner.
"""

from __future__ import annotations

from test_people_changes import world  # noqa: F401  (the people routes on a real store and a real mind)

APPROVE_REQUIRED = "approve_required"


def _refused(response) -> str:
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["code"] == APPROVE_REQUIRED, detail
    return detail["message"]


async def test_the_operator_cannot_raise_a_permission_and_may_lower_it(world):
    path = f"/v1/mind/people/{world.anna.contact_id}/permission"
    for body in ({"may_contact": "auto"}, {"may_contact": "auto", "by": "cli"}, {"may_contact": "auto", "by": "owner"}):
        message = _refused(await world.client.post(path, json=body))
        assert "chat" in message and "Anna Berg" in message, message
    assert (await world.store.get(world.anna.contact_id)).may_contact == "ask"
    assert world.mind.store.open_ask_codes() == []               # refused, not filed: the owner states it in chat
    same = await world.client.post(path, json={"may_contact": "ask", "by": "cli"})
    assert same.status_code == 200 and same.json()["may_contact"] == "ask"
    lowered = await world.client.post(path, json={"may_contact": "never", "by": "cli"})
    assert lowered.status_code == 200 and lowered.json()["may_contact"] == "never"
    _refused(await world.client.post(path, json={"may_contact": "ask", "by": "cli"}))     # never -> ask widens too


async def test_the_operator_cannot_set_a_new_or_faster_cadence_and_may_slow_or_clear_one(world):
    path = f"/v1/mind/people/{world.anna.contact_id}/cadence"
    _refused(await world.client.post(path, json={"minutes": 60, "by": "cli"}))
    assert (await world.store.get(world.anna.contact_id)).cadence_minutes is None
    await world.store.set_cadence(world.anna.contact_id, 1440, by="owner-ask:ABC")
    _refused(await world.client.post(path, json={"minutes": 60, "by": "cli"}))
    slower = await world.client.post(path, json={"minutes": 2880, "by": "cli"})
    assert slower.status_code == 200 and slower.json()["cadence_minutes"] == 2880
    cleared = await world.client.post(path, json={"minutes": None, "by": "cli"})
    assert cleared.status_code == 200 and cleared.json()["cadence_minutes"] is None


async def test_the_operator_cannot_merge_into_more_reach_and_may_fold_what_widens_nothing(world):
    store = world.store
    shadow = await store.create(display_name="Anna B", trust_tier="regular")
    await store.add_handle(shadow.contact_id, "telegram", "4242", verified=True)
    await store.set_may_contact(world.anna.contact_id, "auto", by="owner-ask:ABC")
    # Anna is auto, the shadow record ask: folded in, its handle would be messaged without asking.
    _refused(await world.client.post("/v1/mind/people/merge", json={
        "keep": world.anna.contact_id, "drop": shadow.contact_id, "by": "cli"}))
    assert await store.get(shadow.contact_id) is not None
    # Into the owner, a record always widens.
    _refused(await world.client.post("/v1/mind/people/merge", json={
        "keep": world.owner.contact_id, "drop": shadow.contact_id, "by": "cli"}))
    # Folding the shadow into a keeper on the same terms widens nothing, and the console may do it.
    await store.set_may_contact(world.anna.contact_id, "ask", by="cli")
    folded = await world.client.post("/v1/mind/people/merge", json={
        "keep": world.anna.contact_id, "drop": shadow.contact_id, "by": "cli"})
    assert folded.status_code == 200, folded.text
    assert folded.json()["dropped"] == shadow.contact_id


async def test_the_owners_session_still_files_the_raise_as_their_ask(world):
    """Unchanged: from the owner's own session a raise is their ask (``/changes``), answered in their chat."""
    reply = await world.client.post("/v1/mind/people/changes", json={
        "viewer": world.owner.contact_id, "operation": "set_permission", "contact_id": "Anna", "value": "auto"})
    assert reply.status_code == 200 and reply.json()["status"] == "asked"
    direct = await world.client.post(f"/v1/mind/people/{world.anna.contact_id}/permission",
                                     json={"may_contact": "auto", "contact_id": world.owner.contact_id})
    assert direct.status_code == 409 and direct.json()["detail"]["code"] == "needs_owner_ask"
