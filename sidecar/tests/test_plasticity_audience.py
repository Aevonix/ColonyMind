"""Audience (docs/PLASTICITY.md, Audience): a stack's audience is the narrowest audience of its training data; an
owner-trained stack serves owner turns only, and everyone else gets the base. A named exception is an explicit
per-learner declaration with its reason, and a safety point can never take one."""

from __future__ import annotations

import numpy as np
import pytest

from colonymind.plasticity import builds
from colonymind.plasticity.audience import (AUDIENCES, AudienceError, AudienceException, AudiencePolicy, may_serve,
                                           narrowest)
from colonymind.plasticity.objectives import Objective
from colonymind.plasticity.revisions import RevisionStore
from colonymind.plasticity.synthetic import OBJECTIVE


def _objective(points):
    return Objective(name="probe", model="probe_model", kind="supervised_refit", target_type="label",
                     losses={"owner_label": "cross_entropy"}, blend={"owner_label": 1.0}, points=points)


def _serving(tmp_path, audience, *, objective=OBJECTIVE.identity):
    store = RevisionStore(tmp_path, "probe", base="base@1", declared=("threshold",))
    state = {"w": np.ones(2)}
    closure = store.compose(state=builds.state_digest(state), objective=objective, calibration=("c",),
                            dependents={"threshold": "t"}, parent=store.base.id, audience=audience, mode="rebuild")
    store.write(closure, state)
    store.promote(closure.id)
    return store


def test_a_stacks_audience_is_the_narrowest_of_its_training_data():
    assert AUDIENCES == ("owner", "contact", "guest")
    assert narrowest(["guest", "contact", "guest"]) == "contact"
    assert narrowest(["guest", "owner"]) == "owner"
    with pytest.raises(ValueError):
        narrowest([])
    with pytest.raises(ValueError):
        narrowest(["guest", "everyone"])


def test_an_owner_trained_stack_is_never_served_to_a_guest_or_contact(tmp_path):
    store = _serving(tmp_path, "owner")
    policy = AudiencePolicy()
    serving = store.serving()
    assert policy.select("probe", "owner", serving=serving, base=store.base) == serving
    for requester in ("contact", "guest", "someone_unknown", None):
        assert policy.select("probe", requester, serving=serving, base=store.base) == store.base
    assert not may_serve("owner", "contact") and not may_serve("owner", "guest") and may_serve("owner", "owner")
    assert all(may_serve("guest", requester) for requester in AUDIENCES)
    for stack in AUDIENCES:
        assert not may_serve(stack, "someone_unknown") and not may_serve(stack, None)


def test_a_stack_learned_from_contacts_rows_serves_the_owner_only(tmp_path):
    # A contact row may be seen by the owner and the contacts it concerns; the core cannot tell which contact a
    # stack's rows concern, so another contact never reads it.
    assert may_serve("contact", "owner")
    assert not may_serve("contact", "contact") and not may_serve("contact", "guest")
    store = _serving(tmp_path, "contact")
    policy = AudiencePolicy()
    assert policy.select("probe", "owner", serving=store.serving(), base=store.base) == store.serving()
    for requester in ("contact", "guest"):
        assert policy.select("probe", requester, serving=store.serving(), base=store.base) == store.base


def test_a_stack_whose_audience_is_not_known_serves_nobody_but_the_base(tmp_path):
    store = _serving(tmp_path, "everyone")
    policy = AudiencePolicy()
    for requester in AUDIENCES:
        assert policy.select("probe", requester, serving=store.serving(), base=store.base) == store.base


def test_a_named_exception_must_carry_its_reason(tmp_path):
    policy = AudiencePolicy()
    with pytest.raises(AudienceError, match="reason"):
        policy.declare(AudienceException("probe", frozenset({"contact", "guest"}), reason="  "),
                       objective=OBJECTIVE)
    policy.declare(AudienceException("probe", frozenset({"contact"}), reason="only a label leaves the model"),
                   objective=OBJECTIVE)
    store = _serving(tmp_path, "owner")
    serving = store.serving()
    assert policy.select("probe", "contact", serving=serving, base=store.base) == serving
    assert policy.select("probe", "guest", serving=serving, base=store.base) == store.base  # not named
    assert policy.select("other", "contact", serving=serving, base=store.base) == store.base  # per learner


def test_a_named_exception_holds_only_for_the_objective_it_was_declared_with(tmp_path):
    # The exception was reasoned about one objective; a stack trained toward another (a changed blend, new
    # points) is not covered until the learner declares it again.
    policy = AudiencePolicy()
    policy.declare(AudienceException("probe", frozenset({"contact"}), reason="only a label leaves the model"),
                   objective=OBJECTIVE)
    store = _serving(tmp_path, "owner", objective="another-objective")
    assert policy.select("probe", "contact", serving=store.serving(), base=store.base) == store.base


@pytest.mark.parametrize("points", [("opt_out",), ("synthetic_yes_no", "tool_screen"), ()])
def test_a_safety_point_can_never_take_an_exception(points):
    # No declared points means any point, a safety point among them: refused too.
    with pytest.raises(AudienceError, match="safety"):
        AudiencePolicy().declare(AudienceException("probe", frozenset({"guest"}), reason="a reason"),
                                 objective=_objective(points))


def test_an_exception_names_known_audiences_only():
    with pytest.raises(AudienceError):
        AudiencePolicy().declare(AudienceException("probe", frozenset({"everyone"}), reason="a reason"),
                                 objective=OBJECTIVE)


def test_pinning_for_a_requester_loads_only_what_that_requester_may_be_served(tmp_path):
    store = _serving(tmp_path, "owner")
    policy = AudiencePolicy()
    owner = policy.pin(store, "owner")
    guest = policy.pin(store, "guest")
    assert owner.closure == store.serving() and np.array_equal(owner.state["w"], np.ones(2))
    assert guest.closure == store.base and guest.state == {}


def test_every_request_pin_names_its_requester_and_an_owner_stack_pins_the_base_for_anyone_else(tmp_path):
    # No audience-blind read of the serving stack exists: the store's own pin applies the audience rule too.
    store = _serving(tmp_path, "owner")
    with pytest.raises(TypeError):
        store.pin()
    owner = store.pin("owner")
    assert owner.closure == store.serving() and np.array_equal(owner.state["w"], np.ones(2))
    for requester in ("contact", "guest", "someone_unknown", None):
        pinned = store.pin(requester)
        assert pinned.closure == store.base and pinned.state == {}
