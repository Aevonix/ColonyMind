"""Revision lineage (docs/DECISIONS.md, Calibration and learning): a tier live under one revision keeps its state
when the server serves a learned child of it that has its own record, reaches the revision the tier went live under
through a record at every link, and passes the offline pass (and, for a tier that withholds a label, the adversarial
pass); anything else is demoted as a fingerprint change. Under a learned revision, drift counts a disagreement with
the existing path only when an explicit outcome labels the tier's answer wrong. Offline: every provider is a fake."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from colonymind.instinct import calibration, fit, outcomes, state
from colonymind.instinct.log import Store
from colonymind.instinct.providers import Answer
from colonymind.instinct.registry import Point, Tier
from colonymind.instinct.tiers import Instinct
from test_instinct_fit import KIND, NOW, Varying, answer_many, frozen_set, record, shadowed, table

BARS = {"yes": 0.8, "no": 0.8}


#: The answers each fixture gives before a promotion (the shadow count 1.12 deleted, kept here as a size).
ANSWERS = 200


def fingerprint(point, revision):
    return calibration.fingerprint(point.version, "fast", KIND, "", revision)


class Live:
    """A tier promoted under rev-1 on its evidence, and the provider and engine that answer for it."""

    def __init__(self, tmp_path, point=None, *, provider=None):
        self.point = point or shadowed()
        self.registry = table(self.point)
        self.store = Store(tmp_path / "instinct.db")
        self.provider = provider or Varying(0.1)
        self.engine = Instinct(providers={"local": self.provider}, fast="local", store=self.store,
                               registry=self.registry)
        self.served = 0

    async def promoted(self, frozen=None):
        self.store.put_record(record(self.point, bars=BARS))
        self.frozen("rev-1", frozen)
        await answer_many(self.engine, self.point.name, ANSWERS)
        self.served = ANSWERS
        state.promote(self.store, self.point.name, "fast", now=NOW, registry=self.registry)
        return self

    def frozen(self, revision, items=None, *, flip=0.05):
        items, rates = items(revision) if items is not None else frozen_set(self.point, flip=flip, revision=revision)
        self.store.put_examples(items, set_name=f"frozen-{revision}", flip_rates=rates)

    def child(self, revision, parent, **kwargs):
        made = record(self.point, revision=revision, parent=parent, bars=BARS, **kwargs)
        self.store.put_record(made)
        return made

    async def serves(self, revision, n=1):
        self.provider.revision = revision
        self.engine.monotonic = lambda: 10 ** 12 + self.served         # the engine reads the store again
        await answer_many(self.engine, self.point.name, n, start=1000 + self.served)
        self.served += n

    def check(self):
        return state.check(self.store, now=NOW, registry=self.registry)

    def effective(self):
        return state.effective(self.point, "fast", self.store.states())

    def compared(self, revision, label="no"):
        """The existing path's answer logged beside each settle under ``revision``, as the runner logs it when it
        asks that path to compare (never applied)."""
        for row in self.rows_under(revision):
            if row["tiers"]["fast"].get("outcome") == "settled" and row["answered_by"] == "fast":
                row["tiers"]["deliberate"] = {"state": "live", "outcome": fit.COMPARED, "label": label}
                self.store.write(row)

    def rows_under(self, revision):
        wanted = fingerprint(self.point, revision)
        return [row for row in self.store.rows() if row["tiers"]["fast"].get("fingerprint") == wanted]


# -- 1. a learned child with its record and an offline pass keeps the tier live --------------------------------------

async def test_a_learned_child_with_its_record_and_an_offline_pass_keeps_the_tier_live(tmp_path, monkeypatch):
    live = await Live(tmp_path).promoted()
    parent = live.store.records()[0]
    child = live.child("rev-2", "rev-1")
    live.frozen("rev-2")
    await live.serves("rev-2")
    assert live.check() == []                                          # nothing demoted
    assert live.effective() == "live"
    carried, promoted = live.store.transitions()[:2]
    assert promoted["source"] == "promoted by owner"
    assert (carried["from_state"], carried["to_state"]) == ("live", "live")
    assert carried["source"].startswith(state.CARRIED)
    evidence = carried["evidence"]
    assert (evidence["record"], evidence["parent"]) == (child.id, parent.id)
    assert evidence["fingerprint"] == child.fingerprint and evidence["reached"] == parent.fingerprint
    assert [link["record"] for link in evidence["lineage"]] == [parent.id]
    assert evidence["frozen"]["all"]["record"] >= evidence["frozen"]["all"]["existing"]
    assert evidence["frozen"]["flip_rate"] == 0.05
    # The shadow evidence is waived: one answer under rev-2, and it was a live one.
    [answered] = live.rows_under("rev-2")
    assert answered["tiers"]["fast"]["state"] == "live" and len(live.rows_under("rev-2")) < ANSWERS
    assert state.promoted_under(live.store, live.point, "fast") == (child.fingerprint,)
    assert live.check() == [] and len(live.store.transitions()) == 2     # carried once, not every night
    from colonymind.reports import instinct
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    # The report's 7-day window is anchored at the fixture's clock, not the wall clock (it aged out on 10-12).
    monkeypatch.setattr(instinct, "now_utc", lambda: NOW)
    text = markdown(document(collect(tmp_path, registry=live.registry)))
    assert "State changes: 1 promotion, 1 carried by lineage" in text and "0 demotions" in text


# -- 2. without its record, the offline pass or the adversarial pass, the child is demoted as today ------------------

@pytest.mark.parametrize("case, why", [
    ("no record", "no calibration record"),
    ("no frozen set", "no offline pass: no frozen set imported"),
    ("flip rate", "no offline pass: flip rate 0.20"),
    ("less accurate", "no offline pass: the frozen sets are less accurate"),
])
async def test_a_child_without_its_record_or_the_offline_pass_is_demoted_as_a_fingerprint_change(tmp_path, case,
                                                                                                    why):
    live = await Live(tmp_path).promoted()
    if case != "no record":
        live.child("rev-2", "rev-1")
    if case == "flip rate":
        live.frozen("rev-2", flip=0.2)
    elif case == "less accurate":
        def wrong(revision):
            items, rates = frozen_set(live.point, revision=revision)
            return [dataclasses.replace(item, gold="no" if item.gold == "yes" else "yes") for item in items], rates
        live.frozen("rev-2", wrong)
    await live.serves("rev-2")
    [demoted] = live.check()
    assert demoted["state"] == "demoted" and live.effective() == "demoted"
    assert "fingerprint change" in demoted["evidence"]["reason"]
    assert demoted["evidence"]["lineage"]["why"].startswith(why)
    assert [item["to_state"] for item in live.store.transitions()] == ["demoted", "live"]


def asks(name="asked"):
    """A choice point whose fast tier withholds ``approve`` (read at its bar, only ever asked about)."""
    return Point(name=name, type="choice", instructions="What do the words answer?",
                 labels={"approve": "a plain yes", "decline": "a plain no", "none": "no answer"},
                 template='Words: "{text}"', effect="read",
                 tiers=(Tier("fast", may_settle=("decline", "none"), withholds=("approve",)),),
                 fallback=lambda fields, rules: "none")


class Declines:
    kind, model, private, max_state_chars, batch, timeout_ms = KIND, "", True, 1200, False, 250
    types = frozenset({"yes_no", "choice"})

    def __init__(self, revision="rev-1"):
        self.id, self.revision = "local", revision

    async def answer(self, state_, questions):
        return Answer({name: {"approve": 0.005, "decline": 0.005, "none": 0.99} for name in questions},
                      revision=self.revision)


def ask_items(point, adversarial):
    """Frozen items where the record helps, and ``adversarial`` ones whose right label is not a yes."""
    from colonymind.instinct.fit import Example

    def made(revision):
        items = []
        for n in range(40):
            gold = "decline" if n % 2 else "none"
            raw = {"approve": 0.005, "decline": 0.99 if gold == "decline" else 0.005,
                   "none": 0.99 if gold == "none" else 0.005}
            items.append(Example(key=f"frozen-{n}", point=point.name, version=point.version, tier="fast", kind=KIND,
                                 model="", revision=revision, raw=raw, gold=gold, existing="none", subset="frozen",
                                 tags=("repository",) if n < 20 else ("novel",)))
        items += [Example(key=f"adversarial-{n}", point=point.name, version=point.version, tier="fast", kind=KIND,
                          model="", revision=revision, raw={"approve": 0.005, "decline": 0.005, "none": 0.99},
                          gold="none", existing="none", subset="frozen", tags=("adversarial",))
                  for n in range(adversarial)]
        return items, {point.name: 0.02}
    return made


@pytest.mark.parametrize("adversarial, carried", [(state.ADVERSARIAL_MIN, True), (state.ADVERSARIAL_MIN - 1, False)])
async def test_a_withholding_tiers_child_carries_only_on_the_adversarial_pass(tmp_path, adversarial, carried):
    point = asks()
    live = Live(tmp_path, point, provider=Declines())
    live.store.put_record(record(point, bars={"approve": 0.97, "decline": 0.9, "none": 0.9}))
    live.frozen("rev-1", ask_items(point, state.ADVERSARIAL_MIN))
    await answer_many(live.engine, point.name, ANSWERS)
    live.served = ANSWERS
    state.promote(live.store, point.name, "fast", now=NOW, registry=live.registry)
    live.store.put_record(record(point, revision="rev-2", parent="rev-1",
                                 bars={"approve": 0.97, "decline": 0.9, "none": 0.9}))
    live.frozen("rev-2", ask_items(point, adversarial))
    await live.serves("rev-2")
    made = live.check()
    if carried:
        assert made == [] and live.effective() == "live"
        assert live.store.transitions()[0]["evidence"]["adversarial"] == {"approve": {"n": adversarial, "false": 0}}
        return
    [demoted] = made
    assert demoted["evidence"]["lineage"]["why"].startswith(f"no adversarial pass: {adversarial} adversarial")


@pytest.mark.parametrize("by", ["owner", "the tighten rule"])
async def test_a_demotion_made_while_the_carry_reads_its_gates_stands(tmp_path, monkeypatch, by):
    """The owner (or the tighten rule) demotes the tier while the nightly check reads the child's gates: the carry
    never writes live over that demotion."""
    live = await Live(tmp_path).promoted()
    live.child("rev-2", "rev-1")
    live.frozen("rev-2")
    await live.serves("rev-2")
    real = state.offline

    def meanwhile(*args, **kwargs):
        state.demote(live.store, "probe", "fast", reason="not now", now=NOW, by=by, registry=live.registry)
        return real(*args, **kwargs)
    monkeypatch.setattr(state, "offline", meanwhile)
    assert live.check() == []
    assert live.effective() == "demoted"
    newest = live.store.transitions()[0]
    assert newest["to_state"] == "demoted" and newest["source"].startswith(f"demoted by {by}")
    assert not any(item["source"].startswith(state.CARRIED) for item in live.store.transitions())


# -- 3. a parent the tier never went live under, or no parent, is demoted as today ------------------------------------

@pytest.mark.parametrize("parent", ["rev-0", ""])
async def test_a_revision_whose_parent_never_went_live_or_with_no_parent_is_demoted(tmp_path, parent):
    live = await Live(tmp_path).promoted()
    live.child("rev-0", "")
    live.child("rev-2", parent)
    live.frozen("rev-2")
    await live.serves("rev-2")
    [demoted] = live.check()
    assert demoted["evidence"]["parent_revision"] == parent
    why = demoted["evidence"]["lineage"]["why"]
    assert why.startswith("its lineage reaches no revision" if parent else "the revision names no parent")


# -- 4. a chain carries when every link has a record and passes; a missing or failing middle link demotes -----------

@pytest.mark.parametrize("middle, why", [
    ("passes", None),
    ("no record", "its lineage reaches no revision"),
    ("no frozen set", "no offline pass for the link rev-2: no frozen set imported"),
    ("flip rate", "no offline pass for the link rev-2: flip rate 0.20"),
])
async def test_a_grandchild_carries_only_when_every_link_has_a_record_and_passes(tmp_path, middle, why):
    """rev-3 learned from rev-2, learned from rev-1 (the revision the tier went live under): each revision the chain
    walks through keeps the state only as the running one would, on its own record and passes."""
    live = await Live(tmp_path).promoted()
    if middle != "no record":
        middle_record = live.child("rev-2", "rev-1")
    if middle == "passes":
        live.frozen("rev-2")
    elif middle == "flip rate":
        live.frozen("rev-2", flip=0.2)
    grandchild = live.child("rev-3", "rev-2")
    live.frozen("rev-3")
    await live.serves("rev-3")
    made = live.check()
    if why is None:
        assert made == [] and live.effective() == "live"
        evidence = live.store.transitions()[0]["evidence"]
        assert evidence["record"] == grandchild.id and evidence["parent"] == middle_record.id
        first, reached = evidence["lineage"]
        assert first["record"] == middle_record.id and first["frozen"]["flip_rate"] == 0.05
        assert reached["fingerprint"] == evidence["reached"] == fingerprint(live.point, "rev-1")
        assert "frozen" not in reached                      # the revision it went live under passed then
        return
    [demoted] = made
    assert demoted["evidence"]["lineage"]["why"].startswith(why)
    assert live.effective() == "demoted"


async def test_a_withholding_tiers_middle_link_must_pass_the_adversarial_pass_too(tmp_path):
    point = asks()
    bars = {"approve": 0.97, "decline": 0.9, "none": 0.9}
    live = Live(tmp_path, point, provider=Declines())
    live.store.put_record(record(point, bars=bars))
    live.frozen("rev-1", ask_items(point, state.ADVERSARIAL_MIN))
    await answer_many(live.engine, point.name, ANSWERS)
    live.served = ANSWERS
    state.promote(live.store, point.name, "fast", now=NOW, registry=live.registry)
    live.store.put_record(record(point, revision="rev-2", parent="rev-1", bars=bars))
    live.frozen("rev-2", ask_items(point, state.ADVERSARIAL_MIN - 1))
    live.store.put_record(record(point, revision="rev-3", parent="rev-2", bars=bars))
    live.frozen("rev-3", ask_items(point, state.ADVERSARIAL_MIN))
    await live.serves("rev-3")
    [demoted] = live.check()
    assert demoted["evidence"]["lineage"]["why"].startswith("no adversarial pass for the link rev-2: ")


async def test_a_return_to_a_revision_of_the_live_stretch_carries_on_the_same_gates(tmp_path):
    """After a carry to rev-2, a server back on rev-1 (a rolled-back learner) keeps the tier live only on rev-1's
    record and offline pass read again now; a revision outside the stretch with no lineage is demoted."""
    live = await Live(tmp_path).promoted()
    live.child("rev-2", "rev-1")
    live.frozen("rev-2")
    await live.serves("rev-2")
    assert live.check() == []
    await live.serves("rev-1")
    assert live.check() == [] and live.effective() == "live"
    back = live.store.transitions()[0]
    assert back["source"].startswith(state.CARRIED) and back["evidence"]["lineage"] == []
    assert back["evidence"]["reached"] == fingerprint(live.point, "rev-1")
    live.child("rev-9", "")
    live.frozen("rev-9")
    await live.serves("rev-9")
    [demoted] = live.check()
    assert demoted["state"] == "demoted"


# -- 5. drift under a learned revision counts only outcome-labelled disagreements ------------------------------------

async def carried_to_rev_2(tmp_path):
    live = await Live(tmp_path).promoted()
    live.child("rev-2", "rev-1")
    live.frozen("rev-2")
    await live.serves("rev-2")
    assert live.check() == []
    return live


def resolved(store, decision, *, label, point="probe", at=NOW):
    """An explicit outcome a resolver recorded (the tighten rule is not run: that is ``resolve``'s and ``label``'s)."""
    store.put_outcome({"decision": decision, "point": point, "resolver": point, "source": "explicit",
                       "verdict": "wrong", "judged": "yes", "label": label, "at": at.isoformat(), "evidence": {}})


async def test_under_a_learned_revision_only_disagreements_an_outcome_labels_wrong_count_toward_drift(tmp_path):
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    live = await carried_to_rev_2(tmp_path)
    live.provider.p_yes = 0.95                                         # settles yes; the existing path says no
    await live.serves("rev-2", state.DRIFT_WINDOW)
    live.compared("rev-2")
    disagreeing = [row for row in live.rows_under("rev-2") if row["tiers"]["fast"].get("label") == "yes"]
    assert len(disagreeing) == state.DRIFT_WINDOW
    measured = state.agreement(live.store, "probe", "fast", limit=state.DRIFT_WINDOW, registry=live.registry)
    assert measured["unlabelled"] == state.DRIFT_WINDOW and measured["unmeasured"] >= state.DRIFT_WINDOW
    assert measured["disagree"] == 0
    assert live.check() == [] and live.effective() == "live"
    [gap] = state.unmeasured(live.store, registry=live.registry)
    assert gap["unlabelled"] == state.DRIFT_WINDOW
    text = markdown(document(collect(tmp_path, registry=live.registry)))
    assert f"{state.DRIFT_WINDOW} disagreements under a learned revision with no outcome" in text
    # A behavioural outcome is no proof: still unmeasured.
    for row in disagreeing:
        live.store.put_outcome({"decision": row["id"], "point": "probe", "resolver": "reaction",
                                "source": "behavioural", "verdict": "wrong", "judged": "yes", "label": None,
                                "at": NOW.isoformat(), "evidence": {}})
    assert live.check() == []
    # Explicit outcomes that label the tier's yes wrong: the same disagreements now count, and the tier drifts.
    for row in disagreeing:
        resolved(live.store, row["id"], label="no")
    [demoted] = live.check()
    assert "drift" in demoted["evidence"]["reason"]
    assert demoted["evidence"]["agreement"]["disagree"] == state.DRIFT_WINDOW


async def test_a_labelled_disagreement_the_tier_was_right_about_never_counts_against_it(tmp_path):
    live = await carried_to_rev_2(tmp_path)
    live.provider.p_yes = 0.95
    await live.serves("rev-2", state.DRIFT_WINDOW)
    live.compared("rev-2")
    for row in live.rows_under("rev-2"):
        if row["tiers"]["fast"].get("label") == "yes":
            live.store.put_outcome({"decision": row["id"], "point": "probe", "resolver": "probe",
                                    "source": "explicit", "verdict": "right", "judged": "yes", "label": "yes",
                                    "at": NOW.isoformat(), "evidence": {}})
    measured = state.agreement(live.store, "probe", "fast", limit=state.DRIFT_WINDOW, registry=live.registry)
    assert (measured["n"], measured["disagree"], measured["against"], measured["explicit_model_right"]) == (
        state.DRIFT_WINDOW, state.DRIFT_WINDOW, 0, state.DRIFT_WINDOW)
    assert measured["agreement"] == 1.0
    assert live.check() == []


async def test_disagreements_and_what_drift_counts_against_the_tier_are_kept_apart(tmp_path):
    """Under a learned revision, 100 disagreements an explicit outcome labels (the tier right in 99, wrong in 1) are
    100 disagreements with the deciding path, 100 with an outcome, of which drift counts 1 against the tier: the
    agreement with the path and drift's own reading are two numbers, and each is named."""
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    live = await carried_to_rev_2(tmp_path)
    live.provider.p_yes = 0.95
    await live.serves("rev-2", state.DRIFT_WINDOW)
    live.compared("rev-2")
    disagreeing = [row for row in live.rows_under("rev-2") if row["tiers"]["fast"].get("label") == "yes"]
    assert len(disagreeing) == state.DRIFT_WINDOW
    for number, row in enumerate(disagreeing):
        right = number < state.DRIFT_WINDOW - 1
        live.store.put_outcome({"decision": row["id"], "point": "probe", "resolver": "probe", "source": "explicit",
                                "verdict": "right" if right else "wrong", "judged": "yes",
                                "label": "yes" if right else "no", "at": NOW.isoformat(), "evidence": {}})
    measured = state.agreement(live.store, "probe", "fast", limit=state.DRIFT_WINDOW, registry=live.registry)
    assert (measured["n"], measured["disagree"], measured["against"]) == (state.DRIFT_WINDOW, state.DRIFT_WINDOW, 1)
    assert (measured["explicit_model_right"], measured["explicit_existing_right"]) == (state.DRIFT_WINDOW - 1, 1)
    assert measured["agreement"] == 0.99 and measured["agree"] == 0
    assert live.check() == [] and live.effective() == "live"
    text = markdown(document(collect(tmp_path, registry=live.registry)))
    line = next(line for line in text.splitlines() if line.startswith("- fast (live): coverage"))
    answered = len(live.store.rows())                  # the promotion's shadow answers and every one since
    assert f"agreement {round(100 * (answered - state.DRIFT_WINDOW) / answered)}% with the deciding path" in line
    assert f"{state.DRIFT_WINDOW} disagreements, {state.DRIFT_WINDOW} with an outcome: the model right in 99, " \
           "the existing path in 1" in line
    assert f"1 counted against it (drift agreement {round(100 * (answered - 1) / answered)}%)" in line


@pytest.mark.parametrize("point, said, outcome, graded", [
    # The right label known: the answer is right exactly when it is that label.
    (shadowed(), "yes", {"verdict": "wrong", "judged": "yes", "label": "no"}, "wrong"),
    (shadowed(), "yes", {"verdict": "right", "judged": "yes", "label": "yes"}, "right"),
    # Only the wrong label known: at a yes/no point the answer is the other one or that one.
    (shadowed(), "yes", {"verdict": "wrong", "judged": "yes", "label": None}, "wrong"),
    (shadowed(), "yes", {"verdict": "wrong", "judged": "no", "label": None}, "right"),
    # At a choice point another label being wrong says nothing of this one.
    (asks(), "none", {"verdict": "wrong", "judged": "decline", "label": None}, None),
    (asks(), "none", {"verdict": "wrong", "judged": "none", "label": None}, "wrong"),
    (asks(), "none", {"verdict": "right", "judged": "decline", "label": "decline"}, "wrong"),
    (asks(), "none", {"verdict": "wrong", "judged": None, "label": None}, None),
])
def test_what_an_outcome_says_of_a_tiers_answer(point, said, outcome, graded):
    assert state.graded(point, said, outcome) == graded


async def test_an_outcome_that_says_nothing_of_the_tiers_answer_leaves_its_disagreement_unmeasured(tmp_path):
    """At a choice point, an explicit outcome that the existing path's label was wrong does not say the tier's other
    label was right: under a learned revision that disagreement stays unlabelled, never an agreement, in drift and in
    the report alike."""
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    point = asks()
    bars = {"approve": 0.97, "decline": 0.9, "none": 0.9}
    live = Live(tmp_path, point, provider=Declines())
    live.store.put_record(record(point, bars=bars))
    live.frozen("rev-1", ask_items(point, state.ADVERSARIAL_MIN))
    await answer_many(live.engine, point.name, ANSWERS)
    live.served = ANSWERS
    state.promote(live.store, point.name, "fast", now=NOW, registry=live.registry)
    live.store.put_record(record(point, revision="rev-2", parent="rev-1", bars=bars))
    live.frozen("rev-2", ask_items(point, state.ADVERSARIAL_MIN))
    await live.serves("rev-2", 10)
    assert live.check() == [] and live.effective() == "live"
    live.compared("rev-2", label="decline")                            # the tier settled none; the path said decline
    disagreeing = [row for row in live.rows_under("rev-2") if row["tiers"]["fast"].get("label") == "none"]
    assert len(disagreeing) == 10
    for row in disagreeing:
        live.store.put_outcome({"decision": row["id"], "point": point.name, "resolver": point.name,
                                "source": "explicit", "verdict": "wrong", "judged": "decline", "label": None,
                                "at": NOW.isoformat(), "evidence": {}})
    measured = state.agreement(live.store, point.name, "fast", fingerprint=fingerprint(point, "rev-2"),
                               registry=live.registry)
    assert (measured["n"], measured["disagree"], measured["unlabelled"]) == (0, 0, 10)
    text = markdown(document(collect(tmp_path, registry=live.registry)))
    assert "10 disagreements under a learned revision with no outcome" in text


async def test_a_revision_without_a_parent_drifts_on_unlabelled_disagreements_as_before(tmp_path):
    live = await Live(tmp_path).promoted()
    live.provider.p_yes = 0.95
    await live.serves("rev-1", state.DRIFT_WINDOW)
    live.compared("rev-1")
    [demoted] = live.check()
    assert "drift" in demoted["evidence"]["reason"] and demoted["evidence"]["agreement"]["unlabelled"] == 0


async def test_a_first_promotion_under_a_learned_revision_records_every_disagreement(tmp_path):
    """Promotion goes on offline evidence (release/v1.12 GL-06): its drift refusal reads exactly what the drift rule
    reads, which under a learned revision counts only labelled disagreements, so unlabelled ones no longer refuse it.
    The promotion's evidence (the replay under the record) still counts every one of them against the tier."""
    live = Live(tmp_path)
    live.store.put_record(record(live.point, revision="rev-2", parent="rev-1", bars=BARS))
    live.frozen("rev-2")
    live.provider.revision = "rev-2"
    await answer_many(live.engine, "probe", ANSWERS)                  # agrees: settles no, as the path does
    live.provider.p_yes = 0.95
    await answer_many(live.engine, "probe", 30, start=ANSWERS)       # disagrees, and nothing labels it
    promoted = state.promote(live.store, "probe", "fast", now=NOW, registry=live.registry)
    answered = promoted["evidence"]["answered"]
    assert answered["n"] == ANSWERS + 30 and answered["against"] == answered["disagree"] >= 25
    assert answered["unlabelled"] == 0
    assert live.effective() == "live"


# -- 8. the tighten rule still demotes a carried tier at once ---------------------------------------------------------

async def test_the_tighten_rule_demotes_a_carried_tier_at_once(tmp_path):
    live = await carried_to_rev_2(tmp_path)
    live.provider.p_yes = 0.95
    await live.serves("rev-2", 2)
    wrong = [row for row in live.rows_under("rev-2") if row["tiers"]["fast"].get("label") == "yes"]
    assert len(wrong) == 2
    outcomes.label(live.store, wrong[0]["id"], "no", now=NOW, registry=live.registry)
    assert live.effective() == "live"
    outcomes.label(live.store, wrong[1]["id"], "no", now=NOW + timedelta(minutes=1), registry=live.registry)
    assert live.effective() == "demoted"
    assert live.store.transitions()[0]["source"].startswith("demoted by the tighten rule")


async def test_a_demotion_between_the_carry_reread_and_write_wins(tmp_path, monkeypatch):
    live = await Live(tmp_path).promoted()
    live.child("rev-2", "rev-1")
    live.frozen("rev-2")
    await live.serves("rev-2")
    real = live.store.set_state

    def meanwhile(*args, **kwargs):
        if kwargs["source"].startswith(state.CARRIED):
            Store(live.store.path).set_state("probe", "fast", "demoted", source="owner", at=NOW)
        return real(*args, **kwargs)

    monkeypatch.setattr(live.store, "set_state", meanwhile)
    assert live.check() == []
    assert live.effective() == "demoted"
    assert not any(change["source"].startswith(state.CARRIED) for change in live.store.transitions())


async def test_nightly_and_cli_count_carries(tmp_path, monkeypatch, capsys):
    import argparse

    from colonymind.instinct import cli

    live = await Live(tmp_path).promoted()
    live.child("rev-2", "rev-1")
    live.frozen("rev-2")
    await live.serves("rev-2")
    summary = fit.nightly(tmp_path, now=NOW, registry=live.registry)
    assert len(summary["carried"]) == 1
    assert fit.nightly(tmp_path, now=NOW, registry=live.registry)["carried"] == []
    monkeypatch.setattr(fit, "nightly", lambda *args, **kwargs: summary)
    cli._learning("fit", argparse.Namespace(sets=[], json=False), tmp_path)
    assert "carried by lineage probe fast" in capsys.readouterr().out
