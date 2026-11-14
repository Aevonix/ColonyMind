"""Instinct's learning without training (docs/DECISIONS.md, Calibration and learning): isotonic records, the record
history, the nightly refit with its loosen and tighten rules, the frozen holdout and the canaries, and promotion and
demotion with their evidence. Offline: every provider here is a fake that answers fixed probabilities."""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from colonymind.instinct import calibration, records
from colonymind.instinct.calibration import Record
from colonymind.instinct.log import Store
from colonymind.instinct.providers import Answer
from colonymind.instinct.registry import Point, Tier, register
from colonymind.instinct.tiers import Instinct

KIND = "local-decision.v1"
REVISION = "rev-1"
NOW = datetime(2026, 10, 5, 3, 0, tzinfo=timezone.utc)


class Provider:
    kind, model, private, max_state_chars, batch, timeout_ms = KIND, "", True, 1200, False, 250
    types = frozenset({"yes_no", "choice"})

    def __init__(self, answers, *, revision=REVISION):
        self.id, self.answers, self.revision = "local", answers, revision

    async def answer(self, state, questions):
        return Answer({name: dict(self.answers[name]) for name in questions}, revision=self.revision)


def yes_no(name="probe", *, state="live", floors=None):
    return Point(name=name, type="yes_no", instructions="Is it so?", labels={"yes": "it is so", "no": "it is not so"},
                 template='Words: "{text}"', effect="filter", literal=True,
                 tiers=(Tier("fast", may_settle=("yes", "no"), state=state),), floors=floors or {})


def choice(name="pick", *, state="shadow"):
    return Point(name=name, type="choice", instructions="Which kind is it?",
                 labels={"a": "the first kind", "b": "the second kind", "none": "neither kind"},
                 template='Words: "{text}"', effect="route", tiers=(Tier("fast", may_settle=("a", "b"), state=state),))


def table(*points):
    out = {}
    for point in points:
        register(point, registry=out)
    return out


def record(point, *, revision=REVISION, parent="", **kwargs):
    kwargs.setdefault("bars", {label: 0.8 for label in point.labels})
    return Record(point=point.name, version=point.version, tier="fast", provider_kind=KIND, model="",
                  revision=revision, parent_revision=parent, source="test", **kwargs)


def test_refit_uses_the_strongest_sound_outcome(tmp_path):
    from colonymind.instinct import fit, outcomes

    class Sources:
        def erased(self, turns):
            return set()

        def withheld(self, turns):
            return set(turns) & {"grading-turn"}

    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    key = record(point)
    store.write({"id": "decision-1", "at": NOW.isoformat(), "point": point.name, "version": point.version,
                 "tiers": {"fast": {"raw": {"yes": 0.9, "no": 0.1}, "fingerprint": key.fingerprint,
                                      "kind": KIND, "model": "", "revision": REVISION}}})
    store.put_outcome({"decision": "decision-1", "point": point.name, "resolver": "owner_verdict",
                       "source": "behavioural", "verdict": "wrong", "judged": "yes", "label": "no",
                       "at": NOW.isoformat(), "evidence": {"restated": "grading-turn"}})
    assert not outcomes.sound(store.outcome_of("decision-1"), store.row("decision-1"), Sources())
    assert fit.examples_from_log(store, registry=table(point), sources=Sources()) == []
    store.put_outcome({"decision": "decision-1", "point": point.name, "resolver": "fallback",
                       "source": "synthetic", "verdict": "right", "judged": "yes", "label": "yes",
                       "at": NOW.isoformat(), "evidence": {}})
    [example] = fit.examples_from_log(store, registry=table(point), sources=Sources())
    assert example.gold == "yes" and example.source == "synthetic"


# -- 1. isotonic records -------------------------------------------------------------------------------------------

def test_an_isotonic_record_reads_each_label_through_its_own_map():
    point = yes_no()
    iso = record(point, method="isotonic", maps={"yes": ((0.3, 0.1), (0.7, 0.5), (1.0, 0.95))})
    assert calibration.calibrated(iso, "yes_no", {"yes": 0.8, "no": 0.2}) == {"yes": 0.95, "no": pytest.approx(0.05)}
    assert calibration.calibrated(iso, "yes_no", {"yes": 0.2, "no": 0.8})["yes"] == 0.1
    assert calibration.calibrated(iso, "yes_no", {"yes": 0.5, "no": 0.5})["yes"] == 0.5
    picked = choice()
    per_label = record(picked, method="isotonic", maps={"a": ((0.5, 0.2), (1.0, 0.9)), "b": ((1.0, 0.3),)})
    read = calibration.calibrated(per_label, "choice", {"a": 0.6, "b": 0.3, "none": 0.1})
    assert read == {"a": 0.9, "b": 0.3, "none": 0.1}          # a label with no map keeps its raw probability
    with pytest.raises(ValueError):
        calibration.calibrated(record(point, method="platt"), "yes_no", {"yes": 0.5, "no": 0.5})


async def test_the_runner_settles_on_an_isotonic_record_and_logs_its_method(tmp_path):
    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    iso = record(point, method="isotonic", maps={"yes": ((0.5, 0.2), (1.0, 0.97))}, bars={"yes": 0.9, "no": 0.9})
    engine = Instinct(providers={"local": Provider({"probe": {"yes": 0.7, "no": 0.3}})}, fast="local", store=store,
                      records=[iso], registry=table(point))
    decision = await engine.decide("probe", {"text": "it is"}, {"turn": "t-1"})
    assert decision.applied and decision.label == "yes" and decision.probability == pytest.approx(0.97)
    [row] = store.rows()
    assert row["tiers"]["fast"]["record"]["method"] == "isotonic"


@pytest.mark.parametrize("maintenance", ["refit", "tighten"])
async def test_data_shipped_record_is_used_by_runtime_and_nightly_maintenance(tmp_path, maintenance):
    from colonymind.instinct import REGISTRY, fit, outcomes, state
    from colonymind.instinct.providers import calibration_kind

    [shipped] = records.load(strict=True)
    point = REGISTRY[shipped.point]
    store = Store(tmp_path / "instinct.db")
    provider = Provider({point.name: {"yes": 0.83, "no": 0.17}}, revision=shipped.revision)
    provider.readout = shipped.provider_kind.split("/", 1)[1]
    provider.model = shipped.model
    engine = Instinct(providers={"local": provider}, fast="local", store=store, clock=lambda: NOW)
    engine.set_state(point.name, "fast", "live", source="test", evidence={"fingerprint": shipped.fingerprint})
    assert store.records() == []
    assert state.promoted_under(store, point, "fast") == (shipped.fingerprint,)
    for index in range(2):
        decision = await engine.decide(point.name, {"text": f"Fixture message {index}."},
                                       {"contact": f"contact-{index}", "source": f"turn:source-{index}"})
        assert decision.applied and decision.tier == "fast" and decision.label == "yes"
    await engine.drain()
    assert all(row["tiers"]["fast"]["fingerprint"] == shipped.fingerprint
               and row["tiers"]["fast"]["record"]["source"] == shipped.source for row in store.rows())
    example = fit.Example(key="probe", point=point.name, version=point.version, tier="fast",
                          kind=calibration_kind(provider), model=provider.model, revision=shipped.revision,
                          raw={"yes": 0.83, "no": 0.17}, gold="no", existing="no", source="synthetic")
    assert fit.in_force(store, point, "fast", example) == shipped
    assert fit.record_by_fingerprint(store, shipped.fingerprint) == shipped

    class Sources(outcomes.Sources):
        def contact_audit(self, contact_id):
            if maintenance != "tighten":
                return []
            index = contact_id.split("-")[-1]
            return [
                {"id": f"lower-{index}", "action": "opt_out", "at": NOW,
                 "detail": {"source_ref": f"turn:source-{index}"}},
                {"id": f"reset-{index}", "action": "may_contact_set", "at": NOW,
                 "detail": {"from": "never", "to": "ask"}},
            ]

    if maintenance == "refit":
        store.put_examples([dataclass_replace(example, key=f"fit-{index}") for index in range(fit.MIN_LABELS)],
                           set_name="synthetic", at=NOW)
    summary = fit.nightly(tmp_path, now=NOW, sources=Sources(tmp_path))
    if maintenance == "refit":
        [refitted] = summary["refits"]
        assert refitted["action"] == "skip" and "fewer labels" in refitted["reason"]
        assert store.records() == []
        assert state.effective(point, "fast", store.states()) == "demoted"
        assert store.transitions()[0]["evidence"]["current_labels"] == shipped.n
        assert fit.in_force(store, point, "fast", example) == shipped
        return
    [local] = store.records()
    assert local.fingerprint == shipped.fingerprint and local.id != shipped.id
    assert summary["resolved"][point.name]["wrong"] == 2
    assert local.bars["yes"] > shipped.bars["yes"]
    assert state.effective(point, "fast", store.states()) == "demoted"
    assert fit.in_force(store, point, "fast", example) == local
    assert fit.record_by_fingerprint(store, shipped.fingerprint) == local
    refreshed = Instinct(store=store)
    assert refreshed.record_for(point, "fast", provider, shipped.revision) == local
    assert store.record_history()[0]["replaced"] == shipped.id
    assert store.restore_record(shipped.id) == shipped


def test_a_record_round_trips_with_its_maps_and_its_id_is_its_content():
    point = yes_no()
    iso = record(point, method="isotonic", maps={"yes": ((0.5, 0.2), (1.0, 0.97))}, fitted_at="2026-10-05")
    again = Record.from_dict(json.loads(json.dumps(iso.as_dict())))
    assert again == iso and again.id == iso.id and len(iso.id) == 16
    assert record(point, method="isotonic", maps={"yes": ((1.0, 0.9),)}).id != iso.id


def test_every_record_change_is_kept_and_the_previous_one_can_be_restored(tmp_path):
    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    first = record(point, bars={"yes": 0.9, "no": 0.9}, fitted_at="2026-10-01")
    second = record(point, bars={"yes": 0.97, "no": 0.97}, fitted_at="2026-10-02")
    store.put_record(first, reason="fitted")
    store.put_record(second, reason="tightened")
    assert [item.id for item in store.records()] == [second.id]
    history = store.record_history()
    assert [entry["record"].id for entry in history] == [second.id, first.id]
    assert history[0]["replaced"] == first.id and history[0]["reason"] == "tightened"
    restored = store.restore_record(first.id)
    assert restored == first and [item.id for item in store.records()] == [first.id]
    assert [entry["record"].id for entry in store.record_history()][:1] == [first.id]
    assert store.record_history()[0]["reason"] == f"restored {first.id}"
    with pytest.raises(KeyError):
        store.restore_record("0" * 16)


def test_the_record_a_first_replacement_displaces_can_be_restored(tmp_path, monkeypatch):
    """The predecessor of a first replacement is kept too: a record put in force before the history existed (the
    upgrade's carried records) and the shipped record a first local one overrides."""
    import sqlite3
    point = yes_no()
    older = tmp_path / "older.db"
    carried = dataclass_replace(record(point, bars={"yes": 0.85, "no": 0.85}),
                                source="carried from the 1.11 decisions section")
    with sqlite3.connect(older) as conn:
        conn.execute("CREATE TABLE instinct_records (fingerprint TEXT PRIMARY KEY, point TEXT NOT NULL, tier TEXT "
                     "NOT NULL, record_json TEXT NOT NULL, at TEXT NOT NULL)")
        conn.execute("INSERT INTO instinct_records VALUES (?,?,?,?,?)", (carried.fingerprint, point.name, "fast",
                                                                          json.dumps(carried.as_dict()), "2026-09-01"))
    store = Store(older)
    assert [entry["record"].id for entry in store.record_history()] == [carried.id]
    fitted = dataclass_replace(record(point, bars={"yes": 0.95, "no": 0.95}), source="nightly fit")
    store.put_record(fitted)
    assert store.record_history()[0]["replaced"] == carried.id
    assert store.restore_record(carried.id) == carried and store.records() == [carried]
    # Synthetic shipped records exercise history independently of the measured data bundle.
    shipped = record(point)
    other = record(point, revision="rev-2")
    monkeypatch.setattr(records, "load", lambda: (shipped, other))
    local = Store(tmp_path / "instinct.db")
    local.put_record(dataclass_replace(shipped, bars={"yes": 0.99, "no": 0.99}, source="tightened"))
    history = local.record_history()
    assert [entry["record"].id for entry in history][1:] == [shipped.id] and history[0]["replaced"] == shipped.id
    assert local.restore_record(shipped.id) == shipped and local.records() == [shipped]
    # A shipped record nothing displaced yet is restorable by its id as well.
    assert Store(tmp_path / "third.db").restore_record(other.id) == other


async def test_external_data_record_is_used_archived_restored_and_read_for_lineage(tmp_path, monkeypatch):
    from colonymind.instinct import REGISTRY, fit, state

    point = yes_no("external_probe")
    assert point.name not in REGISTRY
    shipped = record(point, n=71, parent="parent-revision")
    directory = tmp_path / "records"
    directory.mkdir()
    (directory / "external.json").write_text(json.dumps({"records": [
        {**shipped.as_dict(), "fingerprint": shipped.fingerprint}
    ]}))
    monkeypatch.setattr(records, "DIRECTORY", directory)
    assert records.load(strict=True) == (shipped,) and records.shipped() == ()
    store = Store(tmp_path / "instinct.db")
    engine = Instinct(providers={"local": Provider({point.name: {"yes": 0.9, "no": 0.1}})}, fast="local",
                      store=store, registry=table(point))
    decision = await engine.decide(point.name, {"text": "fixture"}, {"turn": "external-turn"})
    await engine.drain()
    assert decision.applied and decision.label == "yes"
    assert fit.record_by_fingerprint(store, shipped.fingerprint) == shipped
    assert state.learned_revisions(store)(shipped.fingerprint)
    # Restoring a shipped record also works before any local replacement archives it.
    assert Store(tmp_path / "restore.db").restore_record(shipped.id) == shipped
    replacement = dataclass_replace(shipped, bars={"yes": 0.85, "no": 0.85}, source="nightly fit")
    store.put_record(replacement)
    assert store.record_history()[0]["replaced"] == shipped.id
    assert store.record_history()[1]["record"] == shipped
    assert fit.record_by_fingerprint(store, shipped.fingerprint) == replacement
    assert store.restore_record(shipped.id) == shipped


# -- Q1. the calibration key admits a parent revision, with no change in what matches ----------------------------

def test_the_key_of_a_record_with_no_parent_is_unchanged():
    fixed = Record(point="probe", version="v", tier="fast", provider_kind=KIND, model="", revision="rev-1")
    expected = "188ff2ec9ffe84d9cf63ea6e66c205504a9371b2ea671decde962058a94b5cdf"
    assert fixed.parent_revision == "" and fixed.fingerprint == expected
    assert calibration.fingerprint("v", "fast", KIND, "", "rev-1") == expected


def test_a_child_record_names_its_parent_but_matches_only_its_own_revision(tmp_path):
    point = yes_no()
    parent = record(point, revision="rev-1")
    child = record(point, revision="rev-2", parent="rev-1")
    assert child.fingerprint == calibration.fingerprint(point.version, "fast", KIND, "", "rev-2")
    assert child.key == {"version": point.version, "tier": "fast", "provider_kind": KIND, "model": "",
                         "revision": "rev-2", "parent_revision": "rev-1"}
    found = lambda revision: calibration.find([parent, child], version=point.version, tier="fast",  # noqa: E731
                                              provider_kind=KIND, model="", revision=revision)
    assert found("rev-2") is child and found("rev-1") is parent
    assert calibration.find([child], version=point.version, tier="fast", provider_kind=KIND, model="",
                            revision="rev-1") is None
    assert calibration.lineage([parent, child], child) == [parent]
    assert calibration.lineage([child], child) == []
    looped = record(point, revision="rev-1", parent="rev-2")
    assert calibration.lineage([looped, child], child) == [looped]          # a loop ends, never recurses
    store = Store(tmp_path / "instinct.db")
    store.put_record(child)
    assert store.records() == [child] and store.records()[0].parent_revision == "rev-1"


async def test_a_learned_revision_never_inherits_its_parents_record(tmp_path):
    """Lineage carries a tier's state, never a record (D3): a child revision is refitted, so a provider serving it
    with only its parent's record abstains."""
    point = yes_no()
    engine = Instinct(providers={"local": Provider({"probe": {"yes": 0.99, "no": 0.01}}, revision="rev-2")},
                      fast="local", records=[record(point, revision="rev-1")], registry=table(point))
    decision = await engine.decide("probe", {"text": "it is"})
    assert not decision.applied and decision.tier == "fallback"


# -- 2. the refit: isotonic from fifty labels, bars on held-out folds, loosen only on evidence -------------------

def examples(point, n, *, seed=7, truth=lambda p: p ** 3, existing="no", source="synthetic", subset="fit",
             tags=(), prefix="x", revision=REVISION):
    """``n`` answers of a yes/no point whose raw probability of yes is uniform and right with ``truth(p)``: an
    overconfident model when ``truth`` bends below the diagonal."""
    from colonymind.instinct.fit import Example
    rng = random.Random(seed)
    out = []
    for index in range(n):
        p = rng.random()
        gold = "yes" if rng.random() < truth(p) else "no"
        out.append(Example(key=f"{prefix}-{index}", point=point.name, version=point.version, tier="fast", kind=KIND,
                           model="", revision=revision, raw={"yes": p, "no": 1 - p}, gold=gold, existing=existing,
                           source=source, subset=subset, tags=tuple(tags)))
    return out


def test_on_a_known_miscalibration_the_isotonic_record_lowers_held_out_ece_and_no_bar_falls_below_its_floor():
    from colonymind.instinct import fit
    point = yes_no(floors={"yes": 0.9, "no": 0.6})
    overconfident = lambda p: 0.5 + (p - 0.5) * 0.4                     # noqa: E731 (says 0.9 where 0.66 holds)
    train = examples(point, 300, seed=1, truth=overconfident)
    held = examples(point, 200, seed=2, prefix="h", truth=overconfident)
    made = fit.candidate(point, "fast", train)
    assert made.method == "isotonic" and made.n == 300 and set(made.maps) == {"yes"}
    raw = fit.ece([(max(e.raw.values()), max(e.raw, key=e.raw.get) == e.gold) for e in held])
    calibrated = fit.ece([(max(read.values()), max(read, key=read.get) == e.gold) for e in held
                          for read in [calibration.calibrated(made, "yes_no", e.raw)]])
    assert calibrated < raw / 2
    assert made.bars["yes"] >= 0.9 and made.bars["no"] >= 0.6
    few = fit.candidate(point, "fast", train[:30])
    assert few.method == "temperature" and few.n == 30 and not few.maps
    assert fit.candidate(point, "fast", train[:fit.MIN_LABELS - 1]) is None


def test_the_bars_are_the_most_accurate_with_the_fallback_and_ties_act_least():
    from colonymind.instinct import fit
    point = yes_no()
    always_right = examples(point, 120, truth=lambda p: 1.0 if p >= 0.5 else 0.0)
    made = fit.candidate(point, "fast", always_right)
    # The existing path already says no, so settling no is a tie: the bar that acts least ("never") stands.
    assert made.bars["yes"] < calibration.NEVER and made.bars["no"] == calibration.NEVER
    assert made.accuracy >= 0.97
    # The existing path is right on every item: the model can only tie, so it never acts.
    tied = [dataclass_replace(e, existing=e.gold) for e in always_right]
    assert fit.candidate(point, "fast", tied).bars == {"yes": calibration.NEVER, "no": calibration.NEVER}


def dataclass_replace(item, **changes):
    import dataclasses
    return dataclasses.replace(item, **changes)


def test_a_refit_that_would_loosen_while_the_frozen_holdout_fails_writes_no_record(tmp_path):
    from colonymind.instinct import fit
    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    good = examples(point, 200, truth=lambda p: 1.0 if p >= 0.5 else 0.0)
    # The frozen items the model gets wrong with confidence, and the existing path right: a fit that acts fails.
    failing = [dataclass_replace(e, gold="no" if e.raw["yes"] >= 0.5 else "yes", existing="no" if e.raw["yes"] >= 0.5
                                 else "yes", subset="frozen", key=f"f-{e.key}")
               for e in examples(point, 40, seed=3, truth=lambda p: 1.0 if p >= 0.5 else 0.0)]
    result = fit.refit(store, point, "fast", good, frozen=failing, now=NOW)
    assert result.action == "skip" and "frozen holdout" in result.reason and store.records() == []
    assert fit.refit(store, point, "fast", good, frozen=[], now=NOW).reason.startswith("no frozen holdout")
    assert store.records() == []
    passing = [dataclass_replace(e, subset="frozen", key=f"p-{e.key}")
               for e in examples(point, 40, seed=4, truth=lambda p: 1.0 if p >= 0.5 else 0.0)]
    written = fit.refit(store, point, "fast", good, frozen=passing, now=NOW)
    assert written.action == "write" and [item.id for item in store.records()] == [written.record.id]
    assert store.record_history()[0]["reason"].startswith("nightly fit")


def test_a_refit_loosens_only_when_it_beats_the_existing_path_and_the_current_record_on_the_same_rows(tmp_path,
                                                                                                     monkeypatch):
    from colonymind.instinct import fit
    point = yes_no()
    rows = examples(point, 200, truth=lambda p: 1.0 if p >= 0.7 else 0.0)
    frozen = [dataclass_replace(e, subset="frozen", key=f"f-{e.key}")
              for e in examples(point, 40, seed=4, truth=lambda p: 1.0 if p >= 0.7 else 0.0)]
    current = record(point, bars={"yes": 0.7, "no": calibration.NEVER})      # right on every row
    loose = record(point, bars={"yes": 0.5, "no": calibration.NEVER}, n=200)

    def fitted(accuracy):
        return lambda point_, tier, examples_: dataclass_replace(loose, accuracy=accuracy)

    store = Store(tmp_path / "instinct.db")
    monkeypatch.setattr(fit, "candidate", fitted(0.4))
    result = fit.refit(store, point, "fast", rows, frozen=frozen, current=current, now=NOW)
    assert result.action == "skip" and "existing path" in result.reason and result.direction == "loosen"
    monkeypatch.setattr(fit, "candidate", fitted(0.85))
    result = fit.refit(store, point, "fast", rows, frozen=frozen, current=current, now=NOW)
    assert result.action == "skip" and "current record" in result.reason
    assert result.evidence["current"] == 1.0 and store.records() == []


def test_a_tightening_refit_is_written_without_the_frozen_holdout(tmp_path, monkeypatch):
    from colonymind.instinct import fit
    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    loose = record(point, bars={"yes": 0.8, "no": 0.8})
    rows = examples(point, 200, truth=lambda p: float(p >= 0.5))
    made = record(point, bars={"yes": 0.85, "no": 0.85}, accuracy=1.0, n=200)
    monkeypatch.setattr(fit, "candidate", lambda *args: made)
    result = fit.refit(store, point, "fast", rows, frozen=[], current=loose, now=NOW)
    assert result.action == "write" and result.direction == "tighten"
    assert all(result.record.bars[label] >= loose.bars[label] for label in ("yes", "no"))
    assert acts_only_where(point, result.record, loose)


@pytest.mark.parametrize("count, tier_state", [(10, "live"), (10, "shadow"), (71, "shadow")])
def test_never_acting_refit_needs_at_least_current_label_count(tmp_path, caplog, count, tier_state):
    from colonymind.instinct import fit, state

    point = yes_no(state=tier_state)
    current = record(point, n=71, accuracy=0.9)
    store = Store(tmp_path / "instinct.db")
    store.put_record(current)
    rows = [fit.Example(key=f"row-{index}", point=point.name, version=point.version, tier="fast", kind=KIND,
                        model="", revision=REVISION, raw={"yes": 0.9, "no": 0.1}, gold="no", existing="no")
            for index in range(count)]
    result = fit.refit(store, point, "fast", rows, now=NOW)
    if count < current.n:
        assert result.action == "skip" and "fewer labels" in result.reason
        assert store.records() == [current] and len(store.record_history()) == 1
        assert "current record is kept" in caplog.text
        assert state.effective(point, "fast", store.states()) == ("demoted" if tier_state == "live" else "shadow")
    else:
        assert result.action == "write" and all(bar == calibration.NEVER for bar in result.record.bars.values())


@pytest.mark.parametrize("tier_state, before, after, allowed", [
    ("live", 0.8, 0.97, False), ("live", 0.8, 0.6, False),
    ("live", 0.8, 0.85, True), ("live", 0.8, 0.75, True),
    ("live", 0.91, 0.93, True), ("live", 0.91, 0.95, False),
    ("shadow", 0.8, 0.97, True),
])
def test_live_refit_bars_move_at_most_one_step(tmp_path, monkeypatch, tier_state, before, after, allowed):
    from colonymind.instinct import fit

    point = yes_no(state=tier_state)
    current = record(point, bars={"yes": before, "no": calibration.NEVER}, n=40, accuracy=1.0)
    made = dataclass_replace(current, bars={"yes": after, "no": calibration.NEVER})
    rows = [dataclass_replace(e, existing=e.gold)
            for e in examples(point, 100, truth=lambda p: float(p >= 0.5))]
    frozen = [dataclass_replace(e, subset="frozen") for e in rows]
    monkeypatch.setattr(fit, "candidate", lambda *args: made)
    store = Store(tmp_path / "instinct.db")
    store.put_record(current)
    result = fit.refit(store, point, "fast", rows, frozen=frozen, now=NOW)
    assert result.action == ("write" if allowed else "skip")
    assert store.records() == ([result.record] if allowed else [current])
    if not allowed:
        assert "one BARS step" in result.reason


@pytest.mark.parametrize("held, stored, same_rows", [(0.9, None, 1.0), (0.9, 0.95, 0.0), (None, 0.95, 0.0)])
def test_tightening_refit_must_match_current_accuracy(tmp_path, monkeypatch, held, stored, same_rows):
    from colonymind.instinct import fit

    point = yes_no()
    current = record(point, bars={"yes": 0.8, "no": calibration.NEVER}, n=40, accuracy=stored)
    made = dataclass_replace(current, bars={"yes": 0.85, "no": calibration.NEVER}, accuracy=held)
    rows = [fit.Example(key=f"row-{index}", point=point.name, version=point.version, tier="fast", kind=KIND,
                        model="", revision=REVISION, raw={"yes": 0.9, "no": 0.1},
                        gold="yes" if same_rows else "no", existing="no") for index in range(40)]
    monkeypatch.setattr(fit, "candidate", lambda *args: made)
    store = Store(tmp_path / "instinct.db")
    store.put_record(current)
    result = fit.refit(store, point, "fast", rows, now=NOW)
    assert result.action == "skip" and result.direction == "tighten" and "held-out accuracy" in result.reason
    assert result.evidence["current"] == same_rows and store.records() == [current]


def acts_only_where(point, new, current, steps=20000):
    """Whether, over a fine grid of raw answers, ``new`` settles only what ``current`` settles the same way."""
    from colonymind.instinct import fit
    tier = point.tier("fast")
    read_new, read_current = fit._reader(point, tier, new), fit._reader(point, tier, current)
    for index in range(steps + 1):
        raw = {"yes": index / steps, "no": 1 - index / steps}
        settled = read_new(raw)
        if settled is not None and read_current(raw) != settled:
            return False
    return True


def test_a_new_calibration_shape_with_higher_bars_is_a_loosening_where_it_acts_more(tmp_path, monkeypatch):
    """Higher bars under a sharper temperature still settle answers the current record left to the existing path
    (raw no 0.77: under 0.8 at T 1, over 0.99 at T 0.25): that needs the frozen holdout like any loosening."""
    from colonymind.instinct import fit
    point = yes_no()
    current = record(point, temperature=1.0, bars={"yes": 0.8, "no": 0.8})
    sharper = record(point, temperature=0.25, bars={"yes": 0.8, "no": 0.99}, n=40)
    raw = {"yes": 0.23, "no": 0.77}
    assert fit._reader(point, point.tier("fast"), current)(raw) is None
    assert fit._reader(point, point.tier("fast"), sharper)(raw) == "no"
    assert not fit.contained(point, point.tier("fast"), sharper, current)
    assert not acts_only_where(point, sharper, current)
    rows = [dataclass_replace(e, existing=e.gold) for e in examples(point, 40, truth=lambda p: 1.0 if p >= .5 else 0.)
            if not 0.2 < e.raw["yes"] < 0.25]                   # no row sits where the two differ
    monkeypatch.setattr(fit, "candidate", lambda *args: dataclass_replace(sharper, accuracy=1.0))
    store = Store(tmp_path / "instinct.db")
    result = fit.refit(store, point, "fast", rows, frozen=[], current=current, now=NOW)
    assert result.action == "skip" and result.direction == "loosen" and "frozen holdout" in result.reason
    assert store.records() == []
    # A softer temperature with the same bars acts only where the current record acts, the same way: a tightening.
    softer = record(point, temperature=2.0, bars={"yes": 0.8, "no": 0.8}, n=40)
    assert fit.contained(point, point.tier("fast"), softer, current) and acts_only_where(point, softer, current)
    monkeypatch.setattr(fit, "candidate", lambda *args: dataclass_replace(softer, accuracy=1.0))
    result = fit.refit(store, point, "fast", rows, frozen=[], current=current, now=NOW)
    assert result.action == "write" and result.direction == "tighten"


def test_raising_one_bar_of_a_yes_no_can_hand_its_answers_to_the_other_label():
    """Bars that overlap (yes at 0.5, no at 0.45) read 0.52 as yes; raising yes to 0.6 reads it as no, an answer
    the current record gives otherwise: not contained, whatever the bars did."""
    from colonymind.instinct import fit
    point = yes_no()
    current = record(point, bars={"yes": 0.5, "no": 0.45})
    raised = record(point, bars={"yes": 0.6, "no": 0.45})
    assert fit._reader(point, point.tier("fast"), current)({"yes": 0.52, "no": 0.48}) == "yes"
    assert fit._reader(point, point.tier("fast"), raised)({"yes": 0.52, "no": 0.48}) == "no"
    assert not fit.contained(point, point.tier("fast"), raised, current)
    picked = choice()
    same = record(picked, bars={"a": 0.8, "b": 0.8})
    assert fit.contained(picked, picked.tier("fast"), record(picked, bars={"a": 0.9, "b": 0.8}), same)
    assert not fit.contained(picked, picked.tier("fast"), record(picked, bars={"a": 0.9, "b": 0.8}, temperature=0.5),
                             same)                              # a choice's shape change is never proved contained


def test_behavioural_labels_only_tighten():
    from colonymind.instinct import fit
    from colonymind.instinct.fit import Example
    point = yes_no()
    base = examples(point, 300, truth=lambda p: p)
    plain = fit.candidate(point, "fast", base)
    # Behavioural wrongs where the model leaned yes: the yes bar rises; behavioural rights never lower a bar.
    rng = random.Random(11)
    wrongs = [Example(key=f"b-{n}", point=point.name, version=point.version, tier="fast", kind=KIND, model="",
                      revision=REVISION, raw={"yes": p, "no": 1 - p}, gold="no", existing="no", source="behavioural")
              for n, p in enumerate(0.5 + 0.3 * rng.random() for _ in range(150))]
    tightened = fit.candidate(point, "fast", base + wrongs)
    assert tightened.bars["yes"] > plain.bars["yes"] and tightened.n == plain.n == 300
    assert tightened.maps == plain.maps                               # they shape no map
    rights = [dataclass_replace(e, source="behavioural", key=f"r-{e.key}") for e in base]
    assert fit.candidate(point, "fast", base + rights).bars == plain.bars


def test_a_canary_scored_perfectly_opens_an_audit_and_writes_no_record(tmp_path):
    from colonymind.instinct import fit
    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    honest = examples(point, 120, truth=lambda p: 1.0 if p >= 0.5 else 0.0)
    frozen = [dataclass_replace(e, subset="frozen", key=f"f-{e.key}")
              for e in examples(point, 40, seed=4, truth=lambda p: 1.0 if p >= 0.5 else 0.0)]
    planted = fit.canaries(point, "fast", KIND, "", REVISION)
    assert planted and all(item.subset == "canary" for item in planted)
    assert fit.refit(store, point, "fast", honest, frozen=frozen, now=NOW).action == "write"
    # The canaries leak into the fit data (as a learned phrase or a careless import would carry them): a fit that
    # reads every one of them as its planted label cannot be honest.
    leaked = honest + [dataclass_replace(item, subset="fit", key=f"{item.key}-{n}", existing="no") for n in range(40)
                       for item in planted if item.gold == "yes"]
    other = Store(tmp_path / "other.db")
    result = fit.refit(other, point, "fast", leaked, frozen=frozen, now=NOW)
    assert result.action == "audit" and other.records() == []
    [audit] = other.audits()
    assert audit["point"] == "probe" and audit["tier"] == "fast" and "canar" in audit["reason"]


@pytest.mark.parametrize("gold, low, high", [("yes", 0.6, 1.0), ("no", 0.0, 0.4)])
def test_honest_one_sided_labels_open_no_audit_and_the_map_never_settles_beyond_what_it_saw(tmp_path, gold, low,
                                                                                             high):
    """Sixty honest answers all right for one label, all at one end: the map is flat over what it saw. Read beyond
    that range it has no evidence, so it settles nothing there, and the canaries (planted beyond it) are not
    settled: no audit, and the record is judged on its merits."""
    from colonymind.instinct import fit
    from colonymind.instinct.fit import Example
    point = yes_no()
    rng = random.Random(3)
    honest = []
    for n in range(60):
        p = low + (high - low) * rng.random()
        honest.append(Example(key=f"x-{n}", point=point.name, version=point.version, tier="fast", kind=KIND, model="",
                              revision=REVISION, raw={"yes": p, "no": 1 - p}, gold=gold,
                              existing="no" if gold == "yes" else "yes"))
    frozen = [dataclass_replace(example, key=f"f-{example.key}", subset="frozen") for example in honest[:10]]
    made = fit.candidate(point, "fast", honest)
    assert made.method == "isotonic" and made.support["yes"][0] >= low and made.support["yes"][1] <= high
    beyond = {"yes": 0.0, "no": 1.0} if gold == "yes" else {"yes": 1.0, "no": 0.0}
    assert fit._reader(point, point.tier("fast"), made)(beyond) is None
    assert fit.canary_leak(point, "fast", made, fit.canaries(point, "fast", KIND, "", REVISION)) is None
    store = Store(tmp_path / "instinct.db")
    result = fit.refit(store, point, "fast", honest, frozen=frozen, now=NOW)
    assert result.action == "write" and store.audits() == []


def test_an_isotonic_map_reads_its_safe_bound_beyond_the_range_it_was_fitted_on():
    point = yes_no()
    iso = record(point, method="isotonic", maps={"yes": ((0.7, 0.4), (1.0, 0.9))}, support={"yes": (0.6, 0.95)})
    assert calibration.calibrated(iso, "yes_no", {"yes": 0.65, "no": 0.35}) == {"yes": 0.4, "no": pytest.approx(0.6)}
    assert calibration.calibrated(iso, "yes_no", {"yes": 0.1, "no": 0.9}) == {"yes": 0.0, "no": pytest.approx(0.6)}
    assert calibration.calibrated(iso, "yes_no", {"yes": 0.99, "no": 0.01}) == {"yes": 0.9, "no": 0.0}
    picked = choice()
    per_label = record(picked, method="isotonic", maps={"a": ((0.5, 0.2), (0.9, 0.9))}, support={"a": (0.3, 0.9)})
    assert calibration.calibrated(per_label, "choice", {"a": 0.1, "b": 0.6, "none": 0.3})["a"] == 0.0
    assert calibration.calibrated(per_label, "choice", {"a": 0.95, "b": 0.03, "none": 0.02})["a"] == 0.9
    again = Record.from_dict(json.loads(json.dumps(iso.as_dict())))
    assert again == iso and again.id == iso.id
    assert "support" not in record(point).as_dict()            # a record without one keeps its id


# -- 6. promotion refuses without its evidence; demotion on drift and on a new revision; every transition kept ----

class Varying(Provider):
    """A fast provider whose probability of yes and revision change between calls."""

    def __init__(self, p_yes=0.1, revision=REVISION):
        super().__init__({}, revision=revision)
        self.p_yes = p_yes

    async def answer(self, state, questions):
        return Answer({name: {"yes": self.p_yes, "no": 1 - self.p_yes} for name in questions},
                      revision=self.revision)


def shadowed(name="probe"):
    """A yes/no point whose fast tier ships in shadow, and whose existing path answers no."""
    return Point(name=name, type="yes_no", instructions="Is it so?", labels={"yes": "it is so", "no": "it is not so"},
                 template='Words: "{text}"', effect="filter", literal=True,
                 tiers=(Tier("fast", may_settle=("yes", "no")),), fallback=lambda fields, rules: "no")


def frozen_set(point, *, flip=0.05, revision=REVISION):
    """A frozen set on which the record helps: the existing path says no, the model is right where it is sure."""
    from colonymind.instinct.fit import Example
    items = []
    for n in range(40):
        p = 0.95 if n % 2 else 0.05
        items.append(Example(key=f"frozen-{n}", point=point.name, version=point.version, tier="fast", kind=KIND,
                             model="", revision=revision, raw={"yes": p, "no": 1 - p}, gold="yes" if p > 0.5 else "no",
                             existing="no", subset="frozen", tags=("repository",) if n < 20 else ("novel",)))
    return items, {point.name: flip}


async def answer_many(engine, point, n, *, start=0):
    for index in range(start, start + n):
        await engine.decide(point, {"text": f"item {index}"}, {"turn": f"t-{index}"})
        await engine.drain()                    # a shadow ask in flight for each: never "busy"


async def test_promote_refuses_without_a_fingerprint_a_record_or_the_offline_pass(tmp_path):
    from colonymind.instinct import calibration, state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    provider = Varying(0.1)
    engine = Instinct(providers={"local": provider}, fast="local", store=store, registry=registry)
    promote = lambda **kw: state.promote(store, "probe", "fast", now=NOW, registry=registry, **kw)  # noqa: E731
    with pytest.raises(state.Refused, match="has not answered yet: name the revision"):
        promote()
    await answer_many(engine, "probe", 10)
    with pytest.raises(state.Refused, match="no calibration record"):
        promote()
    store.put_record(record(point, bars={"yes": 0.8, "no": 0.8}))
    with pytest.raises(state.Refused, match="no offline pass.*no frozen set"):
        promote()
    items, rates = frozen_set(point, flip=0.2)
    store.put_examples(items, set_name="frozen-1", flip_rates=rates)
    with pytest.raises(state.Refused, match="flip rate 0.20"):
        promote()
    other = calibration.fingerprint(point.version, "fast", "local-decision.v1", "", "another-revision")
    with pytest.raises(state.Refused, match="answers under another fingerprint"):
        promote(fingerprint=other)
    assert store.transitions() == [] and state.effective(point, "fast", store.states()) == "shadow"
    store.put_examples(items, set_name="frozen-1", flip_rates={"probe": 0.05})
    done = promote()                          # ten earlier answers: no count of them is asked for
    assert done["state"] == "live" and done["evidence"]["answered"]["n"] == 10
    assert "shadow" not in done["evidence"]


async def test_a_tier_that_never_answered_goes_live_under_the_fingerprint_named(tmp_path):
    """Before its first answer a tier is promoted under the fingerprint the caller names (the CLI builds it from the
    configured provider and the revision it serves); its record and offline pass are that fingerprint's, and the
    nightly check then holds its answers to it."""
    from colonymind.instinct import state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    made = record(point, bars={"yes": 0.8, "no": 0.8})
    store.put_record(made)
    items, rates = frozen_set(point)
    store.put_examples(items, set_name="frozen-1", flip_rates=rates)
    done = state.promote(store, "probe", "fast", now=NOW, registry=registry, fingerprint=made.fingerprint)
    assert done["evidence"]["fingerprint"] == made.fingerprint and done["evidence"]["answered"]["n"] == 0
    assert state.promoted_under(store, point, "fast") == (made.fingerprint,)
    engine = Instinct(providers={"local": Varying(0.1)}, fast="local", store=store, registry=registry)
    engine.monotonic = lambda: 10 ** 12
    await answer_many(engine, "probe", 3)
    assert state.check(store, now=NOW, registry=registry) == []


async def test_promotion_reads_earlier_answers_under_the_record_it_would_put_live(tmp_path):
    """Two hundred answers below the bars of an earlier record agreed only because that record never settled: the
    record put live now settles every one of them against the existing path. The evidence is that record's own
    reading; the drift guard reads what the nightly check reads (what each answer acted on when it was logged), so
    the promotion goes through and the check does not undo it at once."""
    from colonymind.instinct import state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(point, bars={"yes": 0.99, "no": 0.99}))
    engine = Instinct(providers={"local": Varying(0.9)}, fast="local", store=store, registry=registry)
    await answer_many(engine, "probe", 200)
    assert {row["tiers"]["fast"]["outcome"] for row in store.rows()} == {"below_bar"}
    replacement = record(point, bars={"yes": 0.8, "no": 0.8})
    store.put_record(replacement)
    items, rates = frozen_set(point)
    store.put_examples(items, set_name="frozen-1", flip_rates=rates)
    done = state.promote(store, "probe", "fast", now=NOW, registry=registry)
    assert done["evidence"]["answered"]["n"] == 200 and done["evidence"]["answered"]["agreement"] == 0.0
    assert done["evidence"]["answered"]["record"] == replacement.id == done["evidence"]["record"]
    assert state.check(store, now=NOW, registry=registry) == []


async def test_a_promotion_the_drift_rule_would_undo_at_once_is_refused(tmp_path):
    """The guard reads the last ``DRIFT_WINDOW`` compared answers exactly as ``check`` does: 980 agreeing answers and
    then 20 that settle against the existing path would pass an average over all of them and be demoted at the next
    check, so the promotion is refused."""
    from colonymind.instinct import state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(point, bars={"yes": 0.8, "no": 0.8}))
    items, rates = frozen_set(point)
    store.put_examples(items, set_name="frozen-1", flip_rates=rates)
    provider = Varying(0.1)
    engine = Instinct(providers={"local": provider}, fast="local", store=store, registry=registry)
    await answer_many(engine, "probe", 180)
    provider.p_yes = 0.9
    await answer_many(engine, "probe", 20, start=180)
    assert state.agreement(store, "probe", "fast", limit=state.DRIFT_WINDOW, registry=registry)["agreement"] < 0.9
    with pytest.raises(state.Refused, match="over the last 100 compared answers.*drift rule would demote it at once"):
        state.promote(store, "probe", "fast", now=NOW, registry=registry)
    assert store.transitions() == []


async def test_an_explicit_wrong_with_no_right_label_counts_against_the_model(tmp_path):
    from colonymind.instinct import state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(point, bars={"yes": 0.8, "no": 0.8}))
    items, rates = frozen_set(point)
    store.put_examples(items, set_name="frozen-1", flip_rates=rates)
    engine = Instinct(providers={"local": Varying(0.9)}, fast="local", store=store, registry=registry)
    await answer_many(engine, "probe", 3)
    [row, *_] = store.rows()
    store.put_outcome({"decision": row["id"], "point": "probe", "resolver": "a resolver", "source": "explicit",
                       "verdict": "wrong", "judged": "yes", "label": None, "at": NOW.isoformat(), "evidence": {}})
    answered = state.agreement(store, "probe", "fast", registry=registry)
    assert answered["explicit_model_wrong"] == 1
    with pytest.raises(state.Refused, match="favour the existing path"):
        state.promote(store, "probe", "fast", now=NOW, registry=registry)


async def test_explicit_outcomes_that_favour_the_existing_path_refuse_the_promotion(tmp_path):
    from colonymind.instinct import state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(point, bars={"yes": 0.8, "no": 0.8}))
    items, rates = frozen_set(point)
    store.put_examples(items, set_name="frozen-1", flip_rates=rates)
    engine = Instinct(providers={"local": Varying(0.9)}, fast="local", store=store, registry=registry)
    await answer_many(engine, "probe", 3)
    for row in store.rows():                   # the owner's word: the existing path's no was right
        store.put_outcome({"decision": row["id"], "point": "probe", "resolver": "owner", "source": "explicit",
                           "verdict": "right", "judged": "no", "label": "no", "at": NOW.isoformat(), "evidence": {}})
    with pytest.raises(state.Refused, match="favour the existing path"):
        state.promote(store, "probe", "fast", now=NOW, registry=registry)


async def test_a_promotion_on_its_evidence_is_kept_with_it_and_demotion_fires_on_a_new_revision(tmp_path):
    from colonymind.instinct import state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(point, bars={"yes": 0.8, "no": 0.8}))
    items, rates = frozen_set(point)
    store.put_examples(items, set_name="frozen-1", flip_rates=rates)
    provider = Varying(0.1)
    engine = Instinct(providers={"local": provider}, fast="local", store=store, registry=registry)
    await answer_many(engine, "probe", 200)
    done = state.promote(store, "probe", "fast", now=NOW, registry=registry)
    assert done["state"] == "live" and state.effective(point, "fast", store.states()) == "live"
    [transition] = store.transitions()
    assert (transition["from_state"], transition["to_state"], transition["source"]) == (None, "live",
                                                                                         "promoted by owner")
    evidence = transition["evidence"]
    assert evidence["record"] == store.records()[0].id and evidence["answered"]["n"] == 200
    assert evidence["frozen"]["repository"]["record"] >= evidence["frozen"]["repository"]["existing"]
    assert state.check(store, now=NOW, registry=registry) == []          # nothing drifted
    # The server now serves a learned child revision of the promoted one, with its own record but no frozen set
    # imported for it: lineage does not carry it (D3; test_instinct_lineage carries one that passes), so it is
    # demoted as a fingerprint change, with the parent and why lineage did not carry in the evidence.
    store.put_record(record(point, revision="rev-2", parent="rev-1", bars={"yes": 0.8, "no": 0.8}))
    provider.revision = "rev-2"
    engine.monotonic = lambda: 10 ** 12
    await answer_many(engine, "probe", 1, start=500)
    [demoted] = state.check(store, now=NOW, registry=registry)
    assert demoted["state"] == "demoted" and demoted["evidence"]["parent_revision"] == "rev-1"
    assert demoted["evidence"]["lineage"]["why"].startswith("no offline pass: no frozen set")
    assert state.effective(point, "fast", store.states()) == "demoted"
    assert [item["to_state"] for item in store.transitions()] == ["demoted", "live"]
    assert "fingerprint change" in store.transitions()[0]["source"]


async def test_a_live_tier_is_demoted_when_its_agreement_over_the_last_hundred_drifts(tmp_path):
    """A synthetic local record makes owner_verdict settle no (withdraw the verdict), where its existing path
    would have kept the verdict, which asks nothing, so every live answer has its comparison."""
    from colonymind.instinct import REGISTRY, state
    point = REGISTRY["owner_verdict"]
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(point))
    provider = Varying(0.5)
    engine = Instinct(providers={"local": provider}, fast="local", store=store)

    async def verdicts(n, start):
        for index in range(start, start + n):
            await engine.decide(point.name, {"reply": "Done.", "text": f"item {index}"}, {"turn": f"t-{index}"})

    await verdicts(95, 0)                                    # unsure: the verdict stands, as the existing path keeps it
    provider.p_yes = 0.001
    await verdicts(5, 95)                                    # settled no: the verdict withdrawn
    assert {row["answered_by"] for row in store.rows()} == {"fallback", "fast"}
    assert state.check(store, now=NOW) == []                 # 95 of 100 agree
    await verdicts(10, 100)
    [demoted] = state.check(store, now=NOW)
    assert demoted["point"] == point.name and "drift" in demoted["evidence"]["reason"]
    assert demoted["evidence"]["agreement"]["n"] == 100 and demoted["evidence"]["agreement"]["agreement"] == 0.85
    assert store.transitions()[0]["source"].startswith("demoted by the nightly check: drift")
    assert state.effective(point, "fast", store.states()) == "demoted"


async def test_live_settles_with_nothing_to_compare_are_counted_unmeasured_never_agreement(tmp_path):
    """A live owner_send drop skips the existing path (the second opinion). The runner asks it beside the settle,
    only to compare, within the point's hourly budget; beyond the budget nothing says what it would have decided.
    Those answers are unmeasured, said so in the check, the nightly summary and the report, never counted as
    agreement; once the existing path's answer is logged beside them as a comparison, drift is measured."""
    from colonymind.instinct import REGISTRY, fit, state
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    home = tmp_path / "instance"
    home.mkdir()
    store = Store(home / "instinct.db")
    point = REGISTRY["owner_send"]
    live = Record(point="owner_send", version=point.version, tier="fast", provider_kind=KIND, revision=REVISION,
                  bars={"drop": 0.95, "digest": 0.95}, source="fitted")
    store.put_record(live)
    store.set_state("owner_send", "fast", "live", source="promoted by owner",
                    evidence={"fingerprint": live.fingerprint})

    class Drops(Provider):
        async def answer(self, state_, questions):
            return Answer({name: {"send_now": 0.005, "digest": 0.005, "drop": 0.99} for name in questions},
                          revision=self.revision)

    engine = Instinct(providers={"local": Drops({})}, fast="local", store=store)
    judged = []

    async def judge():
        judged.append(1)
        return "send_now"

    fields = {"facts": "", "urgency": "normal", "kind": "finding", "topic": "t", "text": "x", "source": "mind",
              "confidence": 0.99}
    for n in range(100):
        decision = await engine.decide("owner_send", fields, {"intention": f"i-{n}"}, deliberate=judge)
        assert decision.applied and (decision.tier, decision.label) == ("fast", "drop")
    await engine.drain()
    budget = point.budget_per_hour
    assert len(judged) == budget                             # compared beside the first settles, never applied
    measured = state.agreement(store, "owner_send", "fast", limit=state.DRIFT_WINDOW)
    assert (measured["n"], measured["agreement"], measured["unmeasured"]) == (budget, 0.0, 100 - budget)
    assert state.check(store, now=NOW) == []                 # nothing measured is never a reason to demote
    [gap] = state.unmeasured(store)
    assert (gap["point"], gap["tier"], gap["unmeasured"], gap["compared"]) == ("owner_send", "fast", 100 - budget,
                                                                               budget)
    assert fit.nightly(home, now=NOW)["unmeasured"] == [gap]
    text = markdown(document(collect(home)))
    line = next(item for item in text.splitlines() if item.startswith("- fast (live)") and "coverage" in item)
    assert f"{100 - budget} settled with nothing to compare them with" in line and "agreement 100%" not in line
    # The existing path's answer logged beside each (what the runner records when it asks it to compare).
    for row in store.rows():
        row["tiers"]["deliberate"] = {"state": "live", "outcome": fit.COMPARED, "label": "send_now"}
        store.write(row)
    assert state.agreement(store, "owner_send", "fast", limit=state.DRIFT_WINDOW)["agreement"] == 0.0
    [demoted] = state.check(store, now=NOW)
    assert demoted["point"] == "owner_send" and "drift" in demoted["evidence"]["reason"]


async def test_a_tier_the_upgrade_carried_live_is_demoted_when_its_server_serves_another_checkpoint(tmp_path,
                                                                                                 monkeypatch):
    """`colonymind upgrade` carries a 1.11 `enabled: true` as a live state whose transition names a setting, not a
    fingerprint: the tier went live under the records carried with it (and the shipped ones), so a checkpoint no
    measurement covers, even one a nightly refit gave a record, is a fingerprint change."""
    import yaml

    from colonymind import config
    from colonymind.init import upgrade_decisions
    from colonymind.instinct import REGISTRY, state
    for name in ("COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "COLONYMIND_INSTINCT"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    (home / config.CONFIG_FILE).write_text(yaml.safe_dump({"decisions": {"url": "http://127.0.0.1:9", "points": {
        "opt_out": {"enabled": True, "temperature": 0.33}}}}))
    upgrade_decisions(home)
    store = Store(home / "instinct.db")
    point = REGISTRY["opt_out"]
    [carried] = store.transitions(point="opt_out")
    assert carried["to_state"] == "live" and "fingerprint" not in carried["evidence"]
    went_live = state.promoted_under(store, point, "fast")
    assert {record.fingerprint for record in store.records()} <= set(went_live) and went_live
    measured = Instinct(providers={"local": Varying(0.99, revision=calibration.MEASURED_REVISIONS[0])},
                        fast="local", store=store)
    await measured.decide("opt_out", {"text": "words"}, {"contact": "p-02", "source": "turn:c-1"})
    assert state.check(store, now=NOW) == []
    store.put_record(Record(point="opt_out", version=point.version, tier="fast", provider_kind=KIND,
                            revision="another@0f0f", bars={"yes": 0.9, "no": 0.9}, n=60, source="nightly fit"))
    other = Instinct(providers={"local": Varying(0.99, revision="another@0f0f")}, fast="local", store=store)
    decision = await other.decide("opt_out", {"text": "words"}, {"contact": "p-02", "source": "turn:c-2"})
    assert decision.applied                                   # live, on a record nobody promoted
    [demoted] = state.check(store, now=NOW)
    assert demoted["point"] == "opt_out" and "fingerprint change" in demoted["evidence"]["reason"]


async def test_a_live_tier_whose_basis_cannot_be_established_is_demoted_not_left_unchecked(tmp_path):
    """A live state recorded with no fingerprint, at a point with no shipped or carried record: missing evidence
    never switches the fingerprint check off."""
    from colonymind.instinct import state
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    store.set_state("probe", "fast", "live", source="set by hand", evidence={})
    store.put_record(record(point, bars={"yes": 0.8, "no": 0.8}, n=60))
    engine = Instinct(providers={"local": Varying(0.95)}, fast="local", store=store, registry=registry)
    await answer_many(engine, "probe", 1)
    assert state.promoted_under(store, point, "fast") == ()
    [demoted] = state.check(store, now=NOW, registry=registry)
    assert "fingerprint" in demoted["evidence"]["reason"] and demoted["evidence"]["from"] == []


async def test_a_default_live_tier_is_demoted_when_its_server_serves_another_checkpoint(tmp_path):
    from colonymind.instinct import state
    point = dataclass_replace(shadowed(), tiers=(Tier("fast", may_settle=("yes", "no"), state="live"),))
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(point))
    engine = Instinct(providers={"local": Varying(0.5)}, fast="local", store=store, registry=registry)
    await answer_many(engine, point.name, 1)
    assert state.check(store, now=NOW, registry=registry) == []
    other = Instinct(providers={"local": Varying(0.5, revision="another@0f0f")}, fast="local", store=store,
                     registry=registry)
    await answer_many(other, point.name, 1, start=1)
    [demoted] = state.check(store, now=NOW, registry=registry)
    assert demoted["point"] == point.name and state.effective(point, "fast", store.states()) == "demoted"


# -- labelled sets: only a local tier's answers, never a hosted one's; frozen items never fitted on -----------------

def set_document(point, items, **changes):
    document = {"format": "colonymind-instinct-sets/1", "set": "dev-1", "hash": "abc", "tier": "fast",
                "provider": {"kind": KIND, "model": "", "revision": REVISION}, "flip_rates": {point.name: 0.03},
                "items": items}
    document.update(changes)
    return document


def test_a_labelled_set_is_kept_by_reference_and_a_hosted_providers_answers_are_refused(tmp_path):
    from colonymind.instinct import fit
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    items = [{"key": "a-1", "point": "probe", "raw": {"yes": 0.9, "no": 0.1}, "gold": "yes", "existing": "no",
              "subset": "fit", "tags": ["novel"], "fields": {"text": "the flowerpot words"}},
             {"key": "a-2", "point": "probe", "raw": {"yes": 0.2, "no": 0.8}, "gold": "no", "existing": "no",
              "subset": "frozen"},
             {"key": "a-3", "point": "probe", "raw": {"yes": 0.2, "no": 0.8}, "gold": "maybe", "subset": "fit"},
             {"key": "a-4", "point": "probe", "raw": {"yes": 0.2}, "gold": "no", "subset": "fit"},
             {"key": "a-5", "point": "probe", "version": "0000", "raw": {"yes": 0.2, "no": 0.8}, "gold": "no"}]
    done = fit.import_sets(store, set_document(point, items), now=NOW, registry=registry)
    assert done["kept"] == 2 and done["skipped"] == {"label": 1, "answer": 1, "version": 1}
    [kept] = store.examples(subset="fit")
    assert kept.state_hash == store.state_hash('Words: "the flowerpot words"') and kept.tags == ("novel",)
    assert "flowerpot" not in (tmp_path / "instinct.db").read_bytes().decode("utf-8", "replace")
    again = fit.import_sets(store, set_document(point, [{**items[1], "subset": "fit"}]), now=NOW, registry=registry)
    assert again["skipped"] == {"subset": 1}                   # a frozen item is never fitted on
    for provider in ({"kind": "hosted-decision.v1"}, {"kind": KIND, "private": False}):
        with pytest.raises(fit.SetError, match="never labels"):
            fit.import_sets(store, set_document(point, items, provider=provider), now=NOW, registry=registry)
    with pytest.raises(fit.SetError):
        fit.import_sets(store, {"items": items}, now=NOW, registry=registry)


def test_an_imported_items_variants_stay_in_one_fold_through_the_database(tmp_path):
    """Renamed or reordered variants of one item share a group: held out together, never fitted on in one fold and
    scored in another. The group is kept with each answer, and an older table without it still reads."""
    import sqlite3

    from colonymind.instinct import fit
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    items = [{"key": f"v-{n}", "group": f"item-{n % 3}", "point": "probe", "raw": {"yes": 0.9, "no": 0.1},
              "gold": "yes", "existing": "no", "subset": "fit"} for n in range(30)]
    items += [{"key": f"solo-{n}", "point": "probe", "raw": {"yes": 0.2, "no": 0.8}, "gold": "no", "existing": "no",
               "subset": "fit"} for n in range(10)]
    fit.import_sets(store, set_document(point, items), now=NOW, registry=registry)
    kept = store.examples(subset="fit")
    groups = {example.key: example.group for example in kept}
    assert groups["v-0"] == groups["v-3"] == "item-0" and groups["solo-4"] == "solo-4"
    parts = fit.folds(kept)
    for name in ("item-0", "item-1", "item-2"):
        assert sum(any(example.group == name for example in part) for part in parts) == 1
    older = tmp_path / "older.db"
    with sqlite3.connect(older) as conn:
        conn.execute("CREATE TABLE instinct_examples (key TEXT NOT NULL, fingerprint TEXT NOT NULL, point TEXT NOT "
                     "NULL, version TEXT NOT NULL, tier TEXT NOT NULL, kind TEXT NOT NULL, model TEXT NOT NULL, "
                     "revision TEXT NOT NULL, raw_json TEXT NOT NULL, gold TEXT NOT NULL, existing TEXT, subset TEXT "
                     "NOT NULL, tags_json TEXT, state_hash TEXT, set_name TEXT NOT NULL, at TEXT NOT NULL, "
                     "PRIMARY KEY (key, fingerprint))")
        conn.execute("INSERT INTO instinct_examples VALUES ('old-1','fp','probe','v','fast','k','','r',"
                     "'{\"yes\": 0.5, \"no\": 0.5}','yes',NULL,'fit','[]',NULL,'s','2026-10-01')")
    [old] = Store(older).examples()
    assert old.group == "old-1"
    fit.import_sets(Store(older), set_document(point, items[:2]), now=NOW, registry=registry)
    assert {example.group for example in Store(older).examples() if example.key.startswith("v-")} == \
        {"item-0", "item-1"}


# -- the owner's commands ---------------------------------------------------------------------------------------------

def _cli(home, *args):
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-m", "colonymind", "--instance", str(home), "instinct", *args],
                          capture_output=True, text=True, timeout=120)


async def test_the_owners_commands_label_promote_demote_fit_and_restore(tmp_path):
    from colonymind.instinct import REGISTRY
    home = tmp_path / "instance"
    home.mkdir()
    store = Store(home / "instinct.db")
    store.put_record(record(REGISTRY["owner_verdict"]))
    engine = Instinct(providers={"local": Varying(0.02)}, fast="local", store=store)
    decision = await engine.decide("owner_verdict", {"reply": "Done.", "text": "the spare key words"},
                                   {"turn": "t-1"})
    assert decision.applied and decision.label == "no"
    done = _cli(home, "label", decision.id, "yes")
    assert done.returncode == 0 and "corrects" in done.stdout, done.stderr
    assert store.outcome_of(decision.id)["source"] == "explicit"
    bad = _cli(home, "label", decision.id, "maybe")
    assert bad.returncode == 2 and "no label 'maybe'" in bad.stderr
    refused = _cli(home, "promote", "opt_out", "--tier", "fast")
    assert refused.returncode == 2 and "refused" in refused.stderr and "has not answered" in refused.stderr
    demoted = _cli(home, "demote", "owner_verdict", "--tier", "fast", "--reason", "I want to watch it first")
    assert demoted.returncode == 0 and "demoted" in demoted.stdout
    assert store.transitions()[0]["source"] == "demoted by owner: I want to watch it first"
    assert _cli(home, "demote", "owner_verdict", "--tier", "fast").stdout.strip().endswith("unchanged")
    point = REGISTRY["opt_out"]
    items = [{"key": f"o-{n}", "point": "opt_out", "raw": {"yes": 0.9, "no": 0.1}, "gold": "yes", "existing": "no",
              "subset": "fit"} for n in range(12)]
    sets = tmp_path / "sets.json"
    sets.write_text(json.dumps({"format": "colonymind-instinct-sets/1", "set": "dev", "tier": "fast",
                                "provider": {"kind": "local-decision.v1", "revision": "rev-9"}, "items": items}))
    fitted = _cli(home, "fit", "--sets", str(sets), "--json")
    assert fitted.returncode == 0, fitted.stderr
    summary = json.loads(fitted.stdout)
    assert summary["imported"][0]["kept"] == 12
    [refit] = [entry for entry in summary["refits"] if entry["point"] == "opt_out"]
    assert refit["action"] == "skip" and "frozen holdout" in refit["reason"]      # a first record loosens
    first = Record(point="opt_out", version=point.version, tier="fast", provider_kind=KIND, revision="rev-9",
                   bars={"yes": 0.9, "no": 0.9}, source="one")
    second = Record(point="opt_out", version=point.version, tier="fast", provider_kind=KIND, revision="rev-9",
                    bars={"yes": 0.99, "no": 0.99}, source="two")
    store.put_record(first)
    store.put_record(second)
    listed = _cli(home, "records", "--point", "opt_out")
    assert listed.returncode == 0 and first.id in listed.stdout and second.id in listed.stdout
    assert _cli(home, "restore", first.id).returncode == 0
    assert [item.id for item in store.records() if item.point == "opt_out"] == [first.id]
    assert _cli(home, "restore", "0" * 16).returncode == 2
    for output in (done.stdout, listed.stdout, fitted.stdout):
        assert "spare key" not in output
    # Before a tier's first answer the owner names the revision its provider serves; the fingerprint is built from
    # the configured provider, and the gates are that fingerprint's (the local record is found; no frozen set yet).
    served = "rev-before-answer"
    store.put_record(record(point, revision=served))
    off = _cli(home, "promote", "opt_out", "--tier", "fast", "--revision", served)
    assert off.returncode == 2 and "no fast provider configured" in off.stderr, off.stderr   # no url: the tier is off
    (home / "colonymind.yaml").write_text("instinct:\n  providers:\n    local:\n      kind: local-decision.v1\n"
                                         "      url: http://127.0.0.1:9\n  fast: local\n")
    named = _cli(home, "promote", "opt_out", "--tier", "fast", "--revision", served)
    assert named.returncode == 2 and "no offline pass" in named.stderr, named.stderr


# -- the nightly pass runs in the sidecar's daily task ----------------------------------------------------------

async def test_the_daily_maintenance_runs_the_nightly_pass(tmp_path):
    from colonymind.instinct import REGISTRY, maintenance, state
    store = Store(tmp_path / "instinct.db")
    store.put_record(record(REGISTRY["owner_verdict"]))
    engine = Instinct(providers={"local": Varying(0.5, revision="another@0f0f")}, fast="local", store=store)
    await engine.decide("owner_verdict", {"reply": "Done.", "text": "that was wrong"}, {"turn": "t-1"})
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        if len(slept) > 1:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await maintenance(tmp_path, sleep=sleep, first_delay=0)
    assert state.effective(REGISTRY["owner_verdict"], "fast", store.states()) == "demoted"
    assert store.transitions()[0]["source"].startswith("demoted by the nightly check")


# -- the shadow statistics in the report ----------------------------------------------------------------------------

async def test_the_report_shows_each_shadow_tiers_coverage_agreement_disagreements_and_gold_split(tmp_path):
    from colonymind.instinct import fit, outcomes
    from colonymind.reports.instinct import collect, document
    from colonymind.reports.text import markdown
    home = tmp_path / "instance"
    home.mkdir()
    store = Store(home / "instinct.db")
    provider = Varying(0.02, revision="rev-5")
    point = REGISTRY_POINT("opt_out")
    store.put_record(Record(point="opt_out", version=point.version, tier="fast", provider_kind=KIND,
                            revision="rev-5", bars={"yes": 0.9, "no": 0.9}, source="fitted"))
    # A labelled set that has seen one of the states the tier will read.
    fit.import_sets(store, {"format": "colonymind-instinct-sets/1", "set": "dev", "tier": "fast",
                            "provider": {"kind": KIND, "revision": "rev-5"},
                            "items": [{"key": "s-1", "point": "opt_out", "raw": {"yes": 0.1, "no": 0.9}, "gold": "no",
                                       "existing": "no", "subset": "fit",
                                       "fields": {"text": "the spare key words 9"}}]}, now=NOW)
    engine = Instinct(providers={"local": provider}, fast="local", store=store)
    ids = []
    for n in range(10):
        provider.p_yes = 0.97 if n < 2 else (0.5 if n < 4 else 0.02)
        decision = await engine.decide("opt_out", {"text": f"the spare key words {n}"},
                                       {"contact": "p-02", "source": f"turn:c-{n}"})
        await engine.drain()
        ids.append(decision.id)
    outcomes.label(store, ids[0], "yes", now=datetime.now(timezone.utc))      # the model's yes was right
    outcomes.label(store, ids[1], "no", now=datetime.now(timezone.utc))       # and here the existing path was
    store.open_audit("opt_out", "fast", "fp", "the fit settles every planted canary", {}, at=NOW)
    text = markdown(document(collect(home)))
    line = next(item for item in text.splitlines() if item.startswith("- fast (shadow)") and "coverage" in item)
    assert "coverage 80% (8 of 10 answers would have settled)" in line
    assert "agreement 80% with the deciding path" in line
    assert "2 disagreements, 2 with an outcome: the model right in 1, the existing path in 1" in line
    # Seen: the imported item's state and the two labelled decisions (they are fit examples too).
    assert "states seen in the fit sets: 3 (agreement 33%), new: 7 (agreement 100%)" in line
    assert "Outcomes: 1 right, 1 wrong (2 explicit)" in text
    assert "1 open audit" in text and "the fit settles every planted canary" in text
    assert "spare key" not in text


def REGISTRY_POINT(name):
    from colonymind.instinct import REGISTRY
    return REGISTRY[name]


# -- review: a first record that never acts is not written; a frozen state is never fitted on ----------------------

def test_a_first_record_that_would_never_act_is_not_written(tmp_path):
    """No record: the tier's answer holds the point on its existing path. A record whose every bar is "never"
    would turn that into an unsure answer, which escalates: it is not written."""
    from colonymind.instinct import fit
    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    tied = [dataclass_replace(e, existing=e.gold) for e in examples(point, 60)]      # the model can only tie
    result = fit.refit(store, point, "fast", tied, frozen=[], now=NOW)
    assert result.action == "skip" and "never act" in result.reason and store.records() == []


def test_a_labelled_decision_whose_state_is_a_frozen_item_is_never_fitted_on(tmp_path, monkeypatch):
    from colonymind.instinct import fit
    point = yes_no()
    store = Store(tmp_path / "instinct.db")
    rows = [dataclass_replace(e, state_hash=f"h-{n}") for n, e in
            enumerate(examples(point, 60, truth=lambda p: 1.0 if p >= 0.5 else 0.0))]
    frozen = [dataclass_replace(e, subset="frozen", key=f"f-{e.key}", state_hash=f"h-{n}") for n, e in
              enumerate(examples(point, 20, seed=5, truth=lambda p: 1.0 if p >= 0.5 else 0.0))]
    seen = []
    real = fit.candidate

    def spy(point_, tier, examples_):
        seen.append({example.state_hash for example in examples_})
        return real(point_, tier, examples_)

    monkeypatch.setattr(fit, "candidate", spy)
    fit.refit(store, point, "fast", rows, frozen=frozen, now=NOW)
    assert seen and not seen[0] & {example.state_hash for example in frozen}
    assert len(seen[0]) == 40


def test_a_labelled_set_keeps_a_readout_qualified_identity_whole(tmp_path):
    """A set answered under a declared readout (``kind/readout``) is a local tier's; its examples keep the whole
    identity, so records fitted from it never apply to the bare kind. A hosted kind with a readout stays refused."""
    from colonymind.instinct import fit
    point = shadowed()
    registry = table(point)
    store = Store(tmp_path / "instinct.db")
    items = [{"key": "r-1", "point": "probe", "raw": {"yes": 0.9, "no": 0.1}, "gold": "yes", "existing": "no",
              "subset": "fit"}]
    identity = KIND + "/letters-r1"
    done = fit.import_sets(store, set_document(point, items, provider={"kind": identity, "model": "",
                                                                         "revision": REVISION}),
                           now=NOW, registry=registry)
    assert done["kept"] == 1
    [kept] = store.examples(subset="fit")
    assert kept.kind == identity
    with pytest.raises(fit.SetError, match="never labels"):
        fit.import_sets(store, set_document(point, items, provider={"kind": "hosted-decision.v1/letters-r1"}),
                        now=NOW, registry=registry)
