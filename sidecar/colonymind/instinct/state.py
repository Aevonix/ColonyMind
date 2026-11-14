"""A point's state per model tier: ``shadow`` (asked where eligible, logged, never applied), ``live`` or
``demoted`` (behaves as shadow; set automatically; live again only by a new promotion).

The shipped state is the registry's; a state recorded in ``instinct.db`` (by ``colonymind upgrade`` carrying a 1.11
setting, by a promotion or by a demotion, each kept as a transition with who made it and its evidence) overrides
it. A deliberate tier that is the point's existing path has no state: it is the path. A state is not configuration.

**Promotion** (``promote``, ``colonymind instinct promote <point> --tier fast|reader``) goes on offline evidence, as
soon as it is there. A live tier is judged by what it does, with its comparator logged and automatic demotion.
It refuses without both: a calibration record for the
fingerprint the tier answers under (the one it last answered under; before its first answer, the one the caller
names from the configured provider and the revision it serves); and an offline pass on the frozen sets imported for
it (with the fallback at least as accurate as the existing path overall, on the repository's items and on the novel
ones, and a measured flip rate under renaming and reordering of at most ``FLIP_MAX``). Any answers the tier already
gave under that fingerprint are read again under the record being put live (never what an earlier record made of
them) and kept with the evidence; among their disagreements with an explicit outcome, the existing path right more
often than the model (an outcome that names no right label but shows the model's label wrong counts against the
model) refuses the promotion, and so does what ``check`` would read at once: agreement under ``DRIFT_FLOOR`` over the
last ``DRIFT_WINDOW`` compared answers. A tier that withholds a label (``Tier.withholds``)
needs a third, the adversarial pass (``adversarial``): at least ``ADVERSARIAL_MIN`` adversarial frozen items whose
right label is another one, none read as it under the record, whatever the accuracy overall.

**Demotion** is automatic (``check``, in the nightly pass): a live tier whose agreement with the existing path over its
last ``DRIFT_WINDOW`` compared answers falls below ``DRIFT_FLOOR`` (a settled answer, or a withheld label read at its
bar, ``acted``, is compared where the existing path asks nothing, where its answer was asked beside it and logged, or
where the decision is the existing path's own, as a withheld label's is; one with nothing to compare it with is counted
unmeasured and named, ``unmeasured``, never counted as an agreement); the tighten rule (two explicit wrongs in seven
days); and a fingerprint change: the tier answering under another point version, model or revision than the one it went
live under (a promotion or a carry names it; a state carried by the upgrade or shipped live went live under the shipped
and carried records; when none can be established, any answer is a change), unless lineage carries it. ``demote`` is
also the owner's. These numbers are chosen values, not measured rules.

**Lineage** (``carry``, decision D3 of the 1.13 design). A live tier now answering under a learned revision keeps its
state when all hold: the revision has its own calibration record for the running fingerprint (never its parent's: a
child is refitted); its record's ``parent_revision`` chain (``calibration.lineage``) reaches a fingerprint of the tier's
live stretch (``live_stretch``: the promotion and every carry since), with a record at every link; and the revision
passes the offline pass (``offline``) and, for a tier that withholds a label, the adversarial pass (``adversarial``),
both read now on the frozen sets imported for its own fingerprint, as does every revision the chain walks through before
the one it reaches (each would keep the state only on its own passes). A server back on a revision of the stretch (a
learner rolled back) is held to the same record and passes, with no link to walk. The shadow evidence is waived for
such a revision; the offline gates and the automatic demotions (drift, the tighten rule) are the guard. The carry is a
recorded transition, ``live`` to ``live`` with the source ``CARRIED``, whose evidence names the running fingerprint, the
child's and the parent's record ids, each link (with its passes, for a link walked through), the fingerprint the chain
reached and both passes. Anything else is demoted as a fingerprint change, with why lineage did not carry in the
evidence. A demotion made while the gates are read (the owner's, the tighten rule's) stands: the carry is written
only when the tier is still live with no state change since the check began. A tier in shadow or demoted keeps its
state as before: nothing demotes or promotes it on a new revision.

**Drift under a learned revision** (``agreement``). For an answer given under a revision whose record names a parent, a
disagreement with the existing path counts against the tier only when an explicit outcome (the owner's own label, or a
resolver's grade from the owner's own words or acts: ``PROOF``) labels the tier's answer wrong (``graded``); one
that labels it right is measured and not against it; a disagreement with no such outcome (none, only a behavioural
or interpreted one, or one that says nothing of the tier's answer, as another label being wrong at a choice point
does) is counted as ``unlabelled`` and unmeasured, never as an agreement and never against the tier.
Agreements, and every answer under a revision without a parent, are counted as before. A promotion's shadow evidence
(every answer read again under the record, ``replay``) counts every disagreement, as before: the labelled-only count is
drift's, for a tier already live, and a first promotion keeps the stricter one.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .registry import STATES, Point

logger = logging.getLogger(__name__)

#: The drift that demotes and the flip rate an offline pass allows (chosen values, not measured).
DRIFT_WINDOW = 100
DRIFT_FLOOR = 0.90
FLIP_MAX = 0.10
#: A tier that withholds a label (``Tier.withholds``: ``ask_answer``'s ``approve``, which only ever asks the owner
#: back) goes live only on at least this many adversarial frozen items whose right label is another one (someone
#: else's yes, a quoted or injected one), none of them read as it (my choice: about two thirds of the bake-off's
#: adversarial ask items; at none wrong in 20 the one-sided 95% bound on the rate is about 14%).
ADVERSARIAL_MIN = 20
#: The answers that count toward agreement: the tier answered under a record (a settle, or an unsure answer that
#: would have left the existing path to decide).
ANSWERED = frozenset({"settled", "below_bar", "excluded", "high_stakes"})

APPLIES = "live"
#: The source of a transition lineage made (D3): a live tier kept live under a learned revision.
CARRIED = "carried by lineage"
#: The outcome sources that label a disagreement under a learned revision (D 2.1, what counts as proof): the owner's
#: own label and a resolver's grade from their own words or acts. Behavioural and interpreted outcomes never do.
PROOF = frozenset({"explicit"})


def shipped(point: Point, tier: str) -> Optional[str]:
    found = point.tier(tier)
    if found is None:
        return None
    return "live" if found.on_path else found.state


def promotable(point: Point) -> Tuple[str, ...]:
    """The tiers of ``point`` that have a state."""
    return tuple(tier.name for tier in point.tiers if not tier.on_path)


def effective(point: Point, tier: str, overrides: Mapping[Tuple[str, str], str]) -> Optional[str]:
    found = point.tier(tier)
    if found is None:
        return None
    if found.on_path:
        return "live"
    recorded = overrides.get((point.name, tier))
    return recorded if recorded in STATES else found.state


def states_of(point: Point, overrides: Mapping[Tuple[str, str], str]) -> Dict[str, str]:
    return {tier: effective(point, tier, overrides) or "" for tier in promotable(point)}


class Refused(ValueError):
    """A promotion the evidence does not carry, and why."""


def _tier_with_state(point: Point, tier: str) -> Any:
    found = point.tier(tier)
    if found is None or found.on_path or tier not in {"fast", "reader"}:
        raise Refused(f"{point.name} has no {tier} tier with a state")
    return found


def acted(point: Optional[Point], tier: str, entry: Mapping[str, Any]) -> Optional[str]:
    """The label a model tier's logged answer acts on: the one it settled, or a label its tier withholds read at its
    bar (``Tier.withholds``, logged ``excluded``: the caller asks the owner about it, or approves on it once trusted),
    which is measured as the reading it is, as a promotion's replay reads it. None: it abstained. Drift, the report
    and the tighten rule read a logged answer through this, so all of them measure what a promotion measured."""
    outcome = entry.get("outcome")
    if outcome == "settled":
        return entry.get("label")
    spec = point.tier(tier) if point is not None else None
    if outcome == "excluded" and spec is not None and entry.get("label") in spec.withholds:
        return entry.get("label")
    return None


def deciding(row: Mapping[str, Any]) -> Optional[str]:
    """What the existing path answered at a logged decision (``fit.existing_answer``)."""
    from .fit import existing_answer
    return existing_answer(row)


def running_fingerprint(store: Any, point: str, tier: str) -> Optional[str]:
    """The fingerprint the tier last answered under (the log's newest answer), or None."""
    for row in store.rows(point=point, limit=500):
        entry = (row.get("tiers") or {}).get(tier) or {}
        if entry.get("fingerprint") and "raw" in entry:
            return str(entry["fingerprint"])
    return None


def _replayed(point: Point, tier: str, record: Any) -> Any:
    """How the tier would read a logged answer under ``record``: its settled label, or None. A case logged as high
    stakes for the label it would settle again stays unsettled (it escalates whatever the confidence)."""
    from .fit import _reader
    read = _reader(point, point.tier(tier), record)

    def settled(entry: Mapping[str, Any]) -> Optional[str]:
        label = read(entry["raw"])
        if label is not None and entry.get("outcome") == "high_stakes" and entry.get("label") == label:
            return None
        return label
    return settled


def graded(point: Optional[Point], said: Optional[str], outcome: Mapping[str, Any]) -> Optional[str]:
    """What ``outcome`` says of the answer ``said`` at ``point``: ``wrong``, ``right``, or None when it says nothing
    of it. The right label known (``label``): right exactly when ``said`` is it. Else the outcome grades one label
    (``judged``): that label is ``said``, or, at a yes/no point, ``said`` is the other one; at a choice point another
    label being wrong says nothing of ``said``, and another being right makes it wrong (one label is right)."""
    if said is None:
        return None
    gold = outcome.get("label")
    if gold is not None:
        return "right" if said == gold else "wrong"
    verdict, judged = outcome.get("verdict"), outcome.get("judged")
    if judged is None or verdict not in {"right", "wrong"}:
        return None
    if judged == said:
        return str(verdict)
    if point is not None and point.type == "yes_no":
        return "right" if verdict == "wrong" else "wrong"
    return "wrong" if verdict == "right" else None


def learned_revisions(store: Any) -> Callable[[str], bool]:
    """Whether a fingerprint is a learned revision's: its calibration record names the revision it was learned from.
    One lookup per fingerprint."""
    from .fit import record_by_fingerprint
    known: Dict[str, bool] = {}

    def learned(fingerprint: str) -> bool:
        if fingerprint not in known:
            record = record_by_fingerprint(store, fingerprint) if fingerprint else None
            known[fingerprint] = bool(record is not None and record.parent_revision)
        return known[fingerprint]
    return learned


def agreement(store: Any, point: str, tier: str, *, fingerprint: Optional[str] = None,
              limit: Optional[int] = None, since: Optional[datetime] = None,
              replay: Optional[Tuple[Point, Any]] = None,
              registry: Optional[Mapping[str, Point]] = None) -> Dict[str, Any]:
    """How often the tier, had it been live, would have answered what the existing path answered: over its answers
    that have that comparison (newest first, at most ``limit``, under ``fingerprint`` when given); a tier that did
    not act agrees by construction. With the disagreements that have an explicit outcome: who was right. An answer
    that acted with nothing to compare it with (``fit.existing_answer`` None) is counted ``unmeasured``, never as an
    agreement. ``disagree`` counts the measured answers that disagreed, ``against`` those of them counted against the
    tier (every one, except under a learned revision), and ``agreement`` is the share of measured answers not counted
    against it. Under a learned revision (module docstring), except in a replay (a promotion's evidence), a
    disagreement no explicit outcome grades (``graded``) is counted ``unlabelled`` (and unmeasured), and one an
    explicit outcome labels right is measured: a disagreement, never counted against the tier.

    ``replay`` (a point and a record): every raw answer under the fingerprint is read again under that record, so
    the evidence is the record's own; without it, what the tier acted on under the record in force at the time
    (``acted``: a settled label, or a withheld one read at its bar, as the replay reads it)."""
    from . import REGISTRY
    from .fit import strongest
    settle = _replayed(replay[0], tier, replay[1]) if replay is not None else None
    spec = (REGISTRY if registry is None else registry).get(point)
    learned = learned_revisions(store)
    proof: Optional[Dict[str, Mapping[str, Any]]] = None

    def labelled() -> Dict[str, Mapping[str, Any]]:
        nonlocal proof
        if proof is None:
            proof = strongest(item for item in store.outcomes() if item["source"] in PROOF)
        return proof

    rows = []
    unmeasured = unlabelled = 0
    for row in store.rows(point=point, since=since):
        entry = (row.get("tiers") or {}).get(tier) or {}
        if fingerprint and entry.get("fingerprint") != fingerprint:
            continue
        if settle is not None:
            if not isinstance(entry.get("raw"), Mapping):
                continue
            label = settle(entry)
        elif entry.get("outcome") in ANSWERED:
            label = acted(spec, tier, entry)
        else:
            continue
        existing = deciding(row)
        if existing is None:
            unmeasured += label is not None       # settled, with nothing to compare it with: never an agreement
            continue
        against = label is not None and label != existing
        if against and settle is None and learned(str(entry.get("fingerprint") or "")):
            outcome = labelled().get(row["id"])
            said = graded(spec, label, outcome) if outcome is not None else None
            if said is None:
                unlabelled += 1                   # a learned revision's disagreement nothing labels: unmeasured
                unmeasured += 1
                continue
            against = said == "wrong"
        rows.append((row, label, existing, against))
        if limit is not None and len(rows) >= limit:
            break
    best = labelled() if rows else {}
    disagree = counted = model_right = existing_right = model_wrong = 0
    for row, label, existing, against in rows:
        counted += against
        if label is not None and label != existing:
            disagree += 1
            outcome = best.get(row["id"])
            gold = outcome.get("label") if outcome else None
            if gold is not None:
                model_right += label == gold
                existing_right += existing == gold
            elif outcome and outcome.get("verdict") == "wrong" and outcome.get("judged") == label:
                model_wrong += 1                    # known wrong, the right label unknown (the tighten rule's case)
    n = len(rows)
    out = {"n": n, "agree": n - disagree, "agreement": (n - counted) / n if n else None, "disagree": disagree,
           "against": counted, "explicit_model_right": model_right, "explicit_existing_right": existing_right,
           "explicit_model_wrong": model_wrong, "unmeasured": unmeasured, "unlabelled": unlabelled}
    if replay is not None:
        out["record"] = replay[1].id
    return out


def offline(store: Any, point: Point, tier: str, record: Any) -> Tuple[bool, Dict[str, Any]]:
    """The offline pass on the frozen sets imported for the record's fingerprint (module docstring)."""
    from . import fit
    frozen = store.examples(fingerprint=record.fingerprint, subset="frozen", point=point.name)
    spec = point.tier(tier)
    evidence: Dict[str, Any] = {"frozen": len(frozen)}
    if not frozen:
        return False, {**evidence, "why": "no frozen set imported for this fingerprint"}
    ok = True
    for name, subset in (("all", frozen), ("repository", [e for e in frozen if "repository" in e.tags]),
                         ("novel", [e for e in frozen if "novel" in e.tags])):
        with_record, existing = fit.accuracy(point, spec, record, subset), fit.accuracy(point, spec, None, subset)
        if not with_record["n"]:
            if name == "all":
                return False, {**evidence, "why": "no frozen item with a label and an existing answer"}
            continue
        evidence[name] = {"n": with_record["n"], "record": with_record["accuracy"], "existing": existing["accuracy"]}
        ok = ok and with_record["right"] >= existing["right"]
    rates = [entry["flip_rate"] for entry in store.sets(fingerprint=record.fingerprint)
             if entry["point"] == point.name and entry["flip_rate"] is not None]
    evidence["flip_rate"] = max(rates) if rates else None
    if not rates:
        return False, {**evidence, "why": "no flip rate measured for this fingerprint"}
    if max(rates) > FLIP_MAX:
        return False, {**evidence, "why": f"flip rate {max(rates):.2f} over {FLIP_MAX:.2f}"}
    if not ok:
        return False, {**evidence, "why": "the frozen sets are less accurate with it than the existing path"}
    return True, evidence


def adversarial(store: Any, point: Point, tier: str, record: Any) -> Tuple[bool, Dict[str, Any]]:
    """The adversarial pass of a tier that withholds a label (module docstring): per such label, at least
    ``ADVERSARIAL_MIN`` adversarial frozen items for the record's fingerprint whose right label is another one, and
    none read as it under the record (``fit.false_withheld``). A tier that withholds nothing passes with no evidence."""
    from .fit import false_withheld
    spec = point.tier(tier)
    if spec is None or not getattr(spec, "withholds", ()):
        return True, {}
    frozen = store.examples(fingerprint=record.fingerprint, subset="frozen", point=point.name)
    counted = false_withheld(point, spec, record, frozen)
    for label, entry in counted.items():
        if entry["false"]:
            return False, {**counted, "why": f"{entry['false']} of {entry['n']} adversarial frozen items read as "
                                             f"{label}, whose right label is another; none may be"}
        if entry["n"] < ADVERSARIAL_MIN:
            return False, {**counted, "why": f"{entry['n']} adversarial frozen items whose right label is not {label}; "
                                             f"promotion needs {ADVERSARIAL_MIN}, none read as {label}"}
    return True, counted


def promote(store: Any, point_name: str, tier: str, *, now: datetime, by: str = "owner",
            fingerprint: Optional[str] = None, registry: Optional[Mapping[str, Point]] = None) -> Dict[str, Any]:
    """Put a tier live on its offline evidence (module docstring); ``Refused`` names what is missing. ``fingerprint``:
    the one the tier will answer under, needed only before its first answer; once it has answered, a named one must
    be the one it answers under. The transition and its evidence are kept."""
    from . import REGISTRY
    from .fit import record_by_fingerprint
    table = REGISTRY if registry is None else registry
    point = table.get(point_name)
    if point is None:
        raise Refused(f"no point {point_name!r}")
    _tier_with_state(point, tier)
    if effective(point, tier, store.states()) == "live":
        raise Refused(f"{point_name} {tier} is live already")
    running = running_fingerprint(store, point_name, tier)
    if running is not None and fingerprint is not None and fingerprint != running:
        raise Refused(f"{point_name} {tier} answers under another fingerprint than the one named")
    fingerprint = running or fingerprint
    if fingerprint is None:
        raise Refused(f"{point_name} {tier} has not answered yet: name the revision it serves")
    record = record_by_fingerprint(store, fingerprint, registry=table)
    if record is None or record.version != point.version:
        raise Refused(f"no calibration record for the fingerprint {point_name} {tier} answers under")
    passed, frozen = offline(store, point, tier, record)
    if not passed:
        raise Refused(f"no offline pass on the frozen sets: {frozen['why']}")
    guarded, withheld = adversarial(store, point, tier, record)
    if not guarded:
        raise Refused(f"no adversarial pass: {withheld['why']}")
    answered = agreement(store, point_name, tier, fingerprint=fingerprint, replay=(point, record), registry=table)
    if answered["explicit_model_right"] < answered["explicit_existing_right"] + answered["explicit_model_wrong"]:
        raise Refused("the disagreements with an explicit outcome favour the existing path")
    recent = agreement(store, point_name, tier, limit=DRIFT_WINDOW, registry=table)     # exactly what ``check`` reads
    if recent["n"] >= DRIFT_WINDOW and recent["agreement"] < DRIFT_FLOOR:
        raise Refused(f"agreement {recent['agreement']:.3f} over the last {recent['n']} compared answers, under "
                      f"{DRIFT_FLOOR}: the drift rule would demote it at once")
    evidence = {"fingerprint": fingerprint, "record": record.id, "answered": answered, "frozen": frozen,
                "window": "every answer under this fingerprint, read under this record",
                **({"adversarial": withheld} if withheld else {})}
    store.set_state(point_name, tier, "live", source=f"promoted by {by}", evidence=evidence, at=now)
    return {"point": point_name, "tier": tier, "state": "live", "evidence": evidence}


def demote(store: Any, point_name: str, tier: str, *, reason: str, now: datetime, by: str,
           evidence: Optional[Mapping[str, Any]] = None, registry: Optional[Mapping[str, Point]] = None
           ) -> Optional[Dict[str, Any]]:
    """Demote a live tier (None when it is not live), keeping the transition, who made it and why."""
    from . import REGISTRY
    table = REGISTRY if registry is None else registry
    point = table.get(point_name)
    if point is None:
        raise Refused(f"no point {point_name!r}")
    _tier_with_state(point, tier)
    if effective(point, tier, store.states()) != "live":
        return None
    kept = {"reason": reason, **dict(evidence or {})}
    store.set_state(point_name, tier, "demoted", source=f"demoted by {by}: {reason}", evidence=kept, at=now)
    logger.warning("instinct: %s %s demoted (%s)", point_name, tier, reason)
    return {"point": point_name, "tier": tier, "state": "demoted", "evidence": kept}


def demote_tightened(store: Any, point: Point, tier: str, *, now: datetime, evidence: Mapping[str, Any],
                     registry: Optional[Mapping[str, Point]] = None) -> Optional[Dict[str, Any]]:
    """The tighten rule's demotion of a live tier."""
    return demote(store, point.name, tier, reason="the tighten rule: explicit wrongs within seven days", now=now,
                  by="the tighten rule", evidence=evidence, registry=registry)


def _went_live(transition: Mapping[str, Any]) -> Optional[str]:
    """The fingerprint a transition to ``live`` names (a promotion's, a carry's), or None."""
    fingerprint = (transition.get("evidence") or {}).get("fingerprint")
    return str(fingerprint) if fingerprint else None


def _unnamed(store: Any, point: Point, tier: str) -> List[str]:
    """What a live state recorded without a fingerprint went live under: the shipped records, and those put in force
    without labels (``n`` 0: measured or carried), whatever point version they were made for."""
    from . import records
    found = [record.fingerprint for record in records.shipped({point.name: point}) if record.tier == tier]
    found += [entry["record"].fingerprint for entry in store.record_history(point=point.name, limit=10 ** 6)
              if entry["record"].tier == tier and not entry["record"].n]
    return found


def promoted_under(store: Any, point: Point, tier: str) -> Tuple[str, ...]:
    """The fingerprints a live tier went live under: its latest promotion's, or its latest carry's (``CARRIED``). A
    live state recorded without one (the upgrade carrying a 1.11 setting, a tier shipped live, any writer that named
    none) went live under the records it could act on with no fit here (``_unnamed``). Empty when none can be
    established, which ``check`` reads as a fingerprint change, never as a reason to skip the check."""
    for transition in store.transitions(point=point.name, tier=tier):
        if transition["to_state"] == "live":
            fingerprint = _went_live(transition)
            if fingerprint:
                return (fingerprint,)
            break
    return tuple(dict.fromkeys(_unnamed(store, point, tier)))


def live_stretch(store: Any, point: Point, tier: str) -> Tuple[str, ...]:
    """Every fingerprint the tier's current live stretch went live under, newest first: each carry since its latest
    promotion, then that promotion's (or, for a live state recorded without one, what ``_unnamed`` gives). What
    lineage may reach (``carry``)."""
    found: List[str] = []
    for transition in store.transitions(point=point.name, tier=tier, limit=10 ** 6):
        if transition["to_state"] != "live":
            break
        fingerprint = _went_live(transition)
        if str(transition.get("source") or "").startswith(CARRIED) and fingerprint:
            found.append(fingerprint)
            continue
        if fingerprint:
            found.append(fingerprint)
            return tuple(dict.fromkeys(found))
        break
    return tuple(dict.fromkeys([*found, *_unnamed(store, point, tier)]))


def carry(store: Any, point: Point, tier: str, running: str) -> Tuple[bool, Dict[str, Any]]:
    """Whether a live tier answering under ``running`` keeps its state by lineage (module docstring, Lineage), and the
    evidence: on a carry, what the transition keeps; otherwise ``why`` it does not carry."""
    from . import calibration, records
    from .fit import record_by_fingerprint
    record = record_by_fingerprint(store, running, registry={point.name: point})
    evidence: Dict[str, Any] = {"fingerprint": running}
    if record is None or record.tier != tier:
        return False, {**evidence, "why": "no calibration record for the fingerprint the tier now answers under"}
    if record.version != point.version:
        return False, {**evidence, "why": "the point's question changed (another point version)"}
    evidence["record"] = record.id
    stretch = live_stretch(store, point, tier)
    if running in stretch:
        links: List[Any] = []                 # a revision the tier went live under before: no link to walk
        reached = running
    else:
        if not record.parent_revision:
            return False, {**evidence, "why": "the revision names no parent"}
        chain = calibration.lineage([*store.records(), *records.shipped({point.name: point})], record)
        at = next((index for index, parent in enumerate(chain) if parent.fingerprint in stretch), None)
        if at is None:
            return False, {**evidence, "why": "its lineage reaches no revision the tier went live under with a record "
                                              "at every link",
                           "lineage": [{"record": parent.id, "fingerprint": parent.fingerprint} for parent in chain]}
        links, reached = chain[:at + 1], chain[at].fingerprint
    lineage = [{"record": parent.id, "fingerprint": parent.fingerprint} for parent in links]
    evidence.update(parent=links[0].id if links else None, reached=reached, lineage=lineage)
    # Each revision the chain walks through keeps the state only as the running one does, on its own passes; the
    # one it reaches passed when the tier went live under it.
    for parent, link in zip(links[:-1], lineage):
        failed = _gates(store, point, tier, parent, link)
        if failed is not None:
            return False, {**evidence, "why": f"no {failed[0]} pass for the link {parent.revision}: {failed[1]}"}
    failed = _gates(store, point, tier, record, evidence)
    if failed is not None:
        return False, {**evidence, "why": f"no {failed[0]} pass: {failed[1]}"}
    return True, evidence


def _gates(store: Any, point: Point, tier: str, record: Any, evidence: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """The offline pass and, for a tier that withholds a label, the adversarial pass of one record, read now on the
    frozen sets imported for its fingerprint, their results kept in ``evidence``: None when both pass, else the pass
    that failed (``offline``, ``adversarial``) and why."""
    passed, frozen = offline(store, point, tier, record)
    evidence["frozen"] = frozen
    if not passed:
        return "offline", frozen["why"]
    guarded, withheld = adversarial(store, point, tier, record)
    if withheld:
        evidence["adversarial"] = withheld
    if not guarded:
        return "adversarial", withheld["why"]
    return None


def unmeasured(store: Any, *, registry: Optional[Mapping[str, Point]] = None) -> List[Dict[str, Any]]:
    """The live tiers whose settled answers had nothing to compare them with, or disagreed under a learned revision
    with no explicit outcome to say who was right (``unlabelled``), in the span of their last ``DRIFT_WINDOW`` compared
    answers (the whole log when fewer): drift is not measured on those. Said, never a reason to demote (a tier is not
    switched off for being unmeasured)."""
    from . import REGISTRY
    table = REGISTRY if registry is None else registry
    overrides = store.states()
    out = []
    for name, point in table.items():
        for tier in promotable(point):
            if effective(point, tier, overrides) != "live":
                continue
            recent = agreement(store, name, tier, limit=DRIFT_WINDOW, registry=table)
            if recent["unmeasured"]:
                out.append({"point": name, "tier": tier, "unmeasured": recent["unmeasured"], "compared": recent["n"],
                            "unlabelled": recent["unlabelled"]})
    return out


def _newest(store: Any, point: str, tier: str) -> Optional[int]:
    """The sequence number of a tier's newest state change, or None."""
    found = store.transitions(point=point, tier=tier, limit=1)
    return int(found[0]["seq"]) if found else None


def check(store: Any, *, now: datetime, registry: Optional[Mapping[str, Point]] = None) -> List[Dict[str, Any]]:
    """The automatic demotions (drift, a fingerprint change lineage does not carry) of every live tier; the demotions
    made. A carry is kept as its own transition (``CARRIED``) and is not among them."""
    from . import REGISTRY
    from .fit import record_by_fingerprint
    table = REGISTRY if registry is None else registry
    overrides = store.states()
    made: List[Dict[str, Any]] = []
    for name, point in table.items():
        for tier in promotable(point):
            if effective(point, tier, overrides) != "live":
                continue
            before = _newest(store, name, tier)
            running = running_fingerprint(store, name, tier)
            allowed = promoted_under(store, point, tier)
            if running is not None and running not in allowed:
                kept, lineage = carry(store, point, tier, running)
                if kept:
                    # The gates take a while to read: a demotion made meanwhile (the owner's, the tighten rule's)
                    # stands, and the carry is not written over it.
                    if effective(point, tier, store.states()) != "live" or _newest(store, name, tier) != before:
                        logger.info("instinct: %s %s changed state while lineage was read; no carry", name, tier)
                        continue
                    if store.set_state(name, tier, "live", source=f"{CARRIED} (the nightly check)",
                                       evidence={**lineage, "from": list(allowed), "to": running}, at=now,
                                       expected_seq=before):
                        logger.info("instinct: %s %s carried by lineage", name, tier)
                else:
                    record = record_by_fingerprint(store, running, registry={point.name: point})
                    evidence = {"from": list(allowed), "to": running,
                                "parent_revision": record.parent_revision if record is not None else None,
                                "lineage": lineage}
                    reason = ("a new model revision or point version (fingerprint change)" if allowed else
                              "no fingerprint it went live under can be established (fingerprint change)")
                    done = demote(store, name, tier, reason=reason, now=now, by="the nightly check",
                                  evidence=evidence, registry=table)
                    if done:
                        made.append(done)
                    continue
            recent = agreement(store, name, tier, limit=DRIFT_WINDOW, registry=table)
            if recent["n"] >= DRIFT_WINDOW and recent["agreement"] < DRIFT_FLOOR:
                done = demote(store, name, tier, reason=f"drift: agreement {recent['agreement']:.3f} over the last "
                              f"{DRIFT_WINDOW} answers, under {DRIFT_FLOOR}", now=now, by="the nightly check",
                              evidence={"window": DRIFT_WINDOW, "agreement": recent}, registry=table)
                if done:
                    made.append(done)
    return made


__all__ = ["ADVERSARIAL_MIN", "APPLIES", "CARRIED", "DRIFT_FLOOR", "DRIFT_WINDOW", "FLIP_MAX", "PROOF", "STATES",
           "Refused", "acted", "adversarial", "agreement", "carry", "check", "deciding", "demote", "demote_tightened",
           "effective", "graded", "learned_revisions", "live_stretch", "offline", "promote", "promotable",
           "promoted_under", "running_fingerprint", "shipped", "states_of", "unmeasured"]
