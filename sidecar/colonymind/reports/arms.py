"""What a paired benchmark arm switches, read from its frozen profile, and what a contrast of two arms measures.

The profile a plan froze is read with the worker's own functions (``native_memory_worker.mind_section``,
``context_section`` and ``arm_environment``), so an arm means here what it meant in its container: its mind flags,
its drive weights, its two context switches, what the served host reads from its environment and any overlay that
sets one of them before the plugin loads. A contrast measures one faculty when its two arms differ in that faculty
and nothing else; any other overlay one arm carries is a difference of its own.

A profile names only what an arm switches: the rest is the shipped defaults of the image that ran it. Five switches
have shipped both ways (``EARLIER_OFF``), so where one arm sets such a switch and the other leaves it to the
defaults, the two may differ in it or not. A report that records the defaults its image shipped
(``recorded_off``) is read with them; one that does not is read both ways, and a contrast whose reading depends on
it is not read as any one faculty's (``unrecorded``). An arm's name never stands in for that record: an arm that
turns on a switch which ships on today runs what the arm without it runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet, Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from .registry import BY_KEY
from .text import listing

TRUTHY = frozenset({"1", "true", "yes", "on"})
# What an arm runs whether or not its mind is on: the memory blocks (the host's fit) and the claim-side learning
# rules (the claim projection). The status block renders from the served mind, so without one it is not there.
OUTSIDE_MIND = ("blocks", "learning")
# Overlays that set a faculty's switch the way an install's environment does (config.apply_environment).
OVERLAY_SWITCHES = {"COLONYMIND_CONTEXT_STATUS": "status", "COLONYMIND_LEARNING": "learning"}
# The switches that shipped off before faculties shipped on unless shown harmful, and ship on today: an arm that
# leaves one to the defaults ran it off or on depending on the image that ran it.
EARLIER_OFF = frozenset({"owner_gate", "learning", "wishes", "drift_check", "status"})


@dataclass(frozen=True)
class ArmState:
    plugin: bool
    mind: bool
    full: bool
    all_on: bool
    faculties: Tuple[Tuple[str, bool], ...]
    drives: Tuple[Tuple[str, float], ...]
    comparator: str = ""          # a stock-Hermes comparator's extra: a heartbeat or the curator
    overlay: Tuple[Tuple[str, str], ...] = ()   # the overlays that set no faculty's switch
    summary_only: bool = False

    def on(self) -> Dict[str, bool]:
        return dict(self.faculties)


@dataclass
class Subject:
    """What a contrast measures. ``kind``: ``faculty`` (one faculty, ``key``), ``drive`` (one drive's weight,
    ``drive``; the drives faculty's diagnostics), ``system`` (ColonyMind against stock Hermes), ``mixed`` (a
    mechanism comparison or more than one change, ``keys``), ``same`` (the same arm twice: the noise floor) or
    ``unrecorded`` (whether the arms differ in ``keys`` depends on the defaults of the image that ran them,
    which the report does not record).
    ``faculty_on`` names the side that has the faculty (or the drive, or ColonyMind) on: ``treatment`` or
    ``comparator``."""

    kind: str
    key: Optional[str] = None
    faculty_on: Optional[str] = None
    drive: Optional[str] = None
    keys: List[str] = field(default_factory=list)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def recorded_off(report: Mapping[str, Any]) -> Optional[FrozenSet[str]]:
    """The switches of ``EARLIER_OFF`` the report records as shipped off in the image that ran it, or None when it
    does not record them all. The record is the report's ``shipped_defaults``: each switch by its faculty key
    (``owner_gate``, ``status``), with whether the image shipped it on."""
    recorded = report.get("shipped_defaults") if isinstance(report, Mapping) else None
    if not isinstance(recorded, Mapping) or any(not isinstance(recorded.get(name), bool) for name in EARLIER_OFF):
        return None
    return frozenset(name for name in EARLIER_OFF if recorded[name] is False)


def state(profile: Mapping[str, Any], off: AbstractSet[str] = frozenset()) -> ArmState:
    """What the arm runs. ``off``: the switches of ``EARLIER_OFF`` shipped off in the image that ran it, where
    today's defaults ship them on; a value the arm's own switches set is kept."""
    from colonymind.config import DEFAULTS
    from colonymind.qualification import native_memory_worker as worker
    from colonymind.qualification import paired_worker
    plugin = bool(profile.get("plugin"))
    comparator = ("heartbeat_checkin" if profile.get("heartbeat_checkin") else
                  "heartbeat" if profile.get("heartbeat") else "curator" if profile.get("curator") else "")
    summary_only = profile.get("wm_summary_only") is True
    if not plugin:
        return ArmState(False, False, False, False, (), (), comparator, summary_only=summary_only)
    mind = paired_worker.mind_switches(profile)
    switches = mind or {}
    section = worker.mind_section(mind)
    context = worker.context_section({**switches, **(paired_worker.context_switches(profile) or {})}) or {}
    enabled = bool(section.get("enabled"))
    faculties = {name: bool(value) for name, value in (section.get("faculties") or {}).items()} if enabled else {}
    faculties.pop("initiative", None)
    faculties["status"] = bool(context.get("status", DEFAULTS["context"]["status"])) and enabled
    faculties["blocks"] = True  # Mandatory context fitting is a capability, with no configuration switch.
    # The claim-side learning rules run outside the mind: the served host reads them from its environment.
    faculties["learning"] = _truthy(worker.arm_environment(section, context).get("COLONYMIND_LEARNING", "0"))
    for name in off:
        # Only a value the defaults gave: a full arm's flag it names no switch for, the status block without
        # all_on, and the learning rules of an arm with the mind off.
        if name == "status" and not switches.get("all_on"):
            faculties["status"] = False
        elif name in OUTSIDE_MIND and not enabled:
            faculties[name] = False
        elif (name in faculties and switches.get("full") and not switches.get("all_on")
              and not switches.get(f"plus_{name}")):
            faculties[name] = False
    overlay = profile.get("overlay") if isinstance(profile.get("overlay"), dict) else {}
    for name, key in OVERLAY_SWITCHES.items():
        # Applied after the arm's own switches, with the mind on or off: the served host reads both from its
        # environment, and the claim-side learning rules run outside the mind.
        if name in overlay:
            faculties[key] = _truthy(overlay[name]) and (key != "status" or enabled)
    other = tuple(sorted((str(name), str(value)) for name, value in overlay.items() if name not in OVERLAY_SWITCHES))
    drives = tuple(sorted((str(name), float(value)) for name, value in (section.get("drives") or {}).items()))
    return ArmState(True, enabled, bool(mind and mind.get("full")), bool(mind and mind.get("all_on")),
                    tuple(sorted(faculties.items())), drives, comparator, other, summary_only)


def subject(treatment: Mapping[str, Any], comparator: Mapping[str, Any], off: Optional[AbstractSet[str]] = None
            ) -> Subject:
    """What the contrast measures. ``off``: the switches its report records as shipped off (``recorded_off``);
    None when it records nothing, and then the contrast is read with today's defaults and with the earlier ones
    alike: a reading that differs between them is ``unrecorded``, unless both read as more than one change."""
    if off is not None:
        return _subject(treatment, comparator, off)
    today, earlier = _subject(treatment, comparator, frozenset()), _subject(treatment, comparator, EARLIER_OFF)
    if today == earlier or (today.kind == "mixed" and earlier.kind == "mixed"):
        return today
    keys = [name for name in sorted(EARLIER_OFF) if _subject(treatment, comparator, frozenset({name})) != today]
    return Subject("unrecorded", keys=keys)


def _subject(treatment: Mapping[str, Any], comparator: Mapping[str, Any], off: AbstractSet[str]) -> Subject:
    first, second = state(treatment, off), state(comparator, off)
    if first.summary_only != second.summary_only:
        # Like the faculty dispatcher's working_memory.benchmark_evidence, keep mechanism contrasts out of
        # whole-faculty verdicts, including those that also change other settings.
        return Subject("mixed", keys=["the working-memory summary-only mechanism"])
    if first == second:
        return Subject("same")
    if first.plugin != second.plugin:
        if first.comparator or second.comparator:
            return Subject("mixed", keys=["a stock-Hermes comparator"])
        return Subject("system", faculty_on="treatment" if first.plugin else "comparator")
    if first.comparator != second.comparator:
        return Subject("mixed", keys=["a stock-Hermes comparator"])
    left, right = first.on(), second.on()
    changed = sorted(name for name in set(left) | set(right) if left.get(name, False) != right.get(name, False))
    ours, theirs = dict(first.overlay), dict(second.overlay)
    overlays = [f"its own {name} setting" for name in sorted(set(ours) | set(theirs)) if ours.get(name) != theirs.get(name)]
    if left.get("affect_rules") and right.get("affect_rules"):
        # The rules replace the running feelings whatever the affect flag says (mind/affect.py): a flipped
        # affect flag changes nothing the arm runs.
        changed = [name for name in changed if name != "affect"]
    if first.mind != second.mind:
        # The mind on against the mind off (the plugin in both): initiative, the mind itself, with everything that
        # needs it (the status block among them). A switch that acts without the mind changed with it makes it
        # more than that.
        outside = [name for name in OUTSIDE_MIND if left.get(name, False) != right.get(name, False)]
        if outside or overlays:
            return Subject("mixed", keys=["initiative", *outside, *overlays])
        return Subject("faculty", key="initiative", faculty_on="treatment" if first.mind else "comparator")
    drives = [name for name, _ in first.drives
              if dict(first.drives).get(name) != dict(second.drives).get(name)]
    if overlays:
        return Subject("mixed", keys=[*changed, *(f"the {name} drive" for name in drives), *overlays])
    if not changed and not drives:
        return Subject("same")
    if not changed and len(drives) == 1:
        name = drives[0]
        on = "treatment" if dict(first.drives)[name] > dict(second.drives)[name] else "comparator"
        return Subject("drive", key="drives", drive=name, faculty_on=on)
    if len(changed) == 1 and not drives:
        name = changed[0]
        return Subject("faculty", key=name, faculty_on="treatment" if left.get(name) else "comparator")
    return Subject("mixed", keys=[*changed, *(f"the {name} drive" for name in drives)])


def _noun(key: str) -> str:
    faculty = BY_KEY.get(key)
    return faculty.noun if faculty is not None else key.replace("_", " ")


def describe(profile: Mapping[str, Any], off: AbstractSet[str] = frozenset()) -> str:
    """The arm in plain words, relative to the mind it starts from; ``off`` as in ``state``."""
    arm = state(profile, off)
    if not arm.plugin:
        return {"heartbeat": "stock Hermes with a heartbeat", "heartbeat_checkin": "stock Hermes with a check-in "
                "heartbeat", "curator": "stock Hermes with its curator"}.get(arm.comparator, "stock Hermes")
    if not arm.mind:
        base_profile: Dict[str, Any] = {"plugin": True, "overlay": {}}
        base = "ColonyMind with the mind off"
    elif not arm.full:
        return "ColonyMind with only initiative on"
    elif arm.all_on:
        base_profile, base = {"plugin": True, "overlay": {}, "full": True, "all_on": True}, \
            "the mind with every faculty on"
    else:
        base_profile, base = {"plugin": True, "overlay": {}, "full": True}, "the mind at its shipped settings"
    start = state(base_profile, off)
    now, before = arm.on(), start.on()
    removed = [_noun(name) for name in sorted(now) if before.get(name) and not now.get(name)]
    added = [_noun(name) for name in sorted(now) if now.get(name) and not before.get(name)]
    zeroed = [f"the {name} drive at zero" for name, value in arm.drives
              if value == 0 and dict(start.drives).get(name, 0) != 0]
    parts = []
    if arm.summary_only:
        parts.append("with summaries instead of branch consults")
    if removed:
        parts.append(f"without {listing(removed)}")
    if added:
        parts.append(f"with {listing(added)} added")
    if zeroed:
        parts.append(f"with {listing(zeroed)}")
    if arm.overlay:
        parts.append(f"with its own {listing([name for name, _ in arm.overlay])} "
                     + ("setting" if len(arm.overlay) == 1 else "settings"))
    return base if not parts else f"{base}, {', '.join(parts)}"


__all__ = ["EARLIER_OFF", "ArmState", "Subject", "describe", "recorded_off", "state", "subject"]
