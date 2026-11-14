"""The ``instinct`` section of ``colonymind.yaml``: providers by name, the provider each model tier asks, and bars a
point's labels may be raised to.

```yaml
instinct:
  providers:
    local: {kind: local-decision.v1, url: "", timeout_ms: 250}   # your decision server; empty: no fast tier
  fast: local
  reader: ""                          # empty: no reader tier
  points:
    owner_send: {bars: {drop: 0.97}}  # bars can only be raised above the code floors
  external: {}                        # operator points of the Hermes gates
```

``validate`` is ``colonymind.yaml``'s check (it refuses a malformed section by name); ``parse`` reads the section the
sidecar receives leniently (whatever is unusable is left out, so a point keeps its existing path). A point's state
is not configuration: it lives in ``instinct.db``.

``carry_decisions`` maps a 1.11 ``decisions`` section onto this one, once, for ``colonymind upgrade``: ``url`` and
``timeout_ms`` become the ``local`` provider's; a point's ``enabled``, read as the 1.11 loader read it (a boolean,
or a word or number for one), becomes its fast tier's state (true: live, false: shadow); its ``temperature`` and
``abstain`` band become local calibration records for the revisions the shipped records apply to (the measured
checkpoint, and a server that reports no revision): the setting was a tuning of those measured defaults, and a
server that reports another checkpoint gets no record, so its tier abstains until one is fitted for it.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

KINDS = ("local-decision.v1", "logits")
PROVIDER_KEYS = {"kind", "url", "timeout_ms", "model"}
SECTION_KEYS = {"providers", "fast", "reader", "points", "external"}
TIMEOUT_MS_BOUNDS = (1, 5000)
_URL = re.compile(r"^https?://[^\s/]+(?:/\S*)?$")

DEFAULT: Dict[str, Any] = {"providers": {"local": {"kind": "local-decision.v1", "url": "", "timeout_ms": 250}},
                           "fast": "local", "reader": "", "points": {}, "external": {}}


class SettingsError(ValueError):
    """A malformed ``instinct`` section; the message names the key."""


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def default_timeout_ms(kind: str) -> int:
    """The time limit a provider of ``kind`` gets when its entry sets none: its own kind's default (a decision
    server answers in one short call; a reader makes a completions call per rotation of the options)."""
    from .providers import local_decision, logits
    return {local_decision.KIND: local_decision.DEFAULT_TIMEOUT_MS,
            logits.KIND: logits.DEFAULT_TIMEOUT_MS}[kind]


def _provider(name: str, entry: Any) -> Dict[str, Any]:
    where = f"instinct.providers.{name}"
    if not isinstance(entry, dict):
        raise SettingsError(f"{where} must be a mapping with kind, url and timeout_ms")
    provider_keys = PROVIDER_KEYS | ({"route", "thinking", "rotations"} if entry.get("kind") == "logits" else set())
    unknown = sorted(set(entry) - provider_keys)
    if unknown:
        raise SettingsError(f"{where}: unknown key {unknown[0]} ({', '.join(sorted(provider_keys))})")
    kind = entry.get("kind")
    if kind not in KINDS:
        raise SettingsError(f"{where}.kind must be one of {', '.join(KINDS)}")
    url = str(entry.get("url") or "").strip()
    if url and not _URL.match(url):
        raise SettingsError(f"{where}.url must be an http(s) URL, or empty to leave its tier off")
    timeout = _number(entry.get("timeout_ms", default_timeout_ms(kind)))
    if timeout is None or not TIMEOUT_MS_BOUNDS[0] <= timeout <= TIMEOUT_MS_BOUNDS[1]:
        raise SettingsError(f"{where}.timeout_ms must be a number of milliseconds from %d to %d" % TIMEOUT_MS_BOUNDS)
    model = entry.get("model", "")
    if model is not None and not isinstance(model, str):
        raise SettingsError(f"{where}.model must be a string")
    out = {"kind": kind, "url": url, "timeout_ms": int(timeout) if timeout.is_integer() else timeout}
    if model:
        out["model"] = model
    if kind == "logits":
        from .providers.logits import MAX_ROTATIONS, ROUTES

        route = entry.get("route", "completions")
        if route not in ROUTES:
            raise SettingsError(f"{where}.route must be one of {', '.join(ROUTES)}")
        if ("thinking" in entry or route == "chat") and entry.get("thinking") != "off":
            raise SettingsError(f"{where}.thinking must be off (required for the chat route)")
        rotations = entry.get("rotations", MAX_ROTATIONS)
        if isinstance(rotations, bool) or not isinstance(rotations, int) or not 1 <= rotations <= MAX_ROTATIONS:
            raise SettingsError(f"{where}.rotations must be a whole number from 1 to {MAX_ROTATIONS}")
        out.update({key: entry[key] for key in ("route", "thinking", "rotations") if key in entry})
    return out


def _points(value: Any, registry: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SettingsError("instinct.points must be a mapping of decision points")
    checked: Dict[str, Any] = {}
    for name, entry in value.items():
        where = f"instinct.points.{name}"
        point = registry.get(name)
        if point is None:
            raise SettingsError(f"{where}: no such decision point ({', '.join(sorted(registry))})")
        if not isinstance(entry, dict) or set(entry) - {"bars"}:
            raise SettingsError(f"{where} must be a mapping with bars (a point's state is not configuration)")
        bars = entry.get("bars") or {}
        if not isinstance(bars, dict):
            raise SettingsError(f"{where}.bars must map labels to probabilities")
        out = {}
        for label, bar in bars.items():
            number = _number(bar)
            if label not in point.labels or number is None or not 0.0 <= number <= 1.0:
                raise SettingsError(f"{where}.bars: {label} must be one of the point's labels "
                                    f"({', '.join(point.labels)}) with a bar from 0 to 1")
            out[str(label)] = number
        checked[str(name)] = {"bars": out}
    return checked


def validate(value: Any, registry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The section checked and normalised; SettingsError names the first malformed key."""
    if registry is None:
        from .registry import REGISTRY as registry
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise SettingsError("instinct must be a mapping with providers, fast, reader, points and external")
    unknown = sorted(set(value) - SECTION_KEYS)
    if unknown:
        raise SettingsError(f"instinct has unknown keys: {', '.join(unknown)}")
    providers = value.get("providers", copy.deepcopy(DEFAULT["providers"]))
    if not isinstance(providers, dict):
        raise SettingsError("instinct.providers must be a mapping of provider names to {kind, url, timeout_ms}")
    checked = {str(name): _provider(str(name), entry) for name, entry in providers.items()}
    out: Dict[str, Any] = {"providers": checked}
    for tier in ("fast", "reader"):
        chosen = value.get(tier, DEFAULT[tier])
        chosen = "" if chosen is None else chosen
        if not isinstance(chosen, str) or (chosen and chosen not in checked):
            raise SettingsError(f"instinct.{tier} must name a configured provider ({', '.join(checked) or 'none'}), "
                                "or be empty")
        out[tier] = chosen
    out["points"] = _points(value.get("points") or {}, registry)
    external = value.get("external", {})
    external = {} if external is None else external
    if not isinstance(external, dict):
        raise SettingsError("instinct.external must be a mapping of operator points")
    out["external"] = external
    return out


def parse(value: Any, registry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The section as the sidecar reads it: each usable provider and point entry, the rest left out."""
    if registry is None:
        from .registry import REGISTRY as registry
    value = value if isinstance(value, dict) else {}
    providers: Dict[str, Any] = {}
    for name, entry in (value.get("providers") if isinstance(value.get("providers"), dict) else {}).items():
        try:
            providers[str(name)] = _provider(str(name), entry)
        except SettingsError:
            continue
    out: Dict[str, Any] = {"providers": providers}
    for tier in ("fast", "reader"):
        chosen = value.get(tier, DEFAULT[tier])
        out[tier] = chosen if isinstance(chosen, str) and chosen in providers else ""
    out["points"] = points_section(value.get("points"), registry)
    out["external"] = value.get("external") if isinstance(value.get("external"), dict) else {}
    return out


def points_section(value: Any, registry: Mapping[str, Any]) -> Dict[str, Any]:
    """Each usable ``points`` entry; an unusable one is left out."""
    out: Dict[str, Any] = {}
    for name, entry in (value if isinstance(value, dict) else {}).items():
        try:
            out.update(_points({name: entry}, registry))
        except SettingsError:
            continue
    return out


# -- the 1.11 decisions section, carried once ------------------------------------------------------------------

@dataclass
class Carried:
    section: Dict[str, Any]
    states: Dict[Tuple[str, str], str] = field(default_factory=dict)
    records: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _enabled_1_11(value: Any, where: str) -> bool:
    """A 1.11 ``enabled`` as 1.11's loader read it (``colonymind.config._parse_bool``): a boolean, or a word or number
    for one ("false", "off", 0, "on" ...). SettingsError, naming the key, for a value that loader refused."""
    from colonymind.config import ConfigError, _parse_bool
    try:
        return _parse_bool(value, field_name=where)
    except ConfigError as error:
        raise SettingsError(str(error)) from None


def carry_decisions(value: Any, registry: Optional[Mapping[str, Any]] = None) -> Carried:
    """A 1.11 ``decisions`` section as an ``instinct`` section, states and local records (see the module).
    SettingsError for an ``enabled`` the 1.11 loader refused: the upgrade stops before it writes anything."""
    from . import calibration
    if registry is None:
        from .registry import REGISTRY as registry
    value = value if isinstance(value, dict) else {}
    url = str(value.get("url") or "").strip()
    timeout = _number(value.get("timeout_ms", 250))
    if timeout is None or not TIMEOUT_MS_BOUNDS[0] <= timeout <= TIMEOUT_MS_BOUNDS[1]:
        timeout = 250.0
    section = copy.deepcopy(DEFAULT)
    section["providers"]["local"] = {"kind": "local-decision.v1", "url": url,
                                     "timeout_ms": int(timeout) if timeout.is_integer() else timeout}
    carried = Carried(section)
    carried.notes.append("the decisions section is now instinct: " + (
        "the decision server became the local provider of the fast tier" if url
        else "no decision server was configured, so every point keeps its existing path"))
    for name, entry in (value.get("points") if isinstance(value.get("points"), dict) else {}).items():
        point = registry.get(name)
        if point is None or point.tier("fast") is None or not isinstance(entry, dict):
            carried.notes.append(f"decisions.points.{name} was not carried: no such fast point")
            continue
        if "enabled" in entry:
            state = "live" if _enabled_1_11(entry["enabled"], f"decisions.points.{name}.enabled") else "shadow"
            carried.states[(name, "fast")] = state
            carried.notes.append(f"{name}: its fast tier is {state}, as decisions.points.{name}.enabled said")
        if "temperature" in entry or "abstain" in entry:
            temperature, band = 1.0, (0.1, 0.9)
            temperature = _number(entry.get("temperature", temperature)) or temperature
            abstain = entry.get("abstain", band)
            if not (isinstance(abstain, (list, tuple)) and len(abstain) == 2
                    and all(_number(item) is not None for item in abstain)):
                abstain = band
            bars = calibration.band_bars(point.labels, point.type, (float(abstain[0]), float(abstain[1])))
            carried.records.extend(calibration.Record(
                point=name, version=point.version, tier="fast", provider_kind="local-decision.v1", model="",
                revision=revision, temperature=float(temperature), bars=bars,
                source="carried from the 1.11 decisions section") for revision in calibration.MEASURED_REVISIONS)
            carried.notes.append(f"{name}: its 1.11 temperature and abstain band are a local calibration record for "
                                 "the measured checkpoint (and a server that reports no revision); a server that "
                                 "reports another checkpoint gets none, and the tier abstains until it is calibrated")
    return carried


__all__ = ["DEFAULT", "KINDS", "Carried", "SettingsError", "carry_decisions", "default_timeout_ms", "parse",
           "points_section", "validate"]
