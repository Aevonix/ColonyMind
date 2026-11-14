"""Retained branches, consult outcomes and automatic withdrawal, read without writes."""
from __future__ import annotations

import json
from dataclasses import replace
from math import isfinite
from typing import Any, Mapping

from .text import Document, plural

SETTING = "mind.faculties.working_memory"


def collect(config, *, runtime=None) -> dict:
    """Read persisted totals without opening or migrating the writable branch store."""
    if runtime is not None:
        return runtime.report()
    from colonymind.memory.branches import read_report
    return read_report(config)


def lines(collected: Mapping[str, Any]) -> list[str]:
    result = [f"It retains {plural(collected.get('branches', 0), 'branch', 'branches')} with "
              f"{plural(collected.get('spans', 0), 'span')}; "
              f"{plural(collected.get('consults', 0), 'consult')} recorded."]
    if not collected.get("enabled", True):
        result.append("Working memory is switched off; capture, consults and the branches context are off.")
    elif collected.get("summary_only"):
        result.append("Only summaries are available: capture stays on and the consult tool is off.")
    refusals = collected.get("refusals", {})
    if refusals:
        result.append("Refusals by reason: " + "; ".join(
            f"{str(reason).replace('_', ' ')}: {count}" for reason, count in sorted(refusals.items())) + ".")
    else:
        result.append("No consults have been refused.")
    comparator = collected.get("comparator", {})
    if comparator.get("labelled"):
        result.append(f"The live comparison with summaries has {plural(comparator['labelled'], 'labelled pair')}: "
                      f"{plural(comparator.get('losses', 0), 'loss', 'losses')} and "
                      f"{plural(comparator.get('gains', 0), 'gain')}.")
    else:
        result.append("The live comparison with summaries has no labelled outcomes yet.")
    if collected.get("withdrawn"):
        reason = str(collected.get("withdrawal_reason") or "the live comparison found a decline").rstrip(".")
        result.append(f"The consult tool is withdrawn: {reason}. Run colonymind wm reset to clear withdrawal.")
    share = collected.get("cached_token_share")
    if share is None:
        result.append("The consult lane's cached-token share is unavailable in its usage record.")
    else:
        result.append(f"The consult lane served {float(share):.1%} of input tokens from cache.")
    return result


def document(collected: Mapping[str, Any]) -> Document:
    report = Document("Working memory")
    report.bullets(lines(collected))
    report.paragraph(f"Setting: {SETTING}")
    return report


def to_json(collected: Mapping[str, Any]) -> str:
    return json.dumps(collected, sort_keys=True, allow_nan=False) + "\n"


def faculty(reading, section) -> None:
    collected = collect(reading.instance.config)
    section.lines.append("Branches and spans are retained totals; consults and refusals are lifetime totals. "
                         "The live comparison uses its own rolling window.")
    section.lines.extend(lines(collected))
    # These totals cannot establish whether activity happened inside the report's selected window.
    section.records = False


def benchmark_evidence(loaded):
    """Keep summary-only contrasts as mechanism diagnostics before faculty classification."""
    from .bench import family
    ordinary, diagnostics = [], []
    for item in sorted(loaded, key=lambda item: -item.mtime):
        report = item.report
        profiles = report.get("profiles") or {}
        statistics = report.get("statistics") or {}
        kept = []
        for contrast in statistics.get("contrasts") or []:
            if not isinstance(contrast, dict):
                kept.append(contrast)
                continue
            first, second = (profiles.get(contrast.get(name)) or {} for name in ("treatment", "comparator"))
            if not all(isinstance(profile, dict) for profile in (first, second)):
                kept.append(contrast)
                continue
            summary = [profile.get("wm_summary_only") is True for profile in (first, second)]
            if summary[0] == summary[1]:
                kept.append(contrast)
                continue
            normalized = [{key: value for key, value in profile.items() if key not in {"name", "wm_summary_only"}}
                          for profile in (first, second)]
            line = f"Working-memory mechanism comparison on {family(report).lower()}: "
            if (normalized[0] != normalized[1] or not first.get("plugin") or not first.get("full")
                    or first.get("minus_working_memory")):
                line += "the summary setting changes alongside other settings, so its effect cannot be isolated."
            else:
                rates = [contrast.get(name + "_pass_rate") for name in ("treatment", "comparator")]
                if (contrast.get("verdict") in {"demonstrated", "not_demonstrated"}
                        and all(type(rate) in (int, float) and isfinite(rate) and 0 <= rate <= 1 for rate in rates)):
                    full, brief = rates[::-1] if summary[0] else rates
                    line += f"full branch consults passed {full:.1%}; summaries with capture enabled passed {brief:.1%}."
                else:
                    line += "full branch consults were compared with summaries, but usable results are unavailable."
            diagnostics.append(line + " This diagnostic does not establish a faculty-wide verdict.")
        ordinary.append(replace(item, report={**report, "statistics": {**statistics, "contrasts": kept}}))
    return ordinary, diagnostics


__all__ = ["benchmark_evidence", "collect", "document", "faculty", "lines", "to_json"]
