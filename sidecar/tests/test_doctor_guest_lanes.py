"""``colonymind doctor``: which non-owner lanes could run the agent's machine? (docs/HERMES-ADAPTER.md, Owner and guest)

Hermes gives its tools per platform, not per sender, so a contact on a platform the owner also uses gets that
platform's terminal, code and file tools unless something else refuses them. The plugin's guard refuses them to every
non-owner session unless the operator opts the platform in (``guests.executor_platforms``). The doctor names the lanes
Hermes serves whose platforms offer those tools on this machine (``guest_lane_facts``, asked of Hermes itself): it
warns for an opted-in one, and passes naming the rest, which the guard refuses. These tests give Hermes' answer.
"""

from __future__ import annotations

import copy

import pytest

from colonymind import config, doctor
from colonymind.config import ConfigError, validate
from colonymind.doctor import PASS, SKIP, WARN


def facts(backend="local", lanes=None, **extra):
    return {"backend": backend, "lanes": {"whatsapp": ["execute_code", "terminal", "write_file"], "webhook": []}
            if lanes is None else lanes, **extra}


def only(results):
    result, = results
    return result


def test_lanes_the_guard_refuses_pass_and_are_named():
    result = only(doctor.guest_lane_results(facts(), []))
    assert result.name == "guest-lane-tools" and result.status == PASS, result
    assert "whatsapp" in result.detail and "terminal" in result.detail and "webhook" not in result.detail
    assert "refuses" in result.detail and "guests.executor_platforms" in result.detail


def test_an_opted_in_lane_on_this_machine_warns_with_its_remedy():
    result = only(doctor.guest_lane_results(facts(lanes={"api_server": ["execute_code", "terminal"],
                                                         "whatsapp": ["terminal"]}), ["api_server"]))
    assert result.status == WARN, result
    assert "api_server" in result.detail and "whatsapp" not in result.detail.split(";")[0]
    assert "instance key" in result.detail
    assert "guests.executor_platforms" in result.remedy and "terminal.backend" in result.remedy


def test_a_container_backend_or_no_executor_tools_pass():
    remote = only(doctor.guest_lane_results(facts(backend="docker"), ["whatsapp"]))
    assert remote.status == PASS and "docker" in remote.detail
    bare = only(doctor.guest_lane_results(facts(lanes={"webhook": [], "telegram": []}), ["telegram"]))
    assert bare.status == PASS and "no non-owner lane" in bare.detail


def test_what_hermes_cannot_say_is_said_never_guessed():
    unknown = only(doctor.guest_lane_results(facts(lanes={"rcs": None}), []))
    assert unknown.status == SKIP and "rcs" in unknown.detail
    mixed = only(doctor.guest_lane_results(facts(lanes={"rcs": None, "whatsapp": ["terminal"]}), ["rcs"]))
    assert mixed.status == WARN and "rcs" in mixed.detail          # opted in and unknown: it may run anything
    failed = only(doctor.guest_lane_results({"error": "TimeoutExpired"}, []))
    assert failed.status == SKIP and "TimeoutExpired" in failed.detail
    partial = only(doctor.guest_lane_results(facts(lanes={"webhook": []}, partial=True), []))
    assert partial.status == SKIP and "could not list" in partial.detail      # never a clean pass on a partial list
    desktop = only(doctor.guest_lane_results(facts(lanes={"telegram": ["focus_pane", "setup_mcp"]}), ["telegram"]))
    assert desktop.status == WARN and "focus_pane" in desktop.detail


def test_a_partial_inspection_never_passes_and_an_opted_in_platform_it_missed_warns():
    """Hermes could not list every platform its gateway serves (``partial``): the lanes it did list are named with
    what they offer, never as a clean pass; an opted-in platform missing from the list may be served with every tool,
    so it warns, with its remedy. Complete, an opted-in platform Hermes does not serve opens nothing."""
    seen = only(doctor.guest_lane_results(facts(lanes={"telegram": ["terminal"]}, partial=True), []))
    assert seen.status == SKIP and "telegram" in seen.detail and "terminal" in seen.detail, seen
    assert "could not list" in seen.detail
    missed = only(doctor.guest_lane_results(facts(lanes={"telegram": ["terminal"]}, partial=True), ["api_server"]))
    assert missed.status == WARN and "api_server" in missed.detail and "could not list" in missed.detail, missed
    assert "guests.executor_platforms" in missed.remedy
    bare = only(doctor.guest_lane_results(facts(lanes={"telegram": []}, partial=True), ["api_server"]))
    assert bare.status == WARN and "api_server" in bare.detail, bare
    complete = only(doctor.guest_lane_results(facts(lanes={"telegram": ["terminal"]}), ["api_server"]))
    assert complete.status == PASS, complete


def _config(**extra):
    return {**copy.deepcopy(config.DEFAULTS), **extra}


def test_the_opt_in_is_a_list_of_platform_names():
    data = validate(_config(guests={"executor_platforms": ["API_Server", " telegram "], "tools": ["mcp__docs__search"]}))
    assert data["guests"]["executor_platforms"] == ["api_server", "telegram"]
    assert data["guests"]["tools"] == ["mcp__docs__search"]
    assert "guests" not in validate(_config())
    for bad in ({"executor_platforms": "api_server"}, {"executor_platforms": [1]}, "api_server",
                {"executor_platforms": ["telegram", 17]}, {"executor_platforms": [""]}, {"tools": "memory"}):
        with pytest.raises(ConfigError, match="guests"):
            validate(_config(guests=bad))
