"""The shared part of every check.

A check is one scheduled job. Its script runs before the model and decides whether the model wakes:
it looks only at what is new since its last look, keeps its own small marker, prints one line for the
scheduler and records the run. A quiet run costs no model turn.

    check_base.main("mail", look)    the script reads the source itself
    check_base.main("projects")      no connector: the model looks with its own tools on every run

``look(since, settings, until=None)`` returns ``(marker, found)`` or ``(marker, found, problem)``:
``since`` is the marker the last look returned (``None`` the first time), ``found`` lists what is new
after it, oldest first, and ``problem`` is a short code label when part of the source could not be
read. With ``until`` the look stops at that earlier marker.

Run by the scheduler, the script prints ``{"wakeAgent": false}`` when nothing is new and otherwise
``{"wakeAgent": true, "check": <name>, "new": <count>, "read": <command>}``. That line never holds
anything read from the source. ``<script> read`` prints what the last waking run found, as data.

Standard library only. Python 3.9 or newer.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "check-state")
ENGINE_URL = "http://127.0.0.1:7777"
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # a local address, never a proxy


def load_settings():
    """``checks.json`` beside the scripts. A missing file is no settings."""
    try:
        with open(os.path.join(HERE, "checks.json"), encoding="utf-8") as handle:
            settings = json.load(handle)
    except FileNotFoundError:
        return {}
    if not isinstance(settings, dict):
        raise ValueError("checks.json holds no mapping")
    return settings


def load_state(name):
    try:
        with open(os.path.join(STATE, name + ".json"), encoding="utf-8") as handle:
            state = json.load(handle)
    except (FileNotFoundError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(name, state):
    os.makedirs(STATE, exist_ok=True)
    path = os.path.join(STATE, name + ".json")
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    os.replace(path + ".tmp", path)


def engine(settings):
    """Address and authorisation of the local engine, found the way its other local callers find them."""
    url = settings.get("engine_url") or os.environ.get("COLONYMIND_URL") or ENGINE_URL
    key = os.environ.get("COLONYMIND_API_KEY", "").strip()
    if not key:
        home = os.environ.get("COLONYMIND_HOME") or os.path.join(os.path.expanduser("~"), ".colonymind")
        try:
            with open(os.path.join(home, "api.key"), encoding="utf-8") as handle:
                key = (handle.read().strip().splitlines() or [""])[0].strip()
        except OSError:
            key = ""
    return url.rstrip("/"), ({"Authorization": "Bearer " + key} if key else {})


def get_json(url, headers=None, *, direct=False, timeout=20):
    """GET ``url`` and return its JSON answer. ``direct`` reaches a local address without a proxy."""
    request = urllib.request.Request(url, headers=headers or {})
    with (_DIRECT.open if direct else urllib.request.urlopen)(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def record(settings, name, result, choice="", detail="", took_ms=0):
    """One recorder line for this run: code labels and a timing. A recorder that is down changes nothing."""
    try:
        url, headers = engine(settings)
        line = {"kind": "check", "source": "check." + name, "result": result, "took_ms": took_ms}
        if choice:
            line.update(choice=choice, chooser="rule")
        if detail:
            line["detail"] = detail
        request = urllib.request.Request(
            url + "/v1/recorder/once", data=json.dumps(line).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers}, method="POST")
        _DIRECT.open(request, timeout=0.3).close()
    except Exception:  # noqa: BLE001 - recording never fails a check
        pass


def _gate(name, look):
    started = time.monotonic()
    settings, line, outcome = {}, {"wakeAgent": False}, ("silent", "ignore", "")
    try:
        settings = load_settings()
        if look is None:
            os.makedirs(STATE, exist_ok=True)
            line = {"wakeAgent": True, "check": name, "notes": os.path.join(STATE, name + ".notes.md")}
            outcome = ("pending", "act", "")
        else:
            state = load_state(name)
            since = state.get("marker")
            marker, found, *problem = look(since, settings)
            if found:
                save_state(name, {"marker": marker, "previous": since})
                line = {"wakeAgent": True, "check": name, "new": len(found),
                        "read": shlex.join([sys.executable, os.path.abspath(sys.argv[0]), "read"])}
                outcome = ("pending", "act", "")
            else:
                if marker != since:
                    save_state(name, dict(state, marker=marker))
                if problem and problem[0]:
                    outcome = ("failed", "", problem[0])
    except Exception as error:  # noqa: BLE001 - a look that failed is recorded and the model stays asleep
        outcome = ("failed", "", type(error).__name__.lower())
    print(json.dumps(line), flush=True)
    record(settings, name, *outcome, took_ms=int((time.monotonic() - started) * 1000))
    return 0


def _read(name, look):
    state = load_state(name)
    found = look(state.get("previous"), load_settings(), state.get("marker"))[1]
    text = json.dumps({"check": name, "content_trust": "untrusted", "items": found}, ensure_ascii=False, indent=1)
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    return 0


def main(name, look=None):
    """Run the check ``name``: the gate when the scheduler runs the script, ``read`` for the woken run."""
    if look is not None and sys.argv[1:2] == ["read"]:
        return _read(name, look)
    return _gate(name, look)
