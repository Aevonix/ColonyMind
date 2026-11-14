"""Hermes' vault calls in both shapes: one tool per action, and one tool with an action.

Hermes' older sources register ``browser_vault_fill``, ``browser_vault_list`` and three more; its newer ones
register ONE tool, ``browser_vault``, whose ``action`` says which. The product reads the one tool as the older tool
of its action (``read_as``): the same effect, the same deny entry, the same grant name, and the owner's yes to a
payment fill handed to Hermes' consent once. The yes ends with the fill it was given for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from types import SimpleNamespace

import pytest

from colonymind.initiatives.store import InitiativeStore
from colonymind.mind.authority import Authority, Policy, effect_of, executor_call, grantable
from colonymind_shared.effects import VAULT_ACTIONS, VAULT_TOOL, read_as
from test_mind_loop import OWNER, pinned_clock  # noqa: F401  (pinned_clock: autouse fixture)

needs_hermes = pytest.mark.skipif(importlib.util.find_spec("hermes_time") is None,
                                  reason="needs stock Hermes in the test interpreter")
# What each action takes, in the older tool's arguments; the one tool takes the same and its ``action``.
ACTIONS = {"list": {}, "unlock": {"backend": "bitwarden"}, "fill": {"handle": "vault_card"},
           "save_login": {"label": "shop"}, "enter_code": {"handle": "vault_card"}}
# One payment fill as each shape calls it, and the newer one on the in-app browser.
FILLS = (("browser_vault_fill", {"handle": "vault_card"}),
         ("browser_vault", {"action": "fill", "handle": "vault_card"}),
         ("browser_vault", {"action": "fill", "handle": "vault_card", "target": "preview"}))


def test_the_one_vault_tool_is_read_as_the_older_tool_of_its_action():
    assert VAULT_TOOL == "browser_vault" and set(VAULT_ACTIONS) == set(ACTIONS)
    for action, args in ACTIONS.items():
        older, call = f"browser_vault_{action}", {"action": action, **args}
        assert read_as("browser_vault", call) == older and read_as(older, args) == older
        for context in ({}, {"vault_kind": "payment"}, {"vault_kind": "login"}, {"vault_kind": "address"}):
            was, now = effect_of(older, args, context=context), effect_of("browser_vault", call, context=context)
            assert (was is None) == (now is None), (action, context)
            if was is not None:
                assert (now.cls, now.tool, now.bounded, now.floor) == (was.cls, was.tool, was.bounded, was.floor)
                # A permit binds to the call Hermes gets: the one tool, its action and every argument.
                assert json.loads(now.call) == {"tool": "browser_vault", "args": call}
    fill = effect_of("browser_vault", {"action": "fill", "handle": "vault_card"})
    assert (fill.cls, fill.tool, fill.target) == ("money_movement", "browser_vault_fill", "vault item vault_card")
    assert fill.target == effect_of("browser_vault_fill", {"handle": "vault_card"}).target
    for kind in ("login", "address"):
        assert effect_of("browser_vault", {"action": "fill", "handle": "h"}, context={"vault_kind": kind}) is None
    assert effect_of("browser_vault", {"action": "list"}) is None
    save = effect_of("browser_vault", {"action": "save_login", "label": "shop"})
    assert (save.cls, save.tool) == ("credential_change", "browser_vault_save_login")
    for action in ("unlock", "enter_code"):
        held = effect_of("browser_vault", {"action": action, **ACTIONS[action]})
        assert (held.cls, held.tool, held.bounded) == ("unbounded", f"browser_vault_{action}", False)
    # The in-app browser is the same fill behind the same origin check: the same reading, another call.
    preview = effect_of("browser_vault", {"action": "fill", "handle": "vault_card", "target": "preview"})
    assert (preview.cls, preview.tool, preview.target) == (fill.cls, fill.tool, fill.target)
    assert preview.digest != fill.digest
    # Only the vault tool is read by an action.
    assert read_as("terminal", {"action": "fill"}) == "terminal" and read_as(None, None) == ""
    assert read_as("browser_navigate", {"action": "fill"}) == "browser_navigate"


@pytest.mark.parametrize("args", [{}, {"handle": "vault_card"}, {"action": ""}, {"action": None}, {"action": "Fill"},
                                  {"action": " fill"}, {"action": "fill "}, {"action": "pay"}, {"action": ["fill"]},
                                  {"action": {"fill": True}}, {"action": 1}])
def test_a_missing_or_unknown_action_is_the_browser_call_it_is(args):
    """Hermes runs nothing for such a call (it reads the exact word), and the product reads no older tool into it:
    an unbounded browser call, which no grant covers, whatever the item."""
    assert read_as("browser_vault", args) == "browser_vault"
    effect = effect_of("browser_vault", args, context={"vault_kind": "login"})
    unknown = effect_of("browser_navigate", args)
    assert (effect.cls, effect.tool, effect.bounded) == ("unbounded", "browser_vault", False)
    assert (effect.cls, effect.bounded, effect.target) == (unknown.cls, unknown.bounded, unknown.target)
    assert executor_call(effect.tool) and not grantable(effect.tool)


def test_a_grant_and_a_deny_entry_name_the_older_tool_in_both_shapes(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    try:
        row, _ = store.create_intention(kind="task", type="upkeep", title="order the paper", drive="upkeep",
                                        cls="external", decision="act", decision_reason="trusted",
                                        status="dispatched", dedup_key="paper")
        grant = {"id": "shop-card", "class": "money_movement", "scope": "the shop's card",
                 "tools": ["browser_vault_fill"], "match": ["vault item vault_card"]}
        policy = Policy.from_config({"autonomy": "trusted", "grants": [grant]})
        assert [item.valid for item in policy.grants] == [True]
        guard = Authority(policy, store, owner_id=OWNER)
        payment = {"vault_kind": "payment"}
        for tool, args in FILLS:
            verdict = guard.guard(tool=tool, args=args, task=row, context=payment)
            assert verdict["allow"] is True and verdict["grant"] == "shop-card", (tool, verdict)
            assert verdict["effect"]["tool"] == "browser_vault_fill"
            other = guard.guard(tool=tool, args={**args, "handle": "vault_other"}, task=row, context=payment)
            assert other["action"] == "ask" and other["hold"] is True and "grant" not in other, tool
        # ``browser_vault`` itself would be every action at once: never a grant's name, and it covers nothing.
        wide = Policy.from_config({"autonomy": "trusted", "grants": [{**grant, "tools": ["browser_vault"]}]})
        assert [item.valid for item in wide.grants] == [False]
        for tool, args in FILLS:
            held = Authority(wide, store, owner_id=OWNER).guard(tool=tool, args=args, task=row, context=payment)
            assert held["action"] == "ask" and "grant" not in held, tool
        # The owner's deny entry for the older tool denies the call in both shapes, and only that action.
        denying = Authority(Policy.from_config({"autonomy": "trusted", "deny": {"tools": ["browser_vault_fill"]}}),
                            store, owner_id=OWNER)
        for tool, args in FILLS:
            verdict = denying.guard(tool=tool, args=args, task=row, context=payment)
            assert verdict["action"] == "block" and verdict["reason"] == "tool browser_vault_fill is on the deny list"
        assert denying.guard(tool="browser_vault", args={"action": "list"}, task=row).get("action") != "block"
        whole = Authority(Policy.from_config({"autonomy": "trusted", "deny": {"tools": ["browser_vault"]}}),
                          store, owner_id=OWNER)
        assert whole.guard(tool="browser_vault", args={"action": "list"}, task=row)["action"] == "block"
        assert whole.guard(tool="browser_vault_list", args={}, task=row).get("action") != "block"
    finally:
        store.close()


# A stand-in for Hermes' vault module in the two shapes its sources have. Older: one tool per action, and the fill
# asks its consent (with the ITEM's origin) before it looks at the page. Newer: one tool that calls the fill by
# keyword, and the fill checks the page's origin BEFORE it asks its consent (with the PAGE's origin).
VAULT_MODULE = '''
import json

SHAPE = %r
ITEMS = {"vault_card": ("payment", "Card", "https://shop.example"),
         "vault_twin": ("payment", "Card", "https://shop.example")}
PAGE = {"origin": "https://shop.example"}
ASKED = []


def _confirm_payment_fill(label, origin):
    ASKED.append((label, origin))
    return False        # a worker's session has no surface: nobody answers Hermes' own consent


def browser_vault_fill(handle, task_id=None, *, with_identifier=False):
    kind, label, origin = ITEMS[handle]
    if SHAPE == "newer" and PAGE["origin"] != origin:
        return json.dumps({"success": False, "error_type": "origin_mismatch"})
    if not _confirm_payment_fill(label, PAGE["origin"] if SHAPE == "newer" else origin):
        return json.dumps({"success": False, "error_type": "payment_declined"})
    if PAGE["origin"] != origin:
        return json.dumps({"success": False, "error_type": "origin_mismatch"})
    return json.dumps({"success": True, "filled": handle})


def _handle_vault(args, **kwargs):
    assert SHAPE == "newer" and args["action"] == "fill"
    return browser_vault_fill(handle=str(args.get("handle") or ""), task_id=kwargs.get("task_id"))
'''


def _plugin(shape, monkeypatch, tmp_path, mind=None):
    """The plugin's guard in a mind run, over a stand-in vault module of that shape and a sidecar that answers
    every guard request with the owner's yes to exactly that call (a permit)."""
    import tools
    from colonymind_hermes import guard
    module = types.ModuleType("tools.browser_vault_tool")
    exec(VAULT_MODULE % shape, module.__dict__)
    monkeypatch.setitem(sys.modules, "tools.browser_vault_tool", module)
    monkeypatch.setattr(tools, "browser_vault_tool", module, raising=False)
    monkeypatch.setattr(guard, "_PAYMENT_CONSENTS", {})
    monkeypatch.setattr(guard, "vault_meta", lambda handle: SimpleNamespace(kind=module.ITEMS[handle][0])
                        if handle in module.ITEMS else None)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    sent = []

    def post(path, **kwargs):
        sent.append(kwargs["json"])
        answer = {"allow": True, "action": "allow", "permit": "floor-task-1", "effect": kwargs["json"]["effect"]}
        return SimpleNamespace(status_code=200, is_success=True, json=lambda: answer)
    client = SimpleNamespace(has_mind_routes=lambda: True, post=post)
    sessions = SimpleNamespace(standing=lambda sid: ("owner", "sms"), is_owner=lambda sid: False,
                               standings=SimpleNamespace(get=lambda *args: None), get=lambda sid: None)
    settings = SimpleNamespace(key_file=tmp_path / "api.key", guests=lambda: {"executor_platforms": [], "tools": []},
                               mind=lambda: dict(mind or {}), worker_profile="colonymind-act")
    instance = guard.Guard(client, sessions, settings)
    instance._enabled_cache = (time.monotonic(), True)

    def fill(handle="vault_card", task="task-01"):
        """The fill as that source's tool runs it."""
        answer = module.browser_vault_fill(handle, task_id=task) if shape == "older" else \
            module._handle_vault({"action": "fill", "handle": handle}, task_id=task)
        return json.loads(answer)
    call = FILLS[0] if shape == "older" else FILLS[1]
    return guard, instance, module, sent, fill, call


@needs_hermes
@pytest.mark.parametrize("shape", ["older", "newer"])
def test_the_owners_yes_to_a_payment_fill_reaches_hermes_consent_once_in_both_shapes(shape, monkeypatch, tmp_path):
    guard, instance, module, sent, fill, (tool, args) = _plugin(shape, monkeypatch, tmp_path)
    assert fill()["error_type"] == "payment_declined"                   # no yes: Hermes' own consent, unanswered
    assert instance._rules(tool, args, "session-1", True, task_id="task-01") is None
    assert sent[-1]["tool"] == tool and sent[-1]["args"] == args        # the sidecar gets the call as Hermes has it
    assert sent[-1]["context"] == {"vault_kind": "payment"}             # ... with the item's kind, read by the plugin
    assert (sent[-1]["effect"]["cls"], sent[-1]["effect"]["tool"]) == ("money_movement", "browser_vault_fill")
    assert list(guard._PAYMENT_CONSENTS) == [("vault_card", "task-01")]
    assert fill("vault_twin")["error_type"] == "payment_declined"       # a twin item: same label, same site
    assert fill(task="task-02")["error_type"] == "payment_declined"     # the same item from another task
    asked = len(module.ASKED)
    assert fill() == {"success": True, "filled": "vault_card"}          # that item in that task
    assert len(module.ASKED) == asked                                   # ... and Hermes' own consent was not asked
    assert fill()["error_type"] == "payment_declined"                   # once
    assert guard._PAYMENT_CONSENTS == {}


@needs_hermes
@pytest.mark.parametrize("shape", ["older", "newer"])
def test_a_fill_of_a_login_or_a_denied_fill_stores_no_yes(shape, monkeypatch, tmp_path):
    guard, instance, module, sent, fill, (tool, args) = _plugin(shape, monkeypatch, tmp_path,
                                                                mind={"deny": {"tools": ["browser_vault_fill"]}})
    verdict = instance._rules(tool, args, "session-1", True, task_id="task-01")
    assert verdict["action"] == "block" and "browser_vault_fill is on the deny list" in verdict["message"]
    assert sent == [] and guard._PAYMENT_CONSENTS == {}
    guard, instance, module, sent, fill, (tool, args) = _plugin(shape, monkeypatch, tmp_path)
    module.ITEMS["vault_login"] = ("login", "Shop", "https://shop.example")
    assert instance._rules(tool, {**args, "handle": "vault_login"}, "session-1", True, task_id="task-01") is None
    assert sent[-1]["context"] == {"vault_kind": "login"} and sent[-1]["effect"] is None
    assert guard._PAYMENT_CONSENTS == {}                                # a login's fill is no payment: no yes to hand over
    assert fill()["error_type"] == "payment_declined"


@needs_hermes
@pytest.mark.parametrize("shape", ["older", "newer"])
def test_the_yes_ends_with_the_fill_it_was_given_for(shape, monkeypatch, tmp_path):
    """A newer Hermes refuses a fill on another origin's page BEFORE it asks its consent: the yes given for that fill
    must not wait for the next one. (An older Hermes asks first, which spends it.)"""
    guard, instance, module, sent, fill, call = _plugin(shape, monkeypatch, tmp_path)
    assert guard.consent_payment_fill("vault_card", "task-01") is True
    module.PAGE["origin"] = "https://elsewhere.example"
    assert fill()["error_type"] == "origin_mismatch"
    assert module.ASKED == []                                           # Hermes' own consent was never asked
    assert guard._PAYMENT_CONSENTS == {}
    module.PAGE["origin"] = "https://shop.example"
    assert fill()["error_type"] == "payment_declined"                   # the yes went with the fill it was given for
    # A fill that ends in an error ends it too.
    assert guard.consent_payment_fill("vault_card", "task-01") is True
    del module.ITEMS["vault_card"]
    with pytest.raises(KeyError):
        fill()
    module.ITEMS["vault_card"] = ("payment", "Card", "https://shop.example")
    assert guard._PAYMENT_CONSENTS == {} and fill()["error_type"] == "payment_declined"
