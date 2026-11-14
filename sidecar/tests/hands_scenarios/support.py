"""Fixtures use production runtime code; only external Hermes seams are faked."""

import ast
import importlib
import importlib.util
import inspect
import sys
import textwrap
from types import SimpleNamespace

import pytest

from hands_fake_hermes import FakeController, FakeHermes, Parent, load_hands
from .catalog import SCENARIOS, SafetyViolation, check


# Units this suite depends on that may not be merged yet. A scenario that needs
# one of them xfails only while the unit is absent, so the gate turns into a real
# test, and fails loudly, the moment the unit lands.
EVIDENCE_ABSENT = importlib.util.find_spec("colonymind.hands.evidence") is None


def needs_unit(missing, unit, what):
    if missing:
        pytest.xfail(f"{unit} not merged: {what}")


paired = pytest.mark.parametrize("mutated", [
    pytest.param(False, id="guard-on"),
    pytest.param(True, id="guard-off", marks=pytest.mark.hands_mutation),
])


def assert_scenario(number, observations, mutated):
    if mutated:
        with pytest.raises(SafetyViolation) as caught:
            check(number, observations)
        assert caught.value.code == SCENARIOS[number - 1].assertion
    else:
        check(number, observations)


def disable_branch(monkeypatch, owner, name, expression):
    """Disable exactly one production branch, or fail if the source seam moved.

    Mutated code stays in memory. This avoids editing another unit's owned files
    and avoids replacing the implementation with a fake result.
    """
    function = getattr(owner, name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    expected = ast.dump(ast.parse(expression, mode="eval").body)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.dump(node.test) == expected]
    assert len(matches) == 1, f"mutation seam moved: {owner.__name__}.{name}: {expression}"
    matches[0].test = ast.Constant(False)
    tree.body[0].decorator_list = []
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    monkeypatch.setattr(owner, name, namespace[name])


@pytest.fixture
def runtime_factory(tmp_path):
    rigs = []

    def create(*, max_running=1, directory=None, client=None):
        package = load_hands()
        module = lambda part: importlib.import_module(package.__name__ + "." + part)
        dispatch = module("dispatch")
        native, client = FakeHermes(capacity=max_running + 1), client or FakeController()
        config = {"faculty": True, "max_running": max_running, "platforms": ["whatsapp", "sms"],
            "contracts": {"text-route": {"version": "1", "family": "text-family", "modalities": ["text"],
                "context_tokens": 100000, "max_output_tokens": 2000, "tool_calls": True, "decode_tok_s": 100}},
            "notices": {"routine_per_day": 30, "actionable_per_day": 20}}
        runtime = dispatch.Runtime(config=config, client=client,
            spool_dir=directory or tmp_path / f"spool-{len(rigs)}", native=native,
            owner_check=lambda sid: sid.startswith("owner"))
        native.runtime = runtime
        runtime.install()
        package.register_route_resolver(lambda role, effort: native.native_policy())
        parent = Parent()

        def start(key="start", **task):
            answer = runtime.dispatch_tasks([{"goal": "Check the artifact", "kind": "deep", **task}],
                                            parent=parent, tool_call_id=key)["hands"][0]
            assert answer["status"] == "dispatched", answer
            return runtime.registry.get(answer["hand_id"])

        rig = SimpleNamespace(runtime=runtime, native=native, client=client, module=module, package=package,
                              parent=parent, start=start)
        rigs.append(rig)
        return rig

    yield create
    for rig in reversed(rigs):
        rig.runtime.close()
        for name in list(sys.modules):
            if name == rig.package.__name__ or name.startswith(rig.package.__name__ + "."):
                sys.modules.pop(name, None)
