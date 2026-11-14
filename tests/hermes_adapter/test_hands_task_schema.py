"""The front sees her task fields on delegate_task (design 6.2): without them every hand defaulted to deep, read-only.

Found on the rc clone: the front delegated a shell task with no kind (Hermes' schema offers goal, context,
output_schema, images and group only), the deep hand could not run the command, and the front tried three times.
"""
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import typing

import pytest

from hands_fake_hermes import FakeController, FakeHermes, load_hands

NATIVE_ITEMS = {'type': 'object', 'properties': {'goal': {'type': 'string'}, 'context': {'type': 'string'}},
                'required': ['goal']}


def entry():
    schema = {'parameters': {'type': 'object', 'properties': {'tasks': {'type': 'array', 'items': NATIVE_ITEMS}}}}
    return SimpleNamespace(schema=schema, dynamic_schema_overrides=lambda: {
        'description': 'native limits', 'parameters': schema['parameters']})


def runtime_with(tool, tmp_path):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    native = FakeHermes()
    native.tool_entry = lambda name: tool if name == 'delegate_task' else None
    runtime = dispatch.Runtime(config={}, client=FakeController(), spool_dir=tmp_path / 'spool', native=native,
                               owner_check=lambda sid: True)
    native.runtime = runtime
    return runtime


def test_task_items_carry_her_fields_and_native_limits_stay(tmp_path):
    tool = entry()
    native = tool.dynamic_schema_overrides
    runtime = runtime_with(tool, tmp_path)
    runtime.install()
    try:
        value = tool.dynamic_schema_overrides()
        items = value['parameters']['properties']['tasks']['items']
        assert value['description'] == 'native limits'
        assert {'goal', 'context', 'kind', 'effort', 'risk', 'needs', 'accept', 'touches', 'workdir', 'name',
                'priority'} <= set(items['properties'])
        assert items['required'] == ['goal']
        assert 'kind' not in NATIVE_ITEMS['properties'], 'the native schema object is never changed'
        runtime.install()
        assert tool.dynamic_schema_overrides()['parameters'] == value['parameters'], 'installing twice adds nothing'
    finally:
        runtime.close()
    assert tool.dynamic_schema_overrides is native, 'close restores the native schema'


@pytest.mark.requires_sidecar
def test_fields_and_values_are_the_controllers_wire_contract():
    from colonymind.hands import wire
    package = load_hands()
    fields = importlib.import_module(package.__name__ + '.wrap').TASK_FIELDS
    model = wire.PlanTask.model_fields
    assert set(fields) <= set(model)
    def literal(annotation):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        values = []
        for arg in args:
            values += [m.value for m in arg] if isinstance(arg, type) and hasattr(arg, '__members__') else list(typing.get_args(arg)) or [arg]
        return values
    for name in ('kind', 'risk', 'priority'):
        assert sorted(fields[name]['enum']) == sorted(literal(model[name].annotation)), name
    need = typing.get_args(model['needs'].annotation)[0]
    assert sorted(fields['needs']['items']['enum']) == sorted(literal(need))
    assert set(fields['accept']['properties']) <= set(wire.AcceptanceCriteria.model_fields)


@pytest.mark.skipif(not os.environ.get('HERMES_SOURCE_ROOT'), reason='integrator: set HERMES_SOURCE_ROOT')
def test_pinned_hermes_tool_definitions_show_the_fields(tmp_path):
    code = '''
import importlib, json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from hands_fake_hermes import FakeController, load_hands
import tools.delegate_tool
from tools.registry import registry
package = load_hands()
dispatch = importlib.import_module(package.__name__ + '.dispatch')
native = dispatch.NativeHermes()
runtime = dispatch.Runtime(config={}, client=FakeController(), spool_dir=sys.argv[3], native=native, owner_check=lambda s: True)
runtime.install()
items = registry.get_definitions({'delegate_task'}, quiet=True)[0]['function']['parameters']['properties']['tasks']['items']
runtime.close()
after = registry.get_definitions({'delegate_task'}, quiet=True)[0]['function']['parameters']['properties']['tasks']['items']
print(json.dumps([sorted(items['properties']), items['properties']['kind']['enum'], sorted(after['properties'])]))
'''
    source = Path(os.environ['HERMES_SOURCE_ROOT'])
    here = Path(__file__).resolve().parent
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'), PYTHONDONTWRITEBYTECODE='1')
    (tmp_path / 'home').mkdir()
    done = subprocess.run([sys.executable, '-c', code, str(source), str(here), str(tmp_path / 'spool')],
                          capture_output=True, text=True, timeout=120, env=env, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    import json
    shown, kinds, after = json.loads(done.stdout.strip().splitlines()[-1])
    assert {'kind', 'effort', 'risk', 'needs', 'accept', 'touches'} <= set(shown)
    assert 'ops' in kinds and 'coding' in kinds
    assert 'kind' not in after
