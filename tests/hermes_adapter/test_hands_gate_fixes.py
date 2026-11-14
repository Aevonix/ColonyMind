"""Two defects the rc clone gate found in the plugin's planning path."""
import importlib
from types import SimpleNamespace

from hands_fake_hermes import FakeController, FakeHermes, Parent, load_hands


class QueueingController(FakeController):
    """A controller with no free slot: every plan answers queued."""

    def post(self, path, body, key):
        answer = super().post(path, body, key)
        if path.endswith('/plan'):
            for plan in answer['plans']:
                plan.update(admit='queued', reason='capacity')
                plan.pop('lease_token', None)
        return answer


def runtime(tmp_path, client):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    native = FakeHermes()
    config = {'faculty': True, 'max_running': 1, 'platforms': ['whatsapp', 'sms'],
              'contracts': {'text-route': {'version': '1', 'family': 'text-family', 'modalities': ['text'],
                                           'context_tokens': 100000, 'max_output_tokens': 2000,
                                           'tool_calls': {'qualified': True, 'success_share': 1.0}, 'decode_tok_s': 100}}}
    rt = dispatch.Runtime(config=config, client=client, spool_dir=tmp_path / 'spool', native=native,
                          owner_check=lambda sid: sid.startswith('owner'))
    native.runtime = rt
    rt.install()
    package.register_route_resolver(lambda role, effort: native.native_policy())
    return rt, native, dispatch


def test_unsupported_platform_is_refused_before_anything_is_planned(tmp_path):
    client = QueueingController()
    rt, native, _ = runtime(tmp_path, client)
    try:
        for platform in ('rcs', 'api', 'voice', 'unknown'):
            native.platform = platform
            answer = rt.dispatch_tasks([{'goal': 'check the artifact'}], parent=Parent(), tool_call_id=platform)['hands'][0]
            assert answer['status'] == 'refused' and answer['reason'] == 'platform', answer
        assert not [w for w in client.writes if w[0].endswith('/plan')], 'no plan reaches the controller'
        assert native.calls == [] and native.inline == []
        native.platform = 'sms'
        answer = rt.dispatch_tasks([{'goal': 'check the artifact'}], parent=Parent(), tool_call_id='sms')['hands'][0]
        assert answer['status'] == 'queued', 'an allowed platform still plans (and here queues)'
    finally:
        rt.close()


def test_planning_waits_for_the_brief_other_calls_stay_short(tmp_path, monkeypatch):
    package = load_hands()
    dispatch = importlib.import_module(package.__name__ + '.dispatch')
    seen = []

    class Client:
        def __init__(self, timeout, trust_env):
            seen.append(timeout)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, method, url, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    import httpx
    monkeypatch.setattr(httpx, 'Client', Client)
    client = dispatch.ControllerClient(SimpleNamespace(api_key='k', sidecar_url='http://127.0.0.1:9'))
    client.post('/v1/hands/plan', {}, 'plan-key')
    client.post('/v1/hands/h-1/events', {}, 'events-key')
    client.get('/v1/hands/h-1')
    assert seen[0] >= 10 and seen[1] == seen[2] == 2


class OneSlotController(FakeController):
    """A controller with one slot: the first plan starts, every later plan queues (the stage 1 controller)."""

    def post(self, path, body, key):
        first = path.endswith('/plan') and key not in self.responses
        answer = super().post(path, body, key)
        if first:
            started = sum(1 for p, _, _ in self.writes if p.endswith('/plan')) > 1
            for plan in answer['plans']:
                if started:
                    plan.update(admit='queued', reason='capacity')
                    plan.pop('lease_token', None)
        return answer


def test_an_outage_replay_sends_only_work_that_ran(tmp_path):
    """Found on the rc clone (scenario 9): plans refused during the outage were still spooled and replayed; the
    first took the only slot, the hand that did run could not be adopted, and its records stayed pending."""
    client = OneSlotController()
    rt, native, dispatch = runtime(tmp_path, client)
    rt.config['max_running'] = 4
    try:
        parent = Parent()
        client.down = True
        first = rt.dispatch_tasks([{'goal': 'first'}], parent=parent, tool_call_id='first')['hands'][0]
        assert first['reason'] == 'controller_unavailable'
        ran = rt.dispatch_tasks([{'goal': 'second'}], parent=parent, tool_call_id='second')['hands'][0]
        assert ran['status'] == 'dispatched', ran
        own = rt.dispatch_tasks([{'goal': 'initiative', 'priority': 'own'}], parent=parent, tool_call_id='own')['hands'][0]
        third = rt.dispatch_tasks([{'goal': 'other owner work'}], parent=parent, tool_call_id='third')['hands'][0]
        assert own['reason'] == 'degraded_owner_only' and third['reason'] == 'slots'
        hand = rt.registry.get(ran['hand_id'])
        rt.subagent_stop(child_session_id=hand.child_session_id, child_status='completed', child_summary='result')
        client.down = False
        rt.maintain()
        rt.maintain()
        goals = [body['tasks'][0]['goal'] for path, body, _ in client.writes if path.endswith('/plan')]
        assert goals == ['second'], goals
        assert rt.spool.pending() == []
        assert len(native.calls) == 1
    finally:
        rt.close()
