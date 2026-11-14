"""A held ops stop has honest checks and keeps hold provenance after restart."""
from hands_fake_hermes import Parent
from test_hands_dispatch import setup  # noqa: F401
from test_hands_runtime_recovery import factory  # noqa: F401


def test_ops_hold_stop_without_planned_probe_has_no_typeerror_receipt(setup):
    runtime, native, client = setup
    runtime.dispatch_tasks([{'goal': 'Inspect, then await approval', 'kind': 'ops'}], parent=Parent())
    hand, = runtime.registry.all()
    hand.plan['acceptance'] = {}
    hand.activity, hand.blocked = 'awaiting_approval', True
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_status='completed',
                          child_summary='The exact call is held for the owner.')
    result = next(body for path, body, _ in client.writes if path.endswith('/result'))
    assert result['status'] == 'blocked'
    assert result['receipts'] == []


def test_recovered_hold_retains_executor_provenance_for_silent_completion(factory):
    runtime, native, client = factory()
    runtime.dispatch_tasks([{'goal': 'Inspect, then await approval', 'kind': 'ops'}], parent=Parent())
    hand, = runtime.registry.all()
    runtime.emit(f'/v1/hands/{hand.hand_id}/hold', {
        'epoch': runtime.epoch, 'call': {'tool': 'terminal', 'arguments': {'command': 'rm fixture.txt'}},
        'workdir': '/work', 'targets': []}, 'hold-restart')
    hand.plan['acceptance'] = {}
    hand.activity, hand.blocked = 'awaiting_approval', True
    runtime.subagent_stop(child_session_id=hand.child_session_id, child_status='completed',
                          child_summary='The exact call is held for the owner.')
    runtime.close()
    restored, _, _ = factory(client=client)
    recovered = restored.registry.get(hand.hand_id)
    assert recovered.blocked and recovered.activity == 'awaiting_approval'
    assert recovered.parent_session_id == hand.parent_session_id
    assert recovered.delegation_id == hand.delegation_id
