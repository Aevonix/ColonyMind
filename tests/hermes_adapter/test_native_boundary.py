import pytest
from test_hands_definition_dispatch import setup
from test_hands_definition_pickup import enqueue, pickup
from test_hands_pickup_review import sessions_for
from hands_fake_hermes import Parent

# The borrowed fixture `setup` builds the sidecar's own controller.
pytestmark = pytest.mark.requires_sidecar


@pytest.mark.parametrize('change', ['identity', 'handles', 'chat_type', 'destination', 'mind'])
@pytest.mark.parametrize('boundary', ['queue_bind', 'dispatch_intent'])
def test_current_owner_required_at_native_launch(setup, tmp_path, monkeypatch, change, boundary):
    runtime, native, client, c, http, _ = setup
    sessions, handles = sessions_for(runtime, tmp_path, monkeypatch)
    parent = Parent()
    runtime.remember_parent(parent)
    queued = enqueue(setup)
    post = client.post
    changed = []
    def mutate():
        info = sessions.get('owner-session')
        if change == 'identity':
            parent.session_id = 'guest-session'
        elif change == 'handles':
            handles.clear()
        elif change == 'chat_type':
            info.chat_type = 'group'
        elif change == 'destination':
            info.chat_id = 'other-chat'
        else:
            info.inherited_standing = 'mind'
        changed.append(change)
    def intercept(path, body, key):
        response = post(path, body, key)
        if (boundary == 'queue_bind' and path.endswith('/definitions/queue/bind')
                or boundary == 'dispatch_intent' and path.endswith('/events')
                and any(e['type'] == 'dispatch_intent' for e in body.get('events', []))):
            mutate()
        return response
    monkeypatch.setattr(client, 'post', intercept)
    pickup(runtime)
    assert changed, 'The chosen boundary was not reached'
    hand, = runtime.registry.all()
    print({'boundary': boundary, 'change': change, 'native_calls': len(native.calls),
           'selected_parent': hand.parent_session_id, 'actual_parent': parent.session_id,
           'bound_child': hand.subagent_id, 'task_state': client.get('/v1/hands/tasks/' + queued['id'])['state']})
    assert not native.calls, 'Native dispatch used authority revoked after the pre-bind check'
    saved = client.get('/v1/hands/tasks/' + queued['id'])
    assert saved['state'] == 'waiting' and saved['status'] == 'failed'
    assert saved['status_detail']['attempt_state'] == 'refused'
    assert saved['status_detail']['outcome'] == 'waiting_for_main_session'
    queue = http.portal.call(c.one, 'SELECT * FROM definition_task_queue WHERE task_id=?', (queued['id'],))
    assert queue['dispatched'] == 1  # consumed even though native execution was refused
    assert runtime._pending.hand is None
    # Restoration, repeated polls and spool replay cannot execute a consumed binding.
    monkeypatch.setattr(client, 'post', post)
    parent.session_id = 'owner-session'
    handles['telegram'] = ['owner-handle']
    info = sessions.get(parent.session_id)
    info.chat_id, info.chat_type, info.inherited_standing = 'owner-chat', 'dm', ''
    runtime.spool.replay(runtime._send)
    pickup(runtime)
    pickup(runtime)
    assert not native.calls
    # The retained task remains addressable through the normal owner controls.
    assert client.post('/v1/hands/tasks/' + queued['id'] + '/cancel',
                       {'reason': 'Owner stops refused work'}, 'manual-cancel')['state'] == 'cancelled'


def test_native_construction_and_binding_keep_the_checked_identity(setup, monkeypatch):
    runtime, native, client, c, http, _ = setup
    parent = Parent()
    runtime.remember_parent(parent)
    queued = enqueue(setup)
    original = runtime.original_delegate
    def construct(**kwargs):
        # Native entry has passed the final check. Compression while constructing
        # a child must not change its registered parent or lifecycle-hook identity.
        parent.session_id = 'owner-compressed'
        launch = kwargs['parent_agent']
        assert launch.session_id == 'owner-session'
        launch.session_estimated_cost_usd = 1.5
        assert parent.session_estimated_cost_usd == 1.5
        with pytest.raises(AttributeError):
            launch.session_id = 'another-session'
        return original(**kwargs)
    monkeypatch.setattr(runtime, 'original_delegate', construct)
    pickup(runtime)
    hand, = runtime.registry.all()
    assert len(native.calls) == 1 and hand.subagent_id
    saved = client.get('/v1/hands/' + hand.hand_id)
    assert saved['state'] == 'running' and saved['child_session'] == hand.child_session_id
    assert saved['parent_session'] == hand.parent_session_id == 'owner-session'
    assert hand.task_id == queued['id']
