"""Durable owner grants schedule one normal continuation of the holding hand."""
from copy import deepcopy
import json

from .spool import Spool


def original_workspace(registry, hand):
    """The workspace of the task's first attempt, or None.

    The controller binds every continuation to its predecessor's task, so the root
    of the local continuation chain holds the workspace the owner's task was
    planned with. An approval never widens it to the held call's directory, even
    for a continuation recorded before this rule. A broken chain yields no
    workspace, so every deletion needs its own approval.
    """
    seen, current = set(), hand
    while current is not None and current.hand_id not in seen:
        seen.add(current.hand_id)
        task = current.plan.get('task') or {}
        if not task.get('continue'):
            return (current.plan.get('workspace') or current.plan.get('workdir') or current.plan.get('cwd')
                    or task.get('workdir') or None)
        current = registry.get(task['continue'])
    return None


class ApprovalResumes:
    def __init__(self, runtime):
        self.runtime = runtime
        self.spool = Spool(runtime.spool_dir / 'approval-resumes')

    def existing(self, hand):
        for row in self.spool.records():
            if row['path'] == 'approval-resume' and row['body']['hand_id'] == hand.hand_id:
                return {'status': 'approval_resume', 'hand_id': hand.hand_id,
                        'permit_id': row['key'], 'result': self.spool.response(row['key'])}
        return None

    def cancel(self, hand):
        from .dispatch import DISPATCH_LOCK
        with DISPATCH_LOCK:
            hand.plan['_approval_resume_stopped'] = True
            key = 'stop:' + hand.hand_id
            self.spool.append('approval-cancel', {'hand_id': hand.hand_id}, key)
            self.spool.acknowledge(key, {'stopped': True})

    def cancelled(self, hand):
        return hand.plan.get('_approval_resume_stopped') or any(
            row['path'] == 'approval-cancel' and row['body']['hand_id'] == hand.hand_id
            for row in self.spool.records())

    def enqueue(self, session, grants, *, wake=True):
        runtime = self.runtime
        if not runtime.authorized_owner(session):
            return ''
        accepted = []
        for grant in grants:
            if not isinstance(grant, dict) or grant.get('state', 'granted') != 'granted':
                continue
            hand = runtime.registry.get(grant.get('hand_id'))
            if (hand is None or hand.parent_session_id != session or grant.get('parent_session') != session
                    or hand.activity != 'awaiting_approval' or not grant.get('permit_id')
                    or not grant.get('ask_id') or not isinstance(grant.get('call'), dict)
                    or not grant.get('workdir') or self.cancelled(hand)):
                continue
            key = grant['permit_id']
            body = {name: deepcopy(grant[name]) for name in
                    ('permit_id', 'ask_id', 'hand_id', 'parent_session', 'call', 'workdir')}
            self.spool.append('approval-resume', body, key)
            accepted.append(hand.hand_id)
        if not accepted:
            return ''
        if wake:
            runtime._resume_ready.set()
        return ('The runtime is resuming hand ' + ', '.join(dict.fromkeys(accepted)) +
                ' automatically. It waits for the current attempt to finish, then runs the held call '
                'through its one-use permit and reports through the normal hand completion. '
                'Do not continue it again or perform the action yourself.')

    def pump(self):
        from .dispatch import DISPATCH_LOCK, TERMINAL
        runtime = self.runtime
        sessions = {hand.parent_session_id for hand in runtime.registry.all()
                    if hand.activity == 'awaiting_approval'}
        for session in sessions:
            if not runtime.authorized_owner(session):
                continue
            try:
                response = runtime.client.get('/v1/hands/permits/held', params={'session': session})
                if response.get('refreshed') is not True:
                    continue
                rows = response['permits']
            except Exception:
                continue
            current = {row['permit_id']: row for row in rows
                       if row.get('state') == 'granted' and row.get('parent_session') == session}
            with DISPATCH_LOCK:
                self.enqueue(session, list(current.values()), wake=False)
                for row in self.spool.pending():
                    if row['path'] != 'approval-resume':
                        continue
                    grant = row['body']
                    if grant['parent_session'] != session or row['key'] not in current:
                        continue
                    # Recheck the binding before dispatch; a queued notification grants nothing.
                    if any(current[row['key']].get(key) != value for key, value in grant.items()):
                        continue
                    hand = runtime.registry.get(grant['hand_id'])
                    if (hand is None or hand.status not in TERMINAL or not hand.plan.get('_result_submitted')
                            or hand.status == 'unknown_effect' or runtime.effects.open_effects(hand.hand_id)
                            or self.cancelled(hand)):
                        continue
                    durable = runtime.native.ledger(hand.delegation_id)
                    if not durable or durable.get('state', durable.get('status')) not in {
                            'completed', 'done', 'success', 'failed', 'error', 'interrupted', 'cancelled'}:
                        continue
                    running = sum(other.status not in TERMINAL and other.status != 'planned'
                                  and other.activity not in {'awaiting_input', 'awaiting_approval'}
                                  for other in runtime.registry.all())
                    if (runtime.parents.get(session) is None or not runtime.config.get('faculty', True)
                            or runtime.degraded.is_degraded
                            or running >= int(runtime.config.get('max_running', 1))
                            or runtime.native.active_count() + 1 >= runtime.native.max_children()):
                        continue
                    task = deepcopy(hand.plan['task'])
                    # The permit's directory is where the held call runs, never a workspace grant: the
                    # continuation keeps the original task's workspace, or has none.
                    task.pop('workdir', None)
                    workspace = original_workspace(runtime.registry, hand)
                    if workspace:
                        task['workdir'] = workspace
                    task['continue'] = hand.hand_id
                    if isinstance(hand.plan.get('output_schema'), dict):
                        task['output_schema'] = deepcopy(hand.plan['output_schema'])
                    task['context'] = (task.get('context', '') + '\nThe owner approved this exact held call: ' +
                        json.dumps(grant['call'], sort_keys=True) + '\nRun it in ' + grant['workdir'] +
                        ' through the normal permit check, then finish and report the result. '
                        'The approval covers only this call and its bound target values. '
                        'It does not widen your workspace: any other deletion or floor effect needs its own approval.')
                    context = dict(hand.plan.get('origin_context') or runtime.contexts[session])
                    execution = hand.plan.get('_execution_context') or runtime.contexts[session].get('_execution_context')
                    if execution is not None:
                        context['_execution_context'] = execution
                    supported = (execution.copy().run(runtime.native.push_supported, context) if execution
                                 else runtime.native.push_supported(context))
                    if not supported:
                        continue
                    # The exact approved call goes through the permit check even where a workspace
                    # exemption would otherwise let it run, so it consumes this one-use permit.
                    result = runtime.dispatch_tasks([task], session_id=session,
                        plan_key='approval-resume:' + row['key'], context_override=context,
                        approved_call={'permit_id': row['key'], 'call': deepcopy(grant['call']),
                                       'workdir': grant['workdir']})
                    # Dispatch records its intent before starting. Reusing this stable plan key after a
                    # crash returns that attempt, including an uncertain outcome, without executing again.
                    if not any(answer.get('retry') for answer in result.get('hands', [])):
                        self.spool.acknowledge(row['key'], result)
