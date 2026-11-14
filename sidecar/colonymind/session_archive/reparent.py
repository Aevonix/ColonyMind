"""Recurring-delivery reparenting for the session archive. No routes, no store.

In production this module runs ONLY on the Hermes side, in the gateway's venv.
The sidecar never imports it or cron. The adapter packages this same source via
``colonymind_hermes.archive_core``; relative imports keep that package independent.

This is the integration seam ``ArchiveStore`` documents: one synchronous callable
driven through validate, plan, apply, commit and rollback, with the archive
operation row as the single durable intent. It owns no state of its own.

Only recurring DELIVERY moves (the session-archive contract v1.5). A delivery is a Hermes CORE
cron job, retargeted by CALLING the public ``cron.jobs`` API; core is not edited
and no second scheduler appears. Nothing local moves, so the overlay commit stays
a single-store transaction and ``commit`` has nothing to stage.

Three rules the tests pin, each one a way this could lie to the owner:

* A job is in scope only when its recorded ``origin`` equals ``source_origin``
  exactly AND CORE'S OWN RESOLVER puts one of the job's delivery targets in that
  chat. We no longer re-implement core's delivery grammar: two rounds of review
  found two different branches our rule missed, each one leaving a live delivery
  aimed at the archived chat while the receipt claimed it had moved.
* ``main_origin`` is SET, never patched, so a side session's ``thread_id`` cannot
  survive the move. No other field changes, ``attach_to_session`` included.
* Every mutation is proved by an independent read of the public job API, before
  and after. ``update_job``'s return value is never trusted, and neither is an
  earlier attempt's readback.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from .store import ArchiveError


def _core():
    """The public core cron API, imported late so this module loads without Hermes."""
    from cron.jobs import get_job, list_jobs, update_job
    return list_jobs, get_job, update_job


def _resolver():
    """Core's own delivery resolver, so classification cannot drift from delivery.

    ``cron.scheduler_delivery._resolve_delivery_targets`` is PRIVATE core API and
    this is a deliberate exception to standing on the public surface. The public
    ``cron_delivery_targets()`` lists configured platforms, not a job's targets,
    and nothing else in core answers "where would this job actually deliver".
    Re-deriving that answer is what failed twice: ``origin,local`` delivers to the
    origin, and a BARE PLATFORM token equal to the origin's platform ALSO falls
    back to the origin chat when that platform has no configured home channel
    (``scheduler_delivery._resolve_single_delivery_target``). Calling core settles
    every such branch, including ones we have not read.

    Absence or failure is a LOUD refusal, never a fallback to a rule of our own.
    A refused archive costs the owner a retry; a false receipt costs them a
    reminder that still fires into a chat they archived.

    Resolution runs in the gateway process, with the scheduler's config and
    environment. No classification or mutation crosses the process boundary.
    """
    try:
        from cron import scheduler_delivery
    except Exception as exc:
        raise ArchiveError('uncertain', reason='core_resolver_unavailable') from exc
    resolve = getattr(scheduler_delivery, '_resolve_delivery_targets', None)
    if not callable(resolve):
        raise ArchiveError('uncertain', reason='core_resolver_unavailable')

    cache: dict = {}

    def targets(job: Mapping[str, Any]) -> list:
        # Both lanes: ``deliver`` routes output and ``failure_deliver`` independently
        # routes failure reports, so a job with ``deliver: local`` can still report
        # failures into the chat. Either one is a live delivery the owner would see.
        #
        # Memoised on the three fields resolution depends on, because scope asks
        # core about each job several times per phase (as it stands, and with Main
        # as its origin) and a home-resolving job costs real milliseconds: a
        # review measured about 1 s per 100 such jobs without this. The key is the
        # job's own content, so a changed row is a different key and never a stale
        # answer; the cache lives for one phase only.
        key = _cache_key(job)
        if key in cache:
            return cache[key]
        found = []
        for for_failure in (False, True):
            try:
                resolved = resolve(dict(job), for_failure=for_failure)
            except Exception as exc:
                raise ArchiveError('uncertain', reason='core_resolver_failed') from exc
            if isinstance(resolved, list):
                found.extend(item for item in resolved if isinstance(item, Mapping))
        cache[key] = found
        return found

    return targets


def _cache_key(job: Mapping[str, Any]) -> str:
    return json.dumps([job.get('id'), job.get('origin'), job.get('deliver'),
                       job.get('failure_deliver')], sort_keys=True, default=str)


def _identity(origin: Mapping[str, Any]) -> tuple:
    thread = origin.get('thread_id')
    return (str(origin['platform']).lower(), str(origin['chat_id']),
            str(thread) if thread is not None else None)


def _target_identity(target: Mapping[str, Any]) -> tuple | None:
    if target.get('platform') is None or target.get('chat_id') is None:
        return None
    thread = target.get('thread_id')
    return (str(target['platform']).lower(), str(target['chat_id']),
            str(thread) if thread is not None else None)


def _matches(origin: Any, source: Mapping[str, Any]) -> bool:
    # Explicit dict origins only: a provenance string or a hand-edited jobs.json
    # entry is not an address, and core treats those as missing too. No home-channel
    # inference and no unpinned-thread wildcard; a thread is a different session.
    return (isinstance(origin, dict) and bool(origin.get('platform'))
            and bool(origin.get('chat_id')) and _identity(origin) == _identity(source))


def _delivers_to(targets, job: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    """Whether core resolves any of this job's delivery targets to ``where``.

    A job that merely records the chat as provenance resolves no target there and
    is left alone: moving it would change no delivery and would rewrite
    provenance. A chat-level delivery is not a thread's delivery and the reverse
    is also true, which is why the thread id is part of the identity.
    """
    want = _identity(where)
    return any(_target_identity(target) == want for target in targets(job))


def _origin_now(get_job, job_id: str):
    job = get_job(job_id)
    if not isinstance(job, Mapping) or job.get('id') != job_id:
        raise ArchiveError('source_unavailable')
    return job.get('origin')


def _job_now(get_job, job_id: str) -> Mapping[str, Any]:
    """One read, so origin and delivery lanes come from the SAME snapshot.

    Reading them separately let a job change origin between the two reads and
    pass both checks, which is how a job already back at the archived chat got
    counted as moved.
    """
    job = get_job(job_id)
    if not isinstance(job, Mapping) or job.get('id') != job_id:
        raise ArchiveError('source_unavailable')
    return job


def _with_origin(job: Mapping[str, Any], origin: Mapping[str, Any]) -> dict:
    probe = dict(job)
    probe['origin'] = deepcopy(dict(origin))
    return probe


def _origin_decides_delivery(targets, job: Mapping[str, Any],
                             source: Mapping[str, Any], main: Mapping[str, Any]) -> bool:
    """Whether retargeting ``origin`` actually moves this job's delivery.

    Asked by DRY RUN against core rather than by reading tokens: resolve the job
    as it stands, then resolve a copy whose origin is Main, and keep only the jobs
    whose delivery into the archived chat GOES AWAY when the origin moves. That is
    the only question the owner cares about, and it is the one our token rules kept
    getting wrong in both directions.

    It settles the cases a token rule cannot. An explicit ``whatsapp:chat-side``
    still resolves to the archived chat with Main as the origin, so moving the
    origin would change nothing while the receipt claimed it had: not in scope. A
    bare platform token that falls back to the origin chat DOES follow the origin:
    in scope. A home channel that happens to be the archived chat follows neither:
    not in scope, and honestly so, because no origin change can redirect it.
    """
    if not _delivers_to(targets, job, source):
        return False
    return not _delivers_to(targets, _with_origin(job, main), source)


def _delivery_survives_the_move(targets, job: Mapping[str, Any],
                                main: Mapping[str, Any]) -> bool:
    """Whether the job still delivers somewhere the owner sees once it moves.

    A move that silently drops the delivery is worse than a refusal, so a job
    whose Main-origin copy resolves to no target at Main is named at plan time
    instead of being mutated and rolled back.
    """
    return _delivers_to(targets, _with_origin(job, main), main)


def _plan(session_id: str, request: Mapping[str, Any]) -> dict:
    list_jobs, _, _ = _core()
    targets = _resolver()
    source, main = request['source_origin'], request['main_origin']
    # include_disabled: a paused recurring delivery must move too, or it wakes up
    # later still aimed at a chat the owner archived.
    candidates, seen = [], set()
    for job in list_jobs(include_disabled=True):
        if not isinstance(job, Mapping) or not _matches(job.get('origin'), source):
            continue
        if not _origin_decides_delivery(targets, job, source, main):
            continue
        ident = job.get('id')
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ArchiveError('source_unavailable')
        seen.add(ident)
        candidates.append((ident, job))
    candidates.sort(key=lambda pair: pair[0])
    if not request['reparent_to_main']:
        # Declined reparenting still has to be reported honestly: the archive
        # succeeds, and tasks_reparented is true only because there was nothing to
        # move. NOTHING is examined for safety here, because nothing will be
        # touched: running the delivery-loss check on a hypothetical move refused a
        # valid declined request over a job this operation was never going to move.
        return {'jobs': [], 'declined': [ident for ident, _job in candidates]}
    lost = [ident for ident, job in candidates
            if not _delivery_survives_the_move(targets, job, main)]
    if lost:
        raise ArchiveError('uncertain', refused=sorted(lost),
                           reason='delivery_would_be_lost')
    jobs = [{'id': ident, 'original_origin': deepcopy(job['origin']),
             'intended_origin': deepcopy(dict(main))} for ident, job in candidates]
    return {'jobs': jobs, 'declined': []}


def _apply(request: Mapping[str, Any], intent: Mapping[str, Any], record) -> dict:
    """Retarget every in-scope job, then prove the WHOLE SET once more.

    ``record`` durably saves the intent. It is called BEFORE each core mutation,
    because an update can persist and then raise, or the process can die between
    the two: rollback must know which jobs this operation touched, and must not
    compensate a job it never attempted. A second flag, ``moved``, is written only
    after a verified readback, so rollback can tell our move from someone else's.

    The final scan RE-LISTS jobs rather than re-reading the planned set, because a
    job that becomes in-scope while we work is a delivery the receipt would
    otherwise claim had moved. The gateway adapter holds the checked core cron
    lock around the entire unit, including this scan and compensation. Direct
    callers must provide equivalent exclusion for concurrent core writers.
    """
    list_jobs, get_job, update_job = _core()
    targets = _resolver()
    source, main = request['source_origin'], request['main_origin']
    for job in intent['jobs']:
        current = _job_now(get_job, job['id'])
        origin = current.get('origin')
        if origin == job['intended_origin'] and job.get('attempted'):
            # Already where we want it, from an earlier attempt of THIS operation
            # (its durable flag was written before that update). Ours to restore.
            job['moved'] = True
            record(intent)
            continue
        if origin != job['original_origin']:
            # Someone moved this job after the intent was recorded, and no attempt
            # of ours explains it. Replaying a stale intended origin over that
            # would silently undo their change, and claiming it as ours would make
            # rollback drag their work back.
            raise ArchiveError('uncertain', conflicted=[job['id']],
                               reason='job_moved_by_another')
        if not _origin_decides_delivery(targets, current, source, main):
            # The lane changed between the plan and now: this job no longer
            # delivers into the archived chat, so the contract says leave it alone.
            # Checked BEFORE the mutation, so a provenance-only job never reaches
            # update_job, whose _fill_missing_next_run can refuse an unrelated
            # enabled one-shot outright and fail the owner's archive for it.
            raise ArchiveError('uncertain', conflicted=[job['id']],
                               reason='delivery_lane_changed')
        job['attempted'] = True
        record(intent)
        try:
            update_job(job['id'], {'origin': deepcopy(job['intended_origin'])})
        except Exception as exc:
            # Core refuses some updates outright: _fill_missing_next_run raises for
            # an enabled one-shot whose time has passed. Only in-scope jobs ever
            # reach this call, and the refusal names the job rather than leaving
            # the owner with an archive that fails for no stated reason.
            raise ArchiveError('uncertain', refused=[job['id']],
                               reason='cron_update_refused') from exc
        if _origin_now(get_job, job['id']) != job['intended_origin']:
            raise ArchiveError('uncertain', unverified=[job['id']])
        job['moved'] = True
        record(intent)
    return _verify_delivery(request, intent, list_jobs(include_disabled=True), targets)


def _verify_delivery(request, intent, jobs, targets=None):
    """Read-only proof, also rerun by the gateway after the child commits."""
    targets = _resolver() if targets is None else targets
    source, main = request['source_origin'], request['main_origin']
    stale = []
    if request['reparent_to_main']:
        # Only a reparenting archive promises that nothing is left aimed at the
        # archived chat, and it proves BOTH halves of that promise from ONE
        # snapshot: every job we moved, and every job now in scope at the source.
        # Verifying the planned jobs from their own separate reads let a job pass
        # whose lane had changed to an explicit source target after we moved it,
        # because the inventory filter only looks at rows still ORIGINATING at the
        # source and that row no longer did. One listing, both questions.
        snapshot = {}
        for job in jobs:
            if not isinstance(job, Mapping):
                continue
            ident = job.get('id')
            snapshot[ident if isinstance(ident, str) and ident else 'unknown'] = job
        for job in intent['jobs']:
            current = snapshot.get(job['id'])
            if (current is None or current.get('origin') != job['intended_origin']
                    or not _delivers_to(targets, current, main)
                    or _delivers_to(targets, current, source)):
                stale.append(job['id'])
        for ident, job in snapshot.items():
            if not _matches(job.get('origin'), source):
                continue
            if not _origin_decides_delivery(targets, job, source, main):
                continue
            stale.append(ident)
    # A declined archive promises the opposite: a job still in scope at the source
    # is the expected outcome, and nothing was moved, so there is nothing to verify.
    if stale:
        raise ArchiveError('uncertain', unverified=sorted(set(stale)))
    return {'reparented': {'cron_jobs': len(intent['jobs'])},
            'tasks_reparented': not intent.get('declined')}


def _rollback(intent: Mapping[str, Any]) -> dict:
    """Restore the jobs THIS operation moved, and nothing else.

    Only a job this operation actually moved is compensated: ``moved`` is written
    after a verified readback, and ``attempted`` covers the window where an update
    may have persisted before a raise or a crash. An attempted job found anywhere
    other than where we aimed it was never moved by us, so it is left exactly as
    it is and named, rather than being set back over a third party's change.

    A job we cannot READ is also left alone and named. Writing an origin without
    first establishing the job's current one is how an independent move got
    overwritten after a transient read failure.

    Idempotent by construction: each restore is a SET to a known value followed by
    a read, so a rollback interrupted anywhere is simply replayed. A job that
    cannot be verified back at its original origin is named, not folded into a
    count the owner cannot act on.
    """
    _, get_job, update_job = _core()
    restored, left, conflicted, unreadable = 0, [], [], []
    for job in intent.get('jobs', []):
        if not (job.get('moved') or job.get('attempted')):
            continue
        # Every write is preceded by its OWN read, including the retry: the retry
        # exists because an update can persist and then raise, and in exactly that
        # case the job may already be back, or may have been moved on by someone
        # else between our attempts. Writing from the first read's verdict a second
        # time is how an independent move got overwritten.
        outcome = None
        for _attempt in range(3):
            try:
                current = _origin_now(get_job, job['id'])
            except Exception:
                outcome = 'unreadable'
                break
            if current == job['original_origin']:
                outcome = 'restored'
                break
            if current != job['intended_origin']:
                # Neither where we put it nor where it started: it moved again
                # after we touched it, or it was never ours. Setting it back would
                # overwrite that change, so it is reported instead.
                outcome = 'conflicted'
                break
            if _attempt == 2:
                break           # Three reads, two writes; the last read decides.
            try:
                update_job(job['id'], {'origin': deepcopy(job['original_origin'])})
            except Exception:
                pass                # An update can persist and then raise.
        if outcome == 'restored':
            restored += 1
        elif outcome == 'conflicted':
            conflicted.append(job['id'])
        elif outcome == 'unreadable':
            unreadable.append(job['id'])
        else:
            left.append(job['id'])
    left.extend(unreadable)
    result = {'restored': {'cron_jobs': restored}, 'half_applied': bool(left),
              'jobs_left_moved': sorted(set(left))}
    if conflicted:
        # Not ours to restore and not left moved by us: named so the owner is told
        # rather than having it disappear into a count.
        result['jobs_moved_elsewhere'] = sorted(set(conflicted))
    return result


def archive_reparent(*, phase: str, session_id: str, request: Mapping[str, Any],
                     intent: Mapping[str, Any] | None,
                     connection=None, record=None):
    if phase == 'validate':
        # A receipt read carries no request; there is nothing for this engine to
        # refuse there. Whether the session exists is the gateway's to answer.
        if request and request.get('reparent_to_main') is not None:
            for key in ('source_origin', 'main_origin'):
                origin = request.get(key)
                if not isinstance(origin, Mapping) or not origin.get('platform') or not origin.get('chat_id'):
                    raise ArchiveError('invalid_request')
            if _identity(request['source_origin']) == _identity(request['main_origin']):
                raise ArchiveError('invalid_request')
        return None
    if phase == 'plan':
        return _plan(session_id, request)
    if phase == 'apply':
        return _apply(request, intent, record if record is not None else lambda _intent: None)
    if phase == 'commit':
        return None             # Nothing local moves; the overlay receipt is the commit.
    if phase == 'rollback':
        return _rollback(intent)
    raise ArchiveError('source_unavailable')
