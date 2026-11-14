"""One independent watch pass. Observations never imply authority to stop work."""
from contextlib import closing
from datetime import datetime, timezone
import json
import logging
import os
import signal
import subprocess
import time
from uuid import uuid4

import httpx

from colonymind.health_counters import bump
from colonymind.work.hermes_jobs import ACTIVE, digest, failed, read_jobs
from . import store
from .activity import probe
from .network import fetch
from .settings import DEPENDENCY_FAILURE_PASSES, NOTICE_TIMEOUT_S

log = logging.getLogger(__name__)
NETWORK_ERRORS = (OSError, TimeoutError, httpx.HTTPError, ValueError, TypeError, KeyError)
MESSAGES = {
    'stuck': ('{subject} has used no CPU and printed nothing for {minutes:g} minutes.',
              '{subject} has resumed or ended.'),
    'repeat': ('{subject} has failed repeatedly with the same error.', '{subject}: repeated failure has cleared.'),
    'limit': ('{subject} has passed its declared runtime or turn limit.', '{subject}: over-limit task has ended.'),
    'download': ('Download {name} has had no sources for an hour.',
                 'Download {name} has sources again or has completed.'),
    'sidecar': ('ColonyMind is down or frozen. Job watch is still running independently.',
                'ColonyMind has recovered. Job watch is still running independently.'),
    'endpoint': ('Watched endpoint {name} is unavailable.', 'Watched endpoint {name} has recovered.'),
    'counters': ('ColonyMind reported new counts: {counts}.', ''),
}


def job_notice(job):
    return {'name': job.label, 'kind': job.kind, 'started_at': job.started_at}


def notice_message(check, state, data):
    fields = dict(data)
    if check in {'stuck', 'repeat', 'limit'}:
        started = data['started_at']
        clock = datetime.fromtimestamp(started).strftime('%H:%M') if started is not None else 'unknown time'
        fields['subject'] = f'{data["name"]} ({data["kind"]}, started {clock})'
    if check == 'counters':
        fields['counts'] = ', '.join(f'{name} +{count}' for name, count in sorted(data['counts'].items()) if count)
    if check == 'repeat' and state == 'closed':
        if data.get('close_reason') == 'aged_out':
            return '{subject}: no new failures in the last {hours:g} hours.'.format(**fields)
        if data.get('close_reason') != 'succeeded':
            return '{subject}: no longer meets the repeated-failure threshold.'.format(**fields)
    return MESSAGES[check][state == 'closed'].format(**fields)


class Pass:
    def __init__(self, conn, now, *, queue_notices=True):
        self.conn, self.now = conn, now
        self.queue_notices = queue_notices

    def notice(self, identity, check, state, data):
        if not self.queue_notices:
            return
        # Deleted queue rows must never allow a later event to reuse its delivery id.
        last = self.conn.execute('SELECT COALESCE(MAX(id),0) FROM notices').fetchone()[0]
        sequence = max(last, store.get(self.conn, 'notice_sequence', 0)) + 1
        store.put(self.conn, 'notice_sequence', sequence)
        self.conn.execute('INSERT INTO notices (id,finding_id,"check",state,data) VALUES (?,?,?,?,?)',
                          (sequence, identity, check, state, json.dumps({**data, '_queued_at': self.now})))
        return sequence

    def finding(self, check, subject, broken, evidence=None, *, notice_data=None, close_notice=True,
                close_data=None):
        identity = digest(check + ':' + subject)
        old = self.conn.execute('SELECT * FROM findings WHERE id=?', (identity,)).fetchone()
        if broken:
            opened = old is None or old['state'] == 'closed'
            data = json.dumps(notice_data or {}) if opened else old['notice_data']
            opening_notice = self.notice(identity, check, 'open', json.loads(data)) if opened else old['opening_notice']
            self.conn.execute('INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?,?,?)',
                (identity, check, subject, self.now if opened else old['first_seen'], self.now, 'open',
                 json.dumps(evidence or {}), data, opening_notice))
        elif old and old['state'] == 'open':
            self.conn.execute('UPDATE findings SET state=\'closed\',last_seen=? WHERE id=?', (self.now, identity))
            if close_notice and self.conn.execute('SELECT 1 FROM notices WHERE id=?',
                                                   (old['opening_notice'],)).fetchone():
                self.notice(identity, check, 'closed', {**json.loads(old['notice_data']), **(close_data or {})})

    def announce_open_findings(self):
        if not self.queue_notices:
            return
        for row in self.conn.execute("SELECT * FROM findings WHERE state='open' AND opening_notice IS NULL").fetchall():
            identity = self.notice(row['id'], row['check'], 'open', json.loads(row['notice_data']))
            self.conn.execute('UPDATE findings SET opening_notice=? WHERE id=?', (identity, row['id']))

    def jobs(self, jobs, config, since):
        seen = set()
        # A retained terminal receipt wins over a stale running checkpoint.
        terminal = {job.subject for job in jobs if job.ended_at is not None}
        for job in jobs:
            subject = job.subject
            seen.add(subject)
            active = job.state in ACTIVE and subject not in terminal
            if not active:
                self.finding('stuck', subject, False)
                self.finding('limit', subject, False)
                continue
            seconds = config['stuck_minutes'] * 60
            observation = store.get(self.conn, 'activity:' + subject)
            if job.cpu is not None and job.output is not None:
                changed = observation and (observation['cpu'] != job.cpu or observation['output'] != job.output)
                # A gap is neither proof of inactivity nor proof of recovery.
                gap = not observation or self.now - observation['seen'] > 600
                idle_since = self.now if changed or gap else observation['since']
                store.put(self.conn, 'activity:' + subject,
                          {'cpu': job.cpu, 'output': job.output, 'since': idle_since, 'seen': self.now})
                if changed or self.now - idle_since >= seconds:
                    self.finding('stuck', subject, self.now - idle_since >= seconds,
                                 {'since': idle_since, 'seconds': self.now - idle_since},
                                 notice_data={**job_notice(job), 'minutes': config['stuck_minutes']})
            elif observation:
                observation['since'] = self.now
                store.put(self.conn, 'activity:' + subject, observation)
            elapsed = self.now - job.started_at if job.started_at is not None else 0
            wall = job.limits.get('wall_seconds', 0)
            turns = job.limits.get('max_turns', job.limits.get('turns', 0))
            over = (wall > 0 and max(elapsed, job.usage.get('wall_seconds', 0)) > wall) or (
                turns > 0 and job.usage.get('turns', 0) > turns)
            if over:
                self.finding('limit', subject, True, {'seconds': elapsed, 'limits': job.limits, 'usage': job.usage},
                             notice_data=job_notice(job))

        # Only a successful inventory can prove a live job disappeared.
        previous = store.get(self.conn, 'inventory', {})
        current = {job.subject: job.source for job in jobs if job.state in ACTIVE}
        for subject, source in previous.items():
            if subject not in seen and source in jobs.available:
                self.finding('stuck', subject, False)
                self.finding('limit', subject, False)
            elif subject not in current and source not in jobs.available:
                current[subject] = source
        store.put(self.conn, 'inventory', current)

        cutoff = max(self.now - config['repeat_window_hours'] * 3600, since or float('-inf'))
        history = store.get(self.conn, 'results', {})
        for job in jobs:
            if job.kind != 'process' or job.ended_at is None or not job.signature:
                continue
            if cutoff <= job.ended_at <= self.now:
                history[job.subject] = {'signature': job.signature, 'error': job.error_hash,
                    'ended': job.ended_at, 'failed': job.exit_code is not None and job.exit_code != 0,
                    'succeeded': job.exit_code == 0,
                    'notice_data': job_notice(job)}
        history = {key: value for key, value in history.items() if value['ended'] >= cutoff}
        store.put(self.conn, 'results', history)
        streaks = {}
        for identity, item in sorted(history.items(), key=lambda pair: (pair[1]['ended'], pair[0])):
            signature, error = item['signature'], item['error']
            old = streaks.get(signature, {})
            count = (old.get('count', 0) + 1 if old.get('error') == error else 1) if item['failed'] and error else 0
            streaks[signature] = {'error': error, 'count': count, 'ended': item['ended'],
                                  'succeeded': item.get('succeeded', False),
                                  'notice_data': item['notice_data']}
        for signature, streak in streaks.items():
            data = streak.pop('notice_data')
            self.finding('repeat', signature, streak['count'] >= config['repeat_failures'], streak,
                         notice_data=data, close_data={'close_reason': 'succeeded' if streak['succeeded'] else 'threshold'})
        for row in self.conn.execute('SELECT subject FROM findings WHERE "check"=\'repeat\' AND state=\'open\''):
            if row['subject'] not in streaks:
                self.finding('repeat', row['subject'], False,
                             close_data={'close_reason': 'aged_out', 'hours': config['repeat_window_hours']})

    def dependencies(self, config, results):
        for check, subject, broken, data in results['findings']:
            key = 'dependency:' + check + ':' + subject
            failures = store.get(self.conn, key, 0) + 1 if broken else 0
            store.put(self.conn, key, failures)
            if not broken or failures >= config.get('dependency_failure_passes', DEPENDENCY_FAILURE_PASSES):
                self.finding(check, subject, broken, notice_data=data)
        if results['counters'] is not None:
            self.counters(results['counters'])
        if results['downloads'] is not None:
            try:
                self.downloads(results['downloads'])
            except NETWORK_ERRORS as error:
                failed(error)

    def counters(self, counters):
        values = {key: value for key, value in counters.items()
                  if (key == 'context_writes_dropped' or key.startswith('mind_incomplete.'))
                  and isinstance(value, int) and value >= 0}
        old = store.get(self.conn, 'health_counters')
        store.put(self.conn, 'health_counters', values)
        if old is None:
            return
        deltas = {key: value - old.get(key, 0) if value >= old.get(key, 0) else value
                  for key, value in values.items()}
        if not any(deltas.values()):
            return
        day = datetime.fromtimestamp(self.now, timezone.utc).date().isoformat()
        self.finding('counters', day, True, {'counts': deltas}, notice_data={'counts': deltas})
        for row in self.conn.execute('SELECT subject FROM findings WHERE "check"=\'counters\' AND state=\'open\''):
            if row['subject'] != day:
                self.finding('counters', row['subject'], False, close_notice=False)

    def downloads(self, downloads):
        # Validate the whole snapshot before inferring that an item disappeared.
        items = [(digest(str(item['hash'])), float(item['progress']),
                  int(item.get('num_seeds', 0)) + int(item.get('num_leechs', 0)),
                  int(item.get('num_complete', 0)) + int(item.get('num_incomplete', 0)),
                  item.get('name', 'unnamed download')) for item in downloads]
        previous = store.get(self.conn, 'downloads', {})
        current = {}
        for subject, progress, connected, known, name in items:
            if progress < 1 and connected <= 0 and known <= 0:
                old = previous.get(subject, {})
                start = old.get('since', self.now) if self.now - old.get('seen', self.now) <= 600 else self.now
                current[subject] = {'since': start, 'seen': self.now}
                if self.now - start >= 3600:
                    self.finding('download', subject, True, {'since': start}, notice_data={'name': name})
            else:
                self.finding('download', subject, False)
        for subject in previous.keys() - {item[0] for item in items}:
            self.finding('download', subject, False)
        store.put(self.conn, 'downloads', current)


def check_dependencies(config, sidecar_url, key, request):
    results = {'findings': [], 'counters': None, 'downloads': None}
    timeout = config['health_timeout_s']
    try:
        status, _ = request(sidecar_url + '/v1/host/health', timeout, key=key)
        healthy = status == 200
    except NETWORK_ERRORS as error:
        failed(error)
        healthy = False
    results['findings'].append(('sidecar', 'sidecar', not healthy, {}))
    if healthy:
        try:
            status, payload = request(sidecar_url + '/v1/mind/status', timeout, key=key, body=True)
            if status == 200 and isinstance(payload, dict) and isinstance(payload.get('health_counters'), dict):
                results['counters'] = payload['health_counters']
        except NETWORK_ERRORS as error:
            failed(error)
    for endpoint in config['endpoints']:
        subject = digest(endpoint['name'] + ':' + endpoint['url'])
        try:
            status, _ = request(endpoint['url'], timeout)
            broken = status != endpoint['expect_status']
        except NETWORK_ERRORS as error:
            failed(error)
            broken = True
        results['findings'].append(('endpoint', subject, broken, {'name': endpoint['name']}))
    url = config['downloads']['url']
    if url:
        try:
            status, downloads = request(url.rstrip('/') + '/api/v2/torrents/info', timeout, body=True)
            if status == 200 and isinstance(downloads, list):
                results['downloads'] = downloads
        except NETWORK_ERRORS as error:
            failed(error)
    return results


def deliver(message, command, timeout=NOTICE_TIMEOUT_S, *, notice_id):
    if not command:
        raise ValueError('jobwatch notice command is unset')
    with subprocess.Popen(command, stdin=subprocess.PIPE, text=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=True,
                          env={**os.environ, 'COLONYMIND_NOTICE_ID': notice_id}) as process:
        try:
            process.communicate(message + '\n', timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The whole group already exited.
                pass
            process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def group_notices(conn):
    for state in ('open', 'closed'):
        rows = conn.execute('SELECT n.* FROM notices n WHERE n.sent=0 AND n.state=? AND NOT EXISTS '
                            '(SELECT 1 FROM notices older WHERE older.sent=0 '
                            'AND older.finding_id=n.finding_id AND older.id<n.id) ORDER BY n.id',
                            (state,)).fetchall()
        rows = [row for row in rows if not store.get(conn, 'notice_batch:' + str(row['id']))
                and not store.get(conn, 'notice_identity:' + str(row['id']))]
        if len(rows) > 3:
            members = [row['id'] for row in rows]
            notice_id = digest(json.dumps([(row['finding_id'], row['state'], row['id']) for row in rows]))
            for identity in members:
                store.put(conn, 'notice_batch:' + str(identity), members)
            store.put(conn, 'notice_identity:' + str(members[0]), notice_id)


def prune_notices(conn, now):
    rows = conn.execute('SELECT n.*, f.state AS finding_state, f.first_seen, f.last_seen '
                        'FROM notices n JOIN findings f ON f.id=n.finding_id WHERE n.sent=0 ORDER BY n.id').fetchall()
    dropped, openings = set(), {}
    for row in rows:
        members = store.get(conn, 'notice_batch:' + str(row['id']), [])
        if members and not store.get(conn, 'notice_identity:' + str(members[0])):
            # Preserve ids of batches queued before identities were persisted, even if members now close.
            batch = [conn.execute('SELECT * FROM notices WHERE id=?', (identity,)).fetchone()
                     for identity in members]
            notice_id = digest(json.dumps([(item['finding_id'], item['state'], item['id'])
                                           for item in batch if item is not None]))
            store.put(conn, 'notice_identity:' + str(members[0]), notice_id)
        if row['state'] == 'open':
            openings[row['finding_id']] = row['id']
            if row['finding_state'] == 'closed':
                dropped.add(row['id'])
        elif row['finding_id'] in openings:
            dropped.update((openings.pop(row['finding_id']), row['id']))
    expired = set()
    for row in rows:
        queued = json.loads(row['data']).get('_queued_at',
                    row['first_seen'] if row['state'] == 'open' else row['last_seen'])
        if now - queued > 86400:
            expired.add(row['id'])
    if dropped or expired:
        last = conn.execute('SELECT COALESCE(MAX(id),0) FROM notices').fetchone()[0]
        store.put(conn, 'notice_sequence', max(last, store.get(conn, 'notice_sequence', 0)))
        conn.executemany('DELETE FROM notices WHERE id=?', [(identity,) for identity in dropped | expired])
    return len(expired)


def deliver_pending(conn, command, notify, now, timeout):
    if not command:
        return
    conn.execute('BEGIN IMMEDIATE')
    if store.get(conn, 'delivery_claim', {}).get('until', 0) > now:
        conn.rollback()
        return
    expired = prune_notices(conn, now)
    group_notices(conn)
    conn.commit()
    if expired:
        bump('writes_dropped.jobwatch_notice_expired', expired)
    blocked = set()
    pending, seen = [], set()
    for item in conn.execute('SELECT id,state FROM notices WHERE sent=0 ORDER BY id').fetchall():
        members = store.get(conn, 'notice_batch:' + str(item['id']), [])
        identity = members[0] if members else item['id']
        if identity in seen:
            continue
        seen.add(identity)
        offered = store.get(conn, 'notice_offered:' + str(identity), False)
        # Existing singles and timed-out groups already have durable evidence of an offer.
        offered = offered or store.get(conn, 'delivery_retry:' + str(identity), 0) or (
            not members and store.get(conn, 'notice_identity:' + str(identity)))
        pending.append((bool(offered), not bool(members), identity, item))
    delivered_groups = {'open': 0, 'closed': 0}
    delivered_singles = {'open': 0, 'closed': 0}
    started = time.monotonic()
    # Prioritize never-offered notices and groups only among each finding's oldest pending notices.
    pending.sort()
    while pending:
        clock = now + time.monotonic() - started
        conn.execute('BEGIN IMMEDIATE')
        claim = store.get(conn, 'delivery_claim', {})
        if claim.get('until', 0) > clock:
            conn.rollback()
            return
        for index, (_, single, identity, item) in enumerate(pending):
            members = store.get(conn, 'notice_batch:' + str(item['id']), [item['id']])
            rows = [conn.execute('SELECT * FROM notices WHERE id=? AND sent=0', (identity,)).fetchone()
                    for identity in members]
            rows = [row for row in rows if row is not None]
            if not any(conn.execute('SELECT 1 FROM notices WHERE sent=0 AND finding_id=? AND id<?',
                                    (row['finding_id'], row['id'])).fetchone() for row in rows):
                break
        else:
            conn.rollback()
            return
        pending.pop(index)
        subjects = {row['finding_id'] for row in rows}
        retry_key = 'delivery_retry:' + str(members[0])
        if not rows or subjects & blocked or store.get(conn, retry_key, 0) > clock:
            blocked.update(subjects)
            conn.rollback()
            continue
        delivered = delivered_singles if single else delivered_groups
        if delivered[item['state']] >= (3 if single else 1):
            blocked.update(subjects)
            conn.rollback()
            continue
        token = uuid4().hex
        # Persist a single's first offer with its claim so it can never join a later group.
        identity_key = 'notice_identity:' + str(members[0])
        notice_id = store.get(conn, identity_key) or digest(
            json.dumps([(row['finding_id'], row['state'], row['id']) for row in rows]))
        store.put(conn, identity_key, notice_id)
        store.put(conn, 'notice_offered:' + str(members[0]), True)
        # A crashed sender leaves a bounded lease, never a database write lock.
        store.put(conn, 'delivery_claim', {'token': token, 'until': clock + timeout + 60})
        conn.commit()
        if len(rows) > 1:
            counts = {}
            for row in rows:
                counts[row['check']] = counts.get(row['check'], 0) + 1
            action = 'opened' if rows[0]['state'] == 'open' else 'closed'
            message = 'Job watch ' + action + ' ' + str(len(rows)) + ' findings: ' + ', '.join(
                f'{check}: {count}' for check, count in sorted(counts.items())) + '.'
            message += ' ' + ' '.join(notice_message(row['check'], row['state'], json.loads(row['data']))
                                      for row in rows[:3])
        else:
            row = rows[0]
            message = notice_message(row['check'], row['state'], json.loads(row['data']))
        error = None
        try:
            notify(message, command, notice_id)
        except Exception as caught:
            error = caught
            log.warning('jobwatch delivery failed: %s sqlite=%s', type(error).__name__,
                        getattr(error, 'sqlite_errorname', None))
            bump('writes_dropped.jobwatch_delivery')
        conn.execute('BEGIN IMMEDIATE')
        if store.get(conn, 'delivery_claim', {}).get('token') != token:
            conn.rollback()
            return
        store.put(conn, 'delivery_claim', {})
        if error is not None:
            store.put(conn, 'delivery_failures', store.get(conn, 'delivery_failures', 0) + 1)
            if isinstance(error, (subprocess.TimeoutExpired, TimeoutError)):
                store.put(conn, retry_key, now + time.monotonic() - started + 3600)
            blocked.update(subjects)
        else:
            conn.executemany('UPDATE notices SET sent=1 WHERE id=?', [(row['id'],) for row in rows])
            delivered[item['state']] += 1
        conn.commit()
        if isinstance(error, (subprocess.TimeoutExpired, TimeoutError)):
            return


def run(config, data_dir, sidecar_url, *, key=None, now=None, since=None, dry_run=False,
        request=fetch, jobs=None, activity=probe, notify=deliver):
    now = time.time() if now is None else now
    started = time.monotonic()
    if not config['enabled']:
        return []
    if jobs is None:
        jobs = read_jobs(config['hermes_home'], now, data_dir=data_dir, since=since)
        activity(jobs, tmux=config['tmux'])
        # Delegated hands may have no PID; their owned process supplies activity.
        owned = {}
        for job in jobs:
            if job.kind == 'process' and job.state in ACTIVE and job.owner_task_id:
                owned.setdefault(job.owner_task_id, []).append(job)
        for job in jobs:
            if job.kind == 'hand' and job.owner_task_id in owned:
                processes = sorted(owned[job.owner_task_id], key=lambda item: item.subject)
                if all(item.cpu is not None and item.output is not None for item in processes):
                    job.cpu = sum(item.cpu for item in processes)
                    job.output = digest(str([(item.subject, item.output) for item in processes]))
    from .activity import record_activity
    record_activity(jobs, config['hermes_home'])
    results = check_dependencies(config, sidecar_url, key, request)
    if not config['notice_command']:
        log.warning('jobwatch notice command is unset; findings recorded without notices')
    with closing(store.connect(data_dir / 'jobwatch.db', dry_run=dry_run)) as conn:
        conn.execute('BEGIN IMMEDIATE')
        watch = Pass(conn, now, queue_notices=bool(config['notice_command']))
        watch.jobs(jobs, config, since)
        watch.dependencies(config, results)
        watch.announce_open_findings()
        findings = store.project(conn)
        conn.commit()
        if not dry_run:
            timeout = config.get('notice_timeout_s', NOTICE_TIMEOUT_S)
            sender = (lambda message, command, notice_id: deliver(message, command, timeout, notice_id=notice_id)) \
                if notify is deliver else (lambda message, command, notice_id: notify(message, command))
            deliver_pending(conn, config['notice_command'], sender, now + time.monotonic() - started, timeout)
    return findings
