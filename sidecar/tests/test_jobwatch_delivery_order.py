"""New notices precede retries, and only successful deliveries consume caps."""
from contextlib import closing
import json
import subprocess
import sys

import pytest

from colonymind.jobwatch import runner, store
from test_jobwatch import cfg, notice_rows, snapshot


def queue(conn, subject, state='open'):
    watch = runner.Pass(conn, 0)
    watch.finding('endpoint', subject, True, notice_data={'name': subject})
    if state == 'closed':
        conn.execute('UPDATE notices SET sent=1 WHERE id=(SELECT MAX(id) FROM notices)')
        watch.finding('endpoint', subject, False)


@pytest.mark.parametrize('later_sidecar', [False, True])
def test_new_single_delivered_next_pass_after_three_rejections(tmp_path, cfg, later_sidecar):
    cfg['dependency_failure_passes'] = 1
    cfg['endpoints'] = [{'name': f'service{i}', 'url': f'http://service/{i}', 'expect_status': 200}
                        for i in range(3 if later_sidecar else 4)]
    offered, sent = [], []
    target = 'ColonyMind is down' if later_sidecar else 'service3'
    def notify(message, command):
        offered.append(message)
        if target not in message:
            raise OSError
        sent.append(message)
    for index in range(4):
        def request(url, *args, **kwargs):
            broken = (url.startswith('http://service/') and int(url.rsplit('/', 1)[1]) <= index
                      or later_sidecar and index == 3)
            return (503 if broken else 200), None
        offered.clear()
        runner.run(cfg, tmp_path, 'http://sidecar', now=index * 300, jobs=snapshot(),
                   request=request, notify=notify)
    assert len(sent) == 1 and target in sent[0]
    assert offered[0] == sent[0] and len(offered) == 4
    assert sum(row['sent'] for row in notice_rows(tmp_path)) == 1


@pytest.mark.parametrize('error', [OSError, TimeoutError])
def test_later_group_delivered_next_pass(tmp_path, cfg, error):
    cfg['dependency_failure_passes'] = 1
    cfg['endpoints'] = [{'name': f'service{i}', 'url': f'http://service/{i}', 'expect_status': 200}
                        for i in range(8)]
    offered, sent = [], []
    def notify(message, command):
        offered.append(message)
        if 'service0' in message:
            raise error
        sent.append(message)
    for now, count in ((0, 4), (300, 8)):
        runner.run(cfg, tmp_path, 'http://sidecar', now=now, jobs=snapshot(),
            request=lambda url, *a, **k: (503 if url.startswith('http://service/')
                and int(url.rsplit('/', 1)[1]) < count else 200, None), notify=notify)
    assert len(sent) == 1 and 'service4' in sent[0]
    assert offered == [offered[0], sent[0]]
    assert sum(row['sent'] for row in notice_rows(tmp_path)) == 4


@pytest.mark.parametrize('state', ['open', 'closed'])
@pytest.mark.parametrize('backoff', [False, True])
def test_every_retry_group_considered_after_failure_or_backoff(tmp_path, state, backoff):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        attempts = []
        def fail(message, command, notice_id):
            attempts.append((message, notice_id))
            if backoff and 'first0' in message:
                raise TimeoutError
            raise OSError
        for now, prefix in ((0, 'first'), (300, 'later')):
            with conn:
                for index in range(4):
                    queue(conn, f'{prefix}{index}', state)
            runner.deliver_pending(conn, ['sender'], fail, now, 30)
        later = next(attempt for attempt in attempts if 'later0' in attempt[0])
        offered, sent = [], []
        def notify(message, command, notice_id):
            offered.append((message, notice_id))
            if 'first0' in message:
                raise OSError
            sent.append((message, notice_id))
        runner.deliver_pending(conn, ['sender'], notify, 600, 30)
        assert sent == [later]
        assert len(offered) == (1 if backoff else 2)
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 4


@pytest.mark.parametrize('state', ['open', 'closed'])
def test_failed_singles_use_no_allowance_and_keep_retry_ids(tmp_path, state):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        attempts = []
        def fail(message, command, notice_id):
            attempts.append((message, notice_id))
            raise OSError
        for index in range(4):
            with conn:
                queue(conn, f'service{index}', state)
            runner.deliver_pending(conn, ['sender'], fail, index * 300, 30)
        for index in range(4):
            ids = [ident for message, ident in attempts if f'service{index}' in message]
            assert len(ids) == 4 - index and len(set(ids)) == 1
        offered, sent = [], []
        def notify(message, command, notice_id):
            offered.append((message, notice_id))
            if 'service3' not in message:
                raise OSError
            sent.append((message, notice_id))
        runner.deliver_pending(conn, ['sender'], notify, 1200, 30)
        assert len(offered) == 4 and sent == [offered[-1]]
        assert all(f'service{index}' in offered[index][0] for index in range(4))
        assert sent[0] in attempts


@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('error', [TimeoutError(), subprocess.TimeoutExpired(['sender'], 30)])
def test_timeout_ends_pass_and_next_pass_starts_with_new_notices(tmp_path, grouped, error):
    path = tmp_path / 'jobwatch.db'
    offered = []
    def notify(message, command, notice_id):
        offered.append((message, notice_id))
        if 'first0' in message:
            raise error
    with closing(store.connect(path)) as conn:
        with conn:
            for index in range(4 if grouped else 1):
                queue(conn, f'first{index}')
            if grouped:
                runner.group_notices(conn)
            queue(conn, 'waiting')
        runner.deliver_pending(conn, ['sender'], notify, 0, 30)
        assert len(offered) == 1
        first = offered[0]
        assert store.get(conn, 'delivery_retry:1') >= 3600
        assert store.get(conn, 'delivery_claim') == {}
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=1').fetchone()[0] == 0
    with closing(store.connect(path)) as conn:
        with conn:
            queue(conn, 'new')
        runner.deliver_pending(conn, ['sender'], notify, 300, 30)
        assert len(offered) == 3
        assert 'waiting' in offered[1][0] and 'new' in offered[2][0]
        with conn:
            queue(conn, 'newest')
        runner.deliver_pending(conn, ['sender'], notify, 3602, 30)
        assert len(offered) == 5
        assert 'newest' in offered[3][0] and offered[4] == first
        assert store.get(conn, 'delivery_retry:1') >= 7202


def test_reopening_follows_recovery_retry_with_real_sender(tmp_path, cfg):
    cfg['dependency_failure_passes'] = 1
    cfg['endpoints'] = [{'name': 'service', 'url': 'http://service', 'expect_status': 200}]
    sender = tmp_path / 'sender.py'
    rejecting = tmp_path / 'reject-recovery'
    delivered = tmp_path / 'delivered'
    sender.write_text('''import json, os, sys
from pathlib import Path
recovered = 'has recovered' in sys.stdin.read()
if recovered and Path(sys.argv[2]).exists():
    sys.exit(3)
with open(sys.argv[1], 'a') as output:
    output.write(json.dumps([recovered, os.environ['COLONYMIND_NOTICE_ID']]) + '\\n')
''')
    cfg['notice_command'] = [sys.executable, str(sender), str(delivered), str(rejecting)]
    def run(now, broken):
        return runner.run(cfg, tmp_path, 'http://sidecar', now=now, jobs=snapshot(),
            request=lambda url, *a, **k: (503 if broken and url == 'http://service' else 200, None))
    run(0, True)
    rejecting.touch()
    run(300, False)
    rejecting.unlink()
    findings = run(600, True)
    assert len(findings) == 1 and findings[0]['state'] == 'open'
    events = [json.loads(line) for line in delivered.read_text().splitlines()]
    assert [recovered for recovered, identity in events] == [False, True, False]
    assert len({identity for recovered, identity in events}) == 3


@pytest.mark.parametrize('others', [3, 4])
@pytest.mark.parametrize('error', [OSError, TimeoutError])
def test_pending_recovery_holds_reopening_out_of_groups_only(tmp_path, others, error):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        with conn:
            queue(conn, 'original', 'closed')
        attempts = []
        def fail(message, command, notice_id):
            attempts.append((message, notice_id))
            raise error
        runner.deliver_pending(conn, ['sender'], fail, 0, 30)
        recovery = attempts[0]
        with conn:
            queue(conn, 'original')
            reopening = conn.execute('SELECT MAX(id) FROM notices').fetchone()[0]
            for index in range(others):
                queue(conn, f'other{index}')
            runner.group_notices(conn)
        assert store.get(conn, 'notice_batch:' + str(reopening)) is None
        sent = []
        def notify(message, command, notice_id):
            if 'original' in message:
                fail(message, command, notice_id)
            sent.append((message, notice_id))
        for now in (300, 600):
            runner.deliver_pending(conn, ['sender'], notify, now, 30)
        assert len(sent) == (3 if others == 3 else 1)
        assert all('original' not in message for message, identity in sent)
        assert attempts == [recovery] * (3 if error is OSError else 1)
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 2
        runner.deliver_pending(conn, ['sender'], lambda m, c, ident: sent.append((m, ident)), 4000, 30)
        assert sent[-2] == recovery
        assert sent[-1][0] == 'Watched endpoint original is unavailable.'
        assert sent[-1][1] != recovery[1]
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0
        assert store.get(conn, 'notice_batch:' + str(reopening)) is None


def test_reopening_waits_for_next_pass_when_single_allowance_is_used(tmp_path):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        with conn:
            queue(conn, 'original', 'closed')
        def fail(message, command, notice_id):
            raise OSError
        runner.deliver_pending(conn, ['sender'], fail, 0, 30)
        with conn:
            queue(conn, 'original')
            for index in range(3):
                queue(conn, f'other{index}')
        sent = []
        notify = lambda m, c, ident: sent.append(m)
        runner.deliver_pending(conn, ['sender'], notify, 300, 30)
        assert sent == [f'Watched endpoint other{index} is unavailable.' for index in range(3)] + [
            'Watched endpoint original has recovered.']
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 1
        runner.deliver_pending(conn, ['sender'], notify, 600, 30)
        assert sent[-1] == 'Watched endpoint original is unavailable.'
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0
