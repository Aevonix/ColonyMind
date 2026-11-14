"""Delayed announcements and grouping of the durable delivery queue."""
from contextlib import closing

import pytest

from colonymind.jobwatch import runner, store
from test_jobwatch import cfg, invoke, notice_rows, snapshot


def test_current_outage_announced_once_when_sender_enabled(tmp_path, cfg):
    cfg.update(notice_command=[], dependency_failure_passes=1)
    down = lambda *a, **k: (503, None)
    for now in (0, 300):
        assert len(invoke(tmp_path, cfg, now, snapshot(), request=down)) == 1
    assert notice_rows(tmp_path) == []
    cfg['notice_command'] = ['test-sender']
    notices = []
    for now in (600, 900):
        assert len(invoke(tmp_path, cfg, now, snapshot(), request=down, notices=notices)) == 1
    assert notices == [runner.MESSAGES['sidecar'][0]]
    assert len(notice_rows(tmp_path)) == 1
    invoke(tmp_path, cfg, 1200, snapshot(), notices=notices)
    assert notices == list(runner.MESSAGES['sidecar'])


def test_unannounced_finding_closes_silently_then_reopens(tmp_path, cfg):
    cfg.update(notice_command=[], dependency_failure_passes=1)
    down = lambda *a, **k: (503, None)
    invoke(tmp_path, cfg, 0, snapshot(), request=down)
    cfg['notice_command'] = ['test-sender']
    notices = []
    invoke(tmp_path, cfg, 300, snapshot(), notices=notices)
    assert notice_rows(tmp_path) == [] and notices == []
    invoke(tmp_path, cfg, 600, snapshot(), request=down, notices=notices)
    assert notices == [runner.MESSAGES['sidecar'][0]]


def test_unchanged_open_finding_is_announced_without_new_observation(tmp_path, cfg):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn, conn:
        runner.Pass(conn, 0, queue_notices=False).finding('stuck', 'job', True,
            notice_data={'name': 'make', 'kind': 'process', 'started_at': 0, 'minutes': 30})
    notices = []
    for now in (300, 600):
        invoke(tmp_path, cfg, now, snapshot(available=()), notices=notices)
    assert len(notices) == 1 and 'printed nothing' in notices[0]


def test_expired_opening_is_not_requeued_or_followed_by_recovery(tmp_path, cfg):
    cfg['dependency_failure_passes'] = 1
    def fail(*args):
        raise OSError
    runner.run(cfg, tmp_path, 'http://sidecar', now=0, jobs=snapshot(),
               request=lambda *a, **k: (503, None), notify=fail)
    notices = []
    for now in (86401, 86701):
        invoke(tmp_path, cfg, now, snapshot(), request=lambda *a, **k: (503, None), notices=notices)
    invoke(tmp_path, cfg, 87001, snapshot(), notices=notices)
    assert notice_rows(tmp_path) == [] and notices == []


@pytest.mark.parametrize('state', ['open', 'closed'])
@pytest.mark.parametrize('count', [2, 6])
def test_delivery_groups_pending_notices_from_separate_passes(tmp_path, state, count):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        for index in range(count):
            with conn:
                watch = runner.Pass(conn, index * 300)
                watch.finding('endpoint', str(index), True, notice_data={'name': f'service{index}'})
                if state == 'closed':
                    conn.execute('UPDATE notices SET sent=1')
        if state == 'closed':
            for index in range(count):
                with conn:
                    runner.Pass(conn, 1800 + index * 300).finding('endpoint', str(index), False)
        sent = []
        runner.deliver_pending(conn, ['test-sender'], lambda m, c, ident: sent.append((m, ident)), 4000, 30)
        assert len(sent) == (count if count <= 3 else 1)
        if count > 3:
            action = 'opened' if state == 'open' else 'closed'
            assert sent[0][0].startswith(f'Job watch {action} 6 findings: endpoint: 6.')
            assert sum(f'service{index}' in sent[0][0] for index in range(count)) == 3
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0


@pytest.mark.parametrize('later_count', [1, 4])
def test_failed_group_is_committed_and_retry_keeps_members_and_id(tmp_path, later_count):
    path = tmp_path / 'jobwatch.db'
    attempts = []
    with closing(store.connect(path)) as conn:
        with conn:
            for index in range(4):
                runner.Pass(conn, index).finding('endpoint', str(index), True,
                                                notice_data={'name': f'service{index}'})
        def fail(message, command, notice_id):
            attempts.append((message, notice_id))
            with closing(store.connect(path)) as other, other:
                other.execute('BEGIN IMMEDIATE')
                assert store.get(other, 'notice_batch:1') == [1, 2, 3, 4]
                assert store.get(other, 'notice_identity:1') == notice_id
            raise OSError
        runner.deliver_pending(conn, ['test-sender'], fail, 10, 30)
    with closing(store.connect(path)) as conn:
        with conn:
            for index in range(later_count):
                runner.Pass(conn, 20).finding('endpoint', f'later{index}', True, notice_data={'name': f'later{index}'})
        sent = []
        runner.deliver_pending(conn, ['test-sender'], lambda m, c, ident: sent.append((m, ident)), 30, 30)
        assert 'later' in sent[0][0]
        assert len(sent) == (2 if later_count == 1 else 1)
        runner.deliver_pending(conn, ['test-sender'], lambda m, c, ident: sent.append((m, ident)), 330, 30)
        assert len(sent) == 2 and sent[1:] == attempts
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0


def test_offered_single_keeps_identity_and_backoff_when_new_notices_arrive(tmp_path):
    attempts = []
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        with conn:
            runner.Pass(conn, 0).finding('endpoint', '0', True, notice_data={'name': 'service0'})
        def timeout(message, command, notice_id):
            attempts.append((message, notice_id))
            raise TimeoutError
        runner.deliver_pending(conn, ['test-sender'], timeout, 0, 30)
        with conn:
            for index in range(1, 4):
                runner.Pass(conn, 300).finding('endpoint', str(index), True,
                                              notice_data={'name': f'service{index}'})
        sent = []
        notify = lambda m, c, ident: sent.append((m, ident))
        runner.deliver_pending(conn, ['test-sender'], notify, 300, 30)
        assert len(sent) == 3
        assert all('service0' not in message for message, _ in sent)
        runner.deliver_pending(conn, ['test-sender'], notify, 3602, 30)
        assert len(sent) == 4 and sent[-1] == attempts[0]
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0


def test_recovered_sender_does_not_burst_staggered_findings(tmp_path, cfg):
    cfg['dependency_failure_passes'] = 1
    cfg['endpoints'] = [{'name': f'service{i}', 'url': f'http://service/{i}', 'expect_status': 200}
                        for i in range(6)]
    failed_through = -1
    def request(url, *args, **kwargs):
        broken = url.startswith('http://service/') and int(url.rsplit('/', 1)[1]) <= failed_through
        return (503 if broken else 200), None
    def fail(*args):
        raise OSError
    for index in range(6):
        failed_through = index
        runner.run(cfg, tmp_path, 'http://sidecar', now=index * 300, jobs=snapshot(), request=request, notify=fail)
    notices = []
    invoke(tmp_path, cfg, 1800, snapshot(), request=request, notices=notices)
    assert len(notices) == 3 and all(message.startswith('Watched endpoint ') for message in notices)
    assert sum(row['sent'] for row in notice_rows(tmp_path)) == 3
    invoke(tmp_path, cfg, 2100, snapshot(), request=request, notices=notices)
    assert len(notices) == 6 and all(row['sent'] for row in notice_rows(tmp_path))
    invoke(tmp_path, cfg, 2400, snapshot(), request=request, notices=notices)
    assert len(notices) == 6


@pytest.mark.parametrize('error', [OSError, TimeoutError])
def test_failed_group_does_not_hold_later_sidecar_notice(tmp_path, cfg, error):
    cfg['dependency_failure_passes'] = 1
    cfg['endpoints'] = [{'name': f'service{i}', 'url': f'http://service/{i}', 'expect_status': 200}
                        for i in range(4)]
    attempts, sent = [], []
    def notify(message, command):
        attempts.append(message)
        if message.startswith('Job watch opened 4'):
            raise error
        sent.append(message)
    runner.run(cfg, tmp_path, 'http://sidecar', now=0, jobs=snapshot(),
               request=lambda url, *a, **k: (503 if url.startswith('http://service/') else 200, None),
               notify=notify)
    runner.run(cfg, tmp_path, 'http://sidecar', now=300, jobs=snapshot(),
               request=lambda *a, **k: (503, None), notify=notify)
    assert sent == [runner.MESSAGES['sidecar'][0]]
    assert len(attempts) == (2 if error is TimeoutError else 3)
    assert sum(row['sent'] for row in notice_rows(tmp_path)) == 1


@pytest.mark.parametrize('state', ['open', 'closed'])
def test_healthy_pass_delivers_group_then_two_singles(tmp_path, state):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        with conn:
            for index in range(6):
                watch = runner.Pass(conn, index)
                watch.finding('endpoint', str(index), True, notice_data={'name': f'service{index}'})
                if state == 'closed':
                    conn.execute("UPDATE notices SET sent=1 WHERE state='open'")
                    watch.finding('endpoint', str(index), False)
                if index == 3:
                    runner.group_notices(conn)
        sent = []
        runner.deliver_pending(conn, ['test-sender'], lambda m, c, ident: sent.append((m, ident)), 10, 30)
        action = 'opened' if state == 'open' else 'closed'
        assert len(sent) == 3
        assert sent[0][0].startswith(f'Job watch {action} 4 findings:')
        assert 'service4' in sent[1][0] and 'service5' in sent[2][0]
        assert len({ident for _, ident in sent}) == 3
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0


@pytest.mark.parametrize('error', [OSError, TimeoutError, SystemExit])
def test_first_offer_identity_is_committed_with_claim_and_survives_restart(tmp_path, error):
    path = tmp_path / 'jobwatch.db'
    attempts = []
    with closing(store.connect(path)) as conn:
        with conn:
            runner.Pass(conn, 0).finding('endpoint', 'original', True, notice_data={'name': 'original'})
        def fail(message, command, notice_id):
            attempts.append((message, notice_id))
            with closing(store.connect(path)) as other, other:
                other.execute('BEGIN IMMEDIATE')
                assert store.get(other, 'notice_identity:1') == notice_id
                assert store.get(other, 'delivery_claim')['until'] > 0
            raise error
        if error is SystemExit:
            with pytest.raises(SystemExit):
                runner.deliver_pending(conn, ['test-sender'], fail, 0, 30)
        else:
            runner.deliver_pending(conn, ['test-sender'], fail, 0, 30)
    with closing(store.connect(path)) as conn:
        with conn:
            for index in range(4):
                runner.Pass(conn, 300).finding('endpoint', str(index), True,
                                              notice_data={'name': f'service{index}'})
        sent = []
        runner.deliver_pending(conn, ['test-sender'], lambda m, c, ident: sent.append((m, ident)), 3602, 30)
        assert len(sent) == 2
        assert sent[0][0].startswith('Job watch opened 4 findings:')
        assert 'original' not in sent[0][0]
        assert sent[1] == attempts[0]
        assert not store.get(conn, 'notice_batch:1')
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0


def test_pass_caps_each_state_at_one_group_and_three_delivered_singles(tmp_path):
    with closing(store.connect(tmp_path / 'jobwatch.db')) as conn:
        def queue(state, subject):
            watch = runner.Pass(conn, 0)
            watch.finding('endpoint', subject, True, notice_data={'name': subject})
            if state == 'closed':
                conn.execute('UPDATE notices SET sent=1 WHERE id=(SELECT MAX(id) FROM notices)')
                watch.finding('endpoint', subject, False)
        def fail(*args):
            raise OSError
        for state in ('open', 'closed'):
            for index in range(4):
                with conn:
                    queue(state, f'{state}-single{index}')
                runner.deliver_pending(conn, ['test-sender'], fail, index, 30)
            with conn:
                for batch in range(2):
                    for index in range(4):
                        queue(state, f'{state}-group{batch}-{index}')
                    runner.group_notices(conn)
        sent = []
        notify = lambda m, c, ident: sent.append((m, ident))
        runner.deliver_pending(conn, ['test-sender'], notify, 4000, 30)
        assert len(sent) == 8
        assert all(message.startswith('Job watch ') for message, _ in sent[:2])
        assert all(message.startswith('Watched endpoint ') for message, _ in sent[2:])
        for state in ('open', 'closed'):
            assert sum(f'{state}-' in message for message, _ in sent) == 4
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 10
        runner.deliver_pending(conn, ['test-sender'], notify, 4300, 30)
        assert len(sent) == 12
        assert conn.execute('SELECT count(*) FROM notices WHERE sent=0').fetchone()[0] == 0
