"""The live pass: each new send to a contact is served once as it happens and labelled from the comms ledger.

The pass runs every 60 seconds (``Driver.live``). A send before the learner started, or one whose reply is already
recorded, is history and never served; a served send is labelled 1 once its reply is recorded and 0 once the
declared 24-hour horizon passed without one. Forgetting a row's source unserves what it fed.
"""
from datetime import timedelta

from colonymind.contacts.comms import CommsLog
from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from test_plasticity_contact_reply_timing import HOUR, NOW, PRIVATE, comms, contacts, service, talk


def live(tmp_path, now):
    contacts(tmp_path)
    runtime = service(tmp_path, clock=lambda: now[0])
    binding = runtime.loaded['contacts']
    return runtime, binding, binding.driver


def records(binding):
    return binding.comparator.records()


def test_a_new_send_is_served_once_and_labelled_by_its_reply(tmp_path):
    now = [NOW]
    runtime, binding, driver = live(tmp_path, now)
    log = comms(tmp_path)
    try:
        talk(log, 'cid-sam', 'in', NOW - timedelta(minutes=1))
        talk(log, 'cid-sam', 'out', NOW)
        now[0] += timedelta(minutes=1)
        driver.poll(runtime)
        driver.poll(runtime)
        record, = records(binding)
        assert record['output'] == record['reference_output'] == .5 and record['outcome'] is None
        now[0] += 3 * HOUR
        talk(log, 'cid-sam', 'in', now[0] - HOUR)
        driver.poll(runtime)
        record, = records(binding)
        assert record['outcome']['observed'] == 1. and record['outcome']['teacher'] == 'realized_value'
        for path in (tmp_path / 'plasticity').rglob('*'):
            if path.is_file():
                assert all(value.encode() not in path.read_bytes() for value in PRIVATE), path
    finally:
        runtime.close()
        log._conn.close()


def test_no_reply_by_the_horizon_labels_0(tmp_path):
    now = [NOW]
    runtime, binding, driver = live(tmp_path, now)
    log = comms(tmp_path)
    try:
        talk(log, 'cid-sam', 'out', NOW + timedelta(seconds=30))
        now[0] += timedelta(minutes=1)
        driver.poll(runtime)
        now[0] += 23 * HOUR
        driver.poll(runtime)
        assert records(binding)[0]['outcome'] is None                      # the horizon is still ahead
        talk(log, 'cid-sam', 'in', now[0] + 2 * HOUR)                       # a reply after the horizon
        now[0] += 3 * HOUR
        driver.poll(runtime)
        assert records(binding)[0]['outcome']['observed'] == 0.
    finally:
        runtime.close()
        log._conn.close()


def test_history_is_never_served(tmp_path):
    now = [NOW]
    runtime, binding, driver = live(tmp_path, now)
    log = comms(tmp_path)
    try:
        talk(log, 'cid-sam', 'out', NOW - HOUR)                             # before the learner started
        talk(log, 'cid-robin', 'out', NOW + timedelta(seconds=10))
        talk(log, 'cid-robin', 'in', NOW + timedelta(seconds=20))           # replied before the first pass
        now[0] += timedelta(minutes=1)
        driver.poll(runtime)
        assert records(binding) == []
    finally:
        runtime.close()
        log._conn.close()


def test_a_disabled_or_absent_learner_serves_nothing(tmp_path):
    now = [NOW]
    runtime, binding, driver = live(tmp_path, now)
    log = comms(tmp_path)
    try:
        talk(log, 'cid-sam', 'out', NOW + timedelta(seconds=10))
        now[0] += timedelta(minutes=1)
        runtime.enabled = False
        driver.poll(runtime)
        assert records(binding) == []
    finally:
        runtime.close()
        log._conn.close()


def test_forgetting_a_rows_source_unserves_the_send(tmp_path):
    now = [NOW]
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    runtime, binding, driver = live(tmp_path, now)
    log = CommsLog(str(tmp_path / 'colonymind-comms.db'), source_ledger=ledger)
    try:
        ledger.record_source('contact-turn', contact_id='cid-sam', session_id='contact-session',
                             messages=[{'role': 'user', 'content': PRIVATE[-1]}], occurred_at=NOW.isoformat(),
                             derive_claims=False)
        lineage, _ = log.source_input('contact-turn', 'cid-sam')
        talk(log, 'cid-sam', 'in', NOW, source_lineage=lineage)
        talk(log, 'cid-sam', 'out', NOW + timedelta(seconds=5), source_lineage=lineage)
        now[0] += timedelta(minutes=1)
        driver.poll(runtime)
        record, = records(binding)
        ref = driver.rows.ref(driver.rows.snapshot()[0][0]['id'])
        assert 'contact-turn' in driver.rows.source_ids(ref)
        curated.forget_sources(ledger, 'cid-sam', source_ids=['contact-turn'])
        assert records(binding) == []
        assert driver.rows.snapshot() == ([], [])
    finally:
        runtime.close()
        log._conn.close()


# Reply-evidence and retraction regression scenarios.

def test_a_positive_rests_on_its_reply_so_withholding_the_reply_retracts_it(tmp_path, monkeypatch):
    """Finding 2: a forecast served from ``send-turn`` and labelled 1 by ``reply-turn`` depends on both, so the
    dependency index finds it by the reply and withholding the reply takes the positive back."""
    now = [NOW]
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    runtime, binding, driver = live(tmp_path, now)
    log = CommsLog(str(tmp_path / 'colonymind-comms.db'), source_ledger=ledger)
    try:
        for ident in ('send-turn', 'reply-turn'):
            ledger.record_source(ident, contact_id='cid-sam', session_id='s',
                                 messages=[{'role': 'user', 'content': ident}], derive_claims=False)
        send_lineage, _ = log.source_input('send-turn', 'cid-sam')
        reply_lineage, _ = log.source_input('reply-turn', 'cid-sam')
        talk(log, 'cid-sam', 'out', NOW, source_lineage=send_lineage)
        now[0] = NOW + timedelta(minutes=1)
        driver.poll(runtime)
        ref = driver.rows.ref(driver.rows.snapshot()[0][0]['id'])
        talk(log, 'cid-sam', 'in', NOW + HOUR, source_lineage=reply_lineage)
        now[0] = NOW + 2 * HOUR
        driver.poll(runtime)
        record, = records(binding)
        assert record['outcome']['observed'] == 1
        assert {'send-turn', 'reply-turn'} <= set(driver.rows.source_ids(ref))
        assert ref in binding.store.refs_resting_on(['reply-turn'])[0]          # the persisted index too
        monkeypatch.setattr(driver.rows.sources, 'withheld', lambda ids: {'reply-turn'}.intersection(ids))
        assert driver.rows.snapshot()[0][0]['reply_at'] is None
        assert records(binding) == []                                             # the positive is taken back
    finally:
        runtime.close()
        log._conn.close()


def test_a_burst_polled_while_it_grows_has_one_forecast_and_one_label(tmp_path):
    """Finding 3: send at 12:00, poll at 12:01, send again at 12:02, poll at 12:03, reply at 12:04: one burst, so
    one forecast (for its last row; the first row's forecast is superseded) and one label."""
    now = [NOW]
    runtime, binding, driver = live(tmp_path, now)
    log = comms(tmp_path)
    try:
        talk(log, 'cid-sam', 'out', NOW)
        now[0] = NOW + timedelta(minutes=1)
        driver.poll(runtime)
        talk(log, 'cid-sam', 'out', NOW + timedelta(minutes=2))
        now[0] = NOW + timedelta(minutes=3)
        driver.poll(runtime)
        talk(log, 'cid-sam', 'in', NOW + timedelta(minutes=4))
        now[0] = NOW + timedelta(hours=25)
        driver.poll(runtime)
        sends = driver.rows.snapshot()[0]
        result = records(binding)
        assert len(sends) == 1 and len(result) == 1, (len(sends), result)
        assert result[0]['outcome']['observed'] == 1
        assert result[0]['ref_hash'] == binding.comparator._path(driver.rows.ref(sends[0]['id'])).stem
    finally:
        runtime.close()
        log._conn.close()

