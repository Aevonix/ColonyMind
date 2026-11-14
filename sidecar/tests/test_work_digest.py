"""Digest data uses execution status and owner overlays as separate dimensions."""
from datetime import timedelta

from httpx import ASGITransport, AsyncClient

from colonymind.work.digest import summary_line
from test_work_projection import stores, admit, envelope, operation


def populate(stores):
    for i in range(13):
        ident = f'root-{i:02}'
        op = operation(success=i % 2 == 0, evidence={'kind': 'none'})
        admit(stores, envelope(ident, operations=[op]), source_id=ident)
    stores.work.reconcile()
    for i in range(12):
        ident = f'root-{i:02}'
        row = stores.work.get_root(ident, contact_id='owner')
        stores.work.write_derived(ident, contact_id='owner', expected_revision=row['revision'],
            title=f'Update report number {i}', overlay='confirmed' if i % 2 == 0 else 'corrected')


def test_summary_has_bounded_titles_counts_and_overlay_rates(stores):
    populate(stores)
    summary = stores.work.summary(contact_id='owner')
    assert summary['counts'] == {'proven': 0, 'partial': 0, 'unproven': 7, 'failed': 6}
    assert summary['overlays'] == {'confirmed': 6, 'corrected': 6, 'none': 1}
    assert summary['untitled'] == 1 and summary['owner_corrected'] == 6
    assert len(summary['titles']['unproven']) == len(summary['titles']['failed']) == 5
    assert all(title.startswith('Update report number') for title in summary['titles']['failed'])


async def test_summary_endpoint_supplies_digest_fields(stores, monkeypatch):
    from test_work_api import app, key_authority
    from colonymind.api.routers import work
    populate(stores)
    monkeypatch.setattr(work, 'get_work_store', lambda **kw: stores.work)
    monkeypatch.setattr(work, 'owner_person_id', lambda: 'owner')
    async with AsyncClient(transport=ASGITransport(app=app(key_authority())), base_url='http://local') as client:
        response = await client.get('/v1/host/work/summary')
    assert response.status_code == 200
    summary = response.json()
    assert summary['overlays'] == {'confirmed': 6, 'corrected': 6, 'none': 1}
    assert summary['owner_corrected_rate'] == 6 / 13
    assert len(summary['titles']['unproven']) == len(summary['titles']['failed']) == 5
    assert summary['verdict_coverage']['session_limit'] == 6
    assert summary['verdict_coverage']['reviewed_at'] is None


async def test_mind_caller_supplies_summary_and_off_preserves_digest(fx, monkeypatch):
    # Fixture from consolidation explicitly disables work unless enabled here.
    from test_mind_consolidate import OWNER
    from colonymind.work.store import WorkStore
    fx.mind.faculties['work_records'] = True
    fx.mind.digest_hour = 0
    fx.ledger.record_source('root', contact_id=OWNER, session_id='session',
        messages=[{'role': 'user', 'content': 'Update the report.'}],
        work={'envelope': envelope(), 'principal_id': 'host'}, derive_claims=False)
    WorkStore(fx.ledger).reconcile()
    # Source admission stamps real time; the digest covers the current day.
    from colonymind.util.temporal import now_utc
    fx.now = now_utc() + timedelta(seconds=1)
    captured = []
    async def build(**kwargs):
        captured.append(kwargs)
        return None
    monkeypatch.setattr(fx.mind.outbox, 'build_digest', build)
    await fx.mind._digest(fx.now)
    assert captured[-1]['work'].startswith('Work for you: 1 proven')
    fx.mind._daily.clear()
    fx.mind.faculties['work_records'] = False
    await fx.mind._digest(fx.now)
    assert captured[-1]['work'] is None


def test_no_work_adds_no_digest_line(stores):
    assert summary_line(stores.path, contact_id='owner', since=None, until=None) is None


from test_mind_consolidate import fx  # noqa: E402,F401


async def test_outbox_renders_supplied_work_once_and_none_is_invariant(fx):
    kwargs = {'since': fx.now - timedelta(days=1), 'level': fx.mind.level}
    baseline = await fx.mind.outbox.build_digest(**kwargs)
    assert await fx.mind.outbox.build_digest(**kwargs, work=None) == baseline
    line = 'Work for you: 1 proven, 0 partial, 2 unproven, 1 failed; 0 owner-confirmed, 1 owner-corrected.'
    rendered = await fx.mind.outbox.build_digest(**kwargs, work=line)
    assert rendered.count(line) == 1
