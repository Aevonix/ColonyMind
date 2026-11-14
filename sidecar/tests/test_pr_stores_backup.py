"""Regression coverage for restoring the nightly full, vector-free archive."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

from colonymind.backup import create_full_backup, restore_full_backup
from colonymind import cli


def state_at(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'instance-id').write_text('pr-stores-fixture\n')
    return state


def hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def restore_cli(monkeypatch, archive, destination):
    monkeypatch.setattr(cli, '_load_dotenv', lambda: None)
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(destination))
    cli._cmd_restore(SimpleNamespace(input=str(archive), passphrase=None))


def test_learned_files_and_missing_state_are_reported(tmp_path, monkeypatch, capsys):
    state = state_at(tmp_path)
    weights = state / 'plasticity'
    (weights / 'adapters').mkdir(parents=True)
    (weights / 'adapters' / 'query.json').write_text('{"weights": [0.25, 0.75]}\n')
    (weights / 'state.json').write_text('{"revision": 3}\n')
    expected = hashes(weights)
    archive = create_full_backup(state, tmp_path / 'backups', include_vectors=False)
    dest = tmp_path / 'restored'
    restore_cli(monkeypatch, archive, dest)
    assert hashes(dest / 'plasticity') == expected
    output = capsys.readouterr().out
    from colonymind.secrets.vault import KEY_NAME
    for name in ('api.key', 'instinct.key', KEY_NAME, 'lancedb/', 'host/',
                 'identity.yaml', 'colonymind.yaml', '.env', 'channels.json', '.colonymind-llm-config.json'):
        assert any('Not in the archive:' in line and name in line for line in output.splitlines())
    assert 'Not in the archive: plasticity/' not in output


def test_older_archive_reports_missing_learned_weights(tmp_path):
    state = state_at(tmp_path)
    archive = create_full_backup(state, tmp_path / 'backups', include_vectors=False)
    summary = restore_full_backup(archive, tmp_path / 'restored')
    assert 'plasticity/ (learned weights)' in summary['not_in_archive']


def test_redaction_keeps_live_values_and_reports_missing_values(tmp_path, monkeypatch, capsys):
    import yaml
    state = state_at(tmp_path)
    (state / 'colonymind.yaml').write_text(
        'environment: {SYNTHETIC_API_KEY: fixture-before, NORMAL: archived}\n')
    (state / '.env').write_text('SYNTHETIC_API_KEY=fixture-before\nNORMAL=archived\n')
    archive = create_full_backup(state, tmp_path / 'backups', include_vectors=False)
    dest = tmp_path / 'restored'
    dest.mkdir()
    (dest / 'colonymind.yaml').write_text(
        'environment: {SYNTHETIC_API_KEY: fixture-live, NORMAL: live}\n')
    (dest / '.env').write_text('SYNTHETIC_API_KEY=fixture-live\nNORMAL=live\n')
    restore_cli(monkeypatch, archive, dest)
    assert yaml.safe_load((dest / 'colonymind.yaml').read_text())['environment'] == {
        'SYNTHETIC_API_KEY': 'fixture-live', 'NORMAL': 'archived'}
    assert (dest / '.env').read_text() == 'SYNTHETIC_API_KEY=fixture-live\nNORMAL=archived\n'
    output = capsys.readouterr().out
    assert 'Not in the archive: colonymind.yaml:environment.SYNTHETIC_API_KEY (redacted value)' in output
    assert 'Not in the archive: .env:SYNTHETIC_API_KEY (redacted value)' in output
    assert 'fixture-before' not in output and 'fixture-live' not in output


import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('include_vectors', [False, True])
async def test_restore_requeues_only_without_vectors(tmp_path, include_vectors):
    from test_source_vectors import setup, drain
    state = state_at(tmp_path)
    ledger, store, _, projection = await setup(state)
    ledger.record_source('turn', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'My office is by the orchard.'}])
    await drain(projection)
    assert projection.status('c')['projected_turns'] == 1
    await store.close()
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=include_vectors)
    dest = tmp_path / 'restored'
    restore_full_backup(archive, dest)
    restored, store, _, projection = await setup(dest)
    try:
        with restored._connect() as conn:
            row = conn.execute('SELECT status,cursor FROM source_vector_jobs').fetchone()
        assert tuple(row) == (('complete', 1) if include_vectors else ('pending', 0))
        await drain(projection)
        hits, _ = await projection.search('workplace', contact_id='c', session_id='later')
        assert any(hit['content'] == 'My office is by the orchard.' for hit in hits)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_nightly_round_trip_gate(tmp_path, monkeypatch, capsys):
    """Real CLI flags, SQLite ledger, curated memory and Lance; no remote model."""
    import sys
    import yaml
    from colonymind.memory import curated
    from test_source_vectors import setup, drain
    state = state_at(tmp_path)
    (state / 'identity.yaml').write_text('owner: {name: Fixture}\n')
    (state / 'colonymind.yaml').write_text(
        'environment: {SYNTHETIC_API_KEY: fixture-before, NORMAL: archived}\n')
    weights = state / 'plasticity' / 'query-adapter'
    weights.mkdir(parents=True)
    (weights / 'weights.json').write_text('{"weights": [0.25, 0.75]}\n')
    (weights / 'candidate.bin').write_bytes(bytes(range(32)))
    # These private files are deliberately absent from an unencrypted archive.
    import os
    import sqlite3
    import tarfile
    (state / 'plasticity' / 'comparator.key').write_bytes(os.urandom(32))
    with sqlite3.connect(weights / 'query-vectors.db') as conn:
        conn.execute('CREATE TABLE fixture (value TEXT)')
    excluded = {'comparator.key', 'query-adapter/query-vectors.db'}
    expected_weights = {name: digest for name, digest in hashes(state / 'plasticity').items()
                        if name not in excluded}
    ledger, store, _, projection = await setup(state)
    ledger.record_source('turn', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'The hydrofoil departure code is cedar-42.'}])
    curated.mutate(ledger, 'c', 'mem_fixture', 'remember', request_id='remember',
                   content='I prefer green tea.')
    await drain(projection)
    assert projection.status('c')['projected_turns'] == 1
    await store.close()
    monkeypatch.setattr(cli, '_load_dotenv', lambda: None)
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(state))
    monkeypatch.setattr(sys, 'argv', ['colonymind', 'backup', '--full', '--no-vectors',
                                     '--output', str(tmp_path / 'archives')])
    cli.main()
    archive, = (tmp_path / 'archives').glob('*.tar.gz')
    with tarfile.open(archive) as tar:
        assert not any(Path(member.name).name in ('comparator.key', 'query-vectors.db',
                                                  'query-vectors.db-journal') for member in tar.getmembers())
    # First reconstruct an entirely empty directory. Then provision the live
    # configuration and repeat restore to prove that redactions preserve it.
    dest = tmp_path / 'restored'
    dest.mkdir()
    assert not list(dest.iterdir())
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(dest))
    monkeypatch.setattr(sys, 'argv', ['colonymind', 'restore', '--full', '--input', str(archive)])
    cli.main()
    assert hashes(dest / 'plasticity') == expected_weights
    (dest / 'colonymind.yaml').write_text('environment: {SYNTHETIC_API_KEY: fixture-live}\n')
    cli.main()
    output = capsys.readouterr().out
    assert hashes(dest / 'plasticity') == expected_weights
    assert yaml.safe_load((dest / 'colonymind.yaml').read_text())['environment']['SYNTHETIC_API_KEY'] == 'fixture-live'
    assert not (dest / 'lancedb').exists()
    from colonymind.secrets.vault import KEY_NAME
    for absent in ('api.key', 'instinct.key', KEY_NAME, '.env', 'channels.json',
                   '.colonymind-llm-config.json', 'lancedb/', 'host/',
                   'colonymind.yaml:environment.SYNTHETIC_API_KEY'):
        assert any('Not in the archive:' in line and absent in line for line in output.splitlines())
    assert 'Vector jobs queued for rebuild: 1' in output
    restored, store, _, projection = await setup(dest)
    try:
        assert curated.read(restored, 'c', 'mem_fixture')['content'] == 'I prefer green tea.'
        assert projection.status('c')['pending_turns'] == 1
        with restored._connect() as conn:
            assert tuple(conn.execute('SELECT status,cursor FROM source_vector_jobs').fetchone()) == ('pending', 0)
        await drain(projection)
        hits, _ = await projection.search('vessel identifier', contact_id='c', session_id='later')
        assert any('cedar-42' in hit['content'] for hit in hits)
        assert projection.status('c')['projected_turns'] == 1
    finally:
        await store.close()


@pytest.mark.parametrize('encrypted', [False, True])
def test_comparator_key_and_query_log_follow_encryption(tmp_path, encrypted):
    import os
    import sqlite3
    from colonymind.backup import _decrypt_file
    import tarfile

    state = state_at(tmp_path)
    root = state / 'plasticity'
    query = root / 'learners' / 'fixture' / 'query-vectors.db'
    query.parent.mkdir(parents=True)
    key = os.urandom(32)
    (root / 'comparator.key').write_bytes(key)
    (root / 'erased-refs.json').write_text('[]')
    with sqlite3.connect(query) as conn:
        conn.execute('CREATE TABLE fixture (value TEXT)')
    passphrase = os.urandom(32) if encrypted else None
    archive = create_full_backup(state, tmp_path / 'archives', passphrase=passphrase, include_vectors=False)
    if encrypted:
        decrypted = tmp_path / 'decrypted.tar.gz'
        _decrypt_file(archive, decrypted, passphrase)
        archive = decrypted
    with tarfile.open(archive) as tar:
        names = {Path(member.name).name for member in tar.getmembers()}
    assert ('comparator.key' in names) == encrypted
    assert ('query-vectors.db' in names) == encrypted
    restored = tmp_path / 'restored'
    summary = restore_full_backup(archive, restored)
    assert ('comparator.key (plasticity/)' in summary['not_in_archive']) != encrypted
    assert any('query-vectors.db' in name for name in summary['not_in_archive']) != encrypted
    if encrypted:
        assert (restored / 'plasticity' / 'comparator.key').read_bytes() == key
        assert (restored / 'plasticity' / 'comparator.key').stat().st_mode & 0o777 == 0o600


def test_keyless_restore_reads_weights_and_restarts_learner_with_new_hashes(tmp_path, monkeypatch, capsys):
    from colonymind.plasticity import _io
    from colonymind.plasticity.nightly import _erased_hash
    from colonymind.plasticity.synthetic import SyntheticSource, make_rows
    from colonymind.plasticity.store import LearnerStore
    from test_plasticity_wiring_runtime import runtime

    state = state_at(tmp_path)
    source = SyntheticSource(make_rows(60))
    active = runtime(state, sources={'probe': source})
    try:
        assert active.run()[0]['result'] == 'promoted'
        original = active.pin('probe', 'owner')
        store = active.loaded['probe'].store
        store.record_source_refs({'fixture-row': ['fixture-turn']})
        old_hash = store.source_hash('fixture-turn')
        old_erased_hash = _erased_hash(active.root, 'fixture-row')
        _io.write_json(active.root / 'erased-refs.json', [old_erased_hash])
    finally:
        active.close()
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    dest = tmp_path / 'restored'
    restore_cli(monkeypatch, archive, dest)
    output = capsys.readouterr().out
    assert not (dest / 'plasticity' / 'comparator.key').exists()
    reader = LearnerStore(dest / 'plasticity', 'probe', base=store.revisions.base_identity,
                          declared=['threshold'])
    assert _io.state_digest(reader.revisions.pin('owner').state) == original.closure.state
    restarted = runtime(dest, sources={'probe': source})
    try:
        assert not restarted.rejected
        assert restarted.pin('probe', 'owner').closure.id == original.closure.id
        assert _io.state_digest(restarted.pin('probe', 'owner').state) == original.closure.state
        assert restarted.loaded['probe'].store.source_hash('fixture-turn') != old_hash
        assert _erased_hash(restarted.root, 'fixture-row') != old_erased_hash
        assert restarted.loaded['probe'].store.source_refs()['fixture-row'] == [old_hash]
        assert _io.read_json(restarted.root / 'erased-refs.json') == [old_erased_hash]
        source.add(make_rows(30, prefix='after-restore'))
        assert restarted.run()[0]['result'] == 'promoted'
    finally:
        restarted.close()
    assert ('Learned weights remain usable; old erasure and source hashes no longer match unless this is the instance '
            'that made the archive. Learners create a comparator key if needed.') in output


def test_plasticity_snapshot_skips_vanishing_temporaries(tmp_path, monkeypatch):
    import tarfile
    state = state_at(tmp_path)
    root = state / 'plasticity'
    root.mkdir()
    (root / 'weights.json').write_text('{}')
    for name in ('.lock', 'fixture.db-journal', 'fixture.db-wal', 'fixture.db-shm'):
        (root / name).write_bytes(b'')
    original = Path.iterdir

    def listing(path):
        yield from original(path)
        if path == root:
            # A learner has already renamed this temporary by the time it is copied.
            yield root / '.weights.json.vanished.tmp'

    monkeypatch.setattr(Path, 'iterdir', listing)
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    with tarfile.open(archive) as tar:
        names = [Path(member.name).name for member in tar.getmembers() if member.isfile()]
    assert 'weights.json' in names
    assert not any(name.endswith(('.tmp', '-journal', '-wal', '-shm')) or name == '.lock' for name in names)


def test_plasticity_backup_and_restore_hold_the_state_lock(tmp_path, monkeypatch):
    from colonymind import backup
    from colonymind.plasticity.locking import StateLock
    state = state_at(tmp_path)
    root = state / 'plasticity'
    root.mkdir()
    (root / 'weights.json').write_text('{}')
    destination = tmp_path / 'restored'
    copy = backup._copy_private
    locked_copies = []

    def checked(source, target):
        if source.name == 'weights.json':
            locked_root = root if source.parent == root else destination / 'plasticity'
            assert StateLock(locked_root)._lease.depth > 0
            locked_copies.append(locked_root)
        return copy(source, target)

    monkeypatch.setattr(backup, '_copy_private', checked)
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    restore_full_backup(archive, destination)
    assert locked_copies == [root, destination / 'plasticity']


def test_plasticity_sqlite_snapshot_includes_committed_wal(tmp_path):
    import os
    import sqlite3
    state = state_at(tmp_path)
    root = state / 'plasticity' / 'learners' / 'fixture'
    root.mkdir(parents=True)
    with sqlite3.connect(root / 'query-vectors.db') as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE fixture (value TEXT)')
        conn.execute("INSERT INTO fixture VALUES ('committed')")
        conn.commit()
        passphrase = os.urandom(32)
        archive = create_full_backup(state, tmp_path / 'archives', passphrase=passphrase, include_vectors=False)
    dest = tmp_path / 'restored'
    restore_full_backup(archive, dest, passphrase=passphrase)
    restored = dest / 'plasticity' / 'learners' / 'fixture'
    assert not (restored / 'query-vectors.db-wal').exists()
    assert not (restored / 'query-vectors.db-shm').exists()
    with sqlite3.connect(restored / 'query-vectors.db') as conn:
        assert conn.execute('SELECT value FROM fixture').fetchall() == [('committed',)]
        assert conn.execute('PRAGMA integrity_check').fetchone() == ('ok',)


def test_restore_replaces_newer_learned_files_but_keeps_an_absent_key(tmp_path, monkeypatch, capsys):
    import os
    from colonymind.plasticity import _io
    from colonymind.plasticity.synthetic import SyntheticSource, make_rows
    from test_plasticity_wiring_runtime import runtime
    state = state_at(tmp_path)
    source = SyntheticSource(make_rows(60))
    active = runtime(state, sources={'probe': source})
    try:
        active.run()
        old = active.pin('probe', 'owner')
        key = (active.root / 'comparator.key').read_bytes()
    finally:
        active.close()
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    active = runtime(state, sources={'probe': source})
    try:
        source.add(make_rows(30, prefix='newer'))
        active.run()
        assert active.pin('probe', 'owner').closure.id != old.closure.id
        newer_manifest = active.root / 'learners' / 'probe' / 'manifests' / 'newer.json'
        _io.write_json(newer_manifest, {'revision': 'not-in-archive'})
    finally:
        active.close()
    restore_cli(monkeypatch, archive, state)
    assert not newer_manifest.exists()
    assert (state / 'plasticity' / 'comparator.key').read_bytes() == key
    restored = runtime(state, sources={'probe': source})
    try:
        assert restored.pin('probe', 'owner').closure.id == old.closure.id
        assert _io.state_digest(restored.pin('probe', 'owner').state) == old.closure.state
    finally:
        restored.close()
    assert 'Learned weights restored (plasticity/).' in capsys.readouterr().out


def test_legacy_metadata_instance_id_is_not_reported_missing(tmp_path):
    import json
    import tarfile
    state = state_at(tmp_path)
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    staging = tmp_path / 'staging'
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter='data')
    root, = staging.iterdir()
    (root / 'identity' / 'instance-id').unlink()
    meta = json.loads((root / 'meta.json').read_text())
    meta['colonymind_id'] = meta.pop('instance_id')
    (root / 'meta.json').write_text(json.dumps(meta))
    legacy = tmp_path / 'legacy.tar.gz'
    with tarfile.open(legacy, 'w:gz') as tar:
        tar.add(root, arcname=root.name)
    dest = tmp_path / 'restored'
    summary = restore_full_backup(legacy, dest)
    assert (dest / 'instance-id').read_text().strip() == 'pr-stores-fixture'
    assert 'instance-id' not in summary['not_in_archive']


@pytest.mark.parametrize('malformed', ['live', 'archived'])
def test_malformed_configuration_finishes_restore_without_disclosing_lines(tmp_path, monkeypatch, capsys, caplog, malformed):
    import os
    import tarfile
    from colonymind.turns import TurnIdempotencyLedger
    state = state_at(tmp_path)
    valid = b'environment: {SYNTHETIC_API_KEY: <REDACTED>}\r\n'
    invalid = b'environment: {SYNTHETIC_API_KEY: fixture-malformed-private ]\r\n'
    (state / 'colonymind.yaml').write_bytes(invalid if malformed == 'archived' else valid)
    (state / 'plasticity').mkdir()
    (state / 'plasticity' / 'weights.json').write_text('{}')
    ledger = TurnIdempotencyLedger(state / 'turn-idempotency.db')
    ledger.record_source('fixture', contact_id='c', session_id='s', messages=[{'role': 'user', 'content': 'Fixture'}])
    with ledger._connect() as conn:
        conn.execute("UPDATE source_vector_jobs SET status='complete',cursor=1")
    passphrase = os.urandom(32)
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False, passphrase=passphrase)
    dest = tmp_path / 'restored'
    dest.mkdir()
    (dest / 'colonymind.yaml').write_bytes(invalid if malformed == 'live' else valid)
    summary = restore_full_backup(archive, dest, passphrase=passphrase)
    assert (dest / 'colonymind.yaml').read_bytes() == invalid
    assert (dest / 'plasticity' / 'weights.json').read_text() == '{}'
    assert summary['vector_jobs_queued'] == 1
    assert summary['errors'] == ([
        'Could not parse colonymind.yaml; archived bytes kept.'
    ] if malformed == 'archived' else [
        'colonymind.yaml not applied: could not fill archived placeholders from the live file.'
    ])
    assert 'fixture-malformed-private' not in repr(summary) + caplog.text + capsys.readouterr().out
    # Exercise the CLI's reporting of exactly the same summary without asking for a passphrase.
    import colonymind.backup as backup
    monkeypatch.setattr(backup, 'restore_full_backup', lambda *a, **kw: summary)
    import getpass
    monkeypatch.setattr(getpass, 'getpass', lambda *a, **kw: 'fixture-passphrase')
    restore_cli(monkeypatch, archive, dest)
    output = capsys.readouterr().out
    assert ('Could not parse colonymind.yaml; archived bytes kept.' if malformed == 'archived' else
            'colonymind.yaml not applied: could not fill archived placeholders from the live file.') in output
    assert 'fixture-malformed-private' not in output


@pytest.mark.parametrize('filename', ['colonymind.yaml', '.env'])
@pytest.mark.parametrize('interrupted', [False, True])
def test_configuration_rewrite_is_atomic_and_private(tmp_path, monkeypatch, filename, interrupted):
    import os
    from colonymind import backup
    archived, live = tmp_path / 'archive', tmp_path / 'live'
    archived.mkdir()
    live.mkdir()
    old = 'environment: {SYNTHETIC_API_KEY: fixture-live}\n' if filename.endswith('yaml') else 'SYNTHETIC_API_KEY=fixture-live\n'
    new = 'environment: {SYNTHETIC_API_KEY: <REDACTED>, NORMAL: archived}\n' if filename.endswith('yaml') else 'SYNTHETIC_API_KEY=<REDACTED>\nNORMAL=archived\n'
    (archived / filename).write_text(new)
    target = live / filename
    target.write_text(old)
    original = os.replace
    swaps = []

    def replace(source, destination):
        assert target.read_text() == old
        assert Path(source).parent == target.parent
        assert Path(source).stat().st_mode & 0o777 == 0o600
        swaps.append(destination)
        if interrupted:
            raise OSError('fixture interrupted swap')
        original(source, destination)

    monkeypatch.setattr(backup.os, 'replace', replace)
    if interrupted:
        with pytest.raises(OSError, match='fixture interrupted swap'):
            backup._restore_directory(archived, live)
        assert target.read_text() == old
    else:
        backup._restore_directory(archived, live)
        assert 'fixture-live' in target.read_text() and 'archived' in target.read_text()
        assert target.stat().st_mode & 0o777 == 0o600
    assert swaps == [target]
    assert not list(live.glob('*.tmp'))


def test_named_list_redactions_follow_the_entry_and_never_its_position(tmp_path):
    import yaml
    from colonymind.backup import _restore_config_values
    archived, live = tmp_path / 'archive', tmp_path / 'live'
    archived.mkdir()
    live.mkdir()
    a, b = archived / 'colonymind.yaml', live / 'colonymind.yaml'
    a.write_text('providers:\n- {name: second, key: <REDACTED>}\n- {name: first, key: <REDACTED>}\n'
                 '- {name: new, key: <REDACTED>}\n- {key: <REDACTED>}\n')
    b.write_text('providers:\n- {name: first, key: fixture-first}\n- {name: second, key: fixture-second}\n'
                 '- {name: retired, key: fixture-retired}\n- {key: fixture-anonymous}\n')
    assert yaml.safe_load(_restore_config_values(a, b))['providers'] == [
        {'name': 'second', 'key': 'fixture-second'}, {'name': 'first', 'key': 'fixture-first'},
        {'name': 'new', 'key': '<REDACTED>'}, {'key': '<REDACTED>'}]


@pytest.mark.parametrize('archived_log', [False, True])
def test_restore_keeps_unarchived_query_files_in_place(tmp_path, archived_log):
    import os
    import sqlite3
    from contextlib import closing

    state = state_at(tmp_path)
    learner = state / 'plasticity' / 'learners' / 'query_adapter'
    learner.mkdir(parents=True)
    (learner / 'weights.json').write_text('{"revision": 1}')
    log = learner / 'query-vectors.db'
    with closing(sqlite3.connect(log)) as conn, conn:
        conn.execute('CREATE TABLE fixture (value TEXT)')
        conn.execute("INSERT INTO fixture VALUES ('archived')")
    passphrase = os.urandom(32) if archived_log else None
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False, passphrase=passphrase)
    with closing(sqlite3.connect(log)) as conn, conn:
        conn.execute("INSERT INTO fixture VALUES ('live')")
    journal = log.with_name(log.name + '-journal')
    journal.write_bytes(b'synthetic journal')
    extra = learner.parent / 'live_only'
    extra.mkdir()
    extra_log = extra / 'query-vectors.db'
    extra_log.write_bytes(b'synthetic live-only query log')
    extra_journal = extra / 'query-vectors.db-journal'
    extra_journal.write_bytes(b'synthetic live-only journal')
    (learner / 'weights.json').write_text('{"revision": 2}')
    (extra / 'newer.json').write_text('{}')

    def facts(path):
        stat = path.stat()
        return stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, path.read_bytes()

    kept = [extra_log, extra_journal] + ([] if archived_log else [log, journal])
    before = {path: facts(path) for path in kept}
    restore_full_backup(archive, state, passphrase=passphrase)
    assert {path: facts(path) for path in kept} == before
    assert (learner / 'weights.json').read_text() == '{"revision": 1}'
    assert not (extra / 'newer.json').exists()
    if archived_log:
        assert not journal.exists()
        with closing(sqlite3.connect(log)) as conn:
            assert conn.execute('SELECT value FROM fixture').fetchall() == [('archived',)]


@pytest.mark.parametrize('same_instance', [False, True])
def test_keyless_archive_warns_even_with_a_destination_key(tmp_path, monkeypatch, capsys, same_instance):
    import os

    state = state_at(tmp_path)
    root = state / 'plasticity'
    root.mkdir()
    original_key = os.urandom(32)
    (root / 'comparator.key').write_bytes(original_key)
    (root / 'weights.json').write_text('{}')
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    destination = state if same_instance else tmp_path / 'other'
    if not same_instance:
        (destination / 'plasticity').mkdir(parents=True)
        (destination / 'plasticity' / 'comparator.key').write_bytes(os.urandom(32))
    key = (destination / 'plasticity' / 'comparator.key').read_bytes()
    summary = restore_full_backup(archive, destination)
    assert summary.get('plasticity_key_missing') is True
    assert (destination / 'plasticity' / 'comparator.key').read_bytes() == key
    restore_cli(monkeypatch, archive, destination)
    assert ('old erasure and source hashes no longer match unless this is the instance '
            'that made the archive') in capsys.readouterr().out


@pytest.mark.parametrize('filename', ['colonymind.yaml', '.env'])
@pytest.mark.parametrize('existing', [False, True])
def test_non_utf8_archived_settings_finish_restore_with_exact_bytes(tmp_path, caplog, filename, existing):
    import io
    import tarfile
    from colonymind.turns import TurnIdempotencyLedger

    state = state_at(tmp_path)
    raw = b'# synthetic archived setting\r\nNORMAL=fixture-\xff\xfe\r\n'
    (state / 'plasticity').mkdir()
    (state / 'plasticity' / 'weights.json').write_text('{}')
    ledger = TurnIdempotencyLedger(state / 'turn-idempotency.db')
    ledger.record_source('fixture', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Fixture'}])
    with ledger._connect() as conn:
        conn.execute("UPDATE source_vector_jobs SET status='complete',cursor=1")
    original = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    archive = tmp_path / 'invalid-settings.tar.gz'
    with tarfile.open(original) as source, tarfile.open(archive, 'w:gz') as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
        root = source.getmembers()[0].name.split('/')[0]
        folder = 'identity' if filename == 'colonymind.yaml' else 'config'
        member = tarfile.TarInfo(f'{root}/{folder}/{filename}')
        member.size = len(raw)
        target.addfile(member, io.BytesIO(raw))
    destination = tmp_path / 'restored'
    if existing:
        destination.mkdir()
        (destination / filename).write_text('# synthetic live setting\n')

    summary = restore_full_backup(archive, destination)
    assert (destination / filename).read_bytes() == raw
    assert (destination / filename).stat().st_mode & 0o777 == 0o600
    assert (destination / 'plasticity' / 'weights.json').read_text() == '{}'
    assert summary['vector_jobs_queued'] == 1
    assert summary['errors'] == [f'Could not decode {filename} as UTF-8; archived bytes kept.']
    assert 'NORMAL' not in repr(summary) + caplog.text


@pytest.mark.parametrize('relative', ['linked', 'learners/linked'])
def test_restore_replaces_directory_links_without_touching_targets(tmp_path, relative):
    state = state_at(tmp_path)
    archived = state / 'plasticity' / relative
    archived.mkdir(parents=True)
    (archived / 'weights.json').write_text('{"revision": 1}')
    archive = create_full_backup(state, tmp_path / 'archives', include_vectors=False)

    destination = tmp_path / 'restored'
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'weights.json').write_text('{"revision": 2}')
    (outside / 'later.json').write_text('{"revision": 3}')
    link = destination / 'plasticity' / relative
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    before = hashes(outside)

    restore_full_backup(archive, destination)

    assert hashes(outside) == before
    assert not link.is_symlink()
    assert (link / 'weights.json').read_text() == '{"revision": 1}'
    assert not (link / 'later.json').exists()


@pytest.mark.parametrize('filename,problem', [
    ('.env', 'archived_encoding'), ('colonymind.yaml', 'archived_encoding'),
    ('.env', 'live_encoding'), ('colonymind.yaml', 'live_encoding'),
    ('.env', 'live_read'), ('colonymind.yaml', 'live_read'),
    ('colonymind.yaml', 'archived_yaml'), ('colonymind.yaml', 'live_yaml'),
    ('colonymind.yaml', 'live_value'),
])
def test_unfillable_placeholders_leave_live_configuration_untouched(tmp_path, monkeypatch, caplog,
                                                                   filename, problem):
    import io
    import tarfile

    state = state_at(tmp_path)
    (state / 'plasticity').mkdir()
    (state / 'plasticity' / 'weights.json').write_text('{"revision": 1}')
    original = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    archived = b'SETTING=<REDACTED>\r\n' if filename == '.env' else b'setting: <REDACTED>\r\n'
    live = b'SETTING=fixture-live\r\n' if filename == '.env' else b'setting: fixture-live\r\n'
    if problem == 'archived_encoding':
        archived += b'# fixture-archived-\xff\n'
    elif problem == 'live_encoding':
        live += b'# fixture-live-\xff\n'
    elif problem == 'archived_yaml':
        archived += b'broken: [fixture-archived\n'
    elif problem == 'live_yaml':
        live += b'broken: [fixture-live\n'
    elif problem == 'live_value':
        # Well-formed YAML whose date does not exist: the loader raises ValueError, not a YAML error.
        live += b'fixture-live-when: 2026-02-30\n'
    archive = tmp_path / 'placeholder-settings.tar.gz'
    with tarfile.open(original) as source, tarfile.open(archive, 'w:gz') as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
        root = source.getmembers()[0].name.split('/')[0]
        member = tarfile.TarInfo(f'{root}/config/{filename}')
        member.size = len(archived)
        target.addfile(member, io.BytesIO(archived))
    destination = tmp_path / 'restored'
    destination.mkdir()
    installed = destination / filename
    installed.write_bytes(live)
    installed.chmod(0o640)
    before = installed.stat()
    if problem == 'live_read':
        read_text = Path.read_text

        def unreadable(path, *args, **kwargs):
            if path == installed:
                raise OSError('fixture-live read unavailable')
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', unreadable)

    summary = restore_full_backup(archive, destination)

    assert installed.read_bytes() == live
    after = installed.stat()
    for field in ('st_ino', 'st_mode', 'st_mtime_ns', 'st_ctime_ns'):
        assert getattr(after, field) == getattr(before, field)
    assert (destination / 'plasticity' / 'weights.json').read_text() == '{"revision": 1}'
    assert len(summary['errors']) == 1
    assert filename in summary['errors'][0] and 'not applied' in summary['errors'][0]
    assert 'fixture-' not in repr(summary['errors']) + caplog.text
    assert '2026-02-30' not in repr(summary['errors']) + caplog.text


# Well-formed YAML that the loader answers with ValueError (a date that does not exist) or with
# RecursionError (nesting deeper than the interpreter allows) instead of a YAML error.
UNLOADABLE = {'date': b'2026-02-30', 'depth': b'[' * 5000 + b']' * 5000}


def archive_with_settings(tmp_path, archived):
    """A vector-free archive whose identity/colonymind.yaml holds exactly the given bytes."""
    import io
    import tarfile
    from colonymind.turns import TurnIdempotencyLedger

    state = state_at(tmp_path)
    (state / 'plasticity').mkdir()
    (state / 'plasticity' / 'weights.json').write_text('{"revision": 1}')
    ledger = TurnIdempotencyLedger(state / 'turn-idempotency.db')
    ledger.record_source('fixture', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Fixture'}])
    with ledger._connect() as conn:
        conn.execute("UPDATE source_vector_jobs SET status='complete',cursor=1")
    original = create_full_backup(state, tmp_path / 'archives', include_vectors=False)
    archive = tmp_path / 'unloadable-settings.tar.gz'
    with tarfile.open(original) as source, tarfile.open(archive, 'w:gz') as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
        root = source.getmembers()[0].name.split('/')[0]
        member = tarfile.TarInfo(f'{root}/identity/colonymind.yaml')
        member.size = len(archived)
        target.addfile(member, io.BytesIO(archived))
    return archive


@pytest.mark.parametrize('value', ['date', 'depth'])
@pytest.mark.parametrize('existing', [False, True])
def test_unloadable_archived_settings_finish_restore_with_exact_bytes(tmp_path, caplog, value, existing):
    raw = b'# synthetic archived setting\r\nfixture-archived-when: ' + UNLOADABLE[value] + b'\r\n'
    archive = archive_with_settings(tmp_path, raw)
    destination = tmp_path / 'restored'
    if existing:
        destination.mkdir()
        (destination / 'colonymind.yaml').write_text('# synthetic live setting\n')

    summary = restore_full_backup(archive, destination)

    assert (destination / 'colonymind.yaml').read_bytes() == raw
    assert (destination / 'plasticity' / 'weights.json').read_text() == '{"revision": 1}'
    assert summary['vector_jobs_queued'] == 1
    assert summary['errors'] == ['Could not parse colonymind.yaml; archived bytes kept.']
    disclosed = repr(summary) + caplog.text
    assert 'fixture-' not in disclosed and UNLOADABLE[value][:10].decode() not in disclosed


@pytest.mark.parametrize('value', ['date', 'depth'])
@pytest.mark.parametrize('holder', ['live', 'archived'])
def test_unloadable_settings_beside_placeholders_leave_live_file_untouched(tmp_path, caplog, value, holder):
    archived = b'setting: <REDACTED>\r\n'
    live = b'setting: fixture-live\r\n'
    unloadable = b'fixture-' + holder.encode() + b'-when: ' + UNLOADABLE[value] + b'\r\n'
    if holder == 'live':
        live += unloadable
    else:
        archived += unloadable
    archive = archive_with_settings(tmp_path, archived)
    destination = tmp_path / 'restored'
    destination.mkdir()
    installed = destination / 'colonymind.yaml'
    installed.write_bytes(live)
    installed.chmod(0o640)
    before = installed.stat()

    summary = restore_full_backup(archive, destination)

    assert installed.read_bytes() == live
    after = installed.stat()
    for field in ('st_ino', 'st_mode', 'st_mtime_ns', 'st_ctime_ns'):
        assert getattr(after, field) == getattr(before, field)
    assert (destination / 'plasticity' / 'weights.json').read_text() == '{"revision": 1}'
    assert summary['vector_jobs_queued'] == 1
    assert len(summary['errors']) == 1
    assert 'colonymind.yaml' in summary['errors'][0] and 'not applied' in summary['errors'][0]
    disclosed = repr(summary) + caplog.text
    assert 'fixture-' not in disclosed and UNLOADABLE[value][:10].decode() not in disclosed
