"""Well-formed YAML that the loader, the merge or the dump answers with another error does not end a restore."""
import io
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from colonymind import cli
from colonymind.backup import create_full_backup, restore_full_backup
from colonymind.turns import TurnIdempotencyLedger

P = b'<REDACTED>'
KEPT = 'Could not parse colonymind.yaml; archived bytes kept.'
NOT_APPLIED = 'colonymind.yaml not applied: could not fill archived placeholders from the live file.'
UNINSPECTED = 'Could not inspect redacted values in colonymind.yaml'
REDACTED_VALUE = 'colonymind.yaml:environment.FIXTURE_API_KEY (redacted value)'

# Values the loader answers with an error of its own kind: a key, an index or an attribute it does not find.
LOADER = {
    'bool-word': b'n: !!bool fixture-word\n',
    'int-empty': b'n: !!int ""\n',
    'float-empty': b'n: !!float ""\n',
    'timestamp-word': b'n: !!timestamp fixture-word\n',
}
# Values that load, and that the merge or the dump of the merged file cannot take.
MERGE = {
    'depth-350': b'deep: ' + b'[' * 350 + b']' * 350 + b'\n',
    'depth-400': b'deep: ' + b'[' * 400 + b']' * 400 + b'\n',
    'anchor-in-a-list': b'a: &a [*a]\n',
    'anchor-in-a-map': b'a: &a {k: *a}\n',
    'hex-4000-digits': b'n: 0x' + b'f' * 4000 + b'\n',
}
# Values the loader refuses with a YAML error, a ValueError or a RecursionError.
REFUSED = {
    'date-feb-30': b'when: 2026-02-30\n',
    'date-month-13': b'when: 2026-13-01\n',
    'hour-25': b'when: 2026-01-01 25:00:00\n',
    'year-0': b'when: 0000-01-01\n',
    'minute-61': b'when: 2026-01-01T10:61:00\n',
    'zone-99': b'when: 2026-01-01T10:00:00+99:00\n',
    'int-5000-digits': b'n: ' + b'1' * 5000 + b'\n',
    'int-0b-underscore': b'n: 0b_\n',
    'depth-5000': b'deep: ' + b'[' * 5000 + b']' * 5000 + b'\n',
    'merge-key-scalar': b'<<: 5\n',
    'duplicate-anchor': b'a: &x fixture-one\nb: &x fixture-two\nc: *x\n',
    'int-word': b'n: !!int fixture-word\n',
    'binary-bad': b'n: !!binary "@@fixture@@"\n',
    'two-documents': b'---\na: 1\n---\nb: 2\n',
}
# Values that load, merge and dump.
TAKEN = {
    'depth-200': b'deep: ' + b'[' * 200 + b']' * 200 + b'\n',
    'depth-250': b'deep: ' + b'[' * 250 + b']' * 250 + b'\n',
    'depth-300': b'deep: ' + b'[' * 300 + b']' * 300 + b'\n',
    'merge-key-self': b'a: &a {<<: *a}\n',
}
VALUES = {**LOADER, **MERGE, **REFUSED, **TAKEN}

# Where the value sits, and whether the archived file carries a placeholder that needs the live file.
SHAPES = ('no-placeholder', 'live-holds-value', 'archived-holds-value')
SETTING = b'environment:\n  FIXTURE_API_KEY: '
ENDED = ([(case, shape) for case in LOADER for shape in SHAPES]
         + [(case, 'archived-holds-value') for case in MERGE])


def _files(case, shape):
    """The archived and the live colonymind.yaml."""
    if shape == 'no-placeholder':
        return b'setting: fixture-archived\n' + VALUES[case], b'setting: fixture-live\n'
    if shape == 'live-holds-value':
        return SETTING + P + b'\n', SETTING + b'fixture-live\n' + VALUES[case]
    return SETTING + P + b'\n' + VALUES[case], SETTING + b'fixture-live\n'


def _expected(case, shape):
    """The file a finished restore leaves, and its error lines."""
    if case in LOADER or case in REFUSED:
        return ('archived', [KEPT]) if shape == 'no-placeholder' else ('live', [NOT_APPLIED])
    if shape == 'no-placeholder':
        return 'archived', []
    if case in MERGE and shape == 'archived-holds-value':
        return 'live', [NOT_APPLIED]
    return 'merged', []


def _backup(tmp_path):
    """A vector-free archive of a small instance: its id, a config file, learned weights and one source."""
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'instance-id').write_text('pr-fu-a-instance\n')
    (state / 'channels.json').write_text('{"synthetic": true}\n')
    (state / 'plasticity').mkdir()
    (state / 'plasticity' / 'weights.json').write_text('{"revision": 1}')
    ledger = TurnIdempotencyLedger(state / 'turn-idempotency.db')
    ledger.record_source('synthetic', contact_id='c', session_id='s',
                         messages=[{'role': 'user', 'content': 'Synthetic'}])
    with ledger._connect() as conn:
        conn.execute("UPDATE source_vector_jobs SET status='complete',cursor=1")
    return create_full_backup(state, tmp_path / 'archives', include_vectors=False)


def _with_settings(original, path, archived):
    """A copy of the archive whose identity/colonymind.yaml holds exactly the given bytes."""
    with tarfile.open(original) as source, tarfile.open(path, 'w:gz') as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member) if member.isfile() else None)
        root = source.getmembers()[0].name.split('/')[0]
        member = tarfile.TarInfo(f'{root}/identity/colonymind.yaml')
        member.size = len(archived)
        target.addfile(member, io.BytesIO(archived))
    return path


def _installed(directory, live):
    directory.mkdir()
    installed = directory / 'colonymind.yaml'
    installed.write_bytes(live)
    installed.chmod(0o640)
    return installed


def _disclosed(text):
    """A value of either file: a fixture word, a run of brackets or a run of digits."""
    return [token for token in ('fixture-', '@@', '[' * 10, 'f' * 40, '1' * 40) if token in text]


@pytest.mark.parametrize(('case', 'shape'), ENDED)
def test_a_value_that_ended_the_restore_lets_it_finish_its_other_stages(tmp_path, caplog, case, shape):
    archived, live = _files(case, shape)
    archive = _with_settings(_backup(tmp_path), tmp_path / 'settings.tar.gz', archived)
    destination = tmp_path / 'restored'
    installed = _installed(destination, live)
    before = installed.stat()
    caplog.clear()

    summary = restore_full_backup(archive, destination)

    # Identity, databases, config, learned weights and the queue of vector jobs: every other stage ran.
    assert (destination / 'instance-id').read_text() == 'pr-fu-a-instance\n'
    assert summary['databases'] == ['turn-idempotency.db']
    assert (destination / 'channels.json').read_text() == '{"synthetic": true}\n'
    assert (destination / 'plasticity' / 'weights.json').read_text() == '{"revision": 1}'
    assert summary['vector_jobs_queued'] == 1
    if shape == 'no-placeholder':
        # No placeholder to fill: the archived bytes are the file, as after a YAML error.
        assert installed.read_bytes() == archived
        assert summary['errors'] == [KEPT]
    else:
        # A placeholder that could not be filled: the live file is not touched, and the summary says so.
        assert installed.read_bytes() == live
        after = installed.stat()
        for field in ('st_ino', 'st_mode', 'st_mtime_ns', 'st_ctime_ns'):
            assert getattr(after, field) == getattr(before, field)
        assert summary['errors'] == [NOT_APPLIED]
    # The archived file is read once more for its placeholders; one that does not load is passed over.
    unloadable = case in LOADER and shape != 'live-holds-value'
    assert (REDACTED_VALUE in summary['not_in_archive']) is (shape != 'no-placeholder' and not unloadable)
    assert (UNINSPECTED in caplog.text) is (unloadable or case.startswith('anchor'))
    assert _disclosed(repr(summary) + caplog.text) == []
    assert 'Traceback' not in caplog.text and all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize('shape', ['no-placeholder', 'archived-holds-value'])
def test_the_restore_command_prints_the_error_line_and_never_the_word(tmp_path, monkeypatch, capsys, shape):
    archived, live = _files('bool-word', shape)
    archive = _with_settings(_backup(tmp_path), tmp_path / 'settings.tar.gz', archived)
    destination = tmp_path / 'restored'
    _installed(destination, live)
    monkeypatch.setattr(cli, '_load_dotenv', lambda: None)
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(destination))
    capsys.readouterr()

    cli._cmd_restore(SimpleNamespace(input=str(archive), passphrase=None))

    printed = capsys.readouterr()
    assert '  ' + _expected('bool-word', shape)[1][0] in printed.out.splitlines()
    assert '  Vector jobs queued for rebuild: 1' in printed.out.splitlines()
    assert _disclosed(printed.out + printed.err) == []


def _left(now, archived, live):
    if now == archived:
        return 'archived'
    if now == live:
        return 'live'
    try:
        # Merged: the archived file with the live value in place of the placeholder.
        merged = P not in now and yaml.safe_load(now) == yaml.safe_load(archived.replace(P, b'fixture-live'))
    except Exception:
        merged = False
    return 'merged' if merged else 'other'


def test_every_value_finishes_and_one_that_finished_before_ends_as_it_did(tmp_path, caplog):
    original = _backup(tmp_path)
    wrong = []
    for n, (case, shape) in enumerate((case, shape) for case in VALUES for shape in SHAPES):
        archived, live = _files(case, shape)
        archive = _with_settings(original, tmp_path / f'settings-{n}.tar.gz', archived)
        destination = tmp_path / f'restored-{n}'
        installed = _installed(destination, live)
        caplog.clear()
        try:
            summary = restore_full_backup(archive, destination)
        except Exception as ended:
            wrong.append((case, shape, 'ended', type(ended).__name__))
            continue
        outcome = (_left(installed.read_bytes(), archived, live), summary['errors'])
        if outcome != _expected(case, shape):
            wrong.append((case, shape, *outcome))
        if summary.get('vector_jobs_queued') != 1 or _disclosed(repr(summary) + caplog.text):
            wrong.append((case, shape, 'unfinished or disclosed'))
    assert wrong == []
