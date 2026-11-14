"""A completed suite and an interrupted one must leave distinguishable daily records."""
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/daily_suite_verdict.sh'
SERVICE = SCRIPT.parent / 'systemd/colonymind-suite-verdict.service'
VERDICT = '== known failures: 0 of 2 still fail; 0 new; 2 now pass'


@pytest.mark.parametrize('summary,status,verdict', [
    ('== known failures: 0 of 2 still fail; 0 new; 2 now pass', 0, 'PASS'),
    ('== known failures: 2 of 2 still fail; 0 new; 0 now pass', 0, 'KNOWN_FAILURES'),
    ('== known failures: 2 of 2 still fail; 1 new; 0 now pass', 1, 'NEW_FAILURES'),
    ('suite was interrupted', 1, 'INCOMPLETE'),
    ('== known failures: 2 of 2 still fail; 0 new; 0 now pass', 2, 'INCOMPLETE'),
])
def test_daily_record_preserves_verdict_and_exit_status(tmp_path, summary, status, verdict):
    tree = tmp_path / 'checkout'
    (tree / 'scripts').mkdir(parents=True)
    runner = tree / 'scripts/suite_env.sh'
    runner.write_text(f"#!/bin/bash\nprintf '%s\\n' '{summary}'\nexit {status}\n")
    subprocess.run(['git', 'init', '-q', str(tree)], check=True)
    subprocess.run(['git', '-C', str(tree), 'add', 'scripts'], check=True)
    subprocess.run(['git', '-C', str(tree), '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture runner'], check=True)
    commit = subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip()
    log = tmp_path / 'verdicts.log'
    result = subprocess.run(['bash', str(SCRIPT), str(tree), str(log)], capture_output=True, text=True,
                            env={**os.environ, 'SUITE_PRIVATE_ROOT': str(tmp_path)})
    assert result.returncode == (2 if verdict == 'INCOMPLETE' else status), result.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and f'commit={commit} verdict={verdict}' in lines[0]
    if verdict != 'INCOMPLETE':
        assert 'known=' + ('0/2' if verdict == 'PASS' else '2/2') in lines[0]
    assert result.stdout.strip() == lines[0]
    output = Path(lines[0].split(' output=', 1)[1])
    assert output.read_text().strip() == summary


def checkout(path, runner, commit=True):
    """A checkout whose runner is the given script, and its commit ID (``?`` when it is left without a commit)."""
    (path / 'scripts').mkdir(parents=True)
    (path / 'scripts/suite_env.sh').write_text('#!/bin/bash\n' + runner)
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    if not commit:
        return path, '?'
    subprocess.run(['git', '-C', str(path), 'add', 'scripts'], check=True)
    subprocess.run(['git', '-C', str(path), '-c', 'user.name=Fixture', '-c', 'user.email=fixture.invalid',
                    'commit', '-qm', 'Fixture runner'], check=True)
    return path, subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def daily(tree, log, root=None, first=None, changes=None):
    """Run the wrapper: without ``root`` the private root is unset; commands in ``first`` are found first;
    ``changes`` sets environment variables and takes out those it maps to ``None``."""
    env = {name: value for name, value in os.environ.items() if name != 'SUITE_PRIVATE_ROOT'}
    if root is not None:
        env['SUITE_PRIVATE_ROOT'] = str(root)
    if first is not None:
        env['PATH'] = f'{first}{os.pathsep}{env["PATH"]}'
    for name, value in (changes or {}).items():
        env.pop(name, None)
        if value is not None:
            env[name] = value
    return subprocess.run(['bash', str(SCRIPT), str(tree), str(log)], capture_output=True, text=True, env=env)


def failing(directory, name):
    """A directory with a command of that name which only fails."""
    directory.mkdir()
    (directory / name).write_text('#!/bin/sh\nexit 1\n')
    (directory / name).chmod(0o755)
    return directory


def aged(path, hours):
    """Date the entry itself (a link, not its target) that many hours before now; after now when negative."""
    at = time.time() - 3600 * hours
    os.utime(path, (at, at), follow_symlinks=False)
    return path


def earlier_runs(root, ages):
    """Output as earlier runs left it, one run per age in hours: the run directories and the wrapper's logs."""
    runs, logs = [], []
    for age in ages:
        run = root / f'colonymind-suite.Run{abs(age):03d}'
        run.mkdir()
        (run / 'sidecar.log').write_text('earlier')
        log = root / f'daily-suite.Run{abs(age):03d}.log'
        log.write_text('earlier')
        runs.append(aged(run, age))
        logs.append(aged(log, age))
    return runs, logs


COMPLETE = f"printf '%s\\n' '{VERDICT}'\n"
COMPLETE_IN_ITS_OWN_DIRECTORY = 'mktemp -d "$SUITE_PRIVATE_ROOT/colonymind-suite.XXXXXX"\n' + COMPLETE


@pytest.mark.parametrize('case,status,named,reason', [
    ('no private root', 1, False, 'SUITE_PRIVATE_ROOT is not set to a private output directory'),
    ('private root is a file', 1, True, 'SUITE_PRIVATE_ROOT is not a directory'),
    ('no checkout', 1, False, 'the checkout path cannot be entered'),
    ('checkout without a commit', 128, False, 'git finds no commit in the checkout'),
    ('mktemp fails', 1, True, 'mktemp made no output file in SUITE_PRIVATE_ROOT'),
])
def test_a_start_that_cannot_run_leaves_one_incomplete_line_and_its_status(tmp_path, case, status, named, reason):
    tree, commit = checkout(tmp_path / 'checkout', COMPLETE, commit=case != 'checkout without a commit')
    root = tmp_path / 'private'
    if case == 'private root is a file':
        root.write_text('')
    else:
        root.mkdir()
    log = tmp_path / 'verdicts.log'
    result = daily(tmp_path / 'absent' if case == 'no checkout' else tree, log,
                   root=None if case == 'no private root' else root,
                   first=failing(tmp_path / 'bin', 'mktemp') if case == 'mktemp fails' else None)
    assert result.returncode == status, result.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and lines[0].endswith(
        f' commit={commit if named else "?"} verdict=INCOMPLETE reason="{reason}" known=?/? new=? now_passing=?'
        ' output=?')
    assert result.stdout.strip() == lines[0]
    assert root.is_file() or list(root.iterdir()) == []


@pytest.mark.parametrize('summary,status,reason', [
    ('suite was interrupted', 1, 'the runner ended with status 1 and without one final verdict line'),
    (VERDICT, 2, "the runner's status 2 does not fit its verdict line"),
])
def test_a_run_that_ends_without_a_verdict_says_why(tmp_path, summary, status, reason):
    tree, commit = checkout(tmp_path / 'checkout', f"printf '%s\\n' '{summary}'\nexit {status}\n")
    root = tmp_path / 'private'
    root.mkdir()
    log = tmp_path / 'verdicts.log'
    result = daily(tree, log, root=root)
    assert result.returncode == 2, result.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and f' commit={commit} verdict=INCOMPLETE reason="{reason}" known=' in lines[0]
    assert Path(lines[0].split(' output=', 1)[1]).read_text().strip() == summary


def test_a_line_that_the_log_cannot_take_goes_to_standard_error(tmp_path):
    tree, commit = checkout(tmp_path / 'checkout', COMPLETE)
    root = tmp_path / 'private'
    root.mkdir()
    log = tmp_path / 'absent/verdicts.log'
    result = daily(tree, log, root=root)
    assert result.returncode == 1 and result.stdout == ''
    lines = result.stderr.splitlines()
    assert len(lines) == 1 and (f' commit={commit} verdict=PASS known=0/2 new=0 now_passing=2 '
                                f'output={root}/daily-suite.') in lines[0]
    refused = daily(tree, log)
    assert refused.returncode == 1 and refused.stdout == ''
    lines = refused.stderr.splitlines()
    assert len(lines) == 1 and lines[0].endswith(
        ' commit=? verdict=INCOMPLETE reason="SUITE_PRIVATE_ROOT is not set to a private output directory"'
        ' known=?/? new=? now_passing=? output=?')
    assert not log.parent.exists()


def test_output_older_than_the_three_newest_runs_is_removed_and_nothing_else(tmp_path):
    """With this run there are four: the fourth-newest run directory and log go. A foreign directory, names of
    another form or kind and symbolic links with the runner's and the wrapper's names stay, however old."""
    tree, _ = checkout(tmp_path / 'checkout', COMPLETE_IN_ITS_OWN_DIRECTORY)
    root = tmp_path / 'private'
    root.mkdir()
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    (elsewhere / 'kept.txt').write_text('kept')
    runs, logs = earlier_runs(root, (3, 2, 1))
    for name in ('archive', 'colonymind-suite.Run0009', 'daily-suite.ADir01.log'):
        (root / name).mkdir()
        (root / name / 'kept.txt').write_text('kept')
        aged(root / name, 9)
    for name in ('daily-suite.Run009.txt', 'colonymind-suite.AFile1'):
        (root / name).write_text('kept')
        aged(root / name, 9)
    aged(elsewhere, 9)
    for name, target in (('colonymind-suite.Linked', elsewhere), ('daily-suite.Linked.log', elsewhere / 'kept.txt')):
        (root / name).symlink_to(target)
        aged(root / name, 9)
    before = {entry.name for entry in root.iterdir()}
    log = tmp_path / 'verdicts.log'
    result = daily(tree, log, root=root)
    assert result.returncode == 0, result.stderr
    after = {entry.name for entry in root.iterdir()}
    assert before - after == {runs[0].name, logs[0].name}
    made = sorted(after - before)
    assert len(made) == 2 and (root / made[0]).is_dir() and (root / made[1]).is_file()
    line = log.read_text().splitlines()[0]
    assert ' verdict=PASS ' in line and 'not_removed=' not in line and line.endswith(f' output={root / made[1]}')
    assert all((entry / 'sidecar.log').read_text() == 'earlier' for entry in runs[1:])
    assert all(entry.read_text() == 'earlier' for entry in logs[1:])
    assert all((root / name / 'kept.txt').read_text() == 'kept'
               for name in ('archive', 'colonymind-suite.Run0009', 'daily-suite.ADir01.log'))
    assert (root / 'colonymind-suite.Linked').is_symlink() and (root / 'daily-suite.Linked.log').is_symlink()
    assert (elsewhere / 'kept.txt').read_text() == 'kept'


def needs_locale(name):
    """Skip the test, naming the locale, unless ``locale -a`` lists it (it writes ``UTF-8`` as ``utf8``)."""
    def spelled(text):
        return text.lower().replace('-', '')
    try:
        listed = subprocess.run(['locale', '-a'], capture_output=True, text=True, errors='replace').stdout.split()
    except OSError:
        listed = []
    if spelled(name) not in map(spelled, listed):
        pytest.skip(f'the locale {name} is not installed on this machine')


@pytest.mark.parametrize('variable', ['LANG', 'LC_ALL'])
def test_names_with_characters_outside_ascii_stay_under_a_language_locale(tmp_path, variable):
    """Under a language locale a range such as ``a-z`` also takes letters outside ASCII. Neither ``mktemp`` nor the
    runner makes a name with one, so such an entry is not their output: it stays with its content, however old,
    and own output older than the three newest runs still goes."""
    needs_locale('en_US.UTF-8')
    tree, _ = checkout(tmp_path / 'checkout', COMPLETE_IN_ITS_OWN_DIRECTORY)
    root = tmp_path / 'private'
    root.mkdir()
    runs, logs = earlier_runs(root, (3, 2, 1))
    suffixes = ('gardé1', 'abcßef', 'abc１ef')  # e acute, sharp s, a full-width digit one
    for suffix in suffixes:
        (root / f'colonymind-suite.{suffix}').mkdir()
        (root / f'colonymind-suite.{suffix}/kept.txt').write_text('kept')
        (root / f'daily-suite.{suffix}.log').write_text('kept')
        aged(root / f'colonymind-suite.{suffix}', 9)
        aged(root / f'daily-suite.{suffix}.log', 9)
    before = {entry.name for entry in root.iterdir()}
    changes = {name: None for name in os.environ if name == 'LANG' or name.startswith('LC_')}
    result = daily(tree, tmp_path / 'verdicts.log', root=root, changes={**changes, variable: 'en_US.UTF-8'})
    assert result.returncode == 0, result.stderr
    after = {entry.name for entry in root.iterdir()}
    assert before - after == {runs[0].name, logs[0].name}
    assert all((root / f'colonymind-suite.{suffix}/kept.txt').read_text() == 'kept' for suffix in suffixes)
    assert all((root / f'daily-suite.{suffix}.log').read_text() == 'kept' for suffix in suffixes)


def test_inherited_case_options_do_not_widen_what_is_removed(tmp_path):
    """Bash turns on the shell options that an exported ``BASHOPTS`` names. Entries whose names differ from the
    runner's and the wrapper's only in case are not their output: they stay with their content, however old, and
    own output older than the three newest runs still goes."""
    tree, _ = checkout(tmp_path / 'checkout', COMPLETE_IN_ITS_OWN_DIRECTORY)
    root = tmp_path / 'private'
    root.mkdir()
    runs, logs = earlier_runs(root, (3, 2, 1))
    (root / 'ColonyMind-Suite.Old050').mkdir()
    kept = [root / 'ColonyMind-Suite.Old050/kept.txt', root / 'Daily-Suite.Old050.log', root / 'daily-suite.Old050.LOG']
    for entry in kept:
        entry.write_text('kept')
    for entry in (kept[0].parent, *kept[1:]):
        aged(entry, 9)
    before = {entry.name for entry in root.iterdir()}
    result = daily(tree, tmp_path / 'verdicts.log', root=root, changes={'BASHOPTS': 'nocaseglob:nocasematch'})
    assert result.returncode == 0, result.stderr
    after = {entry.name for entry in root.iterdir()}
    assert before - after == {runs[0].name, logs[0].name}
    assert all(entry.read_text() == 'kept' for entry in kept)


@pytest.mark.parametrize('spelling', ['its path in the private root', 'a path through a link to the private root'])
def test_the_verdict_log_is_never_removed(tmp_path, spelling):
    """A verdict log that lies in the private root under a name of the wrapper's own form, with three newer
    output logs: it keeps its earlier lines and takes the new one, however its path is spelled. It is not
    counted as output either: the log older than the three newest runs still goes."""
    tree, commit = checkout(tmp_path / 'checkout', COMPLETE)
    root = tmp_path / 'private'
    root.mkdir()
    _, logs = earlier_runs(root, (3, 2, 1))
    log = root / 'daily-suite.result.log'
    log.write_text('an earlier line\nanother earlier line\n')
    aged(log, 9)
    (tmp_path / 'linked').symlink_to(root)
    named = log if spelling == 'its path in the private root' else tmp_path / 'linked' / log.name
    result = daily(tree, named, root=root)
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert lines[:2] == ['an earlier line', 'another earlier line'] and len(lines) == 3
    assert f' commit={commit} verdict=PASS known=0/2 new=0 now_passing=2 output={root}/daily-suite.' in lines[2]
    assert result.stdout.strip() == lines[2]
    assert not logs[0].exists() and all(entry.read_text() == 'earlier' for entry in logs[1:])


def test_the_run_in_progress_is_never_removed(tmp_path):
    """Earlier output dated after this run (a clock set back) makes the run's own the oldest by date. It stays;
    of the four earlier runs the one dated oldest goes."""
    tree, _ = checkout(tmp_path / 'checkout', COMPLETE_IN_ITS_OWN_DIRECTORY)
    root = tmp_path / 'private'
    root.mkdir()
    runs, logs = earlier_runs(root, (-1, -2, -3, -4))
    before = {entry.name for entry in root.iterdir()}
    log = tmp_path / 'verdicts.log'
    result = daily(tree, log, root=root)
    assert result.returncode == 0, result.stderr
    after = {entry.name for entry in root.iterdir()}
    assert before - after == {runs[0].name, logs[0].name}
    made = sorted(after - before)
    assert len(made) == 2 and (root / made[0]).is_dir() and (root / made[1]).is_file()
    assert log.read_text().splitlines()[0].endswith(f' output={root / made[1]}')


def test_a_removal_that_fails_is_named_in_the_line_and_keeps_the_verdict(tmp_path):
    tree, _ = checkout(tmp_path / 'checkout', COMPLETE_IN_ITS_OWN_DIRECTORY)
    root = tmp_path / 'private'
    root.mkdir()
    runs, logs = earlier_runs(root, (3, 2, 1))
    log = tmp_path / 'verdicts.log'
    result = daily(tree, log, root=root, first=failing(tmp_path / 'bin', 'rm'))
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and (' verdict=PASS known=0/2 new=0 now_passing=2 '
                                f'not_removed={runs[0].name},{logs[0].name} output=') in lines[0]
    assert runs[0].is_dir() and logs[0].is_file()


def test_a_run_stopped_by_sigterm_leaves_an_incomplete_line(tmp_path):
    """A service manager that stops the run signals each of its processes. The wrapper leaves its line and ends
    by the signal, as it did before it left a line."""
    tree, commit = checkout(tmp_path / 'checkout', 'touch "$SUITE_PRIVATE_ROOT/started"\nsleep 60\n')
    root = tmp_path / 'private'
    root.mkdir()
    log = tmp_path / 'verdicts.log'
    run = subprocess.Popen(['bash', str(SCRIPT), str(tree), str(log)], stdout=subprocess.PIPE, text=True,
                           env={**os.environ, 'SUITE_PRIVATE_ROOT': str(root)}, start_new_session=True)
    try:
        waited = time.monotonic() + 30
        while not (root / 'started').exists() and time.monotonic() < waited:
            time.sleep(0.05)
        assert (root / 'started').exists()
        os.killpg(run.pid, signal.SIGTERM)
        stdout, _ = run.communicate(timeout=30)
    finally:
        if run.poll() is None:
            os.killpg(run.pid, signal.SIGKILL)
            run.wait()
    assert run.returncode == -signal.SIGTERM
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and (f' commit={commit} verdict=INCOMPLETE reason="stopped by SIGTERM before the runner '
                                f'ended" known=?/? new=? now_passing=? output={root}/daily-suite.') in lines[0]
    assert stdout.strip() == lines[0]


def test_the_example_service_ends_a_run_after_three_hours():
    assert [line for line in SERVICE.read_text().splitlines() if line.startswith('TimeoutStartSec=')] == [
        'TimeoutStartSec=3h']
