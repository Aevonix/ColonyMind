"""Offline upgrade gates for the retired global launchd registration."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import plistlib
import socket
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from colonymind import __version__, cli, init
from colonymind.services import instance as services


class Launchd:
    """Only exact user-domain jobs; no native service command can escape the fake."""

    def __init__(self):
        self.loaded = set()
        self.calls = []
        self.fail = None

    def __call__(self, args, **kwargs):
        assert args[0] == 'launchctl'
        self.calls.append(tuple(args))
        command, target = args[1:3]
        code, output = 0, ''
        if command == self.fail:
            code = 1
        elif command == 'print':
            if target.count('/') == 2:
                code = 0 if target in self.loaded else 113
                output = '  pid = 4242\n' if code == 0 else ''
        elif command == 'bootstrap':
            self.loaded.add(target + '/' + plistlib.loads(Path(args[3]).read_bytes())['Label'])
        elif command == 'bootout':
            self.loaded.discard(target)
        elif command == 'kickstart':
            assert target == '-k'
            self.loaded.add(args[3])
        else:
            assert command == 'enable', args
        return subprocess.CompletedProcess(args, code, output, '')


def snapshot(root):
    """Include every file's bytes and nanosecond mtime, plus symlink identities."""
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns,
                                      os.readlink(p) if p.is_symlink() else None)
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def upgraded(tmp_path, monkeypatch):
    from colonymind.config import DEFAULTS, save_config, write_api_key
    home, hermes_home = tmp_path / 'colonymind', tmp_path / 'hermes'
    home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setenv('COLONYMIND_HOME', str(home))
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(home))
    monkeypatch.setenv('HERMES_HOME', str(hermes_home))
    monkeypatch.delenv('COLONYMIND_API_KEY', raising=False)
    data = copy.deepcopy(DEFAULTS)
    data['hermes'].update(home=str(hermes_home), python=sys.executable)
    save_config(data, home)
    write_api_key('upgrade-test-key', home)
    monkeypatch.setattr(init, 'resolve_hermes_python', lambda *a: (Path(sys.executable), '0.21.3'))
    monkeypatch.setattr(init, 'installed_adapter_version', lambda python: '0.0.0')
    monkeypatch.setattr(init, 'install_adapter', lambda *a, **kw: True)
    monkeypatch.setattr(init, 'pip_check', lambda python: (True, ''))
    monkeypatch.setattr(init, 'colonymind_in_hermes_environment', lambda python: False)
    real_service_status = init.service_status
    monkeypatch.setattr(init, 'service_status', lambda cfg: None)
    monkeypatch.setattr(socket, 'create_connection',
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionRefusedError()))
    instance = SimpleNamespace(home=home, hermes_home=hermes_home, args=SimpleNamespace(
        home=str(home), hermes_home=None, hermes_python=None, adapter_source=None))
    # Bring the unrelated store/config upgrades current before staging the old service.
    monkeypatch.setattr(init, 'diagnostic_skills', lambda: ([], []))
    assert init.run_upgrade(instance.args) == 0
    monkeypatch.setattr(init, 'installed_adapter_version', lambda python: __version__)
    monkeypatch.setattr(init, 'sys', SimpleNamespace(platform='darwin'))
    manager = Launchd()
    service = services.InstanceService(instance.home, instance.hermes_home, home=tmp_path / 'user',
                                       platform='darwin', runner=manager)
    monkeypatch.setattr(init, '_service', lambda cfg: service)
    # From here on, status and stop/start decisions must exercise fake launchd.
    monkeypatch.setattr(init, 'service_status', real_service_status)
    monkeypatch.setattr(service, 'health', lambda: {'status': 'ok', 'problems': []})
    assert init.run_upgrade(instance.args) == 0
    return SimpleNamespace(instance=instance, service=service, manager=manager, root=tmp_path)


def old_global(upgraded, *, loaded=True, state=None):
    service, manager = upgraded.service, upgraded.manager
    path = service.link.parent / 'ai.aevonix.colonymind-sidecar.plist'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps({'Label': path.stem, 'EnvironmentVariables': {
        'COLONYMIND_STATE_DIR': str(state or service.state)}}))
    target = f'gui/{os.getuid()}/{path.stem}'
    if loaded:
        manager.loaded.add(target)
    return path, target


@pytest.mark.parametrize('registration', ['symlink', 'plain', 'retained-id'])
def test_upgrade_twice_preserves_existing_instance_plist(upgraded, registration):
    """Required fake-root gate: both labels, one instance survives byte/mtime identical."""
    service, manager = upgraded.service, upgraded.manager
    path, old_target = old_global(upgraded)
    content = service.render()
    if registration == 'symlink':
        service.install()
        existing = service.link
    else:
        existing = service.link
        if registration == 'retained-id':
            existing = existing.with_name('ai.colonymind.instance.retained.plist')
            payload = plistlib.loads(content)
            payload['Label'] = existing.stem
            content = plistlib.dumps(payload)
        existing.write_bytes(content)
    target = f'gui/{os.getuid()}/{existing.stem}'
    manager.loaded.add(target)
    before = snapshot(service.home)
    instance_before = snapshot(service.state)
    manager.calls.clear()

    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and old_target not in manager.loaded
    assert list(path.parent.glob('ai.colonymind.instance.*.plist')) == [existing]
    assert existing.read_bytes() == content
    assert snapshot(service.home) == {k: v for k, v in before.items() if k != str(path.relative_to(service.home))}
    assert snapshot(service.state) == instance_before
    assert target in manager.loaded
    # The only call that may name the existing registration is the read-only status check;
    # it is never booted out, bootstrapped, kickstarted or rewritten.
    assert all(call[1] == 'print' or call[2] not in {target, str(existing)} for call in manager.calls)
    assert all(str(existing) not in call for call in manager.calls)
    after, jobs = snapshot(upgraded.root), set(manager.loaded)
    manager.calls.clear()
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert snapshot(upgraded.root) == after and manager.loaded == jobs
    assert manager.calls == []


@pytest.mark.parametrize('loaded', [True, False])
@pytest.mark.parametrize('other_upgrade', [True, False])
def test_global_only_upgrade_installs_one_instance_and_repeats_without_changes(upgraded, loaded, other_upgrade):
    service, manager = upgraded.service, upgraded.manager
    path, old_target = old_global(upgraded, loaded=loaded)
    if other_upgrade:
        upgraded.instance.args.adapter_source = 'fixture-source'
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and old_target not in manager.loaded
    assert list(path.parent.glob('ai.colonymind.instance.*.plist')) == [service.link]
    assert service.link.resolve() == service.definition
    assert (service.target in manager.loaded) == loaded
    upgraded.instance.args.adapter_source = None
    after = snapshot(upgraded.root)
    manager.calls.clear()
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert snapshot(upgraded.root) == after
    assert manager.calls == []


def test_failed_install_keeps_the_old_service_running_and_retryable(upgraded):
    path, target = old_global(upgraded)
    before = path.read_bytes(), path.stat().st_mtime_ns
    upgraded.manager.fail = 'enable'
    assert init.run_upgrade(upgraded.instance.args) == 1
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert target in upgraded.manager.loaded
    assert not upgraded.service.link.exists()
    upgraded.manager.fail = None
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and target not in upgraded.manager.loaded
    assert upgraded.service.target in upgraded.manager.loaded


def test_failed_bootout_retains_the_global_plist_and_the_existing_instance(upgraded):
    path, target = old_global(upgraded)
    upgraded.service.install()
    before = snapshot(upgraded.root)
    upgraded.manager.fail = 'bootout'
    assert init.run_upgrade(upgraded.instance.args) == 1
    assert snapshot(upgraded.root) == before and target in upgraded.manager.loaded
    upgraded.manager.fail = None
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and target not in upgraded.manager.loaded


def test_a_global_service_for_another_instance_is_left_alone(upgraded):
    old_global(upgraded, state=upgraded.root / 'other-instance')
    before = snapshot(upgraded.root)
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert snapshot(upgraded.root) == before
    assert upgraded.manager.calls == []


def test_an_unidentifiable_global_plist_is_retained(upgraded):
    path, _ = old_global(upgraded)
    path.write_bytes(plistlib.dumps({'Label': path.stem}))
    before = snapshot(upgraded.root)
    assert init.run_upgrade(upgraded.instance.args) == 1
    assert snapshot(upgraded.root) == before
    assert upgraded.manager.calls == []


@pytest.mark.parametrize('failure', ['unlink', 'timeout', 'interrupt'])
@pytest.mark.parametrize('other_upgrade', [False, True])
def test_interrupted_conversion_recovers_and_retry_starts_the_installed_service(
        upgraded, monkeypatch, capsys, failure, other_upgrade):
    """Reviewer probe F1, with assertions for recovery before exit and on retry."""
    service, manager = upgraded.service, upgraded.manager
    path, target = old_global(upgraded)
    if other_upgrade:
        upgraded.instance.args.adapter_source = 'fixture-source'
    real_unlink, real_run = Path.unlink, service._run
    booted_out, fault_used, now = False, False, 0

    def unlink(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError('denied')
        return real_unlink(candidate, *args, **kwargs)

    def run(*args, **kwargs):
        nonlocal booted_out, fault_used, now
        result = real_run(*args, **kwargs)
        if args[1:3] == ('bootout', target):
            booted_out = True
        if args[1:3] == ('print', target) and booted_out and not fault_used and failure != 'unlink':
            fault_used = True
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            # Bootout finished, but its first status response is stale past the deadline.
            now = 31
            return subprocess.CompletedProcess(args, 0, '  pid = 4242\n', '')
        return result

    monkeypatch.setattr(service, '_run', run)
    monkeypatch.setattr(services, 'time', SimpleNamespace(monotonic=lambda: now, sleep=lambda _: None))
    if failure == 'unlink':
        monkeypatch.setattr(Path, 'unlink', unlink)
    if failure == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            init.run_upgrade(upgraded.instance.args)
    else:
        assert init.run_upgrade(upgraded.instance.args) == 1
    assert target not in manager.loaded
    assert service.target in manager.loaded  # Recovery happens before returning/raising.
    assert service.link.is_symlink() and path.exists()
    output = capsys.readouterr().out
    if failure == 'unlink':
        assert str(path) in output and 'can still start at login' in output
        assert f'launchctl bootout {target}; rm -- {shlex.quote(str(path))}' in output

    monkeypatch.setattr(Path, 'unlink', real_unlink)
    # Simulate an unloaded replacement before retry: installation alone is insufficient.
    manager.loaded.discard(service.target)
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and target not in manager.loaded
    assert service.target in manager.loaded
    upgraded.instance.args.adapter_source = None
    before = snapshot(upgraded.root)
    manager.calls.clear()
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert snapshot(upgraded.root) == before and manager.calls == []


@pytest.mark.parametrize('registration', ['symlink', 'plain', 'retained-id'])
def test_loaded_global_starts_an_existing_unloaded_instance(upgraded, registration):
    """Reviewer probe F2: hand serving to an existing registration without rewriting it."""
    service, manager = upgraded.service, upgraded.manager
    path, target = old_global(upgraded)
    if registration == 'symlink':
        service.install()
        existing = service.link
    else:
        existing = service.link
        payload = plistlib.loads(service.render())
        if registration == 'retained-id':
            existing = existing.with_name('ai.colonymind.instance.retained.plist')
            payload['Label'] = existing.stem
        existing.write_bytes(plistlib.dumps(payload))
    before = existing.read_bytes(), existing.stat().st_mtime_ns
    manager.calls.clear()
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and target not in manager.loaded
    assert f'gui/{os.getuid()}/{existing.stem}' in manager.loaded
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == before
    assert ('launchctl', 'bootstrap', f'gui/{os.getuid()}', str(existing)) in manager.calls
    assert not any(call[1] in {'enable', 'bootout'} and call[2] != target for call in manager.calls)
    manager.calls.clear()
    after = snapshot(upgraded.root)
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert snapshot(upgraded.root) == after and manager.calls == []


def test_old_default_data_directory_is_this_instance(upgraded, capsys):
    path, target = old_global(upgraded, state=upgraded.service.state / 'data')
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and target not in upgraded.manager.loaded
    assert upgraded.service.target in upgraded.manager.loaded
    assert 'old global service converted' in capsys.readouterr().out


def test_timeout_with_global_still_loaded_restarts_it(upgraded, monkeypatch, capsys):
    path, target = old_global(upgraded)
    service, manager = upgraded.service, upgraded.manager
    real_run, now = service._run, 0

    def run(*args, **kwargs):
        nonlocal now
        result = real_run(*args, **kwargs)
        if args[1:3] == ('bootout', target):
            manager.loaded.add(target)  # The manager has not yet finished bootout.
        elif args[1:3] == ('print', target) and any(call[1] == 'bootout' for call in manager.calls):
            now = 31
        return result

    monkeypatch.setattr(service, '_run', run)
    monkeypatch.setattr(services, 'time', SimpleNamespace(monotonic=lambda: now, sleep=lambda _: None))
    assert init.run_upgrade(upgraded.instance.args) == 1
    assert path.exists() and target in manager.loaded
    assert service.target not in manager.loaded
    assert ('launchctl', 'kickstart', '-k', target) in manager.calls
    assert 'old global service restarted' in capsys.readouterr().out
    monkeypatch.setattr(service, '_run', real_run)
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert service.target in manager.loaded and target not in manager.loaded


@pytest.mark.parametrize('delete_fails', [False, True])
def test_replacement_start_failure_restores_the_global_service(upgraded, monkeypatch, capsys, delete_fails):
    service, manager = upgraded.service, upgraded.manager
    path, target = old_global(upgraded)
    original = path.read_bytes()
    real_run, real_unlink = service._run, Path.unlink

    def run(*args, **kwargs):
        if args[1] == 'bootstrap' and args[3] == str(service.link):
            raise services.ServiceError('replacement bootstrap failed')
        return real_run(*args, **kwargs)

    def unlink(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError('denied')
        return real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(service, '_run', run)
    if delete_fails:
        monkeypatch.setattr(Path, 'unlink', unlink)
    assert init.run_upgrade(upgraded.instance.args) == (1 if delete_fails else 0)
    assert path.read_bytes() == original and target in manager.loaded
    assert service.target not in manager.loaded
    assert 'old global service restarted' in capsys.readouterr().out
    monkeypatch.setattr(service, '_run', real_run)
    monkeypatch.setattr(Path, 'unlink', real_unlink)
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert not path.exists() and service.target in manager.loaded


def test_foreign_global_has_an_exact_removal_command(upgraded, capsys):
    state = upgraded.root / 'another instance'
    path, target = old_global(upgraded, state=state)
    before = snapshot(upgraded.root)
    assert init.run_upgrade(upgraded.instance.args) == 0
    output = capsys.readouterr().out
    assert str(path) in output and str(state) in output
    assert f'launchctl bootout {target}; rm -- {shlex.quote(str(path))}' in output
    assert snapshot(upgraded.root) == before and upgraded.manager.calls == []


@pytest.mark.parametrize('unreadable', ['dangling', 'permission', 'not-dict', 'corrupt'])
def test_unreadable_global_is_unchanged_and_names_a_remedy(upgraded, monkeypatch, capsys, unreadable):
    path, target = old_global(upgraded)
    if unreadable == 'dangling':
        path.unlink()
        path.symlink_to(upgraded.root / 'missing')
    elif unreadable == 'not-dict':
        path.write_bytes(plistlib.dumps(['not a dictionary']))
    elif unreadable == 'corrupt':
        path.write_bytes(b'not a plist')
    before, metadata = snapshot(upgraded.root), path.lstat()
    real_read = Path.read_bytes

    def read(candidate):
        if candidate == path:
            raise PermissionError('denied')
        return real_read(candidate)

    if unreadable == 'permission':
        monkeypatch.setattr(Path, 'read_bytes', read)
    assert init.run_upgrade(upgraded.instance.args) == 1
    output = capsys.readouterr().out
    assert str(path) in output and 'Repair this plist' in output
    assert f'launchctl bootout {target}; rm -- {shlex.quote(str(path))}' in output
    monkeypatch.setattr(Path, 'read_bytes', real_read)
    # Reading a symlink may update atime on a relatime filesystem. Its identity,
    # mode, target and modification time must still be unchanged.
    after = path.lstat()
    assert (after.st_mode, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
        metadata.st_mode, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
    assert snapshot(upgraded.root) == before
    assert upgraded.manager.calls == []


@pytest.mark.parametrize('label,variable', [
    ('ai.aevonix.colony-sidecar', 'COLONY_STATE_DIR'),
    ('ai.aevonix.earlier-sidecar', 'EARLIER_STATE_DIR'),
])
def test_earlier_era_global_labels_are_not_converted(upgraded, label, variable):
    path = upgraded.service.link.parent / f'{label}.plist'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps({'Label': label, 'EnvironmentVariables': {
        variable: str(upgraded.service.state)}}))
    before = snapshot(upgraded.root)
    assert init.run_upgrade(upgraded.instance.args) == 0
    assert snapshot(upgraded.root) == before and upgraded.manager.calls == []


@pytest.mark.parametrize('action', ['install', 'uninstall', 'start', 'stop', 'restart', 'status'])
def test_cli_service_commands_always_use_the_instance_manager(monkeypatch, action):
    called = []
    monkeypatch.setattr(cli.sys, 'argv', ['colonymind', 'service', action])
    monkeypatch.setattr(services, 'manage', called.append)
    monkeypatch.setattr(cli, '_configured_instance', lambda: False)
    cli.main()
    assert called == [action]
