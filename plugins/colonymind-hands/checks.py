"""Acceptance checks: saved plan commands bound to final artifacts, never result prose.

The runner has the assistant's shell authority. It is not a sandbox. No sidecar
import is needed; receipts follow the fixed CheckReceipt wire shape.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
from urllib.parse import urlsplit
import wave

RUNNER = 'hands-checks-v1'
CHECK_TIMEOUT = 300  # Design assumption: seconds for a complete check batch.


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def plan_hash(acceptance: Mapping) -> str:
    return digest(json.dumps(dict(acceptance), sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())


def _file_hash(path: Path, deadline: float | None = None) -> str:
    if not path.is_file():
        raise ValueError('artifact is not a regular file')
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):  # Assumption: bounded read buffer.
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('artifact binding timed out')
            hasher.update(block)
    return hasher.hexdigest()


def _command(command, workdir, timeout):
    deadline = time.monotonic() + timeout
    if not command or not isinstance(command, (str, list)):
        raise ValueError('check command must come from the plan')
    if isinstance(command, list) and not all(isinstance(arg, str) for arg in command):
        raise ValueError('command arguments must be strings')
    if timeout <= 0:
        return None, digest(b''), '', True
    # A temporary file prevents verbose planned commands from exhausting RAM.
    with tempfile.TemporaryFile() as output:
        # Bytecode written by a planned Python test would change the bound tree
        # and turn every passing check on an unignored repository into a failure.
        process = subprocess.Popen(command, cwd=workdir, shell=isinstance(command, str),
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        timed_out = False
        try:
            process.wait(timeout=max(0, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        output.seek(0)
        hasher, sample = hashlib.sha256(), b''
        for block in iter(lambda: output.read(65536), b''):
            if time.monotonic() >= deadline:
                raise TimeoutError('check output hashing timed out')
            hasher.update(block)
            # Assumption: interpret at most 1 MiB for a named-state probe.
            if len(sample) < 1048576:
                sample += block[:1048576 - len(sample)]
        return process.returncode, hasher.hexdigest(), sample.decode('utf-8', 'replace'), timed_out


def git_binding(workdir: Path, *, timeout: float = CHECK_TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=workdir, stderr=subprocess.DEVNULL,
                                       timeout=max(.001, deadline-time.monotonic()))
    root = Path(os.fsdecode(git('rev-parse', '--show-toplevel')).strip())
    workdir = root  # A check in a subdirectory still binds the entire repository.
    head = git('rev-parse', 'HEAD').decode().strip()
    porcelain = git('status', '--porcelain', '--untracked-files=all')
    # Porcelain alone cannot distinguish two edits to an already dirty file.
    files = git('ls-files', '--full-name', '-c', '-o', '--exclude-standard', '-z').split(b'\0')
    contents = []
    for name in sorted(set(files) - {b''}):
        if time.monotonic() > deadline:
            raise TimeoutError('artifact binding timed out')
        path = root / os.fsdecode(name)
        if path.is_symlink():
            value = ['symlink', os.readlink(path)]
        elif path.is_file():
            value = [path.stat().st_mode, _file_hash(path, deadline)]
        elif path.is_dir():
            raise ValueError('nested working tree requires its own check')
        else:
            value = ['missing']
        contents.append([os.fsdecode(name), value])
    return {'reference': head, 'dirty_tree_sha256': digest(porcelain),
            'tree_sha256': digest(json.dumps(contents, sort_keys=True).encode())}


class CheckRunner:
    def __init__(self, *, timeout: float = CHECK_TIMEOUT):
        if timeout <= 0:
            raise ValueError('check timeout must be positive')
        self.timeout = min(timeout, CHECK_TIMEOUT)

    @staticmethod
    def failure(kind, acceptance, error):
        return {'scope': {'kind': kind, 'details': {
            'runner': RUNNER, 'plan_hash': plan_hash(acceptance), 'current': False,
            'artifact_validated': False, 'requirement_passed': False,
            'error': type(error).__name__, 'timed_out': isinstance(error, TimeoutError)}},
            'passed': False, 'exit_code': None, 'output_hash': None}

    def probe(self, *, kind, acceptance, workdir):
        """Observe unknown effects without retrying the effect or its acceptance test."""
        try:
            workdir = Path(workdir).resolve(strict=True)
            if kind in {'coding', 'integrate'}:
                command = ['git', 'status', '--porcelain']
                code, hashed, output, timeout = _command(command, workdir, self.timeout)
                head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workdir,
                                               timeout=self.timeout).decode().strip()
                output = 'HEAD=' + head + '\n' + output
            elif kind == 'ops':
                command = acceptance.get('probe')
                code, hashed, output, timeout = _command(command, workdir, self.timeout)
            else:
                raise ValueError('no reconciliation probe for task kind')
            return {'scope': {'kind': kind, 'probe': command, 'details': {
                'runner': RUNNER, 'plan_hash': plan_hash(acceptance), 'output': output,
                'timed_out': timeout}}, 'passed': code == 0 and not timeout,
                'exit_code': code, 'output_hash': digest(output.encode())}
        except Exception as error:
            return self.failure(kind, acceptance, error)

    def is_current(self, receipt: Mapping, workdir) -> bool:
        try:
            scope = receipt['scope']
            if scope['kind'] in {'coding', 'integrate'}:
                current = git_binding(Path(workdir), timeout=self.timeout)
                return (scope.get('reference') == current['reference']
                        and scope.get('dirty_tree_sha256') == current['dirty_tree_sha256']
                        and scope['details'].get('tree_sha256') == current['tree_sha256'])
            if scope.get('artifacts'):
                deadline = time.monotonic() + self.timeout
                return all(_file_hash(Path(a['reference']), deadline) == a['sha256'] for a in scope['artifacts'])
        except (KeyError, TypeError, OSError, ValueError, subprocess.SubprocessError):
            return False
        return False  # State probes and fetched snapshots are point-in-time receipts.

    def finalize(self, receipts, workdir) -> list[dict]:
        """Recheck code/files immediately before delivery; never revive a failed check.

        Research snapshots and ops probes attest the instant they were observed.
        The caller must run those at stop time, rather than replay an older probe.
        """
        final = deepcopy(list(receipts))
        for receipt in final:
            scope = receipt.get('scope', {})
            if scope.get('kind') in {'coding', 'integrate'} or scope.get('artifacts'):
                if not self.is_current(receipt, workdir):
                    receipt['passed'] = False
                    scope['details'].update(current=False, artifact_validated=False, requirement_passed=False)
        return final

    def run(self, *, kind: str, acceptance: Mapping, workdir, artifacts=(), fetched=None, quotes=()) -> list[dict]:
        """Inputs are executor observations and saved plan criteria.

        Fetched texts must come from this hand's observed fetches, never URLs
        supplied by its summary. Artifacts are observed output paths. Quotes
        may be proposed, but every byte is checked against those fetched texts.
        """
        plan = deepcopy(dict(acceptance))
        details = {'runner': RUNNER, 'plan_hash': plan_hash(plan), 'current': False,
                   'artifact_validated': False, 'requirement_passed': False}
        scope = {'kind': kind, 'details': details}
        receipt = {'scope': scope, 'passed': False, 'exit_code': None, 'output_hash': None}
        deadline = time.monotonic() + self.timeout
        try:
            # A stopped hand can have no saved probe or task workdir. There is
            # no check to run then, and resolving None used to invent TypeError
            # evidence for an ops hand waiting at its approval hold.
            if kind in {'coding', 'integrate', 'ops'} and not plan.get('probe' if kind == 'ops' else 'test_command'):
                return []
            if kind == 'research' and not quotes:
                return []
            if kind not in {'coding', 'integrate', 'ops', 'research'} and not artifacts:
                return []
            if workdir is None or not str(workdir).strip():
                raise ValueError('a planned check needs a workdir')
            workdir = Path(workdir).resolve(strict=True)
            if kind in {'coding', 'integrate', 'ops'}:
                field = 'probe' if kind == 'ops' else 'test_command'
                command = plan.get(field)
                if not command:
                    return []
                scope['probe' if kind == 'ops' else 'command'] = command
                before = None if kind == 'ops' else git_binding(workdir, timeout=self.timeout)
                if before:
                    scope.update({key: before[key] for key in ('reference', 'dirty_tree_sha256')})
                    details['tree_sha256'] = before['tree_sha256']
                code, output_hash, output, timed_out = _command(command, workdir, deadline-time.monotonic())
                receipt.update(exit_code=code, output_hash=output_hash)
                details['timed_out'] = timed_out
                if kind == 'ops':
                    scope.update(reference='probe:' + output_hash, sha256=output_hash,
                                 criterion='; '.join(plan.get('requirements') or []))
                    # Assumption: named-state requirements are exact output
                    # lines such as service=ready, not substring matches.
                    requirements = plan.get('requirements') or []
                    details['current'] = True
                    passed = bool(requirements) and all(line in output.splitlines() for line in requirements)
                    details['matched_requirements'] = [line for line in requirements if line in output.splitlines()]
                    details['final_binding'] = {'sha256': output_hash}
                else:
                    after = git_binding(workdir, timeout=max(.001, deadline-time.monotonic()))
                    details['current'] = before == after
                    details['final_binding'] = after
                    passed = before == after
                passed = passed and code == 0 and not timed_out and time.monotonic() <= deadline
                details.update(artifact_validated=passed, requirement_passed=passed)
                receipt['passed'] = passed
            elif kind == 'research':
                if not quotes:
                    return []  # Nothing proposed: no check ran, and no failure was observed.
                fetched = dict(fetched or {})
                checked, valid = [], bool(quotes)
                for quote in quotes:
                    url, words = quote.get('url'), quote.get('quote')
                    text = fetched.get(url)
                    if not isinstance(text, str) or not isinstance(words, str) or not words or words not in text:
                        valid = False
                        continue
                    start = text.index(words)
                    checked.append({'url': url, 'content_hash': digest(text.encode()), 'quote': words,
                                    'start': start, 'end': start + len(words)})
                scope['quotes'] = checked
                receipt['output_hash'] = plan_hash(fetched)
                details.update(current=True, artifact_validated=valid)
                details['final_binding'] = {q['url']: q['content_hash'] for q in checked}
                required = plan.get('required_sources') or 2  # Design default.
                urls = {q['url'] for q in checked}
                # Assumption: distinct hostnames are a structural source rule,
                # not proof of editorial independence or truth.
                hosts = {(urlsplit(url).hostname or '').removeprefix('www.') for url in urls}
                independent = plan.get('independent_sources', True)
                enough = len(hosts - {''}) >= required if independent else len(urls) >= required
                details['requirement_passed'] = bool(valid and enough)
                scope['criterion'] = f'{required} fetched sources with exact quotes'
                receipt['passed'] = valid
            elif artifacts:
                output_type = plan.get('output_type')
                validated = []
                scope['artifacts'] = []
                for raw in artifacts:
                    path = Path(raw)
                    path = (workdir / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
                    before = _file_hash(path, deadline)
                    scope['artifacts'].append({'kind': output_type or 'unknown', 'reference': str(path), 'sha256': before})
                    validated.append(self._validate(path, output_type, deadline))
                    if before != _file_hash(path, deadline):
                        raise ValueError('artifact changed during validation')
                receipt['output_hash'] = plan_hash({'validated': validated})
                receipt['passed'] = True
                details.update(current=True, artifact_validated=True,
                               requirement_passed=kind == 'media' and bool(output_type), validation=validated)
                details['final_binding'] = deepcopy(scope['artifacts'])
                scope['criterion'] = f'output decodes as {output_type}'
            else:
                return []
            if time.monotonic() > deadline:
                raise TimeoutError('check timed out')
        except Exception as error:
            details.update(current=False, artifact_validated=False, requirement_passed=False,
                           error=type(error).__name__)
            if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
                details['timed_out'] = True
            receipt['passed'] = False
        return [receipt]

    def _validate(self, path, output_type, deadline):
        # Parsing and decompression run out of process so the batch deadline also
        # bounds corrupt media and unexpectedly costly parser work.
        if output_type in {'image', 'pdf', 'json', 'python', 'text'} or (output_type == 'audio' and path.suffix.lower() == '.wav'):
            code, _, output, timed_out = _command(
                [sys.executable, str(Path(__file__).resolve()), '--validate', str(path), output_type],
                path.parent, deadline-time.monotonic())
            if timed_out:
                raise TimeoutError('artifact validation timed out')
            if code:
                raise ValueError('artifact failed to decode')
            return json.loads(output)
        if output_type in {'audio', 'video'}:
            stream = 'a' if output_type == 'audio' else 'v'
            code, _, _, timed_out = _command(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                '-map', f'0:{stream}:0', '-f', 'null', '-'], path.parent, deadline-time.monotonic())
            if timed_out:
                raise TimeoutError('media decoding timed out')
            if code:
                raise ValueError('media failed to decode')
            code, _, output, timed_out = _command(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'json', str(path)], path.parent, deadline-time.monotonic())
            if timed_out:
                raise TimeoutError('media duration probe timed out')
            duration = float(json.loads(output)['format']['duration'])
            if code or not math.isfinite(duration) or duration <= 0:
                raise ValueError('media has no duration')
            return {'duration': duration}
        raise ValueError('unsupported artifact type')


def _decode(path: Path, output_type: str) -> dict:
    """Fixed local parser worker, invoked only by the runner."""
    if output_type == 'image':
        from PIL import Image
        with Image.open(path) as image:
            image.load()
            if min(image.size) <= 0:
                raise ValueError('image has no dimensions')
            return {'dimensions': list(image.size)}
    if output_type == 'pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        if not reader.pages:
            raise ValueError('PDF has no pages')
        for page in reader.pages:
            page.extract_text()
        return {'pages': len(reader.pages)}
    if output_type == 'audio' and path.suffix.lower() == '.wav':
        with wave.open(str(path), 'rb') as audio:
            count, rate = audio.getnframes(), audio.getframerate()
            if count <= 0 or rate <= 0:
                raise ValueError('audio has no duration')
            data = audio.readframes(count)
            if len(data) != count * audio.getsampwidth() * audio.getnchannels():
                raise ValueError('audio is truncated')
            return {'duration': count / rate}
    if output_type == 'json':
        json.loads(path.read_text())
    elif output_type == 'python':
        ast.parse(path.read_text())
    elif output_type == 'text':
        path.read_text(encoding='utf-8')
    else:
        raise ValueError('unsupported artifact type')
    return {'type': output_type}


def register(ctx: Any) -> None:
    """Runtime.subagent_stop owns runner ordering while Hermes retains its slot."""


if __name__ == '__main__' and len(sys.argv) == 4 and sys.argv[1] == '--validate':
    print(json.dumps(_decode(Path(sys.argv[2]), sys.argv[3])))
