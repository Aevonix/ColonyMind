"""A test that needs the checkout to be a git repository skips with the missing prerequisite named."""
from pathlib import Path
import shutil
import subprocess

import pytest


def require_history(root, revision):
    root = Path(root)
    if not (root / '.git').exists():
        pytest.skip(f'checkout has no .git history; requires commit {revision}')
    if shutil.which('git') is None:
        pytest.skip(f'git executable is missing; requires checkout commit {revision}')
    result = subprocess.run(['git', 'cat-file', '-e', f'{revision}^{{commit}}'],
                            cwd=root, capture_output=True, timeout=30)
    if result.returncode:
        pytest.skip(f'checkout history is missing commit {revision}')
