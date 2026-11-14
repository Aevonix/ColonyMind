#!/usr/bin/env python3
"""Run the staged voice gate offline; recorded embeddings remain outside git.

VOICE_RECALL_EMBEDDINGS=/path/to/voice-recall-embeddings.json python .../measure_voice_speculation.py
Optional VOICE_RECALL_VECTOR_REPORT=/path/to/report.json.
"""
import os
import tempfile
from pathlib import Path
import sys


def main():
    path = os.environ.get('VOICE_RECALL_EMBEDDINGS')
    if not path or not Path(path).is_file():
        print('SKIP: set VOICE_RECALL_EMBEDDINGS to the external embedding JSON')
        return 0
    os.environ.setdefault('TMPDIR', tempfile.mkdtemp(prefix='voice-speculation-'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['COLONYMIND_TEST_POLL_LOOP'] = '1'
    os.environ.pop('COLONYMIND_URL', None)
    os.environ.pop('COLONYMIND_API_KEY', None)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'sidecar'))
    import pytest
    return pytest.main(['-c', str(root / 'sidecar/pytest.ini'),
        str(root / 'sidecar/tests/test_voice_recall_t2.py') + '::test_real_embedding_t2_gate', '-q', '-s'])


if __name__ == '__main__':
    sys.exit(main())
