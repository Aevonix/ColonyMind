"""Integrator socket checks: COLONYMIND_TEST_HANDS_SOCKETS=1 PYTHONPATH=sidecar.

This exercises a real local HTTP fetch, checker receipts and sidecar evaluation.
It does not claim to exercise a live Hermes stop hook, whose receipt integration
remains owned by the dispatch integrator.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import os
import threading
from urllib.request import urlopen

import pytest

from hands_fake_hermes import load_hands

pytestmark = pytest.mark.skipif(os.environ.get('COLONYMIND_TEST_HANDS_SOCKETS') != '1',
                                reason='Integrator: opt in to real loopback socket checks')


@pytest.fixture
def source():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = b'Sam and Robin checked the fixture for Owner.'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/source'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('quote,expected', [('Sam and Robin checked the fixture', 'E3'), ('Invented quotation', 'E1')])
@pytest.mark.requires_sidecar
def test_real_fetch_quote_receipt_and_evaluation(source, tmp_path, quote, expected):
    from colonymind.hands.evidence import assess

    with urlopen(source, timeout=2) as response:
        payload = response.read()
    observed_hash = hashlib.sha256(payload).hexdigest()
    acceptance = {'required_sources': 1, 'independent_sources': False, 'quotes_required': True}
    runner = importlib.import_module(load_hands().__name__ + '.checks').CheckRunner()
    receipts = runner.run(kind='research', acceptance=acceptance, workdir=tmp_path,
                          fetched={source: payload.decode()}, quotes=[{'url': source, 'quote': quote}])
    evidence = assess(kind='research', acceptance=acceptance, receipts=receipts,
                      tool_digest=[{'tool': 'web_fetch', 'url': source, 'content_hash': observed_hash}])
    assert evidence.level == expected
    if expected == 'E3':
        assert receipts[0]['scope']['quotes'][0]['content_hash'] == observed_hash
    else:
        assert receipts[0]['passed'] is False
