"""Integrator: planned state probes over an isolated real HTTP socket.

Run with COLONYMIND_TEST_HANDS_SOCKETS=1 outside the restricted sandbox.
"""
import importlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import sys
import threading

import pytest

from hands_fake_hermes import load_hands


pytestmark = pytest.mark.skipif(
    os.environ.get('COLONYMIND_TEST_HANDS_SOCKETS') != '1',
    reason='Integrator: opt in to real loopback socket checks',
)


def test_planned_probe_checks_named_state_over_real_socket(tmp_path):
    state = ['service=starting']

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(state[0].encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        checks = importlib.import_module(load_hands().__name__ + '.checks')
        endpoint = 'http://127.0.0.1:%d/state' % server.server_address[1]
        command = [sys.executable, '-c',
                   'import sys, urllib.request; '
                   'print(urllib.request.urlopen(sys.argv[1]).read().decode())', endpoint]
        plan = {'probe': command, 'requirements': ['service=ready']}
        runner = checks.CheckRunner()
        starting, = runner.run(kind='ops', acceptance=plan, workdir=tmp_path)
        assert starting['exit_code'] == 0 and starting['passed'] is False
        state[0] = 'service=ready'
        ready, = runner.run(kind='ops', acceptance=plan, workdir=tmp_path)
        assert ready['passed'] is True
        assert ready['scope']['probe'] == command
        assert ready['scope']['sha256'] == ready['output_hash']
        assert ready['output_hash'] != starting['output_hash']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
