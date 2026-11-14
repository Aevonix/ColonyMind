"""A local model endpoint for tests: real HTTP/1.1 with keep-alive over real TCP, counting connections and requests.

``answer(path, body)`` returns ``(status, payload)``; the server sends it as JSON. It serves ``/v1/rerank`` and
``/v1/embeddings`` the way an OpenAI-compatible server does, only as the test says.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ModelEndpoint:
    def __init__(self, answer, *, delay: float = 0.0):
        self.connections = 0        # opened, ever
        self.open_connections = 0   # not yet closed by the client
        self.requests: list[tuple[str, dict]] = []
        self.delay = delay
        self.lock = threading.Lock()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                # As a production server does: a response is not held back waiting for the client's ACK.
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with endpoint.lock:
                    endpoint.connections += 1
                    endpoint.open_connections += 1

            def finish(self):
                try:
                    super().finish()
                finally:
                    with endpoint.lock:
                        endpoint.open_connections -= 1

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                endpoint.requests.append((self.path, body))
                if endpoint.delay:
                    time.sleep(endpoint.delay)
                status, payload = answer(self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def idle(self, within: float = 2.0) -> bool:
        """Whether every connection a client opened has been closed by it (waiting up to ``within`` seconds)."""
        deadline = time.monotonic() + within
        while self.open_connections and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.open_connections == 0

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def rerank_answer(path, body):
    """Scores every document; the first one highest."""
    documents = body.get("documents") or []
    return 200, {"results": [{"index": i, "relevance_score": 1.0 / (1 + i)} for i in range(len(documents))]}


def embedding_answer(path, body):
    """A two-dimensional unit vector per input."""
    inputs = body.get("input") or []
    return 200, {"model": "fixture-served", "data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))]}
