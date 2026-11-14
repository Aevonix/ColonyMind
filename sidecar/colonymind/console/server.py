"""The owner console's HTTP server: 127.0.0.1 only, on the port given (a free one by default).

A small threaded server from the standard library. The address is fixed: nothing sets another. It reads a request
body only up to ``app.MAX_BODY`` (a longer one is answered 413 and the connection closed), logs no request line
(the link's one-time token rides in one), and answers every refusal it makes itself (a malformed request, an
unknown method) with the console's own headers and page policy.
"""
from __future__ import annotations

import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .app import HOST, MAX_BODY, Console, Response, refusal, refused
from .source import Source

#: How much of an over-long body is read and dropped before the 413, so the client reads the answer (a builder's
#: choice: 1 MiB, far past any form and small enough to read at once).
_DRAIN = 1 << 20


class _Handler(BaseHTTPRequestHandler):
    server_version = "colonymind-console"
    sys_version = ""
    #: Seconds a connection may sit idle mid-request (a builder's choice: a browser sends a request at once; a client
    #: that holds a connection open without one does not hold a thread for long).
    timeout = 60

    def do_GET(self) -> None:  # noqa: N802 (the standard library's name)
        self._serve("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._serve("POST")

    def _serve(self, method: str) -> None:
        lengths = self.headers.get_all("Content-Length") or []
        body = b""
        if lengths:
            if len(lengths) != 1 or not lengths[0].strip().isdigit():
                self._write(refusal(400, "Bad request", "The request's length is not readable."), close=True)
                return
            size = int(lengths[0])
            if size > MAX_BODY:
                self.rfile.read(min(size, _DRAIN))
                self._write(refusal(413, "Too large", "The console takes only short forms."), close=True)
                return
            body = self.rfile.read(size)
        pairs = [(name, value) for name, value in self.headers.items()]
        try:
            response = self.server.console.handle(method, self.path, pairs, body)
        except Exception:               # never a traceback on the page, never a dropped connection
            response = refused(500)
        self._write(response)

    def _write(self, response: Response, *, close: bool = False) -> None:
        self.send_response(response.status)
        for name, value in response.all_headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(response.body)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response.body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self._write(refused(code), close=True)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (the standard library's name)
        """Nothing: a request line may carry the one-time token."""


class ConsoleServer(ThreadingHTTPServer):
    """The console, listening on 127.0.0.1 and nowhere else. ``url`` is the one-time link to print."""

    daemon_threads = True

    def __init__(self, source: Source, *, port: int = 0):
        super().__init__((HOST, int(port)), _Handler)
        self.console = Console(source, port=self.server_address[1])

    def server_bind(self) -> None:
        # The address is a literal: no name lookup (the standard server asks the resolver for its own name).
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = HOST, self.server_address[1]

    @property
    def url(self) -> str:
        return self.console.url


__all__ = ["ConsoleServer"]
