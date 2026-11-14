"""``colonymind console``: the owner console on loopback (docs/CONSOLE.md)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("console", help="The owner console: read-only local web pages (127.0.0.1 only) with the "
                                            "mind's off switch")
    parser.add_argument("--port", type=int, default=0,
                        help="The port on 127.0.0.1 (default: a free one); the address is always loopback")
    parser.add_argument("--bench", default=None,
                        help="A paired benchmark results directory: the bench report, and the evidence beside each "
                             "faculty")


def run(args: argparse.Namespace) -> int:
    from colonymind.config import ConfigError

    from .server import ConsoleServer
    from .source import Source
    if not 0 <= args.port <= 65535:
        print("colonymind console: --port takes 0 to 65535", file=sys.stderr)
        return 2
    if args.bench and not Path(args.bench).expanduser().is_dir():
        print(f"colonymind console: no results directory at {args.bench}", file=sys.stderr)
        return 2
    try:
        source = Source(getattr(args, "instance", None), bench=args.bench)
        server = ConsoleServer(source, port=args.port)
    except ConfigError as error:
        print(f"colonymind console: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"colonymind console: cannot listen on 127.0.0.1:{args.port} ({error.strerror or type(error).__name__})",
              file=sys.stderr)
        return 1
    print(f"Owner console for {source.home}: read-only, with one control, the mind's off switch.\n"
          "Open this link in a browser on this machine. It works once and opens one session:\n"
          f"  {server.url}\n"
          "Ctrl-C stops the console.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = ["add_parser", "run"]
