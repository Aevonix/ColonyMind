"""Run or inspect the watcher without starting the sidecar."""
import json
import sqlite3
import sys

from colonymind.config import ConfigError, load_config, read_api_key


def add_parser(sub):
    parser = sub.add_parser('watch', help='Check jobs and dependencies in a separate, short-lived process')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='Run one pass (the default)')
    mode.add_argument('--list', action='store_true', help='Read open findings without running checks')
    parser.add_argument('--dry-run', action='store_true', help='Print findings without writes or notices')
    parser.add_argument('--since', help='Replay retained finished records since an ISO date')


def main(args):
    from colonymind.work.hermes_jobs import failed, timestamp
    from .runner import run
    from .store import read_status

    try:
        config = load_config(args.instance, strict_watch=True)
        if args.list:
            findings = read_status(config.home / 'jobwatch.db')
        else:
            since = timestamp(args.since) if args.since else None
            findings = run(config.get('watch'), config.home, config.sidecar_url,
                           key=read_api_key(config.home), since=since, dry_run=args.dry_run)
        print(json.dumps(findings, indent=2))
        return 0
    except (OSError, sqlite3.Error, ValueError, TypeError) as error:
        failed(error, 'jobwatch')
        detail = ': ' + str(error) if isinstance(error, ConfigError) and str(error).startswith('watch') else ''
        print('Job watch failed: ' + type(error).__name__ + detail, file=sys.stderr)
        return 1
