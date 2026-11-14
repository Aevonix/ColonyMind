"""Retired execution writers leave stable reads and lightweight CLI help."""
import subprocess
import sys

import pytest

from colonymind.api.routers import executions


def test_watch_parser_does_not_import_the_watcher_runtime():
    subprocess.run([sys.executable, '-c', """
import argparse
import sys
from colonymind.jobwatch.cli import add_parser
add_parser(argparse.ArgumentParser().add_subparsers())
assert 'colonymind.jobwatch.runner' not in sys.modules
assert 'colonymind.jobwatch.store' not in sys.modules
assert 'colonymind.work.hermes_jobs' not in sys.modules
"""], check=True)


@pytest.mark.parametrize('owner', [False, True])
async def test_execution_envelope_survives_registry_retirement(monkeypatch, owner):
    from colonymind.turns import board_observations, hermes_work, reported_workers

    def read(**kwargs):
        return {'items': [{'id': 'live-record'}], 'available': True}

    for module, name in ((board_observations, 'kanban_view'), (hermes_work, 'cron_view'),
                         (reported_workers, 'reported_worker_view')):
        monkeypatch.setattr(module, name, read)
    view = await executions.with_queue_work({}, owner=owner)
    assert view['schema'] == 'ColonyMindExecutionViewV1'
    assert view['items'] == [] and view['recent'] == [] and view['total'] == 0
    assert not view['truncated'] and not view['recent_truncated']
    assert not view['complete'] and not view['commitments_enforced']
    assert view['observed_at'] > 0 and view['coverage']
    for source in ('native_cron', 'native_kanban', 'reported_worker'):
        if owner:
            assert view[source]['items'] == [{'id': 'live-record'}]
        else:
            assert source not in view
