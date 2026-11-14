"""The evidence chart (``scripts/render_evidence_chart.py``): accuracy, evidence reached and trap rates against the
memory budget, per arm with the stock Hermes arm, as an SVG generated from a results file and nothing else, so the
committed chart is always the committed results drawn the same way."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ElementTree

import pytest

from test_public_memory_export import build, export, records  # noqa: F401  (pytest fixtures)

ROOT = Path(__file__).resolve().parents[2]
SVG = '{http://www.w3.org/2000/svg}'


@pytest.fixture(scope='module')
def chart():
    spec = importlib.util.spec_from_file_location('render_evidence_chart', ROOT / 'scripts' / 'render_evidence_chart.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['render_evidence_chart'] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def results(export, records, tmp_path):
    return build(export, records, tmp_path / 'run')


def test_the_chart_reads_the_exporters_results(chart, export):
    assert (chart.PROTOCOL, chart.WHOLE) == (export.PROTOCOL, export.WHOLE)


def test_three_panels_of_rates_against_the_budget_with_every_arm_run(chart, results):
    svg = chart.render(results)
    tree = ElementTree.fromstring(svg)
    assert tree.tag == f'{SVG}svg' and tree.find(f'{SVG}title').text.startswith('LongMemEval-S')
    panels = [group.get('aria-label') for group in tree.iter(f'{SVG}g') if group.get('class') == 'panel']
    assert panels == ['Accuracy', 'Evidence reached', 'Trap rate (lower is better)']
    series = {group.get('data-arm') for group in tree.iter(f'{SVG}g') if group.get('class') == 'series'}
    assert series == {'all-on', 'import-only', 'session-search'}
    legend = [text.text for text in tree.iter(f'{SVG}text') if text.get('class') == 'legend']
    # In the arms' fixed order, the one each color belongs to.
    assert legend == ['all-on', 'session-search (stock Hermes)', 'import-only']
    for budget in ('4,000', '24,000'):
        assert any(text.text == budget for text in tree.iter(f'{SVG}text'))
    # Every point carries its value and interval for a reader without the colors.
    titles = [node.text for node in tree.iter(f'{SVG}title')]
    assert 'all-on, 4,000 characters: accuracy 100.0% [51.0, 100.0] (4/4)' in titles


def test_an_arm_keeps_its_color_whatever_else_was_run(chart, results):
    """Color follows the arm, never its position among the arms a run happened to include."""
    fewer = dict(results, arms=['session-search'],
                 tables={'session-search': results['tables']['session-search']})
    for document in (results, fewer):
        tree = ElementTree.fromstring(chart.render(document))
        groups = [group for group in tree.iter(f'{SVG}g') if group.get('data-arm') == 'session-search']
        assert len(groups) == 3                     # one line in each panel
        for group in groups:
            assert group.get('class') == 'series'
            assert group.get('style') == f'color: var(--series-{chart.SLOTS["session-search"]})'


def test_the_same_results_always_draw_the_same_bytes(chart, results):
    shuffled = json.loads(json.dumps(results))
    keys = list(shuffled['tables'])
    random.Random(3).shuffle(keys)
    shuffled['tables'] = {key: shuffled['tables'][key] for key in keys}
    shuffled['arms'] = list(reversed(shuffled['arms']))
    assert chart.render(results) == chart.render(results) == chart.render(shuffled)


def test_an_arm_with_nothing_to_draw_is_left_out_and_said_so(chart, results):
    tables = json.loads(json.dumps(results['tables']))
    tables['import-only']['4000']['overall']['trap_rate'] = None
    tree = ElementTree.fromstring(chart.render(dict(results, tables=tables)))
    panel, = [group for group in tree.iter(f'{SVG}g') if group.get('aria-label') == 'Trap rate (lower is better)']
    assert 'import-only' not in {group.get('data-arm') for group in panel.iter(f'{SVG}g')}


def test_a_file_that_is_not_a_results_file_is_refused(chart, tmp_path):
    (tmp_path / 'x.json').write_text(json.dumps({'protocol': 'public-memory-report-1'}))
    with pytest.raises(SystemExit):
        chart.main([str(tmp_path / 'x.json')])


def test_the_command_writes_beside_the_results_and_checks_a_committed_chart(chart, export, results, tmp_path):
    path = tmp_path / 'longmemeval_s.json'
    path.write_text(export.dumps(results))
    assert chart.main([str(path), '--check']) == 1                  # no chart yet
    assert chart.main([str(path)]) == 0
    assert (tmp_path / 'longmemeval_s.svg').read_text() == chart.render(results)
    assert chart.main([str(path), '--check']) == 0
    (tmp_path / 'longmemeval_s.svg').write_text('<svg/>')
    assert chart.main([str(path), '--check']) == 1


def test_every_committed_chart_is_its_committed_results_drawn(chart, export):
    for path in sorted(export.RESULTS_DIR.glob('*.json')) if export.RESULTS_DIR.is_dir() else []:
        drawn = path.with_suffix('.svg')
        assert drawn.is_file() and drawn.read_text() == chart.render(json.loads(path.read_text())), path.name


def test_the_whole_history_is_drawn_past_the_largest_budget(chart, export, records, tmp_path):
    from test_public_memory_export import with_full_context
    tree = ElementTree.fromstring(chart.render(with_full_context(export, records, tmp_path)))
    labels = [text.text for text in tree.iter(f'{SVG}text') if text.get('class') == 'tick']
    assert labels[-3:] == ['4,000', '24,000', 'whole history']
    titles = [node.text for node in tree.iter(f'{SVG}title')]
    assert 'full-context, whole history: accuracy 100.0% [20.7, 100.0] (1/1)' in titles
