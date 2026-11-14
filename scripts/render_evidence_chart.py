"""Draw the public memory benchmark's evidence chart from a results file, as an SVG.

Three panels share one scale (0 to 100%): accuracy, evidence reached and the trap rate, each against the memory
budget, one line per arm with its 95% Wilson interval at every budget. The stock Hermes arm (``session-search``) is
drawn beside ColonyMind's. An arm keeps one color whatever else a run included (``SLOTS``, the reference
categorical order, validated for colorblind separation in light and dark); arms without a slot are in the page's
tables, which also carry every number drawn here. Each point's value and interval are in its tooltip text.

The chart is generated from the results file alone (``benchmarks/public_memory/export.py results``) and the same
file always draws the same bytes: no date, no random order, fixed number formats. It is committed beside the results
(``<results>.svg``); ``--check`` exits 1 when the committed chart is not the results drawn.

    python scripts/render_evidence_chart.py benchmarks/public_memory/results/longmemeval_s.json [--check]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional
from xml.sax.saxutils import escape, quoteattr

PROTOCOL = 'public-memory-results-1'
# The arms the chart draws, each in its own categorical slot; the order is also the legend's.
SLOTS = {'all-on': 1, 'session-search': 2, 'import-only': 3, 'oracle': 4, 'bm25-sessions': 5, 'lexical-only': 6,
         'no-night': 7, 'full-context': 8}
LABELS = {'session-search': 'session-search (stock Hermes)'}
TITLES = {'longmemeval_s': 'LongMemEval-S', 'beam': 'BEAM, 100K split'}
# The full-context baseline's tables: the whole history, drawn past the largest budget.
WHOLE = 'whole'
PANELS = (('Accuracy', 'right', 'accuracy'), ('Evidence reached', 'reach_any', 'evidence reached'),
          ('Trap rate (lower is better)', 'trap_rate', 'trap rate'))
LIGHT = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
DARK = ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767')
WIDTH = 960
LEFT, PANEL_WIDTH, GAP = 64, 250, 56
# The first legend row's baseline, the height of a legend row, where the panels' plots start below the legend, and
# their height. A legend item is as wide as its key and label at an estimated 6.8px per character.
LEGEND_Y, LEGEND_ROW, PLOT_BELOW_LEGEND, PLOT_HEIGHT = 76, 22, 52, 220
CHAR_WIDTH = 6.8
TICKS = (0, 25, 50, 75, 100)
DODGE = 5.0


def _tokens(ink, ink_2, muted, grid, axis, surface, series) -> str:
    colors = ' '.join(f'--series-{slot}: {color};' for slot, color in enumerate(series, 1))
    return (f'--surface: {surface}; --ink: {ink}; --ink-2: {ink_2}; --muted: {muted}; --grid: {grid}; '
            f'--axis: {axis}; {colors}')


STYLE = (
    f'svg {{ {_tokens("#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", LIGHT)} }}\n'
    f'@media (prefers-color-scheme: dark) {{ svg {{ '
    f'{_tokens("#ffffff", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "#1a1a19", DARK)} }} }}\n'
    'text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 12px; fill: var(--ink-2); }\n'
    '.title { font-size: 16px; font-weight: 600; fill: var(--ink); }\n'
    '.heading { font-size: 13px; font-weight: 600; fill: var(--ink); }\n'
    '.tick, .note { fill: var(--muted); font-variant-numeric: tabular-nums; }\n'
    '.legend { fill: var(--ink-2); }\n'
    '.grid { stroke: var(--grid); stroke-width: 1; }\n'
    '.axis { stroke: var(--axis); stroke-width: 1; }\n'
    '.line { fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }\n'
    '.whisker { stroke: currentColor; stroke-width: 1.5; stroke-linecap: round; opacity: 0.55; }\n'
    '.dot { fill: currentColor; stroke: var(--surface); stroke-width: 2; }\n')


def _n(value: float) -> str:
    return f'{value:.1f}'


def _pct(value: float) -> str:
    return f'{100 * value:.1f}'


def _y(rate: float, top: float) -> float:
    return top + PLOT_HEIGHT * (1 - rate)


def _key(budget: Optional[int]) -> str:
    return WHOLE if budget is None else str(budget)


def _text(budget: Optional[int]) -> str:
    return 'whole history' if budget is None else f'{budget:,}'


def _xs(left: float, budgets) -> dict:
    pad = 28.0
    if len(budgets) == 1:
        return {budgets[0]: left + PANEL_WIDTH / 2}
    step = (PANEL_WIDTH - 2 * pad) / (len(budgets) - 1)
    return {budget: left + pad + index * step for index, budget in enumerate(budgets)}


def _point(results: Mapping[str, Any], arm: str, budget: int, key: str) -> Optional[Mapping[str, Any]]:
    table = results['tables'].get(arm, {}).get(_key(budget))
    item = (table or {}).get('overall', {}).get(key)
    return item if isinstance(item, Mapping) and item.get('trials') else None


def arms(results: Mapping[str, Any]) -> list:
    """The arms drawn: those the run holds that have a slot, in slot order."""
    return sorted((arm for arm in set(results['arms']) | set(results['tables']) if arm in SLOTS), key=SLOTS.get)


def _panel(results: Mapping[str, Any], index: int, title: str, key: str, noun: str, budgets: list,
           top: float) -> list:
    left, bottom = LEFT + index * (PANEL_WIDTH + GAP), top + PLOT_HEIGHT
    xs = _xs(left, budgets)
    out = [f'<g class="panel" aria-label={quoteattr(title)}>',
           f'<text class="heading" x="{_n(left)}" y="{_n(top - 16)}">{escape(title)}</text>']
    for tick in TICKS:
        y = _y(tick / 100, top)
        out.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{_n(left)}" x2="{_n(left + PANEL_WIDTH)}" '
                   f'y1="{_n(y)}" y2="{_n(y)}"/>')
        out.append(f'<text class="tick" x="{_n(left - 8)}" y="{_n(y + 4)}" text-anchor="end">{tick}%</text>')
    for budget in budgets:
        out.append(f'<text class="tick" x="{_n(xs[budget])}" y="{_n(bottom + 20)}" text-anchor="middle">'
                   f'{_text(budget)}</text>')
    drawn = [arm for arm in arms(results) if any(_point(results, arm, budget, key) for budget in budgets)]
    for position, arm in enumerate(drawn):
        offset = (position - (len(drawn) - 1) / 2) * DODGE
        points = [(budget, _point(results, arm, budget, key)) for budget in budgets]
        out.append(f'<g class="series" data-arm={quoteattr(arm)} style="color: var(--series-{SLOTS[arm]})">')
        run: list = []
        for budget, item in [*points, (None, None)]:
            if item is not None:
                run.append((xs[budget] + offset, _y(item['rate'], top)))
                continue
            if len(run) > 1:
                out.append('<path class="line" d="' + ' '.join(f'{"M" if i == 0 else "L"}{_n(x)},{_n(y)}'
                                                                 for i, (x, y) in enumerate(run)) + '"/>')
            run = []
        for budget, item in points:
            if item is None:
                continue
            x = xs[budget] + offset
            out.append(f'<line class="whisker" x1="{_n(x)}" x2="{_n(x)}" y1="{_n(_y(item["lower"], top))}" '
                       f'y2="{_n(_y(item["upper"], top))}"/>')
            where = 'whole history' if budget is None else f'{budget:,} characters'
            label = (f'{arm}, {where}: {noun} {_pct(item["rate"])}% [{_pct(item["lower"])}, '
                     f'{_pct(item["upper"])}] ({item["successes"]}/{item["trials"]})')
            out.append(f'<circle class="dot" cx="{_n(x)}" cy="{_n(_y(item["rate"], top))}" r="4.5">'
                       f'<title>{escape(label)}</title></circle>')
        out.append('</g>')
    out.append('</g>')
    return out


def render(results: Mapping[str, Any]) -> str:
    """The chart's SVG text for one results document."""
    if not isinstance(results, Mapping) or results.get('protocol') != PROTOCOL:
        raise ValueError('not a results file (benchmarks/public_memory/export.py results)')
    name = results['dataset']['name']
    title = TITLES.get(name, name)
    budgets = sorted({None if key == WHOLE else int(key) for tables in results['tables'].values() for key in tables},
                     key=lambda budget: (budget is None, budget or 0))
    shown = arms(results)
    description = (f'Accuracy, evidence reached and trap rate against the memory budget ('
                   f'{", ".join(_text(budget) for budget in budgets)}) for the arms '
                   f'{", ".join(LABELS.get(arm, arm) for arm in shown)}, each with a 95% Wilson interval. '
                   'Every value is also in the benchmark page\'s tables.')
    keys, x, y = [], float(LEFT), float(LEGEND_Y)
    for arm in shown:
        label = LABELS.get(arm, arm)
        width = 24 + CHAR_WIDTH * len(label) + 20
        if x > LEFT and x + width - 20 > WIDTH - LEFT / 2:
            x, y = float(LEFT), y + LEGEND_ROW
        keys += [f'<g class="key" style="color: var(--series-{SLOTS[arm]})">',
                 f'<line class="line" x1="{_n(x)}" x2="{_n(x + 18)}" y1="{_n(y - 4)}" y2="{_n(y - 4)}"/>',
                 f'<circle class="dot" cx="{_n(x + 9)}" cy="{_n(y - 4)}" r="4.5"/></g>',
                 f'<text class="legend" x="{_n(x + 24)}" y="{_n(y)}">{escape(label)}</text>']
        x += width
    top = y + PLOT_BELOW_LEGEND
    bottom = top + PLOT_HEIGHT
    height = int(bottom + 86)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" '
           f'height="{height}" role="img" aria-labelledby="chart-title chart-desc">',
           f'<title id="chart-title">{escape(title)}: accuracy, evidence reached and traps by memory budget</title>',
           f'<desc id="chart-desc">{escape(description)}</desc>',
           f'<style>\n{STYLE}</style>',
           f'<rect x="0" y="0" width="{WIDTH}" height="{height}" rx="8" fill="var(--surface)"/>',
           f'<text class="title" x="{LEFT}" y="30">{escape(title)}: what reached the reader, and what it answered'
           '</text>',
           f'<text x="{LEFT}" y="50">Rates per arm against the memory budget, with 95% Wilson intervals</text>',
           *keys]
    for index, (heading, key, noun) in enumerate(PANELS):
        out += _panel(results, index, heading, key, noun, budgets, top)
    out += [f'<text class="note" x="{WIDTH / 2:.1f}" y="{_n(bottom + 44)}" text-anchor="middle">Memory budget '
            '(characters)</text>',
            f'<text class="note" x="{LEFT}" y="{height - 12}">Evidence reached is structural for ColonyMind\'s arms '
            'and a gold session shown for the baselines. Arms not drawn are in the page\'s tables.</text>',
            '</svg>']
    return '\n'.join(out) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path, help='a results file (export.py results)')
    parser.add_argument('--output', type=Path, help='the SVG to write (default: beside the results, .svg)')
    parser.add_argument('--check', action='store_true', help='exit 1 when the chart is not the results drawn')
    args = parser.parse_args(argv)
    try:
        svg = render(json.loads(args.results.read_text(encoding='utf-8')))
    except (ValueError, KeyError, TypeError) as error:
        raise SystemExit(f'{args.results}: {error}') from None
    output = args.output or args.results.with_suffix('.svg')
    if args.check:
        if not output.is_file() or output.read_text(encoding='utf-8') != svg:
            print(f'{output} is not {args.results.name} drawn: run scripts/render_evidence_chart.py',
                  file=sys.stderr)
            return 1
        return 0
    output.write_text(svg, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
