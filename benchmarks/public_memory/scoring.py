"""Scoring for memory benchmarks: whether the evidence reached the reader, whether the answer was right, and traps.

The three are counted apart and never averaged into one accuracy. A right answer from evidence that never reached
the reader is a guess or prior knowledge; a wrong answer from evidence that did reach it is a reading failure; the
two-by-two keeps them apart.

**Reach** is structural. Each gold evidence item is a set of identifiers that name the same stored message (the turn
id and the message hash recorded when the history was replayed); each item the memory section showed is the set of
identifiers it cites. An item is reached when the section cites any of its identifiers. No text is matched. A
baseline that returns whole sessions reaches when a returned session id names a gold session. Where neither is
available, ``span_reach`` checks that the answer span appears as whole words in what the reader was shown, and every
result it gives is labelled ``weaker``.

**Traps** are counted by kind, from structured judgments of what the answer asserts or chooses (a judge's verdict,
or a structured answer's own fields), never from words found in its text: a mention is no assertion ("no longer the
corner office" names the old value and asserts none), and a choice can name the side it rejects ("09:15, not
10:45").

- ``update``: the answer asserts a value that was current once and has since been replaced or withdrawn (a stale
  value), or asserts it beside the current one without choosing; ``asked`` when the answer flags the change and asks
  which value holds.
- ``conflict``: two statements disagree and nothing settles them; choosing one side is the trap, ``asked`` is the
  outcome the question wants.
- ``abstain``: nothing answers the question; a specific value given as answerable is the trap.
- ``recall``: no trap; the answer is right or it is not.

**Statistics** reuse the paired track's methods (``colonymind.qualification.paired_statistics``): the exact sign test
over paired questions and a seeded percentile bootstrap that resamples whole clusters (the questions that share one
conversation's store), plus Wilson intervals per arm. As in the paired report, the repetitions of one scenario are
averaged into one value per arm before testing, and a contrast with any unit whose outcome is missing in either arm
is withheld: no partial cohort is tested.
"""
from pathlib import Path
import math
import re
import statistics
import sys
import unicodedata

try:
    from colonymind.qualification import paired_statistics
except ImportError:  # a checkout without the package installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sidecar'))
    from colonymind.qualification import paired_statistics

STRUCTURAL, SESSION, SPAN = 'structural', 'session', 'span'
KINDS = ('recall', 'update', 'conflict', 'abstain')
CORRECT, ASKED, TRAP, OTHER = 'correct', 'asked', 'trap', 'other'
OUTCOMES = (CORRECT, ASKED, TRAP, OTHER)
Z95 = 1.959963984540054
# The scout's sizing assumption: a quarter of the questions come out differently between two arms.
DISAGREEMENT = 0.25
_WORD = re.compile(r'\w+')


def words(text):
    """Case- and width-folded words: the unit a span or value is matched in (``span_reach``, traps)."""
    return _WORD.findall(unicodedata.normalize('NFKC', str(text or '')).casefold())


def shows(text_words, value):
    """Whether ``value``'s words appear, contiguous and whole, in ``text_words``."""
    needle = words(value)
    if not needle:
        return False
    width = len(needle)
    return any(text_words[index:index + width] == needle for index in range(len(text_words) - width + 1))


def _result(basis, gold, hits, first):
    count = len(gold)
    reached = sum(hits)
    return {'basis': basis, 'weaker': basis == SPAN, 'gold': count, 'reached': reached,
            'any': reached > 0 if count else None, 'all': reached == count if count else None,
            'first_rank': first if count else None}


def reach(gold, shown):
    """Structural reach: ``gold`` is the evidence items, each an iterable of identifiers for one stored message;
    ``shown`` is the memory section's items in order, each the identifiers it cites. ``first_rank`` is the 1-based
    position of the first shown item that cites any gold item."""
    gold = [frozenset(item) for item in gold]
    shown = [frozenset(item) for item in shown]
    if any(not item for item in gold):
        raise ValueError('A gold evidence item needs at least one identifier')
    hits = [any(item & cited for cited in shown) for item in gold]
    wanted = frozenset().union(*gold) if gold else frozenset()
    first = next((rank for rank, cited in enumerate(shown, 1) if cited & wanted), None)
    return _result(STRUCTURAL, gold, hits, first)


def session_reach(gold_sessions, returned_sessions):
    """A baseline that returns whole sessions: reached when a returned session id names a gold session."""
    gold = sorted(set(gold_sessions))
    returned = list(returned_sessions)
    hits = [session in returned for session in gold]
    first = next((rank for rank, session in enumerate(returned, 1) if session in gold), None)
    return _result(SESSION, gold, hits, first)


def span_reach(spans, texts):
    """The weaker check: each answer span shown as whole words in any of ``texts`` (in the order shown)."""
    spans = [span for span in spans if words(span)]
    texts = [words(text) for text in texts]
    hits = [any(shows(text, span) for text in texts) for span in spans]
    first = next((rank for rank, text in enumerate(texts, 1) if any(shows(text, span) for span in spans)), None)
    return _result(SPAN, spans, hits, first)


def two_by_two(rows):
    """Counts of reached and right. A row with no applicable reach (no gold evidence) or no verdict is unscored."""
    table = {'reached_right': 0, 'reached_wrong': 0, 'missed_right': 0, 'missed_wrong': 0, 'unscored': 0}
    for row in rows:
        reached = (row.get('reach') or {}).get('any')
        right = row.get('right')
        if reached is None or right is None:
            table['unscored'] += 1
        else:
            table[('reached_' if reached else 'missed_') + ('right' if right else 'wrong')] += 1
    return table


def same_value(first, second):
    """Whether two values are the same value: equal as whole words, case and width folded."""
    return bool(words(first)) and words(first) == words(second)


def trap_outcome(kind, *, right, asserted=(), asked=False, answerable=None, stale=(), sides=()):
    """The trap outcome of one answer: ``correct``, ``asked``, ``trap`` or ``other`` (wrong without a trap).

    ``asserted`` is a structured judgment: the values the answer asserts as current (``update``), the side it
    chooses (``conflict``), or the specific values it gives as the answer (``abstain``). No text is searched: a value
    handed over is compared whole (``same_value``). ``right`` is the grader's or judge's verdict, ``asked`` whether
    the answer flagged the conflict and asked which value holds, ``answerable`` the reader's own claim that an
    answer exists, ``stale`` the values that were current once (``update``), ``sides`` the disagreeing values
    (``conflict``). An asserted stale value or a chosen side is the trap even beside a right verdict or an ask.
    ``matched`` lists the stale values or sides asserted.
    """
    if kind not in KINDS:
        raise ValueError(f'A trap kind is one of {", ".join(KINDS)}')
    asserted = [value for value in asserted if words(value)]
    outcome, matched = OTHER, []
    if kind == 'update':
        matched = [value for value in stale if any(same_value(value, given) for given in asserted)]
        # "Moved from X to Y" asserts Y alone; X asserted as current, beside Y or not, is the trap.
        outcome = TRAP if matched else ASKED if asked else CORRECT if right else OTHER
    elif kind == 'conflict':
        matched = [value for value in sides if any(same_value(value, given) for given in asserted)]
        # One side chosen is the trap; both given chooses neither.
        outcome = TRAP if len(matched) == 1 else ASKED if asked else OTHER
    elif kind == 'abstain':
        if answerable is False or (answerable is None and right):
            outcome = CORRECT
        elif asserted:
            outcome = TRAP
    else:
        outcome = CORRECT if right else OTHER
    return {'kind': kind, 'outcome': outcome, 'matched': matched}


def wilson(successes, trials, *, z=Z95):
    """The Wilson score interval of a proportion; None when there are no trials."""
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise ValueError('Successes are a count within the trials')
    if trials == 0:
        return None
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return {'successes': successes, 'trials': trials, 'rate': rate,
            'lower': max(0.0, centre - margin), 'upper': min(1.0, centre + margin)}


def _binary(row, key):
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f'{key} must be true or false for every paired question')
    return value


def _outcome(row, key):
    value = row.get(key)
    if value is not None and type(value) is not bool:
        raise ValueError(f'{key} is true, false, or None where it was not attributed, for every paired question')
    return value


def contrast(treatment, comparator, *, key='right', seed, unit_key=None, cluster_key=None):
    """One arm against another over the same questions: the exact sign test on per-unit outcomes and a seeded
    bootstrap interval of the difference, resampling whole clusters when ``cluster_key`` names them.

    Rows pair by ``question``, one attempt each. ``unit_key`` names the unit that repetitions share (the scenario):
    as in the paired report, a unit's repetitions are averaged into one pass value per arm before anything is tested.
    Without it, each question is its own unit.

    Pass every question the measure applies to. One whose outcome is None in either arm (not attributed, or not
    observed) makes its unit unavailable, and a contrast with any unavailable unit is withheld (verdict
    ``unavailable``): no partial cohort is tested. ``declared_units`` and ``unavailable_units`` report the coverage
    either way."""
    first = {row['question']: row for row in treatment}
    second = {row['question']: row for row in comparator}
    if set(first) != set(second) or len(first) != len(treatment) or len(second) != len(comparator):
        raise ValueError('A paired contrast needs the same questions, once each, in both arms')
    grouped, clusters = {}, {}
    for question in sorted(first):
        left, right = first[question], second[question]
        unit = question if unit_key is None else left[unit_key]
        if unit_key is not None and right[unit_key] != unit:
            raise ValueError('An attempt belongs to the same unit in both arms')
        grouped.setdefault(unit, []).append((_outcome(left, key), _outcome(right, key)))
        if cluster_key is not None:
            cluster = left[cluster_key]
            if right[cluster_key] != cluster or clusters.setdefault(unit, cluster) != cluster:
                raise ValueError('A unit belongs to one cluster, the same in both arms')
    unavailable = sum(any(None in pair for pair in values) for values in grouped.values())
    coverage = {'declared_units': len(grouped), 'unavailable_units': unavailable}
    if unavailable or not grouped:
        return {'verdict': 'unavailable', **coverage}
    units = {unit: (sum(left for left, _ in values) / len(values), sum(right for _, right in values) / len(values))
             for unit, values in grouped.items()}
    return {**paired_statistics.contrast(units, seed=seed, clusters=clusters if cluster_key is not None else None),
            **coverage}


def interval(rows, *, key='right', seed, cluster_key=None):
    """A seeded percentile bootstrap interval of one arm's rate, resampling whole clusters when named."""
    groups = {}
    for index, row in enumerate(rows):
        cluster = row[cluster_key] if cluster_key is not None else index
        groups.setdefault(cluster, []).append(float(_binary(row, key)))
    return paired_statistics.cluster_bootstrap(list(groups.values()), seed=seed)


def _rate(rows, test):
    values = [test(row) for row in rows]
    values = [value for value in values if value is not None]
    return wilson(sum(values), len(values)) if values else None


def _table(rows):
    reaches = [row.get('reach') or {} for row in rows]
    scored = [item for item in reaches if item.get('any') is not None]
    ranks = [item['first_rank'] for item in scored if item.get('first_rank') is not None]
    fills = [row['chars'] / row['budget'] for row in rows
             if isinstance(row.get('chars'), (int, float)) and isinstance(row.get('budget'), (int, float))
             and row['budget'] > 0]
    tokens = [row['tokens'] for row in rows if isinstance(row.get('tokens'), (int, float))]
    traps = [(row.get('trap') or {}).get('outcome') for row in rows]
    traps = [outcome for outcome in traps if outcome is not None]
    basis = {}
    for item in scored:
        basis[item['basis']] = basis.get(item['basis'], 0) + 1
    return {'questions': len(rows),
            'right': _rate(rows, lambda row: row.get('right')),
            'reach_any': _rate(rows, lambda row: (row.get('reach') or {}).get('any')),
            'reach_all': _rate(rows, lambda row: (row.get('reach') or {}).get('all')),
            'reach_basis': basis,
            'two_by_two': two_by_two(rows),
            'first_rank_median': statistics.median(ranks) if ranks else None,
            'tokens_mean': statistics.fmean(tokens) if tokens else None,
            'fill_ratio_mean': statistics.fmean(fills) if fills else None,
            'traps': {outcome: traps.count(outcome) for outcome in OUTCOMES if outcome in traps},
            'trap_rate': wilson(traps.count(TRAP), len(traps)) if traps else None}


def summarize(rows, *, key='type'):
    """Per-type tables and the overall table for one arm's question rows.

    A row is ``{question, type, reach, right, trap, chars, budget, tokens}``: ``reach`` from ``reach``,
    ``session_reach`` or ``span_reach``, ``right`` the verdict, ``trap`` from ``trap_outcome``, ``chars`` and
    ``tokens`` what the memory section used of its ``budget``. Missing fields are left out of their counts.
    """
    types = {}
    for row in rows:
        types.setdefault(row.get(key), []).append(row)
    return {'types': {name: _table(members) for name, members in sorted(types.items(), key=lambda item: str(item[0]))},
            'overall': _table(list(rows))}
