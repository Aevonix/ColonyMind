"""The public memory benchmark page (``docs/PUBLIC-MEMORY-BENCHMARK.md``): its method is written by hand, and every
prompt, pin and number on it is generated (``export.py page``) from the code and the committed results files, so the
page can never say what the harness does not do or what no run measured."""
import json
from pathlib import Path
import shutil

import pytest

from test_public_memory_export import build, export, frozen_labels, records  # noqa: F401  (pytest fixtures)

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'docs' / 'PUBLIC-MEMORY-BENCHMARK.md'


def test_the_committed_page_is_what_the_committed_results_render(export):
    text = PAGE.read_text(encoding='utf-8')
    assert export.render_page(text, export.RESULTS_DIR) == text


def test_the_page_prints_the_prompts_and_pins_the_harness_uses(export):
    text = PAGE.read_text(encoding='utf-8')
    for prompt in (export.reader.SYSTEM, export.judge.TRAP_SYSTEM, export.judge.ABSTAIN_SYSTEM):
        assert prompt in text
    assert export.reader.READING_SHA256 in text and export.judge.PROMPTS_SHA256 in text
    for name, pin in export.PUBLIC.items():
        assert pin['sha256'] in text and pin['licence'] in text
    for budget in export.runner.BUDGETS:
        assert f'{budget:,}' in text


def test_the_page_describes_every_arm_the_harness_runs(export):
    text = PAGE.read_text(encoding='utf-8')
    for arm in sorted(export.ARMS):
        assert f'| `{arm}` |' in text, arm


def test_without_a_published_run_the_page_claims_nothing(export, tmp_path):
    block = export.render_results(tmp_path / 'none')
    assert 'No run has been published yet' in block and 'not demonstrated' in block


def test_a_run_renders_its_pins_models_tables_tests_failures_and_agreement(export, records, tmp_path):
    results = build(export, records, tmp_path / 'run', subset=None)
    directory = tmp_path / 'results'
    directory.mkdir()
    (directory / 'longmemeval_s.json').write_text(export.dumps(results))
    block = export.render_results(directory)
    assert block.startswith('### LongMemEval-S, all 4 questions\n')
    assert export.PUBLIC['longmemeval_s']['sha256'] in block and 'a' * 40 in block and frozen_labels()[1] in block
    assert '`open-reader-7b` | `rev-r`' in block and '`open-judge-70b` | `rev-j`' in block
    assert '[page](https://huggingface.co/example/open-reader-7b)' in block
    assert '| `all-on` | 100.0% [51.0, 100.0] (4/4) |' in block
    assert '| `temporal-reasoning` |' in block
    # Per question type, each arm's accuracy with its Wilson interval, as the overall table prints it.
    assert '#### Accuracy by question type, 4,000 characters' in block
    kinds = results['tables']['all-on']['4000']['types']
    assert (kinds['knowledge-update']['right']['successes'], kinds['knowledge-update']['right']['trials']) == (2, 2)
    assert '| `knowledge-update` | 100.0% [34.2, 100.0] (2/2) |' in block
    assert f'| `single-session-user` | {export._rate(kinds["single-session-user"]["right"])} |' in block
    # The paired tests: the withheld one says why, and one short of the rule says it is not demonstrated.
    assert 'withheld: a question without a verdict in either arm' in block
    assert 'not demonstrated' in block
    assert '| `session-search` | 4,000 | `q-tr` | judge_undecided |' in block
    assert '87.5% [52.9, 97.8] of 8 labelled, kappa 0.71 (1 left unlabelled)' in block
    assert 'superseded 1' in block
    assert 'q-ab_abs' in block                  # the question ids
    assert '![' not in block                    # no chart until one is committed beside the results
    (directory / 'longmemeval_s.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    assert '![Accuracy, evidence reached and traps' in export.render_results(directory)
    assert export.render_results(directory) == export.render_results(directory)


def test_a_file_that_is_not_a_results_file_is_refused(export, tmp_path):
    (tmp_path / 'beam.json').write_text(json.dumps({'protocol': 'something-else'}))
    with pytest.raises(export.ExportError, match='results file'):
        export.render_results(tmp_path)


def test_the_page_command_rewrites_a_stale_page_and_checks_a_current_one(export, tmp_path, capsys):
    page = tmp_path / 'PAGE.md'
    shutil.copy(PAGE, page)
    stale = page.read_text().replace('No run has been published yet', 'A run was published')
    page.write_text(stale)
    assert export.main(['page', '--page', str(page), '--results-dir', str(tmp_path / 'none'), '--check']) == 1
    assert page.read_text() == stale
    assert export.main(['page', '--page', str(page), '--results-dir', str(tmp_path / 'none')]) == 0
    assert export.main(['page', '--page', str(page), '--results-dir', str(tmp_path / 'none'), '--check']) == 0
    assert page.read_text() == export.render_page(PAGE.read_text(), tmp_path / 'none')


def test_the_whole_history_is_its_own_column(export, records, tmp_path):
    from test_public_memory_export import with_full_context
    directory = tmp_path / 'results'
    directory.mkdir()
    (directory / 'longmemeval_s.json').write_text(export.dumps(with_full_context(export, records, tmp_path / 'r')))
    block = export.render_results(directory)
    assert '| Arm | 4,000 characters | 24,000 characters | whole history |' in block
    assert '| `full-context` | not run | not run | 100.0% [20.7, 100.0] (1/1) |' in block
    assert '#### Accuracy by question type, whole history' in block
    assert '| `full-context` | whole history | `q-su` | does_not_fit |' in block


def test_a_subset_run_says_so_in_its_heading(export, records, tmp_path):
    from test_public_memory_export import campaign, declaration, labels_beside
    data, answers, verdicts, storage, _ = campaign(export, records, tmp_path / 'r')
    subset = export.subsets.draw(data, seed=5, size=4).record()
    results = export.build(declaration(subset=subset), data, answers=[answers], verdicts=[verdicts],
                           storage=[storage], labels=labels_beside(answers))
    directory = tmp_path / 'results'
    directory.mkdir()
    (directory / 'longmemeval_s-subset.json').write_text(export.dumps(results))
    block = export.render_results(directory)
    assert block.startswith('### LongMemEval-S, a subset of 4 questions (seed 5)\n')
    assert 'drawn by sha256(seed:id) within strata, largest remainder' in block
    # The section links the files it was rendered from, never another run's of the same dataset.
    link = export.PAGE_LINK
    assert f'[`longmemeval_s-subset.json`]({link}longmemeval_s-subset.json)' in block
    assert f'({link}longmemeval_s.json)' not in block
    (directory / 'longmemeval_s-subset.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    block = export.render_results(directory)
    assert f']({link}longmemeval_s-subset.svg)' in block and f'({link}longmemeval_s.svg)' not in block
