"""The handback experiment (``benchmarks/working_memory/handback``): synthetic finished worker tasks, 40 keyed
detail questions per child with gold by construction, four arms answered through one parent prompt (the child's
summary, verbatim excerpts with handles, a consult of the child's exact context, the full text), a runner that only
talks to a loopback server, and a scorer with paired bootstrap intervals."""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "working_memory"


@pytest.fixture(scope="module")
def hb():
    sys.path.insert(0, str(BENCH))
    try:
        from handback import arms, run, score, sets
    finally:
        sys.path.remove(str(BENCH))

    class Modules:
        pass

    modules = Modules()
    modules.sets, modules.arms, modules.run, modules.score = sets, arms, run, score
    return modules


@pytest.fixture(scope="module")
def one_set(hb):
    return hb.sets.build_set(hb.sets.frozen()["sets"][0]["seed"])


def chars4(text):
    """A stub token count: one token per four characters, rounded up."""
    return -(-len(text) // 4)


# 4. The sets.

def test_sets_are_deterministic_and_frozen_by_hash(hb):
    frozen = hb.sets.frozen()
    assert frozen["version"] == hb.sets.VERSION and len(frozen["sets"]) >= 1
    for entry in frozen["sets"]:
        built = hb.sets.build_set(entry["seed"])
        assert hb.sets.set_hash(built) == entry["sha256"]
        assert hb.sets.set_hash(hb.sets.build_set(entry["seed"])) == entry["sha256"]
    first = frozen["sets"][0]["seed"]
    assert hb.sets.set_hash(hb.sets.build_set(first + 1)) != frozen["sets"][0]["sha256"]
    assert hb.sets.load_frozen_set(first)["seed"] == first


def test_a_set_whose_hash_is_not_frozen_is_refused(hb, monkeypatch):
    seed = hb.sets.frozen()["sets"][0]["seed"]
    real = hb.sets.build_set

    def changed(value):
        built = real(value)
        built["children"][0]["units"][0]["text"] += " One more sentence."
        return built

    monkeypatch.setattr(hb.sets, "build_set", changed)
    with pytest.raises(ValueError, match="frozen"):
        hb.sets.load_frozen_set(seed)
    with pytest.raises(ValueError, match="frozen"):
        hb.sets.load_frozen_set(10 ** 9)


def test_each_set_has_eight_children_and_forty_questions_per_child_in_the_declared_shares(hb, one_set):
    children = one_set["children"]
    assert len(children) == hb.sets.CHILDREN == 8
    assert hb.sets.QUESTIONS_PER_CHILD == 40
    assert hb.sets.SHARES == {"contradiction": 10, "relation": 10, "cross_child": 10, "single": 10}
    for child in children:
        mine = [q for q in one_set["questions"] if q["child"] == child["id"]]
        assert len(mine) == 40
        assert {t: sum(1 for q in mine if q["type"] == t) for t in hb.sets.SHARES} == hb.sets.SHARES
    ids = [q["id"] for q in one_set["questions"]]
    assert len(ids) == len(set(ids))


def test_every_gold_answer_occurs_in_its_childs_material_and_never_in_its_question(hb, one_set):
    by_id = {child["id"]: child for child in one_set["children"]}
    for question in one_set["questions"]:
        child = by_id[question["child"]]
        units = {unit["handle"]: unit for unit in child["units"]}
        assert question["gold"] in units[question["gold_handle"]]["text"]
        assert question["gold"] in hb.sets.material_text(child)
        assert question["gold"].lower() not in question["question"].lower()
        assert question["gold"] not in question["distractors"]
        assert question["child"] in question["children"]
        if question["type"] == "cross_child":
            assert len(question["children"]) >= 2
            others = [by_id[c] for c in question["children"] if c != question["child"]]
            # The link that picks the entity is only in another child's material.
            assert any(question["link"] in hb.sets.material_text(other) for other in others)
            assert question["link"] not in hb.sets.material_text(child)
        else:
            assert question["children"] == [question["child"]]


def _relations(text):
    """The direct links the material states: each person's manager and each task's prerequisites."""
    managers, prerequisites = {}, {}
    for a, b in re.findall(r"(p-\d+) reports to (p-\d+)\.", text):
        managers.setdefault(a, set()).add(b)
    for b, a in re.findall(r"(p-\d+) is the direct manager of (p-\d+)\.", text):
        managers.setdefault(a, set()).add(b)
    for x, y in re.findall(r"Task (T-\d+) depends on task (T-\d+)\.", text):
        prerequisites.setdefault(x, set()).add(y)
    for x, y in re.findall(r"Task (T-\d+) cannot start until task (T-\d+) is finished\.", text):
        prerequisites.setdefault(x, set()).add(y)
    return managers, prerequisites


def _closure(graph, start):
    seen, stack = set(), list(graph.get(start, ()))
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(graph.get(node, ()))
    return seen


def _valid_relation_answers(question, text):
    """Every answer the question's wording admits, read from the material: a question about what must be finished
    before a task can start admits every task it depends on, directly or through others."""
    managers, prerequisites = _relations(text)
    wording = question["question"]
    match = re.fullmatch(r"Who is the direct manager of (p-\d+)\?", wording)
    if match:
        return managers.get(match.group(1), set())
    match = re.fullmatch(r"Which task must be finished before task (T-\d+) can start\?", wording)
    if match:
        return _closure(prerequisites, match.group(1))
    match = re.fullmatch(r"Which task does task (T-\d+) depend on directly, not through another task\?", wording)
    if match:
        return prerequisites.get(match.group(1), set())
    raise AssertionError(f"a relation question with unknown wording: {wording}")


def test_every_relation_question_admits_exactly_its_gold_and_never_a_distractor(hb):
    for entry in hb.sets.frozen()["sets"]:
        built = hb.sets.build_set(entry["seed"])
        by_id = {child["id"]: child for child in built["children"]}
        relations = [q for q in built["questions"] if q["type"] == "relation"]
        assert len(relations) == hb.sets.SHARES["relation"] * hb.sets.CHILDREN
        for question in relations:
            valid = _valid_relation_answers(question, hb.sets.material_text(by_id[question["child"]]))
            assert valid == {question["gold"]}, question["id"]
            assert not valid & set(question["distractors"])


def test_a_contradictions_gold_is_the_later_value(hb, one_set):
    by_id = {child["id"]: child for child in one_set["children"]}
    for question in (q for q in one_set["questions"] if q["type"] == "contradiction"):
        order = [unit["handle"] for unit in by_id[question["child"]]["units"]]
        stale = question["distractors"][0]
        units = {unit["handle"]: unit for unit in by_id[question["child"]]["units"]}
        assert stale in units[question["stale_handle"]]["text"]
        assert order.index(question["stale_handle"]) < order.index(question["gold_handle"])


def test_each_childs_material_is_in_the_declared_size_range_and_uses_generic_words_only(hb, one_set):
    low, high = hb.sets.MATERIAL_TOKENS
    for child in one_set["children"]:
        approx = len(hb.sets.material_text(child)) / hb.sets.CHARS_PER_TOKEN
        assert low <= approx <= high * 1.1
        text = hb.sets.material_text(child) + child["work"]
        assert chr(0x2014) not in text and "@" not in text


def test_the_childs_context_is_its_task_material_and_work(hb, one_set):
    child = one_set["children"][0]
    context = child["context"]
    assert [m["role"] for m in context] == ["system", "user", "assistant"]
    for unit in child["units"]:
        assert unit["text"] in context[1]["content"]
    assert context[2]["content"] == child["work"]


# 5. The arms.

def _outputs(hb, one_set):
    summaries = {child["id"]: f"SUMMARY-{child['id']}" for child in one_set["children"]}
    selections = {child["id"]: "\n".join(unit["handle"] for unit in child["units"]) + "\nzz99"
                  for child in one_set["children"]}
    return summaries, selections


def test_the_summary_arm_never_sees_material(hb, one_set):
    summaries, _ = _outputs(hb, one_set)
    for question in one_set["questions"][:80]:
        messages = hb.arms.parent_messages("summary", one_set, question, summaries=summaries)
        text = json.dumps(messages)
        for cid in question["children"]:
            assert f"SUMMARY-{cid}" in text
        for child in one_set["children"]:
            assert not any(json.dumps(unit["text"])[1:-1] in text for unit in child["units"])
        assert question["gold"] not in text


def test_the_handoff_and_excerpt_requests_carry_no_question(hb, one_set):
    child = one_set["children"][0]
    requests = [hb.arms.summary_request(child), hb.arms.excerpt_request(child)]
    for messages in requests:
        assert messages[:len(child["context"])] == child["context"]
        tail = json.dumps(messages[len(child["context"]):])
        assert not any(q["question"] in tail for q in one_set["questions"])


def test_the_excerpt_arm_is_verbatim_with_handles_and_stays_under_its_budget(hb, one_set):
    _, selections = _outputs(hb, one_set)
    for child in one_set["children"]:
        handles = hb.arms.parse_handles(selections[child["id"]], child)
        assert "zz99" not in handles and handles[0] == child["units"][0]["handle"]
        block, used = hb.arms.excerpt_block(child, handles, chars4)
        assert chars4(block) <= hb.arms.EXCERPT_BUDGET
        units = {unit["handle"]: unit for unit in child["units"]}
        assert used and all(hb.arms.render_unit(units[h]) in block for h in used)
        assert used == handles[:len(used)]
    question = one_set["questions"][0]
    excerpts = {child["id"]: hb.arms.excerpt_block(child, hb.arms.parse_handles(selections[child["id"]], child),
                                                   chars4)[0] for child in one_set["children"]}
    text = json.dumps(hb.arms.parent_messages("excerpts", one_set, question, excerpts=excerpts))
    assert json.dumps(excerpts[question["child"]])[1:-1] in text


def test_the_consult_prompt_begins_with_the_childs_exact_context_byte_for_byte(hb, one_set):
    by_id = {child["id"]: child for child in one_set["children"]}
    for question in one_set["questions"][:40]:
        for cid in question["children"]:
            child = by_id[cid]
            messages = hb.arms.consult_request(child, question)
            n = len(child["context"])
            assert messages[:n] == child["context"]
            assert json.dumps(messages[:n], ensure_ascii=False).encode() == \
                json.dumps(child["context"], ensure_ascii=False).encode()
            assert len(messages) == n + 1 and question["question"] in messages[-1]["content"]
            body = hb.run.request_body("m", messages, max_tokens=64, salt="s")
            assert "tools" not in body and "tool_choice" not in body
            assert body["messages"] is messages or body["messages"] == messages


def test_the_full_text_arm_sees_all_of_the_material(hb, one_set):
    by_id = {child["id"]: child for child in one_set["children"]}
    cross = next(q for q in one_set["questions"] if q["type"] == "cross_child")
    text = json.dumps(hb.arms.parent_messages("full", one_set, cross))
    for cid in cross["children"]:
        for unit in by_id[cid]["units"]:
            assert json.dumps(unit["text"])[1:-1] in text


def test_every_arm_answers_through_the_same_parent_prompt(hb, one_set):
    summaries, selections = _outputs(hb, one_set)
    question = one_set["questions"][5]
    excerpts = {cid: "EX" for cid in question["children"]}
    consults = {cid: "ANSWER" for cid in question["children"]}
    systems = {arm: hb.arms.parent_messages(arm, one_set, question, summaries=summaries, excerpts=excerpts,
                                            consults=consults)[0]["content"] for arm in hb.arms.ARMS}
    assert len(set(systems.values())) == 1
    for arm in hb.arms.ARMS:
        last = hb.arms.parent_messages(arm, one_set, question, summaries=summaries, excerpts=excerpts,
                                       consults=consults)[-1]["content"]
        assert last.endswith("Question: " + question["question"])


# 6. The scorer.

def test_an_answer_is_right_only_with_the_gold_and_no_distractor(hb):
    correct = hb.score.correct
    assert correct("KX-4471", "KX-4471", [])
    assert correct("The code is kx-4471.", "KX-4471", [])
    assert not correct("KX-44712", "KX-4471", [])
    assert not correct("KX-4471 or KX-1200", "KX-4471", ["KX-1200"])
    assert not correct("unknown", "KX-4471", [])
    assert not correct("", "KX-4471", [])


def _answers(hb, pattern):
    """A synthetic answers file: ``pattern`` maps arm to the list of right (1) and wrong (0) per question."""
    questions, rows = [], []
    n = len(next(iter(pattern.values())))
    types = list(hb.sets.SHARES)
    for i in range(n):
        questions.append({"id": f"q{i}", "type": types[i % 4], "gold": f"GD-{1000 + i}", "distractors": ["ST-1"]})
    for arm, marks in pattern.items():
        for i, mark in enumerate(marks):
            rows.append({"question_id": f"q{i}", "arm": arm, "answer": f"GD-{1000 + i}" if mark else "ST-1",
                         "prompt_tokens": 100 * (i + 1), "cached_tokens": 50 * (i + 1) if arm == "consult" else 0,
                         "wall_s": 0.5})
    return questions, rows


def test_the_scorer_reproduces_known_accuracy_and_interval_values(hb):
    pattern = {"full": [1] * 18 + [0] * 2, "consult": [1] * 17 + [0] * 3,
               "excerpts": [1] * 12 + [0] * 8, "summary": [1] * 6 + [0] * 14}
    questions, rows = _answers(hb, pattern)
    report = hb.score.score(questions, rows, resamples=2000, seed=7)
    arms = report["arms"]
    assert arms["full"]["accuracy"] == 0.9 and arms["consult"]["accuracy"] == 0.85
    assert arms["excerpts"]["accuracy"] == 0.6 and arms["summary"]["accuracy"] == 0.3
    assert arms["full"]["correct"] == 18 and arms["full"]["n"] == 20
    # Types cycle every four questions and "full" misses questions 18 and 19 (a cross-child one and a single one).
    assert arms["full"]["by_type"]["contradiction"] == {"n": 5, "correct": 5, "accuracy": 1.0}
    assert arms["full"]["by_type"]["single"] == {"n": 5, "correct": 4, "accuracy": 0.8}
    assert arms["full"]["by_type"]["cross_child"] == {"n": 5, "correct": 4, "accuracy": 0.8}
    pair = report["pairs"]["consult-summary"]
    assert pair["difference"] == pytest.approx(0.55)
    assert pair["low"] <= 0.55 <= pair["high"]
    assert report == hb.score.score(questions, rows, resamples=2000, seed=7)
    assert arms["consult"]["cost"]["prompt_tokens_mean"] == pytest.approx(1050.0)
    assert arms["consult"]["cost"]["incremental_tokens_mean"] == pytest.approx(525.0)
    assert arms["consult"]["cost"]["cache_hit_share"] == 1.0
    assert arms["full"]["cost"]["cache_hit_share"] == 0.0


def test_a_constant_paired_difference_has_a_degenerate_interval(hb):
    difference, low, high = hb.score.paired_bootstrap([1] * 30, [0] * 30, resamples=500, seed=1)
    assert difference == low == high == 1.0


def test_the_interval_matches_the_normal_approximation_on_a_large_set(hb):
    rng = random.Random(3)
    a = [1 if rng.random() < 0.7 else 0 for _ in range(1000)]
    b = [1 if rng.random() < 0.6 else 0 for _ in range(1000)]
    difference, low, high = hb.score.paired_bootstrap(a, b, resamples=4000, seed=11)
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
    half = 1.96 * sd / math.sqrt(len(diffs))
    assert difference == pytest.approx(mean)
    assert low == pytest.approx(mean - half, abs=0.01) and high == pytest.approx(mean + half, abs=0.01)


def test_the_verdict_applies_the_preregistered_bars(hb):
    pattern = {"full": [1] * 18 + [0] * 2, "consult": [1] * 17 + [0] * 3,
               "excerpts": [1] * 12 + [0] * 8, "summary": [1] * 6 + [0] * 14}
    questions, rows = _answers(hb, pattern)
    verdict = hb.score.score(questions, rows, resamples=200, seed=1)["verdict"]
    assert verdict["consult_within_one_question_of_full"] is True
    assert verdict["consult_points_above_summary"] == pytest.approx(55.0)
    assert verdict["consult_passes"] is True
    assert verdict["excerpts_within_one_question_of_full"] is False
    assert verdict["outcome"] == "branches_earned"
    pattern["excerpts"] = [1] * 17 + [0] * 3
    questions, rows = _answers(hb, pattern)
    assert hb.score.score(questions, rows, resamples=200, seed=1)["verdict"]["outcome"] == "excerpts_suffice"
    pattern["excerpts"], pattern["consult"] = [0] * 20, [1] * 8 + [0] * 12
    questions, rows = _answers(hb, pattern)
    assert hb.score.score(questions, rows, resamples=200, seed=1)["verdict"]["outcome"] == "neither"


def test_a_question_missing_an_arm_is_left_out_of_every_pair(hb):
    pattern = {"full": [1] * 4, "summary": [0] * 4}
    questions, rows = _answers(hb, pattern)
    rows = [row for row in rows if not (row["arm"] == "summary" and row["question_id"] == "q0")]
    report = hb.score.score(questions, rows, resamples=100, seed=1)
    assert report["arms"]["summary"]["n"] == 3 and report["pairs"]["full-summary"]["n"] == 3
    assert report["verdict"]["outcome"] == "incomplete"


def test_no_verdict_is_given_unless_every_arm_answered_every_question_once(hb):
    questions = [{"id": f"q{i}", "type": "single", "gold": f"GD-{1000 + i}", "distractors": []} for i in range(4)]
    # A full-text answer for one question and an excerpt answer for another: no question is common to the pair.
    rows = [{"question_id": "q0", "arm": "full", "answer": "GD-1000"},
            {"question_id": "q1", "arm": "excerpts", "answer": "wrong"}]
    report = hb.score.score(questions, rows, resamples=10, seed=1)
    assert report["pairs"]["full-excerpts"]["n"] == 0
    assert report["verdict"] == {"outcome": "incomplete", "complete": False}
    # Different partial cohorts per pair would otherwise earn branches.
    rows = [{"question_id": "q0", "arm": "full", "answer": "wrong"},
            {"question_id": "q0", "arm": "consult", "answer": "GD-1000"},
            {"question_id": "q1", "arm": "summary", "answer": "wrong"},
            {"question_id": "q1", "arm": "consult", "answer": "GD-1001"}]
    report = hb.score.score(questions, rows, resamples=10, seed=1)
    assert report["verdict"]["outcome"] == "incomplete"
    assert report["coverage"]["arms"]["full"] == {"answered": 1, "missing": 3, "duplicates": 0}
    assert report["coverage"]["arms"]["excerpts"] == {"answered": 0, "missing": 4, "duplicates": 0}
    # Complete coverage gives a verdict; a duplicate row, or a row for a question outside the set, takes it away.
    complete = [{"question_id": f"q{i}", "arm": arm, "answer": f"GD-{1000 + i}" if arm != "summary" else "wrong"}
                for i in range(4) for arm in hb.arms.ARMS]
    assert hb.score.score(questions, complete, resamples=10, seed=1)["verdict"]["outcome"] == "excerpts_suffice"
    duplicated = complete + [dict(complete[0], answer="wrong")]
    report = hb.score.score(questions, duplicated, resamples=10, seed=1)
    assert report["verdict"]["outcome"] == "incomplete"
    assert report["coverage"]["arms"][complete[0]["arm"]]["duplicates"] == 1
    stray = complete + [{"question_id": "q9", "arm": "full", "answer": "GD-1009"}]
    report = hb.score.score(questions, stray, resamples=10, seed=1)
    assert report["verdict"]["outcome"] == "incomplete" and report["coverage"]["unmatched_rows"] == 1


# 7. The runner.

@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "http://[::1]:8000/v1",
                                 "http://127.0.0.2:9/v1"])
def test_the_runner_accepts_a_loopback_base_url(hb, url):
    assert hb.run.check_base_url(url) == url.rstrip("/")


@pytest.mark.parametrize("url", ["http://192.0.2.1:8000/v1", "http://192.0.2.1:8000/v1", "http://example.com/v1",
                                 "http://127.0.0.1.example.com/v1", "https://localhost.example:8000/v1",
                                 "http://0.0.0.0:8000/v1", "ftp://127.0.0.1/v1", "http:///v1", "127.0.0.1:8000"])
def test_the_runner_refuses_any_base_url_that_is_not_loopback(hb, url):
    with pytest.raises(ValueError, match="loopback"):
        hb.run.check_base_url(url)


def test_the_client_refuses_a_remote_url_before_any_request(hb, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("a request was attempted")

    monkeypatch.setattr(hb.run.urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="loopback"):
        hb.run.Client("http://192.0.2.1:8000/v1", "m")


class StubClient:
    """Answers every request of a run: ``done`` as a summary, three handles as a selection, the gold with a handle
    as a consult, the gold as a parent's answer. ``failures`` requests fail first (a reset connection)."""

    def __init__(self, hb, the_set, model="stub", failures=0):
        self.hb, self.the_set, self.model, self.failures = hb, the_set, model, failures
        self.calls = []

    def complete(self, messages, *, max_tokens, salt):
        hb = self.hb
        self.calls.append((messages, salt))
        if len(self.calls) <= self.failures:
            raise ConnectionResetError("reset")
        last = messages[-1]["content"]
        if last == hb.arms.HANDOFF_INSTRUCTION:
            return {"text": "done", "prompt_tokens": 10, "cached_tokens": 0, "completion_tokens": 1}
        if last == hb.arms.EXCERPT_INSTRUCTION:
            child = next(c for c in self.the_set["children"] if messages[:3] == c["context"])
            return {"text": "\n".join(u["handle"] for u in child["units"][:3]), "prompt_tokens": 10,
                    "cached_tokens": 0, "completion_tokens": 1}
        question = next(q for q in self.the_set["questions"] if last.endswith(q["question"]))
        if messages[0]["content"] == hb.arms.PARENT_SYSTEM:
            text = question["gold"] if "Question:" in last else "unknown"
            return {"text": text, "prompt_tokens": 20, "cached_tokens": 0, "completion_tokens": 1}
        return {"text": f"{question['gold']} [x]", "prompt_tokens": 30, "cached_tokens": 25, "completion_tokens": 2}


def _identity(hb, model="stub", revision="rev-a", tokenizer="tok-a"):
    return hb.run.run_identity(model, revision, tokenizer)


def test_the_runner_drives_every_arm_through_a_stub_client_and_the_scorer_reads_its_answers(hb, one_set, tmp_path):
    by_id = {child["id"]: child for child in one_set["children"]}
    questions = one_set["questions"][:6] + [q for q in one_set["questions"] if q["type"] == "cross_child"][:2]
    subset = dict(one_set, questions=questions)
    client = StubClient(hb, subset)
    out = hb.run.run_set(subset, client, chars4, tmp_path, identity=_identity(hb))
    rows = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert {row["arm"] for row in rows} == set(hb.arms.ARMS)
    assert len(rows) == 4 * len(questions)
    consult_calls = [salt for messages, salt in client.calls if messages[0]["content"] != hb.arms.PARENT_SYSTEM
                     and messages[-1]["content"] not in (hb.arms.HANDOFF_INSTRUCTION, hb.arms.EXCERPT_INSTRUCTION)]
    assert len(consult_calls) == sum(len(q["children"]) for q in questions)
    # A consult shares its child's cache salt; every parent call is cold under a salt of its own.
    parent_salts = [salt for messages, salt in client.calls if messages[0]["content"] == hb.arms.PARENT_SYSTEM]
    assert len(parent_salts) == len(set(parent_salts))
    nonce = json.loads((tmp_path / "run.json").read_text())["nonce"]
    assert set(consult_calls) <= {hb.run.child_salt(subset, cid, nonce) for cid in by_id}
    report = hb.score.score(questions, rows, resamples=100, seed=1)
    assert report["arms"]["full"]["accuracy"] == 1.0 and report["verdict"]["complete"] is True
    assert out["answers"] == len(rows)


def test_a_run_records_its_identity_and_resumes_only_under_the_same_one(hb, one_set, tmp_path):
    subset = dict(one_set, questions=one_set["questions"][:2])
    identity = _identity(hb)
    hb.run.run_set(subset, StubClient(hb, subset), chars4, tmp_path, identity=identity)
    stored = json.loads((tmp_path / "run.json").read_text())["identity"]
    assert stored["model"] == "stub" and stored["model_revision"] == "rev-a" and stored["tokenizer"] == "tok-a"
    assert stored["set_sha256"] == hb.sets.set_hash(subset) and stored["set_version"] == hb.sets.VERSION
    assert stored["max_tokens"] == hb.run.MAX_TOKENS and stored["decoding"] == hb.run.DECODING
    assert stored["prompts_sha256"] == hb.run.prompts_fingerprint()
    # The same identity resumes and asks nothing again.
    again = StubClient(hb, subset)
    hb.run.run_set(subset, again, chars4, tmp_path, identity=identity)
    assert again.calls == []
    before = (tmp_path / "answers.jsonl").read_text()
    changed = [_identity(hb, revision="rev-b"), _identity(hb, tokenizer="tok-b"), _identity(hb, model="other"),
               dict(identity, max_tokens=dict(identity["max_tokens"], parent=32)),
               dict(identity, decoding=dict(identity["decoding"], temperature=0.7)),
               dict(identity, prompts_sha256="0" * 64)]
    for other in changed:
        client = StubClient(hb, subset, model=other["model"])
        with pytest.raises(ValueError, match="identity"):
            hb.run.run_set(subset, client, chars4, tmp_path, identity=other)
        assert client.calls == []
    # A client serving another model than the identity names is refused before any request.
    client = StubClient(hb, subset, model="other")
    with pytest.raises(ValueError, match="model"):
        hb.run.run_set(subset, client, chars4, tmp_path, identity=identity)
    assert client.calls == []
    # Another set (or another version of it) in the same directory.
    other_set = dict(one_set, questions=one_set["questions"][2:4])
    client = StubClient(hb, other_set)
    with pytest.raises(ValueError, match="identity"):
        hb.run.run_set(other_set, client, chars4, tmp_path, identity=identity)
    assert client.calls == [] and (tmp_path / "answers.jsonl").read_text() == before
    # Outputs with no identity beside them are never resumed.
    for name in ("answers.jsonl", "consults.jsonl", "handbacks.json"):
        bare = tmp_path / f"bare-{name}"
        bare.mkdir()
        (bare / name).write_text((tmp_path / name).read_text())
        client = StubClient(hb, subset)
        with pytest.raises(ValueError, match="identity"):
            hb.run.run_set(subset, client, chars4, bare, identity=identity)
        assert client.calls == [] and not (bare / "run.json").exists()


def _is_parent(messages, hb):
    return messages[0]["content"] == hb.arms.PARENT_SYSTEM


def test_runs_into_fresh_directories_against_one_server_never_share_a_cache_salt(hb, one_set, tmp_path):
    subset = dict(one_set, questions=one_set["questions"][:3])
    first, second = StubClient(hb, subset), StubClient(hb, subset)
    hb.run.run_set(subset, first, chars4, tmp_path / "a", identity=_identity(hb))
    hb.run.run_set(subset, second, chars4, tmp_path / "b", identity=_identity(hb))
    salts = [{salt for _, salt in client.calls} for client in (first, second)]
    assert salts[0] and salts[1] and not salts[0] & salts[1]
    nonce = json.loads((tmp_path / "a" / "run.json").read_text())["nonce"]
    assert len(nonce) >= 16 and all(nonce in salt for salt in salts[0])


def test_a_resumed_run_keeps_its_childs_salt_and_asks_a_parent_again_under_a_fresh_one(hb, one_set, tmp_path):
    subset = dict(one_set, questions=one_set["questions"][:2])
    first = StubClient(hb, subset)
    hb.run.run_set(subset, first, chars4, tmp_path, identity=_identity(hb))
    # The last parent request was in flight at a stop: its row was never written.
    answers = tmp_path / "answers.jsonl"
    lines = answers.read_text().splitlines(keepends=True)
    answers.write_text("".join(lines[:-1]))
    (tmp_path / "consults.jsonl").write_text("")
    again = StubClient(hb, subset)
    hb.run.run_set(subset, again, chars4, tmp_path, identity=_identity(hb))
    child_salts = {salt for messages, salt in first.calls if not _is_parent(messages, hb)}
    assert {salt for messages, salt in again.calls if not _is_parent(messages, hb)} <= child_salts
    asked_again = [salt for messages, salt in again.calls if _is_parent(messages, hb)]
    assert len(asked_again) == 1
    assert asked_again[0] not in {salt for messages, salt in first.calls if _is_parent(messages, hb)}


def test_a_parent_retry_draws_a_fresh_salt_and_a_childs_retry_keeps_its_own(hb, one_set, tmp_path, monkeypatch):
    monkeypatch.setattr(hb.run.time, "sleep", lambda seconds: None)
    subset = dict(one_set, questions=one_set["questions"][:1])

    class FailsOnceEach(StubClient):
        """Every request fails on its first try (a reset connection) and succeeds on the next."""

        def complete(self, messages, *, max_tokens, salt):
            key = json.dumps(messages)
            self.seen = getattr(self, "seen", set())
            if key not in self.seen:
                self.seen.add(key)
                self.calls.append((messages, salt))
                raise ConnectionResetError("reset")
            return super().complete(messages, max_tokens=max_tokens, salt=salt)

    client = FailsOnceEach(hb, subset)
    hb.run.run_set(subset, client, chars4, tmp_path, identity=_identity(hb))
    tries = {}
    for messages, salt in client.calls:
        tries.setdefault(json.dumps(messages), []).append((salt, _is_parent(messages, hb)))
    assert tries and all(len(pair) == 2 for pair in tries.values())
    for pair in tries.values():
        (a, parent), (b, _) = pair
        assert (a != b) if parent else (a == b)
    parents = [salt for messages, salt in client.calls if _is_parent(messages, hb)]
    assert len(parents) == 2 * len(hb.arms.ARMS) and len(set(parents)) == len(parents)


def test_the_identity_covers_every_prompt_the_run_sends(hb, monkeypatch):
    base = hb.run.prompts_fingerprint()
    for name in ("HANDOFF_INSTRUCTION", "EXCERPT_INSTRUCTION", "CONSULT_INSTRUCTION", "PARENT_SYSTEM"):
        with monkeypatch.context() as patch:
            patch.setattr(hb.arms, name, getattr(hb.arms, name) + " Changed.")
            assert hb.run.prompts_fingerprint() != base
    with monkeypatch.context() as patch:
        real = hb.arms.parent_messages

        def reworded(*args, **kwargs):
            messages = real(*args, **kwargs)
            return messages[:-1] + [dict(messages[-1], content=messages[-1]["content"].replace("handed back", "sent"))]

        patch.setattr(hb.arms, "parent_messages", reworded)
        assert hb.run.prompts_fingerprint() != base
    assert hb.run.prompts_fingerprint() == base


def test_a_tokenizer_is_identified_by_its_files(hb, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for folder in (first, second):
        folder.mkdir()
        (folder / "tokenizer_config.json").write_text("{}")
        (folder / "tokenizer.json").write_text('{"v": 1}')
        (folder / "model.safetensors").write_text("weights")
    assert hb.run.tokenizer_fingerprint(first) == hb.run.tokenizer_fingerprint(second)
    (second / "model.safetensors").write_text("other weights")
    assert hb.run.tokenizer_fingerprint(first) == hb.run.tokenizer_fingerprint(second)
    (second / "tokenizer.json").write_text('{"v": 2}')
    assert hb.run.tokenizer_fingerprint(first) != hb.run.tokenizer_fingerprint(second)
    with pytest.raises(ValueError, match="tokenizer"):
        hb.run.tokenizer_fingerprint(tmp_path / "missing")


def test_the_run_manifest_is_written_once_and_a_start_under_another_identity_is_refused(hb, tmp_path):
    identity = _identity(hb)
    hb.run.record_start(tmp_path, identity, [1301])
    hb.run.record_start(tmp_path, identity, [1302])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["identity"] == identity and [s["seeds"] for s in manifest["starts"]] == [[1301], [1302]]
    with pytest.raises(ValueError, match="identity"):
        hb.run.record_start(tmp_path, _identity(hb, revision="rev-b"), [1301])
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_a_failing_request_is_tried_again_and_after_its_tries_it_is_an_error_scored_wrong(hb, monkeypatch):
    monkeypatch.setattr(hb.run.time, "sleep", lambda seconds: None)

    class Flaky:
        def __init__(self, failures):
            self.failures, self.calls = failures, 0

        def complete(self, messages, *, max_tokens, salt):
            self.calls += 1
            if self.calls <= self.failures:
                raise ConnectionResetError("reset")
            return {"text": "GD-1000", "prompt_tokens": 5, "cached_tokens": 0, "completion_tokens": 1}

    once = Flaky(1)
    assert hb.run._timed(once, [], max_tokens=4, salt="s")["text"] == "GD-1000" and once.calls == 2
    never = Flaky(99)
    result = hb.run._timed(never, [], max_tokens=4, salt="s")
    assert never.calls == hb.run.TRIES and result["text"] == "" and result["error"] == "ConnectionResetError"
    assert not hb.score.correct(result["text"], "GD-1000", [])
