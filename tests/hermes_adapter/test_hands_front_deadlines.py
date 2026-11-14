"""The tool-work clock counts each tool interval once and no model wait.

A long owner turn gets no front action: no yield line, no interrupt, no continuation, no session breaker.
Only a hand's result review still gets the yield line, after ten seconds of tool work.
"""

import logging
from types import SimpleNamespace as NS

import pytest

from test_hands_front import begin_tool, end_tool, owner_turn, review_turn, rig  # noqa: F401


def no_action(rig):
    assert rig.agent.redirects == []
    assert rig.agent.steers == []
    assert rig.agent.interrupts == 0
    assert rig.ctx.injected == []


def long_tool(rig, turn='turn-1', ident='message-1'):
    owner_turn(rig, turn=turn, ident=ident)
    begin_tool(rig, turn=turn)
    rig.clock.now += 100
    rig.guard.tick('owner-session', turn)
    end_tool(rig, turn=turn)


def test_150_second_model_without_tools_finishes_once_without_front_action(rig):
    owner_turn(rig)
    for now in (10, 30, 90, 149.9, 150):
        rig.clock.now = now
        rig.guard.tick('owner-session', 'turn-1')
        no_action(rig)
    answer = 'The owner receives the completed answer.'
    rig.guard.post_api_request(session_id='owner-session', turn_id='turn-1',
                               assistant_message=NS(content=answer))
    assert rig.guard.finalize_response(answer, session_id='owner-session', turn_id='turn-1') is None
    rig.guard.post_llm_call(session_id='owner-session', turn_id='turn-1', assistant_response=answer)
    rig.guard.on_native_turn_settled(session_id='owner-session', turn_id='turn-1',
        outcome={'completed': True, 'failed': False, 'interrupted': False})
    rig.clock.now = 300
    rig.guard.tick('owner-session', 'turn-1')
    no_action(rig)


def test_model_waits_between_tools_do_not_count_as_tool_work(rig):
    owner_turn(rig)
    state = rig.guard._state('owner-session', 'turn-1')
    rig.clock.now = 60
    begin_tool(rig, 'call-1')
    rig.clock.now = 80
    end_tool(rig, 'call-1')
    no_action(rig)
    rig.clock.now = 230
    rig.guard.tick('owner-session', 'turn-1')
    no_action(rig)
    begin_tool(rig, 'call-2')
    rig.clock.now = 240
    end_tool(rig, 'call-2')
    assert state.work_seconds == 30
    rig.clock.now = 390
    rig.guard.tick('owner-session', 'turn-1')
    assert rig.agent.interrupts == 0
    begin_tool(rig, 'call-3')
    rig.clock.now = 450
    end_tool(rig, 'call-3')
    assert state.work_seconds == 90
    no_action(rig)


def test_overlapping_tool_intervals_count_once_and_wait_for_last_completion(rig):
    owner_turn(rig)
    begin_tool(rig, 'call-1')
    rig.clock.now = 10
    begin_tool(rig, 'call-2')
    rig.clock.now = 20
    end_tool(rig, 'call-1')
    no_action(rig)
    rig.clock.now = 25
    end_tool(rig, 'call-2')
    no_action(rig)  # 25 seconds elapsed, though summed durations would incorrectly be 35.
    begin_tool(rig, 'call-3')
    rig.clock.now = 30
    end_tool(rig, 'call-3')
    assert rig.guard._state('owner-session', 'turn-1').work_seconds == 30
    assert rig.agent.steers == []
    assert not rig.agent.redirects and rig.agent.interrupts == 0


def test_review_result_uses_ten_seconds_of_tool_work(rig):
    review_turn(rig)
    rig.clock.now = 150
    rig.guard.tick('owner-session', 'turn-1')
    no_action(rig)
    begin_tool(rig)
    rig.clock.now = 159.9
    rig.guard.tick('owner-session', 'turn-1')
    no_action(rig)
    rig.clock.now = 160
    end_tool(rig)
    assert rig.agent.steers == [rig.front.YIELD_LINE]


def test_repeated_long_owner_turns_trip_no_session_breaker(rig, caplog, tmp_path):
    rig.runtime.spool_dir = tmp_path
    with caplog.at_level(logging.WARNING):
        long_tool(rig)
        long_tool(rig, 'turn-2', 'message-2')
        long_tool(rig, 'turn-3', 'message-3')
        long_tool(rig, 'turn-4', 'message-4')
        no_action(rig)
        assert 'deadline_interrupt' not in rig.guard.metrics
        assert 'deadline_breaker_tripped' not in rig.guard.metrics
        assert len([row for row in caplog.records if row.levelno == logging.WARNING]) == 0
        # With no breaker there is nothing to reset and no status to publish.
        assert not (tmp_path / 'front-deadlines.json').exists()


def test_faculty_off_mid_turn_stops_every_front_action_within_one_tick(rig):
    live = {'mind': {'faculties': {'hands': True}}}
    rig.runtime.config_loader = lambda: live
    review_turn(rig)
    begin_tool(rig)
    rig.clock.now = 20
    rig.guard.tick('owner-session', 'turn-1')
    live['mind']['faculties']['hands'] = False
    rig.clock.now += 0.1
    rig.guard.tick('owner-session', 'turn-1')
    rig.clock.now = 200
    end_tool(rig)
    rig.guard.tick('owner-session', 'turn-1')
    rig.guard.on_gateway_turn_settled(session_id='owner-session', session_key='route',
                                     run_generation=1, gateway=rig.gateway)
    no_action(rig)


def test_late_tool_callback_never_interrupts_a_model_request_that_already_started(rig):
    import threading
    rig.agent._pending_redirect_lock = threading.Lock()
    rig.agent._model_request_active = threading.Event()
    review_turn(rig)
    begin_tool(rig)
    rig.clock.now = 100
    # Hermes can time out a slow lifecycle hook. The next request wins the bracket lock.
    rig.agent._model_request_active.set()
    end_tool(rig)
    rig.clock.now += 150
    rig.guard.tick('owner-session', 'turn-1')
    no_action(rig)


def test_duplicate_tool_completion_cannot_repeat_guidance(rig):
    review_turn(rig)
    begin_tool(rig)
    rig.clock.now = 30
    end_tool(rig)
    end_tool(rig)
    assert rig.agent.steers == [rig.front.YIELD_LINE]
    begin_tool(rig, 'call-2')
    rig.clock.now = 100
    end_tool(rig, 'call-2')
    end_tool(rig, 'call-2')
    assert rig.agent.steers == [rig.front.YIELD_LINE]
    assert rig.agent.interrupts == 0 and len(rig.ctx.injected) == 0


def test_review_request_finishes_normally_after_late_tool_completion(rig):
    import threading
    review_turn(rig)
    begin_tool(rig)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    answers, errors = [], []
    def native_request(request):
        entered.set()
        assert release.wait(2)
        return 'Slow model answer'
    def request():
        try:
            answers.append(rig.guard.llm_execution({}, native_request,
                session_id='owner-session', turn_id='turn-1'))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()
    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    try:
        assert entered.wait(1)
        rig.clock.now = 150
        end_tool(rig)
        rig.guard.tick('owner-session', 'turn-1')
        no_action(rig)
        assert not finished.is_set()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and errors == []
    assert answers == ['Slow model answer']


def test_provider_entry_closes_abandoned_tool_interval_before_late_post_callback(rig):
    owner_turn(rig)
    begin_tool(rig, 'abandoned')
    rig.clock.now = 20
    state = rig.guard._state('owner-session', 'turn-1')
    def provider(request):
        rig.clock.now = 150
        end_tool(rig, 'abandoned')
        no_action(rig)
        assert state.work_seconds == 20
        assert state.active_tools == set()
        return 'Model returned'
    assert rig.guard.llm_execution({}, provider,
        session_id='owner-session', turn_id='turn-1') == 'Model returned'
    begin_tool(rig, 'fresh')
    rig.clock.now = 160
    end_tool(rig, 'fresh')
    assert state.work_seconds == 30
    assert rig.agent.steers == []
    assert rig.agent.redirects == [] and rig.agent.interrupts == 0 and rig.ctx.injected == []


def test_late_pre_tool_callback_during_provider_wait_cannot_restart_work_clock(rig):
    owner_turn(rig)
    begin_tool(rig, 'abandoned')
    rig.clock.now = 20
    state = rig.guard._state('owner-session', 'turn-1')
    def provider(request):
        rig.clock.now = 150
        begin_tool(rig, 'late-pre')
        assert state.active_tools == set()
        assert state.work_seconds == 20
        rig.clock.now = 170
        end_tool(rig, 'late-pre')
        end_tool(rig, 'abandoned')
        no_action(rig)
        assert state.work_seconds == 20
        return 'Model returned'
    assert rig.guard.llm_execution({}, provider,
        session_id='owner-session', turn_id='turn-1') == 'Model returned'
    begin_tool(rig, 'fresh')
    rig.clock.now = 180
    end_tool(rig, 'fresh')
    assert state.work_seconds == 30
    assert rig.agent.steers == []
    assert rig.agent.redirects == [] and rig.agent.interrupts == 0 and rig.ctx.injected == []


def test_deadline_lines_ask_for_the_owners_reply_not_a_report_on_the_cut(rig):
    """After a deadline at a tool boundary the front reported that its reply was cut off.

    The review yield asks for the reply Owner asked for, from finished results, and hands off only unfinished work.
    """
    line = rig.front.YIELD_LINE
    assert '\n' not in line
    assert ', not from Owner]' in line and 'in the form Owner asked for' in line
    assert 'hand only unfinished work to a hand' in line.lower()
    assert 'what ran' not in line and 'what you have now' not in line
    assert line.startswith('[runtime deadline, not from Owner]')
