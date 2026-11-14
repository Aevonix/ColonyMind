"""The memory provider's CLI (``hermes colonymind-memory ...``) offers only verbs whose routes the sidecar serves."""

import argparse


def test_the_memory_cli_offers_no_goals_verb():
    """The legacy goal records and their route are gone; the mind's goals are read with ``colonymind mind``."""
    from colonymind_memory.cli import app, register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser)
    verbs, = [action.choices for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    assert set(verbs) == {"status", "context", "sync"}
    assert "goals" not in {command.name or command.callback.__name__ for command in app.registered_commands}
