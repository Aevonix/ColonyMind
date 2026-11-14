"""The named report uses the faculty's existing plasticity presentation."""
import argparse
import json

from colonymind.reports import cli, plasticity


def test_plasticity_command_text_and_json(monkeypatch, capsys):
    collected = {"enabled": True, "learners": [], "rejected": {"example": "unavailable"}}
    monkeypatch.setattr("colonymind.config.load_config", lambda: object())
    monkeypatch.setattr(plasticity, "collect", lambda config: collected)
    parser = argparse.ArgumentParser()
    cli.add_parser(parser.add_subparsers())
    for flags in ([], ["--json"]):
        args = parser.parse_args(["report", "plasticity", *flags])
        assert cli.run(args) == 0
        text = capsys.readouterr().out
        if flags:
            assert json.loads(text) == collected
            assert text == plasticity.to_json(collected)
        else:
            for line in plasticity.lines(collected):
                assert line in text


def test_plasticity_command_reads_selected_instance_without_writes(tmp_path, monkeypatch, capsys):
    from colonymind.config import Config
    config = Config(tmp_path, {"mind": {"faculties": {"plasticity": True}}, "plasticity": {"learners": {}}})
    monkeypatch.setattr("colonymind.config.load_config", lambda: config)
    before = set(tmp_path.rglob("*"))
    parser = argparse.ArgumentParser()
    cli.add_parser(parser.add_subparsers())
    assert cli.run(parser.parse_args(["report", "plasticity", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["learners"] == []
    assert set(tmp_path.rglob("*")) == before
