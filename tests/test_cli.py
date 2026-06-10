import json

import pytest

from src.cli import build_parser, main


def test_subject_flags_parse():
    args = build_parser().parse_args(
        ["-sessions", "10", "-save", "models/10sess.txt",
         "-visual", "off"])
    assert args.sessions == 10
    assert args.save == "models/10sess.txt"
    assert args.visual == "off"


def test_subject_eval_flags_parse():
    args = build_parser().parse_args(
        ["-visual", "on", "-load", "models/100sess.txt",
         "-sessions", "10", "-dontlearn", "-step-by-step"])
    assert args.load == "models/100sess.txt"
    assert args.dontlearn is True
    assert args.step_by_step is True
    assert args.visual == "on"


def test_defaults():
    args = build_parser().parse_args([])
    assert args.sessions == 1
    assert args.visual == "on"
    assert args.board_size == 10
    assert args.dontlearn is False


def test_invalid_sessions_exits():
    with pytest.raises(SystemExit):
        main(["-sessions", "0", "-visual", "off"])


def test_train_and_save_then_load_and_eval(tmp_path):
    model = str(tmp_path / "m.txt")
    code = main(["-sessions", "3", "-visual", "off", "-quiet",
                 "-seed", "7", "-save", model])
    assert code == 0
    with open(model, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["trained_sessions"] == 3
    code = main(["-sessions", "2", "-visual", "off", "-quiet",
                 "-seed", "7", "-load", model, "-dontlearn"])
    assert code == 0


def test_missing_model_file_is_graceful_error(capsys):
    code = main(["-load", "/nonexistent/model.txt", "-visual", "off"])
    assert code == 1
    assert "error" in capsys.readouterr().err
