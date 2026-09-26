import pytest

from income_pipeline.__main__ import main
from income_pipeline.score import main as score_main
from income_pipeline.train import main as train_main


def test_pipeline_help_lists_the_commands(capsys):
    main([])
    captured = capsys.readouterr()
    assert "generate" in captured.out
    assert "train" in captured.out
    assert "score" in captured.out


def test_unknown_command_exits():
    with pytest.raises(SystemExit) as caught:
        main(["nope"])
    assert caught.value.code == 2


def test_train_rejects_a_missing_file():
    with pytest.raises(SystemExit) as caught:
        train_main(["--input", "data/raw/does-not-exist.csv"])
    assert caught.value.code == 1


def test_score_rejects_a_missing_model():
    with pytest.raises(SystemExit) as caught:
        score_main(["--model-dir", "models/does-not-exist"])
    assert caught.value.code == 1
