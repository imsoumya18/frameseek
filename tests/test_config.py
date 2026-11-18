from pathlib import Path

import pytest

from frameseek.config import load_config


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.vision.backend == "ollama"
    assert cfg.device == "auto"


def test_reads_yaml(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text("device: cpu\nvision:\n  model: qwen3-vl:4b-instruct\n")
    cfg = load_config(f)
    assert cfg.device == "cpu"
    assert cfg.vision.model == "qwen3-vl:4b-instruct"


def test_unknown_key_fails_loudly(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text("vision:\n  modle: typo\n")
    with pytest.raises(ValueError, match="modle"):
        load_config(f)
