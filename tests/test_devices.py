import pytest

from frameseek.devices import pick_device


def test_auto_returns_a_real_device() -> None:
    assert pick_device("auto") in {"cuda", "mps", "cpu"}


def test_cpu_always_available() -> None:
    assert pick_device("cpu") == "cpu"


def test_rejects_unknown_device() -> None:
    with pytest.raises(ValueError):
        pick_device("tpu")
