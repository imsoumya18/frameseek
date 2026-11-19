"""Tests for the Ollama client using a fake HTTP server (no real model needed)."""

import json
from pathlib import Path

import httpx
from PIL import Image

from frameseek.config import VisionConfig
from frameseek.models.vision import Done, OllamaError, OllamaVision, Status, Token, encode_image


def _make_image(tmp_path: Path, size=(2000, 1000)) -> Path:
    p = tmp_path / "frame.jpg"
    Image.new("RGB", size, "red").save(p)
    return p


def test_encode_image_downscales_keeping_aspect(tmp_path: Path) -> None:
    _, original, new = encode_image(_make_image(tmp_path), max_side=1024)
    assert original == (2000, 1000)
    assert new == (1024, 512)


def _fake_ollama(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    if request.url.path == "/api/chat":
        chunks = [
            {"message": {"content": "A red "}, "done": False},
            {"message": {"content": "square."}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "load_duration": 2_000_000_000,
                "prompt_eval_count": 300,
                "prompt_eval_duration": 1_500_000_000,
                "eval_count": 4,
                "eval_duration": 500_000_000,
                "total_duration": 4_000_000_000,
            },
        ]
        body = "\n".join(json.dumps(c) for c in chunks) + "\n"
        return httpx.Response(200, content=body.encode())
    return httpx.Response(404)


def _model() -> OllamaVision:
    cfg = VisionConfig()
    client = httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(_fake_ollama))
    return OllamaVision(cfg, client=client)


def test_stream_yields_status_tokens_then_timing(tmp_path: Path) -> None:
    events = list(_model().stream(_make_image(tmp_path), "describe"))

    assert isinstance(events[0], Status)
    assert any(isinstance(e, Status) and "Loading" in e.message for e in events)
    assert "".join(e.text for e in events if isinstance(e, Token)) == "A red square."

    done = events[-1]
    assert isinstance(done, Done)
    assert done.timing.load_s == 2.0
    assert done.timing.prefill_tokens == 300
    assert done.timing.decode_tok_per_s == 8.0


def test_describe_joins_tokens(tmp_path: Path) -> None:
    assert _model().describe(_make_image(tmp_path), "describe") == "A red square."


def _ps(models):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": models})
    cfg = VisionConfig()
    return OllamaVision(cfg, client=httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(handler)))


def test_residency_reports_gpu_share() -> None:
    m = _ps([{"name": "qwen3-vl:8b-instruct", "size": 6_000, "size_vram": 6_000}])
    assert m.residency() == (6_000, 6_000)


def test_residency_none_when_not_loaded() -> None:
    assert _ps([]).residency() is None


def test_unload_sends_keep_alive_zero() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(path=request.url.path, body=json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    cfg = VisionConfig(model="qwen3-vl:4b-instruct")
    OllamaVision(cfg, client=httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(handler))).unload()
    assert seen == {"path": "/api/generate", "body": {"model": "qwen3-vl:4b-instruct", "keep_alive": 0}}


def test_cut_off_json_is_retried_once_with_some_randomness() -> None:
    temps = []
    looped = '{"summary": "book cover", "objects": [{"name": "book", "colour": "' + " " * 50
    good = '{"summary": "book cover"}'

    def handler(request: httpx.Request) -> httpx.Response:
        temps.append(json.loads(request.content)["options"]["temperature"])
        content = looped if len(temps) == 1 else good
        return httpx.Response(200, json={"message": {"content": content}, "done": True})

    cfg = VisionConfig(model="qwen3-vl:4b-instruct")
    vm = OllamaVision(cfg, client=httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(handler)))
    out, _ = vm.structured(None, "describe", {"type": "object"})
    assert out == {"summary": "book cover"} and temps == [0.0, 0.4]


def test_cut_off_twice_raises_with_the_tail_shown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": '{"a": "' + "xy" * 100}, "done": True})

    cfg = VisionConfig()
    vm = OllamaVision(cfg, client=httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(handler)))
    try:
        vm.structured(None, "p", {"type": "object"})
        raise AssertionError("expected OllamaError")
    except OllamaError as exc:
        assert "twice" in str(exc) and "xyxy'" in str(exc)
