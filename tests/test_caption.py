"""Tests for keyframe captioning: caching, resume, retries and the structured-output request."""

import json
from pathlib import Path

import httpx
import pytest

from frameseek.caption import CAPTION_SCHEMA, caption_keyframes, cache_key
from frameseek.config import CaptionConfig, VisionConfig
from frameseek.models.vision import OllamaError, OllamaVision, Timing
from frameseek.shots import Keyframe, Shot

CAPTION = {"summary": "A red car on a street", "shot": "wide",
           "people": {"count": 0, "description": ""},
           "objects": [{"name": "car", "colour": "red"}], "text_on_screen": [],
           "setting": "outdoor, day", "action": "parked"}
TIMING = Timing(load_s=0, prefill_s=4.0, prefill_tokens=400, decode_s=3.0, decode_tokens=50, total_s=7.0)
MODEL = "qwen3-vl:8b-instruct"


class FakeVision:
    def __init__(self, fail_on: set[str] = frozenset()):
        self.calls: list[str] = []
        self.fail_on = fail_on

    def structured(self, image, prompt, schema, max_side=None, num_predict=400):
        self.calls.append(image.name)
        if image.name in self.fail_on:
            raise OllamaError("cut off")
        return CAPTION, TIMING


def _shots() -> list[Shot]:
    return [Shot(0, 0.0, 4.0, [Keyframe(2.0, "kf_0000_00.jpg")]),
            Shot(1, 4.0, 30.0, [Keyframe(8.0, "kf_0001_00.jpg"), Keyframe(20.0, "kf_0001_01.jpg")])]


def test_captions_every_keyframe_once(tmp_path: Path) -> None:
    fake = FakeVision()
    res = caption_keyframes(tmp_path, _shots(), fake, MODEL, CaptionConfig())
    assert (res.newly_done, res.from_cache, res.errors) == (3, 0, 0)
    assert [r["file"] for r in res.records] == ["kf_0000_00.jpg", "kf_0001_00.jpg", "kf_0001_01.jpg"]

    again = FakeVision()
    res = caption_keyframes(tmp_path, _shots(), again, MODEL, CaptionConfig())
    assert again.calls == [] and res.from_cache == 3


def test_resumes_after_a_crash(tmp_path: Path) -> None:
    caption_keyframes(tmp_path, _shots()[:1], FakeVision(), MODEL, CaptionConfig())   # only shot 0 done
    with (tmp_path / "captions.jsonl").open("a") as f:
        f.write('{"file": "kf_0001_00.jpg", "capt')                                  # half-written line
    fake = FakeVision()
    res = caption_keyframes(tmp_path, _shots(), fake, MODEL, CaptionConfig())
    assert fake.calls == ["kf_0001_00.jpg", "kf_0001_01.jpg"]
    assert len(res.records) == 3


def test_changing_image_size_recaptions(tmp_path: Path) -> None:
    caption_keyframes(tmp_path, _shots(), FakeVision(), MODEL, CaptionConfig(max_image_side=640))
    fake = FakeVision()
    caption_keyframes(tmp_path, _shots(), fake, MODEL, CaptionConfig(max_image_side=512))
    assert len(fake.calls) == 3
    assert cache_key(MODEL, 640) != cache_key(MODEL, 512)


def test_errors_are_retried_next_run(tmp_path: Path) -> None:
    res = caption_keyframes(tmp_path, _shots(), FakeVision(fail_on={"kf_0001_00.jpg"}), MODEL, CaptionConfig())
    assert (res.newly_done, res.errors) == (2, 1)
    fake = FakeVision()
    res = caption_keyframes(tmp_path, _shots(), fake, MODEL, CaptionConfig())
    assert fake.calls == ["kf_0001_00.jpg"] and res.errors == 0 and len(res.records) == 3


def test_force_starts_over(tmp_path: Path) -> None:
    caption_keyframes(tmp_path, _shots(), FakeVision(), MODEL, CaptionConfig())
    fake = FakeVision()
    caption_keyframes(tmp_path, _shots(), fake, MODEL, CaptionConfig(), force=True)
    assert len(fake.calls) == 3


# ---------- the request Ollama receives ----------

def _ollama(content: str, seen: dict) -> OllamaVision:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": content}, "done": True,
                                         "prompt_eval_count": 410, "eval_count": 55,
                                         "prompt_eval_duration": 4_000_000_000, "eval_duration": 3_000_000_000})
    cfg = VisionConfig()
    return OllamaVision(cfg, client=httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(handler)))


def _image(tmp_path: Path) -> Path:
    from PIL import Image
    p = tmp_path / "f.jpg"
    Image.new("RGB", (1280, 540), "red").save(p)
    return p


def test_schema_is_sent_for_constrained_decoding(tmp_path: Path) -> None:
    seen: dict = {}
    cap, t = _ollama(json.dumps(CAPTION), seen).structured(_image(tmp_path), "describe", CAPTION_SCHEMA, max_side=512)
    assert seen["format"] == CAPTION_SCHEMA
    assert seen["options"]["temperature"] == 0
    assert cap["objects"][0]["colour"] == "red"
    assert (t.prefill_tokens, t.decode_tokens) == (410, 55)


def test_cut_off_json_raises(tmp_path: Path) -> None:
    with pytest.raises(OllamaError, match="cut off"):
        _ollama('{"summary": "A red', {}).structured(_image(tmp_path), "describe", CAPTION_SCHEMA)
