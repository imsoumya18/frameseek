"""Tests for z-score fusion, moments and answering."""

import json
from pathlib import Path

import httpx
import numpy as np

from frameseek.answer import ANSWER_SCHEMA, answer_question, build_prompt, fuse, zscore
from frameseek.config import AnswerConfig, VisionConfig
from frameseek.models.vision import OllamaVision, Timing


def fr(file, t, summary="x"):
    return {"file": file, "t": t, "summary": summary, "text": summary + ".", "shot": 0, "has_text": True}


FRAMES = [fr("kf_0006_00.jpg", 32.9), fr("kf_0010_00.jpg", 47.5), fr("kf_0012_00.jpg", 54.2),
          fr("kf_0013_00.jpg", 58.8, "Man and woman carry bowls"), fr("kf_0014_00.jpg", 61.9),
          fr("kf_0017_00.jpg", 68.0), fr("kf_0019_00.jpg", 72.2, "Two women laughing with friends"),
          fr("kf_0022_00.jpg", 88.3), fr("kf_0063_00.jpg", 193.5)]
SPEECH = [{"start": 11.3, "end": 25.0, "text": "ভায়, তো রন্দুর"}, {"start": 55.6, "end": 70.0, "text": "এসেছে ঢাক পেটাতে"}]

def scores(cap, frm, sp):
    return {"captions": np.array(cap, float), "frames": np.array(frm, float), "speech": np.array(sp, float)}


# Shaped like the real "guy in a pink cap" scores: captions match "pink" bowls
# (0013), pixels clearly prefer the pink-cap frames (0019, 0014), speech is flat.
PINK_CAP = scores(
    cap=[0.78, 0.77, 0.78, 0.80, 0.77, 0.77, 0.78, 0.77, 0.80],
    frm=[0.05, 0.04, 0.07, 0.08, 0.12, 0.08, 0.15, 0.08, 0.04],
    sp=[0.76, 0.75],
)


def test_zscore_puts_sources_on_one_scale() -> None:
    a, b = zscore([0.1, 0.2, 0.3]), zscore([0.7, 0.8, 0.9])   # SigLIP-like vs e5-like
    assert np.allclose(a, b) and abs(a.mean()) < 1e-9
    assert np.all(zscore([0.5, 0.5]) == 0)                    # no spread: no opinion


def test_clear_pixel_match_beats_a_caption_word_match() -> None:
    m = fuse(FRAMES, SPEECH, PINK_CAP)
    assert m[0].file == "kf_0019_00.jpg"
    assert m[0].parts["frames"] > m[0].parts["captions"]
    assert len(m) == len(FRAMES)                               # every keyframe gets a score


def test_frame_inherits_the_speech_chunk_playing_at_its_time() -> None:
    m = {x.file: x for x in fuse(FRAMES, SPEECH, PINK_CAP)}
    assert m["kf_0013_00.jpg"].speech == "এসেছে ঢাক পেটাতে"     # 58.8 s is inside 55.6-70.0
    assert m["kf_0022_00.jpg"].speech == "" and m["kf_0022_00.jpg"].parts["speech"] == 0.0


def test_lyric_match_lifts_frames_in_that_chunk() -> None:
    flat = [0.5] * len(FRAMES)
    m = fuse(FRAMES, SPEECH, scores(flat, flat, [0.70, 0.90]))
    assert {x.file for x in m[:3]} == {"kf_0013_00.jpg", "kf_0014_00.jpg", "kf_0017_00.jpg"}


def test_empty_captions_never_match() -> None:
    cap = [np.nan] + [0.5] * (len(FRAMES) - 1)
    cap[1] = 0.6
    m = {x.file: x for x in fuse(FRAMES, SPEECH, scores(cap, [0.1] * len(FRAMES), [0.5, 0.5]))}
    assert m["kf_0006_00.jpg"].parts["captions"] == min(x.parts["captions"] for x in m.values())


def test_missing_speech_table_still_ranks() -> None:
    m = fuse(FRAMES, [], {"captions": PINK_CAP["captions"], "frames": PINK_CAP["frames"]})
    assert "speech" not in m[0].parts and m[0].file == "kf_0019_00.jpg"


def test_prompt_numbers_moments_and_images() -> None:
    moments = fuse(FRAMES, SPEECH, PINK_CAP)[:3]
    prompt = build_prompt("guy in a pink cap", moments)
    assert "Moment 1 at" in prompt and "(image 3)" in prompt and "0 if none" in prompt
    assert "speech text is the evidence" in prompt          # lyric questions: transcript counts
    assert "must really be in that moment's text" in prompt  # ...but no quoting what isn't there
    assert "Do not mention moment or image numbers" in prompt


class FakeVision:
    def __init__(self, reply):
        self.reply, self.images = reply, None

    def structured(self, image, prompt, schema, max_side=None, num_predict=400):
        self.images = image
        return self.reply, Timing(0, 30.0, 3700, 4.0, 60, 34.0)


def test_answer_maps_moment_number_to_candidate(tmp_path: Path) -> None:
    moments = fuse(FRAMES, SPEECH, PINK_CAP)
    fake = FakeVision({"answer": "At 1:12.", "moment": 2, "confidence": "high", "evidence": "pink cap"})
    a = answer_question("guy in a pink cap", moments, fake, tmp_path, AnswerConfig(top_n=3))
    assert a.moment is moments[1] and len(fake.images) == 3 and a.confidence == "high"


def test_zero_or_out_of_range_means_no_moment(tmp_path: Path) -> None:
    moments = fuse(FRAMES, SPEECH, PINK_CAP)
    for bad in (0, 9, -1):
        fake = FakeVision({"answer": "Not shown.", "moment": bad, "confidence": "low", "evidence": ""})
        assert answer_question("q", moments, fake, tmp_path, AnswerConfig()).moment is None


def test_several_images_go_in_one_request(tmp_path: Path) -> None:
    from PIL import Image
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 64), "red").save(p)
        paths.append(p)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(
            {"answer": "a", "moment": 1, "confidence": "low", "evidence": ""})}, "done": True})

    cfg = VisionConfig()
    v = OllamaVision(cfg, client=httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(handler)))
    v.structured(paths, "q", ANSWER_SCHEMA)
    assert len(seen["messages"][0]["images"]) == 3
    assert seen["options"]["num_ctx"] == 8192   # Ollama's 4096 default overflowed on a real question


def test_bare_number_answer_falls_back_to_evidence(tmp_path: Path) -> None:
    moments = fuse(FRAMES, SPEECH, PINK_CAP)
    for bare in ("0", "moment 1", " Moment 2. "):
        fake = FakeVision({"evidence": "Image 1 shows Durga idols.", "moment": 1, "answer": bare, "confidence": "high"})
        assert answer_question("q", moments, fake, tmp_path, AnswerConfig()).text == "Image 1 shows Durga idols."


def test_schema_asks_for_evidence_before_answer() -> None:
    order = list(ANSWER_SCHEMA["properties"])
    assert order.index("evidence") < order.index("moment") < order.index("answer")
