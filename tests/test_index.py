"""Tests for the vector index: text flattening, speech chunking, caching and search.

Embedders are fakes (bag-of-words vectors), but LanceDB is real, so storage
and search are exercised end to end without downloading models.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from frameseek.config import EmbedConfig
from frameseek.index import build_index, caption_text, chunk_speech, has_content, search
from frameseek.transcribe import Segment, Transcript

DIM = 64


def bow(text: str) -> np.ndarray:
    """Bag-of-words vector: texts sharing words point in similar directions."""
    v = np.zeros(DIM, dtype="float32")
    for w in text.lower().replace(".", " ").replace(",", " ").split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeText:
    loaded = False

    def load(self):
        self.loaded = True

    def queries(self, texts):
        return np.stack([bow(t) for t in texts])

    def passages(self, texts):
        return np.stack([bow(t) for t in texts]) if texts else np.zeros((0, DIM), "float32")


class FakeImage:
    """Pretends each keyframe 'looks like' a hidden description, like real pixels would."""
    def __init__(self, looks):
        self.looks = looks
        self.calls = 0

    def load(self):
        pass

    def images(self, paths, on_progress=lambda _: None):
        self.calls += 1
        on_progress({"stage": "embed_frames", "done": len(paths), "total": len(paths)})
        return np.stack([bow(self.looks[p.name]) for p in paths])

    def texts(self, texts):
        return np.stack([bow(t) for t in texts])


def cap(summary, objects=(), text=()):
    return {"summary": summary, "shot": "medium", "people": {"count": 0, "description": ""},
            "objects": [{"name": n, "colour": c} for n, c in objects],
            "text_on_screen": list(text), "setting": "outdoor", "action": ""}


RECORDS = [
    {"file": "kf_0001_00.jpg", "t": 12.0, "shot": 1, "caption": cap("Man plays guitar on balcony", [("guitar", "brown")])},
    {"file": "kf_0002_00.jpg", "t": 72.2, "shot": 2, "caption": cap("Two women laughing with friends", text=["あなたの右目に"])},
    {"file": "kf_0003_00.jpg", "t": 10.3, "shot": 0, "caption": cap("Goddess idol display in temple")},
]
# What the frames "look like": frame 2 shows a pink cap that its caption never mentions.
LOOKS = {"kf_0001_00.jpg": "man guitar balcony", "kf_0002_00.jpg": "women laughing pink cap",
         "kf_0003_00.jpg": "goddess idol temple"}
TRANSCRIPT = Transcript("whisper:large-v3-turbo", "bn", "x", {}, [
    Segment(36.0, 40.0, "কাশের ফুলে দোলে"), Segment(40.2, 43.0, "খিল খিলি দাত কপাটি"),
    Segment(43.3, 47.0, "[Music]"), Segment(55.6, 59.0, "এসেছে ঢাক পেটাতে দুখান কাঠি"),
])


# ---------- text preparation ----------

def test_caption_text_includes_every_field() -> None:
    text = caption_text(cap("Red car parked", [("car", "red")], ["STOP"]))
    assert "Red car parked." in text and "red car" in text and "Text on screen: STOP" in text


def test_chunk_speech_merges_and_splits_on_silence() -> None:
    chunks = chunk_speech(TRANSCRIPT.segments, target_s=15, max_gap_s=3)
    assert [round(c["start"], 1) for c in chunks] == [36.0, 55.6]   # 12.6 s gap splits; [Music] dropped
    assert chunks[0]["text"] == "কাশের ফুলে দোলে খিল খিলি দাত কপাটি"


def test_chunk_respects_target_length() -> None:
    segs = [Segment(float(i), float(i) + 0.9, f"w{i}") for i in range(40)]
    chunks = chunk_speech(segs, target_s=10, max_gap_s=3)
    assert all(c["end"] - c["start"] <= 11 for c in chunks) and len(chunks) == 4


# ---------- build + search ----------

@pytest.fixture
def built(tmp_path: Path):
    img = FakeImage(LOOKS)
    stats, cached = build_index(tmp_path, RECORDS, TRANSCRIPT, FakeText(), img, EmbedConfig())
    return tmp_path, stats, cached, img


def test_build_writes_both_tables(built) -> None:
    _, stats, cached, _ = built
    assert (stats.frames, stats.speech_chunks, cached) == (3, 2, False)


def test_rebuild_is_cached_until_inputs_change(built) -> None:
    vdir, _, _, img = built
    _, cached = build_index(vdir, RECORDS, TRANSCRIPT, FakeText(), img, EmbedConfig())
    assert cached is True and img.calls == 1
    _, cached = build_index(vdir, RECORDS, TRANSCRIPT, FakeText(), img, EmbedConfig(speech_chunk_s=5))
    assert cached is False


def test_caption_search_finds_written_facts(built) -> None:
    res = search(built[0], "playing guitar", FakeText(), FakeImage(LOOKS), k=3)
    assert res["captions"][0]["file"] == "kf_0001_00.jpg"
    assert res["captions"][0]["score"] > res["captions"][1]["score"]


def test_frame_search_finds_what_captions_missed(built) -> None:
    """'pink cap' is in no caption, but frame 2's pixels show it."""
    res = search(built[0], "pink cap", FakeText(), FakeImage(LOOKS), k=3)
    assert res["frames"][0]["file"] == "kf_0002_00.jpg"


def test_speech_search_returns_time_ranges(built) -> None:
    res = search(built[0], "ঢাক পেটাতে", FakeText(), FakeImage(LOOKS), k=2)
    top = res["speech"][0]
    assert (top["start"], top["end"]) == (55.6, 59.0) and "img_vec" not in top and "txt_vec" not in top


def test_frames_only_index_without_transcript(tmp_path: Path) -> None:
    stats, _ = build_index(tmp_path, RECORDS, None, FakeText(), FakeImage(LOOKS), EmbedConfig())
    assert stats.speech_chunks == 0
    assert search(tmp_path, "guitar", FakeText(), FakeImage(LOOKS))["speech"] == []


def test_empty_captions_are_left_out_of_caption_search(tmp_path: Path) -> None:
    """A black frame's empty caption must not surface as a caption match."""
    blank = {"file": "kf_0083_00.jpg", "t": 282.4, "shot": 83,
             "caption": cap("") | {"setting": "", "people": {"count": 0, "description": ""}}}
    looks = LOOKS | {"kf_0083_00.jpg": "black"}
    build_index(tmp_path, RECORDS + [blank], None, FakeText(), FakeImage(looks), EmbedConfig())
    assert not has_content(blank["caption"])
    res = search(tmp_path, "medium outdoor", FakeText(), FakeImage(looks), k=10)
    assert "kf_0083_00.jpg" not in [r["file"] for r in res["captions"]]
    assert "kf_0083_00.jpg" in [r["file"] for r in res["frames"]]   # still searchable by pixels
