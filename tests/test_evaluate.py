"""Tests for the question-set scoring."""

from pathlib import Path

from frameseek.answer import Moment
from frameseek.evaluate import Item, first_hit, load_eval, summarise

SPEECH = [{"start": 55.6, "end": 77.0, "text": "এসেছে ঢাক"}, {"start": 77.0, "end": 101.6, "text": "আমি তো প্রেমিক"}]


def test_frame_or_speech_chunk_counts_as_hit() -> None:
    by_frame = Item("pink cap", "visual", {"kf_0019_00"}, [])
    by_lyric = Item("আমি তো প্রেমিক", "speech", set(), [(77, 85)])
    ms = [Moment(t=10, file="kf_0001_00.jpg"), Moment(t=72.2, file="kf_0019_00.jpg", speech="এসেছে ঢাক"),
          Moment(t=90, file="kf_0024_01.jpg", speech="আমি তো প্রেমিক")]
    assert first_hit(ms, by_frame, SPEECH) == 2
    assert first_hit(ms, by_lyric, SPEECH) == 3   # 72.2 s chunk touches 77 only at its edge: not a hit


def test_summary_counts() -> None:
    items = [Item("a", "visual", set(), []), Item("b", "visual", set(), []), Item("c", "speech", set(), [])]
    s = summarise(items, [1, 4, None], top_n=3)
    assert (s.hit1, s.hit_top, round(s.mrr, 3)) == (1, 1, round((1 + 0.25) / 3, 3))
    assert s.by_kind == {"visual": (1, 2), "speech": (0, 1)}


def test_shipped_question_set_loads() -> None:
    video, items = load_eval(Path(__file__).parents[1] / "evals" / "nRWZBq2hcls.yaml")
    assert video == "nRWZBq2hcls" and len(items) == 21
    assert all(it.frames or it.spans for it in items)
