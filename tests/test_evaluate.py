"""Tests for the question-set scoring."""

from pathlib import Path

from frameseek.answer import Moment
from frameseek.evaluate import Item, first_hit, load_eval, outcome, summarise, summarise_answers, text_ok

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


def test_text_check_catches_a_one_sign_misread() -> None:
    title = Item("name in the opening title?", "text", {"kf_0000_00"}, [], ["সৃজিত"])
    assert text_ok("The name is সৃজিত মুখার্জি.", title) is True
    assert text_ok("The name is সুজিত মুখার্জি.", title) is False   # what the 8B wrote in Section 7
    assert text_ok("anything", Item("q", "visual", set(), [])) is None


def test_absent_questions_reward_saying_none() -> None:
    cat = Item("a cat sitting on a sofa", "absent", set(), [], absent=True)
    present = Item("pink cap", "visual", {"kf_0019_00"}, [])
    m = Moment(t=72.2, file="kf_0019_00.jpg")
    assert outcome(None, False, cat) == "abstained_ok" and outcome(m, False, cat) == "false_answer"
    assert [outcome(m, True, present), outcome(m, False, present), outcome(None, False, present)] == \
        ["right", "wrong", "abstained"]
    s = summarise_answers(["right", "right", "wrong", "abstained", "abstained_ok", "false_answer", "error"])
    assert (s.answered, round(s.accuracy_when_answering, 2)) == (4, 0.5)


def test_absent_questions_are_left_out_of_retrieval_scores() -> None:
    items = [Item("a", "visual", {"x"}, []), Item("cat", "absent", set(), [], absent=True)]
    assert summarise(items, [1, None], top_n=3).n == 1


def test_every_shipped_question_set_loads() -> None:
    for path in (Path(__file__).parents[1] / "evals").glob("*.yaml"):
        video, items = load_eval(path)
        assert path.stem == video and items
        assert all(it.absent or it.frames or it.spans for it in items)
