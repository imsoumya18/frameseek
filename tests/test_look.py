"""Tests for the look-again loop, with a scripted fake model."""

from pathlib import Path

from PIL import Image

from frameseek.answer import Moment
from frameseek.config import AnswerConfig
from frameseek.look import LOOK_SCHEMA, look_again, look_prompt, zoom_prompt
from frameseek.models.vision import Timing

T = Timing(0, 20.0, 1500, 4.0, 60, 24.0)
NO = {"seen": "people talking", "present": False, "source": "none", "box": [0, 0, 0, 0], "answer": "No."}
CAP = {"seen": "pink cap on the man at right", "present": True, "source": "image",
       "box": [700, 50, 920, 260], "answer": "A man at the right wears a pink cap."}


class Scripted:
    """Returns the next reply for each call and records what it was shown."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def structured(self, image, prompt, schema, max_side=None, num_predict=400):
        self.calls.append((Path(image).name, schema))
        return self.replies.pop(0), T


def moments(*files, speech=""):
    return [Moment(t=10.0 * i, file=f, speech=speech) for i, f in enumerate(files, start=1)]


def setup(tmp_path: Path, files) -> Path:
    kf = tmp_path / "keyframes"
    kf.mkdir()
    for f in files:
        Image.new("RGB", (1280, 540), "white").save(kf / f)
    return kf


def test_stops_at_first_confirmed_moment(tmp_path: Path) -> None:
    files = ["kf_0006_00.jpg", "kf_0013_00.jpg", "kf_0014_00.jpg", "kf_0019_00.jpg"]
    kf, ms = setup(tmp_path, files), moments(*files)
    zoom_yes = {"seen": "pink baseball cap", "present": True, "answer": "He wears a pink cap.", "confidence": "high"}
    fake = Scripted(NO, NO, CAP, zoom_yes)
    a = look_again("guy in a pink cap", ms, fake, kf, tmp_path / "crops", AnswerConfig())
    assert a.moment is ms[2] and a.text == "He wears a pink cap." and a.confidence == "high"
    assert [s.kind for s in a.steps] == ["look", "look", "look", "zoom"]   # 4th moment never checked
    assert a.crop and a.crop.exists() and fake.calls[-1][0] == a.crop.name
    assert a.timing.prefill_tokens == 4 * 1500


def test_zoom_that_does_not_confirm_keeps_looking(tmp_path: Path) -> None:
    files = ["kf_0013_00.jpg", "kf_0019_00.jpg"]
    kf, ms = setup(tmp_path, files), moments(*files)
    zoom_no = {"seen": "a pink bowl, no cap", "present": False, "answer": "No cap.", "confidence": "low"}
    zoom_yes = {"seen": "pink cap", "present": True, "answer": "Pink cap at right.", "confidence": "high"}
    a = look_again("guy in a pink cap", ms, Scripted(CAP, zoom_no, CAP, zoom_yes), kf, tmp_path / "crops",
                   AnswerConfig())
    assert a.moment is ms[1]
    assert [(s.kind, s.present) for s in a.steps] == [("look", True), ("zoom", False), ("look", True), ("zoom", True)]


def test_not_found_after_look_n(tmp_path: Path) -> None:
    files = [f"kf_{i:04d}_00.jpg" for i in range(8)]
    kf, ms = setup(tmp_path, files), moments(*files)
    fake = Scripted(*[NO] * 8)
    a = look_again("guy in a pink cap", ms, fake, kf, tmp_path / "crops", AnswerConfig(look_n=3))
    assert a.moment is None and len(fake.calls) == 3 and "top 3" in a.text and a.confidence == "low"


def test_whole_scene_and_speech_hits_skip_zoom(tmp_path: Path) -> None:
    kf = setup(tmp_path, ["kf_0026_00.jpg"])
    scene = {**CAP, "seen": "Durga idols", "box": [0, 0, 1000, 1000], "answer": "Durga idols indoors."}
    a = look_again("where is the Durga idol?", moments("kf_0026_00.jpg"), Scripted(scene), kf, tmp_path / "c",
                   AnswerConfig())
    assert a.text == "Durga idols indoors." and a.crop is None and len(a.steps) == 1

    sung = {"seen": "lyric in speech", "present": True, "source": "speech", "box": [0, 0, 0, 0], "answer": "Sung at 0:10."}
    a = look_again("ঢাক পেটাতে", moments("kf_0026_00.jpg", speech="এশেছে ঢাক পেটাতে"), Scripted(sung), kf,
                   tmp_path / "c", AnswerConfig())
    assert a.moment is not None and a.crop is None


def test_speech_claim_without_speech_is_rejected(tmp_path: Path) -> None:
    # The real failure from Section 7: claiming a lyric for a title card with no speech.
    kf = setup(tmp_path, ["kf_0000_00.jpg"])
    sung = {"seen": "the lyric", "present": True, "source": "speech", "box": [0, 0, 0, 0], "answer": "It says it."}
    a = look_again("আমি তো প্রেমিক", moments("kf_0000_00.jpg"), Scripted(sung), kf, tmp_path / "c",
                   AnswerConfig(look_n=1))
    assert a.moment is None and a.steps[0].present is False


def test_prompt_and_schema() -> None:
    p = look_prompt("what is written on the striped t-shirt?",
                    Moment(t=72.2, file="kf_0019_00.jpg", caption_text="Two women laughing", speech="এশেছে"))
    assert "1:12" in p and "0 to 1000" in p and "Said around this time" in p
    order = list(LOOK_SCHEMA["properties"])
    assert order.index("seen") < order.index("present") < order.index("box") < order.index("answer")


def test_zoom_keeps_the_full_frame_answer_as_context() -> None:
    # The crop alone loses the setting: "where is the idol?" was answered "here".
    p = zoom_prompt("where is the Durga idol?", Moment(t=115.0, file="kf_0026_00.jpg"), "idol",
                    "The Durga idols stand indoors in a decorated shrine.")
    assert "decorated shrine" in p and "Do not just repeat the" in p
