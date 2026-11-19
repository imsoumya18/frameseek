"""Tests for the look-again loop, with a scripted fake model."""

from pathlib import Path

from PIL import Image

from frameseek.answer import Moment
from frameseek.config import AnswerConfig
from frameseek.look import LOOK_SCHEMA, look_again, look_prompt, zoom_prompt
from frameseek.models.vision import Timing

T = Timing(0, 20.0, 1500, 4.0, 60, 24.0)
NO = {"seen": "people talking", "present": False, "source": "none", "box": [0, 0, 0, 0], "answer": "No."}
REL = {"reason": "it answers the question", "supports": True}   # check 2 on a visual answer
CAP = {"seen": "pink cap on the man at right", "present": True, "source": "image",
       "box": [700, 50, 920, 260], "answer": "A man at the right wears a pink cap."}


class Scripted:
    """Returns the next reply for each call and records what it was shown."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def structured(self, image, prompt, schema, max_side=None, num_predict=400):
        self.calls.append((Path(image).name if image else None, schema))
        reply = self.replies.pop(0)
        return reply, (Timing(0, 1.0, 180, 1.0, 20, 2.0) if image is None else T)


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
    fake = Scripted(NO, NO, CAP, zoom_yes, REL)
    a = look_again("guy in a pink cap", ms, fake, kf, tmp_path / "crops", AnswerConfig())
    assert a.moment is ms[2] and a.text == "He wears a pink cap." and a.confidence == "high"
    assert [s.kind for s in a.steps] == ["look", "look", "look", "zoom", "check"]   # 4th moment never checked
    assert a.crop and a.crop.exists() and fake.calls[-2][0] == a.crop.name
    assert fake.calls[-1][0] is None                                          # the check is text only
    assert a.timing.prefill_tokens == 4 * 1500 + 180


def test_zoom_that_does_not_confirm_keeps_looking(tmp_path: Path) -> None:
    files = ["kf_0013_00.jpg", "kf_0019_00.jpg"]
    kf, ms = setup(tmp_path, files), moments(*files)
    zoom_no = {"seen": "a pink bowl, no cap", "present": False, "answer": "No cap.", "confidence": "low"}
    zoom_yes = {"seen": "pink cap", "present": True, "answer": "Pink cap at right.", "confidence": "high"}
    a = look_again("guy in a pink cap", ms, Scripted(CAP, zoom_no, CAP, zoom_yes, REL), kf, tmp_path / "crops",
                   AnswerConfig())
    assert a.moment is ms[1]
    assert [(s.kind, s.present) for s in a.steps] == [("look", True), ("zoom", False), ("look", True),
                                                      ("zoom", True), ("check", True)]


def test_not_found_after_look_n(tmp_path: Path) -> None:
    files = [f"kf_{i:04d}_00.jpg" for i in range(8)]
    kf, ms = setup(tmp_path, files), moments(*files)
    fake = Scripted(*[NO] * 8)
    a = look_again("guy in a pink cap", ms, fake, kf, tmp_path / "crops", AnswerConfig(look_n=3))
    assert a.moment is None and len(fake.calls) == 3 and "top 3" in a.text and a.confidence == "low"


def test_whole_scene_and_speech_hits_skip_zoom(tmp_path: Path) -> None:
    kf = setup(tmp_path, ["kf_0026_00.jpg"])
    scene = {**CAP, "seen": "Durga idols", "box": [0, 0, 1000, 1000], "answer": "Durga idols indoors."}
    a = look_again("where is the Durga idol?", moments("kf_0026_00.jpg"), Scripted(scene, REL), kf, tmp_path / "c",
                   AnswerConfig())
    assert a.text == "Durga idols indoors." and a.crop is None and [s.kind for s in a.steps] == ["look", "check"]

    sung = {"seen": "lyric in speech", "present": True, "source": "speech", "quote": "ঢাক পেটাতে",
            "box": [0, 0, 0, 0], "answer": "Sung at 0:10."}
    yes = {"reason": "the line is the one asked about", "supports": True}
    a = look_again("ঢাক পেটাতে", moments("kf_0026_00.jpg", speech="এশেছে ঢাক পেটাতে"), Scripted(sung, yes), kf,
                   tmp_path / "c", AnswerConfig())
    assert a.moment is not None and a.crop is None and [s.kind for s in a.steps] == ["look", "check"]


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


# ---------- Section 9: evidence checks inside the loop ----------

REAL_CHUNK = ("কা জলছা পারুপ দুখান গালে আট পোরে টুল আমি পাঞ্জা ভিতে খাচি খাবি খুল তুমার চিকন কারির মখ মলে "
              "আছল এবার পুজো তে তুমার শাতে গুড্তে হাত ধারাতে")
DHAK_WRONG = ("মিলে আশ্কারা তো আলাতে আটখার হবে দো সুমে পার আবার বছো রপে খার খোরে বে পরো আবান চলো ছেছে "
              "ধরিগান খালিমা দুখা চানে কোতো টাস্লো মোশানে")


def test_real_quote_that_does_not_answer_is_rejected_then_abstains(tmp_path: Path) -> None:
    # The real dhak case: 'ধরিগান খালিমা' is in the transcript, but it isn't about the dhak.
    kf = setup(tmp_path, ["kf_0050_00.jpg"])
    claim = {"seen": "lyrics", "present": True, "source": "speech", "quote": "ধরিগান খালিমা",
             "box": [0, 0, 0, 0], "answer": "They sing about the dhak in 'ধরিগান খালিমা'."}
    no = {"reason": "the line does not mention a dhak", "supports": False}
    a = look_again("when do they sing about the dhak?", moments("kf_0050_00.jpg", speech=DHAK_WRONG),
                   Scripted(claim, no), kf, tmp_path / "c", AnswerConfig(look_n=1))
    assert a.moment is None and a.text.startswith("Can't tell")
    check = a.steps[-1].check
    assert check.quote_score == 1.0 and check.supports is False   # check 1 passed, check 2 caught it


def test_made_up_screen_text_fails_check_1_without_a_model_call(tmp_path: Path) -> None:
    # The real title-card case: "the text on screen says 'আমি তো প্রেমিক'".
    kf = setup(tmp_path, ["kf_0000_00.jpg"])
    claim = {"seen": "white text on black", "present": True, "source": "image", "quote": "আমি তো প্রেমিক",
             "box": [0, 0, 1000, 1000], "answer": "The text on screen says 'আমি তো প্রেমিক'."}
    m = [Moment(t=3.0, file="kf_0000_00.jpg", screen_text="অনেকটা ভালোবাসা | সুজিত মুখার্জি")]
    fake = Scripted(claim)
    a = look_again("আমি তো প্রেমিক", m, fake, kf, tmp_path / "c", AnswerConfig(look_n=1))
    assert a.moment is None and len(fake.calls) == 1             # no support call needed
    assert a.steps[-1].check.status == "failed" and a.steps[-1].check.quote_score < 0.5


def test_misspelled_transcript_still_verifies_a_real_lyric(tmp_path: Path) -> None:
    kf = setup(tmp_path, ["kf_0000_00.jpg"])
    claim = {"seen": "lyric", "present": True, "source": "speech", "quote": "এবার পুজোতে তোমার সাথে",
             "box": [0, 0, 0, 0], "answer": "Sung around 2:15."}
    yes = {"reason": "it is the line asked about", "supports": True}
    a = look_again("এবার পুজোতে তোমার সাথে", moments("kf_0000_00.jpg", speech=REAL_CHUNK),
                   Scripted(claim, yes), kf, tmp_path / "c", AnswerConfig())
    assert a.moment is not None and a.steps[-1].check.status == "verified"


def test_speech_claim_needs_a_quote(tmp_path: Path) -> None:
    kf = setup(tmp_path, ["kf_0000_00.jpg"])
    claim = {"seen": "lyric", "present": True, "source": "speech", "quote": "", "box": [0, 0, 0, 0], "answer": "Yes."}
    a = look_again("q", moments("kf_0000_00.jpg", speech=REAL_CHUNK), Scripted(claim), kf, tmp_path / "c",
                   AnswerConfig(look_n=1))
    assert a.moment is None and a.steps[-1].kind == "check"


def test_verify_off_accepts_without_checking(tmp_path: Path) -> None:
    kf = setup(tmp_path, ["kf_0050_00.jpg"])
    claim = {"seen": "lyrics", "present": True, "source": "speech", "quote": "ধরিগান খালিমা",
             "box": [0, 0, 0, 0], "answer": "In 'ধরিগান খালিমা'."}
    fake = Scripted(claim)
    a = look_again("dhak?", moments("kf_0050_00.jpg", speech=DHAK_WRONG), fake, kf, tmp_path / "c",
                   AnswerConfig(verify=False))
    assert a.moment is not None and len(fake.calls) == 1


def test_visual_answer_that_does_not_address_the_question_is_rejected(tmp_path: Path) -> None:
    # The real "এবার পুজোতে তোমার সাথে" miss: a lyric question answered with "a couple dancing".
    kf = setup(tmp_path, ["kf_0069_00.jpg"])
    dance = {"seen": "couple dancing with smoke", "present": True, "source": "image", "quote": "",
             "box": [0, 0, 1000, 1000], "answer": "A man and a woman are dancing in a temple."}
    no = {"reason": "a dance does not show the lyric being sung", "supports": False}
    a = look_again("এবার পুজোতে তোমার সাথে", moments("kf_0069_00.jpg"), Scripted(dance, no), kf, tmp_path / "c",
                   AnswerConfig(look_n=1))
    assert a.moment is None and a.steps[-1].source == "visual answer"


def test_a_failed_model_call_counts_as_not_here(tmp_path: Path) -> None:
    from frameseek.models.vision import OllamaError

    class CutOff(Scripted):
        def structured(self, image, prompt, schema, max_side=None, num_predict=400):
            if not self.calls:
                self.calls.append(("x", schema))
                raise OllamaError("Output was cut off before the JSON finished")
            return super().structured(image, prompt, schema, max_side, num_predict)

    files = ["kf_0026_00.jpg", "kf_0081_00.jpg"]
    kf = setup(tmp_path, files)
    scene = {**CAP, "box": [0, 0, 1000, 1000], "answer": "Idol."}
    a = look_again("q", moments(*files), CutOff(scene, REL), kf, tmp_path / "c", AnswerConfig())
    assert a.moment is not None and a.moment.file == "kf_0081_00.jpg" and a.steps[0].seen.startswith("error")
