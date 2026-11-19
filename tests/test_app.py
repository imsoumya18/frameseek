"""Tests for the web app's pieces, with fakes (no models, no browser)."""

import json
from pathlib import Path

from frameseek import app
from frameseek.answer import Answer, Moment
from frameseek.look import Step
from frameseek.models.vision import Timing
from frameseek.pipeline import AskResult
from frameseek.progress import Checklist, from_disk
from frameseek.verify import Check

T = Timing(0, 20.0, 1500, 4.0, 60, 24.0)


def test_step_lines_mark_outcomes() -> None:
    m = Moment(t=61.9, file="kf_0014_00.jpg")
    assert "✗" in app._step_line(Step("look", m, 1, False, "pink bowl", "none", timing=T))
    line = app._step_line(Step("check", m, 1, False, "", "speech", quote="ধরিগান খালিমা",
                               check=Check("failed", 1.0, False, "not about the dhak")))
    assert "✗" in line and "1.00" in line and "not about the dhak" in line


def test_card_escapes_text() -> None:
    assert "&lt;script&gt;" in app._card("<script>", [("high confidence", "fs-high")])


def test_seek_restarts_the_player_at_the_clicked_moment() -> None:
    assert "nRWZBq2hcls?start=61" in app.seek({"video_id": "nRWZBq2hcls", "times": [32, 61]}, 1)


def test_checklist_shows_done_running_and_waiting() -> None:
    c = Checklist()
    c.set("download", "done", 1.0, "4:46 · 540p")
    c.set("shots", "running", 0.5, "saving keyframe 44 of 88")
    out = app._checklist_html(c, "Infra - Khoshmejaje")
    assert out.count("✅") == 1 and out.count("⏳") == 1 and out.count("⬜") == 3
    assert "1 of 5" in out and "30%" in out and "saving keyframe 44 of 88" in out


def _fake_video(d: Path, captioned: int, total: int = 4) -> None:
    d.mkdir(parents=True)
    (d / "video.mp4").write_bytes(b"x" * 10)
    (d / "meta.json").write_text(json.dumps({"id": d.name, "url": "u", "file": "video.mp4", "title": "T",
                                             "duration_s": 286.0}))
    (d / "shots.json").write_text(json.dumps({"shots": [{"keyframes": [{"file": f"kf_{i}.jpg"}]} for i in range(total)]}))
    (d / "transcript.json").write_text(json.dumps({"source": "whisper:large-v3-turbo", "language": "bn",
                                                   "segments": [{}, {}]}))
    (d / "captions.jsonl").write_text("\n".join(json.dumps({"file": f"kf_{i}.jpg", "caption": {"summary": "s"}})
                                                for i in range(captioned)))


def test_from_disk_reads_what_is_already_done(tmp_path: Path) -> None:
    _fake_video(tmp_path / "abc", captioned=3)
    c = from_disk(tmp_path / "abc")
    assert [c.steps[k].state for k in ("download", "shots", "speech", "captions", "index")] == \
        ["done", "done", "done", "waiting", "waiting"]
    assert c.steps["captions"].fraction == 0.75 and "3 of 4" in c.steps["captions"].detail
    assert "4:46" in c.steps["download"].detail and "Whisper · bn · 2 lines" == c.steps["speech"].detail
    assert from_disk(tmp_path / "nothing").done == 0


def test_checklist_for_a_pasted_link(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app._cfg, "data_dir", tmp_path)
    assert "Paste a YouTube link" in app.checklist_for("")
    assert "Not prepared yet" in app.checklist_for("https://youtu.be/nRWZBq2hcls")


def test_ask_streams_steps_then_shows_results(tmp_path: Path, monkeypatch) -> None:
    vd = tmp_path / "nRWZBq2hcls"
    (vd / "index.lance").mkdir(parents=True)
    monkeypatch.setattr(app._cfg, "data_dir", tmp_path)
    m = Moment(t=61.9, file="kf_0014_00.jpg")
    steps = [Step("look", m, 1, True, "a man wearing a pink cap", "image", timing=T)]

    class FakeAsker:
        def ask(self, video_dir, question, mode=None, on_step=None):
            for s in steps:
                on_step(s)
            return AskResult(Answer("A man in a pink cap.", m, "high", "pink cap", T, [m], steps=steps), 900.0, 30.0)

    monkeypatch.setattr(app, "_asker", FakeAsker())
    monkeypatch.setattr(app, "_quiet", True)
    outs = list(app.ask_question("https://youtu.be/nRWZBq2hcls", "guy in a pink cap"))
    assert "Searching" in outs[0][0]
    assert "Looking at 1:01" in outs[1][0] and "pink cap" in outs[1][0]   # live feed inside the card
    card, player, evidence, moments, state = outs[-1]
    assert "A man in a pink cap." in card and "start=61" in player
    assert state["times"] == [61] and moments[0][1].startswith("1:01")


def test_new_link_loads_that_video_paused_and_clears_the_old_answer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app._cfg, "data_dir", tmp_path)
    checklist, player, card, evidence, moments, state, shown = app.on_link("https://youtu.be/nRWZBq2hcls", "abc12345678")
    assert "embed/nRWZBq2hcls?start=0&autoplay=0" in player
    assert card == app.IDLE_CARD and evidence is None and moments == [] and state == {} and shown == "nRWZBq2hcls"
    same = app.on_link("https://www.youtube.com/watch?v=nRWZBq2hcls", "nRWZBq2hcls")
    assert "Not prepared yet" in same[0] and all(type(x).__name__ != "str" for x in same[1:])   # nothing reloads


def test_header_links_author_and_repo():
    h = app._header()
    assert "Developed by" in h and "Soumya" in h
    assert 'href="https://github.com/imsoumya18/frameseek"' in h
    assert 'target="_blank"' in h
