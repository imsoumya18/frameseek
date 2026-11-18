"""Tests for the transcribe step. No network or real Whisper model needed."""

import json
from pathlib import Path

import pytest

from frameseek.config import TranscribeConfig
from frameseek.devices import ctranslate2_device
from frameseek.ingest import VideoMeta
from frameseek.transcribe import (
    NoCaptionsError,
    Segment,
    Word,
    build_transcript,
    parse_json3,
    pick_caption_track,
)

# ---------- json3 parsing ----------

AUTO_JSON3 = {
    "events": [
        {"tStartMs": 1000, "dDurationMs": 3000,
         "segs": [{"utf8": "Maa"}, {"utf8": " aschhe", "tOffsetMs": 600}, {"utf8": " ghore", "tOffsetMs": 1400}]},
        {"tStartMs": 4000, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
    ]
}

MANUAL_JSON3 = {
    "events": [{"tStartMs": 2500, "dDurationMs": 2000, "segs": [{"utf8": "Shubho Bijoya to everyone"}]}]
}


def test_auto_captions_give_word_timestamps() -> None:
    [seg] = parse_json3(AUTO_JSON3)   # the newline-only event is skipped
    assert seg.text == "Maa aschhe ghore"
    assert (seg.start, seg.end) == (1.0, 4.0)
    assert [(w.w, w.start, w.end) for w in seg.words] == [
        ("Maa", 1.0, 1.6), ("aschhe", 1.6, 2.4), ("ghore", 2.4, 4.0)]


def test_manual_captions_are_segment_level() -> None:
    [seg] = parse_json3(MANUAL_JSON3)
    assert seg.text == "Shubho Bijoya to everyone"
    assert seg.words == []


# ---------- choosing a caption track ----------

def test_manual_track_wins_and_live_chat_is_ignored() -> None:
    info = {"language": "bn", "subtitles": {"live_chat": [], "en": [], "bn": []},
            "automatic_captions": {"bn-orig": []}}
    assert pick_caption_track(info) == ("manual", "bn")


def test_auto_prefers_original_speech_track_over_translations() -> None:
    info = {"language": "hi", "subtitles": {"live_chat": []},
            "automatic_captions": {"en": [], "fr": [], "bn-orig": [], "bn": []}}
    assert pick_caption_track(info) == ("auto", "bn-orig")


def test_no_captions() -> None:
    assert pick_caption_track({"subtitles": {}, "automatic_captions": {}}) is None


# ---------- build_transcript ----------

META = VideoMeta(id="abc", url="https://www.youtube.com/watch?v=abc", file="video.mp4", duration_s=60.0)
SEGS = [Segment(0.0, 2.0, "hello there", [Word("hello", 0.0, 0.8), Word("there", 0.8, 2.0)])]


def captions_found(url, on_progress):
    return SEGS, "auto", "bn"


def captions_missing(url, on_progress):
    return None


CALLS = {}


def whisper_ok(video, cfg, device, on_progress):
    CALLS["language"] = cfg.language   # what language the transcription model was told
    return SEGS, cfg.language or "bn"


def must_not_call(*a, **k):
    raise AssertionError("should not be called")


def hears(lang, prob=0.95):
    return lambda video, cfg, device, on_progress: (lang, prob)


def manual_captions(url, on_progress):
    return SEGS, "manual", "hi"


def test_auto_uses_captions_when_available(tmp_path: Path) -> None:
    tr, cached = build_transcript(tmp_path, META, TranscribeConfig(),
                                  caption_fetcher=captions_found, whisper_fn=must_not_call,
                                  language_detector=hears("bn"))
    assert (tr.source, tr.language, cached) == ("youtube-auto", "bn", False)
    assert "matching" in tr.note
    saved = json.loads((tmp_path / "transcript.json").read_text())
    assert saved["segments"][0]["words"][1]["w"] == "there"


def test_auto_falls_back_to_whisper(tmp_path: Path) -> None:
    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(),
                             caption_fetcher=captions_missing, whisper_fn=whisper_ok,
                             language_detector=hears("bn"))
    assert tr.source == "whisper:large-v3-turbo"
    assert CALLS["language"] == "bn"                      # detected language is forced
    assert tr.language_from.startswith("detected by Whisper small")


def test_youtube_only_fails_without_captions(tmp_path: Path) -> None:
    with pytest.raises(NoCaptionsError):
        build_transcript(tmp_path, META, TranscribeConfig(source="youtube"),
                         caption_fetcher=captions_missing, whisper_fn=must_not_call)


def test_whisper_mode_skips_youtube(tmp_path: Path) -> None:
    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(source="whisper"),
                             caption_fetcher=must_not_call, whisper_fn=whisper_ok,
                             language_detector=hears("bn"))
    assert tr.source == "whisper:large-v3-turbo"


def test_cached_until_params_change(tmp_path: Path) -> None:
    build_transcript(tmp_path, META, TranscribeConfig(), caption_fetcher=captions_found,
                     language_detector=hears("bn"))
    _, cached = build_transcript(tmp_path, META, TranscribeConfig(), caption_fetcher=must_not_call,
                                 whisper_fn=must_not_call, language_detector=must_not_call)
    assert cached is True
    _, cached = build_transcript(tmp_path, META, TranscribeConfig(whisper_model="medium"),
                                 caption_fetcher=captions_found, language_detector=hears("bn"))
    assert cached is False


def test_transcript_stats(tmp_path: Path) -> None:
    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(), caption_fetcher=captions_found,
                             language_detector=hears("bn"))
    assert tr.speech_seconds() == 2.0
    assert tr.word_count() == 2


def test_wrong_language_captions_fall_back_to_whisper(tmp_path: Path) -> None:
    """The Bengali-song-tagged-as-Hindi case: captions are 'hi', the audio is Bengali."""
    def hindi_captions(url, on_progress):
        return SEGS, "auto", "hi"

    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(), caption_fetcher=hindi_captions,
                             whisper_fn=whisper_ok, language_detector=hears("bn", 0.97))
    assert tr.source == "whisper:large-v3-turbo"
    assert CALLS["language"] == "bn"   # turbo is told Bengali, not left to guess
    assert "'hi'" in tr.note and "'bn'" in tr.note


def test_unsure_detection_keeps_captions(tmp_path: Path) -> None:
    def hindi_captions(url, on_progress):
        return SEGS, "auto", "hi"

    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(), caption_fetcher=hindi_captions,
                             whisper_fn=must_not_call, language_detector=hears("bn", 0.3))
    assert tr.source == "youtube-auto"


def test_manual_captions_are_trusted_without_a_check(tmp_path: Path) -> None:
    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(), caption_fetcher=manual_captions,
                             whisper_fn=must_not_call, language_detector=must_not_call)
    assert tr.source == "youtube-manual"


def test_forced_language_is_labelled_forced(tmp_path: Path) -> None:
    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(source="whisper", language="bn"),
                             caption_fetcher=must_not_call, whisper_fn=whisper_ok,
                             language_detector=must_not_call)   # no detection when forced
    assert tr.language_from == "forced in config"


def test_unsure_language_lets_whisper_decide(tmp_path: Path) -> None:
    tr, _ = build_transcript(tmp_path, META, TranscribeConfig(source="whisper"),
                             caption_fetcher=must_not_call, whisper_fn=whisper_ok,
                             language_detector=hears("en", 0.3))
    assert CALLS["language"] == ""
    assert "unsure" in tr.language_from


def test_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_transcript(tmp_path, META, TranscribeConfig(source="magic"))


# ---------- devices ----------

def test_ctranslate2_mps_only_when_asked() -> None:
    assert ctranslate2_device("mps") == "mps"
    assert ctranslate2_device("auto") in {"cpu", "cuda"}   # never auto-picks experimental MPS


# ---------- speech coverage ----------

def _tr(segs):
    from frameseek.transcribe import Transcript
    return Transcript("youtube-auto", "hi", "x", {}, segs)


def test_overlapping_captions_are_not_double_counted() -> None:
    # rolling auto captions: each line overlaps the next
    tr = _tr([Segment(0.0, 4.0, "a b"), Segment(2.0, 6.0, "c d"), Segment(10.0, 12.0, "e")])
    assert tr.speech_seconds() == 8.0   # [0, 6] + [10, 12], not 4 + 4 + 2 = 10


def test_music_markers_are_not_speech() -> None:
    tr = _tr([Segment(0.0, 5.0, "[संगीत]"), Segment(5.0, 7.0, "[Music] [Applause]"), Segment(7.0, 9.0, "चलो")])
    assert tr.speech_seconds() == 2.0
