"""Timestamped speech for a video: YouTube captions first, local Whisper otherwise.

Output: data/<video_id>/transcript.json

    source          youtube-manual | youtube-auto | whisper:<model>
    language        language code of the speech
    language_from   where that code came from (caption track, YouTube's own
                    speech recognition, or Whisper's detection)
    params          settings used; a change re-runs the step
    segments        [{start, end, text, words: [{w, start, end}]}]

Source order in "auto" mode: manual captions > auto captions > Whisper.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from frameseek.config import TranscribeConfig
from frameseek.devices import ctranslate2_device
from frameseek.ingest import VideoMeta, write_json_atomic

ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class Word:
    w: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    source: str
    language: str | None
    language_from: str
    params: dict[str, Any]
    segments: list[Segment]
    note: str = ""            # why this source was chosen

    def speech_seconds(self) -> float:
        """Seconds covered by speech: the union of segment ranges, skipping "[Music]"-only lines.

        A plain sum over-counts, because YouTube's rolling auto captions overlap in time.
        """
        ranges = sorted((s.start, s.end) for s in self.segments
                        if s.end > s.start and not is_non_speech(s.text))
        total, cur_start, cur_end = 0.0, None, None
        for start, end in ranges:
            if cur_end is None or start > cur_end:
                if cur_end is not None:
                    total += cur_end - cur_start
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        if cur_end is not None:
            total += cur_end - cur_start
        return total

    def word_count(self) -> int:
        return sum(len(s.words) if s.words else len(s.text.split()) for s in self.segments)


NON_SPEECH = re.compile(r"^(\s*\[[^\]]*\]\s*)+$")   # "[Music]", "[संगीत] [हंसी]"


def is_non_speech(text: str) -> bool:
    """True for caption lines that only describe sounds, e.g. "[Music]" or "[संगीत]"."""
    return bool(NON_SPEECH.match(text))


class NoCaptionsError(RuntimeError):
    pass


# ---------- YouTube captions ----------

def pick_caption_track(info: dict[str, Any]) -> tuple[str, str] | None:
    """Choose (kind, language_key) from yt-dlp's caption listings, or None.

    Manual tracks win. For auto captions, YouTube also lists machine
    translations into dozens of languages; the "-orig" key is the original
    speech-recognition track, which is the one we want.
    """
    manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    if manual:
        lang = info.get("language")
        return "manual", lang if lang in manual else sorted(manual)[0]

    auto = info.get("automatic_captions") or {}
    orig = sorted(k for k in auto if k.endswith("-orig"))
    if orig:
        return "auto", orig[0]
    lang = info.get("language")
    if lang and lang in auto:
        return "auto", lang
    return None


def parse_json3(data: dict[str, Any]) -> list[Segment]:
    """Parse YouTube's json3 caption format.

    Each event has a start (tStartMs) and pieces of text (segs). Auto captions
    give each word an offset (tOffsetMs), which becomes a word timestamp.
    Manual captions usually have one piece per line and no offsets.
    """
    segments: list[Segment] = []
    for ev in data.get("events", []):
        pieces = [p for p in ev.get("segs", []) if p.get("utf8", "").strip()]
        if not pieces:
            continue
        start = ev.get("tStartMs", 0) / 1000
        end = start + ev.get("dDurationMs", 0) / 1000
        text = " ".join("".join(p["utf8"] for p in pieces).split())

        words: list[Word] = []
        if len(pieces) > 1:   # word-by-word pieces (auto captions)
            starts = [start + p.get("tOffsetMs", 0) / 1000 for p in pieces]
            for i, p in enumerate(pieces):
                w_end = starts[i + 1] if i + 1 < len(pieces) else end
                words.append(Word(p["utf8"].strip(), round(starts[i], 3), round(max(w_end, starts[i]), 3)))
        segments.append(Segment(round(start, 3), round(end, 3), text, words))
    return segments


def fetch_youtube_captions(url: str, on_progress: ProgressFn) -> tuple[list[Segment], str, str] | None:
    """Return (segments, kind, language) from YouTube, or None if there are no captions."""
    from yt_dlp import YoutubeDL

    on_progress({"stage": "captions"})
    with YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        choice = pick_caption_track(info)
        if choice is None:
            return None
        kind, key = choice
        tracks = (info["subtitles"] if kind == "manual" else info["automatic_captions"])[key]
        track = next((t for t in tracks if t.get("ext") == "json3"), None)
        if track is None:
            return None
        data = json.loads(ydl.urlopen(track["url"]).read().decode("utf-8"))

    segments = parse_json3(data)
    return (segments, kind, key.removesuffix("-orig")) if segments else None


# ---------- Whisper ----------

_MODELS: dict[tuple[str, str], Any] = {}


def _load_whisper(name: str, cfg: TranscribeConfig, device_pref: str, on_progress: ProgressFn):
    """Load each model once per process and reuse it."""
    from faster_whisper import WhisperModel

    from frameseek.hf import quiet_hf
    quiet_hf()

    device = ctranslate2_device(cfg.device if cfg.device != "auto" else device_pref)
    key = (name, device)
    if key not in _MODELS:
        on_progress({"stage": "load_model", "model": name, "device": device})
        compute = {"cuda": "float16", "cpu": "int8"}.get(device, "auto")   # mps: let CTranslate2 choose
        _MODELS[key] = WhisperModel(name, device=device, compute_type=compute)
    return _MODELS[key]


def detect_language_whisper(
    video: Path, cfg: TranscribeConfig, device_pref: str, on_progress: ProgressFn
) -> tuple[str, float]:
    """Detect the spoken language from the audio itself (speech only, several windows).

    Uses cfg.language_model, not the transcription model: Whisper picks the
    language with its decoder, and large-v3-turbo's decoder was cut from 32 to
    4 layers, which hurts language ID far more than transcription. On a Bengali
    song, turbo heard English (83%) while small correctly heard Bengali.
    """
    from faster_whisper import decode_audio

    model = _load_whisper(cfg.language_model, cfg, device_pref, on_progress)
    on_progress({"stage": "check_language"})
    audio = decode_audio(str(video))
    lang, prob, _ = model.detect_language(
        audio, vad_filter=True,               # skip the music intro, listen to voices
        language_detection_segments=3,        # vote across up to three 30 s windows
    )
    return lang, prob


def transcribe_whisper(
    video: Path, cfg: TranscribeConfig, device_pref: str, on_progress: ProgressFn
) -> tuple[list[Segment], str]:
    """Run faster-whisper with VAD and word timestamps. Returns (segments, language)."""
    model = _load_whisper(cfg.whisper_model, cfg, device_pref, on_progress)

    segments_iter, info = model.transcribe(
        str(video),
        language=cfg.language or None,       # None = detect from the first 30 s
        vad_filter=cfg.vad,                  # skip music and silence: no hallucinated text
        word_timestamps=True,                # from cross-attention alignment
        condition_on_previous_text=False,    # one bad line can't start a repetition loop
        beam_size=cfg.beam_size,
    )
    on_progress({"stage": "language", "language": info.language,
                 "probability": info.language_probability,
                 "forced": bool(cfg.language),          # told the language, not detected
                 "speech_s": info.duration_after_vad, "total_s": info.duration})

    out: list[Segment] = []
    for s in segments_iter:   # a generator: work happens as we iterate
        words = [Word(w.word.strip(), round(w.start, 3), round(w.end, 3)) for w in (s.words or [])]
        out.append(Segment(round(s.start, 3), round(s.end, 3), s.text.strip(), words))
        on_progress({"stage": "transcribe", "done": s.end, "total": info.duration})
    return out, info.language


# ---------- the transcribe step ----------

def _params(cfg: TranscribeConfig) -> dict[str, Any]:
    return asdict(cfg)


def load_transcript(video_dir: Path, cfg: TranscribeConfig) -> Transcript | None:
    path = video_dir / "transcript.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("params") != _params(cfg):
        return None
    segments = [Segment(s["start"], s["end"], s["text"], [Word(**w) for w in s["words"]])
                for s in data["segments"]]
    return Transcript(data["source"], data["language"], data["language_from"], data["params"],
                      segments, data.get("note", ""))


def _base(lang: str | None) -> str:
    """'bn-IN' -> 'bn', so regional variants compare equal."""
    return (lang or "").split("-")[0].lower()


def build_transcript(
    video_dir: Path,
    meta: VideoMeta,
    cfg: TranscribeConfig,
    device_pref: str = "auto",
    force: bool = False,
    on_progress: ProgressFn = lambda _: None,
    caption_fetcher=fetch_youtube_captions,
    whisper_fn=transcribe_whisper,
    language_detector=detect_language_whisper,
) -> tuple[Transcript, bool]:
    """Produce transcript.json following cfg.source. Returns (transcript, was_cached).

    In auto mode, YouTube auto captions are only trusted if their language matches
    what Whisper hears: YouTube runs its recognizer in the uploader's declared
    language, which can be wrong. Manual captions are always trusted.
    """
    if cfg.source not in {"auto", "youtube", "whisper"}:
        raise ValueError(f"transcribe.source must be auto, youtube or whisper, got {cfg.source!r}")
    if not force and (cached := load_transcript(video_dir, cfg)) is not None:
        return cached, True

    video = video_dir / meta.file
    transcript = None
    note = ""
    heard: tuple[str, float] | None = None   # (language, probability) from the language model

    if cfg.source in {"auto", "youtube"}:
        found = caption_fetcher(meta.url, on_progress)
        if found:
            segments, kind, lang = found
            use_captions = True
            if kind == "manual":
                note = "Manual captions from the uploader"
            elif cfg.source == "auto":
                heard = language_detector(video, cfg, device_pref, on_progress)
                h_lang, h_prob = heard
                on_progress({"stage": "language_check", "captions": lang,
                             "heard": h_lang, "probability": h_prob})
                if _base(h_lang) != _base(lang) and h_prob >= 0.5:
                    use_captions = False
                    note = (f"YouTube auto captions were in '{lang}' but the audio is '{h_lang}' "
                            f"({h_prob:.0%}); used Whisper instead")
                else:
                    note = f"Whisper hears '{h_lang}' ({h_prob:.0%}), matching the auto captions"
            else:
                note = "YouTube captions requested (--source youtube); language not checked"

            if use_captions:
                transcript = Transcript(
                    source=f"youtube-{kind}",
                    language=lang,
                    language_from=("caption track" if kind == "manual"
                                   else "YouTube auto captions, which use the uploader's declared language"),
                    params=_params(cfg),
                    segments=segments,
                    note=note,
                )
        elif cfg.source == "youtube":
            raise NoCaptionsError("This video has no YouTube captions. Use --source whisper.")
        else:
            note = "No YouTube captions available"

    if transcript is None:
        # Detect with the language model, then force that language on the
        # transcription model, unless the user already forced one.
        if cfg.language:
            run_cfg, language_from = cfg, "forced in config"
        else:
            if heard is None:
                heard = language_detector(video, cfg, device_pref, on_progress)
                on_progress({"stage": "language_check", "captions": None,
                             "heard": heard[0], "probability": heard[1]})
            if heard[1] >= 0.5:
                run_cfg = replace(cfg, language=heard[0])
                language_from = f"detected by Whisper {cfg.language_model} ({heard[1]:.0%})"
            else:
                run_cfg, language_from = cfg, "Whisper detection (language model was unsure)"

        segments, lang = whisper_fn(video, run_cfg, device_pref, on_progress)
        transcript = Transcript(
            source=f"whisper:{cfg.whisper_model}",
            language=lang,
            language_from=language_from,
            params=_params(cfg),
            segments=segments,
            note=note or "Whisper requested (--source whisper)",
        )

    write_json_atomic(video_dir / "transcript.json", asdict(transcript))
    return transcript, False
