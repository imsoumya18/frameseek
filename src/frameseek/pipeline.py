"""The whole flow in two calls, shared by the CLI and the web app.

prepare(url)  indexing: ingest -> shots -> transcribe -> caption -> index.
              Every step is cached, so a prepared video returns in seconds.
Asker.ask()   asking: search -> fuse into moments -> answer, either by looking at
              moments one at a time with zoom (look) or at the top few together (compare).
              Models stay loaded between questions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from frameseek.answer import Answer, answer_question, fuse
from frameseek.caption import caption_keyframes
from frameseek.config import Config
from frameseek.devices import pick_device
from frameseek.index import build_index, score_all
from frameseek.look import StepFn, look_again
from frameseek.ingest import ingest, require_tools
from frameseek.models.embed import ImageEmbedder, TextEmbedder
from frameseek.models.vision import build_vision_model
from frameseek.progress import EventFn, mmss
from frameseek.shots import build_shots
from frameseek.transcribe import build_transcript

StatusFn = Callable[[str], None]


@dataclass
class Prepared:
    video_id: str
    title: str
    video_dir: Path


def prepare(url: str, cfg: Config, on_status: StatusFn = lambda _: None,
            on_event: EventFn = lambda *_: None) -> Prepared:
    """Run every indexing step that isn't cached yet.

    on_status gets one line per step (the CLI prints these); on_event gets
    (step, state, fraction, detail) for the web app's checklist, with progress
    inside each step where it's known.
    """
    require_tools()

    # 1. download
    on_status("Downloading video (or using the cached copy)")
    on_event("download", "running", None, "checking for a cached copy")

    def dl(ev: dict) -> None:
        if ev.get("stage") == "download" and ev.get("total"):
            mb, total = ev["done"] / 1e6, ev["total"] / 1e6
            on_event("download", "running", ev["done"] / ev["total"],
                     f"{ev.get('stream', 'video')}: {mb:.0f} of {total:.0f} MB")
        elif ev.get("stage") in ("merge", "probe"):
            on_event("download", "running", 1.0, "merging" if ev["stage"] == "merge" else "reading the file")

    meta, cached = ingest(url, cfg.data_dir, max_height=cfg.ingest.max_height, on_progress=dl)
    video_dir = cfg.data_dir / meta.id
    on_event("download", "done", 1.0, f"{mmss(meta.duration_s)} · {meta.height}p" + (" · cached" if cached else ""))

    # 2. shots and keyframes
    on_status("Detecting shots")
    on_event("shots", "running", 0.0, "scanning for cuts")

    def sh(ev: dict) -> None:
        if ev.get("stage") == "detect" and ev.get("total"):
            on_event("shots", "running", 0.8 * ev["done"] / ev["total"],
                     f"scanning for cuts: {mmss(ev['done'])} of {mmss(ev['total'])}")
        elif ev.get("stage") == "extract" and ev.get("total"):
            on_event("shots", "running", 0.8 + 0.2 * ev["done"] / ev["total"],
                     f"saving keyframe {ev['done']} of {ev['total']}")

    shots, cached = build_shots(video_dir, meta.file, params=cfg.shots, on_progress=sh)
    n = sum(len(s.keyframes) for s in shots)
    on_event("shots", "done", 1.0, f"{len(shots)} shots · {n} keyframes" + (" · cached" if cached else ""))

    # 3. speech
    on_status("Transcribing speech (minutes on first run)")
    on_event("speech", "running", None, "looking for YouTube captions")

    def sp(ev: dict) -> None:
        stage = ev.get("stage")
        if stage == "transcribe" and ev.get("total"):
            on_event("speech", "running", min(ev["done"] / ev["total"], 1.0),
                     f"Whisper: {mmss(ev['done'])} of {mmss(ev['total'])}")
        elif stage == "load_model":
            on_event("speech", "running", None, f"loading Whisper {ev.get('model', '')}")
        elif stage in ("check_language", "language_check"):
            on_event("speech", "running", None, "checking which language is spoken")

    transcript, cached = build_transcript(video_dir, meta, cfg.transcribe, device_pref=cfg.device, on_progress=sp)
    source = "YouTube captions" if transcript.source.startswith("youtube") else "Whisper"
    on_event("speech", "done", 1.0, f"{source} · {transcript.language} · {len(transcript.segments)} lines"
             + (" · cached" if cached else ""))

    # 4. captions
    cap_model = cfg.caption.model or cfg.vision.model
    vm = build_vision_model(replace(cfg.vision, model=cap_model))
    on_status(f"Captioning {n} keyframes with {cap_model} (about 20 s each on first run)")
    on_event("captions", "running", None, f"loading {cap_model}")
    started = time.perf_counter()

    def cp(ev: dict) -> None:
        on_status(f"Captioning keyframes: {ev['done']}/{ev['total']} new")
        todo, done = ev["total"], ev["done"]
        have = n - todo + done
        per = (time.perf_counter() - started) / max(done, 1)
        left = per * (todo - done)
        on_event("captions", "running", have / n if n else None,
                 f"{have} of {n} · about {mmss(left)} left" if done < todo else f"{have} of {n}")

    result = caption_keyframes(video_dir, shots, vm, cap_model, cfg.caption, on_progress=cp)
    if cap_model != (cfg.answer.model or cfg.vision.model) and result.newly_done:
        vm.unload()   # don't keep the 4B in memory next to the 8B that answers questions
    detail = f"{len(result.records)} of {n} keyframes"
    if result.errors:
        detail += f" · {result.errors} failed (Prepare again to retry)"
    on_event("captions", "done", 1.0, detail)

    # 5. search index
    on_status("Building the search index")
    on_event("index", "running", None, "loading the search models")

    def ix(ev: dict) -> None:
        stage = ev.get("stage")
        label = {"load": f"loading {ev.get('model', '')}", "embed_captions": "embedding frames and captions",
                 "embed_speech": "embedding the speech", "write": "writing the index"}.get(stage)
        if label:
            on_event("index", "running", None, label)

    device = pick_device(cfg.device)
    stats, cached = build_index(video_dir, result.records, transcript,
                                TextEmbedder(cfg.embed.text_model, device, cfg.embed.batch_size),
                                ImageEmbedder(cfg.embed.image_model, device, cfg.embed.batch_size), cfg.embed,
                                on_progress=ix)
    on_event("index", "done", 1.0, f"{stats.frames} frames · {stats.speech_chunks} speech chunks"
             + (" · cached" if cached else ""))
    on_status("Ready")
    return Prepared(meta.id, meta.title, video_dir)


@dataclass
class AskResult:
    answer: Answer
    search_ms: float
    total_s: float


class Asker:
    """Keeps the embedders and the answering model loaded across questions."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        device = pick_device(cfg.device)
        self.text = TextEmbedder(cfg.embed.text_model, device, cfg.embed.batch_size)
        self.image = ImageEmbedder(cfg.embed.image_model, device, cfg.embed.batch_size)
        self.model_name = cfg.answer.model or cfg.vision.model
        self.vision = build_vision_model(replace(cfg.vision, model=self.model_name))

    def moments(self, video_dir: Path, question: str) -> list:
        """Every keyframe, best match first (retrieval only, no answering model)."""
        frames, speech, scores = score_all(video_dir, question, self.text, self.image)
        return fuse(frames, speech, scores)

    def answer(self, video_dir: Path, question: str, moments: list,
               mode: str | None = None, on_step: StepFn | None = None) -> Answer:
        """Answer from ranked moments with the configured mode (or `mode` to override)."""
        mode = mode or self.cfg.answer.mode
        if mode == "look":
            return look_again(question, moments, self.vision, video_dir / "keyframes", video_dir / "crops",
                              self.cfg.answer, self.cfg.vision.max_image_side, on_step)
        if mode == "compare":
            return answer_question(question, moments, self.vision, video_dir / "keyframes", self.cfg.answer)
        raise ValueError(f"answer.mode must be 'look' or 'compare', not {mode!r}")

    def ask(self, video_dir: Path, question: str,
            mode: str | None = None, on_step: StepFn | None = None) -> AskResult:
        start = time.perf_counter()
        moments = self.moments(video_dir, question)
        search_ms = (time.perf_counter() - start) * 1000
        ans = self.answer(video_dir, question, moments, mode, on_step)
        return AskResult(ans, search_ms, time.perf_counter() - start)
