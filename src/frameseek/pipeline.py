"""The whole flow in two calls, shared by the CLI and the web app.

prepare(url)  indexing: ingest -> shots -> transcribe -> caption -> index.
              Every step is cached, so a prepared video returns in seconds.
Asker.ask()   asking: search -> fuse into moments -> answer from the frames.
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
from frameseek.ingest import ingest, require_tools
from frameseek.models.embed import ImageEmbedder, TextEmbedder
from frameseek.models.vision import build_vision_model
from frameseek.shots import build_shots
from frameseek.transcribe import build_transcript

StatusFn = Callable[[str], None]


@dataclass
class Prepared:
    video_id: str
    title: str
    video_dir: Path


def prepare(url: str, cfg: Config, on_status: StatusFn = lambda _: None) -> Prepared:
    """Run every indexing step that isn't cached yet."""
    require_tools()
    on_status("Downloading video (or using the cached copy)")
    meta, _ = ingest(url, cfg.data_dir, max_height=cfg.ingest.max_height)
    video_dir = cfg.data_dir / meta.id

    on_status("Detecting shots")
    shots, _ = build_shots(video_dir, meta.file, params=cfg.shots)

    on_status("Transcribing speech (minutes on first run)")
    transcript, _ = build_transcript(video_dir, meta, cfg.transcribe, device_pref=cfg.device)

    cap_model = cfg.caption.model or cfg.vision.model
    vm = build_vision_model(replace(cfg.vision, model=cap_model))
    n = sum(len(s.keyframes) for s in shots)
    on_status(f"Captioning {n} keyframes with {cap_model} (about 20 s each on first run)")
    result = caption_keyframes(
        video_dir, shots, vm, cap_model, cfg.caption,
        on_progress=lambda ev: on_status(f"Captioning keyframes: {ev['done']}/{ev['total']} new"),
    )

    on_status("Building the search index")
    device = pick_device(cfg.device)
    build_index(video_dir, result.records, transcript,
                TextEmbedder(cfg.embed.text_model, device, cfg.embed.batch_size),
                ImageEmbedder(cfg.embed.image_model, device, cfg.embed.batch_size), cfg.embed)
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

    def ask(self, video_dir: Path, question: str) -> AskResult:
        start = time.perf_counter()
        moments = self.moments(video_dir, question)
        search_ms = (time.perf_counter() - start) * 1000
        ans = answer_question(question, moments, self.vision, video_dir / "keyframes", self.cfg.answer)
        return AskResult(ans, search_ms, time.perf_counter() - start)
