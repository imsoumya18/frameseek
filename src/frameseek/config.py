"""Loads config.yaml into typed settings objects.

Lookup order for the config file:
1. the path in the FRAMESEEK_CONFIG environment variable
2. ./config.yaml in the current directory
3. built-in defaults (so the code still runs with no file at all)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VisionConfig:
    backend: str = "ollama"
    model: str = "qwen3-vl:8b-instruct"
    host: str = "http://localhost:11434"
    max_image_side: int = 1024
    timeout_s: float = 180.0
    keep_alive: str = "30m"
    num_ctx: int = 8192           # context window; Ollama's default 4096 is too small for 3 frames + text


@dataclass
class IngestConfig:
    max_height: int = 720   # download resolution cap; the VLM never sees more than ~1024px


@dataclass
class ShotsConfig:
    """Every setting that changes shot detection. Stored in shots.json as the cache key."""
    adaptive_threshold: float = 3.0   # cut if a frame's change is this many times its neighbours'
    min_content_val: float = 15.0     # ...and its raw change score is at least this
    min_shot_s: float = 0.6           # ignore a new cut this soon after the previous one
    every_s: float = 10.0             # extra keyframes inside shots longer than this


@dataclass
class TranscribeConfig:
    source: str = "auto"            # auto (captions, else Whisper) | youtube | whisper
    whisper_model: str = "large-v3-turbo"  # best for Indian languages; small = quick tests
    language_model: str = "small"   # detects the language; turbo is poor at this (see docs/04)
    vad: bool = True                # transcribe only stretches with speech
    beam_size: int = 5              # 1 = greedy (faster), 5 = beam search (more accurate)
    language: str = ""              # force a language code, e.g. "bn"; empty = detect
    device: str = "auto"            # auto (CUDA, else CPU) | cpu | cuda | mps (experimental)


@dataclass
class CaptionConfig:
    model: str = ""              # model for indexing captions; empty = use vision.model
    max_image_side: int = 1024   # Ollama resizes images itself; sending less only loses detail
    num_predict: int = 300       # hard cap on output tokens per frame


@dataclass
class EmbedConfig:
    text_model: str = "intfloat/multilingual-e5-small"     # question <-> captions and speech (100 languages)
    image_model: str = "google/siglip2-base-patch16-224"   # question <-> frame pixels
    speech_chunk_s: float = 15.0                           # merge transcript lines into ~15 s chunks
    batch_size: int = 16


@dataclass
class AnswerConfig:
    model: str = ""             # answering model; empty = vision.model (the 8B)
    mode: str = "look"          # look: one frame at a time + zoom (Section 8); compare: top_n frames at once (Section 7)
    top_n: int = 3              # compare mode: moments (and frames) shown together
    look_n: int = 6             # look mode: most moments checked before giving up
    zoom_margin: float = 0.3    # look mode: crop margin, as a fraction of the box's size on each side
    zoom_min_px: int = 160      # look mode: smallest crop side in original pixels
    zoom_skip_area: float = 0.5 # look mode: boxes bigger than this fraction of the frame aren't zoomed
    num_predict: int = 300


@dataclass
class Config:
    data_dir: Path = Path("data")
    device: str = "auto"
    vision: VisionConfig = field(default_factory=VisionConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    shots: ShotsConfig = field(default_factory=ShotsConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)


def _pick(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys the dataclass knows about, so a typo in YAML fails loudly."""
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys in config: {sorted(unknown)}")
    return raw


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("FRAMESEEK_CONFIG", "config.yaml"))
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    vision_raw = raw.pop("vision", {}) or {}
    ingest_raw = raw.pop("ingest", {}) or {}
    shots_raw = raw.pop("shots", {}) or {}
    transcribe_raw = raw.pop("transcribe", {}) or {}
    caption_raw = raw.pop("caption", {}) or {}
    embed_raw = raw.pop("embed", {}) or {}
    answer_raw = raw.pop("answer", {}) or {}
    top = _pick(Config, raw)
    return Config(
        data_dir=Path(top.get("data_dir", "data")),
        device=top.get("device", "auto"),
        vision=VisionConfig(**_pick(VisionConfig, vision_raw)),
        ingest=IngestConfig(**_pick(IngestConfig, ingest_raw)),
        shots=ShotsConfig(**_pick(ShotsConfig, shots_raw)),
        transcribe=TranscribeConfig(**_pick(TranscribeConfig, transcribe_raw)),
        caption=CaptionConfig(**_pick(CaptionConfig, caption_raw)),
        embed=EmbedConfig(**_pick(EmbedConfig, embed_raw)),
        answer=AnswerConfig(**_pick(AnswerConfig, answer_raw)),
    )
