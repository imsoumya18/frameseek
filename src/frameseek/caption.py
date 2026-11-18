"""Describe every keyframe with the vision model, as structured JSON.

Output: data/<video_id>/captions.jsonl, one line per keyframe, appended as each
frame finishes so a crash loses at most one frame:

    {"file": "kf_0012_00.jpg", "t": 41.2, "shot": 12, "key": "3f9a...",
     "model": "qwen3-vl:8b-instruct", "side": 640,
     "caption": {"summary": ..., "shot": ..., "people": ..., "objects": [...], ...},
     "timing": {"prefill_tokens": 412, "prefill_s": 5.6, "decode_tokens": 58, "decode_s": 3.8}}

`key` is a hash of everything that changes a caption (model, image size, prompt,
schema). Lines with another key are ignored, so changing any of those
re-captions only what is needed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from frameseek.config import CaptionConfig
from frameseek.models.vision import OllamaError, VisionModel
from frameseek.shots import Shot

ProgressFn = Callable[[dict[str, Any]], None]

SCHEMA_VERSION = 2

CAPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "shot": {"type": "string",
                 "enum": ["extreme close-up", "close-up", "medium", "wide", "extreme wide", "other"]},
        "people": {
            "type": "object",
            "properties": {"count": {"type": "integer"}, "description": {"type": "string"}},
            "required": ["count", "description"],
        },
        "objects": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "colour": {"type": "string"}},
                "required": ["name", "colour"],
            },
        },
        "text_on_screen": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        "setting": {"type": "string"},
        "action": {"type": "string"},
    },
    "required": ["summary", "shot", "people", "objects", "text_on_screen", "setting", "action"],
}

CAPTION_PROMPT = """You are indexing a video frame so it can be searched later.
Describe only what is clearly visible. Write in English. Be brief: every word costs time.

- summary: one sentence, at most 15 words.
- shot: the camera framing.
- people: how many are visible; description in at most 8 words (clothing, what they hold).
- objects: up to 6 notable objects; name in 1-3 words, colour in 1-2 plain words (red, dark blue, gold).
- text_on_screen: each readable piece of text, copied exactly in its original script. Do not translate.
- setting: at most 6 words (place, indoor or outdoor, time of day).
- action: at most 6 words.

If something is unclear, leave it out or use an empty string. Never guess text you cannot read."""


def cache_key(model: str, side: int) -> str:
    """Hash of every input that changes a caption."""
    blob = json.dumps(
        {"model": model, "side": side, "prompt": CAPTION_PROMPT,
         "schema": CAPTION_SCHEMA, "version": SCHEMA_VERSION},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class CaptionResult:
    records: list[dict[str, Any]]   # one per keyframe, in shot order
    newly_done: int
    from_cache: int
    errors: int


def load_done(path: Path, key: str) -> dict[str, dict[str, Any]]:
    """Return successful records for this key, by keyframe file. Errors are retried."""
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue   # a line half-written during a crash
        if rec.get("key") == key and rec.get("caption") is not None:
            done[rec["file"]] = rec
    return done


def caption_keyframes(
    video_dir: Path,
    shots: list[Shot],
    model: VisionModel,
    model_name: str,
    cfg: CaptionConfig,
    force: bool = False,
    on_progress: ProgressFn = lambda _: None,
) -> CaptionResult:
    """Caption every keyframe not already captioned with the same settings."""
    path = video_dir / "captions.jsonl"
    if force and path.exists():
        path.unlink()

    key = cache_key(model_name, cfg.max_image_side)
    done = load_done(path, key)
    frames = [(s.index, k) for s in shots for k in s.keyframes]
    todo = [(i, k) for i, k in frames if k.file not in done]

    errors = 0
    with path.open("a", encoding="utf-8") as out:
        for n, (shot_idx, kf) in enumerate(todo, start=1):
            rec: dict[str, Any] = {"file": kf.file, "t": kf.t, "shot": shot_idx, "key": key,
                                   "model": model_name, "side": cfg.max_image_side}
            try:
                caption, timing = model.structured(
                    video_dir / "keyframes" / kf.file, CAPTION_PROMPT, CAPTION_SCHEMA,
                    max_side=cfg.max_image_side, num_predict=cfg.num_predict,
                )
                rec["caption"] = caption
                rec["timing"] = {"prefill_tokens": timing.prefill_tokens, "prefill_s": round(timing.prefill_s, 2),
                                 "decode_tokens": timing.decode_tokens, "decode_s": round(timing.decode_s, 2)}
                done[kf.file] = rec
            except OllamaError as exc:
                rec["caption"] = None
                rec["error"] = str(exc)
                errors += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()   # on disk now: a crash after this line keeps this frame
            on_progress({"stage": "caption", "done": n, "total": len(todo), "record": rec})

    records = [done[k.file] for _, k in frames if k.file in done]
    return CaptionResult(records, newly_done=len(todo) - errors,
                         from_cache=len(frames) - len(todo), errors=errors)
