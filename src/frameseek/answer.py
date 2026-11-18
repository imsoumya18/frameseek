"""Score every keyframe against the question, then answer from the evidence.

Stage 1, retrieval (cheap, aims for recall): each source (captions, frame
pixels, speech) scores the whole video; each source's scores are turned into
z-scores so their scales match, and a keyframe's score is the sum. A moment is
a keyframe with its caption and the speech playing at that time.

Stage 2, answering (expensive, aims for precision): the top moments' keyframes,
captions and speech go to the vision model, which must answer in a fixed JSON
shape and point at the moment it used (or 0 for none of them).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frameseek.config import AnswerConfig
from frameseek.models.vision import Timing, VisionModel


@dataclass
class Moment:
    t: float                                # where the player should jump
    file: str                               # keyframe shown to the answering model
    summary: str = ""
    caption_text: str = ""
    speech: str = ""                        # what was said around this time
    parts: dict[str, float] = field(default_factory=dict)  # each source's z-score, e.g. {"frames": 1.9}
    score: float = 0.0                      # sum of the parts


SOURCES = ("captions", "frames", "speech")


def zscore(x: np.ndarray) -> np.ndarray:
    """How many standard deviations each item sits above this question's average.

    Raw scores are not comparable across models (SigLIP2 ~0.1, e5 ~0.8), but
    "2 standard deviations above the rest of the video" means the same thing
    for every source.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else np.zeros_like(x)


def covering_chunk(frames: list[dict], speech: list[dict]) -> list[int]:
    """Index of the speech chunk playing at each frame's time, or -1 for none."""
    out = []
    for f in frames:
        hit = next((i for i, c in enumerate(speech) if c["start"] - 1.0 <= f["t"] <= c["end"] + 1.0), -1)
        out.append(hit)
    return out


def fuse(frames: list[dict], speech: list[dict], scores: dict[str, np.ndarray]) -> list[Moment]:
    """Rank every keyframe by the sum of its z-scores from the three sources, best first.

    scores["captions"] and scores["frames"] have one value per frame (NaN for
    frames whose caption is empty); scores["speech"] has one value per chunk.
    A frame inherits the z-score of the chunk being sung or spoken at its time,
    so a lyric match lifts every frame in that chunk, and the pixels and
    captions decide between them.

    Replaced reciprocal rank fusion (Section 7, v0): ranks threw away how far
    ahead a match was, and the speech list (only 8 chunks, all of them always
    returned) added score to some frame for every question.
    """
    n = len(frames)
    parts: dict[str, np.ndarray] = {}
    cap = np.asarray(scores.get("captions", np.full(n, np.nan)), dtype=np.float64)
    if n and np.isfinite(cap).any():
        cap = np.where(np.isfinite(cap), cap, np.nanmin(cap))   # empty captions: lowest, never a match
        parts["captions"] = zscore(cap)
    if n and len(scores.get("frames", [])):
        parts["frames"] = zscore(scores["frames"])
    cover = covering_chunk(frames, speech)
    if speech and len(scores.get("speech", [])):
        zs = zscore(scores["speech"])
        parts["speech"] = np.array([zs[i] if i >= 0 else 0.0 for i in cover])   # no speech: neutral

    total = sum(parts.values()) if parts else np.zeros(n)
    moments = [
        Moment(t=f["t"], file=f["file"], summary=f.get("summary", ""), caption_text=f.get("text", ""),
               speech=speech[cover[j]]["text"] if cover[j] >= 0 else "",
               parts={k: float(v[j]) for k, v in parts.items()}, score=float(total[j]))
        for j, f in enumerate(frames)
    ]
    return sorted(moments, key=lambda m: m.score, reverse=True)


def describe_parts(m: Moment) -> str:
    """e.g. 'frames +1.9, captions +0.9, speech -0.1', strongest first."""
    return ", ".join(f"{k} {v:+.1f}" for k, v in sorted(m.parts.items(), key=lambda kv: -kv[1]))


# ---------- answering ----------

# Field order matters: Ollama generates properties in schema order, so the model
# writes what it sees (evidence) before choosing a moment and phrasing the answer.
# With "answer" first, it wrote "0" or "moment 1" there before looking at anything.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "moment": {"type": "integer"},
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["evidence", "moment", "answer", "confidence"],
}

_BARE_MOMENT = re.compile(r"^\s*(moment\s*)?\d+\s*\.?\s*$", re.IGNORECASE)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def build_prompt(question: str, moments: list[Moment]) -> str:
    lines = [
        "You are answering a question about a video. Below are candidate moments from it.",
        "Image N is the keyframe of moment N. Look at the images themselves: captions can miss details.",
        "",
    ]
    for i, m in enumerate(moments, start=1):
        lines.append(f"Moment {i} at {fmt_time(m.t)} (image {i})")
        lines.append(f"  Caption: {m.caption_text or '(none)'}")
        if m.speech:
            lines.append(f"  Said around this time: {m.speech}")
        lines.append("")
    lines += [
        f"Question: {question}",
        "",
        "Fill the fields in this order:",
        "- evidence: first look at each image and its speech, and say what relates to the question.",
        "- moment: the number of the moment that best answers the question, or 0 if none of them does.",
        "  A moment answers it if its image shows it, OR if what was said at that time contains it:",
        "  for questions about lyrics or speech, the speech text is the evidence. It is an automatic",
        "  transcript with spelling mistakes, but the words must really be in that moment's text;",
        "  if they are in none of them, use 0. Never claim something was said that is not written above.",
        "  If several moments show it, pick the clearest one; do not use 0 for that.",
        "- answer: a complete sentence that answers the question, in the same language as the question.",
        "  Say what is there: who, what, where, or the exact words. Do not repeat the question back.",
        "  Do not mention moment or image numbers; the user sees the time separately.",
        "  When quoting text on screen, read it from the image; the caption's copy can be misspelled.",
        "  If no moment answers it, say so plainly.",
        "- confidence: high only if the image or speech clearly shows it.",
        "Use only what these moments show or say, not outside knowledge about the video.",
    ]
    return "\n".join(lines)


@dataclass
class Answer:
    text: str
    moment: Moment | None        # None when the model chose 0 (no moment answers it)
    confidence: str
    evidence: str
    timing: Timing
    candidates: list[Moment]


def answer_question(
    question: str, moments: list[Moment], model: VisionModel, keyframes_dir: Path, cfg: AnswerConfig
) -> Answer:
    top = moments[: cfg.top_n]
    if not top:
        return Answer("The index has nothing to search for this video.", None, "low", "",
                      Timing(0, 0, 0, 0, 0, 0), [])
    out, timing = model.structured(
        [keyframes_dir / m.file for m in top], build_prompt(question, top), ANSWER_SCHEMA,
        num_predict=cfg.num_predict,
    )
    idx = out.get("moment", 0)
    chosen = top[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(top) else None
    text, evidence = out.get("answer", "").strip(), out.get("evidence", "").strip()
    if not text or _BARE_MOMENT.match(text):   # e.g. "0" or "moment 1": not an answer
        text = evidence or "No answer."
    return Answer(text, chosen, out.get("confidence", "low"), evidence, timing, top)
