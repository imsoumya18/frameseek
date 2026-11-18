"""Look again: check moments one at a time, zoom into the first hit, answer from the crop.

Section 7 shows the model the top 3 frames at once, each at full-frame
resolution. That misses small things (text on a T-shirt gets ~5 of ~1,240
tokens) and never sees the 4th-ranked frame. This loop:

  for each moment, best first, up to look_n:
      look   one frame + its caption and speech: is the answer here, and where (a box)?
      zoom   if it's in a small region: crop that box from the original keyframe and
             ask again. The crop must confirm it, which also catches a wrong "yes".
      stop   at the first confirmed moment
  if none: say it isn't in the top look_n moments

Cost: one call per look (~1,500 tokens) and one per zoom. When the best-ranked
frame is right (most questions), that's 2 calls, less than Section 7's single
3-frame call. When nothing is found, it's look_n calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from frameseek.answer import Answer, Moment, fmt_time
from frameseek.config import AnswerConfig
from frameseek.models.vision import Timing, VisionModel
from frameseek.zoom import Box, parse_box, save_crop

LOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "present": {"type": "boolean"},
        "source": {"type": "string", "enum": ["image", "speech", "none"]},
        "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        "answer": {"type": "string"},
    },
    "required": ["seen", "present", "source", "box", "answer"],
}

ZOOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "present": {"type": "boolean"},
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["seen", "present", "answer", "confidence"],
}


def look_prompt(question: str, m: Moment) -> str:
    lines = [
        f"You are checking one moment of a video, at {fmt_time(m.t)}, for the answer to a question.",
        f"Caption (automatic, can miss details): {m.caption_text or '(none)'}",
    ]
    if m.speech:
        lines.append(f"Said around this time (automatic transcript, has spelling mistakes): {m.speech}")
    lines += [
        "",
        f"Question: {question}",
        "",
        "Fill the fields in this order:",
        "- seen: in a few words, what in the image (or the speech) relates to the question.",
        "- present: true only if the image clearly shows the answer, or the words asked about",
        "  are really in the speech text above. Otherwise false.",
        '- source: "image" if you see it, "speech" if it is in the speech text, "none" if not present.',
        "- box: where the answer is in the image, as [x1, y1, x2, y2] from 0 to 1000",
        "  (top-left to bottom-right). Tight around the thing asked about: the text, the cap,",
        "  the person. [0, 0, 1000, 1000] if it is the whole scene; [0, 0, 0, 0] if not in the image.",
        "- answer: one complete sentence in the same language as the question. Say who, what,",
        "  where, or the exact words, with details such as colours, clothing and setting.",
        "  Do not just repeat the question back. Do not mention image or moment numbers.",
    ]
    return "\n".join(lines)


def zoom_prompt(question: str, m: Moment, seen: str, full_frame_answer: str) -> str:
    return "\n".join([
        f"This is a zoomed-in crop of a video frame at {fmt_time(m.t)}, around: {seen}",
        f"Looking at the whole frame, the answer was: {full_frame_answer}",
        "",
        f"Question: {question}",
        "",
        "Fill the fields in this order:",
        "- seen: what the crop shows that relates to the question. If there is text, copy it",
        "  exactly, character by character, in its own script.",
        "- present: true only if the crop really shows the answer.",
        "- answer: one complete sentence in the same language as the question. Keep what the",
        "  whole frame showed (the setting, the people) and correct or add the details the crop",
        "  shows; quote any text exactly as it appears in the crop. Do not just repeat the",
        "  question back. Do not mention image or moment numbers.",
        "- confidence: high only if the crop shows it clearly.",
    ])


@dataclass
class Step:
    """One call in the loop, for the CLI table, the UI and the docs."""

    kind: str                      # "look" or "zoom"
    moment: Moment
    rank: int                      # 1 = best-ranked moment
    present: bool
    seen: str
    source: str = ""
    box: Box | None = None
    crop: Path | None = None
    timing: Timing | None = None


StepFn = Callable[[Step], None]


def sum_timings(ts: list[Timing]) -> Timing:
    return Timing(*(sum(getattr(t, f) for t in ts) for f in
                    ("load_s", "prefill_s", "prefill_tokens", "decode_s", "decode_tokens", "total_s")))


def look_again(
    question: str, moments: list[Moment], model: VisionModel, keyframes_dir: Path,
    crops_dir: Path, cfg: AnswerConfig, max_side: int = 1024, on_step: StepFn | None = None,
) -> Answer:
    candidates = moments[: cfg.look_n]
    steps: list[Step] = []
    qkey = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]

    def record(step: Step) -> None:
        steps.append(step)
        if on_step:
            on_step(step)

    def finish(text: str, moment: Moment | None, confidence: str, evidence: str, crop: Path | None) -> Answer:
        timing = sum_timings([s.timing for s in steps if s.timing])
        return Answer(text, moment, confidence, evidence, timing, candidates, steps=steps, crop=crop)

    for rank, m in enumerate(candidates, start=1):
        out, t = model.structured(keyframes_dir / m.file, look_prompt(question, m), LOOK_SCHEMA,
                                  num_predict=cfg.num_predict)
        source = out.get("source", "none")
        present = bool(out.get("present")) and source != "none"
        if source == "speech" and not m.speech:   # can't be "in the speech" with no speech here
            present = False
        box = parse_box(out.get("box")) if source == "image" else None
        record(Step("look", m, rank, present, out.get("seen", "").strip(), source, box, None, t))
        if not present:
            continue

        answer = out.get("answer", "").strip()
        if source == "speech" or box is None or box.area > cfg.zoom_skip_area:
            # Heard rather than seen, or the answer is the whole scene: zooming adds nothing.
            return finish(answer, m, "medium", steps[-1].seen, None)

        crop = crops_dir / f"{Path(m.file).stem}_{qkey}.jpg"
        save_crop(keyframes_dir / m.file, box, crop, cfg.zoom_margin, cfg.zoom_min_px, max_side)
        zout, zt = model.structured(crop, zoom_prompt(question, m, steps[-1].seen, answer), ZOOM_SCHEMA,
                                    num_predict=cfg.num_predict)
        confirmed = bool(zout.get("present"))
        record(Step("zoom", m, rank, confirmed, zout.get("seen", "").strip(), "image", box, crop, zt))
        if confirmed:
            return finish(zout.get("answer", "").strip() or answer, m,
                          zout.get("confidence", "medium"), steps[-1].seen, crop)
        # The close-up didn't confirm it: a false "yes" from the full frame. Keep looking.

    n = len(candidates)
    return finish(f"Not found in the top {n} moments.", None, "low",
                  "; ".join(f"{fmt_time(s.moment.t)}: {s.seen}" for s in steps if s.kind == "look"), None)
