"""Look again: check moments one at a time, zoom into the first hit, answer from the crop.

Section 7 shows the model the top 3 frames at once, each at full-frame
resolution. That misses small things (text on a T-shirt gets ~5 of ~1,240
tokens) and never sees the 4th-ranked frame. This loop:

  for each moment, best first, up to look_n:
      look   one frame + its caption and speech: is the answer here, and where (a box)?
      zoom   if it's in a small region: crop that box from the original keyframe and
             ask again. The crop must confirm it, which also catches a wrong "yes".
      check  if the answer quotes speech or on-screen text: is the quote really in that
             source? Then, for every answer: does it answer the question? (verify.py, Section 9)
      stop   at the first moment that passes
  if none: abstain ("not found" / "can't tell"), showing what was seen

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
from frameseek.models.vision import OllamaError, Timing, VisionModel
from frameseek.verify import Check, check_evidence, check_relevance
from frameseek.zoom import Box, parse_box, save_crop

LOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "present": {"type": "boolean"},
        "source": {"type": "string", "enum": ["image", "speech", "none"]},
        "quote": {"type": "string"},
        "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        "answer": {"type": "string"},
    },
    "required": ["seen", "present", "source", "quote", "box", "answer"],
}

ZOOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "present": {"type": "boolean"},
        "quote": {"type": "string"},
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["seen", "present", "quote", "answer", "confidence"],
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
        "- quote: the exact words that give the answer, copied character by character in their own",
        "  script: from the speech text above if source is speech, from the screen if it is text in",
        '  the image. "" if the answer is purely visual (a person, an object, a colour).',
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
        "- quote: any text in the crop that gives the answer, copied exactly in its own script;",
        '  "" if the answer is purely visual.',
        "- answer: one complete sentence in the same language as the question. Keep what the",
        "  whole frame showed (the setting, the people) and correct or add the details the crop",
        "  shows; quote any text exactly as it appears in the crop. Do not just repeat the",
        "  question back. Do not mention image or moment numbers.",
        "- confidence: high only if the crop shows it clearly.",
    ])


@dataclass
class Step:
    """One call in the loop, for the CLI table, the UI and the docs."""

    kind: str                      # "look", "zoom" or "check"
    moment: Moment
    rank: int                      # 1 = best-ranked moment
    present: bool
    seen: str
    source: str = ""
    box: Box | None = None
    crop: Path | None = None
    timing: Timing | None = None
    quote: str = ""
    check: Check | None = None


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

    def verified(m: Moment, rank: int, quote: str, source: str, where: str, description: str) -> bool:
        """Run the checks on a moment's answer; record them; False means treat it as "not here".

        With a quote: check 1 (is it in the source?) then check 2 (does it answer?).
        Without one (a visual answer): check 2 on the model's description of the frame.
        """
        if not cfg.verify:
            return True
        if quote:
            c = check_evidence(model, question, quote, source, where, m.t, cfg.quote_threshold)
        else:
            c = check_relevance(model, question, description, m.t)
            where = "visual answer"
        record(Step("check", m, rank, c.passed, c.reason, where, None, None, c.timing, quote, c))
        return c.passed

    def ask_model(image, prompt: str, schema: dict, kind: str, m: Moment, rank: int) -> tuple[dict, Timing] | None:
        """One look or zoom call; a failed call counts as "not here" instead of ending the loop."""
        try:
            return model.structured(image, prompt, schema, num_predict=cfg.num_predict)
        except OllamaError as exc:
            record(Step(kind, m, rank, False, f"error: {str(exc)[:120]}"))
            return None

    for rank, m in enumerate(candidates, start=1):
        got = ask_model(keyframes_dir / m.file, look_prompt(question, m), LOOK_SCHEMA, "look", m, rank)
        if got is None:
            continue
        out, t = got
        source = out.get("source", "none")
        present = bool(out.get("present")) and source != "none"
        if source == "speech" and not m.speech:   # can't be "in the speech" with no speech here
            present = False
        quote = (out.get("quote") or "").strip()
        seen = out.get("seen", "").strip()
        box = parse_box(out.get("box")) if source == "image" else None
        record(Step("look", m, rank, present, seen, source, box, None, t, quote))
        if not present:
            continue
        answer = out.get("answer", "").strip()

        if source == "speech":
            # Heard, not seen: nothing to zoom into. A speech claim needs its words quoted.
            if cfg.verify and not quote:
                record(Step("check", m, rank, False, "speech claim without a quote", "speech",
                            check=Check("failed", reason="speech claim without a quote")))
                continue
            if verified(m, rank, quote, m.speech, "speech", f"{seen}. {answer}"):
                return finish(answer, m, "medium", seen, None)
            continue

        if box is None or box.area > cfg.zoom_skip_area:
            # The answer is the whole scene: zooming adds nothing.
            if verified(m, rank, quote, m.screen_text, "on-screen text", f"{seen}. {answer}"):
                return finish(answer, m, "medium", seen, None)
            continue

        crop = crops_dir / f"{Path(m.file).stem}_{qkey}.jpg"
        save_crop(keyframes_dir / m.file, box, crop, cfg.zoom_margin, cfg.zoom_min_px, max_side)
        got = ask_model(crop, zoom_prompt(question, m, seen, answer), ZOOM_SCHEMA, "zoom", m, rank)
        if got is None:
            continue
        zout, zt = got
        confirmed = bool(zout.get("present"))
        zquote = (zout.get("quote") or "").strip() or quote
        record(Step("zoom", m, rank, confirmed, zout.get("seen", "").strip(), "image", box, crop, zt, zquote))
        if not confirmed:
            continue   # the close-up didn't confirm it: a false "yes" from the full frame
        zanswer = zout.get("answer", "").strip() or answer
        if verified(m, rank, zquote, m.screen_text, "on-screen text", f"{zout.get('seen', '').strip()}. {zanswer}"):
            return finish(zanswer, m,
                          zout.get("confidence", "medium"), zout.get("seen", "").strip(), crop)

    n = len(candidates)
    rejected = [s for s in steps if s.kind == "check" and not s.present]
    text = (f"Can't tell: {len(rejected)} possible answer(s) in the top {n} moments failed the evidence checks."
            if rejected else f"Not found in the top {n} moments.")
    return finish(text, None, "low",
                  "; ".join(f"{fmt_time(s.moment.t)}: {s.seen}" for s in steps if s.kind in ("look", "check")),
                  None)
