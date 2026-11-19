"""Measure retrieval against a hand-checked question set (evals/<video_id>.yaml).

Questions with an exact expected answer (on-screen text, a name) also list
`contains`: the answer text must include those strings, so a misread such as
সুজিত for সৃজিত counts as wrong without reading every answer by eye.

Questions marked `absent: true` ask about something that isn't in the video:
the right answer is "not found" / "can't tell". They have no retrieval rank and
count as wrong if the model answers them anyway.

A question is a hit at rank r when the r-th moment shows the answer (one of
the listed keyframes) or is playing the speech that contains it (its speech
chunk overlaps a listed span by at least a second).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from frameseek.answer import Moment


@dataclass
class Item:
    q: str
    kind: str
    frames: set[str]
    spans: list[tuple[float, float]]
    contains: list[str] = field(default_factory=list)   # the answer text must include all of these
    absent: bool = False   # not in the video: the right answer is to say so


def load_eval(path: Path) -> tuple[str, list[Item]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = [Item(d["q"], d.get("kind", ""), set(d.get("frames", [])),
                  [tuple(s) for s in d.get("spans", [])], list(d.get("contains", [])), bool(d.get("absent", False)))
             for d in data["questions"]]
    return data["video"], items


def is_hit(m: Moment, item: Item, speech: list[dict[str, Any]]) -> bool:
    if Path(m.file).stem in item.frames:
        return True
    if any(a - 1 <= m.t <= b + 1 for a, b in item.spans):
        return True
    chunks = {c["text"] for c in speech for a, b in item.spans if min(b, c["end"]) - max(a, c["start"]) >= 1}
    return bool(m.speech) and m.speech in chunks


def text_ok(answer: str, item: Item) -> bool | None:
    """Does the answer contain every expected string? None when the item has none to check."""
    if not item.contains:
        return None
    a = answer.casefold()
    return all(c.casefold() in a for c in item.contains)


def first_hit(moments: list[Moment], item: Item, speech: list[dict[str, Any]]) -> int | None:
    """1-based rank of the first correct moment, or None."""
    return next((r for r, m in enumerate(moments, start=1) if is_hit(m, item, speech)), None)


@dataclass
class Summary:
    n: int
    hit1: int
    hit_top: int
    mrr: float
    by_kind: dict[str, tuple[int, int]]


def summarise(items: list[Item], ranks: list[int | None], top_n: int) -> Summary:
    """Retrieval scores over the questions whose answer is in the video (absent ones have no rank)."""
    pairs = [(it, r) for it, r in zip(items, ranks) if not it.absent]
    items, ranks = [it for it, _ in pairs], [r for _, r in pairs]
    by_kind: dict[str, tuple[int, int]] = {}
    for it, r in zip(items, ranks):
        ok, total = by_kind.get(it.kind, (0, 0))
        by_kind[it.kind] = (ok + bool(r and r <= top_n), total + 1)
    return Summary(
        n=len(items),
        hit1=sum(r == 1 for r in ranks),
        hit_top=sum(bool(r and r <= top_n) for r in ranks),
        mrr=sum(1 / r for r in ranks if r) / max(len(items), 1),
        by_kind=by_kind,
    )


def outcome(answered_moment: object, hit: bool, item: Item) -> str:
    """right / wrong / abstained for a present item; abstained_ok / false_answer for an absent one."""
    if item.absent:
        return "abstained_ok" if answered_moment is None else "false_answer"
    if answered_moment is None:
        return "abstained"
    return "right" if hit else "wrong"


@dataclass
class AnswerSummary:
    right: int
    wrong: int
    abstained: int
    errors: int
    absent_ok: int
    false_answers: int

    @property
    def answered(self) -> int:
        return self.right + self.wrong + self.false_answers

    @property
    def accuracy_when_answering(self) -> float:
        return self.right / self.answered if self.answered else 0.0


def summarise_answers(outcomes: list[str]) -> AnswerSummary:
    c = {k: outcomes.count(k) for k in ("right", "wrong", "abstained", "error", "abstained_ok", "false_answer")}
    return AnswerSummary(c["right"], c["wrong"], c["abstained"], c["error"], c["abstained_ok"], c["false_answer"])
