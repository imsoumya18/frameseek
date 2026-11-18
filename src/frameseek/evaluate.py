"""Measure retrieval against a hand-checked question set (evals/<video_id>.yaml).

A question is a hit at rank r when the r-th moment shows the answer (one of
the listed keyframes) or is playing the speech that contains it (its speech
chunk overlaps a listed span by at least a second).
"""

from __future__ import annotations

from dataclasses import dataclass
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


def load_eval(path: Path) -> tuple[str, list[Item]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = [Item(d["q"], d.get("kind", ""), set(d.get("frames", [])),
                  [tuple(s) for s in d.get("spans", [])]) for d in data["questions"]]
    return data["video"], items


def is_hit(m: Moment, item: Item, speech: list[dict[str, Any]]) -> bool:
    if Path(m.file).stem in item.frames:
        return True
    if any(a - 1 <= m.t <= b + 1 for a, b in item.spans):
        return True
    chunks = {c["text"] for c in speech for a, b in item.spans if min(b, c["end"]) - max(a, c["start"]) >= 1}
    return bool(m.speech) and m.speech in chunks


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
