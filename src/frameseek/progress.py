"""The five preparation steps, their progress, and what's already done on disk.

The web app shows these as a checklist: done steps get a tick, the running
step a progress bar, the rest wait their turn. When a link is pasted, the
checklist is filled in from the files already in data/<video_id>/, so a video
prepared yesterday shows all five ticks before anything runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

STEPS: list[tuple[str, str]] = [
    ("download", "Download the video"),
    ("shots", "Find shots and keyframes"),
    ("speech", "Transcribe the speech"),
    ("captions", "Describe each keyframe"),
    ("index", "Build the search index"),
]


@dataclass
class StepState:
    state: str = "waiting"           # waiting / running / done / failed
    fraction: float | None = None    # 0-1 while running, when known
    detail: str = ""


@dataclass
class Checklist:
    steps: dict[str, StepState] = field(default_factory=lambda: {k: StepState() for k, _ in STEPS})

    def set(self, key: str, state: str, fraction: float | None = None, detail: str = "") -> None:
        self.steps[key] = StepState(state, fraction, detail)

    @property
    def done(self) -> int:
        return sum(s.state == "done" for s in self.steps.values())

    @property
    def overall(self) -> float:
        """0-1 over all five steps, counting the running step's own progress."""
        total = 0.0
        for s in self.steps.values():
            total += 1.0 if s.state == "done" else (s.fraction or 0.0) if s.state == "running" else 0.0
        return total / len(STEPS)


EventFn = Callable[[str, str, float | None, str], None]   # (step, state, fraction, detail)


def mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def from_disk(video_dir: Path) -> Checklist:
    """What's already done for this video, judged from the files in its folder.

    It reads the files, not the settings they were made with: a step that the
    current config would redo still shows as done here, and prepare() redoes it.
    """
    c = Checklist()
    meta_path = video_dir / "meta.json"
    if not meta_path.exists():
        return c
    meta = json.loads(meta_path.read_text())
    if (video_dir / meta.get("file", "video.mp4")).exists():
        size = (video_dir / meta.get("file", "video.mp4")).stat().st_size / 1e6
        c.set("download", "done", 1.0, f"{mmss(meta.get('duration_s') or 0)} · {size:.0f} MB")

    total = 0
    if (video_dir / "shots.json").exists():
        shots = json.loads((video_dir / "shots.json").read_text()).get("shots", [])
        total = sum(len(s.get("keyframes", [])) for s in shots)
        c.set("shots", "done", 1.0, f"{len(shots)} shots · {total} keyframes")

    if (video_dir / "transcript.json").exists():
        t = json.loads((video_dir / "transcript.json").read_text())
        source = "YouTube captions" if str(t.get("source", "")).startswith("youtube") else "Whisper"
        c.set("speech", "done", 1.0, f"{source} · {t.get('language', '?')} · {len(t.get('segments', []))} lines")

    cap = video_dir / "captions.jsonl"
    if cap.exists() and total:
        done = set()
        for line in cap.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("caption"):
                done.add(rec.get("file"))
        n = min(len(done), total)
        if n >= total:
            c.set("captions", "done", 1.0, f"{total} keyframes described")
        elif n:
            c.set("captions", "waiting", n / total, f"{n} of {total} done; the rest on Prepare")

    if (video_dir / "index_meta.json").exists() and (video_dir / "index.lance").exists():
        c.set("index", "done", 1.0, "ready to search")
    return c
