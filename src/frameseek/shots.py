"""Split a video into shots and extract keyframes for each one.

Output in data/<video_id>/:

    shots.json     shots with start/end times and their keyframes, plus the
                   settings used (so changed settings invalidate the cache)
    keyframes/     kf_<shot>_<n>.jpg at full download resolution
    contact_sheet.jpg (optional) grid of all keyframes for eyeballing results

Detection runs at low resolution (cuts are obvious even when tiny). Keyframes
are then extracted at full resolution with a fast seek (see docs/03).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from frameseek.config import ShotsConfig as ShotParams
from frameseek.ingest import write_json_atomic

ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class Keyframe:
    t: float
    file: str


@dataclass
class Shot:
    index: int
    start_s: float
    end_s: float
    keyframes: list[Keyframe] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------- detection ----------

def detect_cuts(
    video: Path, params: ShotParams, on_progress: ProgressFn = lambda _: None
) -> list[tuple[float, float]]:
    """Return (start_s, end_s) for every shot, using PySceneDetect's AdaptiveDetector."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector

    stream = open_video(str(video))
    fps = float(stream.frame_rate)
    total_s = stream.duration.seconds

    manager = SceneManager()
    manager.auto_downscale = True   # analyse at ~256px wide: same cuts, ~10x less work
    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=params.adaptive_threshold,
            min_content_val=params.min_content_val,
            min_scene_len=max(1, round(params.min_shot_s * fps)),
        )
    )

    # Process in 5-second chunks so we can report progress between them.
    chunk_frames = max(1, int(5 * fps))
    while manager.detect_scenes(video=stream, duration=chunk_frames) > 0:
        on_progress({"stage": "detect", "done": stream.position.seconds, "total": total_s})

    scenes = manager.get_scene_list(start_in_scene=True)
    return [(round(a.seconds, 3), round(b.seconds, 3)) for a, b in scenes]


# ---------- choosing keyframes ----------

def plan_keyframe_times(start_s: float, end_s: float, every_s: float) -> list[float]:
    """Pick evenly spaced, centred timestamps inside a shot.

    A 6 s shot gets one frame at its middle. A 25 s shot (every_s=10) gets
    three, at the centres of three equal parts. Never the first frame, which
    often still contains the transition.
    """
    duration = max(0.0, end_s - start_s)
    n = max(1, math.ceil(duration / every_s)) if every_s > 0 else 1
    return [round(start_s + duration * (k + 0.5) / n, 3) for k in range(n)]


def extract_frame(video: Path, t: float, out: Path) -> None:
    """Fast seek (-ss before -i): jump to the keyframe before t, decode a short stretch.

    After a mid-stream seek the H.264 decoder sometimes logs harmless messages
    such as "mmco: unref short failure" (an instruction to drop a reference
    frame it never decoded). They are captured here and shown only if the
    extraction actually fails, so they don't break the progress display.
    """
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not extract frame at {t:.3f}s:\n{result.stderr.strip()}")


# ---------- the shots step ----------

def load_shots(video_dir: Path, params: ShotParams) -> list[Shot] | None:
    """Return cached shots only if they were made with the same params and all files exist."""
    path = video_dir / "shots.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("params") != asdict(params):
        return None
    shots = [
        Shot(s["index"], s["start_s"], s["end_s"], [Keyframe(**k) for k in s["keyframes"]])
        for s in data["shots"]
    ]
    kf_dir = video_dir / "keyframes"
    if not all((kf_dir / k.file).exists() for s in shots for k in s.keyframes):
        return None
    return shots


def build_shots(
    video_dir: Path,
    video_file: str,
    params: ShotParams = ShotParams(),
    force: bool = False,
    on_progress: ProgressFn = lambda _: None,
    detector=detect_cuts,
    extractor=extract_frame,
) -> tuple[list[Shot], bool]:
    """Detect shots and extract keyframes. Returns (shots, was_cached)."""
    if not force and (cached := load_shots(video_dir, params)) is not None:
        return cached, True

    video = video_dir / video_file
    cuts = detector(video, params, on_progress)

    kf_dir = video_dir / "keyframes"
    shutil.rmtree(kf_dir, ignore_errors=True)   # stale frames from older params
    kf_dir.mkdir(parents=True)

    shots = []
    for i, (start, end) in enumerate(cuts):
        times = plan_keyframe_times(start, end, params.every_s)
        shots.append(Shot(i, start, end, [Keyframe(t, f"kf_{i:04d}_{n:02d}.jpg") for n, t in enumerate(times)]))

    total = sum(len(s.keyframes) for s in shots)
    done = 0
    for shot in shots:
        for kf in shot.keyframes:
            extractor(video, kf.t, kf_dir / kf.file)
            done += 1
            on_progress({"stage": "extract", "done": done, "total": total})

    write_json_atomic(
        video_dir / "shots.json",
        {"params": asdict(params), "shots": [asdict(s) for s in shots]},
    )
    return shots, False


# ---------- contact sheet ----------

def contact_sheet(video_dir: Path, shots: list[Shot], cols: int = 6, thumb_w: int = 240) -> Path:
    """Tile every keyframe into one image, labelled with shot number and time."""
    from PIL import Image, ImageDraw

    kf_dir = video_dir / "keyframes"
    items = [(s.index, k) for s in shots for k in s.keyframes]
    if not items:
        raise ValueError("No keyframes to tile")

    with Image.open(kf_dir / items[0][1].file) as first:
        thumb_h = round(thumb_w * first.height / first.width)
    label_h = 18
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for n, (shot_idx, kf) in enumerate(items):
        x, y = (n % cols) * thumb_w, (n // cols) * (thumb_h + label_h)
        with Image.open(kf_dir / kf.file) as img:
            sheet.paste(img.convert("RGB").resize((thumb_w, thumb_h)), (x, y))
        mm, ss = divmod(kf.t, 60)
        draw.text((x + 4, y + thumb_h + 3), f"shot {shot_idx}  {int(mm)}:{ss:04.1f}", fill="black")

    out = video_dir / "contact_sheet.jpg"
    sheet.save(out, quality=85)
    return out
