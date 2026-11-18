"""Tests for shot detection and keyframe planning.

Unit tests use fake detectors and extractors. The integration test builds a
real 3-shot video with ffmpeg and is skipped if ffmpeg is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from frameseek.shots import ShotParams, build_shots, contact_sheet, plan_keyframe_times


# ---------- keyframe planning ----------

def test_short_shot_gets_middle_frame() -> None:
    assert plan_keyframe_times(10.0, 16.0, every_s=10) == [13.0]


def test_long_shot_gets_evenly_spaced_frames() -> None:
    # 25 s shot, every 10 s -> 3 parts of ~8.33 s, frames at their centres
    assert plan_keyframe_times(0.0, 25.0, every_s=10) == [4.167, 12.5, 20.833]


def test_never_picks_the_first_frame() -> None:
    assert all(t > 5.0 for t in plan_keyframe_times(5.0, 5.4, every_s=10))


# ---------- caching ----------

def _fake_detector(video, params, on_progress):
    on_progress({"stage": "detect", "done": 30.0, "total": 30.0})
    return [(0.0, 4.0), (4.0, 7.0), (7.0, 30.0)]


def _fake_extractor(video, t, out: Path):
    out.write_bytes(b"jpg")


def _setup(tmp_path: Path) -> Path:
    (tmp_path / "video.mp4").write_bytes(b"fake")
    return tmp_path


def test_builds_shots_and_writes_json(tmp_path: Path) -> None:
    vdir = _setup(tmp_path)
    shots, cached = build_shots(vdir, "video.mp4", detector=_fake_detector, extractor=_fake_extractor)

    assert cached is False
    assert [len(s.keyframes) for s in shots] == [1, 1, 3]   # 23 s shot -> 3 frames
    data = json.loads((vdir / "shots.json").read_text())
    assert data["params"]["every_s"] == 10.0
    assert len(list((vdir / "keyframes").glob("*.jpg"))) == 5


def test_second_run_is_cached(tmp_path: Path) -> None:
    vdir = _setup(tmp_path)
    build_shots(vdir, "video.mp4", detector=_fake_detector, extractor=_fake_extractor)

    def boom(*a, **k):
        raise AssertionError("should not re-detect")

    _, cached = build_shots(vdir, "video.mp4", detector=boom, extractor=boom)
    assert cached is True


def test_changed_params_invalidate_cache(tmp_path: Path) -> None:
    vdir = _setup(tmp_path)
    build_shots(vdir, "video.mp4", detector=_fake_detector, extractor=_fake_extractor)
    shots, cached = build_shots(vdir, "video.mp4", params=ShotParams(every_s=5.0),
                                detector=_fake_detector, extractor=_fake_extractor)
    assert cached is False
    assert len(shots[2].keyframes) == 5   # 23 s / 5 s


def test_missing_keyframe_invalidates_cache(tmp_path: Path) -> None:
    vdir = _setup(tmp_path)
    build_shots(vdir, "video.mp4", detector=_fake_detector, extractor=_fake_extractor)
    next((vdir / "keyframes").glob("*.jpg")).unlink()
    _, cached = build_shots(vdir, "video.mp4", detector=_fake_detector, extractor=_fake_extractor)
    assert cached is False


# ---------- real video ----------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_video_with_two_cuts(tmp_path: Path) -> None:
    """Test pattern (4 s) -> solid red (3 s) -> fractal (5 s): two hard cuts."""
    video = tmp_path / "video.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=d=4:s=320x180:r=25",
         "-f", "lavfi", "-i", "color=c=red:d=3:s=320x180:r=25",
         "-f", "lavfi", "-i", "mandelbrot=s=320x180:r=25",
         "-filter_complex", "[2]trim=duration=5,setpts=PTS-STARTPTS[m];[0][1][m]concat=n=3:v=1:a=0[v]",
         "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    shots, _ = build_shots(tmp_path, "video.mp4")

    assert [(s.start_s, s.end_s) for s in shots] == [(0.0, 4.0), (4.0, 7.0), (7.0, 12.0)]
    assert all((tmp_path / "keyframes" / k.file).exists() for s in shots for k in s.keyframes)
    assert contact_sheet(tmp_path, shots).exists()
