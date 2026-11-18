"""Download a YouTube video once and cache it under data/<video_id>/.

Layout after a successful ingest:

    data/<video_id>/
        video.mp4     the merged video + audio file
        meta.json     title, duration, fps, codecs, keyframe spacing, captions

Re-running ingest on the same video (via any URL form) is a no-op: the video
ID is the cache key, and the step is skipped when meta.json and the file exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}

ProgressFn = Callable[[dict[str, Any]], None]


# ---------- video ID ----------

def parse_video_id(url_or_id: str) -> str:
    """Return the 11-character video ID from any common YouTube URL form.

    Handles watch?v=, youtu.be/, /shorts/, /embed/, /live/ and bare IDs.
    Extra parameters such as &t=45 or &list=... are ignored.
    """
    s = url_or_id.strip()
    if ID_RE.match(s):
        return s

    parsed = urlparse(s if "://" in s else "https://" + s)
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.split("/") if p]

    candidate = None
    if host == "youtu.be" and parts:
        candidate = parts[0]
    elif host in YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            candidate = parts[1]

    if candidate and ID_RE.match(candidate):
        return candidate
    raise ValueError(f"Not a recognisable YouTube video URL or ID: {url_or_id!r}")


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# ---------- metadata ----------

@dataclass
class VideoMeta:
    id: str
    url: str
    file: str
    title: str = ""
    channel: str = ""
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    vcodec: str = ""
    acodec: str = ""
    keyframe_interval_s: float | None = None
    language: str | None = None
    manual_subtitles: list[str] = field(default_factory=list)
    has_auto_captions: bool = False


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write to a temp file, then rename: readers never see a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def load_meta(video_dir: Path) -> VideoMeta | None:
    meta_path = video_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = VideoMeta(**json.loads(meta_path.read_text()))
    return meta if (video_dir / meta.file).exists() else None


# ---------- probing with ffprobe ----------

def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"Missing {', '.join(missing)}. Install ffmpeg "
            "(macOS: brew install ffmpeg, Ubuntu: apt install ffmpeg, Windows: winget install ffmpeg)."
        )


def parse_fps(rate: str) -> float:
    """ffprobe reports frame rates as fractions like '30000/1001' (= 29.97)."""
    if not rate or rate == "0/0":
        return 0.0
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1)


def probe(path: Path) -> dict[str, Any]:
    """Read stream info, and the average spacing between keyframes."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})

    # -skip_frame nokey makes the decoder touch only I-frames, so this is fast
    # even for long videos (see docs/02-video-fundamentals.md).
    kf = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=300,
    )
    times = []
    for line in kf.stdout.splitlines():
        # ffprobe may append extra comma-separated fields; the first is pts_time.
        value = line.split(",")[0].strip()
        if value and value != "N/A":
            times.append(float(value))
    gaps = [b - a for a, b in zip(times, times[1:])]

    return {
        "duration_s": float(info.get("format", {}).get("duration", 0.0)),
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "fps": round(parse_fps(video.get("avg_frame_rate", "")), 3),
        "vcodec": video.get("codec_name", ""),
        "acodec": audio.get("codec_name", ""),
        "keyframe_interval_s": round(sum(gaps) / len(gaps), 2) if gaps else None,
    }


# ---------- downloading with yt-dlp ----------

def download_with_ytdlp(
    url: str, tmp_dir: Path, max_height: int, on_progress: ProgressFn
) -> tuple[Path, dict[str, Any]]:
    """Download best video <= max_height + best audio, merged into one file.

    Prefers H.264 (avc1) video and m4a audio so the result is a plain .mp4
    that every tool can read; falls back to whatever is available.
    """
    from yt_dlp import YoutubeDL  # imported here so tests don't need network

    def hook(d: dict[str, Any]) -> None:
        if d["status"] != "downloading":
            return
        vcodec = (d.get("info_dict") or {}).get("vcodec") or "none"
        on_progress({
            "stage": "download",
            "stream": "video" if vcodec != "none" else "audio",
            "done": d.get("downloaded_bytes") or 0,
            "total": d.get("total_bytes") or d.get("total_bytes_estimate") or 0,
        })

    def pp_hook(d: dict[str, Any]) -> None:
        if d["status"] == "started" and d.get("postprocessor") == "Merger":
            on_progress({"stage": "merge"})

    h = max_height
    opts = {
        "format": (
            f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/"
            f"bv*[height<={h}]+ba/b[height<={h}]/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(tmp_dir / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "postprocessor_hooks": [pp_hook],
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    downloads = info.get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads else next(tmp_dir.glob("video.*"))
    return path, info


# ---------- the ingest step ----------

def ingest(
    url_or_id: str,
    data_dir: Path,
    max_height: int = 720,
    force: bool = False,
    on_progress: ProgressFn = lambda _: None,
    downloader=download_with_ytdlp,
    prober=probe,
) -> tuple[VideoMeta, bool]:
    """Make sure the video is downloaded and described. Returns (meta, was_cached)."""
    video_id = parse_video_id(url_or_id)
    video_dir = data_dir / video_id

    if not force and (cached := load_meta(video_dir)) is not None:
        return cached, True

    video_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = video_dir / ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)   # leftovers from a crashed run
    tmp_dir.mkdir()

    on_progress({"stage": "resolve"})
    path, info = downloader(canonical_url(video_id), tmp_dir, max_height, on_progress)

    final = video_dir / f"video{path.suffix}"
    os.replace(path, final)                      # atomic: file is complete or absent
    shutil.rmtree(tmp_dir, ignore_errors=True)

    on_progress({"stage": "probe"})
    probed = prober(final)

    meta = VideoMeta(
        id=video_id,
        url=canonical_url(video_id),
        file=final.name,
        title=info.get("title", ""),
        channel=info.get("channel") or info.get("uploader") or "",
        language=info.get("language"),
        # yt-dlp lists a live-chat replay under "subtitles"; it is not a caption track.
        manual_subtitles=sorted(k for k in (info.get("subtitles") or {}) if k != "live_chat"),
        has_auto_captions=bool(info.get("automatic_captions")),
        **probed,
    )
    write_json_atomic(video_dir / "meta.json", asdict(meta))
    return meta, False
