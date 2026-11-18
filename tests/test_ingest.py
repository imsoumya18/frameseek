"""Tests for ingest: URL parsing, caching and atomic writes. No network needed."""

import json
from pathlib import Path

import pytest

from frameseek.ingest import ingest, parse_fps, parse_video_id

VID = "nRWZBq2hcls"


@pytest.mark.parametrize(
    "url",
    [
        VID,
        f"https://www.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/watch?v={VID}&t=45s&list=PL123",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?t=10",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://m.youtube.com/watch?v={VID}",
        f"youtube.com/embed/{VID}",
        f"https://www.youtube.com/live/{VID}?feature=share",
    ],
)
def test_every_url_form_maps_to_same_id(url: str) -> None:
    assert parse_video_id(url) == VID


@pytest.mark.parametrize("bad", ["", "hello", "https://vimeo.com/123", "https://youtube.com/watch?v=short"])
def test_rejects_non_youtube(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_video_id(bad)


def test_parse_fps() -> None:
    assert round(parse_fps("30000/1001"), 2) == 29.97
    assert parse_fps("25/1") == 25.0
    assert parse_fps("0/0") == 0.0


def _fake_downloader(url, tmp_dir: Path, max_height, on_progress):
    on_progress({"stage": "download", "stream": "video", "done": 50, "total": 100})
    path = tmp_dir / "video.mp4"
    path.write_bytes(b"fake video")
    return path, {
        "title": "Test",
        "channel": "Me",
        "subtitles": {"en": [], "live_chat": []},   # live_chat is not a caption track
        "automatic_captions": {"hi": []},
    }


def _fake_prober(path: Path) -> dict:
    return {"duration_s": 60.0, "width": 1280, "height": 720, "fps": 30.0,
            "vcodec": "h264", "acodec": "aac", "keyframe_interval_s": 2.0}


def test_first_ingest_downloads_and_writes_meta(tmp_path: Path) -> None:
    events = []
    meta, cached = ingest(f"https://youtu.be/{VID}", tmp_path,
                          on_progress=events.append,
                          downloader=_fake_downloader, prober=_fake_prober)

    assert cached is False
    vdir = tmp_path / VID
    assert (vdir / "video.mp4").read_bytes() == b"fake video"
    assert not (vdir / ".tmp").exists()

    saved = json.loads((vdir / "meta.json").read_text())
    assert saved["title"] == "Test"
    assert saved["manual_subtitles"] == ["en"]
    assert saved["has_auto_captions"] is True
    assert saved["keyframe_interval_s"] == 2.0
    assert {e["stage"] for e in events} >= {"resolve", "download", "probe"}


def test_second_ingest_with_different_url_is_cached(tmp_path: Path) -> None:
    ingest(VID, tmp_path, downloader=_fake_downloader, prober=_fake_prober)

    def must_not_download(*args, **kwargs):
        raise AssertionError("downloader should not run for a cached video")

    meta, cached = ingest(f"https://www.youtube.com/watch?v={VID}&t=45",
                          tmp_path, downloader=must_not_download, prober=_fake_prober)
    assert cached is True
    assert meta.id == VID


def test_force_redownloads(tmp_path: Path) -> None:
    ingest(VID, tmp_path, downloader=_fake_downloader, prober=_fake_prober)
    _, cached = ingest(VID, tmp_path, force=True, downloader=_fake_downloader, prober=_fake_prober)
    assert cached is False


def test_missing_video_file_invalidates_cache(tmp_path: Path) -> None:
    ingest(VID, tmp_path, downloader=_fake_downloader, prober=_fake_prober)
    (tmp_path / VID / "video.mp4").unlink()
    _, cached = ingest(VID, tmp_path, downloader=_fake_downloader, prober=_fake_prober)
    assert cached is False
