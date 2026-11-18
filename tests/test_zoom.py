"""Tests for turning model boxes into crops."""

from pathlib import Path

from PIL import Image

from frameseek.zoom import Box, crop_pixels, parse_box, save_crop


def test_parse_box_scales_orders_and_clamps() -> None:
    assert parse_box([808, 904, 898, 948]) == Box(0.808, 0.904, 0.898, 0.948)   # the real T-shirt text
    assert parse_box([898, 948, 808, 904]) == Box(0.808, 0.904, 0.898, 0.948)   # swapped corners
    assert parse_box([-5, 0, 1200, 1000]) == Box(0.0, 0.0, 1.0, 1.0)            # clamped to the image


def test_parse_box_rejects_non_boxes() -> None:
    for raw in ([0, 0, 0, 0], [1, 2, 3], "box", None, [10, 10, 10, 500], ["a", 1, 2, 3]):
        assert parse_box(raw) is None


def test_crop_adds_margin_and_keeps_inside_the_frame() -> None:
    text = Box(0.808, 0.904, 0.898, 0.948)                 # ~115 x 24 px in 1280 x 540
    l, t, r, b = crop_pixels(text, (1280, 540), margin=0.3, min_px=160)
    assert r - l == 184 and b - t == 160                    # 115 px + 30% each side; height raised to the minimum
    assert 0 <= l and r <= 1280 and 0 <= t and b <= 540     # slid up from the bottom edge, not cut
    assert l <= 0.808 * 1280 and r >= 0.898 * 1280          # still contains the whole box


def test_large_box_gets_margin_capped_at_frame() -> None:
    l, t, r, b = crop_pixels(Box(0.1, 0.1, 0.9, 0.9), (1280, 540), margin=0.3, min_px=160)
    assert (l, t, r, b) == (0, 0, 1280, 540)


def test_save_crop_enlarges_to_out_side(tmp_path: Path) -> None:
    src = tmp_path / "kf.jpg"
    Image.new("RGB", (1280, 540), "white").save(src)
    rect = save_crop(src, Box(0.808, 0.904, 0.898, 0.948), tmp_path / "crops" / "c.jpg",
                     margin=0.3, min_px=160, out_side=1024)
    with Image.open(tmp_path / "crops" / "c.jpg") as crop:
        assert max(crop.size) == 1024 and (rect[2] - rect[0], rect[3] - rect[1]) == (184, 160)
