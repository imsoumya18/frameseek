"""Turn a box from the model into a crop of the original keyframe.

Qwen3-VL gives boxes as [x1, y1, x2, y2] in relative coordinates from 0 to
1000, whatever the image size (Qwen2.5-VL used absolute pixels). So a box
maps onto the original 1280-pixel keyframe directly, even though the model
was shown a smaller copy.

Why crop at all: Ollama spends about the same number of tokens on any image
(~1,240), so a crop spends the whole budget on the region that matters. Text
that got ~5 tokens in the full frame gets hundreds in the crop. The crop is
taken from the original keyframe, so no pixel the video has is thrown away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Box:
    """Fractions of the image width and height, 0 to 1, top-left to bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def parse_box(raw: object, scale: float = 1000.0) -> Box | None:
    """[x1, y1, x2, y2] on a 0-1000 scale -> Box, or None if it isn't a usable box.

    Swapped corners are put in order and values are clamped to the image; a box
    with no area (the model's "not in the image" [0, 0, 0, 0]) is None.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (min(max(float(v) / scale, 0.0), 1.0) for v in raw)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 <= 0.002 or y2 - y1 <= 0.002:
        return None
    return Box(x1, y1, x2, y2)


def crop_pixels(box: Box, size: tuple[int, int], margin: float, min_px: int) -> tuple[int, int, int, int]:
    """Pixel rectangle (left, top, right, bottom) for a box, with a margin and a minimum size.

    margin: fraction of the box's own width/height added on each side, because
            model boxes are often a little tight or a little off.
    min_px: smallest crop side in original pixels; below that, a crop is mostly
            blur once enlarged, and the model loses the context around the thing.
    """
    w, h = size
    bw, bh = (box.x2 - box.x1) * w, (box.y2 - box.y1) * h
    cx, cy = (box.x1 + box.x2) / 2 * w, (box.y1 + box.y2) / 2 * h
    cw = min(max(bw * (1 + 2 * margin), min_px), w)
    ch = min(max(bh * (1 + 2 * margin), min_px), h)
    left = min(max(cx - cw / 2, 0), w - cw)    # slide inside the image rather than shrink
    top = min(max(cy - ch / 2, 0), h - ch)
    return round(left), round(top), round(left + cw), round(top + ch)


def save_crop(src: Path, box: Box, dest: Path, margin: float, min_px: int, out_side: int) -> tuple[int, int, int, int]:
    """Crop the original keyframe and enlarge it so its longest side is out_side.

    Enlarging adds no detail; it means the model sees every original pixel at
    a useful size whatever the backend's own resizing does.
    """
    with Image.open(src) as img:
        img = img.convert("RGB")
        rect = crop_pixels(box, img.size, margin, min_px)
        crop = img.crop(rect)
    scale = out_side / max(crop.size)
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest, format="JPEG", quality=92)
    return rect
