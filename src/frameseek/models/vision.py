"""Vision-language model interface and the Ollama implementation.

Every other part of frameseek depends on the VisionModel interface, never on
Ollama directly. Adding another backend means writing one new class and one
new branch in build_vision_model().

stream() yields three kinds of events so callers can show progress:
  Status -> a step finished or started ("model already in memory")
  Token  -> a piece of the answer as the model generates it
  Done   -> final timing breakdown
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Union

import httpx
from PIL import Image

from frameseek.config import VisionConfig


# ---------- events ----------

@dataclass(frozen=True)
class Status:
    message: str


@dataclass(frozen=True)
class Token:
    text: str


@dataclass(frozen=True)
class Timing:
    load_s: float          # loading model weights into memory (0 if already loaded)
    prefill_s: float       # reading the prompt + image tokens
    prefill_tokens: int
    decode_s: float        # generating the answer, one token at a time
    decode_tokens: int
    total_s: float

    @property
    def decode_tok_per_s(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s else 0.0


@dataclass(frozen=True)
class Done:
    timing: Timing


Event = Union[Status, Token, Done]


# ---------- interface ----------

class VisionModel(Protocol):
    def stream(self, image: Path, prompt: str) -> Iterator[Event]:
        """Yield progress events, answer tokens, then a final Done."""
        ...

    def describe(self, image: Path, prompt: str) -> str:
        """Return the full answer about a single image."""
        ...

    def structured(
        self, image: Path | list[Path], prompt: str, schema: dict,
        max_side: int | None = None, num_predict: int = 400,
    ) -> tuple[dict, Timing]:
        """Return JSON that is guaranteed to match `schema`, plus timing. Accepts one image or several."""
        ...


class OllamaError(RuntimeError):
    pass


# ---------- helpers ----------

def encode_image(path: Path, max_side: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Downscale so the longest side is at most max_side, then base64-encode as JPEG.

    Returns (base64_string, original_size, new_size). Smaller images mean fewer
    image tokens for the model, which is the main lever on speed locally.
    """
    with Image.open(path) as img:
        original = img.size
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        new = img.size
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii"), original, new


def _ns_to_s(ns: int | None) -> float:
    return (ns or 0) / 1e9


def _timing(chunk: dict) -> Timing:
    """Ollama reports durations in nanoseconds on the final response chunk."""
    return Timing(
        load_s=_ns_to_s(chunk.get("load_duration")),
        prefill_s=_ns_to_s(chunk.get("prompt_eval_duration")),
        prefill_tokens=chunk.get("prompt_eval_count", 0),
        decode_s=_ns_to_s(chunk.get("eval_duration")),
        decode_tokens=chunk.get("eval_count", 0),
        total_s=_ns_to_s(chunk.get("total_duration")),
    )


# ---------- Ollama ----------

class OllamaVision:
    """Talks to a local Ollama server over its HTTP API."""

    def __init__(self, cfg: VisionConfig, client: httpx.Client | None = None):
        self.cfg = cfg
        self.client = client or httpx.Client(base_url=cfg.host, timeout=cfg.timeout_s)

    def health(self) -> None:
        """Raise OllamaError with a helpful message if the server or model is missing."""
        try:
            resp = self.client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.cfg.host}. Is the Ollama app running?"
            ) from exc

        names = {m["name"] for m in resp.json().get("models", [])}
        if self.cfg.model not in names:
            raise OllamaError(
                f"Model {self.cfg.model!r} is not pulled. Run: ollama pull {self.cfg.model}"
            )

    def is_loaded(self) -> bool:
        """True if the model is already resident in memory (/api/ps lists running models)."""
        try:
            resp = self.client.get("/api/ps")
            resp.raise_for_status()
        except httpx.HTTPError:
            return False
        running = {m.get("name") or m.get("model") for m in resp.json().get("models", [])}
        return self.cfg.model in running

    def residency(self) -> tuple[int, int] | None:
        """(total_bytes, bytes_in_gpu_memory) for the loaded model, or None if not loaded.

        Ollama picks its own device (Metal on Apple Silicon, CUDA on NVIDIA);
        size_vram tells us how much of the model actually sits on the GPU.
        """
        try:
            resp = self.client.get("/api/ps")
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        for m in resp.json().get("models", []):
            if (m.get("name") or m.get("model")) == self.cfg.model:
                return int(m.get("size", 0)), int(m.get("size_vram", 0))
        return None

    def stream(self, image: Path, prompt: str) -> Iterator[Event]:
        b64, original, new = encode_image(image, self.cfg.max_image_side)
        yield Status(f"Encoded image {original[0]}x{original[1]} -> {new[0]}x{new[1]}")

        if self.is_loaded():
            yield Status("Model already in memory")
        else:
            yield Status("Loading model into memory (slow on first call)")

        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": True,
            "keep_alive": self.cfg.keep_alive,  # keep weights loaded between calls
            "options": {"temperature": 0, "num_ctx": self.cfg.num_ctx},  # deterministic: same frame, same caption
        }
        yield Status("Model is reading the image and prompt")

        with self.client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                resp.read()
                hint = (" Raise vision.num_ctx in config.yaml."
                        if "context size" in resp.text else "")
                raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:300]}{hint}")

            # Ollama streams newline-delimited JSON: one object per chunk.
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise OllamaError(chunk["error"])

                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield Token(text)

                if chunk.get("done"):
                    yield Done(_timing(chunk))

    def structured(
        self, image: Path | list[Path], prompt: str, schema: dict,
        max_side: int | None = None, num_predict: int = 400,
    ) -> tuple[dict, Timing]:
        """Constrained decoding: Ollama only allows tokens that keep the output valid for `schema`.

        Several images can be sent in one message; they reach the model in order,
        so a prompt can refer to them as image 1, image 2, ...
        """
        paths = image if isinstance(image, list) else [image]
        side = max_side or self.cfg.max_image_side
        images = [encode_image(p, side)[0] for p in paths]
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "stream": False,
            "format": schema,                       # the JSON schema the output must follow
            "keep_alive": self.cfg.keep_alive,
            # num_ctx is the same on every call: a different value makes Ollama reload the model
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": self.cfg.num_ctx},
        }
        resp = self.client.post("/api/chat", json=payload)
        if resp.status_code != 200:
            hint = (" Raise vision.num_ctx in config.yaml, or lower answer.top_n."
                    if "context size" in resp.text else "")
            raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:300]}{hint}")
        data = resp.json()
        content = data["message"]["content"]
        try:
            return json.loads(content), _timing(data)
        except json.JSONDecodeError as exc:
            # Only possible if generation hit num_predict before the JSON closed.
            raise OllamaError(f"Output was cut off before the JSON finished: {content[:200]}") from exc

    def describe(self, image: Path, prompt: str) -> str:
        return "".join(e.text for e in self.stream(image, prompt) if isinstance(e, Token)).strip()


def build_vision_model(cfg: VisionConfig) -> VisionModel:
    if cfg.backend == "ollama":
        return OllamaVision(cfg)
    raise ValueError(f"Unknown vision backend: {cfg.backend!r}")
