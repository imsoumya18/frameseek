"""Embedding models: turn text and images into unit-length vectors.

TextEmbedder   multilingual-e5: question <-> caption text and speech chunks.
               Trained with "query: " / "passage: " prefixes (asymmetric
               retrieval); leaving them out measurably hurts results.
ImageEmbedder  SigLIP2: question text <-> frame pixels, in one shared space
               learned by contrastive training on image-text pairs.

All vectors are L2-normalised, so cosine similarity is a plain dot product.
Models load lazily on first use and run on CUDA, MPS or CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from frameseek.hf import is_downloaded, quiet_hf

ProgressFn = Callable[[dict], None]


class TextEmbedder:
    def __init__(self, name: str, device: str, batch_size: int = 32):
        self.name, self.device, self.batch_size = name, device, batch_size
        self._model = None
        # e5 models expect prefixes; other sentence-transformers models do not.
        self._prefixes = ("query: ", "passage: ") if "e5" in name.lower() else ("", "")

    def is_downloaded(self) -> bool:
        return is_downloaded(self.name)

    def load(self) -> None:
        if self._model is None:
            quiet_hf()
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name, device=self.device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.load()
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype="float32")
        vecs = self._model.encode(texts, batch_size=self.batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype("float32")

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._prefixes[0] + t for t in texts])

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._prefixes[1] + t for t in texts])


def _features(out) -> "torch.Tensor":  # noqa: F821
    """transformers 5 returns an output object here; older versions return a tensor."""
    import torch
    return out if torch.is_tensor(out) else out.pooler_output


class ImageEmbedder:
    def __init__(self, name: str, device: str, batch_size: int = 16):
        self.name, self.device, self.batch_size = name, device, batch_size
        self._model = self._proc = None

    def is_downloaded(self) -> bool:
        return is_downloaded(self.name)

    def load(self) -> None:
        if self._model is None:
            quiet_hf()
            from transformers import AutoModel, AutoProcessor
            self._model = AutoModel.from_pretrained(self.name).to(self.device).eval()
            self._proc = AutoProcessor.from_pretrained(self.name)

    def images(self, paths: list[Path], on_progress: ProgressFn = lambda _: None) -> np.ndarray:
        import torch
        from PIL import Image

        self.load()
        out: list[np.ndarray] = []
        for i in range(0, len(paths), self.batch_size):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + self.batch_size]]
            with torch.no_grad():
                inputs = self._proc(images=batch, return_tensors="pt").to(self.device)
                vecs = _features(self._model.get_image_features(**inputs))
                vecs = torch.nn.functional.normalize(vecs, dim=-1)
            out.append(vecs.float().cpu().numpy())
            on_progress({"stage": "embed_frames", "done": min(i + self.batch_size, len(paths)),
                         "total": len(paths)})
        return np.concatenate(out).astype("float32") if out else np.zeros((0, 0), dtype="float32")

    def texts(self, texts: list[str]) -> np.ndarray:
        import torch

        self.load()
        with torch.no_grad():
            # SigLIP was trained on text padded to exactly 64 tokens; other padding hurts matching.
            inputs = self._proc(text=texts, padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
            vecs = _features(self._model.get_text_features(**inputs))
            vecs = torch.nn.functional.normalize(vecs, dim=-1)
        return vecs.float().cpu().numpy().astype("float32")
