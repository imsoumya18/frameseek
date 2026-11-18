"""Chooses where in-process models (SigLIP2, faster-whisper) run.

Ollama manages its own device, so this only matters for models we load
inside our Python process. Libraries differ in what they support:
PyTorch runs on CUDA, MPS or CPU. CTranslate2 (behind faster-whisper) runs on
CUDA or CPU, and from 4.9 has an experimental MPS backend, used only when
asked for explicitly.
"""

from __future__ import annotations

VALID = {"auto", "cuda", "mps", "cpu"}


def pick_device(preferred: str = "auto") -> str:
    """Return 'cuda', 'mps' or 'cpu' for PyTorch models.

    'auto' picks the fastest one available. An explicit choice is honoured
    only if it is actually available; otherwise we fall back to 'cpu'.
    """
    if preferred not in VALID:
        raise ValueError(f"device must be one of {sorted(VALID)}, got {preferred!r}")

    try:
        import torch  # imported lazily: torch is only installed from Section 6
    except ImportError:
        return "cpu"

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }

    if preferred == "auto":
        for name in ("cuda", "mps", "cpu"):
            if available[name]:
                return name
    return preferred if available[preferred] else "cpu"


def ctranslate2_device(preferred: str = "auto") -> str:
    """Return 'cuda', 'mps' or 'cpu' for CTranslate2 models (faster-whisper).

    'auto' never picks MPS: it is experimental in CTranslate2, so it must be
    requested explicitly (transcribe.device: mps).
    """
    if preferred not in VALID:
        raise ValueError(f"device must be one of {sorted(VALID)}, got {preferred!r}")
    if preferred == "mps":
        return "mps"
    if preferred in {"auto", "cuda"}:
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except ImportError:
            pass
    return "cpu"
