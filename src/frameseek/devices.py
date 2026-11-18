"""Chooses where in-process models (SigLIP2, faster-whisper) run.

Ollama manages its own device, so this only matters for models we load
inside our Python process.
"""

from __future__ import annotations

VALID = {"auto", "cuda", "mps", "cpu"}


def pick_device(preferred: str = "auto") -> str:
    """Return 'cuda', 'mps' or 'cpu'.

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
