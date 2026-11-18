"""Quiet Hugging Face downloads so they don't fight frameseek's own progress display.

huggingface_hub and transformers each draw tqdm progress bars. tqdm redraws a
bar by returning to the start of the line, but rich's live display moves the
cursor too, so every redraw lands on a new line: a wall of bars. frameseek turns
those bars off and shows one line of its own instead.
"""

from __future__ import annotations

import logging
import os

# Rough download sizes, only for the message shown on first use.
APPROX_SIZE = {
    "intfloat/multilingual-e5-small": "~470 MB",
    "intfloat/multilingual-e5-base": "~1.1 GB",
    "google/siglip2-base-patch16-224": "~1.5 GB",
}


def quiet_hf() -> None:
    """Silence download bars, weight-loading bars and noisy config warnings."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # PyTorch prints "Redirects are currently not supported in MacOs" when this
    # module is imported (sentence-transformers pulls it in). Harmless noise.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    try:
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import logging as hub_logging
        disable_progress_bars()
        hub_logging.set_verbosity_error()   # e.g. the "unauthenticated requests" notice
    except ImportError:
        pass
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass


def is_downloaded(repo_id: str) -> bool:
    """True if the model is already in the local Hugging Face cache (no network call)."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False
