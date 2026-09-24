from __future__ import annotations

import io
import logging
import resource
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
_MIN_NOFILE = 8192


def harden_process_for_ipc(min_nofile: int = _MIN_NOFILE) -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError) as e:
        logger.warning("sharing_strategy='file_system' non impostabile: %s", e)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < min_nofile:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, min_nofile), hard))
    except (ValueError, OSError) as e:
        logger.warning("RLIMIT_NOFILE non alzabile: %s", e)


def encode_for_ipc(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def decode_from_ipc(payload: bytes, *, map_location: Optional[str] = "cpu") -> Any:
    return torch.load(io.BytesIO(payload), map_location=map_location, weights_only=False)
