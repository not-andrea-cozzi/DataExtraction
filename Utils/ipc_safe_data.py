from __future__ import annotations

import io
import logging
import resource
from typing import Any, Optional
import torch

logger = logging.getLogger("ipc_safe_data")

_RECOMMENDED_MIN_NOFILE = 8192


def harden_process_for_ipc(recommended_min_nofile: int = _RECOMMENDED_MIN_NOFILE) -> None:
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Impossibile impostare sharing_strategy='file_system': {e}")

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < recommended_min_nofile:
            new_soft = min(hard, recommended_min_nofile)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info(
                f"[ipc_safe_data] RLIMIT_NOFILE alzato da {soft} a {new_soft} "
                f"(hard limit={hard})."
            )
        else:
            logger.debug(f"[ipc_safe_data] RLIMIT_NOFILE gia' sufficiente: {soft}.")
    except (ValueError, OSError) as e:
        logger.warning(
            f"[ipc_safe_data] Impossibile alzare RLIMIT_NOFILE ({e}); "
            f"affidamento esclusivo sulla serializzazione a bytes per l'IPC."
        )


def encode_for_ipc(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def decode_from_ipc(payload: bytes, *, map_location: Optional[str] = "cpu") -> Any:
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location=map_location, weights_only=False)