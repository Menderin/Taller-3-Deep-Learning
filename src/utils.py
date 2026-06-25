"""Small reproducibility and environment helpers."""

from __future__ import annotations

import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix common random generators for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    """Select an available PyTorch device."""

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda, pero CUDA no esta disponible.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("DEVICE=mps, pero MPS no esta disponible.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_environment_info(device: torch.device) -> dict[str, Any]:
    """Return versions useful for the reproducibility section of the report."""

    import torchvision

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "N/A",
    }
