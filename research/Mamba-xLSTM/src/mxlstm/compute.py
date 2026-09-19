"""Device selection and Mamba backend detection.

Single source of truth for "what hardware are we on" and "which Mamba
implementation can we import" so the rest of the codebase doesn't sprinkle
``torch.cuda.is_available()`` checks everywhere.

Mamba-3 backends supported (the project uses Mamba-3 throughout):

  * ``mamba_ssm``  - NVIDIA's official CUDA + Triton kernels routed to
                    ``mamba_ssm.Mamba3``. Requires building ``mamba-ssm``
                    from source per the project README
                    (``MAMBA_FORCE_BUILD=TRUE pip install ...``).
  * ``mambapy``    - alxndrTL's wrapper. As of this writing it does not
                    implement Mamba-3, so the Mamba block factory
                    transparently downgrades it to the ``vanilla`` path.
  * ``none``       - no external backend; Mamba block construction falls
                    back to the pure-PyTorch Mamba-3 SISO reference in
                    :mod:`mxlstm.models.mamba_blocks`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

import torch

MambaBackendName = Literal["mamba_ssm", "mambapy", "none"]


@dataclass(frozen=True)
class ComputeProfile:
    device: torch.device
    accelerator: str  # "cuda" | "mps" | "cpu"
    mamba_backend: MambaBackendName
    precision: str  # "32-true" | "bf16-mixed" | "16-mixed"


def _detect_mamba_backend() -> MambaBackendName:
    if torch.cuda.is_available():
        try:
            importlib.import_module("mamba_ssm")
            return "mamba_ssm"
        except ImportError:
            pass
    try:
        importlib.import_module("mambapy")
        return "mambapy"
    except ImportError:
        return "none"


def get_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_precision() -> str:
    """Pick a precision that actually works on the detected accelerator."""
    if torch.cuda.is_available():
        # bf16 is safer than fp16 on most modern NVIDIA hardware.
        major, _ = torch.cuda.get_device_capability()
        return "bf16-mixed" if major >= 8 else "16-mixed"
    # MPS fp16/bf16 mixed precision is unstable for several ops in 2024-2025.
    return "32-true"


def get_compute_profile() -> ComputeProfile:
    return ComputeProfile(
        device=get_device(),
        accelerator=get_accelerator(),
        mamba_backend=_detect_mamba_backend(),
        precision=get_default_precision(),
    )


def assert_mamba_available() -> MambaBackendName:
    backend = _detect_mamba_backend()
    if backend == "none":
        # The project always ships the pure-PyTorch Mamba-3 fallback, so this
        # only guards code paths that explicitly require the optional libs.
        raise ImportError(
            "No external Mamba backend found. The pure-PyTorch Mamba-3 "
            "reference in mxlstm.models.mamba_blocks._VanillaMamba3 is still "
            "usable. For CUDA acceleration install 'mamba-ssm' built from "
            "source (Mamba-3 ships in state-spaces/mamba; the official "
            "instructions are `MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir "
            "--force-reinstall git+https://github.com/state-spaces/mamba.git "
            "--no-build-isolation`). 'mambapy' does not yet implement Mamba-3 "
            "and is auto-downgraded to the vanilla backend by the block factory."
        )
    return backend
