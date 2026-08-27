"""Lazy NumPy/CuPy backend for Diversity custom-optimizer math."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np


SUPPORTED_COMPUTE_DEVICES = frozenset({"cpu", "gpu", "hybrid"})


class GPUBackendError(RuntimeError):
    """Raised when explicit GPU execution cannot initialize CuPy/CUDA."""


@dataclass(frozen=True)
class GPUInfo:
    device_id: int
    device_count: int
    name: str
    cupy_version: str
    total_memory_bytes: int
    free_memory_bytes: int
    memory_fraction: float
    memory_limit_bytes: int


def normalize_compute_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized not in SUPPORTED_COMPUTE_DEVICES:
        choices = ", ".join(sorted(SUPPORTED_COMPUTE_DEVICES))
        raise ValueError(f"Unsupported compute device '{device}'. Choose one of: {choices}.")
    return normalized


def validate_memory_fraction(memory_fraction: float) -> float:
    fraction = float(memory_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("GPU memory fraction must satisfy 0 < fraction <= 1.")
    return fraction


def cupy_installed() -> bool:
    """Check package presence without importing CuPy or initializing CUDA."""
    return find_spec("cupy") is not None


def _load_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        raise GPUBackendError(
            "CuPy/CUDA is unavailable. Use --compute-device cpu, use hybrid "
            "for automatic CPU fallback, or install requirements-linux-gpu.txt. "
            f"Details: {exc}"
        ) from exc
    return cp


def _activate_gpu(cp, device_id: int, memory_fraction: float) -> GPUInfo:
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("no CUDA devices were reported")
        if not 0 <= device_id < device_count:
            raise RuntimeError(
                f"CUDA device {device_id} is invalid; {device_count} device(s) available"
            )
        cp.cuda.Device(device_id).use()
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        memory_pool = cp.get_default_memory_pool()
        memory_limit = max(1, int(total_memory * memory_fraction))
        memory_pool.set_limit(size=memory_limit)
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = properties.get("name", "Unknown NVIDIA GPU")
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        cp.asarray([0.0], dtype=cp.float64)
        cp.cuda.get_current_stream().synchronize()
    except Exception as exc:
        raise GPUBackendError(f"CUDA initialization failed: {exc}") from exc
    return GPUInfo(
        device_id=int(device_id),
        device_count=device_count,
        name=name,
        cupy_version=str(cp.__version__),
        total_memory_bytes=int(total_memory),
        free_memory_bytes=int(free_memory),
        memory_fraction=memory_fraction,
        memory_limit_bytes=int(memory_pool.get_limit()),
    )


def initialize_gpu(device_id: int = 0, memory_fraction: float = 0.85) -> GPUInfo:
    """Explicitly initialize CUDA for validation or strict GPU mode."""
    fraction = validate_memory_fraction(memory_fraction)
    return _activate_gpu(_load_cupy(), int(device_id), fraction)


class ComputeBackend:
    """Array boundary that leaves MAFESE, MEALPY agents, and sklearn on NumPy."""

    def __init__(
        self,
        device: str = "cpu",
        device_id: int = 0,
        memory_fraction: float = 0.85,
    ):
        self.requested_device = normalize_compute_device(device)
        self.device_id = int(device_id)
        self.memory_fraction = validate_memory_fraction(memory_fraction)
        self.device = "cpu"
        self.xp = np
        self.gpu_info: GPUInfo | None = None
        self.fallback_reason: str | None = None

        if self.requested_device == "cpu":
            return
        try:
            cp = _load_cupy()
            self.gpu_info = _activate_gpu(cp, self.device_id, self.memory_fraction)
        except GPUBackendError as exc:
            if self.requested_device == "gpu":
                raise
            self.fallback_reason = str(exc)
            return
        self.device = "gpu"
        self.xp = cp

    @property
    def uses_gpu(self) -> bool:
        return self.device == "gpu"

    def asarray(self, value: Any, dtype=None):
        return self.xp.asarray(value, dtype=dtype)

    def to_cpu(self, value: Any) -> np.ndarray:
        if self.uses_gpu:
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def scalar(self, value: Any) -> float:
        if self.uses_gpu:
            return float(value.item())
        return float(value)

    def synchronize(self) -> None:
        if self.uses_gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def free_cached_blocks(self) -> None:
        if self.uses_gpu:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()

