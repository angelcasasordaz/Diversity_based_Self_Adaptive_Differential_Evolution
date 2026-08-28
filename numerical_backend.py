"""Common NumPy/CuPy numerical backend used by all optimizer integrations.

The backend is deliberately an array boundary, not a replacement for MEALPY's
Agent, RNG, Problem, or fitness machinery. GPU requests fall back to NumPy when
CUDA is unavailable so the complete experiment can continue.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Protocol, runtime_checkable

import numpy as np


SUPPORTED_COMPUTE_DEVICES = frozenset({"cpu", "gpu", "hybrid"})


class GPUBackendError(RuntimeError):
    """Raised when an explicit CuPy/CUDA backend cannot be initialized."""


class UnsupportedOptimizerDeviceError(RuntimeError):
    """Raised when strict GPU mode is requested for an unsupported optimizer."""


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


@runtime_checkable
class NumericalBackend(Protocol):
    requested_device: str
    device: str
    xp: Any
    gpu_info: GPUInfo | None
    fallback_reason: str | None

    @property
    def uses_gpu(self) -> bool: ...
    def asarray(self, value: Any, dtype=None): ...
    def to_cpu(self, value: Any) -> np.ndarray: ...
    def scalar(self, value: Any) -> float: ...
    def synchronize(self) -> None: ...
    def free_cached_blocks(self) -> None: ...


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
    return find_spec("cupy") is not None


def _load_cupy():
    try:
        import cupy as cp
    except Exception as exc:
        raise GPUBackendError(
            "CuPy/CUDA is unavailable. Use CPU or hybrid mode, or install the "
            f"project GPU requirements. Details: {exc}"
        ) from exc
    return cp


def _activate_gpu(cp, device_id: int, memory_fraction: float) -> GPUInfo:
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("no CUDA devices were reported")
        if not 0 <= device_id < device_count:
            raise RuntimeError(f"CUDA device {device_id} is invalid; {device_count} device(s) available")
        cp.cuda.Device(device_id).use()
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        memory_pool = cp.get_default_memory_pool()
        memory_pool.set_limit(size=max(1, int(total_memory * memory_fraction)))
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = properties.get("name", "Unknown NVIDIA GPU")
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        cp.asarray([0.0], dtype=cp.float64)
        cp.cuda.get_current_stream().synchronize()
    except Exception as exc:
        raise GPUBackendError(f"CUDA initialization failed: {exc}") from exc
    return GPUInfo(
        device_id=device_id, device_count=device_count, name=name,
        cupy_version=str(cp.__version__), total_memory_bytes=int(total_memory),
        free_memory_bytes=int(free_memory), memory_fraction=memory_fraction,
        memory_limit_bytes=int(memory_pool.get_limit()),
    )


def initialize_gpu(device_id: int = 0, memory_fraction: float = 0.85) -> GPUInfo:
    return _activate_gpu(_load_cupy(), int(device_id), validate_memory_fraction(memory_fraction))


class ArrayBackend:
    """Array namespace with explicit and observable host/device transfers."""

    def __init__(self, device: str = "cpu", device_id: int = 0, memory_fraction: float = 0.85):
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
        return self.xp.asnumpy(value) if self.uses_gpu else np.asarray(value)

    def scalar(self, value: Any) -> float:
        return float(value.item()) if self.uses_gpu else float(value)

    def synchronize(self) -> None:
        if self.uses_gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def free_cached_blocks(self) -> None:
        if self.uses_gpu:
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()


class NumPyBackend(ArrayBackend):
    def __init__(self):
        super().__init__("cpu")


class CuPyBackend(ArrayBackend):
    def __init__(self, device_id: int = 0, memory_fraction: float = 0.85):
        super().__init__("gpu", device_id, memory_fraction)


class HybridBackend(ArrayBackend):
    def __init__(self, device_id: int = 0, memory_fraction: float = 0.85):
        super().__init__("hybrid", device_id, memory_fraction)


def create_numerical_backend(device: str, device_id: int = 0, memory_fraction: float = 0.85) -> ArrayBackend:
    mode = normalize_compute_device(device)
    if mode == "cpu":
        return NumPyBackend()
    if mode == "gpu":
        return CuPyBackend(device_id, memory_fraction)
    return HybridBackend(device_id, memory_fraction)


# Compatibility name retained for existing callers while infrastructure moves.
ComputeBackend = ArrayBackend
