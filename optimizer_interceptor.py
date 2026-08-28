"""Safe numerical boundaries shared by generic and specialized optimizers.

This module intentionally does not monkey-patch NumPy or rewrite ``evolve``.
Only explicitly validated dense regions may call through this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from numerical_backend import NumericalBackend


@dataclass(frozen=True)
class Workload:
    population_size: int
    dimensions: int
    epochs: int = 1

    @property
    def element_work(self) -> int:
        return max(1, self.population_size) * max(1, self.dimensions) * max(1, self.epochs)

    @property
    def population_bytes(self) -> int:
        return max(1, self.population_size) * max(1, self.dimensions) * 8


class PopulationInterceptor:
    """Pack Agent solutions for a kernel and always return NumPy at its edge."""

    def __init__(self, backend: NumericalBackend):
        self.backend = backend

    def pack(self, population: Iterable, dtype=np.float64):
        host = np.asarray([agent.solution for agent in population], dtype=dtype)
        return self.backend.asarray(host, dtype=dtype)

    def unpack(self, matrix) -> np.ndarray:
        return self.backend.to_cpu(matrix)

    def bounds(self, lower, upper, dtype=np.float64):
        return self.backend.asarray(lower, dtype=dtype), self.backend.asarray(upper, dtype=dtype)


def estimate_dense_kernel_bytes(workload: Workload, temporary_matrices: int = 4) -> int:
    return max(64 * 1024**2, workload.population_bytes * max(2, temporary_matrices))


def hybrid_gpu_is_worthwhile(
    workload: Workload,
    kernel_work: int,
    available_gpu_bytes: int,
    estimated_gpu_bytes: int,
    minimum_kernel_work: int,
    minimum_epochs: int,
) -> tuple[bool, str]:
    if available_gpu_bytes < 2 * estimated_gpu_bytes:
        return False, "available VRAM lacks 2x headroom for the estimated working set"
    if workload.epochs < minimum_epochs or kernel_work < minimum_kernel_work:
        return False, "GPU work is too small to amortize CPU/GPU transfers"
    return True, "validated kernels and workload justify GPU execution"
