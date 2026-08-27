"""Batched numerical kernels for Diversity custom optimizers.

All estimator fitting and fitness evaluation remains on CPU. Only dense
population math crosses the optional CuPy boundary.
"""
from __future__ import annotations

import numpy as np

from compute_backend import ComputeBackend


class DiversityMathBatcher:
    """Execute useful custom-optimizer math on NumPy or optional CuPy."""

    def __init__(
        self,
        compute_device: str = "cpu",
        gpu_device_id: int = 0,
        gpu_memory_fraction: float = 0.85,
    ):
        self.backend = ComputeBackend(
            compute_device,
            device_id=gpu_device_id,
            memory_fraction=gpu_memory_fraction,
        )

    @property
    def uses_gpu(self) -> bool:
        return self.backend.uses_gpu

    @property
    def effective_device(self) -> str:
        return self.backend.device

    def awad(self, population, lb=None, ub=None) -> float:
        """Match the existing AWAD definition; bounds are intentionally unused."""
        _ = lb, ub
        if not self.uses_gpu:
            return self._awad_cpu(population)

        xp = self.backend.xp
        pop = self.backend.asarray(population, dtype=xp.float64)
        npop, n_dims = pop.shape
        median = xp.median(pop, axis=0)
        div = xp.sum(xp.mean(xp.abs(pop - median), axis=0)) / max(n_dims, 1)
        unique_count = xp.unique(pop, axis=0).shape[0]
        non_repeat_percent = unique_count * 100.0 / max(npop, 1)
        std = xp.std(pop, axis=0)
        std = xp.where(std == 0, 1.0e-5, std)
        if npop <= 1:
            min_distance = xp.asarray(0.0)
        else:
            scaled = (pop[:, None, :] - pop[None, :, :]) / std
            distances = xp.sqrt(xp.sum(scaled * scaled, axis=-1))
            diagonal = xp.eye(npop, dtype=bool)
            min_distance = xp.min(xp.where(diagonal, xp.inf, distances))
            min_distance = xp.where(xp.isfinite(min_distance), min_distance, 0.0)
        penalty = ((min_distance + 0.1) ** 2) / (1.0 + min_distance**2)
        return self.backend.scalar(div * 0.1 * non_repeat_percent * penalty)

    @staticmethod
    def _awad_cpu(population) -> float:
        pop = np.asarray(population, dtype=float)
        npop, n_dims = pop.shape
        med_dim = np.median(pop, axis=0)
        div_dim = np.mean(np.abs(pop - med_dim), axis=0)
        div = float(np.sum(div_dim) / max(n_dims, 1))
        unique_count = np.unique(pop, axis=0).shape[0]
        non_repeat_percent = unique_count * 100.0 / max(npop, 1)
        std_devs = np.std(pop, axis=0)
        std_devs[std_devs == 0] = 1.0e-5
        if npop <= 1:
            min_distance = 0.0
        else:
            min_distance = np.inf
            for idx in range(npop - 1):
                diff = (pop[idx + 1:] - pop[idx]) / std_devs
                distances = np.sqrt(np.sum(diff * diff, axis=1))
                if distances.size:
                    min_distance = min(min_distance, float(np.min(distances)))
            if not np.isfinite(min_distance):
                min_distance = 0.0
        penalty = ((min_distance + 0.1) ** 2) / (1.0 + min_distance**2)
        return float(div * 0.1 * non_repeat_percent * penalty)

    def covariance_inverse(self, population, n_dims: int) -> np.ndarray:
        xp = self.backend.xp
        pop = self.backend.asarray(population, dtype=xp.float64)
        sigma = xp.cov(pop, rowvar=False)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1, 1)
        if sigma.shape != (n_dims, n_dims):
            sigma = xp.eye(n_dims, dtype=xp.float64) * 1.0e-6
        sigma = (sigma + sigma.T) / 2.0 + 1.0e-6 * xp.eye(n_dims, dtype=xp.float64)
        try:
            chol = xp.linalg.cholesky(sigma)
            identity = xp.eye(n_dims, dtype=xp.float64)
            inverse = xp.linalg.solve(chol.T, xp.linalg.solve(chol, identity))
        except xp.linalg.LinAlgError:
            inverse = xp.linalg.pinv(sigma)
        return self.backend.to_cpu(inverse)

    def mahalanobis_distances(self, population, n_dims: int) -> np.ndarray:
        xp = self.backend.xp
        pop = self.backend.asarray(population, dtype=xp.float64)
        sigma = xp.cov(pop, rowvar=False)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1, 1)
        if sigma.shape != (n_dims, n_dims):
            sigma = xp.eye(n_dims, dtype=xp.float64) * 1.0e-6
        sigma = (sigma + sigma.T) / 2.0 + 1.0e-6 * xp.eye(n_dims, dtype=xp.float64)
        try:
            chol = xp.linalg.cholesky(sigma)
            identity = xp.eye(n_dims, dtype=xp.float64)
            inverse = xp.linalg.solve(chol.T, xp.linalg.solve(chol, identity))
        except xp.linalg.LinAlgError:
            inverse = xp.linalg.pinv(sigma)
        centered = pop - xp.mean(pop, axis=0)
        distances = xp.sum((centered @ inverse) * centered, axis=1)
        return self.backend.to_cpu(distances)

    def mutate(self, x1, x2, x3, factor) -> np.ndarray:
        if not self.uses_gpu:
            return np.asarray(x1) + np.asarray(factor) * (np.asarray(x2) - np.asarray(x3))
        xp = self.backend.xp
        result = (
            self.backend.asarray(x1, dtype=xp.float64)
            + self.backend.asarray(factor, dtype=xp.float64)
            * (
                self.backend.asarray(x2, dtype=xp.float64)
                - self.backend.asarray(x3, dtype=xp.float64)
            )
        )
        return self.backend.to_cpu(result)

    def crossover(self, parent, mutant, mask) -> np.ndarray:
        if not self.uses_gpu:
            trial = np.asarray(parent).copy()
            trial[np.asarray(mask, dtype=bool)] = np.asarray(mutant)[np.asarray(mask, dtype=bool)]
            return trial
        xp = self.backend.xp
        trial = xp.where(
            self.backend.asarray(mask, dtype=bool),
            self.backend.asarray(mutant, dtype=xp.float64),
            self.backend.asarray(parent, dtype=xp.float64),
        )
        return self.backend.to_cpu(trial)

