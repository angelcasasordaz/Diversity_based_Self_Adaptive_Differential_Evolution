"""Tiny validation for the lazy Diversity NumPy/CuPy backend."""
from __future__ import annotations

import argparse

import numpy as np

from diversity_gpu_batching import DiversityMathBatcher


def validate_backend(requested_device: str = "hybrid") -> str:
    rng = np.random.default_rng(20260827)
    population = rng.normal(size=(10, 8))
    x1, x2, x3 = population[1], population[3], population[7]
    factor = rng.uniform(0.2, 0.8, size=8)
    mask = rng.random(8) <= 0.4

    cpu = DiversityMathBatcher("cpu")
    candidate = DiversityMathBatcher(requested_device)
    np.testing.assert_allclose(
        cpu.awad(population), candidate.awad(population), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        cpu.covariance_inverse(population, 8),
        candidate.covariance_inverse(population, 8),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        cpu.mahalanobis_distances(population, 8),
        candidate.mahalanobis_distances(population, 8),
        rtol=1e-9,
        atol=1e-10,
    )
    cpu_mutant = cpu.mutate(x1, x2, x3, factor)
    candidate_mutant = candidate.mutate(x1, x2, x3, factor)
    np.testing.assert_allclose(cpu_mutant, candidate_mutant, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        cpu.crossover(population[0], cpu_mutant, mask),
        candidate.crossover(population[0], candidate_mutant, mask),
        rtol=1e-12,
        atol=1e-12,
    )
    return candidate.effective_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-device", choices=["cpu", "gpu", "hybrid"], default="hybrid")
    args = parser.parse_args()
    effective = validate_backend(args.compute_device)
    print(f"requested_device={args.compute_device}")
    print(f"effective_math_device={effective}")
    print("compute_backend_equivalent=True")


if __name__ == "__main__":
    main()

