"""One tiny seeded CPU-versus-hybrid Diversity validation."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from mafese import MhaSelector

from dsade_optimizer import DSADE
from main_best import RobustClassificationFeatureSelectionProblem, load_codesmell_dataset
from validate_compute_backend import validate_backend


SEED = 1234
EPOCHS = 1
POP_SIZE = 10


def run_selector(device: str):
    _, X, y = load_codesmell_dataset(Path("Original/LongMethod.csv"))
    optimizer = DSADE(
        epoch=EPOCHS,
        pop_size=POP_SIZE,
        beta_min=0.2,
        beta_max=0.8,
        pcr=0.2,
        mahalanobis_q=0.68,
        compute_device=device,
    )
    selector = MhaSelector(
        problem="classification",
        estimator="knn",
        optimizer=optimizer,
        obj_name="AS",
        seed=SEED,
        verbose=False,
    )
    started = perf_counter()
    selector.fit(
        X,
        y,
        test_size=0.2,
        transfer_func="vstf_01",
        fs_problem=RobustClassificationFeatureSelectionProblem,
    )
    runtime = perf_counter() - started
    return {
        "runtime": runtime,
        "fitness": float(selector.optimizer.g_best.target.fitness),
        "objectives": np.asarray(selector.optimizer.g_best.target.objectives, dtype=float),
        "mask": np.asarray(selector.selected_feature_masks, dtype=bool),
        "curve": np.asarray(selector.optimizer.history.list_global_best_fit, dtype=float),
        "effective_device": optimizer.math_batcher.effective_device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-device", choices=["gpu", "hybrid"], default="hybrid")
    args = parser.parse_args()

    kernel_device = validate_backend(args.hybrid_device)
    cpu = run_selector("cpu")
    hybrid = run_selector(args.hybrid_device)
    np.testing.assert_allclose(cpu["fitness"], hybrid["fitness"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cpu["objectives"], hybrid["objectives"], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(cpu["mask"], hybrid["mask"])
    np.testing.assert_allclose(cpu["curve"], hybrid["curve"], rtol=1e-10, atol=1e-12)

    print(f"seed={SEED} epochs={EPOCHS} pop_size={POP_SIZE}")
    print(f"requested_hybrid={args.hybrid_device} effective_math_device={kernel_device}")
    print(f"cpu_seconds={cpu['runtime']:.6f}")
    print(f"hybrid_seconds={hybrid['runtime']:.6f}")
    print(f"runtime_ratio_cpu_over_hybrid={cpu['runtime'] / hybrid['runtime']:.6f}")
    print("numerically_equivalent=True")


if __name__ == "__main__":
    main()
