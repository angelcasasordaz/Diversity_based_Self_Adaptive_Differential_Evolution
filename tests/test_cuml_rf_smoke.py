"""Optional native GPU smoke check; no optimizers, datasets, or output files.

RUN_CUML_RF_SMOKE=1 .venv/bin/python -B -m unittest tests.test_cuml_rf_smoke -v
"""
import argparse
import os
import unittest

import numpy as np

import main_best as study


@unittest.skipUnless(os.environ.get("RUN_CUML_RF_SMOKE") == "1", "Explicit GPU smoke opt-in required")
class CuMLRFSmokeTests(unittest.TestCase):
    def test_native_project_gpu_rf_fit_predict(self):
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(size=(128, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                # This opt-in native check is independent of the historical
                # sklearn policy selected for normal EXP629 continuation.
                args = argparse.Namespace(compute_device="gpu", rf_cpu_fallback=fallback,
                                          rf_backend_policy="cuml")
                classifier = study.build_run_estimator("rf", args, 42)
                self.assertIsInstance(classifier, RandomForestClassifier)
                self.assertEqual(classifier.get_params()["n_streams"], 1)
                self.assertEqual(classifier.get_params()["n_estimators"], 100)
                classifier.set_params(n_estimators=8, max_depth=4, n_bins=16)
                classifier.fit(cp.asarray(X), cp.asarray(y))
                predicted = classifier.predict(cp.asarray(X))
                cp.cuda.runtime.deviceSynchronize()
                self.assertIsInstance(predicted, np.ndarray)
                self.assertEqual(predicted.shape, y.shape)
                self.assertGreaterEqual(float((predicted == y).mean()), .8)
                self.assertEqual(args.compute_device, "gpu")


if __name__ == "__main__":
    unittest.main()
