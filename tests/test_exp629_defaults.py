"""EXP629 continuation defaults and read-only source reuse; no model fitting."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import main_best as study
from tests import test_full_cache_reuse as cache_fixtures


class EXP629DefaultsTests(unittest.TestCase):
    setUp = cache_fixtures.FullCacheReuseTests.setUp
    run_mocked = cache_fixtures.FullCacheReuseTests.run_mocked

    def defaults(self):
        with patch.object(sys, "argv", ["main_best.py"]):
            return study.parse_args()

    def test_ide_defaults_continue_exp629_with_historical_rf(self):
        args = self.defaults()
        self.assertEqual((args.exp_id, args.reuse_cache_from_exp_id), (629, 627))
        self.assertTrue(args.reuse_cache)
        self.assertEqual(args.compute_device, "gpu")
        self.assertEqual(args.rf_backend_policy, "sklearn")
        self.assertTrue(args.rf_cpu_fallback)
        self.assertFalse(args.full_replica_report_only)
        self.assertEqual((args.runs, args.epochs, args.pop_size), (30, 150, 50))
        self.assertEqual((args.test_size, args.random_state, args.seed_base), (.2, 2, 1234))
        self.assertEqual(args.optimizers, study.OPTIMIZERS)
        self.assertEqual(args.estimators, study.ESTIMATORS)
        self.assertEqual(args.transfer_functions, study.TRANSFER_FUNCTIONS)

    def test_default_gpu_policy_uses_sklearn_even_when_cuml_is_available(self):
        args = self.defaults()
        with patch.object(study, "gpu_random_forest_class", return_value=object()) as probe, \
                patch.object(study, "build_gpu_random_forest", side_effect=AssertionError("cuML forbidden by policy")):
            self.assertEqual(study.resolve_rf_backend(args), "sklearn")
            classifier = study.build_run_estimator("rf", args, args.seed_base)
            self.assertIsInstance(classifier, RandomForestClassifier)
            probe.assert_not_called()
            self.assertEqual(args.compute_device, "gpu")

    def row(self, args, method, accuracy):
        return study.build_label_payload("rf", [accuracy] * args.runs, [.8] * args.runs,
            [.8] * args.runs, [.8] * args.runs, [.2] * args.runs, [2] * args.runs,
            [.01] * args.runs, [np.full(args.epochs, .2)] * args.runs, args.epochs,
            scientific_metadata=study.full_checkpoint_metadata(args, method, "Synthetic", "rf", "vstf_01"))

    def test_current_exp629_legacy_rf_precedes_read_only_exp627_imports(self):
        methods = ("MaCRO-DE-t", "DE", "JADE")
        current = cache_fixtures.arguments(self.root, 629, methods)
        current.estimators = ["rf"]
        source = deepcopy(current)
        source.exp_id, source.reuse_cache_from_exp_id = 627, None
        source_paths = study.make_paths(source)
        rows = {study.optimizer_acronym(method).upper(): self.row(source, method, 70.) for method in methods}
        study.save_cache(str(Path(source_paths.cache_dir) / "EXP627_Synthetic_rf_shared_results.pkl"), rows)
        current_paths = study.make_paths(current)
        row = self.row(current, "MaCRO-DE-t", 90.)
        row["CacheIdentity"].pop("rf_backend")  # Previously complete shared metadata, no RF provenance.
        old_file = Path(current_paths.cache_dir) / "EXP629_Synthetic_rf_old_shared_results.pkl"
        study.save_cache(str(old_file), {"MACRO-DE-T": row})
        before = cache_fixtures.snapshot(self.root / "Results/EXP627")
        legacy_bytes = old_file.read_bytes()
        with patch.object(study, "build_run_estimator", side_effect=AssertionError("RF cache reuse must not construct")):
            calls, log = self.run_mocked(current)
        self.assertEqual(calls, [])
        self.assertEqual(log.count("CACHE IMPORTED | from=EXP627"), 2)
        self.assertIn("CACHE HIT", log)
        self.assertEqual(cache_fixtures.snapshot(self.root / "Results/EXP627"), before)
        self.assertEqual(old_file.read_bytes(), legacy_bytes)
        recovered, _ = study.load_compatible_full_cache_payload(current_paths, current, "Synthetic", "rf")
        np.testing.assert_array_equal(recovered["MACRO-DE-T"]["AccRuns"], [90.] * current.runs)
        np.testing.assert_array_equal(recovered["DE"]["AccRuns"], [70.] * current.runs)

    def test_default_sklearn_identity_rejects_cuml_and_preserves_labels(self):
        args = self.defaults()
        row = self.row(args, "MaCRO-DE-t", 80.)
        expected = study.full_checkpoint_metadata(args, "MaCRO-DE-t", "Synthetic", "rf", "vstf_01")
        self.assertTrue(study.checkpoint_metadata_matches(row, expected))
        row["CacheIdentity"]["rf_backend"] = "cuml"
        self.assertFalse(study.checkpoint_metadata_matches(row, expected))
        for estimator in ("knn", "svm", "rf"):
            self.assertEqual(study.build_alg_label("MaCRO-DE-t", "vstf_01", estimator, False, True),
                             f"MACRO-DE-T_{estimator.upper()}")
        self.assertEqual(expected["CacheIdentity"]["optimizer"], "MaCRO-DE-t")
        self.assertEqual(study.optimizer_display_label("MaCRO-DE-t"), "DSA-DE")


if __name__ == "__main__":
    unittest.main()
