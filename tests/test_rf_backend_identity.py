"""Scientific RF backend separation; temporary caches and mocked execution only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import main_best as study
import tests.test_full_cache_reuse as fixtures


BASELINES = ("MaCRO-DE-t", "DE", "JADE", "SHADE", "PSO", "WOA", "HHO", "GOA", "SA", "BRO", "RUN", "FOX")


class RFBackendIdentityTests(unittest.TestCase):
    setUp = fixtures.FullCacheReuseTests.setUp
    run_mocked = fixtures.FullCacheReuseTests.run_mocked

    def args(self, backend, exp=627, optimizers=("DE", "JADE")):
        args = fixtures.arguments(self.root, exp, optimizers)
        args.estimators = ["rf"]
        args.compute_device = "gpu" if backend == "cuml" else "cpu"
        args.rf_backend_policy = backend
        args.rf_cpu_fallback = True
        return args

    def row(self, args, method="DE", *, metadata=True):
        return study.build_label_payload("rf", [80.] * args.runs, [.8] * args.runs,
            [.8] * args.runs, [.8] * args.runs, [.2] * args.runs, [2] * args.runs,
            [.01] * args.runs, [np.full(args.epochs, .2)] * args.runs, args.epochs,
            scientific_metadata=study.full_checkpoint_metadata(args, method, "Synthetic", "rf", "vstf_01") if metadata else None)

    def legacy_source(self):
        args = self.args("sklearn")
        paths = study.make_paths(args)
        rows = {study.optimizer_acronym(method).upper(): self.row(args, method, metadata=False)
                for method in args.optimizers}
        file = Path(paths.cache_dir) / f"EXP627_Synthetic_rf_{study.build_legacy_cache_signature(args)}_results.pkl"
        study.save_cache(str(file), rows)
        return args, paths

    def test_sklearn_rf_results_are_not_reused_for_cuml_rf(self):
        old = self.args("sklearn", optimizers=BASELINES)
        self.run_mocked(old)
        before = fixtures.snapshot(self.root / "Results/EXP627")
        current = self.args("cuml", 629, BASELINES)
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [(study.optimizer_acronym(method), [0, 1]) for method in BASELINES])
        self.assertIn("CACHE SOURCE INCOMPATIBLE", log)
        self.assertIn("expected=cuml, actual=sklearn", log)
        self.assertNotIn("CACHE IMPORTED", log)
        self.assertEqual(fixtures.snapshot(self.root / "Results/EXP627"), before)
        for method in BASELINES:
            self.assertNotEqual(study.build_cache_signature(old, method, "Synthetic", "rf", "vstf_01"),
                                study.build_cache_signature(current, method, "Synthetic", "rf", "vstf_01"))
        self.assertEqual(self.run_mocked(current)[0], [])

    def test_cuml_rf_results_are_not_reused_for_sklearn_rf(self):
        old = self.args("cuml")
        self.run_mocked(old)
        before = fixtures.snapshot(self.root / "Results/EXP627")
        current = self.args("sklearn", 629)
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [("DE", [0, 1]), ("JADE", [0, 1])])
        self.assertIn("expected=sklearn, actual=cuml", log)
        self.assertNotIn("CACHE IMPORTED", log)
        self.assertEqual(fixtures.snapshot(self.root / "Results/EXP627"), before)

    def test_knn_svm_signatures_and_reuse_do_not_depend_on_rf_policy(self):
        for classifier in ("knn", "svm"):
            with self.subTest(classifier=classifier):
                old = self.args("sklearn")
                old.estimators = [classifier]
                signature = study.build_cache_signature(old, "DE", "Synthetic", classifier, "vstf_01")
                self.run_mocked(old)
                current = deepcopy(old)
                current.exp_id, current.reuse_cache_from_exp_id = 629, 627
                current.compute_device = "gpu"
                for policy in ("auto", "sklearn", "cuml"):
                    current.rf_backend_policy = policy
                    with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("Non-RF must not resolve cuML")):
                        self.assertEqual(signature, study.build_cache_signature(current, "DE", "Synthetic", classifier, "vstf_01"))
                        self.assertEqual(study.classifier_cache_signature(current, "historic", classifier), "historic")
                        calls, _ = self.run_mocked(current)
                        self.assertEqual(calls, [])

    def test_legacy_unknown_is_sklearn_only_and_logged_without_source_writes(self):
        _, paths = self.legacy_source()
        before = fixtures.snapshot(Path(paths.cache_dir))
        current = self.args("sklearn", 629)
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [])
        self.assertIn("rf_backend=sklearn", log)
        self.assertIn("provenance=legacy_unknown_assumed_sklearn", log)
        self.assertEqual(fixtures.snapshot(Path(paths.cache_dir)), before)
        current = self.args("cuml", 630)
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [("DE", [0, 1]), ("JADE", [0, 1])])
        self.assertIn("actual=sklearn/unknown (legacy provenance absent)", log)
        self.assertEqual(fixtures.snapshot(Path(paths.cache_dir)), before)

    def test_old_complete_identity_can_use_recorded_cuml_but_not_unknown(self):
        args = self.args("cuml")
        paths = study.make_paths(args)
        args.optimizers = ["DE"]
        row = self.row(args)
        row["CacheIdentity"].pop("rf_backend")
        file = Path(paths.cache_dir) / "EXP627_Synthetic_rf_old_shared_results.pkl"
        study.save_cache(str(file), {"DE": row})
        payload, reason = study.load_compatible_full_cache_payload(paths, args, "Synthetic", "rf")
        self.assertIsNone(payload)
        self.assertIn("actual=sklearn/unknown", reason)
        row["RFBackend"] = "cuml"
        # A separate historical file; never replace the original observations.
        study.save_cache(str(Path(paths.cache_dir) / "EXP627_Synthetic_rf_recorded_shared_results.pkl"), {"DE": row})
        payload, _ = study.load_compatible_full_cache_payload(paths, args, "Synthetic", "rf")
        self.assertEqual(payload["DE"]["CacheIdentity"]["rf_backend"], "cuml")
        changed = deepcopy(args)
        changed.seed_base += 1
        payload, reason = study.load_compatible_full_cache_payload(paths, changed, "Synthetic", "rf")
        self.assertIsNone(payload)
        self.assertIn("FULL scientific metadata", reason)

    def test_mixed_or_contradictory_provenance_is_never_reused(self):
        args = self.args("cuml")
        for trace in ({"RFBackend": "sklearn"}, {"RFBackend": "mixed"},
                      {"RFExecutionRuns": [{"backend": "unknown"}] * args.runs},
                      {"RFExecutionRuns": [{"backend": "cuml"}]}):
            with self.subTest(trace=trace):
                row = {**self.row(args), **trace}
                values, reason = study.validate_source_label_runs(row, "rf", args.runs,
                    study.full_checkpoint_metadata(args, "DE", "Synthetic", "rf", "vstf_01"))
                self.assertIsNone(values)
                self.assertIn("RF backend provenance", reason)

    def test_all_optimizers_share_selection_and_explicit_cuml_forbids_fallback(self):
        args = self.args("cuml", optimizers=BASELINES)
        sentinel = object()
        with patch.object(study, "build_gpu_random_forest", return_value=sentinel) as build:
            for method in BASELINES:
                worker = deepcopy(args)
                worker.optimizers = [method]
                self.assertEqual(study.full_cache_identity(worker, method, "Synthetic", "rf", "vstf_01")["rf_backend"], "cuml")
                self.assertIs(study.build_run_estimator("rf", worker, args.seed_base), sentinel)
            self.assertEqual(build.call_count, len(BASELINES))
        with patch.dict(sys.modules, {"cuml": None, "cuml.ensemble": None}), self.assertRaises(study.CuMLUnavailableError):
            study.build_run_estimator("rf", args, args.seed_base)
        args.rf_backend_policy = "sklearn"
        with patch.object(study, "build_gpu_random_forest", side_effect=AssertionError("Explicit sklearn must not construct GPU RF")):
            self.assertIsInstance(study.build_run_estimator("rf", args, args.seed_base), RandomForestClassifier)
            self.assertEqual(args.compute_device, "gpu")
        args.rf_backend_policy, args.compute_device = "cuml", "cpu"
        self.assertEqual(study.resolve_rf_backend(args), "sklearn")

    def test_malformed_backend_metadata_is_rejected_without_crashing(self):
        args = self.args("cuml")
        for metadata in ({"RFBackend": []}, {"RFScientificBackend": {}},
                         {"RFExecutionRuns": [{"backend": {}}] * args.runs}):
            with self.subTest(metadata=metadata):
                row = {**self.row(args), **metadata}
                self.assertIn("invalid", study.rf_backend_compatibility(row, "cuml"))
        row = self.row(args)
        row["CacheIdentity"]["rf_backend"] = []
        self.assertIn("invalid", study.rf_backend_compatibility(row, "cuml"))

    def test_other_modes_separate_rf_files_and_verify_legacy_backend(self):
        for mode in ("ablation", "sensitivity", "sensitivity_weights", "transfer_functions"):
            with self.subTest(mode=mode):
                args = self.args("sklearn")
                args.experiment_mode = mode
                paths = study.make_paths(args)
                signature = study.build_cache_signature(args)
                row = self.row(args, metadata=False)
                filename = Path(paths.cache_dir) / f"EXP627_Synthetic_rf_{signature}_results.pkl"
                study.save_cache(str(filename), {"DE": row})
                payload, reason = study.load_mode_cache_payload(paths, args, "Synthetic", "rf", signature)
                self.assertIsNone(reason)
                self.assertIsNotNone(payload)
                current = deepcopy(args)
                current.compute_device, current.rf_backend_policy = "gpu", "cuml"
                self.assertNotEqual(study.classifier_cache_signature(args, signature, "rf"),
                                    study.classifier_cache_signature(current, signature, "rf"))
                payload, reason = study.load_mode_cache_payload(paths, current, "Synthetic", "rf", signature)
                self.assertIsNone(payload)
                self.assertIn("actual=sklearn/unknown", reason)


if __name__ == "__main__":
    unittest.main()
