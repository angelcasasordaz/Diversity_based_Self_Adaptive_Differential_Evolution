"""RF dependency validation after cache lookup, without GPU or optimization."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import main_best as study
from tests.test_full_cache_reuse import arguments, snapshot


RF_GPU_ERROR = "RF GPU execution requires cuML. Use COMPUTE_DEVICE='cpu', remove rf, or install cuML."


class GPURFCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Simulate a machine without cuML independently of installed packages.
        self.stack.enter_context(patch.dict(sys.modules, {"cuml": None, "cuml.ensemble": None}))
        self.output = StringIO()
        self.stack.enter_context(redirect_stdout(self.output))
        for name in ("configure_compute_backend", "report_gpu_acceptance", "validate_comparison_backend"):
            self.stack.enter_context(patch.object(study, name))
        self.stack.enter_context(patch.object(study, "resolve_dataset_specs", return_value=[
            study.DatasetSpec("Synthetic", "codesmell")]))
        self.stack.enter_context(patch.object(study, "export_mode_outputs", return_value=([], "", [], "", None)))
        self.stack.enter_context(patch.object(study, "regenerate_figures_from_cache", return_value=([], "", [], "", None)))
        self.optimizer = self.stack.enter_context(patch.object(study, "build_optimizer", side_effect=AssertionError("Optimization forbidden")))
        self.run = self.stack.enter_context(patch.object(study, "run_single", side_effect=AssertionError("Optimization forbidden")))
        self.strategy = self.stack.enter_context(patch.object(study, "select_execution_strategy", return_value=SimpleNamespace(optimizer_compute_device="gpu")))

    def args(self, exp=629, *, policy="auto"):
        args = arguments(self.root, exp, ("DE",))
        args.estimators, args.compute_device = ["rf"], "gpu"
        args.rf_cpu_fallback = False
        args.rf_backend_policy = policy
        return args

    def cache_rf(self, args, completed=None, *, legacy=False, rf_execution_runs=None):
        paths = study.make_paths(args)
        n = args.runs if completed is None else completed
        row = study.build_label_payload("rf", [80.] * n, [.8] * n, [.8] * n, [.8] * n,
            [.2] * n, [2] * n, [.01] * n, [np.full(args.epochs, .2)] * n, args.epochs,
            scientific_metadata=None if legacy else study.full_checkpoint_metadata(
                args, "DE", "Synthetic", "rf", "vstf_01"))
        if rf_execution_runs is not None:
            row["RFExecutionRuns"] = rf_execution_runs
            row["RFBackend"] = rf_execution_runs[-1]["backend"]
        if legacy:
            signature = study.build_legacy_cache_signature(args)
            path = Path(paths.cache_dir) / f"{paths.exp_tag}_Synthetic_rf_{signature}_results.pkl"
            study.save_cache(str(path), {"DE": row})
        else:
            study.save_full_optimizer_checkpoint(paths, args, "Synthetic", "rf", "DE", "vstf_01", row)
        return paths

    def invoke(self, args):
        with patch.object(study, "parse_args", return_value=args):
            study.main()

    def mock_dataset(self):
        return patch.object(study, "load_dataset", return_value=(
            "Synthetic", np.arange(80, dtype=float).reshape(20, 4), np.tile([0, 1], 10)))

    def test_gpu_rf_current_cache_hit_does_not_require_cuml(self):
        args = self.args()
        args.reuse_cache_from_exp_id = None
        paths = self.cache_rf(args)
        before = snapshot(Path(paths.cache_dir))
        with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")) as resolve, \
                patch.object(study, "load_dataset", side_effect=AssertionError("Cached dataset must not execute")):
            self.invoke(args)
        resolve.assert_not_called()
        self.run.assert_not_called()
        self.optimizer.assert_not_called()
        self.assertIn("CACHE HIT", self.output.getvalue())
        self.assertEqual(snapshot(Path(paths.cache_dir)), before)

    def test_gpu_rf_legacy_cache_import_does_not_require_cuml(self):
        source = self.args(627)
        paths = self.cache_rf(source, legacy=True)
        before = snapshot(Path(paths.cache_dir))
        current = self.args(policy="sklearn")
        with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")) as resolve, self.mock_dataset():
            self.invoke(current)
        resolve.assert_not_called()
        self.run.assert_not_called()
        self.optimizer.assert_not_called()
        self.assertIn("CACHE IMPORTED", self.output.getvalue())
        self.assertEqual(snapshot(Path(paths.cache_dir)), before)
        sig = study.build_cache_signature(current, "DE", "Synthetic", "rf", "vstf_01")
        self.assertTrue((self.root / "Results/EXP629/full/cache" / f"EXP629_Synthetic_rf_{sig}_results.pkl").is_file())

    def test_source_import_preserves_per_run_rf_backend_provenance(self):
        source = self.args(627)
        trace = [{"backend": "cuml", "version": "26.08.00", "parameters": {"n_streams": 1}}
                 for _ in range(source.runs)]
        paths = self.cache_rf(source, rf_execution_runs=trace)
        before = snapshot(Path(paths.cache_dir))
        current = self.args()
        with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("Cached RF must not construct")), self.mock_dataset():
            self.invoke(current)
        row, reason = study.load_compatible_full_cache_payload(study.make_paths(current), current, "Synthetic", "rf")
        self.assertIsNone(reason)
        self.assertEqual(row["DE"]["RFExecutionRuns"], trace)
        self.assertEqual(snapshot(Path(paths.cache_dir)), before)

    def test_gpu_rf_cache_miss_requires_cuml_before_workers_or_optimization(self):
        args = self.args()
        args.reuse_cache_from_exp_id = None
        with self.mock_dataset(), self.assertRaises(RuntimeError) as error:
            self.invoke(args)
        self.assertIsInstance(error.exception, study.CuMLUnavailableError)
        self.assertEqual(str(error.exception), RF_GPU_ERROR)
        self.assertIsInstance(error.exception.__cause__, ModuleNotFoundError)
        self.assertIn("CACHE MISS", self.output.getvalue())
        self.strategy.assert_not_called()
        self.run.assert_not_called()
        self.optimizer.assert_not_called()

    def test_gpu_rf_partial_import_still_requires_cuml_for_the_missing_run(self):
        source = self.args(627)
        paths = self.cache_rf(source, completed=1)
        before = snapshot(Path(paths.cache_dir))
        with self.mock_dataset(), self.assertRaises(RuntimeError) as error:
            self.invoke(self.args())
        self.assertEqual(str(error.exception), RF_GPU_ERROR)
        self.assertIn("CACHE IMPORTED", self.output.getvalue())
        self.assertIn("missing=1", self.output.getvalue())
        self.run.assert_not_called()
        self.assertEqual(snapshot(Path(paths.cache_dir)), before)

    def test_gpu_knn_svm_pending_runs_do_not_check_cuml_even_with_rf_selected(self):
        args = self.args()
        args.rf_cpu_fallback = True
        args.estimators = ["rf", "knn", "svm"]
        self.run.side_effect = None
        self.run.return_value = {"mocked": True}
        with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")) as resolve:
            for estimator in ("knn", "svm"):
                with self.subTest(estimator=estimator):
                    self.assertEqual(study.build_run_estimator(estimator, args, args.seed_base), estimator)
                    completed = study.execute_pending_runs(None, estimator, "DE", "vstf_01", args, [0])
                    self.assertEqual(completed, [(0, {"mocked": True})])
                    self.assertEqual(self.run.call_args.args[1], estimator)
        resolve.assert_not_called()

    def test_cpu_rf_keeps_sklearn_and_does_not_check_cuml(self):
        args = self.args()
        args.compute_device = "cpu"
        self.strategy.return_value = SimpleNamespace(optimizer_compute_device="cpu")
        self.run.side_effect = None
        self.run.return_value = {"mocked": True}
        with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")) as resolve:
            estimator = study.build_run_estimator("rf", args, args.seed_base)
            self.assertEqual(estimator, "rf")
            self.assertIsInstance(study.get_general_estimator("classification", estimator), RandomForestClassifier)
            completed = study.execute_pending_runs(None, "rf", "DE", "vstf_01", args, [0])
            self.assertEqual(completed, [(0, {"mocked": True})])
        resolve.assert_not_called()

    def test_gpu_rf_prefers_cuml_when_available_even_with_fallback_enabled(self):
        class FakeCuMLRF:
            def __init__(self, *, n_streams=4, max_depth=None, **kwargs):
                self.n_streams, self.max_depth, self.parameters = n_streams, max_depth, kwargs
        for fallback in (False, True):
            with self.subTest(fallback=fallback), \
                    patch.object(study, "gpu_random_forest_class", return_value=FakeCuMLRF), \
                    patch.object(study, "get_general_estimator", side_effect=AssertionError("CPU RF forbidden")):
                args = self.args()
                args.rf_cpu_fallback, args._rf_use_sklearn = fallback, True
                classifier = study.build_run_estimator("rf", args, args.seed_base)
                self.assertIsInstance(classifier, FakeCuMLRF)
                self.assertEqual(classifier.n_streams, 1)
                self.assertIsNone(classifier.max_depth)
                self.assertEqual(classifier.parameters["random_state"], args.seed_base)
                self.assertEqual(classifier.parameters["output_type"], "numpy")
                self.assertFalse(args._rf_use_sklearn)
                self.assertEqual(args.compute_device, "gpu")

    def test_gpu_rf_pending_runs_do_not_compete_for_forest_vram(self):
        args = self.args()
        args.parallel, args.n_workers, args.rf_cpu_fallback = "yes", 8, True
        self.run.side_effect = None
        self.run.return_value = {"mocked": True}
        with patch.object(study, "ProcessPoolExecutor", side_effect=AssertionError("RF runs must remain serial")):
            completed = study.execute_pending_runs(None, "rf", "DE", "vstf_01", args, [0, 1])
        self.assertEqual(completed, [(0, {"mocked": True}), (1, {"mocked": True})])
        self.assertEqual(self.run.call_count, 2)
        self.assertTrue(all(call.args[4].optimizer_compute_device == "gpu" for call in self.run.call_args_list))
        self.assertIn("RF GPU run concurrency limited to 1", self.output.getvalue())

    def test_cuml_recomputes_partial_legacy_rf_instead_of_creating_mixed_row(self):
        source = self.args(627)
        paths = self.cache_rf(source, completed=1, legacy=True)
        before = snapshot(Path(paths.cache_dir))
        args = self.args(policy="cuml")
        args.rf_cpu_fallback = True
        signature = study.build_cache_signature(args, "DE", "Synthetic", "rf", "vstf_01")
        self.run.side_effect = None
        self.run.return_value = dict(as_test=80., ps_test=.8, rs_test=.8, f1_test=.8,
            fit_final=.2, n_features=2, runtime=.01, curve=np.full(args.epochs, .2),
            rf_backend="cuml", rf_execution={"backend": "cuml", "version": "26.08.00",
                                           "parameters": {"n_streams": 1, "random_state": args.seed_base + 1}})
        with self.mock_dataset(), patch.object(study, "build_gpu_random_forest", return_value=object()):
            self.invoke(args)
        payload, reason = study.load_compatible_full_cache_payload(study.make_paths(args), args, "Synthetic", "rf")
        self.assertIsNone(reason)
        self.assertEqual(payload["DE"]["RFBackend"], "cuml")
        self.assertEqual(payload["DE"]["RFExecutionRuns"][0]["backend"], "cuml")
        self.assertEqual(payload["DE"]["RFExecutionRuns"][1]["backend"], "cuml")
        self.assertEqual(payload["DE"]["RFExecutionRuns"][1]["version"], "26.08.00")
        self.assertEqual(signature, study.build_cache_signature(args, "DE", "Synthetic", "rf", "vstf_01"))
        self.assertEqual(snapshot(Path(paths.cache_dir)), before)
        self.assertEqual(self.run.call_count, args.runs)
        self.assertIn("actual=sklearn/unknown", self.output.getvalue())

    def test_gpu_rf_with_no_pending_runs_does_not_check_cuml(self):
        args = self.args()
        with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")) as resolve:
            self.assertEqual(study.execute_pending_runs(None, "rf", "DE", "vstf_01", args, []), [])
        resolve.assert_not_called()
        self.run.assert_not_called()

    def test_gpu_rf_cache_miss_uses_sklearn_only_with_explicit_fallback(self):
        args = self.args()
        signature = study.build_cache_signature(args, "DE", "Synthetic", "rf", "vstf_01")
        args.rf_cpu_fallback = True
        self.assertNotEqual(signature, study.build_cache_signature(args, "DE", "Synthetic", "rf", "vstf_01"))
        args.reuse_cache_from_exp_id = None
        def execute(data, estimator, method, tf, worker_args, seed, **kwargs):
            self.assertEqual(worker_args.compute_device, "gpu")
            self.assertEqual(worker_args.optimizer_compute_device, "gpu")
            chosen = study.build_run_estimator(estimator, worker_args, seed)
            self.assertIsInstance(chosen, RandomForestClassifier)
            self.assertEqual(chosen.get_params(), RandomForestClassifier().get_params())
            return dict(as_test=80., ps_test=.8, rs_test=.8, f1_test=.8, fit_final=.2,
                        n_features=2, runtime=.01, curve=np.full(args.epochs, .2), rf_backend="sklearn")
        self.run.side_effect = execute
        with self.mock_dataset():
            self.invoke(args)
        self.assertEqual(self.run.call_count, args.runs)
        self.assertIn("cuML unavailable; using sklearn RF on CPU while optimizer backend remains GPU.", self.output.getvalue())
        payload, reason = study.load_compatible_full_cache_payload(study.make_paths(args), args, "Synthetic", "rf")
        self.assertIsNone(reason)
        self.assertEqual(payload["DE"]["RFBackend"], "sklearn")

    def test_fallback_enabled_skips_cuml_preflight_and_gpu_rf_construction(self):
        args = self.args(policy="sklearn")
        args.rf_cpu_fallback = True
        with patch.object(study, "build_gpu_random_forest", side_effect=AssertionError("GPU RF must not be constructed")), \
                patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("Preflight must not check cuML")) as resolve:
            study.validate_gpu_random_forest_backend(args, "rf")
            resolve.assert_not_called()
        with patch.object(study, "build_gpu_random_forest", side_effect=AssertionError("GPU RF must not be constructed")) as build:
            self.assertIsInstance(study.build_run_estimator("rf", args, args.seed_base), RandomForestClassifier)
            build.assert_not_called()

    def test_global_fallback_is_used_when_namespace_omits_flag(self):
        args = self.args()
        del args.rf_cpu_fallback
        with patch.object(study, "RF_CPU_FALLBACK", True):
            study.validate_gpu_random_forest_backend(args, "rf")
            self.assertIsInstance(study.build_run_estimator("rf", args, args.seed_base), RandomForestClassifier)

    def test_explicit_disabled_fallback_requires_cuml_even_with_global_enabled(self):
        args = self.args()
        with patch.object(study, "RF_CPU_FALLBACK", True), self.assertRaises(study.CuMLUnavailableError) as error:
            study.build_run_estimator("rf", args, args.seed_base)
        self.assertEqual(str(error.exception), RF_GPU_ERROR)

    def test_cli_and_global_fallback_reach_worker_arguments(self):
        for argv, configured in ((["main_best.py", "--rf-cpu-fallback"], False), (["main_best.py"], True)):
            with self.subTest(argv=argv, configured=configured), \
                    patch.object(study, "RF_CPU_FALLBACK", configured), patch.object(sys, "argv", argv):
                args = study.parse_args()
                self.assertTrue(args.rf_cpu_fallback)
                cloned = study.clone_args_for_mode(args, "full")
                cloned.compute_device = "gpu"
                self.assertIsInstance(study.build_run_estimator("rf", cloned, cloned.seed_base), RandomForestClassifier)

    def test_rf_evaluation_recovery_respects_cpu_fallback(self):
        args = self.args()
        args.rf_cpu_fallback = True
        data = SimpleNamespace(X_train=np.array([[0.], [1.], [0.], [1.]]),
                               y_train=np.array([0, 1, 0, 1]),
                               X_test=np.array([[0.], [1.]]), y_test=np.array([0, 1]))
        class FakeSelector:
            def __init__(self, problem, estimator, optimizer, **kwargs):
                self.estimator = estimator
                self.optimizer = SimpleNamespace(history=SimpleNamespace(list_global_best_fit=[.2]))

            def fit(self, X, y):
                pass

            def transform(self, X):
                return X

            def evaluate(self, **kwargs):
                raise ValueError("Invalid y_pred")

        with patch.object(study, "MhaSelector", FakeSelector), \
                patch.object(study, "build_optimizer", return_value=object()), \
                patch.object(study, "build_gpu_random_forest", side_effect=AssertionError("Recovery must not construct GPU RF")) as build:
            result = study._run_single(data, "rf", "DE", "vstf_01", args, args.seed_base)
        build.assert_not_called()
        self.assertEqual(result["as_test"], 100.)
        self.assertEqual(result["rf_backend"], "sklearn")
        self.assertEqual(args.compute_device, "gpu")

    def test_current_shared_prefix_digest_recovers_rf_without_touching_old_cache(self):
        args = self.args(policy="sklearn")
        args.optimizers = ["DE", "JADE", "SHADE"]
        args.reuse_cache_from_exp_id = None
        paths = self.cache_rf(args, legacy=True)
        before = snapshot(Path(paths.cache_dir))
        payload, reason = study.load_compatible_full_cache_payload(paths, args, "Synthetic", "rf")
        self.assertIsNone(reason)
        self.assertEqual(payload["DE"]["CompletedRuns"], args.runs)
        study.materialize_current_full_cache(paths, args, ["Synthetic"])
        after = snapshot(Path(paths.cache_dir))
        for filename, value in before.items():
            self.assertEqual(after[filename], value)
        self.assertGreater(len(after), len(before))
        changed = deepcopy(args)
        changed.seed_base += 1
        payload, reason = study.load_compatible_full_cache_payload(paths, changed, "Synthetic", "rf")
        self.assertIsNone(payload)
        self.assertIn("scientific metadata", reason)

    def test_current_local_then_shared_cache_precedes_source_exp(self):
        args = self.args()
        paths = self.cache_rf(args)
        # Create a scientific-compatible shared row with different observations.
        row = study.load_cache(str(next(Path(paths.cache_dir).glob("*results.pkl"))))
        for value in row.values():
            value["AccRuns"] = np.array([91.] * args.runs)
            value["AccMean"] = 91.
        study.save_cache(str(Path(paths.cache_dir) / "EXP629_Synthetic_rf_shared_results.pkl"), row)
        self.cache_rf(self.args(627))
        loader = study.load_compatible_full_cache_payload
        def load(*values, **kwargs):
            self.assertNotEqual(values[0].exp_tag, "EXP627", "Source EXP should not be read for a complete current cache")
            return loader(*values, **kwargs)
        with patch.object(study, "load_compatible_full_cache_payload", side_effect=load), \
                patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")):
            self.invoke(args)
        chosen, _ = loader(paths, args, "Synthetic", "rf")
        np.testing.assert_array_equal(chosen["DE"]["AccRuns"], [80.] * args.runs)

    def test_complete_current_shared_rf_import_precedes_source_and_cuml(self):
        args = self.args(policy="sklearn")
        paths = self.cache_rf(args, legacy=True)
        self.cache_rf(self.args(627))
        before = snapshot(Path(paths.cache_dir))
        loader = study.load_compatible_full_cache_payload
        def load(*values, **kwargs):
            self.assertEqual(values[0].exp_tag, "EXP629")
            return loader(*values, **kwargs)
        with patch.object(study, "load_compatible_full_cache_payload", side_effect=load), \
                patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")):
            self.invoke(args)
        self.assertIn("from=EXP629 shared/legacy", self.output.getvalue())
        after = snapshot(Path(paths.cache_dir))
        for filename, value in before.items():
            self.assertEqual(after[filename], value)

    def test_matching_shared_continuation_completes_current_local_prefix(self):
        args = self.args(policy="sklearn")
        paths = self.cache_rf(args, completed=1)
        self.cache_rf(args, legacy=True)
        payload, reason = study.load_compatible_full_cache_payload(paths, args, "Synthetic", "rf")
        self.assertIsNone(reason)
        self.assertEqual(payload["DE"]["CompletedRuns"], args.runs)
        with patch.object(study, "gpu_random_forest_class", side_effect=AssertionError("cuML must not be checked")):
            self.invoke(args)
        self.run.assert_not_called()

    def test_fallback_does_not_hide_cuml_constructor_errors(self):
        args = self.args()
        args.rf_cpu_fallback = True
        with patch.object(study, "gpu_random_forest_class", return_value=object()), \
                patch.object(study, "build_gpu_random_forest", side_effect=RuntimeError("constructor failure")):
            with self.assertRaisesRegex(RuntimeError, "constructor failure"):
                study.build_run_estimator("rf", args, args.seed_base)


if __name__ == "__main__":
    unittest.main()
