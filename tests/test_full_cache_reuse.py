"""Optimizer-local FULL cache tests; all execution and exports are mocked."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main_best as study


def arguments(root, exp=627, optimizers=("DE", "JADE")):
    argv = ["main_best.py", "--experiment-mode", "full", "--exp-id", str(exp),
            "--output-root", str(root), "--datasets", "Synthetic", "--estimators", "knn",
            "--optimizers", *optimizers, "--runs", "2", "--epochs", "3", "--pop-size", "10",
            "--parallel", "no"]
    with patch.object(sys, "argv", argv):
        args = study.parse_args()
    args.reuse_cache_from_exp_id = None if exp == 627 else 627
    return args


def snapshot(directory):
    return {p.relative_to(directory): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def aggregate_path(args):
    tag = f"EXP{args.exp_id:03d}"
    return (Path(args.output_root) / "Results" / tag / "full/cache" /
            f"{tag}_Synthetic_knn_{study.build_cache_signature(args)}_results.pkl")


class FullCacheReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_mocked(self, args):
        calls, output = [], StringIO()
        save = study.save_cache
        destination = self.root / f"Results/EXP{args.exp_id:03d}"
        def save_destination_only(path, payload):
            self.assertTrue(Path(path).is_relative_to(destination),
                            f"Attempted cache write outside destination EXP: {path}")
            save(path, payload)
        def execute(data, estimator, method, tf, run_args, pending, on_run_complete, **kwargs):
            calls.append((study.optimizer_acronym(method), list(pending)))
            for run in pending:
                on_run_complete(run, dict(as_test=80. + run, ps_test=.8, rs_test=.8,
                    f1_test=.8, fit_final=.2, n_features=2, runtime=.01,
                    curve=np.full(run_args.epochs, .2)))
        x = np.arange(80, dtype=float).reshape(20, 4)
        with redirect_stdout(output), \
                patch.object(study, "resolve_dataset_specs", return_value=[study.DatasetSpec("Synthetic", "codesmell")]), \
                patch.object(study, "load_dataset", return_value=("Synthetic", x, np.tile([0, 1], 10))), \
                patch.object(study, "execute_pending_runs", side_effect=execute), \
                patch.object(study, "save_cache", side_effect=save_destination_only), \
                patch.object(study, "export_mode_outputs", return_value=([], "", [], "", None)), \
                patch.object(study, "regenerate_figures_from_cache", return_value=([], "", [], "", None)):
            study.run_experiment_mode(args)
        return calls, output.getvalue()

    def source(self):
        args = arguments(self.root)
        calls, _ = self.run_mocked(args)
        self.assertEqual(calls, [("DE", [0, 1]), ("JADE", [0, 1])])
        return args, self.root / "Results/EXP627"

    def test_comparison_change_imports_baselines_and_only_executes_missing_optimizer(self):
        old, source = self.source()
        before = snapshot(source)
        current = arguments(self.root, 629, ("MaCRO-DE-t", "DE", "JADE"))
        self.assertEqual(study.build_cache_signature(old), study.build_cache_signature(current))
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [("MaCRO-DE-t", [0, 1])])
        self.assertEqual(log.count("CACHE IMPORTED"), 2)
        self.assertIn("CACHE MISS", log)
        self.assertNotIn("label=MACRO-DE-T | runs=2 | completed", log)
        self.assertEqual(snapshot(source), before)
        # Changing the selection again reuses current rows independently.
        removed = arguments(self.root, 629, ("JADE",))
        calls, _ = self.run_mocked(removed)
        self.assertEqual(calls, [])
        self.assertEqual(snapshot(source), before)

    def test_scientific_changes_reject_rows_and_execution_settings_do_not(self):
        args, source = self.source()
        paths = study.make_read_only_source_paths(arguments(self.root, 629))
        before = snapshot(source)
        for field, value in (("epochs", 4), ("pop_size", 11), ("runs", 3),
                             ("test_size", .3), ("random_state", 9), ("seed_base", 50),
                             ("fitness_alpha", .8), ("fitness_beta", .2),
                             ("dataset_source", "mafese")):
            with self.subTest(field=field):
                changed = deepcopy(args)
                setattr(changed, field, value)
                payload, reason = study.load_compatible_full_cache_payload(paths, changed, "Synthetic", "knn")
                self.assertIsNone(payload)
                self.assertIn("scientific metadata", reason)
        changed = deepcopy(args)
        changed.compute_device, changed.n_workers = "gpu", 50
        changed.gpu_device_id, changed.gpu_memory_fraction = 2, .5
        changed.dsade_beta_min, changed.dsade_mahal_q = .2, .9
        payload, reason = study.load_compatible_full_cache_payload(paths, changed, "Synthetic", "knn")
        self.assertIsNone(reason)
        self.assertEqual(set(payload), {"DE", "JADE"})
        self.assertEqual(snapshot(source), before)

    def test_optimizer_parameters_revision_tf_and_classifier_control_local_identity(self):
        args = arguments(self.root)
        def sig(a, method="MaCRO-DE-t", dataset="Synthetic", estimator="knn", tf="vstf_01"):
            return study.build_cache_signature(a, method, dataset, estimator, tf)
        baseline, macro = sig(args, "DE"), sig(args)
        changed = deepcopy(args)
        changed.optimizers = ["MaCRO-DE-t", "DE", "JADE"]
        self.assertEqual(baseline, sig(changed, "DE"))
        changed.dsade_mahal_q += .1
        self.assertEqual(baseline, sig(changed, "DE"))
        self.assertNotEqual(macro, sig(changed))
        self.assertNotEqual(baseline, sig(args, "JADE"))
        for kwargs in ({"dataset": "Other"}, {"estimator": "svm"}, {"tf": "vstf_02"}):
            self.assertNotEqual(macro, sig(args, **kwargs))
        cls = study.resolve_optimizer("MaCRO-DE-t").optimizer_class
        with patch.object(cls, "IMPLEMENTATION_REVISION", "changed"):
            self.assertNotEqual(macro, sig(args))

    def test_legacy_comparison_digest_is_verified_before_wildcard_import(self):
        old, source = self.source()
        cache = source / "full/cache"
        payload = study.load_cache(str(aggregate_path(old)))
        for row in payload.values():
            row.pop("CacheIdentity")
            row.pop("ExperimentMode")
        for path in cache.glob("*.pkl"):
            path.unlink()
        legacy = cache / f"EXP627_Synthetic_knn_{study.build_legacy_cache_signature(old)}_results.pkl"
        study.save_cache(str(legacy), payload)
        before = snapshot(source)
        current = arguments(self.root, 629, ("MaCRO-DE-t", "DE", "JADE"))
        self.assertNotEqual(study.build_legacy_cache_signature(old), study.build_cache_signature(current))
        for method in ("DE", "JADE"):
            self.assertNotEqual(study.build_legacy_cache_signature(old),
                study.build_cache_signature(current, method, "Synthetic", "knn", "vstf_01"))
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [("MaCRO-DE-t", [0, 1])])
        self.assertEqual(log.count("CACHE IMPORTED"), 2)
        self.assertEqual(snapshot(source), before)
        destination = self.root / "Results/EXP629/full/cache"
        for method in ("DE", "JADE", "MaCRO-DE-t"):
            signature = study.build_cache_signature(current, method, "Synthetic", "knn", "vstf_01")
            local = destination / f"EXP629_Synthetic_knn_{signature}_results.pkl"
            self.assertTrue(local.is_file())
            row, = study.load_cache(str(local)).values()
            self.assertEqual(row["CacheIdentity"], study.full_cache_identity(
                current, method, "Synthetic", "knn", "vstf_01"))
            self.assertEqual(row["CompletedRuns"], 2)
        changed = deepcopy(current)
        changed.seed_base += 1
        imported, reason = study.load_compatible_full_cache_payload(
            study.make_read_only_source_paths(changed), changed, "Synthetic", "knn")
        self.assertIsNone(imported)
        self.assertIn("scientific metadata", reason)

    def test_incompatible_longer_candidate_does_not_hide_compatible_partial(self):
        args, source = self.source()
        cache = source / "full/cache"
        payload = study.load_cache(str(aggregate_path(args)))
        incompatible = deepcopy(payload)
        incompatible["DE"]["CacheIdentity"]["seed_base"] += 1
        partial = deepcopy(payload)
        partial["DE"] = study.build_label_payload("knn", [80.], [.8], [.8], [.8], [.2], [2],
            [.01], [np.full(3, .2)], 3, study.full_checkpoint_metadata(args, "DE", "Synthetic", "knn", "vstf_01"))
        for path in cache.glob("*.pkl"):
            path.unlink()
        study.save_cache(str(cache / "EXP627_Synthetic_knn_aaa_results.pkl"), incompatible)
        study.save_cache(str(cache / "EXP627_Synthetic_knn_bbb_progress.pkl"), partial)
        current = arguments(self.root, 629)
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [("DE", [1])])
        self.assertIn("CACHE SOURCE INCOMPATIBLE", log)
        self.assertEqual(log.count("CACHE IMPORTED"), 2)

    def test_source_metadata_and_run_arrays_are_not_relaxed(self):
        args, source = self.source()
        cache = source / "full/cache"
        path = aggregate_path(args)
        original = study.load_cache(str(path))
        for candidate in cache.glob("*.pkl"):
            candidate.unlink()
        current = arguments(self.root, 629)
        paths = study.make_read_only_source_paths(current)
        for mutation in (lambda row: row.pop("CacheIdentity"),
                         lambda row: row.update(Estimator="rf"),
                         lambda row: row.update(CompletedRuns=3),
                         lambda row: row.update(FitRuns=[])):
            with self.subTest(mutation=mutation):
                payload = deepcopy(original)
                mutation(payload["DE"])
                study.save_cache(str(path), payload)
                selected, reason = study.load_compatible_full_cache_payload(paths, current, "Synthetic", "knn")
                self.assertEqual(set(selected), {"JADE"})
                self.assertIsNotNone(reason)

    def test_removed_rows_are_preserved_when_a_new_optimizer_is_added(self):
        args = arguments(self.root, 629)
        args.reuse_cache_from_exp_id = None
        self.run_mocked(args)
        changed = arguments(self.root, 629, ("DE", "PSO"))
        changed.reuse_cache_from_exp_id = None
        calls, log = self.run_mocked(changed)
        self.assertEqual(calls, [("PSO", [0, 1])])
        self.assertIn("CACHE HIT", log)
        path = aggregate_path(changed)
        self.assertEqual(set(study.load_cache(str(path))), {"DE", "JADE"})
        payload, _ = study.load_compatible_full_cache_payload(
            study.make_paths(changed), changed, "Synthetic", "knn")
        self.assertEqual(set(payload), {"DE", "PSO"})
        from reporting.core import load_completed_cache, report_guard
        with report_guard(()):
            report = load_completed_cache(changed)
        self.assertEqual(report.algorithms, ["DE", "PSO"])

    def test_changed_optimizer_science_recomputes_only_that_optimizer(self):
        args = arguments(self.root, 629, ("MaCRO-DE-t", "DE", "JADE"))
        args.reuse_cache_from_exp_id = None
        self.run_mocked(args)
        changed = deepcopy(args)
        changed.dsade_mahal_q += .1
        calls, log = self.run_mocked(changed)
        self.assertEqual(calls, [("MaCRO-DE-t", [0, 1])])
        self.assertIn("CACHE SOURCE INCOMPATIBLE", log)
        self.assertEqual(log.count("CACHE HIT"), 2)
        # Both scientific configurations retain their own checkpoint files.
        for settings in (args, changed):
            signature = study.build_cache_signature(settings, "MaCRO-DE-t", "Synthetic", "knn", "vstf_01")
            path = self.root / "Results/EXP629/full/cache" / f"EXP629_Synthetic_knn_{signature}_results.pkl"
            self.assertTrue(path.is_file())
        calls, _ = self.run_mocked(args)
        self.assertEqual(calls, [])

    def test_complete_metadata_is_accepted_under_an_unrelated_filename_signature(self):
        args, source = self.source()
        cache = source / "full/cache"
        payload = study.load_cache(str(aggregate_path(args)))
        for candidate in cache.glob("*.pkl"):
            candidate.unlink()
        path = cache / "EXP627_Synthetic_knn_unrelated_old_comparison_results.pkl"
        study.save_cache(str(path), payload)
        before = snapshot(source)
        current = arguments(self.root, 629, ("MaCRO-DE-t", "DE", "JADE"))
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [("MaCRO-DE-t", [0, 1])])
        self.assertEqual(log.count("CACHE IMPORTED"), 2)
        self.assertNotIn("CACHE SOURCE INCOMPATIBLE", log)
        self.assertEqual(snapshot(source), before)

    def test_current_optimizer_local_files_work_without_the_shared_comparison_file(self):
        args = arguments(self.root, 629)
        args.reuse_cache_from_exp_id = None
        self.run_mocked(args)
        aggregate_path(args).unlink()
        aggregate_path(args).with_name(aggregate_path(args).name.replace("results", "progress")).unlink()
        current = arguments(self.root, 629, ("MaCRO-DE-t", "DE", "JADE"))
        current.reuse_cache_from_exp_id = None
        calls, log = self.run_mocked(current)
        self.assertEqual(calls, [("MaCRO-DE-t", [0, 1])])
        self.assertEqual(log.count("CACHE HIT"), 2)

    def test_local_writer_rejects_the_read_only_source_experiment(self):
        args, source = self.source()
        row = study.load_cache(str(aggregate_path(args)))["DE"]
        current = arguments(self.root, 629)
        before = snapshot(source)
        with self.assertRaisesRegex(ValueError, "destination experiment"):
            study.save_full_optimizer_checkpoint(study.make_read_only_source_paths(current),
                current, "Synthetic", "knn", "DE", "vstf_01", row)
        self.assertEqual(snapshot(source), before)

    def test_invalid_current_run_arrays_are_recomputed(self):
        args = arguments(self.root, 629)
        args.reuse_cache_from_exp_id = None
        self.run_mocked(args)
        cache = self.root / "Results/EXP629/full/cache"
        for path in cache.glob("*.pkl"):
            payload = study.load_cache(str(path))
            for row in payload.values():
                if row["CacheIdentity"]["optimizer"] == study.resolve_optimizer_name("DE"):
                    row["FitRuns"] = []
            study.save_cache(str(path), payload)
        calls, log = self.run_mocked(args)
        self.assertEqual(calls, [("DE", [0, 1])])
        self.assertIn("CACHE SOURCE INCOMPATIBLE", log)

    def test_saved_exp627_baselines_are_discovered_without_writes(self):
        root = Path(study.__file__).resolve().parent
        cache = root / "Results/EXP627/full/cache"
        if not cache.is_dir():
            self.skipTest("Historical EXP627 caches unavailable")
        with patch.object(sys, "argv", ["main_best.py", "--experiment-mode", "full"]):
            args = study.parse_args()
        args.exp_id, args.reuse_cache_from_exp_id, args.output_root = 629, 627, str(root)
        args.rf_backend_policy = "sklearn"  # Historical RF caches predate cuML execution.
        args.optimizers = ["MaCRO-DE-t", "DE", "JADE", "SHADE", "PSO", "WOA", "HHO", "GOA", "SA", "BRO", "RUN", "FOX"]
        args.estimators, args.transfer_functions = ["knn", "svm", "rf"], ["vstf_01"]
        args.runs, args.epochs, args.pop_size = 30, 150, 50
        before = snapshot(cache)
        from reporting.core import load_completed_cache, report_guard
        with report_guard(()):
            paths = study.make_read_only_source_paths(args)
            for dataset in ("DataClass", "FeatureEnvy", "GodClass", "LongMethod", "LongParameterList", "SwitchStatements"):
                for classifier in args.estimators:
                    with self.subTest(dataset=dataset, classifier=classifier):
                        payload, reason = study.load_compatible_full_cache_payload(paths, args, dataset, classifier)
                        self.assertIsNone(reason)
                        self.assertEqual(len(payload), 11)
                        self.assertTrue(all(row["CompletedRuns"] == 30 for row in payload.values()))
                        self.assertFalse(any("MACRO" in label for label in payload))
            historical_report_args = deepcopy(args)
            historical_report_args.exp_id = 627
            historical_report_args.optimizers = ["DSADE", *args.optimizers[1:]]
            report = load_completed_cache(historical_report_args)
            self.assertEqual(len(report.indexed), 216)
        self.assertEqual(snapshot(cache), before)

    def test_reporting_reads_new_metadata_and_exact_historical_files(self):
        args, source = self.source()
        from reporting.core import load_completed_cache, report_guard
        with report_guard(()):
            report = load_completed_cache(args)
        self.assertEqual(report.algorithms, ["DE", "JADE"])
        path = aggregate_path(args)
        payload = study.load_cache(str(path))
        payload["DE"]["CacheIdentity"]["optimizer_parameters"]["wf"] = .7
        study.save_cache(str(path), payload)
        with report_guard(()), self.assertRaisesRegex(ValueError, "Scientific cache identity"):
            load_completed_cache(args)
        payload["DE"].pop("CacheIdentity")
        study.save_cache(str(path), payload)
        with report_guard(()), self.assertRaisesRegex(ValueError, "Scientific cache identity"):
            load_completed_cache(args)


if __name__ == "__main__":
    unittest.main()
