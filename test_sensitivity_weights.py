import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mafese import Data

import main_best as study


def make_args(mode="full"):
    return argparse.Namespace(
        experiment_mode=mode,
        optimizers=["DE", "JADE"],
        estimators=["knn", "svm", "rf"],
        transfer_functions=["vstf_01"],
        runs=3,
        epochs=5,
        pop_size=10,
        test_size=0.2,
        random_state=2,
        seed_base=1234,
        dsade_beta_min=0.2,
        dsade_beta_max=0.8,
        dsade_pcr=0.2,
        dsade_mahal_q=0.68,
        sensitivity_parameter="mahalanobis_q",
        sensitivity_values=[0.50, 0.68, 0.80, 0.90],
        sensitivity_weight_pairs=list(study.SENSITIVITY_WEIGHT_PAIRS),
        fitness_alpha=study.DEFAULT_FITNESS_ALPHA,
        fitness_beta=study.DEFAULT_FITNESS_BETA,
    )


def legacy_cache_signature(args):
    payload = {
        "experiment_mode": str(args.experiment_mode),
        "optimizers": [study.resolve_optimizer_name(name) for name in args.optimizers],
        "transfer_functions": list(args.transfer_functions),
        "runs": int(args.runs),
        "epochs": int(args.epochs),
        "pop_size": int(args.pop_size),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "seed_base": int(args.seed_base),
        "obj_name": "AS",
        "fitness_mode": "minimize_metric_loss_plus_feature_ratio_v1",
        "dsade_beta_min": float(args.dsade_beta_min),
        "dsade_beta_max": float(args.dsade_beta_max),
        "dsade_pcr": float(args.dsade_pcr),
        "dsade_mahal_q": float(args.dsade_mahal_q),
        "sensitivity_parameter": str(args.sensitivity_parameter),
        "sensitivity_values": [float(value) for value in args.sensitivity_values],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]


class SensitivityWeightsTests(unittest.TestCase):
    def test_command_line_accepts_new_mode_and_pairs(self):
        argv = [
            "main_best.py",
            "--experiment-mode",
            "sensitivity_weights",
            "--sensitivity-weight-pairs",
            "0.70,0.30",
            "0.95,0.05",
        ]
        with patch.object(sys, "argv", argv):
            args = study.parse_args()
        self.assertEqual(args.experiment_modes, ["sensitivity_weights"])
        self.assertEqual(args.sensitivity_weight_pairs, [(0.70, 0.30), (0.95, 0.05)])

    def test_existing_modes_keep_configuration_and_cache_identity(self):
        expected_optimizers = {
            "full": ["DE", "JADE"],
            "ablation": list(study.ABLATION_OPTIMIZERS),
            "sensitivity": ["DSA-DE"],
        }
        for mode, optimizers in expected_optimizers.items():
            with self.subTest(mode=mode):
                args = make_args(mode)
                study.apply_experiment_mode(args)
                self.assertEqual(args.optimizers, optimizers)
                self.assertEqual((args.fitness_alpha, args.fitness_beta), (0.90, 0.10))
                self.assertEqual(study.build_cache_signature(args), legacy_cache_signature(args))

    def test_weight_mode_uses_only_proposed_optimizer_and_exact_pairs(self):
        args = make_args("sensitivity_weights")
        study.apply_experiment_mode(args)
        self.assertEqual(args.optimizers, ["DSA-DE"])
        self.assertEqual(study.experiment_variants(args), study.SENSITIVITY_WEIGHT_PAIRS)

    def test_weight_validation(self):
        self.assertEqual(
            study.validate_sensitivity_weight_pairs(study.SENSITIVITY_WEIGHT_PAIRS),
            study.SENSITIVITY_WEIGHT_PAIRS,
        )
        for invalid in [[(0.0, 1.0)], [(0.9, 0.0)], [(0.8, 0.3)]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                study.validate_sensitivity_weight_pairs(invalid)

    def test_fit_receives_active_weights_without_running_an_optimizer(self):
        captured = {}

        class FakeHistory:
            list_global_best_fit = [0.25]

        class FakeOptimizer:
            history = FakeHistory()

        class FakeSelector:
            def __init__(self, **kwargs):
                self.estimator = kwargs["estimator"]
                self.optimizer = FakeOptimizer()

            def fit(self, X, y, test_size=0.2, fit_weights=(0.9, 0.1), transfer_func=None,
                    fs_problem=None, verbose=True):
                captured["fit_weights"] = fit_weights

            def transform(self, X):
                return X[:, :1]

            def evaluate(self, **kwargs):
                return {"AS_test": 0.75, "PS_test": 0.7, "RS_test": 0.8, "F1S_test": 0.74}

        args = make_args("sensitivity_weights")
        study.apply_experiment_mode(args)
        study.apply_sensitivity_weight_pair(args, (0.95, 0.05))
        data = Data()
        data.set_train_test(
            X_train=np.array([[0.0, 1.0], [1.0, 0.0]]),
            y_train=np.array([0, 1]),
            X_test=np.array([[0.1, 0.9], [0.9, 0.1]]),
            y_test=np.array([0, 1]),
        )
        with patch.object(study, "MhaSelector", FakeSelector), patch.object(
            study, "build_optimizer", return_value=FakeOptimizer()
        ):
            study.run_single(data, "knn", "DSA-DE", "vstf_01", args, seed=1234)
        self.assertEqual(captured["fit_weights"], (0.95, 0.05))

    def test_cache_identity_and_cross_experiment_metadata(self):
        args = make_args("sensitivity_weights")
        study.apply_experiment_mode(args)
        signature = study.build_cache_signature(args)

        changed = make_args("sensitivity_weights")
        changed.sensitivity_weight_pairs = [(0.95, 0.05)]
        study.apply_experiment_mode(changed)
        self.assertNotEqual(signature, study.build_cache_signature(changed))

        normal = make_args("sensitivity")
        study.apply_experiment_mode(normal)
        self.assertNotEqual(signature, study.build_cache_signature(normal))

        study.apply_sensitivity_weight_pair(args, (0.95, 0.05))
        metadata = study.weight_checkpoint_metadata(args)
        row = study.build_label_payload(
            "knn", [75.0], [0.7], [0.8], [0.74], [0.25], [1], [0.01],
            [np.array([0.25])], 1, scientific_metadata=metadata,
        )
        source = {"label": row}
        destinations = {key: [] for key in (
            "AccRuns", "PSRuns", "RSRuns", "F1Runs", "FitRuns", "FeatRuns",
            "TimeRuns", "CurvesAll",
        )}
        imported, reason = study.import_source_label_runs(
            source, "label", "label", "knn", 1, destinations,
            expected_metadata=metadata,
        )
        self.assertEqual((imported, reason), (1, None))

        wrong_metadata = dict(metadata, FitnessAlpha=0.90, FitnessBeta=0.10)
        empty_destinations = {key: [] for key in destinations}
        imported, reason = study.import_source_label_runs(
            source, "label", "label", "knn", 1, empty_destinations,
            expected_metadata=wrong_metadata,
        )
        self.assertEqual(imported, 0)
        self.assertIn("metadata does not match", reason)

        with tempfile.TemporaryDirectory() as tmp:
            args.output_root = tmp
            args.exp_id = 702
            args.reuse_cache_from_exp_id = 701
            source_paths = study.make_read_only_source_paths(args)
            Path(source_paths.cache_dir).mkdir(parents=True)
            filename = f"{source_paths.exp_tag}_LongMethod_knn_{signature}_results.pkl"
            study.save_cache(str(Path(source_paths.cache_dir) / filename), source)
            loaded, reason = study.load_best_source_cache_payload(
                source_paths, "LongMethod", "knn", signature
            )
            self.assertIsNone(reason)
            self.assertEqual(loaded["label"]["FitnessAlpha"], 0.95)
            self.assertIn("sensitivity_weights", source_paths.cache_dir)

            normal.output_root = tmp
            normal.exp_id = 702
            normal.reuse_cache_from_exp_id = 701
            normal_source_paths = study.make_read_only_source_paths(normal)
            self.assertIn("sensitivity", normal_source_paths.cache_dir)
            self.assertNotEqual(source_paths.cache_dir, normal_source_paths.cache_dir)

    def test_selected_dataset_and_weight_figure(self):
        args = make_args("sensitivity_weights")
        study.apply_experiment_mode(args)
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "datasets"
            dataset_dir.mkdir()
            (dataset_dir / "LongMethod.csv").touch()
            (dataset_dir / "GodClass.csv").touch()
            selection_args = argparse.Namespace(
                dataset_dir=str(dataset_dir),
                datasets=["LongMethod"],
            )
            specs = study.resolve_codesmell_dataset_specs(selection_args)
            self.assertEqual([spec.name for spec in specs], ["LongMethod"])

            results = {"LongMethod": {}}
            for estimator in args.estimators:
                for idx, pair in enumerate(args.sensitivity_weight_pairs):
                    study.apply_sensitivity_weight_pair(args, pair)
                    suffix = study.sensitivity_weight_label_suffix(pair)
                    label = study.build_alg_label(
                        "DSA-DE", "vstf_01", estimator, False, True, suffix
                    )
                    results["LongMethod"][label] = study.build_label_payload(
                        estimator,
                        [80.0 + idx], [0.7], [0.8], [0.75 + idx / 100.0],
                        [0.2], [3 + idx], [0.01], [np.array([0.2])], 1,
                        scientific_metadata=study.weight_checkpoint_metadata(args),
                    )
            summary = study.generate_summary_dataframe(results, args)
            filename = study.generate_weight_sensitivity_main_figure(
                "LongMethod", summary, tmp, args
            )
            self.assertEqual(filename, "SensitivityWeights_AlphaBeta_LongMethod.png")
            self.assertTrue((Path(tmp) / filename).exists())
            self.assertEqual(study.experiment_output_prefix(args), "SensitivityWeights_")


if __name__ == "__main__":
    unittest.main()
