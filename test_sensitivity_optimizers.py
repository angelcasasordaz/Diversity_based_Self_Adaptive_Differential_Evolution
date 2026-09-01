import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import main_best as study


def make_args(mode="sensitivity", sensitivity_optimizers=None):
    return argparse.Namespace(
        experiment_mode=mode,
        optimizers=["DE", "JADE"],
        sensitivity_optimizers=list(
            sensitivity_optimizers
            if sensitivity_optimizers is not None
            else study.SENSITIVITY_OPTIMIZERS
        ),
        estimators=["knn"],
        transfer_functions=["vstf_01"],
        runs=2,
        epochs=7,
        pop_size=20,
        test_size=0.2,
        random_state=2,
        seed_base=1234,
        dsade_beta_min=0.2,
        dsade_beta_max=0.8,
        dsade_pcr=0.2,
        dsade_mahal_q=0.68,
        sensitivity_parameter="mahalanobis_q",
        sensitivity_values=[0.50, 0.68],
        sensitivity_weight_pairs=list(study.SENSITIVITY_WEIGHT_PAIRS),
        fitness_alpha=study.DEFAULT_FITNESS_ALPHA,
        fitness_beta=study.DEFAULT_FITNESS_BETA,
        compute_device="cpu",
        gpu_device_id=0,
        gpu_memory_fraction=0.85,
    )


class SensitivityOptimizerTests(unittest.TestCase):
    def test_one_or_both_sensitivity_optimizers_are_selected(self):
        for selected in (["DSA-DE"], ["MaCRO-DE"], ["DSA-DE", "MaCRO-DE"]):
            with self.subTest(selected=selected):
                args = make_args(sensitivity_optimizers=selected)
                study.apply_experiment_mode(args)
                self.assertEqual(args.optimizers, selected)

        args = make_args(sensitivity_optimizers=["DSA-DE", "MaCRO-DE"])
        study.apply_experiment_mode(args)
        labels = study.expected_result_labels(args, "knn", False, False)
        self.assertEqual(len(labels), len(args.optimizers) * len(args.sensitivity_values))
        self.assertEqual(len({label for label, _ in labels}), len(labels))

    def test_each_parameter_uses_the_factory_resolved_constructor_name(self):
        cases = {
            "beta_min": (0.3, "dsade_beta_min", "beta_min", 0.3),
            "beta_max": (0.9, "dsade_beta_max", "beta_max", 0.9),
            "pcr": (0.4, "dsade_pcr", "pcr", 0.4),
            "mahalanobis_q": (0.75, "dsade_mahal_q", "mahalanobis_q", 0.75),
            "pop_size": (30, "pop_size", "pop_size", 30),
            "epochs": (11, "epochs", "epoch", 11),
        }
        for optimizer_name in ("DSA-DE", "MaCRO-DE"):
            for parameter, (value, setting_name, constructor_name, expected) in cases.items():
                with self.subTest(optimizer=optimizer_name, parameter=parameter):
                    args = make_args(sensitivity_optimizers=[optimizer_name])
                    study.apply_experiment_mode(args)
                    args.sensitivity_parameter = parameter
                    variant_args = study.sensitivity_variant_args(args, value)
                    constructor_kwargs = study.optimizer_constructor_kwargs(
                        optimizer_name,
                        variant_args,
                    )
                    self.assertEqual(getattr(variant_args, setting_name), expected)
                    self.assertEqual(constructor_kwargs[constructor_name], expected)

        for optimizer_name in ("DSA-DE", "MaCRO-DE"):
            args = make_args(sensitivity_optimizers=[optimizer_name])
            study.apply_experiment_mode(args)
            args.sensitivity_parameter = "beta_min"
            optimizer = study.build_optimizer(
                optimizer_name,
                study.sensitivity_variant_args(args, 0.35),
            )
            self.assertAlmostEqual(optimizer.beta_min, 0.35)

    def test_ofat_variants_do_not_mutate_nominal_configuration(self):
        parameter_settings = {
            "beta_min": "dsade_beta_min",
            "beta_max": "dsade_beta_max",
            "pcr": "dsade_pcr",
            "mahalanobis_q": "dsade_mahal_q",
            "pop_size": "pop_size",
            "epochs": "epochs",
        }
        nominal_args = make_args(sensitivity_optimizers=["DSA-DE", "MaCRO-DE"])
        study.apply_experiment_mode(nominal_args)
        nominal = {
            setting: getattr(nominal_args, setting)
            for setting in parameter_settings.values()
        }
        for parameter, selected_setting in parameter_settings.items():
            with self.subTest(parameter=parameter):
                nominal_args.sensitivity_parameter = parameter
                value = 0.4 if parameter not in {"pop_size", "epochs"} else 40
                variant_args = study.sensitivity_variant_args(nominal_args, value)
                for setting, expected in nominal.items():
                    if setting != selected_setting:
                        self.assertEqual(getattr(variant_args, setting), expected)
                self.assertEqual(
                    {setting: getattr(nominal_args, setting) for setting in nominal},
                    nominal,
                )

    def test_cache_and_checkpoint_identity_separate_optimizers_and_values(self):
        dsade_args = make_args(sensitivity_optimizers=["DSA-DE"])
        macro_args = make_args(sensitivity_optimizers=["MaCRO-DE"])
        study.apply_experiment_mode(dsade_args)
        study.apply_experiment_mode(macro_args)
        self.assertNotEqual(
            study.build_cache_signature(dsade_args),
            study.build_cache_signature(macro_args),
        )

        dsade_variant = study.sensitivity_variant_args(dsade_args, 0.50)
        macro_variant = study.sensitivity_variant_args(macro_args, 0.50)
        dsade_metadata = study.sensitivity_checkpoint_metadata(
            dsade_variant, "DSA-DE", 0.50
        )
        macro_metadata = study.sensitivity_checkpoint_metadata(
            macro_variant, "MaCRO-DE", 0.50
        )
        self.assertIsNotNone(dsade_metadata)
        self.assertIsNotNone(macro_metadata)
        self.assertNotEqual(dsade_metadata, macro_metadata)

        changed_value = study.sensitivity_checkpoint_metadata(
            study.sensitivity_variant_args(dsade_args, 0.80), "DSA-DE", 0.80
        )
        self.assertNotEqual(dsade_metadata, changed_value)

        changed_science_args = study.sensitivity_variant_args(dsade_args, 0.50)
        changed_science_args.dsade_pcr = 0.35
        changed_science = study.sensitivity_checkpoint_metadata(
            changed_science_args, "DSA-DE", 0.50
        )
        self.assertNotEqual(dsade_metadata, changed_science)
        self.assertFalse(
            study.sensitivity_checkpoint_metadata_matches(
                dsade_metadata,
                changed_science,
            )
        )

    def test_other_modes_ignore_sensitivity_optimizer_selector(self):
        full = make_args("full", ["MaCRO-DE"])
        full_signature = study.build_cache_signature(full)
        study.apply_experiment_mode(full)
        self.assertEqual(full.optimizers, ["DE", "JADE"])
        self.assertEqual(study.build_cache_signature(full), full_signature)

        ablation = make_args("ablation", ["MaCRO-DE"])
        study.apply_experiment_mode(ablation)
        self.assertEqual(ablation.optimizers, study.ABLATION_OPTIMIZERS)

        weights = make_args("sensitivity_weights", ["MaCRO-DE"])
        study.apply_experiment_mode(weights)
        self.assertEqual(weights.optimizers, ["DSA-DE"])
        self.assertEqual(
            study.experiment_variants(weights),
            study.SENSITIVITY_WEIGHT_PAIRS,
        )

    def test_multi_optimizer_statistics_and_plot_keep_optimizer_dimension(self):
        args = make_args(sensitivity_optimizers=["DSA-DE", "MaCRO-DE"])
        study.apply_experiment_mode(args)
        args.optimizers = study.resolve_optimizers(args)
        results = {"Synthetic": {}}
        for optimizer_idx, optimizer_name in enumerate(args.optimizers):
            for value_idx, value in enumerate(args.sensitivity_values):
                variant_args = study.sensitivity_variant_args(args, value)
                label = study.build_alg_label(
                    optimizer_name,
                    "vstf_01",
                    "knn",
                    False,
                    False,
                    study.sensitivity_label_suffix(args, value),
                )
                results["Synthetic"][label] = study.build_label_payload(
                    "knn",
                    [70.0 + optimizer_idx + value_idx],
                    [0.70],
                    [0.71],
                    [0.72 + optimizer_idx / 100.0],
                    [0.25],
                    [3 + value_idx],
                    [0.01],
                    [np.full(variant_args.epochs, 0.25)],
                    variant_args.epochs,
                    scientific_metadata=study.sensitivity_checkpoint_metadata(
                        variant_args,
                        optimizer_name,
                        value,
                    ),
                )

        summary = study.generate_summary_dataframe(results, args)
        self.assertEqual(
            len(summary[["Optimizer", "SensitivityValue"]].drop_duplicates()),
            len(args.optimizers) * len(args.sensitivity_values),
        )
        with tempfile.TemporaryDirectory() as tmp:
            chart = study.generate_sensitivity_main_figure(summary, tmp, args)
            self.assertIsNotNone(chart)
            self.assertTrue((Path(tmp) / chart).exists())

            workbook = Path(tmp) / "sensitivity_stats.xlsx"
            study.export_statistical_excel(
                results,
                ["Synthetic"],
                args.optimizers,
                args,
                str(workbook),
            )
            accuracy = pd.read_excel(workbook, sheet_name="Accuracy", index_col=[0, 1])
            expected_groups = {
                f"{optimizer} | {args.sensitivity_parameter}={value:g}"
                for optimizer in args.optimizers
                for value in args.sensitivity_values
            }
            self.assertEqual(set(accuracy.index.get_level_values(0)), expected_groups)


if __name__ == "__main__":
    unittest.main()
