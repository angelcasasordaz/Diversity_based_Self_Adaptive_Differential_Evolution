import argparse
import tempfile
import unittest
from pathlib import Path

import main_best as study


def make_args(source="codesmell", mode="full", datasets=None, dataset_dir="Original"):
    return argparse.Namespace(
        dataset_source=source,
        experiment_mode=mode,
        datasets=datasets,
        dataset_dir=dataset_dir,
        dataset_suite="test14",
    )


class ModeDatasetSelectionTests(unittest.TestCase):
    def test_full_uses_existing_source_specific_lists(self):
        codesmell = study.configured_dataset_names(make_args("codesmell", "full"))
        mafese = study.configured_dataset_names(make_args("mafese", "full"))
        self.assertEqual(codesmell, study.CODE_SMELL_DATASETS)
        self.assertEqual(mafese, study.TEST_DATASETS_CLASSIFICATION_14)

    def test_each_non_full_mode_uses_only_its_dedicated_list(self):
        expected = {
            "ablation": study.ABLATION_DATASETS,
            "sensitivity": study.SENSITIVITY_DATASETS,
            "sensitivity_weights": study.SENSITIVITY_WEIGHTS_DATASETS,
        }
        selected = {
            mode: study.configured_dataset_names(make_args("codesmell", mode))
            for mode in expected
        }
        self.assertEqual(selected, expected)
        for mode, names in selected.items():
            for other_mode, other_names in selected.items():
                if mode != other_mode and expected[mode] != expected[other_mode]:
                    self.assertIsNot(names, other_names)

    def test_selected_names_are_interpreted_by_current_source(self):
        for source in ("codesmell", "mafese"):
            args = make_args(source, "sensitivity")
            self.assertEqual(
                study.configured_dataset_names(args),
                study.SENSITIVITY_DATASETS,
            )

    def test_codesmell_resolver_uses_each_mode_selection(self):
        expected = {
            "full": study.CODE_SMELL_DATASETS,
            "ablation": study.ABLATION_DATASETS,
            "sensitivity": study.SENSITIVITY_DATASETS,
            "sensitivity_weights": study.SENSITIVITY_WEIGHTS_DATASETS,
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name in set().union(*map(set, expected.values())):
                Path(tmp, f"{name}.csv").touch()
            for mode, names in expected.items():
                with self.subTest(mode=mode):
                    args = make_args("codesmell", mode, dataset_dir=tmp)
                    specs = study.resolve_dataset_specs(args)
                    self.assertEqual([spec.name for spec in specs], names)

    def test_source_and_dataset_validation_are_preserved(self):
        with self.assertRaisesRegex(ValueError, "Unsupported dataset source"):
            study.resolve_dataset_specs(make_args("invalid", "full"))

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "DataClass.csv").touch()
            args = make_args(
                "codesmell",
                "full",
                datasets=["MissingDataset"],
                dataset_dir=tmp,
            )
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                study.resolve_dataset_specs(args)

        args = make_args("mafese", "sensitivity")
        original = study.SENSITIVITY_DATASETS
        try:
            study.SENSITIVITY_DATASETS = ["MissingDataset"]
            with self.assertRaisesRegex(ValueError, "Unsupported MAFESE datasets"):
                study.resolve_dataset_specs(args)
        finally:
            study.SENSITIVITY_DATASETS = original

    def test_codesmell_command_line_override_is_preserved(self):
        args = make_args("codesmell", "ablation", datasets=["FeatureEnvy.csv"])
        self.assertEqual(study.configured_dataset_names(args), ["FeatureEnvy.csv"])

    def test_mafese_command_line_dataset_override_remains_rejected(self):
        args = make_args("mafese", "full", datasets=["Wine"])
        with self.assertRaisesRegex(ValueError, "only supported"):
            study.resolve_dataset_specs(args)


if __name__ == "__main__":
    unittest.main()
