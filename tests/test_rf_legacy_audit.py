"""Read-only diagnostics for the observed EXP629 DataClass/RF checkpoint."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import main_best as study


def metadata_differences(actual, expected, prefix=""):
    differences = {}
    for key, value in expected.items():
        name = f"{prefix}.{key}" if prefix else key
        found = actual.get(key, "<missing>") if isinstance(actual, dict) else "<missing>"
        if isinstance(value, dict):
            differences.update(metadata_differences(found, value, name))
        elif found != value:
            differences[name] = {"actual": found, "expected": value}
    return differences


class ObservedRFCheckpointAuditTests(unittest.TestCase):
    def test_observed_shared_prefix_is_recoverable_by_its_full_legacy_digest(self):
        root = Path(study.__file__).resolve().parent
        directory = root / "Results/EXP629/full/cache"
        if not (directory / "EXP629_DataClass_rf_bb1ada81b7_results.pkl").exists():
            self.skipTest("Observed EXP629 checkpoint not available")
        with patch.object(sys, "argv", ["main_best.py", "--experiment-mode", "full"]):
            args = study.parse_args()
        from reporting.core import report_guard
        # Isolate the historical files: subsequent local checkpoints can contain
        # additional completed runs and must not alter this legacy audit fixture.
        with tempfile.TemporaryDirectory() as temporary:
            for path in directory.glob("EXP629_DataClass_rf_bb1ada81b7_*.pkl"):
                shutil.copyfile(path, Path(temporary) / path.name)
            paths = study.Paths("EXP629", "full", "", temporary, temporary)
            with report_guard(()):
                payload, reason = study.load_compatible_full_cache_payload(paths, args, "DataClass", "rf")
        self.assertIsNone(reason)
        self.assertEqual(payload["MACRO-DE-T_RF"]["CompletedRuns"], 30)
        self.assertEqual(payload["HHO_RF"]["CompletedRuns"], 20)
        for path in directory.glob("EXP629_DataClass_rf_bb1ada81b7_*.pkl"):
            row = study.load_cache(str(path))["MACRO-DE-T_RF"]
            recovered = study.legacy_full_cache_args(path, study.load_cache(str(path)), args, "DataClass", "rf")
            self.assertIsNotNone(recovered)
            self.assertEqual(study.build_legacy_cache_signature(recovered), "bb1ada81b7")
            for key in ("AccRuns", "PSRuns", "RSRuns", "F1Runs", "FitRuns", "FeatRuns", "TimeRuns"):
                self.assertTrue((payload["MACRO-DE-T_RF"][key] == row[key]).all())

    def test_observed_checkpoint_metadata_and_run_arrays(self):
        root = Path(study.__file__).resolve().parent
        files = sorted((root / "Results/EXP629/full/cache").glob(
            "EXP629_DataClass_rf_bb1ada81b7_*.pkl"))
        if not files:
            self.skipTest("Observed EXP629 checkpoint not available")
        with patch.object(sys, "argv", ["main_best.py", "--experiment-mode", "full"]):
            args = study.parse_args()
        expected = study.full_checkpoint_metadata(args, "MaCRO-DE-t", "DataClass", "rf", "vstf_01")
        for path in files:
            before = path.read_bytes(), path.stat().st_mtime_ns
            row = study.load_cache(str(path))["MACRO-DE-T_RF"]
            differences = metadata_differences(row, expected)
            print("RF AUDIT", path.name, json.dumps(differences, sort_keys=True))
            self.assertEqual(row["Estimator"], "rf")
            # This regression fixture predates complete FULL metadata.
            self.assertNotIn("CacheIdentity", row)
            self.assertNotIn("ExperimentMode", row)
            self.assertTrue(all(item["actual"] == "<missing>" for item in differences.values()))
            values, reason = study.validate_source_label_runs(row, "rf", args.runs)
            self.assertIsNone(reason)
            self.assertEqual(len(values["AccRuns"]), 30)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)


if __name__ == "__main__":
    unittest.main()
