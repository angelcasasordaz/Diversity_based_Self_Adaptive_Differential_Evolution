"""RF availability diagnostics only; no GPU, model fitting, or experiment."""
import argparse
from contextlib import redirect_stdout
from io import StringIO
import sys
import unittest
from unittest.mock import patch

import main_best as study


class RFImportDiagnosticsTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(compute_device="gpu", rf_backend_policy="auto", rf_cpu_fallback=True)

    def test_auto_fallback_logs_original_import_error_and_interpreter(self):
        cause = ModuleNotFoundError("No module named 'cuml'")
        failure = study.CuMLUnavailableError("RF GPU execution requires cuML.")
        failure.__cause__ = cause
        output, args = StringIO(), self.args()
        with patch.object(study, "gpu_random_forest_class", side_effect=failure) as probe, redirect_stdout(output):
            self.assertEqual(study.resolve_rf_backend(args), "sklearn")
            self.assertEqual(study.resolve_rf_backend(args), "sklearn")
        probe.assert_called_once()
        self.assertIn(f"interpreter={sys.executable}", output.getvalue())
        self.assertIn("cause=ModuleNotFoundError: No module named 'cuml'", output.getvalue())
        self.assertIn("cuML unavailable; using sklearn RF on CPU while optimizer backend remains GPU.", output.getvalue())

    def test_successful_auto_probe_does_not_log_failure_or_use_sklearn(self):
        output = StringIO()
        with patch.object(study, "gpu_random_forest_class", return_value=object()), redirect_stdout(output):
            self.assertEqual(study.resolve_rf_backend(self.args()), "cuml")
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
