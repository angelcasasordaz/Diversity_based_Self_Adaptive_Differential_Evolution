"""Exercise CLI routing without allowing experiment execution or output writes."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from io import StringIO
import sys
import unittest
from unittest.mock import patch

import main_best as m


MODES = ("full", "ablation", "sensitivity", "sensitivity_weights", "transfer_functions")


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m, "FULL_REPLICA_REPORT_ONLY", True))
        self.stack.enter_context(redirect_stdout(StringIO()))
        for name in ("_run_single", "execute_pending_runs", "load_dataset", "save_cache"):
            self.stack.enter_context(patch.object(m, name, side_effect=AssertionError("Scientific execution forbidden")))
        self.report = self.stack.enter_context(patch("reporting.core.run_report"))
        self.run_mode = self.stack.enter_context(patch.object(m, "run_experiment_mode"))
        self.listing = self.stack.enter_context(patch.object(m, "print_available_optimizers"))
        self.backend = self.stack.enter_context(patch.object(m, "configure_compute_backend"))
        for name in ("validate_gpu_random_forest_backend", "report_gpu_acceptance", "validate_comparison_backend"):
            self.stack.enter_context(patch.object(m, name))

    def invoke(self, *argv):
        with patch.object(sys, "argv", ["main_best.py", *argv]):
            m.main()

    def test_plain_ide_run_is_report_only(self):
        self.invoke()
        self.report.assert_called_once()
        self.run_mode.assert_not_called()
        self.backend.assert_not_called()

    def test_explicit_report_aliases_are_safe(self):
        for flag in ("--full-replica-report-only", "--full-rep1-report-only"):
            with self.subTest(flag=flag):
                self.report.reset_mock()
                self.invoke(flag)
                self.report.assert_called_once()
        self.run_mode.assert_not_called()
        self.backend.assert_not_called()

    def test_list_optimizers_precedes_report_default_and_explicit_report(self):
        for extra in ([], ["--full-rep1-report-only"], ["--experiment-mode", "full"]):
            with self.subTest(extra=extra):
                self.listing.reset_mock()
                self.invoke("--list-optimizers", *extra)
                self.listing.assert_called_once()
        self.report.assert_not_called()
        self.run_mode.assert_not_called()
        self.backend.assert_not_called()

    def test_every_explicit_mode_reaches_experiment_dispatch(self):
        for flag in ("--experiment-mode", "--experiment-modes"):
            for mode in MODES:
                with self.subTest(flag=flag, mode=mode):
                    self.run_mode.reset_mock()
                    self.invoke(flag, mode)
                    self.assertTrue(self.run_mode.called)
                    self.assertEqual({c.args[0].experiment_mode for c in self.run_mode.call_args_list}, {mode})
                    self.assertTrue(all(not c.args[0].full_replica_report_only for c in self.run_mode.call_args_list))
        self.report.assert_not_called()

    def test_multiple_modes_and_equals_syntax(self):
        self.invoke("--experiment-modes", *MODES)
        self.assertEqual(list(dict.fromkeys(c.args[0].experiment_mode for c in self.run_mode.call_args_list)), list(MODES))
        self.run_mode.reset_mock()
        self.invoke("--experiment-mode=ablation")
        self.assertEqual(self.run_mode.call_args.args[0].experiment_mode, "ablation")
        self.report.assert_not_called()

    def test_report_mode_and_selected_exp_are_explicit(self):
        self.invoke("--report-only", "--experiment-modes", "ablation", "--exp-id", "913")
        self.report.assert_called_once()
        self.assertEqual(self.report.call_args.args[0].exp_id, 913)
        self.assertEqual(self.report.call_args.args[0].experiment_modes, ["ablation"])
        self.run_mode.assert_not_called()
        self.backend.assert_not_called()

    def test_help_never_dispatches(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke("--help")
        self.assertEqual(error.exception.code, 0)
        self.report.assert_not_called()
        self.run_mode.assert_not_called()
        self.backend.assert_not_called()

    def test_removed_experiment_specific_option_is_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.invoke('--report-preset', 'exp627', '--experiment-mode', 'full')
        self.run_mode.assert_not_called()
        self.backend.assert_not_called()


class DirectModeDispatchTests(unittest.TestCase):
    def test_global_default_cannot_intercept_explicit_mode_arguments(self):
        class StopBeforeExecution(Exception):
            pass
        for mode in MODES:
            with self.subTest(mode=mode), patch.object(m, "FULL_REPLICA_REPORT_ONLY", True), \
                    patch.object(sys, "argv", ["main_best.py", "--experiment-mode", mode]):
                args = m.parse_args()
                with patch.object(m, "apply_experiment_mode", side_effect=StopBeforeExecution) as apply, \
                        patch("reporting.core.run_report") as report:
                    with self.assertRaises(StopBeforeExecution):
                        m.run_experiment_mode(args)
                    apply.assert_called_once_with(args)
                    report.assert_not_called()

    def test_direct_report_only_call_stays_safe(self):
        with patch.object(sys, "argv", ["main_best.py", "--full-rep1-report-only"]):
            args = m.parse_args()
        with patch.object(m, "apply_experiment_mode") as apply, \
                patch("reporting.core.run_report") as report:
            m.run_experiment_mode(args)
        report.assert_called_once_with(args)
        apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
