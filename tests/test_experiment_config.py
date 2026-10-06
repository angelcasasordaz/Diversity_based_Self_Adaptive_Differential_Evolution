"""Manifest provenance, read-only checks, and cache-only reporting defaults."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import json
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import main_best as main
from reporting import core, figures, experiment_config as configs
from tests.test_generic_reporting import arguments, caches


class ExperimentConfigTests(unittest.TestCase):
    def fixture(self, root):
        args = caches(arguments(root))
        with core.report_guard(()):
            report = core.load_cached_figure_report(args)
        config = configs.config_from_report(report)
        configs.write_manifest(args, config)
        return args, config

    def test_manifest_contains_actual_settings_signatures_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config = self.fixture(root)
            stored = json.loads(configs.manifest_path(args).read_text())
            self.assertTrue(all(field in stored for field in configs.FIELDS))
            self.assertEqual(stored['datasets'], ['First', 'Second'])
            self.assertEqual(stored['optimizers'], ['DE', 'PSO', 'JADE'])
            self.assertEqual(len(stored['cache_signatures']), 4)
            self.assertEqual(stored['optimizer_parameters']['DE']['pop_size'], args.pop_size)
            for entry in stored['cache_signatures']:
                path = configs.manifest_path(args).parent / entry['path']
                self.assertIn(entry['signature'], path.name)
                self.assertEqual(entry['sha256'], core.sha256(path))

    def test_report_prefers_manifest_defaults_and_recorded_files_over_current_hash_and_other_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config = self.fixture(root)
            path = next(root.rglob('*_results.pkl'))
            payload = pickle.loads(path.read_bytes())
            row = next(iter(payload.values()))
            row['AccRuns'][0] += 1
            row['AccMean'] = sum(row['AccRuns']) / len(row['AccRuns'])
            path.with_name(path.name.replace('_results.pkl', '_other_results.pkl')).write_bytes(pickle.dumps(payload))
            changed = deepcopy(args)
            changed.runs, changed.epochs, changed.pop_size = 999, 999, 999
            changed.optimizers = ['BRO']
            changed.report_explicit_options = frozenset()
            self.assertNotEqual(main.build_cache_signature(changed), main.build_cache_signature(args))
            with core.report_guard(()):
                report = core.load_cached_figure_report(changed)
            self.assertEqual(report.args.runs, 3)
            self.assertEqual(report.args.pop_size, args.pop_size)
            self.assertEqual(set(report.algorithms), {'DE', 'PSO', 'JADE'})
            self.assertEqual(len(report.sources), 4)

    def test_explicit_science_mismatch_and_changed_cache_fail_without_generating(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = self.fixture(Path(directory))
            changed = deepcopy(args)
            changed.runs += 1
            with core.report_guard(()), self.assertRaisesRegex(ValueError, 'Manifest MISMATCH runs'):
                core.load_cached_figure_report(changed)
            path = configs.manifest_path(args).parent / config['cache_signatures'][0]['path']
            path.write_bytes(b'changed cache')
            with core.report_guard(()), self.assertRaisesRegex(ValueError, 'cache SHA256'):
                core.load_cached_figure_report(args)

    def test_manifest_allows_explicit_subset_of_recorded_optimizers(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = self.fixture(Path(directory))
            args.optimizers = ['DE']
            with core.report_guard(()):
                report = core.load_cached_figure_report(args)
            self.assertEqual(report.algorithms, ['DE'])
            self.assertEqual(len(report.indexed), 4)

    def test_check_is_read_only_even_with_report_only_and_ide_report_default(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            args, config = self.fixture(root)
            before = {p: (core.sha256(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            for name in ('configure_compute_backend', 'load_dataset', 'execute_pending_runs', 'save_cache'):
                stack.enter_context(patch.object(main, name, side_effect=AssertionError(name)))
            stack.enter_context(patch.object(figures, 'generate', side_effect=AssertionError('figures forbidden')))
            stack.enter_context(patch.object(core, 'run_report', side_effect=AssertionError('reports forbidden')))
            with patch.object(sys, 'argv', ['main_best.py', '--check-exp-config', '--report-only',
                '--exp-id', '913', '--experiment-mode', 'full', '--output-root', directory, '--dataset-source', 'codesmell',
                '--datasets', 'First', 'Second', '--optimizers', 'DE', 'PSO', 'JADE',
                '--estimators', 'knn', 'rf', '--runs', '3', '--epochs', '3']):
                result = main.main()
            self.assertTrue(result['match'])
            self.assertEqual(result['optimization_calls'], 0)
            self.assertEqual({p: (core.sha256(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}, before)
            self.assertFalse((root / 'Figures').exists())

    def test_check_mismatch_reports_fields_and_returns_cli_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args, config = self.fixture(Path(directory))
            output = StringIO()
            with redirect_stdout(output), patch.object(sys, 'argv', ['main_best.py', '--check-exp-config',
                '--exp-id', '913', '--experiment-mode', 'full', '--output-root', directory, '--dataset-source', 'codesmell',
                '--datasets', 'First', 'Second', '--optimizers', 'DE', 'PSO', 'JADE',
                '--estimators', 'knn', 'rf', '--runs', '3', '--epochs', '3', '--pop-size', '99']), \
                    self.assertRaises(SystemExit) as error:
                main.main()
            self.assertEqual(error.exception.code, 1)
            self.assertIn('MISMATCH pop_size', output.getvalue())
            self.assertIn('MATCH runs', output.getvalue())

    def test_check_without_manifest_reads_caches_without_creating_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = caches(arguments(root))
            self.assertTrue(configs.check_exp_config(args)['match'])
            self.assertFalse(configs.manifest_path(args).exists())
            self.assertFalse((root / 'Figures').exists())

    def test_reporting_creates_only_configuration_metadata_in_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = caches(arguments(root))
            args.report_only = True
            before = {p: core.sha256(p) for p in root.rglob('*.pkl')}
            with patch.object(figures, 'generate', return_value=[]):
                core.run_report(args)
            self.assertTrue(configs.manifest_path(args).exists())
            self.assertEqual({p: core.sha256(p) for p in root.rglob('*.pkl')}, before)
            original = configs.manifest_path(args).read_bytes()
            with patch.object(figures, 'generate', return_value=[]):
                core.run_report(args)
            self.assertEqual(configs.manifest_path(args).read_bytes(), original)

    def test_completed_report_refreshes_manifest_when_only_progress_paths_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config = self.fixture(root)
            for path in list(root.rglob('*_results.pkl')):
                path.rename(path.with_name(path.name.replace('_results.pkl', '_progress.pkl')))
            args.report_only = True
            with patch.object(figures, 'generate', return_value=[]):
                core.run_report(args)
            updated = configs.read_manifest(args)
            self.assertTrue(all(entry['path'].endswith('_progress.pkl') for entry in updated['cache_signatures']))
            self.assertTrue(configs.check_exp_config(args)['match'])


if __name__ == '__main__':
    unittest.main()
