"""Cache-only CLI reports change figure paths/text without scientific writes."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import pickle
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import main_best as main
from reporting import core, figures
from tests.test_generic_reporting import arguments, caches


class ReportFigureLanguageTests(unittest.TestCase):
    def test_cli_languages_write_separate_figures_and_preserve_all_results(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            caches(arguments(root))
            historical = root / 'Results/EXP913/full/summary.csv'
            historical.write_text('unchanged results')
            before = {p: core.sha256(p) for p in (root / 'Results').rglob('*') if p.is_file()}
            for name in ('_run_single', 'execute_pending_runs', 'build_optimizer',
                         'load_dataset', 'save_cache', 'configure_compute_backend', 'export_global_excel'):
                forbidden = stack.enter_context(patch.object(main, name, side_effect=AssertionError(name)))
                self.addCleanup(forbidden.assert_not_called)
            observed = []

            def one_heatmap(report, destination):
                original = deepcopy(report.indexed)
                metric = next(m for m in report.metrics if m.run_key == 'AccRuns')
                fig = figures.heatmap_figure(report, 'knn', metric)
                from figure_text import localize_figure
                localize_figure(fig, report.args.figure_language, protected=report.algorithms)
                observed.append(fig.axes[0].get_title())
                figures.save_png(fig, Path(destination) / '06_heatmap_accuracy_knn.png')
                np.testing.assert_equal(report.indexed, original)
                return []

            stack.enter_context(patch.object(figures, 'generate', side_effect=one_heatmap))
            for language, folder in (('en', 'full'), ('es', 'full_esp')):
                with patch.object(sys, 'argv', ['main_best.py', '--report-only', '--exp-id', '913',
                    '--output-root', directory, '--experiment-mode', 'full', '--figure-language', language,
                    '--runs', '3', '--epochs', '3']):
                    manifest = main.main()
                self.assertTrue((root / f'Figures/EXP913/{folder}/06_heatmap_accuracy_knn.png').is_file())
                self.assertTrue((root / f'Results/EXP913/{folder}').is_dir())
                self.assertEqual(manifest['optimization_calls'], 0)
                self.assertTrue(manifest['protected_files_unchanged'])
            self.assertEqual(observed, ['KNN — Accuracy (0–1)', 'KNN — Exactitud (0–1)'])
            self.assertEqual({p: core.sha256(p) for p in (root / 'Results').rglob('*')
                              if p.is_file() and p.name != 'experiment_config.json'}, before)
            self.assertTrue((root / 'Results/EXP913/full/experiment_config.json').is_file())
            self.assertFalse(list(root.rglob('full_rep*')))

    def test_missing_cache_never_allocates_figures_or_runs_science(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory))
            args.report_only = True
            with self.assertRaises(FileNotFoundError):
                core.run_report(args)
            self.assertFalse((Path(directory) / 'Figures').exists())

    def test_ambiguous_final_caches_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = caches(arguments(root))
            path = next(root.rglob('*_results.pkl'))
            payload = pickle.loads(path.read_bytes())
            row = next(iter(payload.values()))
            row['AccRuns'][0] += .1
            row['AccMean'] = float(np.mean(row['AccRuns']))
            path.with_name(path.name.replace('_results.pkl', '_other_results.pkl')).write_bytes(pickle.dumps(payload))
            args.report_only = True
            with self.assertRaisesRegex(ValueError, 'Conflicting compatible observations'):
                core.run_report(args)
            self.assertFalse((root / 'Figures').exists())

    def test_existing_exp625_loads_from_cache_with_current_cli_defaults(self):
        root = Path(main.__file__).resolve().parent
        if not (root / 'Results/EXP625/full/cache').is_dir():
            self.skipTest('Historical EXP625 cache unavailable')
        with patch.object(sys, 'argv', ['main_best.py', '--report-only', '--exp-id', '625',
            '--experiment-mode', 'full', '--dataset-source', 'mafese', '--figure-language', 'es',
            '--output-root', str(root)]):
            args = main.parse_args()
        with core.report_guard(()):
            report = core.load_cached_figure_report(args)
        self.assertIn('BreastCancer', report.datasets)
        self.assertEqual(report.classifiers, ['knn', 'svm'])
        self.assertEqual(report.args.runs, 30)
        self.assertTrue(all(core.sha256(root / path) == digest for path, digest in report.sources.items()))

    def test_progress_only_and_arbitrary_hash_with_metadata_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = caches(arguments(root))
            for path in root.rglob('*_results.pkl'):
                path.rename(path.with_name(path.name.rsplit('_', 2)[0] + '_differenthash_progress.pkl'))
            before = {p: core.sha256(p) for p in root.rglob('*.pkl')}
            with core.report_guard(()):
                report = core.load_cached_figure_report(args)
            self.assertEqual(len(report.indexed), 12)
            self.assertTrue(all(path.endswith('_progress.pkl') for path in report.sources))
            self.assertTrue(all(core.sha256(p) == digest for p, digest in before.items()))

    def test_scientific_mismatches_are_audited_and_rejected(self):
        for field, value in (('runs', 4), ('epochs', 4), ('pop_size', 99), ('test_size', .3),
                             ('random_state', 17), ('seed_base', 29)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = caches(arguments(root))
                setattr(args, field, value)
                output = StringIO()
                with redirect_stdout(output), core.report_guard(()), self.assertRaisesRegex(ValueError, field):
                    core.load_cached_figure_report(args)
                self.assertIn('MISMATCH', output.getvalue())
                self.assertIn(field, output.getvalue())
                self.assertFalse((root / 'Figures').exists())

    def test_mismatch_candidate_cannot_hide_compatible_cache_and_identical_duplicates_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = caches(arguments(root))
            for path in list(root.rglob('*_results.pkl')):
                path.with_name(path.name.replace('_results.pkl', '_progress.pkl')).write_bytes(path.read_bytes())
                payload = pickle.loads(path.read_bytes())
                for row in payload.values():
                    row['CacheIdentity']['seed_base'] += 1
                path.with_name(path.name.replace('_results.pkl', '_bad_results.pkl')).write_bytes(pickle.dumps(payload))
            with core.report_guard(()):
                report = core.load_cached_figure_report(args)
            self.assertEqual(len(report.indexed), 12)
            self.assertEqual(len(report.sources), 4)

    def test_explicit_missing_selection_fails_with_clear_audit(self):
        for option, field, value in (('--datasets', 'datasets', ['Missing']),
                                     ('--estimators', 'estimators', ['svm']),
                                     ('--optimizers', 'optimizers', ['BRO'])):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = caches(arguments(root))
                setattr(args, field, value)
                args.report_explicit_options = frozenset({option})
                with core.report_guard(()), self.assertRaises((FileNotFoundError, ValueError)):
                    core.load_cached_figure_report(args)

    def test_legacy_unverified_numeric_settings_are_not_assumed(self):
        root = Path(main.__file__).resolve().parent
        if not (root / 'Results/EXP625/full/cache').is_dir():
            self.skipTest('Historical EXP625 cache unavailable')
        with patch.object(sys, 'argv', ['main_best.py', '--report-only', '--exp-id', '625',
            '--experiment-mode', 'full', '--dataset-source', 'mafese', '--pop-size', '99',
            '--output-root', str(root)]):
            args = main.parse_args()
        with core.report_guard(()), self.assertRaisesRegex(ValueError, 'legacy digest cannot verify'):
            core.load_cached_figure_report(args, use_manifest=False)


if __name__ == '__main__':
    unittest.main()
