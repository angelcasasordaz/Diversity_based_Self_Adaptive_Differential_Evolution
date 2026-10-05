"""Replica figure layout/style contracts; synthetic caches and temporary outputs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.text import Text
import numpy as np

import full_plot_style as style
import main_best as main
from reporting import core, figures, statistics
from tests.test_generic_figures import report_fixture


class FullReplicaLayoutTests(unittest.TestCase):
    def setUp(self):
        self.report = report_fixture(('A', 'B', 'C', 'D', 'E', 'F'),
                                     ('DSADE', 'DE', 'PSO'), ('knn', 'svm', 'rf'))

    def test_root_has_nine_main_views_with_individual_and_statistics_siblings(self):
        """Use the real builders and PNG sink; reduce physical size for this test."""
        def small_save(fig, target):
            fig.set_size_inches(1, 1)
            fig.set_layout_engine(None)
            for ax in list(fig.axes):
                ax.remove()
            original(fig, target)
        original = figures.save_png
        with tempfile.TemporaryDirectory() as directory, patch.object(figures, 'save_png', small_save):
            root = Path(directory)
            report = replace(self.report, datasets=['A', 'B'], classifiers=['knn'])
            figures.generate(report, root)
            statistics.export(report, root / 'statistics', root / 'results')
            self.assertEqual({p.name for p in root.glob('*.png')}, set(figures.base_figure_names(report)))
            self.assertEqual(len(list(root.glob('*.png'))), 9)
            expected = {f'generic_{kind}_knn.png' for kind in
                        ('summary', 'radar', 'precision', 'convergence', 'features_runtime', 'accuracy_boxplot', 'recall_violin')}
            expected |= {f'{kind}_{ds}_knn.png' for ds in report.datasets for kind in ('radar', 'convergence', 'features_runtime')}
            self.assertEqual({p.name for p in (root / 'individual').glob('*.png')}, expected)
            expected_statistics = {'generic_average_rank.png', 'generic_reference_comparisons.png',
                                   'generic_holm_heatmap.png', 'generic_block_distribution.png'}
            self.assertEqual({p.name for p in (root / 'statistics').glob('*.png')}, expected_statistics)
            self.assertFalse((root / 'individual/statistics').exists())
            self.assertEqual(len(list(root.rglob('*.png'))), 9 + len(expected) + len(expected_statistics))
            self.assertFalse(list(root.rglob('*.pdf')))

    def test_full_numbered_base_and_generic_use_one_style_and_identity_palette(self):
        self.assertIs(figures.STYLE, style.STYLE)
        self.assertIs(main.FULL_OPTIMIZER_COLORS, style.OPTIMIZER_COLORS)
        self.assertIs(figures.style_axes, style.style_axes)
        self.assertIs(figures.palette, style.palette)
        for names in (self.report.algorithms, ['PSO', 'DE', 'DSA-DE']):
            colors = figures.palette(names)
            for name in names:
                self.assertEqual(colors[name], main.FULL_OPTIMIZER_COLORS[style.method_key(name)])
                self.assertEqual(style.line_style(name), main.full_optimizer_line_style(name))
        with plt.rc_context(style.STYLE):
            for generator in (figures.base_publication_figures, figures.publication_figures):
                for name, fig in generator(self.report, []):
                    try:
                        style.neutral_text(fig)
                        self.assertEqual(fig.get_facecolor(), to_rgba('white'))
                        for text in fig.findobj(match=Text):
                            self.assertIn(to_rgba(text.get_color()), (to_rgba('black'), to_rgba('white')), name)
                        if 'summary' in name or name.startswith('01_'):
                            self.assertEqual(fig.axes[0].patches[0].get_facecolor(), to_rgba('#0072B2'))
                            self.assertEqual(fig.axes[0].patches[0].get_edgecolor(), to_rgba('black'))
                        if 'convergence' in name:
                            line = next(line for line in fig.axes[0].lines if line.get_label() == 'DSA-DE')
                            self.assertEqual(line.get_color(), '#0072B2')
                            self.assertEqual(line.get_marker(), 'o')
                    finally:
                        plt.close(fig)

    def test_summary_reuses_numbered_renderer_primitive_and_values_are_preserved(self):
        original = style.metric_bars
        calls = []
        def traced(ax, values, algorithms, colors):
            calls.append(np.asarray(values).copy())
            return original(ax, values, algorithms, colors)
        report = self.report
        before = deepcopy(report.indexed)
        with plt.rc_context(style.STYLE), patch.object(style, 'metric_bars', traced):
            fig = figures.summary_figure(report, 'svm')
            plt.close(fig)
            self.assertEqual(len(calls), 7)
            for actual, metric in zip(calls, report.metrics):
                np.testing.assert_array_equal(actual, figures.metric_matrix(report, 'svm', metric).mean(axis=1))
            calls.clear()
            rows = []
            for ds in report.datasets:
                for method in report.algorithms:
                    row = report.indexed[ds, 'svm', method]
                    rows.append(dict(Dataset=ds, Optimizer=method, Estimator='svm',
                        AS_test=np.mean(row['AccRuns'])/100, PS_test=np.mean(row['PSRuns']),
                        RS_test=np.mean(row['RSRuns']), F1_test=np.mean(row['F1Runs'])))
            import pandas as pd
            token = main._FULL_REPORT_STYLE.set(True)
            try:
                with patch.object(main, '_save_chart', side_effect=lambda fig, *a, **k: plt.close(fig)):
                    main.generate_classifier_metric_grid_chart(pd.DataFrame(rows), '/unused', report.algorithms,
                                                               available_estimators_only=True)
            finally:
                main._FULL_REPORT_STYLE.reset(token)
            self.assertEqual(len(calls), 4)
            for actual, key in zip(calls, ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs')):
                metric = next(m for m in report.metrics if m.run_key == key)
                np.testing.assert_allclose(actual, figures.metric_matrix(report, 'svm', metric).mean(axis=1))
        np.testing.assert_equal(report.indexed, before)

    def test_statistical_exports_use_shared_style_without_leaking_rc_changes(self):
        observed = []
        def inspect_save(fig, target):
            try:
                observed.append(Path(target).name)
                self.assertEqual(plt.rcParams['axes.edgecolor'], style.STYLE['axes.edgecolor'])
                self.assertEqual(plt.rcParams['axes.titleweight'], style.STYLE['axes.titleweight'])
                style.neutral_text(fig)
                for text in fig.findobj(match=Text):
                    self.assertIn(to_rgba(text.get_color()), (to_rgba('black'), to_rgba('white')))
            finally:
                plt.close(fig)
        with tempfile.TemporaryDirectory() as directory, plt.rc_context({'axes.edgecolor': 'red'}), \
                patch.object(figures, 'save_png', inspect_save):
            statistics.export(self.report, Path(directory)/'figures', Path(directory)/'results')
            self.assertEqual(plt.rcParams['axes.edgecolor'], 'red')
        self.assertEqual(len(observed), 4)

    def test_orchestration_routes_stats_and_checks_root_without_scientific_execution(self):
        from tests.test_generic_reporting import arguments, caches, tiny_figures, tiny_base_figures, tiny_dataset_figures
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = arguments(root)
            caches(args)
            hashes = {p: core.sha256(p) for p in root.rglob('*.pkl')}
            for version in (1, 2):
                for kind in ('Figures', 'Results'):
                    (root / kind / 'EXP913' / f'full_rep{version}').mkdir(parents=True)
            with patch.object(figures, 'base_publication_figures', tiny_base_figures), \
                    patch.object(figures, 'publication_figures', tiny_figures), \
                    patch.object(figures, 'per_dataset_figures', tiny_dataset_figures), \
                    patch.object(figures, 'statistical_figures', tiny_figures), \
                    patch.object(main, 'build_optimizer', side_effect=AssertionError('No optimizer')):
                manifest = core.run_report(args)
            self.assertEqual(manifest['report_version'], 3)
            self.assertEqual(manifest['optimization_calls'], 0)
            entry = manifest['reports'][0]
            self.assertEqual(len(entry['root_figures']), 9)
            self.assertIn('02_radar_por_dataset_knn.png', entry['root_figures'])
            self.assertEqual(entry['figure_style'], style.STYLE_ID)
            self.assertIn('individual/tiny.png', entry['individual_figures'])
            self.assertEqual(entry['statistical_figures'], ['statistics/tiny.png'])
            self.assertIn('individual/convergence_First_knn.png', entry['individual_figures'])
            self.assertTrue(all(core.sha256(p) == h for p, h in hashes.items()))


if __name__ == '__main__':
    unittest.main()
