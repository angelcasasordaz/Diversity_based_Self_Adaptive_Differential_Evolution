"""Presentation options select stored data and visible text without scientific execution."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.text import Text
import numpy as np

import main_best as main
from figure_layout import main_figure_names, full_figure_category, full_figure_path, METRIC_TOKENS
from optimizer_factory import optimizer_scientific_identity
from reporting import figures
from tests.test_generic_figures import report_fixture
from tests.test_radar_convergence_style import DATASETS


class PublicationOptionsTests(unittest.TestCase):
    def args(self, classifier='knn', metric='accuracy', language='en'):
        with patch.object(sys, 'argv', ['main_best.py', '--report-only', '--experiment-mode', 'full',
            '--plot-global-estimator', classifier, '--plot-global-metric', metric, '--figure-language', language]):
            return main.parse_args()

    def report(self, classifier='knn', metric='accuracy', language='en'):
        report = report_fixture(DATASETS[:2], ('MaCRO-DE-t', 'DE', 'JADE'), ('knn', 'svm', 'rf'))
        report.args.plot_global_estimator = self.args(classifier, metric, language).plot_global_estimator
        report.args.plot_global_metric = metric
        report.args.figure_language = language
        return report

    def views(self, report):
        views = {}
        skipped = []
        with plt.rc_context(figures.STYLE):
            for name, fig in figures.base_publication_figures(report, skipped):
                views[name] = fig
                self.addCleanup(plt.close, fig)
        self.assertFalse(skipped)
        self.assertEqual({f'{name}.png' for name in views}, set(figures.base_figure_names(report)))
        self.assertEqual(len(views), 9)
        return views

    def test_cli_defaults_choices_and_main_config_override(self):
        defaults = self.args()
        self.assertEqual((defaults.plot_global_estimator, defaults.plot_global_metric, defaults.figure_language),
                         ('knn', 'accuracy', 'en'))
        with patch.object(main, 'PLOT_GLOBAL_ESTIMATOR', 'rf'), patch.object(main, 'PLOT_GLOBAL_METRIC', 'precision'), \
                patch.object(main, 'FIGURE_LANGUAGE', 'es'), patch.object(sys, 'argv', ['main_best.py']):
            configured = main.parse_args()
        self.assertEqual((configured.plot_global_estimator, configured.plot_global_metric, configured.figure_language),
                         ('rf', 'precision', 'es'))
        report = report_fixture(DATASETS[:1], ('MaCRO-DE-t', 'DE'), ('knn', 'rf'))
        report.args = configured
        self.assertEqual(figures.base_figure_names(report), main_figure_names('rf', 'precision'))
        for option, invalid in (('--plot-global-estimator', 'tree'), ('--plot-global-metric', 'fitness'),
                                ('--figure-language', 'fr')):
            with self.subTest(option=option), patch.object(sys, 'argv', ['main_best.py', option, invalid]), \
                    patch.object(sys, 'stderr'), self.assertRaises(SystemExit):
                main.parse_args()

    def test_selected_classifier_controls_root_names_and_plotted_values(self):
        for classifier in ('knn', 'svm', 'rf'):
            with self.subTest(classifier=classifier):
                report = self.report(classifier)
                before = deepcopy(report.indexed)
                views = self.views(report)
                metric = next(m for m in report.metrics if m.run_key == 'AccRuns')
                heatmap = views[f'06_heatmap_accuracy_{classifier}'].axes[0]
                np.testing.assert_array_equal(heatmap.images[0].get_array(),
                                              figures.metric_matrix(report, classifier, metric))
                labels, values = figures.radar_values(report, classifier)
                radar = views[f'02_radar_por_dataset_{classifier}']
                for di, ax in enumerate(radar.axes):
                    self.assertEqual(ax.get_title(), f'{report.datasets[di]} / {classifier.upper()}')
                    for line in ax.lines:
                        ai = next(i for i, a in enumerate(report.algorithms)
                                  if figures.base_style.display_label(a, report.algorithms) == line.get_label())
                        np.testing.assert_array_equal(line.get_ydata(), np.r_[values[ai, di], values[ai, di, 0]])
                self.assertEqual(views['08_global_accuracy_distribution'].axes[0].get_title(), classifier.upper())
                self.assertEqual(views['09_global_features_runtime_tradeoff'].axes[0].get_title(), classifier.upper())
                np.testing.assert_equal(report.indexed, before)
                for fig in views.values():
                    plt.close(fig)

    def test_all_four_metrics_drive_heatmap_violin_global_and_dataset_boxplots(self):
        for token in METRIC_TOKENS:
            with self.subTest(metric=token):
                report = self.report('svm', token)
                before = deepcopy(report.indexed)
                metric = next(m for m in report.metrics if figures.metric_token(m) == token)
                calls = {'violin': [], 'boxplot': []}
                original_violin, original_box = Axes.violinplot, Axes.boxplot
                def trace_violin(ax, values, *args, **kwargs):
                    calls['violin'].append((ax, np.asarray(values).copy()))
                    return original_violin(ax, values, *args, **kwargs)
                def trace_box(ax, values, *args, **kwargs):
                    calls['boxplot'].append((ax, np.asarray(values).copy()))
                    return original_box(ax, values, *args, **kwargs)
                with patch.object(Axes, 'violinplot', trace_violin), patch.object(Axes, 'boxplot', trace_box):
                    views = self.views(report)
                heatmap = views[f'06_heatmap_{token}_svm'].axes[0]
                np.testing.assert_array_equal(heatmap.images[0].get_array(), figures.metric_matrix(report, 'svm', metric))
                expected = np.asarray([np.concatenate([before[ds, 'svm', a][metric.run_key]
                    for ds in report.datasets]) / metric.scale for a in report.algorithms])
                violin = views[f'07_violin_{token}_svm'].axes[0]
                np.testing.assert_array_equal([v[0] for ax, v in calls['violin'] if ax is violin], expected)
                global_ax = views[f'08_global_{token}_distribution'].axes[0]
                np.testing.assert_array_equal(next(v for ax, v in calls['boxplot'] if ax is global_ax), expected.T)
                for ax in (violin, global_ax):
                    self.assertFalse(any(isinstance(c, PathCollection) for c in ax.collections))
                    self.assertEqual(ax.get_ylabel(), f'{metric.name} (test): cached runs across datasets')
                per_dataset = views[f'04_boxplot_{token}_por_dataset_svm']
                for ds, axis in zip(report.datasets, per_dataset.axes):
                    expected = np.asarray([before[ds, 'svm', a][metric.run_key] for a in report.algorithms]) / metric.scale
                    np.testing.assert_array_equal(next(v for ax, v in calls['boxplot'] if ax is axis), expected.T)
                    self.assertEqual(axis.get_ylabel(), f'{metric.name} (test)')
                self.assertEqual(heatmap.patches[0].get_edgecolor(), to_rgba('black'))
                self.assertEqual(len(heatmap.texts), len(report.algorithms)*len(report.datasets))
                np.testing.assert_equal(report.indexed, before)
                for fig in views.values():
                    plt.close(fig)

    def test_plotting_options_never_change_cache_or_checkpoint_or_optimizer_identity(self):
        baseline = self.args()
        signature = main.build_cache_signature(baseline)
        legacy = main.build_legacy_cache_signature(baseline)
        identities = {c: main.full_cache_identity(baseline, 'MaCRO-DE-t', 'DataClass', c, 'vstf_01')
                      for c in baseline.estimators}
        optimizer_identity = optimizer_scientific_identity('MaCRO-DE-t', baseline)
        for classifier, token, language in product(('knn', 'svm', 'rf'), METRIC_TOKENS, ('en', 'es')):
            args = self.args(classifier, token, language)
            self.assertEqual(main.build_cache_signature(args), signature)
            self.assertEqual(main.build_legacy_cache_signature(args), legacy)
            self.assertEqual(optimizer_scientific_identity('MaCRO-DE-t', args), optimizer_identity)
            for stored_classifier in baseline.estimators:
                identity = main.full_cache_identity(args, 'MaCRO-DE-t', 'DataClass', stored_classifier, 'vstf_01')
                self.assertEqual(identity, identities[stored_classifier])
                self.assertEqual(main.full_checkpoint_metadata(args, 'MaCRO-DE-t', 'DataClass', stored_classifier, 'vstf_01'),
                                 main.full_checkpoint_metadata(baseline, 'MaCRO-DE-t', 'DataClass', stored_classifier, 'vstf_01'))
                label = main.build_alg_label('MaCRO-DE-t', 'vstf_01', stored_classifier, False, True)
                self.assertEqual(label, f'MACRO-DE-T_{stored_classifier.upper()}')
                self.assertEqual(main.parse_result_label(label, args)['method'], 'MaCRO-DE-t')

    def test_spanish_changes_visible_root_labels_but_not_names_or_aliases(self):
        report = self.report('knn', 'recall', 'es')
        before = deepcopy(report.indexed)
        spanish = self.views(report)
        report.args.figure_language = 'en'
        english = self.views(report)
        self.assertEqual(spanish.keys(), english.keys())
        self.assertEqual(spanish['06_heatmap_recall_knn'].axes[0].get_title(), 'KNN — Sensibilidad (0–1)')
        self.assertEqual(spanish['06_heatmap_recall_knn'].axes[0].get_xlabel(), 'Conjunto de datos')
        self.assertEqual(spanish['06_heatmap_recall_knn'].axes[0].get_ylabel(), 'Metaheurísticas')
        self.assertEqual(spanish['07_violin_recall_knn'].axes[0].get_ylabel(),
                         'Sensibilidad (prueba): Corridas en caché entre conjuntos')
        self.assertEqual([t.get_text() for t in spanish['07_violin_recall_knn'].axes[0].get_legend().get_texts()],
                         ['Media', 'Mediana'])
        self.assertEqual(spanish['03_features_runtime_por_dataset_knn'].axes[0].get_ylabel(),
                         'Promedio de características seleccionadas')
        self.assertEqual([t.get_text() for t in spanish['03_features_runtime_por_dataset_knn'].legends[0].get_texts()],
                         ['Características seleccionadas', 'Tiempo de ejecución'])
        self.assertEqual([t.get_text() for t in spanish['02_radar_por_dataset_knn'].axes[0].get_xticklabels()][:4],
                         ['Exactitud', 'Precisión', 'Sensibilidad', 'F1-Score'])
        for name, fig in spanish.items():
            fig.canvas.draw()
            texts = [t.get_text() for t in fig.findobj(match=Text)]
            self.assertNotIn('MACRO-DE-T', texts)
            self.assertTrue(any('DSA-DE' in text for text in texts), name)
        convergence = spanish['05_convergence_por_dataset_knn']
        for ax in convergence.axes:
            self.assertEqual((ax.get_xlabel(), ax.get_ylabel()), ('Iteración', 'Aptitud'))
            self.assertEqual(ax.child_axes[0].get_title(), 'Etapa final')
        np.testing.assert_equal(report.indexed, before)
        self.assertIn(('DataClass', 'knn', 'MaCRO-DE-t'), report.indexed)

    def test_all_six_convergence_panels_have_final_quarter_insets_with_all_curves(self):
        report = report_fixture(DATASETS, ('MaCRO-DE-t', 'DE', 'JADE'), ('knn',))
        for di, ds in enumerate(DATASETS):
            for ai, algorithm in enumerate(report.algorithms):
                report.indexed[ds, 'knn', algorithm]['Curve'] = (np.linspace(.9, .2, 150) + ai*.003 + di*.001).tolist()
        before = deepcopy(report.indexed)
        for language, title in (('en', 'Final stage'), ('es', 'Etapa final')):
            report.args.figure_language = language
            skipped = []
            fig = figures.convergence_figure(report, 'knn', report.datasets, skipped=skipped)
            try:
                self.assertFalse(skipped)
                self.assertEqual(len(fig.axes), 6)
                self.assertEqual(len(fig.legends), 1)
                self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], ['DSA-DE', 'DE', 'JADE'])
                fig.canvas.draw()
                for ds, ax in zip(DATASETS, fig.axes):
                    self.assertEqual(len(ax.child_axes), 1)
                    inset = ax.child_axes[0]
                    self.assertEqual(inset.get_title(), title)
                    self.assertEqual(inset.get_xlim(), (112, 149))
                    self.assertEqual(len(ax.lines), 3)
                    self.assertEqual(len(inset.lines), 3)
                    np.testing.assert_allclose(
                        ax.transAxes.inverted().transform_bbox(inset.get_window_extent()).bounds,
                        (.56, .16, .39, .30))
                    for main_line, zoom_line in zip(ax.lines, inset.lines):
                        algorithm = next(a for a in report.algorithms
                            if figures.base_style.display_label(a, report.algorithms) == main_line.get_label())
                        curve = before[ds, 'knn', algorithm]['Curve']
                        np.testing.assert_array_equal(main_line.get_ydata(), curve)
                        np.testing.assert_array_equal(zoom_line.get_ydata(), curve[112:])
                        self.assertEqual(zoom_line.get_color(), main_line.get_color())
                        self.assertEqual(zoom_line.get_linewidth(), main_line.get_linewidth())
                        self.assertNotEqual(to_rgba(zoom_line.get_color()), to_rgba('black'))
            finally:
                plt.close(fig)
        np.testing.assert_equal(report.indexed, before)

    def test_missing_or_short_curves_report_explicit_inset_reasons_and_keep_dataset_panels(self):
        report = self.report()
        report.indexed['DataClass', 'knn', 'DE']['Curve'] = []
        report.indexed['FeatureEnvy', 'knn', 'JADE']['Curve'] = [.5, .4]
        skipped = []
        views = dict(figures.base_publication_figures(report, skipped))
        for fig in views.values():
            self.addCleanup(plt.close, fig)
        convergence = views['05_convergence_por_dataset_knn']
        self.assertEqual(len(convergence.axes), 2)
        self.assertTrue(all(not ax.child_axes for ax in convergence.axes))
        self.assertEqual(len(skipped), 2)
        self.assertIn('DataClass/knn', skipped[0]['output'])
        self.assertIn('DE', skipped[0]['reason'])
        self.assertIn('fewer than three', skipped[1]['reason'])
        self.assertEqual([t.get_text() for t in convergence.legends[0].get_texts()], ['DSA-DE', 'DE', 'JADE'])

    def test_insets_magnify_bottom_curves_without_including_high_primary_in_zoom(self):
        algorithms = ('MaCRO-DE-t', 'DE', 'JADE', 'PSO', 'SA', 'BRO')
        for primary_final in (.002, .25):
            with self.subTest(primary_final=primary_final):
                report = report_fixture(DATASETS, algorithms, ('knn',))
                levels = (primary_final, .003, .004, .2, .4, .6)
                for ds in DATASETS:
                    for algorithm, level in zip(algorithms, levels):
                        report.indexed[ds, 'knn', algorithm]['Curve'] = (
                            level + .8*np.exp(-np.arange(150)/12)).tolist()
                before = deepcopy(report.indexed)
                fig = figures.convergence_figure(report, 'knn', report.datasets)
                try:
                    fig.canvas.draw()
                    for ds, ax in zip(DATASETS, fig.axes):
                        inset = ax.child_axes[0]
                        self.assertEqual(inset.get_xlim(), (112, 149))
                        ymin, ymax = inset.get_ylim()
                        self.assertLess(ymax, .01)
                        self.assertLess(ymin, .003)
                        self.assertEqual(len(inset.lines), len(algorithms))
                        primary = inset.lines[-1]
                        self.assertEqual(primary.get_label(), 'DSA-DE')
                        self.assertEqual(primary.get_color(), '#0072B2')
                        self.assertTrue(all(primary.get_zorder() > line.get_zorder() for line in inset.lines[:-1]))
                        if primary_final == .002:
                            self.assertTrue(np.all((primary.get_ydata() >= ymin) & (primary.get_ydata() <= ymax)))
                        else:
                            self.assertTrue(np.all(primary.get_ydata() > ymax))
                        for line in ax.lines:
                            algorithm = next(a for a in algorithms
                                if figures.base_style.display_label(a, algorithms) == line.get_label())
                            np.testing.assert_array_equal(line.get_ydata(), before[ds, 'knn', algorithm]['Curve'])
                        for line in inset.lines:
                            if line.get_label() in ('PSO', 'SA', 'BRO'):
                                self.assertTrue(np.all(line.get_ydata() > ymax))
                    np.testing.assert_equal(report.indexed, before)
                finally:
                    plt.close(fig)

    def test_dense_curves_keep_main_limits_and_lower_right_inset(self):
        algorithms = ['MaCRO-DE-t', *list(figures.base_style.OPTIMIZER_COLORS)[1:]]
        report = report_fixture(DATASETS[:1], algorithms, ('knn',))
        for algorithm, value in zip(algorithms, np.linspace(.2, .9, 12)):
            report.indexed['DataClass', 'knn', algorithm]['Curve'] = [value]*150
        before = deepcopy(report.indexed)
        with patch.object(figures, 'final_stage_inset', return_value=None):
            reference = figures.convergence_figure(report, 'knn', report.datasets)
        fig = figures.convergence_figure(report, 'knn', report.datasets)
        try:
            fig.canvas.draw()
            ax = fig.axes[0]
            self.assertEqual(ax.get_ylim(), reference.axes[0].get_ylim())
            self.assertEqual(ax.get_xlim(), reference.axes[0].get_xlim())
            np.testing.assert_allclose(
                ax.transAxes.inverted().transform_bbox(ax.child_axes[0].get_window_extent()).bounds,
                (.56, .16, .39, .30))
            self.assertEqual(len(ax.child_axes[0].lines), 12)
        finally:
            plt.close(fig)
            plt.close(reference)
        np.testing.assert_equal(report.indexed, before)

    def test_lower_inset_zoom_does_not_change_main_panel_layout_limits_or_lines(self):
        report = report_fixture(DATASETS[:2], ('MaCRO-DE-t', 'DE', 'JADE', 'PSO'), ('knn',))
        for ds in report.datasets:
            for algorithm, level in zip(report.algorithms, (.002, .003, .2, .6)):
                report.indexed[ds, 'knn', algorithm]['Curve'] = (level + np.exp(-np.arange(150)/12)).tolist()
        original = figures.final_stage_inset
        def full_range_inset(*args, **kwargs):
            inset = original(*args, **kwargs)
            start = int(inset.get_xlim()[0])
            tail = np.concatenate([curve[start:] for curve in args[1]])
            padding = max(float(np.ptp(tail))*.12, float(np.max(np.abs(tail)))*.001, 1e-6)
            inset.set_ylim(float(tail.min())-padding, float(tail.max())+padding)
            return inset
        with patch.object(figures, 'final_stage_inset', full_range_inset):
            previous = figures.convergence_figure(report, 'knn', report.datasets)
        current = figures.convergence_figure(report, 'knn', report.datasets)
        try:
            previous.canvas.draw(); current.canvas.draw()
            for old, new in zip(previous.axes, current.axes):
                np.testing.assert_array_equal(old.get_position().bounds, new.get_position().bounds)
                self.assertEqual(old.get_xlim(), new.get_xlim())
                self.assertEqual(old.get_ylim(), new.get_ylim())
                np.testing.assert_array_equal(old.child_axes[0].get_position().bounds,
                                              new.child_axes[0].get_position().bounds)
                self.assertLess(new.child_axes[0].get_ylim()[1], old.child_axes[0].get_ylim()[1])
                for old_line, new_line in zip(old.lines, new.lines):
                    np.testing.assert_array_equal(old_line.get_xdata(), new_line.get_xdata())
                    np.testing.assert_array_equal(old_line.get_ydata(), new_line.get_ydata())
                    for property_name in ('linewidth', 'color', 'linestyle', 'marker', 'zorder', 'label'):
                        self.assertEqual(getattr(old_line, f'get_{property_name}')(),
                                         getattr(new_line, f'get_{property_name}')())
        finally:
            plt.close(previous); plt.close(current)

    def test_dynamic_names_keep_root_individual_statistics_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for classifier, token in product(('knn', 'svm', 'rf'), METRIC_TOKENS):
                names = main_figure_names(classifier, token)
                self.assertEqual(len(names), 9)
                for name in names:
                    self.assertEqual(full_figure_category(name), 'root')
                    self.assertEqual(full_figure_path(root, name), root/name)
            self.assertEqual(full_figure_path(root, 'generic_heatmap_rf_recall.png'),
                             root/'individual/generic_heatmap_rf_recall.png')
            self.assertEqual(full_figure_path(root, 'generic_holm_heatmap.png'), root/'statistics/generic_holm_heatmap.png')

    def test_unavailable_selected_classifier_is_explicit_and_never_silently_switched(self):
        report = self.report('rf')
        report = replace(report, classifiers=['knn'])
        with self.assertRaisesRegex(ValueError, "Selected publication classifier 'rf'.*no stored results"):
            figures.base_figure_names(report)


if __name__ == '__main__':
    unittest.main()
