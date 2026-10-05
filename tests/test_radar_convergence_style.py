"""Balanced publication curves on synthetic data; no scientific execution."""
from copy import deepcopy
import unittest

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

import full_plot_style as style
from reporting import figures
from tests.test_generic_figures import report_fixture


DATASETS = ('DataClass', 'FeatureEnvy', 'GodClass', 'LongMethod',
            'LongParameterList', 'SwitchStatements')


class RadarConvergenceStyleTests(unittest.TestCase):
    def assert_balanced_lines(self, fig, report, classifier, primary_width, other_width):
        colors = style.palette(report.algorithms)
        axes = [ax for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(axes), 6)
        expected_labels = [style.display_label(a, report.algorithms) for a in report.algorithms]
        draw_algorithms = [a for a in report.algorithms if style.method_key(a) != 'DSADE'] + [
            a for a in report.algorithms if style.method_key(a) == 'DSADE']
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], expected_labels)
        for ax, dataset in zip(axes, DATASETS):
            self.assertEqual(ax.get_title(), f'{dataset} / {classifier.upper()}')
            # Include unlabeled artists: the old black backing curve had no label.
            self.assertEqual(len(ax.lines), len(report.algorithms))
            self.assertEqual([line.get_label() for line in ax.lines],
                             [style.display_label(a, report.algorithms) for a in draw_algorithms])
            primary = ax.lines[-1]
            self.assertEqual(primary.get_label(), 'DSA-DE')
            self.assertTrue(all(primary.get_zorder() > line.get_zorder() for line in ax.lines[:-1]))
            self.assertTrue(all(primary.get_linewidth() > line.get_linewidth() for line in ax.lines[:-1]))
            for line, algorithm in zip(ax.lines, draw_algorithms):
                self.assertEqual(line.get_label(), style.display_label(algorithm, report.algorithms))
                self.assertEqual(to_rgba(line.get_color()), to_rgba(colors[algorithm]))
                self.assertNotEqual(to_rgba(line.get_color()), to_rgba('black'))
                self.assertEqual(to_rgba(line.get_markeredgecolor()), to_rgba(colors[algorithm]))
                self.assertEqual(line.get_linestyle(), style.line_style(algorithm)['linestyle'])
                self.assertEqual(line.get_marker(), style.line_style(algorithm)['marker'])
                self.assertFalse(line.get_path_effects())
                self.assertEqual(line.get_zorder(), 3 if style.method_key(algorithm) == 'DSADE' else 2)
                self.assertAlmostEqual(line.get_linewidth(), primary_width
                    if style.method_key(algorithm) == 'DSADE' else other_width)
        for handle, algorithm in zip(fig.legends[0].legend_handles, report.algorithms):
            self.assertEqual(to_rgba(handle.get_color()), to_rgba(colors[algorithm]))
            self.assertAlmostEqual(handle.get_linewidth(), primary_width
                if style.method_key(algorithm) == 'DSADE' else other_width)
        return axes

    def reports(self):
        for classifier, alias in (('knn', 'MaCRO-DE-t'), ('svm', 'DSADE'), ('rf', 'DSA-DE')):
            # Full project palette, with the primary method between other methods.
            algorithms = list(style.OPTIMIZER_COLORS)[1:]
            algorithms.insert(2, alias)
            yield classifier, report_fixture(DATASETS, algorithms, (classifier,))

    def test_radar_has_one_modest_palette_line_per_method_and_subtle_fill(self):
        for classifier, report in self.reports():
            with self.subTest(classifier=classifier):
                before = deepcopy(report.indexed)
                algorithms = report.algorithms.copy()
                labels, values = figures.radar_values(report, classifier)
                original_values = values.copy()
                fig = figures.radar_figure(report, classifier, labels, values)
                try:
                    axes = self.assert_balanced_lines(fig, report, classifier, 2.1, 1.2)
                    for di, ax in enumerate(axes):
                        self.assertEqual(len(ax.patches), len(algorithms))
                        for line, fill in zip(ax.lines, ax.patches):
                            ai = algorithms.index(next(a for a in algorithms
                                if style.display_label(a, algorithms) == line.get_label()))
                            np.testing.assert_array_equal(line.get_ydata(),
                                np.r_[values[ai, di], values[ai, di, 0]])
                            self.assertLessEqual(fill.get_alpha(), .04)
                    fig.canvas.draw()
                finally:
                    plt.close(fig)
                np.testing.assert_array_equal(values, original_values)
                np.testing.assert_equal(report.indexed, before)
                self.assertEqual(report.algorithms, algorithms)

    def test_convergence_has_one_modest_palette_line_per_method_and_preserves_curves(self):
        for classifier, report in self.reports():
            with self.subTest(classifier=classifier):
                algorithms = report.algorithms.copy()
                # Long curves exercise sparse markers and distinguish stored methods.
                for di, dataset in enumerate(DATASETS):
                    for ai, algorithm in enumerate(algorithms):
                        report.indexed[dataset, classifier, algorithm]['Curve'] = (
                            np.linspace(.9, .2, 48) + (di + ai) / 1000).tolist()
                before = deepcopy(report.indexed)
                fig = figures.convergence_figure(report, classifier, report.datasets)
                try:
                    axes = self.assert_balanced_lines(fig, report, classifier, 2.4, 1.3)
                    for ax, dataset in zip(axes, DATASETS):
                        self.assertEqual(ax.get_xlabel(), 'Iteration')
                        self.assertEqual(ax.get_ylabel(), 'Fitness')
                        self.assertTrue(any(line.get_visible() for line in ax.get_ygridlines()))
                        for line in ax.lines:
                            algorithm = next(a for a in algorithms
                                if style.display_label(a, algorithms) == line.get_label())
                            np.testing.assert_array_equal(line.get_xdata(), np.arange(48))
                            np.testing.assert_array_equal(line.get_ydata(),
                                before[dataset, classifier, algorithm]['Curve'])
                            self.assertEqual(line.get_markevery(), 4)
                        self.assertEqual(len(ax.child_axes), 1)
                        inset = ax.child_axes[0]
                        self.assertEqual(len(inset.lines), len(algorithms))
                        self.assertEqual([line.get_label() for line in inset.lines],
                                         [line.get_label() for line in ax.lines])
                        for main, zoom in zip(ax.lines, inset.lines):
                            np.testing.assert_array_equal(zoom.get_xdata(), np.arange(36, 48))
                            np.testing.assert_array_equal(zoom.get_ydata(), main.get_ydata()[36:])
                            self.assertEqual(zoom.get_linewidth(), main.get_linewidth())
                            self.assertEqual(zoom.get_zorder(), main.get_zorder())
                            self.assertEqual(zoom.get_color(), main.get_color())
                            self.assertFalse(zoom.get_path_effects())
                    fig.canvas.draw()
                    for ax in axes:
                        np.testing.assert_allclose(
                            ax.transAxes.inverted().transform_bbox(ax.child_axes[0].get_window_extent()).bounds,
                            (.56, .16, .39, .30))
                finally:
                    plt.close(fig)
                np.testing.assert_equal(report.indexed, before)
                self.assertEqual(report.algorithms, algorithms)


if __name__ == '__main__':
    unittest.main()
