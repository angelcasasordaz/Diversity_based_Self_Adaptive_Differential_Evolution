"""Main publication styling and unchanged samples, using synthetic cached runs."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import to_rgba
import numpy as np

import full_plot_style as style
from reporting import figures
from tests.test_generic_figures import report_fixture
from tests.test_radar_convergence_style import DATASETS


class PublicationPolishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.algorithms = ['MaCRO-DE-t', *list(style.OPTIMIZER_COLORS)[1:]]
        cls.report = report_fixture(DATASETS, cls.algorithms, ('knn', 'svm'))
        for di, dataset in enumerate(DATASETS):
            for ai, algorithm in enumerate(cls.algorithms):
                for classifier in cls.report.classifiers:
                    row = cls.report.indexed[dataset, classifier, algorithm]
                    row['RSRuns'] = (.6 + di*.04 + ai*.003 + np.linspace(-.02, .02, 30)).tolist()
                    row['AccRuns'] = (70 + di*3 + ai*.2 + np.linspace(-4, 3, 30)).tolist()
                    # Nearby paired bar tops stress label overlap on both axes.
                    row['FeatRuns'] = [10 + ai*.08 + di*.1] * 30
                    row['TimeRuns'] = [100 + ai*.08 + di*.1] * 30
        cls.before = deepcopy(cls.report.indexed)
        cls.violin_calls, cls.box_calls = [], []
        violin, boxplot = Axes.violinplot, Axes.boxplot

        def trace_violin(ax, values, *args, **kwargs):
            cls.violin_calls.append((ax, np.asarray(values).copy()))
            return violin(ax, values, *args, **kwargs)

        def trace_boxplot(ax, values, *args, **kwargs):
            result = boxplot(ax, values, *args, **kwargs)
            cls.box_calls.append((ax, np.asarray(values).copy(), result))
            return result

        cls.figures = {}
        with plt.rc_context(style.STYLE), patch.object(Axes, 'violinplot', trace_violin), \
                patch.object(Axes, 'boxplot', trace_boxplot):
            for name, fig in figures.base_publication_figures(cls.report, []):
                cls.figures[name] = fig
                cls.addClassCleanup(plt.close, fig)

    def expected_runs(self, key, scale=1):
        return np.asarray([np.concatenate([self.before[dataset, 'knn', algorithm][key]
            for dataset in DATASETS]) / scale for algorithm in self.algorithms])

    def assert_no_run_points(self, ax):
        self.assertFalse(any(isinstance(c, PathCollection) for c in ax.collections))
        self.assertFalse(any(line.get_marker() in ('o', '.') for line in ax.lines))

    def assert_mean_labels(self, ax, expected):
        self.assertEqual(len(ax.texts), len(expected))
        renderer = ax.figure.canvas.get_renderer()
        markers = [line for line in ax.lines if line.get_marker() == 'D']
        for i, (label, sample, marker) in enumerate(zip(ax.texts, expected, markers)):
            mean = float(np.mean(sample))
            self.assertEqual(label.get_text(), f'{mean:.3f}')
            self.assertEqual(label.get_color(), 'black')
            self.assertEqual(label.get_fontweight(), 'bold')
            self.assertEqual(label.get_ha(), 'center')
            self.assertEqual(label.xy, (i, mean))
            self.assertFalse(label.get_in_layout())
            bounds = label.get_window_extent(renderer)
            self.assertFalse(bounds.overlaps(marker.get_window_extent(renderer)))
            median_y = ax.transData.transform((i, np.median(sample)))[1]
            self.assertFalse(bounds.y0 <= median_y <= bounds.y1)
            low, high = ax.transData.transform([(i, np.min(sample)), (i, np.max(sample))])[:, 1]
            self.assertGreaterEqual(bounds.y0, low)
            self.assertLessEqual(bounds.y1, high)

    def test_accuracy_violin_uses_all_cached_runs_without_points_and_keeps_mean_median(self):
        fig = self.figures['07_violin_accuracy_knn']
        ax = fig.axes[0]
        expected = self.expected_runs('AccRuns', 100)
        self.assertEqual(expected.shape, (12, 180))
        samples = [values[0] for axis, values in self.violin_calls if axis is ax]
        np.testing.assert_array_equal(samples, expected)
        self.assert_no_run_points(ax)
        self.assertEqual(sum(isinstance(c, PolyCollection) for c in ax.collections), 12)
        mean_lines = [line for line in ax.lines if line.get_marker() == 'D']
        self.assertEqual(len(mean_lines), 12)
        for i, line in enumerate(mean_lines):
            np.testing.assert_array_equal(line.get_xdata(), [i])
            np.testing.assert_allclose(line.get_ydata(), [expected[i].mean()])
        medians = [c.get_segments()[0][:, 1] for c in ax.collections if isinstance(c, LineCollection)]
        np.testing.assert_allclose(medians, np.repeat(np.median(expected, axis=1)[:, None], 2, axis=1))
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ['Mean', 'Median'])
        self.assertEqual(ax.get_ylabel(), 'Accuracy (test): cached runs across datasets')
        self.assertEqual(ax.get_xticklabels()[0].get_text(), 'DSA-DE')
        fig.canvas.draw()
        self.assert_mean_labels(ax, expected)
        np.testing.assert_equal(self.report.indexed, self.before)

    def test_global_accuracy_boxplot_preserves_scaled_runs_without_points_or_fliers(self):
        fig = self.figures['08_global_accuracy_distribution']
        ax = fig.axes[0]
        expected = self.expected_runs('AccRuns', 100)
        calls = [(values, artists) for axis, values, artists in self.box_calls if axis is ax]
        self.assertEqual(len(calls), 1)
        values, artists = calls[0]
        np.testing.assert_array_equal(values, expected.T)
        self.assert_no_run_points(ax)
        self.assertFalse(artists['fliers'])
        for i, (mean, median, box) in enumerate(zip(artists['means'], artists['medians'], artists['boxes'])):
            self.assertEqual(mean.get_marker(), 'D')
            np.testing.assert_allclose(mean.get_ydata(), [expected[i].mean()])
            np.testing.assert_allclose(median.get_ydata(), np.repeat(np.median(expected[i]), 2))
            np.testing.assert_allclose(np.unique(box.get_path().vertices[:, 1]), np.percentile(expected[i], [25, 75]))
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ['Mean', 'Median'])
        self.assertEqual(ax.get_ylabel(), 'Accuracy (test): cached runs across datasets')
        self.assertEqual(ax.get_xticklabels()[0].get_text(), 'DSA-DE')
        fig.canvas.draw()
        self.assert_mean_labels(ax, expected)
        np.testing.assert_equal(self.report.indexed, self.before)

    def test_recall_violin_and_global_distribution_have_no_scatter_calls_and_keep_mean_median(self):
        report = deepcopy(self.report)
        report.args.plot_global_metric = 'recall'
        before = deepcopy(report.indexed)
        expected = self.expected_runs('RSRuns')
        with patch.object(Axes, 'scatter', side_effect=AssertionError('No individual scatter points')):
            views = dict(figures.base_publication_figures(report, []))
        try:
            ax = views['07_violin_recall_knn'].axes[0]
            self.assert_no_run_points(ax)
            self.assert_no_run_points(views['08_global_recall_distribution'].axes[0])
            means = [line for line in ax.lines if line.get_marker() == 'D']
            np.testing.assert_allclose([line.get_ydata()[0] for line in means], expected.mean(axis=1))
            medians = [c.get_segments()[0][:, 1] for c in ax.collections if isinstance(c, LineCollection)]
            np.testing.assert_allclose(medians, np.repeat(np.median(expected, axis=1)[:, None], 2, axis=1))
            self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ['Mean', 'Median'])
            self.assertEqual(ax.get_ylabel(), 'Recall (test): cached runs across datasets')
            self.assertEqual(ax.get_xticklabels()[0].get_text(), 'DSA-DE')
            ax.figure.canvas.draw()
            self.assert_mean_labels(ax, expected)
            distribution = views['08_global_recall_distribution'].axes[0]
            distribution.figure.canvas.draw()
            self.assert_mean_labels(distribution, expected)
        finally:
            for fig in views.values():
                plt.close(fig)
        np.testing.assert_equal(report.indexed, before)
        self.assertEqual(report.algorithms, self.algorithms)

    def test_features_runtime_labels_do_not_overlap_and_values_hatching_palette_are_preserved(self):
        fig = self.figures['03_features_runtime_por_dataset_knn']
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left_axes = [ax for ax in fig.axes if ax.get_ylabel() == 'Average selected features']
        right_axes = [ax for ax in fig.axes if ax.get_ylabel() == 'Average runtime (s)']
        self.assertEqual(len(left_axes), 6)
        self.assertEqual(len(right_axes), 6)
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], ['Selected features', 'Runtime'])
        for dataset, left, right in zip(DATASETS, left_axes, right_axes):
            self.assertIsNone(left.get_legend())
            self.assertIsNone(right.get_legend())
            for axis, key, hatch in ((left, 'FeatRuns', None), (right, 'TimeRuns', '///')):
                self.assertEqual(len(axis.patches), 12)
                self.assertEqual(len(axis.texts), 12)
                for bar, label, algorithm in zip(axis.patches, axis.texts, self.algorithms):
                    value = np.mean(self.before[dataset, 'knn', algorithm][key])
                    self.assertAlmostEqual(bar.get_height(), value)
                    self.assertEqual(bar.get_hatch(), hatch)
                    self.assertEqual(bar.get_facecolor()[:3], to_rgba(style.palette(self.algorithms)[algorithm])[:3])
                    self.assertEqual(label.get_text(), f'{value:.1f}s' if hatch else f'{value:.2f}')
                    self.assertEqual(label.get_rotation(), 90)
                    self.assertEqual(label.get_fontsize(), 6)
                    bounds, label_box = axis.get_window_extent(renderer), label.get_window_extent(renderer)
                    self.assertGreaterEqual(label_box.x0, bounds.x0)
                    self.assertLessEqual(label_box.x1, bounds.x1)
                    self.assertLessEqual(label_box.y1, bounds.y1)
            boxes = [t.get_window_extent(renderer) for axis in (left, right) for t in axis.texts]
            for i, box in enumerate(boxes):
                for other in boxes[i+1:]:
                    self.assertFalse(box.overlaps(other), (dataset, box, other))
        np.testing.assert_equal(self.report.indexed, self.before)

    def test_polish_is_limited_to_main_distributions_and_keeps_nine_names_and_internal_identity(self):
        self.assertEqual({f'{name}.png' for name in self.figures}, set(figures.base_figure_names(self.report)))
        self.assertEqual(len(self.figures), 9)
        self.assertEqual(self.report.algorithms, self.algorithms)
        self.assertIn(('DataClass', 'knn', 'MaCRO-DE-t'), self.report.indexed)
        self.assertNotIn(('DataClass', 'knn', 'DSA-DE'), self.report.indexed)
        # Generic distributions still show their observed dataset means.
        for builder in (figures.violin_figure, figures.boxplot_figure):
            values = self.expected_runs('RSRuns')[:, :6]
            before = values.copy()
            fig = builder(values, self.algorithms, 'knn')
            try:
                clouds = [c for c in fig.axes[0].collections if isinstance(c, PathCollection)
                          and len(c.get_offsets()) == 6]
                self.assertEqual(len(clouds), 12)
                np.testing.assert_array_equal(values, before)
            finally:
                plt.close(fig)
        np.testing.assert_equal(self.report.indexed, self.before)


if __name__ == '__main__':
    unittest.main()
