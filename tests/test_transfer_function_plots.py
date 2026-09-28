"""Plot-only regressions; synthetic records never invoke an optimizer."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import main_best as study
import historical_transfer_plots as historical


class TransferFunctionPlotTests(unittest.TestCase):
    def test_blue_defaults_cannot_overwrite_normal_filenames(self):
        for function, renderer, expected in (
            (study.generate_classifier_metric_grid_chart, 'render_classifier_metric_grid', historical.METRICS_FILENAME),
            (study.generate_global_features_runtime, 'render_features_runtime', historical.TRADEOFF_FILENAME),
        ):
            with patch.object(historical, renderer) as render:
                function(None, 'output', ['MaCRO-DE'], blue_transfer=True)
                self.assertEqual(render.call_args.args[-1], expected)

    def test_available_classifiers_and_saved_curves(self):
        args = argparse.Namespace(experiment_mode='transfer_functions',
                                  optimizers=['MaCRO-DE'], estimators=['knn', 'svm'])
        for estimators in [('knn',), ('svm',), ('knn', 'svm')]:
            with self.subTest(estimators=estimators), tempfile.TemporaryDirectory() as tmp:
                records = {}
                for estimator in estimators:
                    for i, tf in enumerate(study.SUPPORTED_TRANSFER_FUNCTIONS):
                        records[f'MaCRO-DE_{tf.upper()}_{estimator.upper()}'] = {
                            'Estimator': estimator, 'AccMean': 80 + i,
                            'PSMean': .8, 'RSMean': .7, 'F1Mean': .75,
                            'FeatMean': 2 + i, 'TimeMean': 1 + i,
                            'Curve': np.array([.8, .5, .2]) + i / 100,
                        }
                results = {'Dataset': records}
                summary = study.generate_summary_dataframe(results, args)
                figures = {}

                def capture(fig, out_dir, filename, *, save_pdf=True):
                    self.assertFalse(save_pdf)
                    figures[filename] = fig
                    Path(out_dir, filename).touch()

                def capture_historical(fig, out_dir, filename):
                    extra_name = {
                        '09_resultados_clasificador_metrica_todos_datasets.png': historical.METRICS_FILENAME,
                        '09_global_features_runtime_tradeoff.png': historical.TRADEOFF_FILENAME,
                    }[filename]
                    self.assertEqual(fig.dpi, 100)
                    figures[extra_name] = fig
                    Path(out_dir, filename).touch()

                try:
                    with patch.object(historical, '_save_chart', side_effect=capture_historical), patch.object(study, '_save_chart', side_effect=capture), patch.object(
                        study, 'execute_pending_runs', side_effect=AssertionError('Optimization forbidden')
                    ):
                        names = study.generate_transfer_function_charts(
                            summary, results, tmp, args.optimizers, args
                        )
                    self.assertEqual(len(names), 3 + 3 * len(estimators))
                    self.assertEqual({name for name in names if name.endswith('_Blue.png')}, {
                        'TransferFunctions_ClassificationMetrics_Blue.png',
                        'TransferFunctions_FeaturesRuntimeTradeoff_Blue.png',
                    })
                    metrics = figures['09_resultados_clasificador_metrica_todos_datasets.png']
                    self.assertEqual(len(metrics.axes), 4 * len(estimators))
                    self.assertEqual({ax.get_ylabel().lower() for ax in metrics.axes} - {''}, set(estimators))
                    for estimator in estimators:
                        tradeoff = figures[f'09_global_features_runtime_tradeoff_{estimator}.png']
                        expected_colors = [bar.get_facecolor()[:3] for bar in metrics.axes[0].patches]
                        self.assertEqual(len(set(expected_colors)), 8)
                        for ax in tradeoff.axes:
                            self.assertEqual([bar.get_facecolor()[:3] for bar in ax.patches], expected_colors)
                        palette = study.muted_color_palette(8)
                        np.testing.assert_array_equal(expected_colors, [
                            palette[list(study.SUPPORTED_TRANSFER_FUNCTIONS).index(tf)]
                            for tf in sorted(study.SUPPORTED_TRANSFER_FUNCTIONS)
                        ])
                        curve = figures[f'05_convergence_por_dataset_{estimator}.png']
                        lines = curve.axes[0].lines
                        self.assertEqual(len(lines), 8)
                        self.assertEqual(len({line.get_marker() for line in lines}), 8)
                        for line, tf in zip(lines, sorted(study.SUPPORTED_TRANSFER_FUNCTIONS)):
                            np.testing.assert_array_equal(
                                line.get_ydata(), records[f'MaCRO-DE_{tf.upper()}_{estimator.upper()}']['Curve']
                            )
                    blue_metrics = figures['TransferFunctions_ClassificationMetrics_Blue.png']
                    blue_tradeoff = figures['TransferFunctions_FeaturesRuntimeTradeoff_Blue.png']
                    labels = [f'MaCRO-DE {tf.upper()}' for tf in sorted(study.SUPPORTED_TRANSFER_FUNCTIONS)]
                    self.assertEqual(len(blue_metrics.axes), 4)
                    self.assertEqual({ax.get_ylabel() for ax in blue_metrics.axes} - {''}, {'KNN'})
                    for ax in blue_metrics.axes + blue_tradeoff.axes[:1]:
                        self.assertEqual([label.get_text() for label in ax.get_xticklabels()], labels)
                    blue_colors = [bar.get_facecolor()[:3] for bar in blue_metrics.axes[0].patches]
                    self.assertEqual(len(set(blue_colors)), 8)
                    luminance = np.asarray(blue_colors) @ np.array([.2126, .7152, .0722])
                    self.assertLess(max(luminance[:4]), min(luminance[4:]))
                    for red, green, blue in blue_colors:
                        self.assertLess(red, green)
                        self.assertLess(green, blue)
                    for ax in blue_metrics.axes + blue_tradeoff.axes:
                        np.testing.assert_array_equal(
                            [bar.get_facecolor()[:3] for bar in ax.patches], blue_colors,
                        )
                    np.testing.assert_array_equal(
                        [patch.get_facecolor()[:3] for patch in blue_metrics.legends[0].get_patches()],
                        blue_colors,
                    )
                    for ax, alpha, hatch in zip(blue_tradeoff.axes, [.85, .45], [None, '///']):
                        for bar in ax.patches:
                            self.assertEqual(bar.get_alpha(), alpha)
                            self.assertEqual(bar.get_hatch(), hatch)
                    np.testing.assert_array_equal(
                        [bar.get_height() for bar in blue_tradeoff.axes[0].patches], [6, 7, 8, 9, 2, 3, 4, 5],
                    )
                    for name, fig in figures.items():
                        if name == historical.METRICS_FILENAME:
                            continue  # An SVM-only fixture has no KNN measurements.
                        self.assertFalse(any(t.get_text() == 'No data' for ax in fig.axes for t in ax.texts))
                finally:
                    for fig in figures.values():
                        plt.close(fig)

    def test_missing_classifier_cache_is_allowed_only_for_transfer_mode(self):
        args = argparse.Namespace(experiment_mode='transfer_functions', estimators=['knn', 'svm'])
        payload = {'variant': {'Estimator': 'knn', 'AccMean': 80}}
        with patch.object(study, 'load_best_cache_payload', side_effect=lambda p, d, e, s: payload if e == 'knn' else None):
            self.assertEqual(study.load_results_from_cache(None, args, ['Dataset'], 'sig'), {'Dataset': payload})
            args.experiment_mode = 'full'
            with self.assertRaises(FileNotFoundError):
                study.load_results_from_cache(None, args, ['Dataset'], 'sig')
        args.experiment_mode = 'transfer_functions'
        with patch.object(study, 'load_best_cache_payload', return_value=None):
            with self.assertRaises(FileNotFoundError):
                study.load_results_from_cache(None, args, ['Dataset'], 'sig')


if __name__ == '__main__':
    unittest.main()
