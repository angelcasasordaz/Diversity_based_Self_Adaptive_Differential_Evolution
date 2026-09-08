"""Plot-only regressions; synthetic records never invoke an optimizer."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import main_best as study


class TransferFunctionPlotTests(unittest.TestCase):
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

                def capture(fig, out_dir, filename):
                    figures[filename] = fig
                    Path(out_dir, filename).touch()

                try:
                    with patch.object(study, '_save_chart', side_effect=capture), patch.object(
                        study, 'execute_pending_runs', side_effect=AssertionError('Optimization forbidden')
                    ):
                        names = study.generate_transfer_function_charts(
                            summary, results, tmp, args.optimizers, args
                        )
                    self.assertEqual(len(names), 1 + 3 * len(estimators))
                    metrics = figures['09_resultados_clasificador_metrica_todos_datasets.png']
                    self.assertEqual(len(metrics.axes), 4 * len(estimators))
                    self.assertEqual({ax.get_ylabel().lower() for ax in metrics.axes} - {''}, set(estimators))
                    for estimator in estimators:
                        curve = figures[f'05_convergence_por_dataset_{estimator}.png']
                        lines = curve.axes[0].lines
                        self.assertEqual(len(lines), 8)
                        self.assertEqual(len({line.get_marker() for line in lines}), 8)
                        for line, tf in zip(lines, sorted(study.SUPPORTED_TRANSFER_FUNCTIONS)):
                            np.testing.assert_array_equal(
                                line.get_ydata(), records[f'MaCRO-DE_{tf.upper()}_{estimator.upper()}']['Curve']
                            )
                    for fig in figures.values():
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
