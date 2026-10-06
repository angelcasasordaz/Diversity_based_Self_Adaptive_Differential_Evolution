"""Presentation-only report defaults, labels and skipped expensive detail builders."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import main_best as main
from figure_text import localize_figure, visible_text
from reporting import core, figures
from tests.test_generic_figures import report_fixture
from tests.test_generic_reporting import tiny_base_figures


class PrimaryReportTests(unittest.TestCase):
    def test_plain_run_is_unversioned_and_uses_global_presentation_defaults(self):
        for language in ('en', 'es'):
            with patch.object(sys, 'argv', ['main_best.py']), patch.object(main, 'FIGURE_LANGUAGE', language):
                args = main.parse_args()
            self.assertTrue(args.report_only)
            self.assertTrue(args.full_replica_report_only)
            self.assertEqual(args.figure_language, language)
            self.assertFalse(args.generate_individual_figures)

    def test_explicit_english_replica_remains_available_but_spanish_is_unversioned(self):
        for language in ('en', 'es'):
            with patch.object(sys, 'argv', ['main_best.py', '--full-rep1-report-only',
                                           '--figure-language', language]):
                args = main.parse_args()
            self.assertFalse(args.report_only)
            if language == 'es':
                with patch.object(core, 'run_figure_report', return_value='spanish') as report:
                    self.assertEqual(core.run_report(args), 'spanish')
                report.assert_called_once_with(args)

    def test_skip_individual_builders_and_export_only_nine_primary_figures(self):
        report = report_fixture(('A', 'B'), ('DE', 'PSO'), ('knn', 'svm'))
        report.args.generate_individual_figures = False
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(figures, 'base_publication_figures', tiny_base_figures), \
                patch.object(figures, 'publication_figures', side_effect=AssertionError('detail builder')) as generic, \
                patch.object(figures, 'per_dataset_figures', side_effect=AssertionError('dataset builder')) as detail, \
                patch.object(figures, 'save_png', side_effect=lambda fig, path: plt.close(fig)) as save, \
                redirect_stdout(StringIO()) as output:
            generated = []
            figures.generate(report, directory, generated=generated)
            self.assertEqual(len(generated), 9)
            self.assertEqual(save.call_count, 9)
            self.assertFalse((Path(directory) / 'individual').exists())
            self.assertIn('individual figures skipped', output.getvalue())
            generic.assert_not_called()
            detail.assert_not_called()

    def test_summary_classifier_row_labels_are_unique_and_bar_data_is_unchanged(self):
        report = report_fixture(('A', 'B'), ('DE', 'PSO'), ('knn', 'svm'))
        original = deepcopy(report.indexed)
        for language in ('en', 'es'):
            fig = figures.summary_figure(report)
            try:
                localize_figure(fig, language, protected=report.algorithms)
                labels = [ax.get_ylabel() for ax in fig.axes]
                self.assertEqual(labels, ['KNN', '', '', '', 'SVM', '', '', ''])
                metrics = {metric.run_key: metric for metric in report.metrics}
                for ci, classifier in enumerate(report.classifiers):
                    for mi, key in enumerate(('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs')):
                        expected = figures.metric_matrix(report, classifier, metrics[key]).mean(axis=1)
                        np.testing.assert_array_equal([bar.get_height() for bar in fig.axes[ci*4+mi].patches], expected)
            finally:
                plt.close(fig)
        np.testing.assert_equal(report.indexed, original)

    def test_statistics_visible_text_respects_language(self):
        for text in ('Average rank (1 = best)', 'Holm-adjusted p; 91-pair family',
                     'F1-Score: cached run mean per matched block',
                     'DSA-DE versus other algorithms (configured reference)'):
            self.assertEqual(visible_text(text, 'en'), text)
            self.assertNotEqual(visible_text(text, 'es'), text)


if __name__ == '__main__':
    unittest.main()
