"""Display aliases must never become scientific or persisted identities."""
from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main_best as m
import full_plot_style
import historical_transfer_plots
from optimizer_factory import optimizer_acronym, optimizer_scientific_identity, resolve_optimizer
from reporting import figures, paper_tables
from tests.test_generic_figures import report_fixture


class PlotLabelOverrideTests(unittest.TestCase):
    def test_internal_labels_parsing_and_identity_are_unchanged(self):
        args = SimpleNamespace(optimizers=['MaCRO-DE-t', 'DE'], estimators=['knn', 'svm', 'rf'])
        self.assertEqual(resolve_optimizer('MaCRO-DE-t').canonical_name, 'MaCRO-DE-t')
        self.assertEqual(optimizer_acronym('MaCRO-DE-t'), 'MaCRO-DE-t')
        for estimator in args.estimators:
            expected = f'MACRO-DE-T_{estimator.upper()}'
            self.assertEqual(m.build_alg_label('MaCRO-DE-t', 'vstf_01', estimator, False, True), expected)
            self.assertEqual(m.build_legacy_alg_label('MaCRO-DE-t', 'vstf_01', estimator, False, True), expected)
            self.assertEqual(m.parse_result_label(expected, args)['method'], 'MaCRO-DE-t')
        self.assertEqual(optimizer_scientific_identity('MaCRO-DE-t', args)['canonical_name'], 'MaCRO-DE-t')

    def test_display_helpers_and_other_optimizers(self):
        for helper in (m.optimizer_display_label, historical_transfer_plots.optimizer_display_label,
                       full_plot_style.display_label):
            for name in ('MaCRO-DE-t', 'MACRO-DE-T'):
                self.assertEqual(helper(name), 'DSA-DE')
            for name in ('DE', 'JADE', 'PSO', 'MaCRO-DE', 'MaCRO-DE-t-v2'):
                self.assertEqual(helper(name), name)

    def test_plot_groups_keep_keys_and_transfer_variants(self):
        for helper in (m.prepare_plot_groups, historical_transfer_plots.prepare_plot_groups):
            df = pd.DataFrame({'Optimizer': ['MaCRO-DE-t', 'MaCRO-DE-t', 'DE'],
                               'TransferFunction': ['vstf_01', 'sstf_01', 'vstf_01']})
            original = df.copy(deep=True)
            plotted, groups, _, labels = helper(df, ['MaCRO-DE-t', 'DE'])
            self.assertEqual(set(groups), {'MaCRO-DE-t_VSTF_01', 'MaCRO-DE-t_SSTF_01', 'DE'})
            self.assertEqual(labels['MaCRO-DE-t_VSTF_01'], 'DSA-DE VSTF_01')
            self.assertEqual(labels['DE'], 'DE')
            pd.testing.assert_frame_equal(df, original)
            self.assertEqual(plotted.Optimizer.tolist(), df.Optimizer.tolist())

    def test_conflict_labels_in_plot_groups_ticks_and_legends(self):
        algorithms = ['DSADE', 'MaCRO-DE-t', 'DE']
        df = pd.DataFrame({'Optimizer': algorithms})
        for helper in (m.prepare_plot_groups, historical_transfer_plots.prepare_plot_groups):
            _, groups, _, labels = helper(df, algorithms)
            self.assertEqual(labels['MaCRO-DE-t'], 'DSA-DE (MaCRO-DE-t)')
            self.assertEqual(len(set(labels.values())), len(groups))
        report = report_fixture(('Only',), algorithms, ('knn',))
        fig = figures.summary_figure(report)
        try:
            expected = ['DSA-DE', 'DSA-DE (MaCRO-DE-t)', 'DE']
            self.assertEqual([t.get_text() for t in fig.axes[0].get_xticklabels()], expected)
            self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], expected)
            self.assertEqual(report.algorithms, algorithms)
        finally:
            plt.close(fig)

    def test_full_legend_without_conflict(self):
        report = report_fixture(('Only',), ('MaCRO-DE-t', 'DE'), ('knn',))
        fig = figures.convergence_figure(report, 'knn', report.datasets)
        try:
            self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], ['DSA-DE', 'DE'])
            self.assertIn(('Only', 'knn', 'MaCRO-DE-t'), report.indexed)
        finally:
            plt.close(fig)

    def test_manuscript_table_labels_leave_values_and_keys_unchanged(self):
        algorithms = ['DSADE', 'MACRO-DE-T', 'DE']
        values = np.arange(12, dtype=float).reshape(12, 1)
        tables = {'Overall': (values, values[2::4])}
        workbook = paper_tables.make_workbook(tables, algorithms,
            {'Overall': [('Accuracy', 'KNN', [], 'knn', paper_tables.METRICS[0])]},
            title='Test', description='Test')
        sheet = workbook['Overall']
        self.assertEqual(sheet['A7'].value, 'DSA-DE (MaCRO-DE-t)')
        self.assertEqual(sheet['A11'].value, 'DE')
        self.assertEqual(sheet['C7'].value, values[4, 0])
        self.assertEqual(algorithms, ['DSADE', 'MACRO-DE-T', 'DE'])
        workbook.close()
