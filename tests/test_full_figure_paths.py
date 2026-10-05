"""New FULL figure routing with temporary exports and no optimization."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import main_best as study
from figure_layout import full_figure_category, main_figure_names
from tests.test_generic_figures import report_fixture
from reporting import figures


class FullFigurePathsTests(unittest.TestCase):
    def test_numbered_exports_use_root_individual_statistics_and_preserve_old_figures(self):
        report = report_fixture(("DataClass", "FeatureEnvy"), ("MaCRO-DE-t", "DE"), ("knn",))
        args = report.args
        args.epochs = 3
        before = deepcopy(report.results)
        summary = study.generate_summary_dataframe(report.results, args)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "radar_6smells_grid_svm.png"
            old.write_bytes(b"historical figure")
            def lightweight_save(fig, target, **kwargs):
                Path(target).with_suffix(".png").write_bytes(b"new figure")
            with patch.object(study, "_save_figure", side_effect=lightweight_save):
                generated = study.generate_seven_global_charts(summary, report.results, str(root), args.optimizers, args)
            self.assertEqual({name for name in generated if "/" not in name},
                             set(main_figure_names("knn")))
            for dataset in report.datasets:
                for kind in ("radar", "convergence", "features_runtime"):
                    self.assertIn(f"individual/{kind}_{dataset}_knn.png", generated)
            self.assertIn("individual/generic_convergence_knn.png", generated)
            self.assertEqual([name for name in generated if "heatmap" in name],
                             ["06_heatmap_accuracy_knn.png"])
            self.assertFalse(list((root / "individual").glob("*heatmap*.png")))
            self.assertTrue(all((root / name).is_file() for name in generated))
            self.assertEqual(old.read_bytes(), b"historical figure")
        np.testing.assert_equal(report.results, before)

    def test_full_make_paths_creates_sibling_directories_without_source_writes(self):
        from tests.test_full_cache_reuse import arguments
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory), 629)
            paths = study.make_paths(args)
            self.assertTrue((Path(paths.fig_dir) / "individual").is_dir())
            self.assertTrue((Path(paths.fig_dir) / "statistics").is_dir())
            study.make_read_only_source_paths(args)
            self.assertFalse((Path(directory) / "Results/EXP627").exists())
            self.assertFalse((Path(directory) / "Figures/EXP627").exists())

    def test_main_distributions_stay_root_and_inference_stays_statistics(self):
        for name in ("generic_average_rank.png", "generic_holm_heatmap.png"):
            self.assertEqual(full_figure_category(name), "statistics")
        for name in ("generic_summary_knn.png", "generic_radar_svm.png", "radar_2datasets_grid_svm.png",
                     "generic_accuracy_boxplot_knn.png", "generic_recall_violin_rf.png",
                     "features_runtime_per_optimizer.png", "new_detail.png"):
            self.assertEqual(full_figure_category(name), "individual")
        for classifier in ("knn", "svm", "rf"):
            for name in main_figure_names(classifier):
                self.assertEqual(full_figure_category(name), "root")

    def test_macro_visual_identity_outlines_numeric_labels_and_run_values_are_preserved(self):
        report = report_fixture(("DataClass", "FeatureEnvy"), ("MaCRO-DE-t", "DE"), ("knn",))
        before = deepcopy(report.indexed)
        f1 = next(m for m in report.metrics if m.run_key == "F1Runs")
        heatmap = figures.heatmap_figure(report, "knn", f1)
        try:
            outline = heatmap.axes[0].patches[0]
            self.assertEqual(outline.get_edgecolor(), to_rgba("black"))
            self.assertFalse(outline.get_fill())
            self.assertEqual(len(heatmap.axes[0].texts), 4)
            self.assertEqual(heatmap.axes[0].get_yticklabels()[0].get_text(), "DSA-DE")
        finally:
            plt.close(heatmap)
        labels, values = figures.radar_values(report, "knn")
        for fig in (figures.radar_figure(report, "knn", labels, values),
                    figures.convergence_figure(report, "knn", report.datasets)):
            try:
                primary = next(line for line in fig.axes[0].lines if line.get_label() == "DSA-DE")
                self.assertEqual(primary.get_color(), "#0072B2")
                self.assertEqual(primary.get_linestyle(), "-")
                self.assertEqual(primary.get_marker(), "o")
                self.assertLessEqual(primary.get_linewidth(), 2.5)
                self.assertIn("DSA-DE", [t.get_text() for t in fig.legends[0].get_texts()])
            finally:
                plt.close(fig)
        tradeoff = figures.dataset_tradeoff_figure(report, "knn")
        try:
            self.assertTrue(any(p.get_hatch() == "///" for ax in tradeoff.axes for p in ax.patches))
            self.assertTrue(any(text.get_text().endswith("s") for ax in tradeoff.axes for text in ax.texts))
            self.assertTrue(any(text.get_fontweight() == "bold" for ax in tradeoff.axes for text in ax.texts))
        finally:
            plt.close(tradeoff)
        expected = np.concatenate([report.indexed[ds, "knn", "MaCRO-DE-t"]["AccRuns"] for ds in report.datasets]) / 100
        np.testing.assert_array_equal(figures.run_values(report, "knn", "AccRuns", 100)[0], expected)
        np.testing.assert_equal(report.indexed, before)

    def test_plot_classifier_changes_names_but_never_cache_identity(self):
        from tests.test_full_cache_reuse import arguments
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory), 629)
            signature = study.build_cache_signature(args, "MaCRO-DE-t", "DataClass", "knn", "vstf_01")
            args.plot_global_estimator = "svm"
            self.assertEqual(study.build_cache_signature(args, "MaCRO-DE-t", "DataClass", "knn", "vstf_01"), signature)
        report = report_fixture(("DataClass",), ("MaCRO-DE-t", "DE"), ("knn", "svm"))
        report.args.plot_global_estimator = "svm"
        self.assertEqual(figures.base_figure_names(report), main_figure_names("svm"))

    def test_normal_full_exports_statistics_in_sibling_folder_and_keeps_workbooks_in_results(self):
        from tests.test_generic_reporting import tiny_figures
        from tests.test_full_cache_reuse import arguments
        from contextlib import ExitStack
        report = report_fixture(("A", "B"), ("MaCRO-DE-t", "DE", "JADE"), ("knn",))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = arguments(Path(directory), 629, ("MaCRO-DE-t", "DE", "JADE"))
            paths = study.make_paths(args)
            targets = []
            stack.enter_context(patch.object(study, "export_global_excel", side_effect=lambda r, d, p: targets.append(Path(p)) or []))
            stack.enter_context(patch.object(study, "export_statistical_excel", side_effect=lambda *a: targets.append(Path(a[-1]))))
            stack.enter_context(patch.object(study, "export_friedman_analysis", side_effect=lambda *a: targets.append(Path(a[-1]))))
            stack.enter_context(patch("reporting.paper_tables.export_paper_tables", return_value="paper.xlsx"))
            stack.enter_context(patch.object(study, "generate_seven_global_charts", return_value=[]))
            stack.enter_context(patch.object(figures, "statistical_figures", tiny_figures))
            study.export_mode_outputs(paths, args, report.datasets, report.results)
            self.assertTrue((Path(paths.fig_dir) / "statistics/tiny.png").is_file())
            self.assertTrue((Path(paths.res_dir) / "statistics/statistical_summary.csv").is_file())
            self.assertTrue(all(path.parent == Path(paths.res_dir) for path in targets))


if __name__ == "__main__":
    unittest.main()
