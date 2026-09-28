"""Exercise the common save policy used across experiment and report modes."""
import ast
from contextlib import ExitStack
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import main_best as m


class SavePolicyTests(unittest.TestCase):
    def test_all_production_figure_writes_use_the_png_sink(self):
        writes = []
        root = Path(__file__).resolve().parents[1]
        for source in (*root.glob("*.py"), *(root / "reporting").glob("*.py")):
            if source.name.startswith("test_"):
                continue
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    self.assertNotIn(name, {"PdfPages", "print_pdf"}, source.name)
                    if name == "savefig":
                        writes.append((source.name, node))
        self.assertEqual(len(writes), 1)
        filename, call = writes[0]
        self.assertEqual(filename, "main_best.py")
        self.assertEqual(next(ast.literal_eval(k.value) for k in call.keywords if k.arg == "format"), "png")

    def test_framework_and_historical_chart_savers_ignore_pdf_requests(self):
        import historical_transfer_plots as historical
        with tempfile.TemporaryDirectory() as folder:
            for name, saver in (("framework", m._save_chart), ("historical", historical._save_chart)):
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.plot([0, 1])
                saver(fig, folder, f"{name}.pdf")
            self.assertEqual({p.name for p in Path(folder).iterdir()}, {"framework.png", "historical.png"})

    def test_legacy_pdf_requests_produce_only_600_dpi_png(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fig, ax = plt.subplots(figsize=(1, 1), dpi=72)
            ax.plot([0, 1])
            try:
                m._save_figure(fig, root / "legacy.png", save_pdf=True)
                m._save_figure(fig, root / "explicit.pdf", save_pdf=True, format="pdf", dpi=72)
                self.assertEqual({p.name for p in root.iterdir()}, {"legacy.png", "explicit.png"})
                with Image.open(root / "explicit.png") as png:
                    self.assertEqual(png.format, "PNG")
                    self.assertAlmostEqual(png.info["dpi"][0], 600, delta=0.1)
                historical = root / "legacy.pdf"
                historical.write_bytes(b"historical PDF")
                m._rename_chart_exports(folder, "legacy.png", "renamed.png", save_pdf=True)
                self.assertEqual(historical.read_bytes(), b"historical PDF")
                self.assertFalse((root / "renamed.pdf").exists())
                m._rename_chart_exports(folder, "renamed.pdf", "final.pdf", save_pdf=True)
                self.assertTrue((root / "final.png").is_file())
                self.assertFalse((root / "final.pdf").exists())
            finally:
                plt.close(fig)


class ReportOrchestrationPolicyTests(unittest.TestCase):
    def test_report_only_orchestration_writes_pngs_without_full_size_rendering(self):
        """Use stored inputs and tiny test figures; never redraw the publication package."""
        import reporting.exp627_figures as report
        import reporting.exp627_statistics as statistics
        source = Path(__file__).resolve().parents[1]
        cache = source / "Results/EXP627/full/cache"
        reference = source / "Results/EXP627/full/RESUMEN_GRAFICAS_EXP627.csv"
        if not cache.is_dir() or not reference.is_file():
            self.skipTest("Saved EXP627 FULL inputs are required for orchestration validation")

        def tiny_figure(*args, **kwargs):
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.plot([0, 1])
            return fig

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            (root / "Figures/EXP627").mkdir(parents=True)
            (root / "Results/EXP627/full").mkdir(parents=True)
            shutil.copytree(cache, root / "Results/EXP627/full/cache")
            shutil.copy2(reference, root / "Results/EXP627/full" / reference.name)
            for generator in ("general_figure", "radar_figure", "heatmap_figure", "precision_figure",
                              "boxplot_figure", "violin_figure", "convergence_figure", "tradeoff_figure"):
                stack.enter_context(patch.object(report, generator, tiny_figure))
            stack.enter_context(patch.object(statistics, "figures", lambda *args: (tiny_figure() for _ in range(4))))
            for name in ("_run_single", "execute_pending_runs", "build_optimizer", "save_cache", "configure_compute_backend"):
                stack.enter_context(patch.object(m, name, side_effect=AssertionError("Scientific execution forbidden")))
            stack.enter_context(patch.object(sys, "argv", ["main_best.py", "--full-rep1-report-only", "--output-root", folder]))
            m.main()
            fig = root / "Figures/EXP627/full_rep1"
            self.assertEqual({p.name for p in fig.glob("*.png")}, {f"{s}.png" for s in report.STEMS})
            self.assertEqual({p.name for p in (fig / "statistics").iterdir()}, {f"{s}.png" for s in statistics.STEMS})
            self.assertFalse(list(root.rglob("*.pdf")))
            import json
            validation = json.loads((root / "Results/EXP627/full_rep1/validation.json").read_text())
            self.assertEqual(validation["optimization_calls"], 0)
            self.assertEqual(validation["statistical_pdfs"], 0)

    def test_all_statistical_figures_are_png_only_even_for_pdf_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            fig = plt.figure(figsize=(1, 1), dpi=72)
            try:
                path = Path(folder) / "statistics.png"
                with self.assertRaises(ValueError):
                    m._save_statistical_figure(fig, path, statistical_figure="standard")
                from reporting.exp627_statistics import STEMS, IDS
                for stem, figure_id in zip(STEMS, IDS):
                    for save_pdf in (True, False):
                        m._save_statistical_figure(
                            fig, Path(folder) / f"{stem}.pdf", statistical_figure=figure_id,
                            save_pdf=save_pdf, format="pdf", dpi=72,
                        )
                self.assertEqual({p.name for p in Path(folder).iterdir()},
                                 {f"{stem}.png" for stem in STEMS})
                for png_path in Path(folder).iterdir():
                    with Image.open(png_path) as png:
                        self.assertEqual(png.format, "PNG")
                        self.assertAlmostEqual(png.info["dpi"][0], 600, delta=0.1)
            finally:
                plt.close(fig)
