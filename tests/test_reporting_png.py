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
        for source in (*root.glob("*.py"), *(root / "reporting").rglob("*.py")):
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
    def test_all_statistical_figures_are_png_only_even_for_pdf_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            fig = plt.figure(figsize=(1, 1), dpi=72)
            try:
                path = Path(folder) / "statistics.png"
                with self.assertRaises(ValueError):
                    m._save_statistical_figure(fig, path, statistical_figure="standard")
                IDS = ('average_rank', 'dsade_pairwise_holm', 'adjusted_pvalue_heatmap', 'f1_distribution_by_algorithm')
                STEMS = tuple(f'statistical_{i}' for i in range(len(IDS)))
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
