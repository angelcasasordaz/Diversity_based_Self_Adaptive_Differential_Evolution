"""Safety regressions for the explicitly authorized FULL_REP1 reporting tree."""
import tempfile
import unittest
from pathlib import Path

from reporting.exp627_figures import validate_output_destination, STEMS
from reporting.exp627_statistics import STEMS as STAT_STEMS, EXPECTED_RES


class FullRep1ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fig = self.root / "Figures/EXP627/full_rep1"
        self.res = self.root / "Results/EXP627/full_rep1"
        for base in (self.fig, self.res):
            (base / "statistics").mkdir(parents=True)
        for stem in STEMS:
            (self.fig / f"{stem}.png").write_bytes(b"report")
        for stem in STAT_STEMS:
            (self.fig / "statistics" / f"{stem}.png").write_bytes(b"report")
        for name in EXPECTED_RES:
            (self.res / "statistics" / name).write_text("report")
        (self.res / "validation.json").write_text("{}")
        (self.res / "latex_tables").mkdir()
        for table in ("overall", "datasets_1", "datasets_2"):
            for variant in ("mean_std", "full_stats"):
                (self.res / "latex_tables" / f"table_{table}_{variant}.tex").write_text("table")
        (self.res / "latex_tables/latex_table_validation.txt").write_text("validated")

    def test_existing_complete_reporting_tree_is_repeatably_accepted(self):
        before = {p: p.read_bytes() for base in (self.fig, self.res) for p in base.rglob("*") if p.is_file()}
        for _ in range(2):
            for base in (self.fig, self.res):
                self.assertEqual(validate_output_destination(self.root, base), base)
        self.assertTrue(all(p.read_bytes() == contents for p, contents in before.items()))

    def test_unknown_and_nested_directories_are_rejected(self):
        for base in (self.fig, self.res):
            for relative in ("cache", "other", "statistics/cache", "statistics/nested"):
                path = base / relative
                path.mkdir()
                try:
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        validate_output_destination(self.root, base)
                finally:
                    path.rmdir()
        (self.fig / "latex_tables").mkdir()
        with self.assertRaises(ValueError):
            validate_output_destination(self.root, self.fig)

    def test_scientific_and_unexpected_files_are_rejected_at_every_depth(self):
        for folder in (self.fig, self.res, self.fig / "statistics", self.res / "statistics", self.res / "latex_tables"):
            base = self.fig if self.fig == folder or self.fig in folder.parents else self.res
            for name in ("saved.pkl", "saved.PKL", "saved.pickle", "saved.ckpt", "checkpoint.json", "cache.csv", "unknown.txt", "stat_fig1_average_rank.pdf"):
                path = folder / name
                path.write_bytes(b"protected")
                try:
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        validate_output_destination(self.root, base)
                finally:
                    path.unlink()

    def test_symlinks_are_rejected_even_with_authorized_names(self):
        for path, base in ((self.res / "validation.json", self.res),
                           (self.fig / "statistics" / f"{STAT_STEMS[0]}.png", self.fig)):
            path.unlink()
            path.symlink_to(self.root / "missing")
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_output_destination(self.root, base)
            path.unlink()
        empty = self.root / "empty"
        empty.mkdir()
        latex = self.res / "latex_tables"
        for p in latex.iterdir():
            p.unlink()
        latex.rmdir()
        latex.symlink_to(empty, target_is_directory=True)
        with self.assertRaises(ValueError):
            validate_output_destination(self.root, self.res)

    def test_paths_outside_exact_reporting_roots_are_rejected(self):
        for kind in ("Figures", "Results"):
            for mode in ("full", "full/cache", "ablation", "sensitivity", "sensitivity_weights", "full_rep1/../full", "full_rep1/statistics"):
                with self.subTest(kind=kind, mode=mode), self.assertRaises(ValueError):
                    validate_output_destination(self.root, self.root / kind / "EXP627" / mode)


if __name__ == "__main__":
    unittest.main()
