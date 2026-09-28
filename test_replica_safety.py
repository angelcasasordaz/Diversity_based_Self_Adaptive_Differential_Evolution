"""Regression checks for destructive replica cleanup boundaries."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from full_replica_report import (
    REPLICA_DIRECTORIES, recreate_replica_directories, report_guard,
    validate_replica_destination,
)
from full_rep1_report import cleanup_duplicates


class ReplicaCleanupSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.targets = [self.root / name for name in REPLICA_DIRECTORIES]
        for path in self.targets:
            (path / "nested").mkdir(parents=True)
            (path / "nested/report.txt").write_text("old replica")
        self.original = self.root / "Results/EXP627/full/cache/original.pkl"
        self.original.parent.mkdir(parents=True)
        self.original.write_bytes(b"immutable")

    def test_only_four_duplicate_directories_deleted_under_guard(self):
        with report_guard(tuple(self.targets)):
            deleted = cleanup_duplicates(self.root)
        self.assertEqual(set(deleted), set(REPLICA_DIRECTORIES))
        for path in self.targets:
            self.assertFalse(path.exists())
        self.assertEqual(self.original.read_bytes(), b"immutable")

    def test_retired_directories_cannot_be_recreated(self):
        with self.assertRaises(ValueError):
            recreate_replica_directories(self.root, *self.targets)

    def test_unsafe_paths_never_reach_rmtree(self):
        paths = [self.root, self.root / "Results/EXP627",
                 self.root / "Results/EXP627/full", self.root / "Figures/EXP627/full",
                 self.targets[0] / "nested", self.targets[0].with_name("full_replica_other")]
        with patch("full_replica_report.shutil.rmtree") as delete:
            for path in paths:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    recreate_replica_directories(self.root, path)
            delete.assert_not_called()

    def test_checkpoint_aborts_whole_batch(self):
        (self.targets[1] / "saved.pkl").write_bytes(b"protected")
        with patch("full_replica_report.shutil.rmtree") as delete:
            with self.assertRaises(ValueError):
                cleanup_duplicates(self.root)
            delete.assert_not_called()
        self.assertTrue((self.targets[0] / "nested/report.txt").exists())

    def test_nested_symlink_is_rejected(self):
        (self.targets[0] / "link").symlink_to(self.original.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            validate_replica_destination(self.root, self.targets[0])

    def test_redirected_destination_is_rejected(self):
        other = self.root / "other"
        (other / "Results/EXP627").mkdir(parents=True)
        (other / REPLICA_DIRECTORIES[0]).symlink_to(self.targets[0], target_is_directory=True)
        with self.assertRaises(ValueError):
            validate_replica_destination(other, other / REPLICA_DIRECTORIES[0])

    def test_descriptor_relative_original_delete_is_blocked(self):
        import os
        fd = os.open(self.original.parent, os.O_RDONLY)
        try:
            with report_guard(tuple(self.targets)), self.assertRaises(PermissionError):
                os.unlink(self.original.name, dir_fd=fd)
        finally:
            os.close(fd)
        self.assertEqual(self.original.read_bytes(), b"immutable")


if __name__ == "__main__":
    unittest.main()
