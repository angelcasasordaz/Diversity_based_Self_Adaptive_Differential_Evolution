"""Append-only report allocation and publication in temporary trees."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reporting.core import next_report_version, staged_version, sha256


class ReportVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def pair(self, n, exp=913):
        for kind in ('Figures', 'Results'):
            base = self.root / kind / f'EXP{exp}' / f'full_rep{n}'
            base.mkdir(parents=True)
            (base / 'historical.txt').write_text(f'{kind} {n}')

    def test_a_b_c_d_e_versions_preserve_bytes_and_selected_exp(self):
        self.assertEqual(next_report_version(self.root, 913), 1)
        self.pair(1)
        self.assertEqual(next_report_version(self.root, 913), 2)
        self.pair(2)
        self.assertEqual(next_report_version(self.root, 913), 3)
        before = {p: sha256(p) for p in self.root.rglob('*.txt')}
        with staged_version(self.root, 913) as (n, fig, res, finals):
            self.assertEqual(n, 3)
            (fig / 'new.txt').write_text('new')
            (res / 'new.txt').write_text('new')
            self.assertFalse(any(p.exists() for p in finals))
        self.assertTrue(all(p.is_dir() for p in finals))
        self.assertEqual(before, {p: sha256(p) for p in before})
        self.assertFalse((self.root / 'Results/EXP914').exists())

    def test_maximum_not_count_and_staging_ignored(self):
        self.pair(1); self.pair(7)
        (self.root / 'Results/EXP913/.report-staging-old').mkdir()
        self.assertEqual(next_report_version(self.root, 913), 8)

    def test_unpaired_versions_fail_before_writes(self):
        (self.root / 'Figures/EXP913/full_rep8').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            with staged_version(self.root, 913):
                self.fail('must not allocate')
        self.assertFalse((self.root / 'Results').exists())

    def test_conflicting_manifest_identity_is_preserved_and_rejected(self):
        self.pair(1)
        path = self.root / 'Results/EXP913/full_rep1/validation.json'
        contents = '{"experiment_id": 914, "report_version": 1}'
        path.write_text(contents)
        with self.assertRaisesRegex(ValueError, 'Conflicting report manifest'):
            next_report_version(self.root, 913)
        self.assertEqual(path.read_text(), contents)

    def test_conflicts_symlinks_and_unsafe_roots(self):
        for kind in ('Figures', 'Results'):
            (self.root / kind / 'EXP913').mkdir(parents=True)
        p = self.root / 'Figures/EXP913/full_rep1'
        p.write_text('file')
        with self.assertRaises(ValueError): next_report_version(self.root, 913)
        p.unlink(); p.symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError): next_report_version(self.root, 913)
        p.unlink()
        with self.assertRaises(ValueError): next_report_version(self.root / 'child/..', 913)
        self.pair(1)
        (p / 'link').symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError): next_report_version(self.root, 913)

    def test_failed_generation_and_second_publication_leave_no_new_version(self):
        self.pair(1)
        with self.assertRaises(RuntimeError):
            with staged_version(self.root, 913) as (_, fig, res, _):
                (res / 'partial.txt').write_text('partial')
                raise RuntimeError('export failed')
        from reporting import core
        real = core._rename_new
        def fail_second(source, target):
            if 'Results' in target.parts and target.name == 'full_rep2':
                raise OSError('second publication failed')
            real(source, target)
        with patch.object(core, '_rename_new', side_effect=fail_second), self.assertRaises(OSError):
            with staged_version(self.root, 913): pass
        self.assertEqual(next_report_version(self.root, 913), 2)
        self.assertFalse(list(self.root.rglob('.report-staging-*')))
        self.assertFalse(list(self.root.rglob('.report-allocation.lock')))

    def test_no_replace_even_if_destination_appears_during_generation(self):
        with self.assertRaises(OSError):
            with staged_version(self.root, 913) as (_, fig, res, finals):
                finals[0].mkdir()
                (finals[0] / 'concurrent.txt').write_text('preserve')
        self.assertEqual((finals[0] / 'concurrent.txt').read_text(), 'preserve')
        self.assertFalse(finals[1].exists())
