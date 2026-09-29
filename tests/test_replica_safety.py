"""Cache-only execution and mutation boundaries; no legacy cleanup entrypoint."""
import os
from pathlib import Path
import tempfile
import unittest

from reporting.core import report_guard


class ReportGuardTests(unittest.TestCase):
    def test_writes_and_descriptor_deletion_outside_stage_are_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage = root / 'stage'; stage.mkdir()
            original = root / 'original.pkl'; original.write_bytes(b'immutable')
            fd = os.open(root, os.O_RDONLY)
            try:
                with report_guard((stage,)):
                    (stage / 'report.txt').write_text('allowed')
                    with self.assertRaises(PermissionError): original.write_bytes(b'bad')
                    with self.assertRaises(PermissionError): os.unlink(original.name, dir_fd=fd)
                    with self.assertRaises(PermissionError): (stage / 'new.pkl').write_bytes(b'bad')
                self.assertEqual(original.read_bytes(), b'immutable')
            finally: os.close(fd)

    def test_scientific_execution_blocked_before_function_body(self):
        calls = []
        def execute_pending_runs(): calls.append('forbidden')
        with report_guard(()) as guard:
            with self.assertRaises(RuntimeError): execute_pending_runs()
        self.assertEqual(calls, [])
        self.assertEqual(guard['optimization_calls'], 1)
