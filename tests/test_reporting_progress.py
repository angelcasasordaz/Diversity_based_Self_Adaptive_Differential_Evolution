"""Progress visibility and the optimized cache-only guard's protections."""
from contextlib import redirect_stdout
from io import StringIO
import sys
import unittest

from reporting.core import report_guard, report_stage


class FlushedOutput(StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


class ReportingProgressTests(unittest.TestCase):
    def test_stage_flushes_before_work_and_after_completion(self):
        output = FlushedOutput()
        with redirect_stdout(output):
            with report_stage('Actual work'):
                self.assertIn('Actual work ...', output.getvalue())
                self.assertEqual(output.flushes, 1)
        self.assertIn('Actual work complete', output.getvalue())
        self.assertEqual(output.flushes, 2)

    def test_failure_is_visible_and_propagates(self):
        output = FlushedOutput()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, 'original error'):
            with report_stage('Validation'):
                raise ValueError('original error')
        self.assertIn('Validation failed', output.getvalue())
        self.assertNotIn('complete', output.getvalue())
        self.assertEqual(output.flushes, 2)

    def test_cached_guard_keeps_name_and_library_checks(self):
        for name, filename in (('_run_single', 'framework.py'), ('build_optimizer', 'factory.py'),
                               ('load_dataset', 'framework.py'), ('solve', 'mealpy/optimizer.py'),
                               ('fit', 'mafese/selector.py')):
            namespace = {'calls': []}
            exec(compile(f'def {name}():\n    calls.append("executed")', filename, 'exec'), namespace)
            with self.subTest(name=name), report_guard(()) as state:
                # Populate the allowed-code cache before visiting forbidden code.
                self.assertEqual(sum([1, 2]), 3)
                with self.assertRaisesRegex(RuntimeError, 'blocked scientific execution'):
                    namespace[name]()
            self.assertEqual(namespace['calls'], [])
            self.assertEqual(state['optimization_calls'], 1)

    def test_guard_restores_existing_profile(self):
        previous = sys.getprofile()
        with report_guard(()) as state:
            for _ in range(100):
                self.assertEqual(str(1), '1')
        self.assertIs(sys.getprofile(), previous)
        self.assertEqual(state['optimization_calls'], 0)


if __name__ == '__main__':
    unittest.main()
