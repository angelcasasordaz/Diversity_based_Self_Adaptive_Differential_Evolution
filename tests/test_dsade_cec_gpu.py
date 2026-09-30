"""Focused CEC GPU wiring, strict validation, and seeded numerical parity.

Transport tests use an explicitly emulated device; real CUDA tests skip when
unavailable. No experiment runners or artifact writers are invoked.
"""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mealpy import FloatVar

import diversity_gpu_batching as batching
from dsade_cec_optimizer import DSADE
from numerical_backend import GPUBackendError, NumPyBackend, create_numerical_backend
from optimizer_factory import (build_optimizer, optimizer_constructor_kwargs,
                               resolve_optimizer, select_execution_strategy)
from optimizer_interceptor import Workload


FIVE = ('DSADE', 'DSADE-CEC', 'MaCRO-DE', 'MaCRO-DE-t', 'MaCRO-DE-t-v2')


class EmulatedDevice(NumPyBackend):
    """Run the device namespace equations on NumPy for transport regression tests."""
    @property
    def uses_gpu(self):
        return True

    def to_cpu(self, value):
        return np.asarray(value)

    def scalar(self, value):
        return float(value)


def remote_batcher():
    owner = object.__new__(batching.DiversityMathBatcher)
    owner.backend, owner.remote = EmulatedDevice(), None
    client = object.__new__(batching.DiversityMathBatcher)
    calls = []
    class Proxy:
        def call(self, operation, *args):
            calls.append(operation)
            return getattr(owner, operation)(*args)
    client.backend, client.remote = None, Proxy()
    return client, calls


def solve(model, mode, dims):
    model.solve(dict(bounds=FloatVar(lb=[-5.]*dims, ub=[5.]*dims),
                     minmax='min', obj_func=lambda x: float(np.sum(x*x)), log_to=None),
                mode=mode, seed=717)
    return model


class CECGPUWiringTests(unittest.TestCase):
    def test_five_optimizer_validation_and_strategy_with_available_gpu(self):
        import main_best as study
        gpu = SimpleNamespace(uses_gpu=True, fallback_reason=None)
        args = SimpleNamespace(compute_device='gpu')
        output = StringIO()
        with patch.object(study, 'GPU_OWNER_BACKEND', gpu), redirect_stdout(output):
            study.validate_comparison_backend(args, list(FIVE))
        self.assertIn('Comparison backend validation: PASSED', output.getvalue())
        self.assertIn('Optimizers validated: 5/5', output.getvalue())
        for name in FIVE:
            resolved = resolve_optimizer(name)
            self.assertTrue(resolved.capability.supports_gpu)
            strategy = select_execution_strategy('gpu', resolved, Workload(10, 6, 2), gpu)
            self.assertEqual(strategy.optimizer_compute_device, 'gpu')
        self.assertEqual(resolve_optimizer('DSADE').optimizer_class.__module__, 'dsade_awad_optimizer')

    def test_validation_and_construction_still_reject_unavailable_gpu(self):
        import main_best as study
        with patch.object(study, 'GPU_OWNER_BACKEND', None), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'backend validation failed'):
                study.validate_comparison_backend(SimpleNamespace(compute_device='gpu'), list(FIVE))
        with patch.object(batching, '_REMOTE_GPU_CONNECTION', None), \
                patch.object(batching, '_LOCAL_GPU_WORKER_BACKEND', None), \
                patch.object(batching, 'ComputeBackend', return_value=NumPyBackend()):
            with self.assertRaises(GPUBackendError):
                DSADE(epoch=2, pop_size=10, compute_device='gpu')

    def test_factory_propagates_gpu_settings_and_service_executes_kernels(self):
        settings = SimpleNamespace(epochs=2, pop_size=10, optimizer_compute_device='gpu',
                                   gpu_device_id=2, gpu_memory_fraction=.7)
        kwargs = optimizer_constructor_kwargs('DSADE-CEC', settings)
        self.assertEqual(kwargs['compute_device'], 'gpu')
        client, calls = remote_batcher()
        with patch('dsade_cec_optimizer.DiversityMathBatcher', return_value=client) as constructor:
            model = build_optimizer('DSADE-CEC', settings)
        constructor.assert_called_once_with('gpu', 2, .7)
        solve(model, 'single', 3)
        self.assertTrue({'awad', 'macro_de_t_covariance', 'mutate', 'crossover'}.issubset(calls))
        # The directly exposed covariance helper also dispatches to the owner.
        model._safe_cov_inv(model._positions(model.pop))
        self.assertIn('covariance_inverse', calls)
        self.assertTrue(all(isinstance(agent.solution, np.ndarray) for agent in model.pop))

    def test_seeded_service_path_matches_cpu_in_both_modes(self):
        for mode in ('single', 'swarm'):
            for dims in (1, 6):
                with self.subTest(mode=mode, dims=dims):
                    cpu = solve(DSADE(epoch=4, pop_size=10), mode, dims)
                    client, _ = remote_batcher()
                    with patch('dsade_cec_optimizer.DiversityMathBatcher', return_value=client):
                        gpu = solve(DSADE(epoch=4, pop_size=10, compute_device='gpu'), mode, dims)
                    self.assert_models_close(cpu, gpu)

    def assert_models_close(self, cpu, gpu):
        for attr in ('div_awad_hist', 'div_norm_hist', 'pcr_hist', 'fmean_hist'):
            np.testing.assert_allclose(getattr(cpu, attr), getattr(gpu, attr), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(cpu._positions(cpu.pop), gpu._positions(gpu.pop), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(cpu.history.list_global_best_fit, gpu.history.list_global_best_fit,
                                   rtol=1e-10, atol=1e-12)
        self.assertEqual(cpu.generator.bit_generator.state, gpu.generator.bit_generator.state)
        # Greedy CEC survivors cannot degrade best fitness, in either mode.
        self.assertTrue(np.all(np.diff(gpu.history.list_global_best_fit) <= 0))

    def test_service_failure_never_falls_back_to_cpu(self):
        client, _ = remote_batcher()
        with patch('dsade_cec_optimizer.DiversityMathBatcher', return_value=client):
            model = DSADE(epoch=2, pop_size=10, compute_device='gpu')
        with patch.object(client.remote, 'call', side_effect=batching.GPUServiceUnavailable('offline')):
            with self.assertRaisesRegex(GPUBackendError, 'offline'):
                model._awad(np.ones((10, 3)), None, None)

    def test_real_cuda_seeded_parity_and_five_optimizer_validation(self):
        backend = create_numerical_backend('gpu')
        if not backend.uses_gpu:
            self.skipTest(backend.fallback_reason)
        import main_best as study
        with patch.object(study, 'GPU_OWNER_BACKEND', backend), redirect_stdout(StringIO()):
            study.validate_comparison_backend(SimpleNamespace(compute_device='gpu'), list(FIVE))
        for mode in ('single', 'swarm'):
            for dims in (1, 6):
                with self.subTest(mode=mode, dims=dims):
                    cpu = solve(DSADE(epoch=4, pop_size=10), mode, dims)
                    gpu = solve(DSADE(epoch=4, pop_size=10, compute_device='gpu'), mode, dims)
                    self.assertTrue(gpu.math_batcher.uses_gpu)
                    self.assert_models_close(cpu, gpu)


if __name__ == '__main__':
    unittest.main()
