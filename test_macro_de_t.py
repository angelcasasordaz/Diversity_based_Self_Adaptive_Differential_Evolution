"""Deterministic comparisons to the user's unmodified authoritative CEC files.

Set CEC_SOURCE_DIR when the source checkout is not the sibling directory below.
No benchmark code, datasets, or 30-run experiments are imported/executed.
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mealpy import FloatVar
from mealpy.utils.agent import Agent
from mealpy.utils.target import Target

from macro_de_t_optimizer import MaCRO_DE_t
from macro_de_t_backend import CECCovarianceKernels
from numerical_backend import NumPyBackend, create_numerical_backend
from optimizer_factory import (build_optimizer, resolve_optimizer, optimizer_acronym,
                               optimizer_constructor_kwargs, optimizer_scientific_identity)


def load_reference():
    directory = Path(os.environ.get('CEC_SOURCE_DIR', Path(__file__).resolve().parent.parent /
                     'Adaptive_Mahalanobis-Cholesky_Differential_Evolution_MaCRO_DE'))
    modules = {}
    with patch.dict(sys.modules):
        for name in ('compute_backend', 'de_ablation_base', 'de_mc_optimizer', 'de_mc_cf_optimizer'):
            path = directory / f'{name}.py'
            if not path.is_file():
                raise FileNotFoundError(f'Authoritative CEC source missing: {path}; set CEC_SOURCE_DIR')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    return modules['de_mc_cf_optimizer'].DE_MC_CF, modules['compute_backend'].ComputeBackend


Reference, ReferenceBackend = load_reference()


def sphere(x):
    return float(np.sum(x*x))


def configured(cls, population, seed=71, route=1., mode='single', q=.68):
    model = cls(epoch=4, pop_size=len(population), mahalanobis_q=q)
    model.problem = SimpleNamespace(n_dims=population.shape[1], lb=np.full(population.shape[1], -5.),
                                    ub=np.full(population.shape[1], 5.), minmax='min')
    model.generator = np.random.default_rng(seed)
    model.mode = mode
    model.correct_solution = lambda x: np.clip(x, -5., 5.)
    model.get_target = lambda x, counted=True: Target(sphere(x))
    model.pop = [Agent(solution=x.copy(), target=Target(sphere(x))) for x in population]
    model.initialize_variables()
    model.before_main_loop()
    model.div_norm_for_update = route
    return model


class Trace:
    def initialize_variables(self):
        super().initialize_variables()
        self.donors, self.mutants, self.masks, self.trials = [], [], [], []

    def _sample_mutation_indices(self, positions, idx):
        result = super()._sample_mutation_indices(positions, idx)
        self.donors.append(result.copy())
        return result

    def _binomial_crossover(self, parent, mutant):
        # Replay RNG from its saved state without consuming the model's stream.
        rng = np.random.default_rng()
        rng.bit_generator.state = self.generator.bit_generator.state
        j0 = rng.integers(0, self.problem.n_dims)
        mask = rng.random(self.problem.n_dims) <= self.cr
        mask[j0] = True
        self.masks.append(mask)
        self.mutants.append(mutant.copy())
        result = super()._binomial_crossover(parent, mutant)
        np.testing.assert_array_equal(result, self.correct_solution(np.where(mask, mutant, parent)))
        self.trials.append(result.copy())
        return result


class TracePort(Trace, MaCRO_DE_t): pass
class TraceReference(Trace, Reference): pass


class EquivalenceTests(unittest.TestCase):
    def test_forced_coordinate_and_inclusive_crossover_boundary(self):
        pop = np.zeros((5, 4))
        class FixedGenerator:
            def integers(self, low, high):
                return 2
            def random(self, size):
                return np.array([.9, .900001, 1., 1.])
        for cls in (MaCRO_DE_t, Reference):
            model = configured(cls, pop)
            model.generator = FixedGenerator()
            result = model._binomial_crossover(pop[0], np.array([1., 2., 3., 4.]))
            np.testing.assert_array_equal(result, [1., 0., 3., 0.])

    def test_diversity_cumulative_max_increase_decrease_and_zero(self):
        initial = np.random.default_rng(18).normal(size=(8, 3))
        for cls in (MaCRO_DE_t, Reference):
            model = configured(cls, initial)
            maximum = model.div_max_seen
            # Isolate the post-generation diversity update from selection.
            parent = cls.__mro__[1]
            with patch.object(parent, 'evolve', lambda self, epoch: None):
                for epoch, scale in enumerate([2., .1, 0., 3.], 1):
                    model.pop = [Agent(solution=scale*x) for x in initial]
                    awad = model._awad(scale*initial, None, None)
                    maximum = max(maximum, awad)
                    model.evolve(epoch)
                    self.assertEqual(model.div_max_seen, maximum)
                    self.assertEqual(model.div_norm_for_update,
                                     float(np.clip(awad/(maximum+model.EPSILON), 0., 1.)))

    def test_awad_safeguards(self):
        rng = np.random.default_rng(42)
        cases = [rng.normal(size=(9, 4)), np.ones((8, 3)), np.ones((1, 3)),
                 np.array([[0., 1.], [0., 1.], [3., 1.], [4., 1.], [5., 1.]]),
                 np.array([[0., 0.], [1e308, 0.], [-1e308, 0.]])]
        a, b = MaCRO_DE_t(epoch=4, pop_size=9), Reference(epoch=4, pop_size=9)
        a.initialize_variables(); b.initialize_variables()
        with np.errstate(all='ignore'):
            for pop in cases:
                np.testing.assert_equal(a._awad(pop, None, None), b._awad(pop, None, None))
        self.assertEqual(a._awad(np.ones((5, 3)), None, None), 0.)

    def test_routing_and_fallback_donor_eligibility(self):
        pop = np.random.default_rng(5).normal(size=(8, 3))
        for route, expected in [(1., 'close'), (.5, 'close'), (.499999, 'far'), (0., 'far')]:
            for selected in [np.array([], dtype=int), np.array([0, 1, 2]), np.array([0, 1, 2, 3]), np.arange(8)]:
                a, b = [configured(cls, pop, route=route) for cls in (MaCRO_DE_t, Reference)]
                self.assertEqual(a._route_for_diversity(route), expected)
                pools = (selected, np.arange(8)) if expected == 'close' else (np.arange(8), selected)
                for obj in (a, b):
                    obj._close_far_indices = lambda positions: pools
                for idx in range(8):
                    pa, pb = a._mutation_pool_indices(pop, idx), b._mutation_pool_indices(pop, idx)
                    np.testing.assert_array_equal(pa, pb)
                    valid = selected[selected != idx]
                    np.testing.assert_array_equal(pa, selected if len(valid) >= 3 else np.arange(8))
                    da, db = a._sample_mutation_indices(pop, idx), b._sample_mutation_indices(pop, idx)
                    np.testing.assert_array_equal(da, db)
                    self.assertNotIn(idx, da)
                    self.assertEqual(len(set(da)), 3)
                self.assertEqual(a.routing_counts, b.routing_counts)

    def test_covariance_cholesky_distances_and_pinv(self):
        a, b = CECCovarianceKernels(NumPyBackend()), ReferenceBackend('cpu')
        rng = np.random.default_rng(72)
        for pop in [rng.normal(size=(9, 4)), np.ones((8, 4)), rng.normal(size=(8, 1)), rng.normal(size=(2, 8, 4))]:
            dims = pop.shape[-1]
            np.testing.assert_array_equal(a.covariance(pop, dims), b.covariance(pop, dims))
            for method in ['cholesky', 'cholesky_solve']:
                for x, y in zip(a.mahalanobis_cpu(pop, dims, method), b.mahalanobis_cpu(pop, dims, method)):
                    np.testing.assert_array_equal(x, y)
            # A mismatched requested shape exercises the covariance shape safeguard.
            np.testing.assert_array_equal(a.covariance(pop, dims+1), b.covariance(pop, dims+1))
        sigma = np.array([[1., 2.], [2., 1.]])
        np.testing.assert_array_equal(a.covariance_inverse(sigma, 'cholesky'), np.linalg.pinv(sigma))
        pop = rng.normal(size=(8, 2))
        left, right = [configured(cls, pop) for cls in (MaCRO_DE_t, Reference)]
        np.testing.assert_array_equal(left._covariance_inverse(sigma), right._covariance_inverse(sigma))
        with patch.object(np.linalg, 'cholesky', side_effect=np.linalg.LinAlgError):
            np.testing.assert_array_equal(a.mahalanobis_distances(pop, 2), b.mahalanobis_distances(pop, 2))
        for q in [.1, .68, .99]:
            left, right = [configured(cls, pop, q=q) for cls in (MaCRO_DE_t, Reference)]
            self.assertEqual(left._mahalanobis_threshold(), right._mahalanobis_threshold())
            for x, y in zip(left._close_far_indices(pop), right._close_far_indices(pop)):
                np.testing.assert_array_equal(x, y)

    def test_generations_mutation_masks_survivors_and_normalization(self):
        for mode in ['single', 'swarm']:
            for seed in [3, 71, 919]:
                for route in [1., .499999]:
                    pop = np.random.default_rng(seed).uniform(-4., 4., (12, 6))
                    a, b = [configured(cls, pop, seed, route, mode) for cls in (TracePort, TraceReference)]
                    max_seen = a.div_max_seen
                    for epoch in range(1, 5):
                        frozen = a._positions(a.pop).copy()
                        a.evolve(epoch); b.evolve(epoch)
                        for attr in ['donors', 'mutants', 'masks', 'trials', 'div_awad_hist', 'div_norm_hist']:
                            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
                        for idx, donor in enumerate(a.donors[-12:]):
                            mutant = np.clip(frozen[donor[0]] + .5*(frozen[donor[1]]-frozen[donor[2]]), -5., 5.)
                            np.testing.assert_array_equal(a.mutants[-12+idx], mutant)
                            trial = a.trials[-12+idx]
                            expected = trial if sphere(trial) < sphere(frozen[idx]) else frozen[idx]
                            np.testing.assert_array_equal(a.pop[idx].solution, expected)
                        np.testing.assert_array_equal(a._positions(a.pop), b._positions(b.pop))
                        awad = a._awad(a._positions(a.pop), None, None)
                        max_seen = max(max_seen, awad)
                        self.assertEqual(a.div_max_seen, max_seen)
                        self.assertEqual(a.div_norm_for_update, float(np.clip(awad/(max_seen+a.EPSILON), 0, 1)))
                        self.assertEqual(a.routing_counts, b.routing_counts)

    def test_solve_lifecycle(self):
        for dims in [1, 6]:
            results = []
            for cls in [MaCRO_DE_t, Reference]:
                model = cls(epoch=4, pop_size=10)
                result = model.solve(dict(bounds=FloatVar(lb=[-5.]*dims, ub=[5.]*dims),
                                          minmax='min', obj_func=sphere, log_to=None), seed=717)
                results.append((result.solution, model.history.list_global_best_fit, model.div_norm_hist))
            for x, y in zip(*results):
                np.testing.assert_array_equal(x, y)

    def test_gpu_owner_transport(self):
        from diversity_gpu_batching import DiversityMathBatcher
        pop = np.random.default_rng(1).normal(size=(8, 4))
        local = DiversityMathBatcher('cpu')
        remote = object.__new__(DiversityMathBatcher)
        class Proxy:
            def call(self, method, *args):
                return getattr(local, method)(*args)
        remote.remote = Proxy()
        for x, y in zip(local.macro_de_t_covariance('close_far_indices', pop, 4, 3., 'cholesky'),
                        remote.macro_de_t_covariance('close_far_indices', pop, 4, 3., 'cholesky')):
            np.testing.assert_array_equal(x, y)

    def test_cuda_kernel_equivalence_when_available(self):
        backend = create_numerical_backend('gpu')
        if not backend.uses_gpu:
            self.skipTest(backend.fallback_reason)
        pop = np.random.default_rng(999).normal(size=(12, 6))
        port, ref = CECCovarianceKernels(backend), ReferenceBackend('gpu')
        for x, y in zip(port.mahalanobis_cpu(pop, 6), ref.mahalanobis_cpu(pop, 6)):
            np.testing.assert_array_equal(x, y)
        for x, y in zip(port.mahalanobis_cpu(pop, 6), ReferenceBackend('cpu').mahalanobis_cpu(pop, 6)):
            np.testing.assert_allclose(x, y, rtol=2e-11, atol=2e-12)


class IntegrationTests(unittest.TestCase):
    def test_registry_fixed_parameters_cache_revision(self):
        from test_sensitivity_optimizers import make_args
        import main_best as study
        args = make_args()
        args.optimizers = ['MaCRO-DE-t']
        args.dsade_pcr = .13
        for alias in ['MaCRO-DE-t', 'MACRO_DE_T', 'macrodet', 'MaCRODEt']:
            self.assertIs(resolve_optimizer(alias).optimizer_class, MaCRO_DE_t)
            self.assertEqual(optimizer_acronym(alias), 'MaCRO-DE-t')
            model = build_optimizer(alias, args)
            self.assertEqual((model.wf, model.cr), (.5, .9))
            self.assertNotIn('pcr', optimizer_constructor_kwargs(alias, args))
        self.assertNotIn('MaCRO-DE-t', study.OPTIMIZERS)
        identity = optimizer_scientific_identity('MaCRO-DE-t', args)
        self.assertEqual(identity['implementation_revision'], 'awad-close-far-v2')
        self.assertEqual(identity['parameters']['wf'], .5)
        sig = study.build_cache_signature(args)
        with patch.object(MaCRO_DE_t, 'IMPLEMENTATION_REVISION', 'incompatible'):
            self.assertNotEqual(sig, study.build_cache_signature(args))
        for old in ['MaCRO-DE', 'DSADE']:
            args.optimizers = [old]
            self.assertNotEqual(sig, study.build_cache_signature(args))
        from optimizer_adapters import get_custom_adapter
        self.assertIsNone(get_custom_adapter('DE-MC-CF'))

    def test_both_dataset_sources_use_main_best_run_path(self):
        import main_best as study
        from test_sensitivity_optimizers import make_args
        from mafese import Data
        rng = np.random.default_rng(55)
        x = rng.normal(size=(48, 6))
        y = np.tile([0, 1], 24)
        for source in ['codesmell', 'mafese']:
            args = make_args()
            args.dataset_source = source
            args.epochs, args.pop_size = 1, 5
            args.optimizer_compute_device = 'cpu'
            data = Data(x, y)
            data.split_train_test(test_size=.25, random_state=7)
            result = study._run_single(data, 'knn', 'MaCRO-DE-t', 'vstf_01', args, 71)
            self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
