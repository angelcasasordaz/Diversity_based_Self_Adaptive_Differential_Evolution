"""Small FS integration runs and deterministic comparisons with the CEC source.

No datasets, experiment runner, reports, or checkpoint files are written.
CEC_SOURCE_DIR can point to another checkout of the audited CEC project.
"""
import ast
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mafese import Data
from mealpy import FloatVar

from dsade_awad_optimizer import DSADE
from dsade_cec_optimizer import DSADE as CEC_DSADE
from macro_de_optimizer import MaCRO_DE
from macro_de_t_optimizer import MaCRO_DE_t
from macro_de_t_v2_optimizer import MaCRO_DE_t_v2
from optimizer_factory import (build_optimizer, optimizer_constructor_kwargs,
                               optimizer_scientific_identity, resolve_optimizer)


NAMES = {
    'DSADE': (DSADE, ('DSA-DE', 'DSA_DE')),
    'MaCRO-DE': (MaCRO_DE, ('MACRO_DE', 'MACRODE')),
    'MaCRO-DE-t': (MaCRO_DE_t, ('MACRO_DE_T', 'MACRODET', 'DE-MC-CF', 'DE_MC_CF')),
    'MaCRO-DE-t-v2': (MaCRO_DE_t_v2, ('MACRO_DE_T_V2', 'MACRODETV2', 'DE-MC-CF-v2', 'DE_MC_CF_V2')),
    'DSADE-CEC': (CEC_DSADE, ('DSADE_CEC', 'DSA-DE-CEC')),
}


def load_cec_sources():
    directory = Path(os.environ.get('CEC_SOURCE_DIR', Path(__file__).resolve().parents[2] /
                     'Adaptive_Mahalanobis-Cholesky_Differential_Evolution_MaCRO_DE'))
    modules = {}
    with patch.dict(sys.modules):
        for name in ('compute_backend', 'de_ablation_base', 'de_mc_optimizer',
                     'de_mc_cf_optimizer', 'de_mc_cf_v2_optimizer',
                     'macro_de_optimizer', 'dsade_optimizer'):
            path = directory / f'{name}.py'
            if not path.is_file():
                raise unittest.SkipTest(f'CEC reference not available: {path}; set CEC_SOURCE_DIR')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    return modules


class FactoryAndFeatureSelectionTests(unittest.TestCase):
    def test_all_aliases_resolve_instantiate_and_keep_distinct_identities(self):
        settings = SimpleNamespace(epochs=2, pop_size=10, compute_device='cpu')
        identities = []
        for name, (cls, aliases) in NAMES.items():
            with self.subTest(name=name):
                identities.append(optimizer_scientific_identity(name, settings)['implementation_revision'])
                for alias in (name, *aliases):
                    resolved = resolve_optimizer(' ' + alias.lower() + ' ')
                    self.assertEqual(resolved.canonical_name, name)
                    self.assertIs(resolved.optimizer_class, cls)
                    self.assertIs(type(build_optimizer(alias, settings)), cls)
                    self.assertTrue(resolved.capability.supports_gpu)
        self.assertEqual(len(set(identities)), len(NAMES))

    def test_source_defaults_and_existing_parameter_bridge(self):
        from tests.test_sensitivity_optimizers import make_args
        settings = make_args()
        for name in ('DSADE', 'MaCRO-DE', 'DSADE-CEC'):
            model = build_optimizer(name, settings)
            self.assertEqual((model.beta_min, model.beta_max, model.pcr, model.mahalanobis_q),
                             (settings.dsade_beta_min, settings.dsade_beta_max,
                              settings.dsade_pcr, settings.dsade_mahal_q))
        fixed = build_optimizer('MaCRO-DE-t', settings)
        self.assertEqual((fixed.wf, fixed.cr), (.5, .9))
        self.assertNotIn('pcr', optimizer_constructor_kwargs('MaCRO-DE-t', settings))
        v2 = MaCRO_DE_t_v2(epoch=2, pop_size=10)
        self.assertEqual((v2.beta_min, v2.beta_max, v2.pcr, v2.mahalanobis_q), (.1, .6, .1, .5))
        kwargs = optimizer_constructor_kwargs('MaCRO-DE-t-v2', settings)
        self.assertEqual(kwargs['beta_min'], settings.dsade_beta_min)
        self.assertEqual(kwargs['beta_max'], settings.dsade_beta_max)
        self.assertFalse({'pcr', 'wf', 'cr'}.intersection(kwargs))
        for forbidden in ('wf', 'cr', 'pcr'):
            with self.assertRaises(TypeError):
                MaCRO_DE_t_v2(**{forbidden: .5})

    def test_all_candidates_execute_real_main_best_feature_selection(self):
        import main_best as study
        from tests.test_sensitivity_optimizers import make_args
        rng = np.random.default_rng(513)
        x = rng.normal(size=(48, 6))
        y = (x[:, 0] + .3*x[:, 1] > 0).astype(int)
        data = Data(x, y)
        data.split_train_test(test_size=.25, random_state=7)
        args = make_args()
        args.epochs, args.pop_size = 2, 10
        args.optimizer_compute_device = 'cpu'
        for name in NAMES:
            with self.subTest(name=name), patch.object(study, 'save_cache', side_effect=AssertionError('No output writes')):
                result = study._run_single(data, 'knn', name, 'vstf_01', args, 71)
                self.assertEqual(len(result['curve']), 2)
                self.assertTrue(np.all(np.isfinite(result['curve'])))
                self.assertTrue(np.isfinite(result['fit_final']))
                self.assertTrue(np.isfinite(result['as_test']))
                self.assertGreaterEqual(result['n_features'], 1)
                self.assertLessEqual(result['n_features'], 6)

    def test_v2_coordinate_uniform_draws_with_macro_d_scaling_and_adaptive_pcr(self):
        model = MaCRO_DE_t_v2(epoch=2, pop_size=10, beta_min=0., beta_max=2.)
        model.problem = SimpleNamespace(n_dims=100)
        model.generator = np.random.default_rng(42)
        expected_rng = np.random.default_rng(42)
        for D in (0., .25, .5, 1.):
            for dM in (0., .25, .5, 1.):
                f, pcr = model._adaptive_control(dM, D)
                draws = expected_rng.uniform(0., 2., 100)
                np.testing.assert_array_equal(f, np.clip(draws * np.clip(1.5-D, .5, 1.5), .1, 1.5))
                self.assertTrue(np.all((f >= .1) & (f <= 1.5)))
                self.assertGreater(np.ptp(f), 0.)
                self.assertEqual(pcr, .1 + .25 * (1 - dM))
        self.assertEqual(model.generator.bit_generator.state, expected_rng.bit_generator.state)

    def test_v2_beta_bound_validation(self):
        for low, high in ((-.1, .6), (.7, .6), (np.nan, .6), (.1, np.inf)):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                MaCRO_DE_t_v2(beta_min=low, beta_max=high)
        model = MaCRO_DE_t_v2(beta_min=.4, beta_max=.4)
        model.problem = SimpleNamespace(n_dims=3)
        model.generator = np.random.default_rng(7)
        np.testing.assert_array_equal(model._sample_scale_factors(.5), [.4, .4, .4])
        np.testing.assert_allclose(model._sample_scale_factors(0.), [.6, .6, .6])
        np.testing.assert_array_equal(model._sample_scale_factors(1.), [.2, .2, .2])

    def test_v2_reuses_macro_diversity_definition(self):
        from mealpy.utils.agent import Agent
        population = np.random.default_rng(33).normal(size=(10, 3))
        models = [cls(epoch=2, pop_size=10) for cls in (MaCRO_DE, MaCRO_DE_t_v2)]
        for model in models:
            model.problem = SimpleNamespace(n_dims=3, lb=np.full(3, -5.), ub=np.full(3, 5.))
            model.pop = [Agent(solution=x.copy()) for x in population]
            model.initialize_variables()
            model.before_main_loop()
        self.assertEqual(models[0].div_max_seen, models[1].div_max_seen)
        for scale in (0., .1, 1., 2.):
            values = [model._awad(scale*population, None, None) for model in models]
            np.testing.assert_allclose(values[0], values[1], rtol=1e-14)

    def test_v2_distance_normalization_and_collapsed_population(self):
        np.testing.assert_array_equal(MaCRO_DE_t_v2._normalized_mahalanobis(
            np.array([0., 1., 4., 16.])), [0., .25, .5, 1.])
        np.testing.assert_array_equal(MaCRO_DE_t_v2._normalized_mahalanobis(np.zeros(5)), np.zeros(5))

    def test_v2_evolution_uses_distance_control_and_greedy_survivors(self):
        from mealpy.utils.agent import Agent
        from mealpy.utils.target import Target
        from scipy.stats import chi2
        population = np.random.default_rng(73).normal(size=(10, 3))
        for mode in ('single', 'swarm'):
            model = MaCRO_DE_t_v2(epoch=1, pop_size=10)
            model.problem = SimpleNamespace(n_dims=3, lb=np.full(3, -5.),
                                            ub=np.full(3, 5.), minmax='min')
            model.mode = mode
            model.generator = np.random.default_rng(71)
            model.correct_solution = lambda x: np.clip(x, -5., 5.)
            model.get_target = lambda x, counted=True: Target(float(np.sum(x*x)))
            model.pop = [Agent(solution=x.copy(), target=model.get_target(x)) for x in population]
            model.initialize_variables()
            model.before_main_loop()
            model.div_norm_for_update = .25
            expected_D = .25
            dist2 = model._mahalanobis_dist2(population)
            expected_dm = np.sqrt(dist2) / np.max(np.sqrt(dist2))
            donors, trials, random_draws, crossover_rates = [], [], [], []
            original_donors = model._sample_mutation_indices
            original_crossover = model._binomial_crossover
            def sample(positions, idx):
                np.testing.assert_array_equal(positions, population)
                result = original_donors(positions, idx)
                self.assertNotIn(idx, result)
                self.assertEqual(len(set(result)), 3)
                donors.append(result)
                rng = np.random.default_rng()
                rng.bit_generator.state = model.generator.bit_generator.state
                random_draws.append(rng.uniform(model.beta_min, model.beta_max, 3))
                return result
            def crossover(parent, mutant):
                idx = len(trials)
                i, j, k = donors[idx]
                expected_f = np.clip(random_draws[idx] * np.clip(1.5-expected_D, .5, 1.5), .1, 1.5)
                np.testing.assert_allclose(mutant, np.clip(
                    population[i] + expected_f*(population[j]-population[k]), -5., 5.))
                crossover_rates.append(model.cr)
                trial = original_crossover(parent, mutant)
                trials.append(trial.copy())
                return trial
            model._sample_mutation_indices = sample
            model._binomial_crossover = crossover
            # AWAD does not enter control or group classification.
            # Historical note above: AWAD now supplies delayed D for F, never dM/groups.
            with patch.object(model, '_awad', return_value=123.):
                model.evolve(1)
            np.testing.assert_allclose(model.dm_hist[0], expected_dm)
            np.testing.assert_allclose(model.pcr_hist[0], .1 + .25*(1-expected_dm))
            np.testing.assert_array_equal(crossover_rates, model.pcr_hist[0])
            expected_scales = np.clip(np.array(random_draws)*(1.5-expected_D), .1, 1.5)
            np.testing.assert_allclose(model.f_hist[0], np.mean(expected_scales, axis=1))
            self.assertEqual(model.d_hist[0], expected_D)
            self.assertEqual(model.div_norm_for_update,
                             123. / (max(model.div_max_seen, 123.) + model.EPSILON))
            self.assertGreater(np.ptp(model.pcr_hist[0]), 0.)
            close, far = model._close_far_indices(population)
            np.testing.assert_array_equal(close, np.flatnonzero(dist2 <= chi2.ppf(.5, 3)))
            np.testing.assert_array_equal(far, np.flatnonzero(dist2 > chi2.ppf(.5, 3)))
            for idx, trial in enumerate(trials):
                expected = trial if np.sum(trial*trial) < np.sum(population[idx]**2) else population[idx]
                np.testing.assert_array_equal(model.pop[idx].solution, expected)

    def test_v2_cache_revision_and_legacy_settings_do_not_override_control(self):
        settings = SimpleNamespace(epochs=2, pop_size=10, dsade_beta_min=.4,
                                   dsade_beta_max=.8, dsade_pcr=.9, dsade_mahal_q=.5)
        for alias in ('MaCRO-DE-t-v2', 'MaCRO_DE_t_v2'):
            model = build_optimizer(alias, settings)
            self.assertEqual(model.pcr, .1)
            self.assertEqual((model.beta_min, model.beta_max), (.4, .8))
            identity = optimizer_scientific_identity(alias, settings)
            self.assertEqual(identity['implementation_revision'], 'macro-d-scaled-coordinate-adaptive-pcr-v5')
            self.assertEqual(set(identity['parameters']),
                             {'epoch', 'pop_size', 'beta_min', 'beta_max', 'mahalanobis_q'})
            self.assertEqual(identity['parameters']['beta_min'], .4)
            self.assertEqual(identity['parameters']['beta_max'], .8)
        import main_best as study
        from tests.test_sensitivity_optimizers import make_args
        args = make_args()
        args.optimizers = ['MaCRO-DE-t-v2']
        signature = study.build_cache_signature(args)
        with patch.object(MaCRO_DE_t_v2, 'IMPLEMENTATION_REVISION', 'coordinate-beta-adaptive-pcr-v4'):
            self.assertNotEqual(signature, study.build_cache_signature(args))
        args.dsade_beta_min += .01
        self.assertNotEqual(signature, study.build_cache_signature(args))


class CECScientificEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = load_cec_sources()
        cls.pairs = (
            (MaCRO_DE, cls.modules['macro_de_optimizer'].MaCRO_DE),
            (MaCRO_DE_t, cls.modules['de_mc_cf_optimizer'].DE_MC_CF),
            (CEC_DSADE, cls.modules['dsade_optimizer'].DSADE),
        )

    def test_scientific_method_asts_match_source(self):
        for port, reference in self.pairs:
            for name, method in vars(reference).items():
                if not inspect.isfunction(method):
                    continue
                # Constructors add only backend transport/configuration.
                if port in (MaCRO_DE, CEC_DSADE) and name == '__init__':
                    continue
                with self.subTest(optimizer=port.__name__, method=name):
                    expected = ast.parse(textwrap.dedent(inspect.getsource(method)))
                    actual = ast.parse(textwrap.dedent(inspect.getsource(getattr(port, name))))
                    if port is CEC_DSADE:
                        # Remove GPU dispatch branches, then normalize the two
                        # backend equations to their exact NumPy equivalents.
                        body = actual.body[0].body
                        body[:] = [node for node in body if not (
                            isinstance(node, ast.If) and
                            ast.unparse(node.test) == 'self.math_batcher.uses_gpu')]
                        for node in ast.walk(actual):
                            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                                call = ast.unparse(node.value.func)
                                if call == 'self.math_batcher.mutate':
                                    node.value = ast.parse('x1 + f_vec * (x2 - x3)', mode='eval').body
                                elif call == 'self.math_batcher.crossover':
                                    original = ast.parse('z[cross_mask] = y[cross_mask]').body[0]
                                    node.targets, node.value = original.targets, original.value
                    self.assertEqual(ast.dump(actual), ast.dump(expected))

    def test_seeded_cpu_trajectories_match_cec_in_both_modes(self):
        # Continuous state exposes donor/crossover differences hidden by binary masks.
        for port, reference in self.pairs:
            for mode in ('single', 'swarm'):
                for dims in (1, 6):
                    with self.subTest(optimizer=port.__name__, mode=mode, dims=dims):
                        models = []
                        for cls in (port, reference):
                            model = cls(epoch=4, pop_size=10)
                            model.solve(dict(bounds=FloatVar(lb=[-5.]*dims, ub=[5.]*dims),
                                             minmax='min', obj_func=lambda x: float(np.sum(x*x)),
                                             log_to=None), mode=mode, seed=717)
                            models.append(model)
                        a, b = models
                        for attr in ('div_awad_hist', 'div_norm_hist', 'pcr_hist', 'fmean_hist', 'routing_counts'):
                            if hasattr(a, attr):
                                np.testing.assert_equal(getattr(a, attr), getattr(b, attr))
                        np.testing.assert_array_equal(a._positions(a.pop), b._positions(b.pop))
                        np.testing.assert_array_equal(a.history.list_global_best_fit, b.history.list_global_best_fit)
                        self.assertEqual(a.generator.bit_generator.state, b.generator.bit_generator.state)

    def test_gpu_feature_selection_when_cuda_is_available(self):
        from numerical_backend import create_numerical_backend
        backend = create_numerical_backend('gpu')
        if not backend.uses_gpu:
            self.skipTest(backend.fallback_reason)
        import main_best as study
        from tests.test_sensitivity_optimizers import make_args
        args = make_args()
        args.epochs, args.pop_size = 1, 10
        args.optimizer_compute_device = 'gpu'
        data = Data(np.random.default_rng(51).normal(size=(48, 6)), np.tile([0, 1], 24))
        data.split_train_test(test_size=.25, random_state=7)
        for name in ('DSADE', 'DSADE-CEC', 'MaCRO-DE', 'MaCRO-DE-t', 'MaCRO-DE-t-v2'):
            with self.subTest(name=name):
                result = study._run_single(data, 'knn', name, 'vstf_01', args, 71)
                self.assertTrue(np.isfinite(result['fit_final']))


if __name__ == '__main__':
    unittest.main()
