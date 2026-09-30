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
                     'Adaptive_Mahalanobis-Cholesky_DIfferential_Evolution'))
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
                    self.assertEqual(resolved.capability.supports_gpu, name != 'DSADE-CEC')
        self.assertEqual(len(set(identities)), len(NAMES))

    def test_source_defaults_and_existing_parameter_bridge(self):
        from tests.test_sensitivity_optimizers import make_args
        settings = make_args()
        for name in ('DSADE', 'MaCRO-DE', 'MaCRO-DE-t-v2', 'DSADE-CEC'):
            model = build_optimizer(name, settings)
            self.assertEqual((model.beta_min, model.beta_max, model.pcr, model.mahalanobis_q),
                             (settings.dsade_beta_min, settings.dsade_beta_max,
                              settings.dsade_pcr, settings.dsade_mahal_q))
        fixed = build_optimizer('MaCRO-DE-t', settings)
        self.assertEqual((fixed.wf, fixed.cr), (.5, .9))
        self.assertNotIn('pcr', optimizer_constructor_kwargs('MaCRO-DE-t', settings))
        v2 = MaCRO_DE_t_v2(epoch=2, pop_size=10)
        self.assertEqual((v2.beta_min, v2.beta_max, v2.pcr, v2.mahalanobis_q), (.1, .6, .1, .5))
        for forbidden in ('wf', 'cr'):
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

    def test_v2_scale_draws_are_unscaled_and_unclipped(self):
        model = MaCRO_DE_t_v2(epoch=2, pop_size=10, beta_min=0., beta_max=2., pcr=0.)
        model.problem = SimpleNamespace(n_dims=100)
        model.generator = np.random.default_rng(42)
        expected = np.random.default_rng(42).uniform(0., 2., 100)
        np.testing.assert_array_equal(model._sample_scale_factors(), expected)
        self.assertTrue(np.any(expected < .1))
        self.assertTrue(np.any(expected > 1.5))


class CECScientificEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = load_cec_sources()
        cls.pairs = (
            (MaCRO_DE, cls.modules['macro_de_optimizer'].MaCRO_DE),
            (MaCRO_DE_t, cls.modules['de_mc_cf_optimizer'].DE_MC_CF),
            (MaCRO_DE_t_v2, cls.modules['de_mc_cf_v2_optimizer'].DE_MC_CF_V2),
            (CEC_DSADE, cls.modules['dsade_optimizer'].DSADE),
        )

    def test_scientific_method_asts_match_source(self):
        for port, reference in self.pairs:
            for name, method in vars(reference).items():
                if not inspect.isfunction(method):
                    continue
                # MaCRO-DE's constructor only substitutes backend transport.
                if port is MaCRO_DE and name == '__init__':
                    continue
                with self.subTest(optimizer=port.__name__, method=name):
                    expected = ast.parse(textwrap.dedent(inspect.getsource(method)))
                    actual = ast.parse(textwrap.dedent(inspect.getsource(getattr(port, name))))
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
        for name in ('DSADE', 'MaCRO-DE', 'MaCRO-DE-t', 'MaCRO-DE-t-v2'):
            with self.subTest(name=name):
                result = study._run_single(data, 'knn', name, 'vstf_01', args, 71)
                self.assertTrue(np.isfinite(result['fit_final']))


if __name__ == '__main__':
    unittest.main()
