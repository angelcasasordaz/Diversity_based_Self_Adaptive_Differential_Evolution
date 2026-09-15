"""Canonical manuscript DSA-DE, scientific preservation, and cache isolation."""
import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mealpy import FloatVar
from mealpy.utils.agent import Agent
from mealpy.utils.target import Target

from dsade_awad_optimizer import DSADE
from optimizer_factory import (build_optimizer, resolve_optimizer, optimizer_acronym,
                               optimizer_scientific_identity)
from optimizer_adapters import CUSTOM_ADAPTERS
import main_best as study
import historical_transfer_plots as historical
from test_sensitivity_optimizers import make_args


class CanonicalDSADETests(unittest.TestCase):
    def test_all_methods_preserve_original_manuscript_implementation(self):
        # Captured from dsade_awad_optimizer.py before migration. Only class name
        # and class-level identity constants may change, never scientific methods.
        cls = ast.parse(inspect.getsource(DSADE)).body[0]
        methods = [ast.dump(node, include_attributes=False) for node in cls.body
                   if isinstance(node, ast.FunctionDef)]
        digest = hashlib.sha256('\n'.join(methods).encode()).hexdigest()
        self.assertEqual(digest, '0d57b6324616eff3b1e0af21c451f3724a4f638677d53358f2fe5d6fc71d48ed')

    def test_canonical_aliases_factory_capabilities_and_plots(self):
        args = make_args()
        for alias in ['DSADE', 'DSA-DE', 'DSA_DE', 'dsa_de']:
            resolved = resolve_optimizer(alias)
            self.assertIs(resolved.optimizer_class, DSADE)
            self.assertEqual(resolved.optimizer_class.__module__, 'dsade_awad_optimizer')
            self.assertEqual(resolved.canonical_name, 'DSADE')
            self.assertEqual(optimizer_acronym(alias), 'DSADE')
            self.assertIs(type(build_optimizer(alias, args)), DSADE)
            self.assertIs(resolved.capability.adapter.optimizer_class, DSADE)
            self.assertTrue(resolved.capability.supports_gpu)
            self.assertTrue(study.is_exact_dsade_method(alias))
            self.assertTrue(study.is_dsade_method(alias))
            self.assertTrue(historical.is_exact_dsade_method(alias))
        self.assertEqual([key for key, item in CUSTOM_ADAPTERS.items()
                          if item.optimizer_class is DSADE], ['DSADE'])

    def test_removed_identity_is_rejected(self):
        for name in ['DSADE_AWAD', 'DSADE-AWAD']:
            with self.assertRaises(ValueError):
                resolve_optimizer(name)
            self.assertFalse(study.is_dsade_method(name))
            self.assertFalse(historical.is_exact_dsade_method(name))

    def test_fitness_then_awad_then_parent(self):
        model = DSADE(epoch=2, pop_size=10)
        model.problem = SimpleNamespace(n_dims=2, lb=np.zeros(2), ub=np.ones(2), minmax='min')
        base = np.zeros((9, 2))
        parent = Agent(solution=np.zeros(2), target=Target(1.))
        better_fitness = Agent(solution=np.zeros(2), target=Target(.5))
        worse_but_diverse = Agent(solution=np.ones(2), target=Target(2.))
        same = Agent(solution=np.zeros(2), target=Target(1.))
        self.assertIs(model.diversity_selection(parent, better_fitness, base), better_fitness)
        self.assertGreater(model.local_awad_contribution(worse_but_diverse.solution, base),
                           model.local_awad_contribution(parent.solution, base)+model.EPSILON)
        self.assertIs(model.diversity_selection(parent, worse_but_diverse, base), worse_but_diverse)
        self.assertIs(model.diversity_selection(parent, same, base), parent)
        self.assertIs(model.diversity_selection(worse_but_diverse,
                      Agent(solution=np.zeros(2), target=Target(3.)), base), worse_but_diverse)

    def test_cache_revision_aliases_and_parameter_identity(self):
        args = make_args()
        args.optimizers = ['DSADE']
        identity = optimizer_scientific_identity('DSADE', args)
        self.assertEqual(identity['implementation_revision'], 'manuscript-awad-survivor-v1')
        self.assertEqual(set(identity['parameters']), set(DSADE.SCIENTIFIC_PARAMETERS))
        for mode in ['full', 'ablation', 'sensitivity', 'sensitivity_weights', 'transfer_functions']:
            args.experiment_mode = mode
            new_signature = study.build_cache_signature(args)
            for alias in ['DSADE', 'DSA-DE', 'DSA_DE']:
                args.optimizers = [alias]
                self.assertEqual(new_signature, study.build_cache_signature(args))
            with patch.object(DSADE, 'IMPLEMENTATION_REVISION', None):
                self.assertNotEqual(new_signature, study.build_cache_signature(args))
            with patch.object(DSADE, 'IMPLEMENTATION_REVISION', 'incompatible'):
                self.assertNotEqual(new_signature, study.build_cache_signature(args))

    def test_old_unversioned_checkpoints_are_rejected(self):
        args = make_args()
        for mode in ['sensitivity', 'sensitivity_weights']:
            args.experiment_mode = mode
            expected = (study.sensitivity_checkpoint_metadata(args, 'DSADE', .68)
                        if mode == 'sensitivity' else study.weight_checkpoint_metadata(args, 'DSADE'))
            self.assertTrue(study.checkpoint_metadata_matches(expected, expected))
            old = dict(expected)
            del old['OptimizerScientificIdentity']
            self.assertFalse(study.checkpoint_metadata_matches(old, expected))
            old.pop('Optimizer')
            self.assertFalse(study.checkpoint_metadata_matches(old, expected))

    def test_synthetic_solve_all_aliases_both_modes(self):
        args = make_args()
        args.epochs, args.pop_size = 2, 10
        for mode in ['single', 'swarm']:
            snapshots = []
            for alias in ['DSADE', 'DSA-DE', 'DSA_DE']:
                model = build_optimizer(alias, args)
                model.solve(dict(bounds=FloatVar(lb=[-2.]*3, ub=[2.]*3), minmax='min',
                                 obj_func=lambda x: float(np.sum(x*x)), log_to=None),
                            mode=mode, seed=71)
                snapshots.append((model._positions(model.pop), model.div_awad_hist,
                                  model.div_norm_hist, model.pcr_hist, model.fmean_hist))
            for snapshot in snapshots[1:]:
                for actual, expected in zip(snapshot, snapshots[0]):
                    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
