import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from scipy.stats import chi2

from diversity_gpu_batching import DiversityMathBatcher


class DE_Mahalanobis(Optimizer):
    """
    Differential Evolution with only Mahalanobis mutation-pool selection.

    Active ablation component:
    - Mahalanobis close/far mutation-pool selection using the same covariance
      stabilization and chi-square threshold as the current DSADE/MaCRO-DE
      implementation.

    Excluded components:
    - AWAD-based mutation-factor or crossover adaptation.
    - Diversity-guided survivor selection.

    The current project Mahalanobis pool switch depends on a delayed normalized
    AWAD diversity value to choose close versus far particles. This class keeps
    that minimal diversity bookkeeping only for the pool switch; F, crossover,
    and survivor selection remain fixed standard DE behavior.
    """

    def __init__(
        self,
        epoch=1000,
        pop_size=50,
        wf=0.1,
        cr=0.9,
        mahalanobis_q=0.68,
        compute_device="cpu",
        gpu_device_id=0,
        gpu_memory_fraction=0.85,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wf = self.validator.check_float("wf", wf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0.0, 1.0))
        self.mahalanobis_q = self.validator.check_float("mahalanobis_q", mahalanobis_q, (0.0, 1.0))
        self.compute_device = str(compute_device)
        self.math_batcher = DiversityMathBatcher(
            self.compute_device, gpu_device_id, gpu_memory_fraction
        )
        self.set_parameters(["epoch", "pop_size", "wf", "cr", "mahalanobis_q"])
        self.sort_flag = False
        self.support_parallel_modes = True

        self.div_awad_hist = None
        self.div_norm_hist = None
        self.div_max_seen = None
        self.div_norm_for_update = 1.0

    def initialize_variables(self):
        self.div_awad_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_for_update = 1.0
        self.div_max_seen = None

    def before_main_loop(self):
        pop_pos = self._positions(self.pop)
        div0 = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_max_seen = max(div0, self.EPSILON)

    def _positions(self, pop):
        return np.array([agent.solution for agent in pop], dtype=float)

    def _awad(self, pop_pos, lb, ub):
        return self.math_batcher.awad(pop_pos, lb, ub)

    def _safe_cov_inv(self, pop_pos):
        return self.math_batcher.covariance_inverse(pop_pos, self.problem.n_dims)

    def _mutation_pool_indices(self, pop_pos, div_norm_used):
        n_dims = self.problem.n_dims
        dist2 = self.math_batcher.mahalanobis_distances(pop_pos, n_dims)
        thr = chi2.ppf(self.mahalanobis_q, max(n_dims, 1))
        close_mask = dist2 <= thr

        close_indices = np.flatnonzero(close_mask)
        far_indices = np.flatnonzero(~close_mask)

        if div_norm_used >= 0.5 and close_indices.size >= 3:
            return close_indices
        if div_norm_used < 0.5 and far_indices.size >= 3:
            return far_indices
        return np.arange(self.pop_size)

    def _sample_pool_indices(self, pool_indices, current_idx):
        candidates = pool_indices[pool_indices != current_idx]
        if candidates.size < 3:
            candidates = np.array(list(set(range(self.pop_size)) - {current_idx}), dtype=int)
        return self.generator.choice(candidates, 3, replace=False)

    def _binomial_crossover(self, parent_pos, mutant_pos):
        j0 = self.generator.integers(0, self.problem.n_dims)
        cross_mask = self.generator.random(self.problem.n_dims) <= self.cr
        cross_mask[j0] = True
        trial = self.math_batcher.crossover(parent_pos, mutant_pos, cross_mask)
        return self.correct_solution(trial)

    def evolve(self, epoch):
        epoch_idx = epoch - 1
        if self.div_max_seen is None:
            self.before_main_loop()

        div_norm_used = float(np.clip(self.div_norm_for_update, 0.0, 1.0))
        pop_new = []

        for idx in range(self.pop_size):
            pop_pos = self._positions(self.pop)
            pool_indices = self._mutation_pool_indices(pop_pos, div_norm_used)
            idxs = self._sample_pool_indices(pool_indices, idx)
            x1, x2, x3 = pop_pos[idxs[0]], pop_pos[idxs[1]], pop_pos[idxs[2]]

            mutant = self.correct_solution(self.math_batcher.mutate(x1, x2, x3, self.wf))
            trial = self._binomial_crossover(self.pop[idx].solution, mutant)
            candidate = Agent(solution=trial)

            if self.mode not in self.AVAILABLE_MODES:
                candidate.target = self.get_target(trial)
                self.pop[idx] = self.get_better_agent(candidate, self.pop[idx], self.problem.minmax)
            else:
                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch_idx] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0))
        self.div_norm_hist[epoch_idx] = div_norm_now
        self.div_norm_for_update = div_norm_now
