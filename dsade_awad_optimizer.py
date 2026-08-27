import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from scipy.stats import chi2

from diversity_gpu_batching import DiversityMathBatcher

class DSADE_AWAD(Optimizer):
    """
    Diversity-based Self-Adaptive Control in Differential Evolution (DSADE)
    with:
    - Delayed AWAD diversity-adaptive control
    - Mahalanobis grouping for mutation pool sampling
    - AWAD-aware survivor selection
    """

    def __init__(
        self,
        epoch=1000,
        pop_size=50,
        beta_min=0.2,
        beta_max=0.8,
        pcr=0.2,
        mahalanobis_q=0.68,
        compute_device="cpu",
        gpu_device_id=0,
        gpu_memory_fraction=0.85,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.beta_min = self.validator.check_float("beta_min", beta_min, (0.0, 2.0))
        self.beta_max = self.validator.check_float("beta_max", beta_max, (0.0, 2.0))
        self.pcr = self.validator.check_float("pcr", pcr, (0.0, 1.0))
        self.mahalanobis_q = self.validator.check_float("mahalanobis_q", mahalanobis_q, (0.0, 1.0))
        self.compute_device = str(compute_device)
        self.math_batcher = DiversityMathBatcher(
            self.compute_device, gpu_device_id, gpu_memory_fraction
        )
        if self.beta_min > self.beta_max:
            raise ValueError("beta_min debe ser <= beta_max.")
        self.set_parameters(
            ["epoch", "pop_size", "beta_min", "beta_max", "pcr", "mahalanobis_q"]
        )
        self.sort_flag = False
        self.support_parallel_modes = True

        self.div_awad_hist = None
        self.div_norm_hist = None
        self.pcr_hist = None
        self.fmean_hist = None
        self.div_max_seen = None
        self.div_norm_for_update = 1.0

    def initialize_variables(self):
        self.div_awad_hist = np.full(self.epoch, np.nan, dtype=float)
        self.div_norm_hist = np.full(self.epoch, np.nan, dtype=float)
        self.pcr_hist = np.full(self.epoch, np.nan, dtype=float)
        self.fmean_hist = np.full(self.epoch, np.nan, dtype=float)
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

    def _mutation_pool(self, pop_pos, div_norm_used):
        n_dims = self.problem.n_dims
        dist2 = self.math_batcher.mahalanobis_distances(pop_pos, n_dims)
        thr = chi2.ppf(self.mahalanobis_q, max(n_dims, 1))
        close_mask = dist2 <= thr
        close_particles = pop_pos[close_mask]
        far_particles = pop_pos[~close_mask]

        if div_norm_used >= 0.5 and close_particles.shape[0] >= 3:
            return close_particles
        if div_norm_used < 0.5 and far_particles.shape[0] >= 3:
            return far_particles
        return pop_pos

    def local_awad_contribution(self, candidate_pos, base_pop_pos):
        """
        Estimate the local AWAD contribution of one candidate in the current
        population context. The compared parent is removed from base_pop_pos by
        the caller, so parent and offspring are scored under the same context.
        """
        candidate_pos = np.asarray(candidate_pos, dtype=float)
        base_pop_pos = np.asarray(base_pop_pos, dtype=float)
        if base_pop_pos.size == 0:
            local_pop = candidate_pos.reshape(1, -1)
        else:
            local_pop = np.vstack((base_pop_pos, candidate_pos))
        return self._awad(local_pop, self.problem.lb, self.problem.ub)

    def diversity_selection(self, parent, offspring, base_pop_pos):
        """
        Select between parent and offspring using fitness first, then AWAD.
        Better fitness is always accepted. Otherwise, the offspring survives
        only if it improves the local AWAD contribution.
        """
        if self.compare_target(offspring.target, parent.target, self.problem.minmax):
            return offspring

        offspring_awad = self.local_awad_contribution(offspring.solution, base_pop_pos)
        parent_awad = self.local_awad_contribution(parent.solution, base_pop_pos)
        if offspring_awad > parent_awad + self.EPSILON:
            return offspring
        return parent

    def _diversity_selection_population(self, pop_old, pop_new):
        old_pos = self._positions(pop_old)
        selected = []
        for idx, (parent, offspring) in enumerate(zip(pop_old, pop_new)):
            base_pop_pos = np.delete(old_pos, idx, axis=0)
            selected.append(self.diversity_selection(parent, offspring, base_pop_pos))
        return selected

    def evolve(self, epoch):
        epoch_idx = epoch - 1
        div_norm_used = float(np.clip(self.div_norm_for_update, 0.0, 1.0))
        scale_f = float(np.clip(0.5 + (1.0 - div_norm_used), 0.5, 1.5))
        pcr_it = float(np.clip(self.pcr + 0.25 * (1.0 - div_norm_used), 0.10, 0.95))
        self.pcr_hist[epoch_idx] = pcr_it

        f_used_sum = 0.0
        pop_new = []

        for idx in range(self.pop_size):
            pop_pos = self._positions(self.pop)
            pool = self._mutation_pool(pop_pos, div_norm_used)
            idxs = self.generator.choice(pool.shape[0], 3, replace=False)
            x1, x2, x3 = pool[idxs[0]], pool[idxs[1]], pool[idxs[2]]

            f_vec = self.generator.uniform(self.beta_min, self.beta_max, self.problem.n_dims) * scale_f
            f_vec = np.clip(f_vec, 0.10, 1.50)
            f_used_sum += float(np.mean(f_vec))

            y = self.math_batcher.mutate(x1, x2, x3, f_vec)
            y = self.correct_solution(y)

            j0 = self.generator.integers(0, self.problem.n_dims)
            cross_mask = self.generator.random(self.problem.n_dims) <= pcr_it
            cross_mask[j0] = True
            z = self.math_batcher.crossover(self.pop[idx].solution, y, cross_mask)
            z = self.correct_solution(z)
            candidate = Agent(solution=z)

            if self.mode not in self.AVAILABLE_MODES:
                candidate.target = self.get_target(z)
                base_pop_pos = np.delete(pop_pos, idx, axis=0)
                self.pop[idx] = self.diversity_selection(self.pop[idx], candidate, base_pop_pos)
            else:
                pop_new.append(candidate)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self._diversity_selection_population(self.pop, pop_new)

        self.fmean_hist[epoch_idx] = f_used_sum / self.pop_size

        pop_pos = self._positions(self.pop)
        div_awad = self._awad(pop_pos, self.problem.lb, self.problem.ub)
        self.div_awad_hist[epoch_idx] = div_awad
        self.div_max_seen = max(self.div_max_seen, div_awad)
        div_norm_now = float(np.clip(div_awad / (self.div_max_seen + self.EPSILON), 0.0, 1.0))
        self.div_norm_hist[epoch_idx] = div_norm_now
        self.div_norm_for_update = div_norm_now


# Backward compatibility aliases
DSADE_AWAD = DSADE_AWAD
