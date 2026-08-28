"""Registry for project optimizers and future specialized backend adapters.

Installed MEALPY optimizers are never listed here; they come from MEALPY's own
registry.  Entries here describe only project-owned classes with validated
backend behavior or deliberate scientific compatibility fixes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from mealpy.evolutionary_based.DE import JADE, OriginalDE
from mealpy.evolutionary_based.SHADE import OriginalSHADE
from mealpy.human_based.BRO import OriginalBRO
from mealpy.math_based.RUN import OriginalRUN
from mealpy.physics_based.SA import OriginalSA
from mealpy.swarm_based.DMOA import OriginalDMOA
from mealpy.swarm_based.FOX import OriginalFOX
from mealpy.swarm_based.GOA import OriginalGOA
from mealpy.swarm_based.HHO import OriginalHHO
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.WOA import OriginalWOA

from dbo_optimizer import DBOOptimizer
from de_awad_optimizer import DE_AWAD
from de_diversity_selection_optimizer import DE_DiversitySelection
from de_mahalanobis_optimizer import DE_Mahalanobis
from dsade_optimizer import DSADE
from dsade_awad_optimizer import DSADE_AWAD
from macro_de_optimizer import MaCRO_DE
from diversity_gpu_batching import DiversityMathBatcher


@dataclass(frozen=True)
class OptimizerAdapter:
    canonical_name: str
    optimizer_class: type
    aliases: tuple[str, ...] = ()
    capability: str = "CPU_ONLY"
    reason: str = "project optimizer has no validated GPU kernel"
    parameter_map: tuple[tuple[str, str], ...] = ()
    kernel_work: Callable[[int, int], int] | None = None
    memory_bytes: Callable[[int, int], int] | None = None


def _diversity_work(population_size: int, dimensions: int) -> int:
    return max(population_size * population_size * dimensions,
               population_size * dimensions * dimensions + dimensions ** 3)


def _diversity_memory(population_size: int, dimensions: int) -> int:
    awad = 8 * (2 * population_size * population_size * dimensions + 8 * population_size ** 2)
    mahal = 8 * (10 * dimensions ** 2 + 4 * population_size * dimensions)
    return max(64 * 1024**2, awad, mahal)


COMMON_PARAMETERS = (
    ("epochs", "epoch"), ("pop_size", "pop_size"),
    ("dsade_beta_min", "beta_min"), ("dsade_beta_max", "beta_max"),
    ("dsade_pcr", "pcr"), ("dsade_mahal_q", "mahalanobis_q"),
    ("optimizer_compute_device", "compute_device"),
    ("gpu_device_id", "gpu_device_id"),
    ("gpu_memory_fraction", "gpu_memory_fraction"),
)

MEALPY_GPU_PARAMETERS = (
    ("epochs", "epoch"), ("pop_size", "pop_size"),
    ("optimizer_compute_device", "compute_device"),
    ("gpu_device_id", "gpu_device_id"),
    ("gpu_memory_fraction", "gpu_memory_fraction"),
)


class _GPUArray(np.ndarray):
    """CPU-resident MEALPY state whose ordered array kernels use the GPU service."""

    __array_priority__ = 1000

    def __new__(cls, value, executor):
        obj = np.asarray(value).view(cls)
        obj._gpu_executor = executor
        return obj

    def __array_finalize__(self, parent):
        self._gpu_executor = getattr(parent, "_gpu_executor", None)

    @staticmethod
    def _plain(value):
        if isinstance(value, np.ndarray):
            return np.asarray(value)
        if isinstance(value, (list, tuple)):
            return type(value)(_GPUArray._plain(item) for item in value)
        return value

    def _wrap(self, value):
        if isinstance(value, np.ndarray):
            return _GPUArray(value, self._gpu_executor)
        if isinstance(value, tuple):
            return tuple(self._wrap(item) for item in value)
        return value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None:
            outputs = kwargs.pop("out")
            result = self._gpu_executor.array_operation(
                "ufunc", ufunc.__name__, method,
                tuple(self._plain(item) for item in inputs), kwargs,
            )
            for output, value in zip(outputs, result if isinstance(result, tuple) else (result,)):
                np.copyto(np.asarray(output), value)
            return outputs[0] if len(outputs) == 1 else outputs
        result = self._gpu_executor.array_operation(
            "ufunc", ufunc.__name__, method,
            tuple(self._plain(item) for item in inputs), kwargs,
        )
        return self._wrap(result)

    def __array_function__(self, func, types, args, kwargs):
        result = self._gpu_executor.array_operation(
            "function", func.__name__, "__call__",
            tuple(self._plain(item) for item in args), kwargs,
        )
        return self._wrap(result)


class _MEALPYGPUAdapter:
    """Preserve MEALPY control/RNG while dispatching its array equations in order."""

    def __init__(self, *args, compute_device="cpu", gpu_device_id=0,
                 gpu_memory_fraction=0.85, **kwargs):
        self.compute_device = str(compute_device).lower()
        self._gpu_math = DiversityMathBatcher(
            self.compute_device, gpu_device_id, gpu_memory_fraction,
        )
        super().__init__(*args, **kwargs)

    def generate_empty_agent(self, solution=None):
        agent = super().generate_empty_agent(solution)
        if self.compute_device == "gpu":
            for name, value in tuple(vars(agent).items()):
                if isinstance(value, np.ndarray) and not isinstance(value, _GPUArray):
                    setattr(agent, name, _GPUArray(value, self._gpu_math))
        return agent


class GPUOriginalDE(_MEALPYGPUAdapter, OriginalDE): pass
class GPUJADE(_MEALPYGPUAdapter, JADE): pass
class GPUOriginalSHADE(_MEALPYGPUAdapter, OriginalSHADE): pass
class GPUOriginalPSO(_MEALPYGPUAdapter, OriginalPSO): pass
class GPUOriginalWOA(_MEALPYGPUAdapter, OriginalWOA): pass
class GPUOriginalHHO(_MEALPYGPUAdapter, OriginalHHO): pass
class GPUOriginalGOA(_MEALPYGPUAdapter, OriginalGOA): pass
class GPUOriginalSA(_MEALPYGPUAdapter, OriginalSA): pass
class GPUOriginalBRO(_MEALPYGPUAdapter, OriginalBRO):
    def find_idx_min_distance__(self, target_pos=None, pop=None):
        if self.compute_device != "gpu":
            return super().find_idx_min_distance__(target_pos, pop)
        positions = _GPUArray(
            np.asarray([np.asarray(agent.solution) for agent in pop]), self._gpu_math,
        )
        target = _GPUArray(np.asarray(target_pos).reshape(1, -1), self._gpu_math)
        distances = np.sqrt(np.sum((positions - target) ** 2, axis=1))
        return self.get_idx_min__(np.asarray(distances))
class GPUOriginalRUN(_MEALPYGPUAdapter, OriginalRUN): pass
class GPUOriginalFOX(_MEALPYGPUAdapter, OriginalFOX): pass


class SafeOriginalDMOA(OriginalDMOA):
    """OriginalDMOA with the project's existing binary-space numerical guard."""

    def evolve(self, epoch):
        cf = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / (mean_cost + self.EPSILON))
        for idx in range(self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = self.generator.choice(list(set(range(self.pop_size)) - {idx, alpha}))
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[alpha].solution + phi * (self.pop[alpha].solution - self.pop[k].solution)
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1
        sm = np.zeros(self.pop_size)
        for idx in range(self.pop_size):
            k = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[idx].solution + phi * (self.pop[idx].solution - self.pop[k].solution)
            agent = self.generate_agent(self.correct_solution(new_pos))
            current_fit, trial_fit = self.pop[idx].target.fitness, agent.target.fitness
            denom = max(abs(trial_fit), abs(current_fit), self.EPSILON)
            sm[idx] = (trial_fit - current_fit) / denom
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1
        for idx in range(self.n_baby_sitter):
            if self.C[idx] >= self.L:
                self.pop[idx] = self.generate_agent()
                self.C[idx] = 0
        new_tau = np.mean(sm)
        for idx in range(self.pop_size):
            m = np.full(self.problem.n_dims, sm[idx], dtype=float)
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > self.tau:
                new_pos = self.pop[idx].solution - cf * phi * self.generator.random() * (self.pop[idx].solution - m)
            else:
                new_pos = self.pop[idx].solution + cf * phi * self.generator.random() * (self.pop[idx].solution - m)
            self.tau = new_tau
            self.pop[idx] = self.generate_agent(self.correct_solution(new_pos))


_GPU_REASON = "project dense kernels use the common NumPy/CuPy backend boundary"
_CUSTOM = (
    OptimizerAdapter("DE", GPUOriginalDE, ("OriginalDE",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("JADE", GPUJADE, (), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("SHADE", GPUOriginalSHADE, ("OriginalSHADE",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("PSO", GPUOriginalPSO, ("OriginalPSO",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("WOA", GPUOriginalWOA, ("OriginalWOA",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("HHO", GPUOriginalHHO, ("OriginalHHO",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("GOA", GPUOriginalGOA, ("OriginalGOA",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("SA", GPUOriginalSA, ("OriginalSA",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("BRO", GPUOriginalBRO, ("OriginalBRO",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("RUN", GPUOriginalRUN, ("OriginalRUN",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("FOX", GPUOriginalFOX, ("OriginalFOX",), "GENERIC_GPU", _GPU_REASON, MEALPY_GPU_PARAMETERS),
    OptimizerAdapter("MaCRO-DE", MaCRO_DE, ("MACRO-DE", "MACRO_DE"), "GENERIC_GPU", _GPU_REASON, COMMON_PARAMETERS, _diversity_work, _diversity_memory),
    OptimizerAdapter("DSADE", DSADE, ("DSA-DE", "DSA_DE"), "GENERIC_GPU", _GPU_REASON, COMMON_PARAMETERS, _diversity_work, _diversity_memory),
    OptimizerAdapter("DSADE_AWAD", DSADE_AWAD, ("DSADE-AWAD",), "GENERIC_GPU", _GPU_REASON, COMMON_PARAMETERS, _diversity_work, _diversity_memory),
    OptimizerAdapter("DE-AWAD", DE_AWAD, ("DE_AWAD",), "GENERIC_GPU", _GPU_REASON, COMMON_PARAMETERS, _diversity_work, _diversity_memory),
    OptimizerAdapter("DE-DiversitySelection", DE_DiversitySelection, ("DE_DIVERSITYSELECTION",), "GENERIC_GPU", _GPU_REASON, COMMON_PARAMETERS, _diversity_work, _diversity_memory),
    OptimizerAdapter("DE-Mahalanobis", DE_Mahalanobis, ("DE_MAHALANOBIS",), "GENERIC_GPU", _GPU_REASON, COMMON_PARAMETERS, _diversity_work, _diversity_memory),
    OptimizerAdapter("DBO", DBOOptimizer, (), "PARTIAL", "per-agent fitness-dependent control remains on CPU", COMMON_PARAMETERS),
    OptimizerAdapter("OriginalDMOA", SafeOriginalDMOA, (), "PARTIAL", "project numerical guard is sequential and CPU-resident", COMMON_PARAMETERS),
)

CUSTOM_ADAPTERS = {item.canonical_name: item for item in _CUSTOM}
CUSTOM_ALIASES = {
    alias.upper(): item.canonical_name
    for item in _CUSTOM for alias in (item.canonical_name, *item.aliases)
}


def get_custom_adapter(name: str) -> OptimizerAdapter | None:
    canonical = CUSTOM_ALIASES.get(str(name).strip().upper())
    return CUSTOM_ADAPTERS.get(canonical) if canonical else None


def custom_optimizer_names() -> tuple[str, ...]:
    return tuple(CUSTOM_ADAPTERS)
