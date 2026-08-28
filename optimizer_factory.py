"""Dynamic optimizer resolution, construction, and device policy."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mealpy

from numerical_backend import UnsupportedOptimizerDeviceError, normalize_compute_device
from optimizer_adapters import CUSTOM_ADAPTERS, get_custom_adapter
from optimizer_capabilities import CapabilityReport, OptimizerCapability, analyze_optimizer
from optimizer_interceptor import Workload, estimate_dense_kernel_bytes


@dataclass(frozen=True)
class ResolvedOptimizer:
    requested_name: str
    canonical_name: str
    optimizer_class: type
    capability: CapabilityReport


@dataclass(frozen=True)
class ExecutionStrategy:
    optimizer_compute_device: str
    gpu_owner_count: int
    name: str
    reason: str
    estimated_gpu_bytes: int
    capability: OptimizerCapability


@lru_cache(maxsize=1)
def installed_optimizers() -> dict[str, type]:
    return mealpy.get_all_optimizers(verbose=False)


@lru_cache(maxsize=1)
def _installed_names() -> dict[str, str]:
    return {name.casefold(): name for name in installed_optimizers()}


def resolve_optimizer(name: str) -> ResolvedOptimizer:
    raw = str(name).strip()
    custom = get_custom_adapter(raw)
    if custom is not None:
        canonical, cls = custom.canonical_name, custom.optimizer_class
    else:
        names = _installed_names()
        candidates = [raw, f"Original{raw}", f"Base{raw}", f"Dev{raw}"]
        if raw.casefold() == "dmoa":
            candidates.insert(0, "DevDMOA")
        canonical = next((names[item.casefold()] for item in candidates if item.casefold() in names), None)
        if canonical is None:
            raise ValueError(f"Unknown MEALPY optimizer '{name}'. Tried: {', '.join(candidates)}")
        cls = installed_optimizers()[canonical]
    return ResolvedOptimizer(raw, canonical, cls, analyze_optimizer(canonical, cls))


def optimizer_acronym(name: str) -> str:
    canonical = resolve_optimizer(name).canonical_name
    if canonical in CUSTOM_ADAPTERS:
        return canonical
    for prefix in ("Original", "Base", "Dev"):
        if canonical.startswith(prefix) and len(canonical) > len(prefix):
            return canonical[len(prefix):]
    return canonical


def _constructor_kwargs(resolved: ResolvedOptimizer, settings: Any) -> dict[str, Any]:
    signature = inspect.signature(resolved.optimizer_class.__init__)
    params = signature.parameters
    accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values())
    adapter = resolved.capability.adapter
    mapping = adapter.parameter_map if adapter else (("epochs", "epoch"), ("pop_size", "pop_size"))
    kwargs = {}
    for setting_name, parameter_name in mapping:
        if parameter_name not in params and not accepts_kwargs:
            continue
        if hasattr(settings, setting_name):
            kwargs[parameter_name] = getattr(settings, setting_name)
        elif setting_name == "optimizer_compute_device" and hasattr(settings, "compute_device"):
            kwargs[parameter_name] = getattr(settings, "compute_device")
    return kwargs


def build_optimizer(name: str, settings: Any):
    resolved = resolve_optimizer(name)
    optimizer = resolved.optimizer_class(**_constructor_kwargs(resolved, settings))
    optimizer.backend_capability = resolved.capability
    return optimizer


def select_execution_strategy(
    requested_device: str,
    resolved: ResolvedOptimizer,
    workload: Workload,
    gpu_backend=None,
    minimum_kernel_work: int = 100_000,
    minimum_epochs: int = 2,
) -> ExecutionStrategy:
    requested = normalize_compute_device(requested_device)
    report, adapter = resolved.capability, resolved.capability.adapter
    if requested == "cpu":
        return ExecutionStrategy("cpu", 0, "forced-cpu", "CPU mode requested", 0, report.capability)
    if requested == "gpu" and not report.supports_gpu:
        raise UnsupportedOptimizerDeviceError(
            f"{resolved.canonical_name} is {report.capability.value}: {report.reason}"
        )
    if not report.supports_gpu:
        raise UnsupportedOptimizerDeviceError(
            f"{resolved.canonical_name} is {report.capability.value}: {report.reason}"
        )
    if gpu_backend is None or not getattr(gpu_backend, "uses_gpu", False):
        reason = getattr(gpu_backend, "fallback_reason", None) or "CUDA/runtime backend is unavailable"
        raise UnsupportedOptimizerDeviceError(reason)

    pop, dims = workload.population_size, workload.dimensions
    estimated = adapter.memory_bytes(pop, dims) if adapter and adapter.memory_bytes else estimate_dense_kernel_bytes(workload)
    if requested == "hybrid":
        reason = "common hybrid policy selected validated GPU kernels with CPU fitness"
        name = "hybrid-gpu-single-owner"
    else:
        reason, name = "strict GPU mode selected validated kernels", "forced-gpu-single-owner"
    return ExecutionStrategy("gpu", 1, name, reason, estimated, report.capability)


def list_available_optimizers() -> str:
    rows = [(optimizer_acronym(name), name) for name in sorted(installed_optimizers(), key=str.casefold)]
    width = max((len(display) for display, _ in rows), default=0)
    lines = [f"{display:<{width}} -> {name}" for display, name in rows]
    lines.extend(("", "Custom:", *CUSTOM_ADAPTERS.keys()))
    return "\n".join(lines)
