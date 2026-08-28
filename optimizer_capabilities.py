"""Conservative, source-derived MEALPY backend capability classification."""
from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import ModuleType

from optimizer_adapters import OptimizerAdapter, get_custom_adapter


class OptimizerCapability(str, Enum):
    GENERIC_GPU = "GENERIC_GPU"
    PARTIAL = "PARTIAL"
    CPU_ONLY = "CPU_ONLY"
    ADAPTER_REQUIRED = "ADAPTER_REQUIRED"


@dataclass(frozen=True)
class CapabilityReport:
    optimizer_name: str
    optimizer_class: type
    capability: OptimizerCapability
    reason: str
    numpy_operations: tuple[str, ...] = ()
    hazards: tuple[str, ...] = ()
    dense_population_regions: int = 0
    adapter: OptimizerAdapter | None = None

    @property
    def supports_gpu(self) -> bool:
        return self.capability is OptimizerCapability.GENERIC_GPU


_DENSE_PATTERNS = (
    "agent.solution for agent in", "item.solution for item in",
    "np.stack(", "np.vstack(", "np.cov(", "cdist(",
)
_CPU_ONLY_PATTERNS = (
    "PermutationVar", "CategoricalVar", "dtype=object",
)


def _class_source(optimizer_class: type) -> str:
    chunks = []
    for cls in reversed(optimizer_class.__mro__):
        if cls is object:
            continue
        try:
            chunks.append(inspect.getsource(cls))
        except (OSError, TypeError):
            pass
    return "\n".join(chunks)


def _analyze_calls(source: str, module: ModuleType | None) -> tuple[set[str], set[str]]:
    numpy_operations, hazards = set(), set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return numpy_operations, {"source could not be parsed safely"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            root_name = root.id if isinstance(root, ast.Name) else ""
            if root_name in {"np", "numpy"}:
                numpy_operations.add(func.attr)
            target = vars(module).get(root_name) if module is not None else None
            target_module = getattr(target, "__module__", getattr(target, "__name__", ""))
            if root_name in {"scipy", "stats", "special", "spatial", "qmc"} or str(target_module).startswith("scipy"):
                hazards.add(f"SciPy call through {root_name}.{func.attr}")
        elif isinstance(func, ast.Name) and module is not None:
            target = vars(module).get(func.id)
            target_module = getattr(target, "__module__", "")
            if target_module.startswith("scipy"):
                hazards.add(f"SciPy call {target_module}.{func.id}")
    return numpy_operations, hazards


@lru_cache(maxsize=None)
def analyze_optimizer(optimizer_name: str, optimizer_class: type) -> CapabilityReport:
    adapter = get_custom_adapter(optimizer_name)
    if adapter is not None:
        return CapabilityReport(
            optimizer_name, optimizer_class, OptimizerCapability(adapter.capability),
            adapter.reason, adapter=adapter,
        )

    source = _class_source(optimizer_class)
    module = inspect.getmodule(optimizer_class)
    numpy_operations, hazards = _analyze_calls(source, module)
    dense_regions = sum(source.count(pattern) for pattern in _DENSE_PATTERNS)
    cpu_markers = tuple(pattern for pattern in _CPU_ONLY_PATTERNS if pattern in source)

    if hazards:
        capability = OptimizerCapability.ADAPTER_REQUIRED
        reason = "update path uses numerical APIs outside the validated backend namespace"
    elif cpu_markers:
        capability = OptimizerCapability.CPU_ONLY
        reason = "optimizer uses non-dense/object-valued search state"
    elif dense_regions:
        capability = OptimizerCapability.PARTIAL
        reason = (
            "dense population math was detected, but no complete safe GPU region is "
            "declared; CPU execution preserves the original method"
        )
    elif numpy_operations or "solution" in source:
        capability = OptimizerCapability.PARTIAL
        reason = "numerical work is per-agent or intertwined with CPU control/fitness"
    else:
        capability = OptimizerCapability.CPU_ONLY
        reason = "no useful dense numerical region was detected"

    return CapabilityReport(
        optimizer_name, optimizer_class, capability, reason,
        tuple(sorted(numpy_operations)), tuple(sorted(hazards)), dense_regions,
    )


def capability_summary(report: CapabilityReport) -> str:
    return f"{report.optimizer_name}: {report.capability.value} - {report.reason}"
