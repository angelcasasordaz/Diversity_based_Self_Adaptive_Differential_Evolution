import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from mafese import Data, MhaSelector, get_dataset
from mafese.utils.mealpy_util import FeatureSelectionProblem
from mafese.utils.estimator import get_general_estimator
from mealpy.swarm_based.DMOA import OriginalDMOA
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dbo_optimizer import DBOOptimizer
from dsade_optimizer import DSADE
from dsade_awad_optimizer import DSADE_AWAD
from macro_de_optimizer import MaCRO_DE
from algorithm_acronym_list import (
    list_available_optimizers,
    optimizer_acronym,
    optimizer_class,
    resolve_optimizer_name,
)

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

DATASET_SOURCE = "codesmell"
# Options:
# "codesmell"
# "mafese"

CODE_SMELL_DATASET_DIR = "Original"

CODE_SMELL_DATASETS = None
# None -> run all CSV files in CODE_SMELL_DATASET_DIR.
#
# Example:
# CODE_SMELL_DATASETS = [
#     "FeatureEnvy",
#     "LongMethod",
#     "GodClass",
# ]

MAFESE_DATASET_SUITE = "test14"

OPTIMIZERS = [
    # "MaCRO-DE",
    # "DSADE",
    # "DE",
    # "JADE",
    # "SHADE",
    # "PSO",
    # "WOA",
    # "GWO",
    # "HHO",
    # "GOA",
    # "SA",
    # "BRO",
    # "RUN",
    # "WOA",
    "FOX",
]

ESTIMATORS = [
    "knn",
    "svm",
    "rf",
]

TRANSFER_FUNCTIONS = [
    "vstf_01",
]

RUNS = 1
EPOCHS = 5
POP_SIZE = 50

CHART_CMAP = "tab20"

# Qualitative:
# "tab10"
# "tab20"
# "Set1"
# "Set2"
# "Set3"
# "Dark2"
# "Paired"
# "Accent"

PARALLEL = True
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)

EXP_ID = 627
TEST_SIZE = 0.2
RANDOM_STATE = 2
SEED_BASE = 1234
OUTPUT_ROOT = "."
REUSE_CACHE = False
FIGURES_ONLY = False

DSADE_BETA_MIN = 0.2
DSADE_BETA_MAX = 0.8
DSADE_PCR = 0.2
DSADE_MAHAL_Q = 0.68

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

TEST_DATASETS_CLASSIFICATION_14 = [
    "BreastCancer",
    "BreastEW",
    "Glass",
    "HeartEW",
    "Ionosphere",
    "Lymphography",
    "Sonar",
    "SpectEW",
    "Tic-tac-toe",
    "Wine",
    "WaveformEW",
    "Zoo",
]
SUPPORTED_ESTIMATORS = ["knn", "svm", "rf", "adaboost", "xgb", "tree", "ann"]
SUPPORTED_TRANSFER_FUNCTIONS = [
    "vstf_01",
    "vstf_02",
    "vstf_03",
    "vstf_04",
    "sstf_01",
    "sstf_02",
    "sstf_03",
    "sstf_04",
]

@dataclass
class Paths:
    exp_tag: str
    fig_dir: str
    res_dir: str
    cache_dir: str

@dataclass
class DatasetSpec:
    name: str
    source: str
    path: Optional[Path] = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-selection comparison framework with cache and multi-run support")
    parser.add_argument("--exp-id", type=int, default=EXP_ID, help="Numeric experiment ID")
    parser.add_argument("--dataset-source", default=DATASET_SOURCE, choices=["mafese", "codesmell"], help="Dataset source")
    parser.add_argument("--dataset-suite", default=MAFESE_DATASET_SUITE, choices=["test14"], help="MAFESE dataset suite")
    parser.add_argument("--dataset-dir", default=CODE_SMELL_DATASET_DIR, help="Directory containing Code Smell CSV files")
    parser.add_argument("--datasets", nargs="*", default=CODE_SMELL_DATASETS, help="Code Smell datasets to run, with or without .csv")
    parser.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS), help="Optimizer names or acronyms")
    parser.add_argument("--estimators", nargs="+", default=list(ESTIMATORS), help="Classifier names")
    parser.add_argument("--transfer-functions", nargs="+", default=list(TRANSFER_FUNCTIONS), help="Transfer functions")
    parser.add_argument("--runs", type=int, default=RUNS, help="Independent runs per combination")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Optimizer iterations")
    parser.add_argument("--pop-size", type=int, default=POP_SIZE, help="Population size")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Holdout ratio")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Train/test split seed")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE, help="Base seed for runs")
    parser.add_argument("--output-root", default=OUTPUT_ROOT, help="Root directory for Figures and Results")
    parser.add_argument("--reuse-cache", action="store_true", default=REUSE_CACHE, help="Reuse cache when available")
    parser.add_argument("--figures-only", action="store_true", default=FIGURES_ONLY, help="Regenerate only charts from existing cache")
    parser.add_argument("--list-optimizers", action="store_true", help="List available optimizers and exit")
    parser.add_argument("--parallel", default="yes" if PARALLEL else "no", choices=["yes", "no"], help="Run independent runs in parallel: yes/no")
    parser.add_argument("--n-workers", type=int, default=N_WORKERS, help="Number of parallel worker processes")
    parser.add_argument("--dsade-beta-min", type=float, default=DSADE_BETA_MIN)
    parser.add_argument("--dsade-beta-max", type=float, default=DSADE_BETA_MAX)
    parser.add_argument("--dsade-pcr", type=float, default=DSADE_PCR)
    parser.add_argument("--dsade-mahal-q", type=float, default=DSADE_MAHAL_Q)
    return parser.parse_args()

def resolve_optimizers(args: argparse.Namespace) -> List[str]:
    optimizers = []
    for name in args.optimizers:
        resolved_name = resolve_optimizer_name(name)
        display_name = resolved_name if resolved_name == "OriginalDMOA" else optimizer_acronym(resolved_name)
        if display_name not in optimizers:
            optimizers.append(display_name)
    return optimizers

def print_available_optimizers() -> None:
    print(list_available_optimizers())

def validate_selection_options(args: argparse.Namespace) -> None:
    invalid_estimators = [e for e in args.estimators if e not in SUPPORTED_ESTIMATORS]
    if invalid_estimators:
        raise ValueError(
            f"Unsupported classifiers: {invalid_estimators}. "
            f"Valid values: {', '.join(SUPPORTED_ESTIMATORS)}"
        )
    invalid_tf = [tf for tf in args.transfer_functions if tf not in SUPPORTED_TRANSFER_FUNCTIONS]
    if invalid_tf:
        raise ValueError(
            f"Unsupported transfer functions: {invalid_tf}. "
            f"Valid values: {', '.join(SUPPORTED_TRANSFER_FUNCTIONS)}"
        )

def make_paths(args: argparse.Namespace) -> Paths:
    exp_tag = f"EXP{args.exp_id:03d}"
    fig_dir = os.path.join(args.output_root, "Figures", exp_tag)
    res_dir = os.path.join(args.output_root, "Results", exp_tag)
    cache_dir = os.path.join(res_dir, "cache")
    for p in (fig_dir, res_dir, cache_dir):
        os.makedirs(p, exist_ok=True)
    return Paths(exp_tag=exp_tag, fig_dir=fig_dir, res_dir=res_dir, cache_dir=cache_dir)

def resolve_mafese_dataset_names(args: argparse.Namespace) -> List[str]:
    if args.dataset_suite == "test14":
        return list(TEST_DATASETS_CLASSIFICATION_14)
    raise ValueError(f"Unsupported dataset suite: {args.dataset_suite}")

def resolve_codesmell_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Code Smell dataset directory not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"--dataset-dir must be a directory: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("*.csv"), key=lambda p: p.stem.lower())
    if args.datasets is not None:
        if not args.datasets:
            raise ValueError("--datasets requires at least one dataset name when used")
        requested = list(dict.fromkeys(Path(name).stem for name in args.datasets))
        by_name = {path.stem: path for path in csv_files}
        missing = [name for name in requested if name not in by_name]
        if missing:
            available = ", ".join(sorted(by_name)) or "(none)"
            raise FileNotFoundError(
                f"Code Smell datasets not found in {dataset_dir}: {missing}. "
                f"Available datasets: {available}"
            )
        csv_files = [by_name[name] for name in requested]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    return [DatasetSpec(name=path.stem, source="codesmell", path=path) for path in csv_files]

def resolve_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    if args.dataset_source == "mafese":
        if args.datasets is not None:
            raise ValueError("--datasets is only supported with --dataset-source codesmell")
        return [DatasetSpec(name=name, source="mafese") for name in resolve_mafese_dataset_names(args)]
    if args.dataset_source == "codesmell":
        return resolve_codesmell_dataset_specs(args)
    raise ValueError(f"Unsupported dataset source: {args.dataset_source}")

def validate_xy(dataset_name: str, X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"Dataset '{dataset_name}': X must be a 2D matrix, shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Dataset '{dataset_name}': y must be a 1D vector, shape={y.shape}")
    if X.shape[1] < 1:
        raise ValueError(f"Dataset '{dataset_name}': X has no numeric features")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Dataset '{dataset_name}': X and y have different sample counts "
            f"({X.shape[0]} vs {y.shape[0]})"
        )
    if np.unique(y).size < 2:
        raise ValueError(f"Dataset '{dataset_name}': y must contain at least two classes")
    if not np.isfinite(X).all():
        raise ValueError(f"Dataset '{dataset_name}': X contains NaN or infinite values after cleaning")

def load_mafese_dataset(dataset_name: str) -> Tuple[str, np.ndarray, np.ndarray]:
    mafese_data = get_dataset(dataset_name)
    if mafese_data is None:
        raise ValueError(
            f"MAFESE could not load '{dataset_name}'. "
            "Verify that it exists in the MAFESE 'test14' suite."
        )
    X = np.asarray(mafese_data.X, dtype=np.float64)
    y = np.asarray(mafese_data.y).astype(np.int32)
    validate_xy(dataset_name, X, y)
    return dataset_name, X, y

def load_codesmell_dataset(csv_path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    dataset_name = csv_path.stem
    df = pd.read_csv(csv_path)
    is_columns = [col for col in df.columns if str(col).startswith("is_")]
    if len(is_columns) != 1:
        raise ValueError(
            f"Code Smell dataset '{dataset_name}' ({csv_path}): expected exactly one "
            f"'is_*' target column, found {len(is_columns)}: {is_columns}"
        )

    target_column = is_columns[0]
    y = df[target_column].astype(int).to_numpy()
    X = (
        df.drop(columns=is_columns)
        .select_dtypes(include=["number"])
        .to_numpy(dtype=np.float64)
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(np.int32)

    validate_xy(dataset_name, X, y)
    return dataset_name, X, y

def load_dataset(spec: DatasetSpec) -> Tuple[str, np.ndarray, np.ndarray]:
    if spec.source == "mafese":
        return load_mafese_dataset(spec.name)
    if spec.source == "codesmell":
        if spec.path is None:
            raise ValueError(f"Code Smell dataset '{spec.name}' has no associated CSV path")
        return load_codesmell_dataset(spec.path)
    raise ValueError(f"Unsupported dataset source: {spec.source}")

def print_dataset_summary(args: argparse.Namespace, dataset_specs: List[DatasetSpec]) -> None:
    print(f"Dataset source: {args.dataset_source}")
    if args.dataset_source == "mafese":
        print(f"Dataset suite: {args.dataset_suite} ({len(dataset_specs)} datasets)")
        print(f"Datasets: {', '.join(spec.name for spec in dataset_specs)}")
        return

    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Datasets found: {len(dataset_specs)}")
    print()
    for spec in dataset_specs:
        print(f"- {spec.name}")


class SafeOriginalDMOA(OriginalDMOA):
    """OriginalDMOA with numerically safe updates for binary feature-selection spaces."""

    def evolve(self, epoch):
        cf = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / (mean_cost + self.EPSILON))

        for idx in range(0, self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx, alpha}))
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[alpha].solution + phi * (self.pop[alpha].solution - self.pop[k].solution)
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        sm = np.zeros(self.pop_size)
        for idx in range(0, self.pop_size):
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[idx].solution + phi * (self.pop[idx].solution - self.pop[k].solution)
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            current_fit = self.pop[idx].target.fitness
            trial_fit = agent.target.fitness
            denom = max(abs(trial_fit), abs(current_fit), self.EPSILON)
            sm[idx] = (trial_fit - current_fit) / denom
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        for idx in range(0, self.n_baby_sitter):
            if self.C[idx] >= self.L:
                self.pop[idx] = self.generate_agent()
                self.C[idx] = 0

        new_tau = np.mean(sm)
        for idx in range(0, self.pop_size):
            m = np.full(self.problem.n_dims, sm[idx], dtype=float)
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > self.tau:
                new_pos = self.pop[idx].solution - cf * phi * self.generator.random() * (self.pop[idx].solution - m)
            else:
                new_pos = self.pop[idx].solution + cf * phi * self.generator.random() * (self.pop[idx].solution - m)
            self.tau = new_tau
            new_pos = self.correct_solution(new_pos)
            self.pop[idx] = self.generate_agent(new_pos)


def _instantiate_mealpy_optimizer(optimizer_cls, args: argparse.Namespace):
    init_params = inspect.signature(optimizer_cls.__init__).parameters
    kwargs = {}
    if "epoch" in init_params:
        kwargs["epoch"] = args.epochs
    if "pop_size" in init_params:
        kwargs["pop_size"] = args.pop_size
    return optimizer_cls(**kwargs)

def build_optimizer(name: str, args: argparse.Namespace):
    resolved_name = resolve_optimizer_name(name)
    if resolved_name == "DSADE":
        return DSADE(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )
    if resolved_name == "DSADE_AWAD":
        return DSADE_AWAD(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )
    if resolved_name == "MaCRO-DE":
        return MaCRO_DE(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )
    if resolved_name == "DBO":
        return DBOOptimizer(epoch=args.epochs, pop_size=args.pop_size)

    # Simple "DMOA" is resolved by algorithm_acronym_list to DevDMOA. This
    # explicit OriginalDMOA path keeps the local numerical guard for users who
    # intentionally request the original variant in binary FS experiments.
    if resolved_name == "OriginalDMOA":
        return SafeOriginalDMOA(epoch=args.epochs, pop_size=args.pop_size)

    return _instantiate_mealpy_optimizer(optimizer_class(resolved_name), args)

def build_cache_signature(args: argparse.Namespace) -> str:
    payload = {
        "optimizers": [resolve_optimizer_name(name) for name in args.optimizers],
        "transfer_functions": list(args.transfer_functions),
        "runs": int(args.runs),
        "epochs": int(args.epochs),
        "pop_size": int(args.pop_size),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "seed_base": int(args.seed_base),
        "obj_name": "AS",
        "fitness_mode": "minimize_metric_loss_plus_feature_ratio_v1",
        "dsade_beta_min": float(args.dsade_beta_min),
        "dsade_beta_max": float(args.dsade_beta_max),
        "dsade_pcr": float(args.dsade_pcr),
        "dsade_mahal_q": float(args.dsade_mahal_q),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]

def build_alg_label(method: str, transfer_function: str, classifier: str, show_tf: bool, show_cls: bool) -> str:
    parts = [optimizer_acronym(method).upper()]
    if show_tf:
        parts.append(str(transfer_function).upper())
    if show_cls:
        parts.append(classifier.upper())
    return "_".join(parts)

def build_legacy_alg_label(method: str, transfer_function: str, classifier: str, show_tf: bool, show_cls: bool) -> str:
    parts = [resolve_optimizer_name(method).upper()]
    if show_tf:
        parts.append(str(transfer_function).upper())
    if show_cls:
        parts.append(classifier.upper())
    return "_".join(parts)

def muted_color_palette(n: int) -> np.ndarray:
    cmap = plt.get_cmap(CHART_CMAP, max(n, 1))
    return cmap(np.arange(max(n, 1)))[:, :3]


class RobustClassificationFeatureSelectionProblem(FeatureSelectionProblem):
    """Classification objective that tolerates validation folds missing classes."""

    def __init__(self, bounds=None, minmax=None, data=None, estimator=None, metric_class=None,
                 obj_name=None, obj_paras=None, fit_weights=(0.9, 0.1), fit_sign=None, **kwargs):
        super().__init__(
            bounds=bounds,
            minmax="min",
            data=data,
            estimator=estimator,
            metric_class=metric_class,
            obj_name=obj_name,
            obj_paras=obj_paras,
            fit_weights=fit_weights,
            fit_sign=1,
            **kwargs,
        )

    def obj_func(self, solution):
        x = self.decode_solution(solution)["my_var"]
        cols = np.flatnonzero(x)
        self.estimator.fit(self.data.X_train[:, cols], self.data.y_train)
        y_valid_pred = self.estimator.predict(self.data.X_test[:, cols])
        obj = self._score(self.data.y_test, y_valid_pred)
        feature_ratio = np.sum(x) / self.n_dims
        fitness = self.fit_weights[0] * (1.0 - obj) + self.fit_weights[1] * feature_ratio
        return [fitness, obj, np.sum(x)]

    def _score(self, y_true, y_pred) -> float:
        metric = str(self.obj_name).upper()
        average = (self.obj_paras or {}).get("average", "macro")
        labels = np.unique(np.concatenate((np.asarray(self.data.y_train), np.asarray(y_true), np.asarray(y_pred))))

        if metric == "AS":
            return float(accuracy_score(y_true, y_pred))
        if metric == "PS":
            return float(precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0))
        if metric == "RS":
            return float(recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0))
        if metric == "F1S":
            return float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0))

        evaluator = self.metric_class(y_true, y_pred)
        try:
            return float(evaluator.get_metric_by_name(self.obj_name, paras=self.obj_paras)[self.obj_name])
        except ValueError as err:
            if "Invalid y_pred" not in str(err):
                raise
            paras = dict(self.obj_paras or {})
            paras["labels"] = labels
            return float(evaluator.get_metric_by_name(self.obj_name, paras=paras)[self.obj_name])


def run_single(data: Data, estimator: str, optimizer_name: str, tf: str, args: argparse.Namespace, seed: int):
    logging.disable(logging.INFO)
    np.random.seed(seed)
    optimizer = build_optimizer(optimizer_name, args)
    selector_kwargs = dict(
        problem="classification",
        estimator=estimator,
        optimizer=optimizer,
        optimizer_paras=({"epoch": args.epochs, "pop_size": args.pop_size} if isinstance(optimizer, str) else None),
        obj_name="AS",
    )
    init_params = inspect.signature(MhaSelector.__init__).parameters
    if "transfer_func" in init_params:
        selector_kwargs["transfer_func"] = tf

    selector = MhaSelector(**selector_kwargs)

    t0 = time.time()
    fit_params = inspect.signature(selector.fit).parameters
    fit_kwargs = {}
    if "transfer_func" in fit_params:
        fit_kwargs["transfer_func"] = tf
    if "verbose" in fit_params:
        fit_kwargs["verbose"] = False
    if "fs_problem" in fit_params:
        fit_kwargs["fs_problem"] = RobustClassificationFeatureSelectionProblem
    selector.fit(data.X_train, data.y_train, **fit_kwargs)
    runtime = time.time() - t0

    fit_curve = np.array(selector.optimizer.history.list_global_best_fit, dtype=float)
    fit_final = float(fit_curve[-1]) if fit_curve.size else np.nan

    selected = selector.transform(data.X_train)
    n_features = int(selected.shape[1])

    try:
        metrics = selector.evaluate(estimator=selector.estimator, data=data, metrics=["AS", "PS", "RS", "F1S"])
        as_test = float(metrics.get("AS_test", np.nan))
        ps_test = float(metrics.get("PS_test", np.nan))
        rs_test = float(metrics.get("RS_test", np.nan))
        f1_test = float(metrics.get("F1S_test", np.nan))
    except ValueError as err:
        # Permetrics can fail when y_pred contains labels absent in y_test.
        if "Invalid y_pred" not in str(err):
            raise
        X_train_sel = selector.transform(data.X_train)
        X_test_sel = selector.transform(data.X_test)
        if isinstance(selector.estimator, str):
            est = get_general_estimator("classification", selector.estimator)
        else:
            est = clone(selector.estimator)
        est.fit(X_train_sel, data.y_train)
        y_pred = est.predict(X_test_sel)
        labels = np.unique(np.concatenate((np.asarray(data.y_test), np.asarray(y_pred))))
        as_test = float(accuracy_score(data.y_test, y_pred))
        ps_test = float(precision_score(data.y_test, y_pred, labels=labels, average="macro", zero_division=0))
        rs_test = float(recall_score(data.y_test, y_pred, labels=labels, average="macro", zero_division=0))
        f1_test = float(f1_score(data.y_test, y_pred, labels=labels, average="macro", zero_division=0))

    return {
        "as_test": 100.0 * as_test,
        "ps_test": ps_test,
        "rs_test": rs_test,
        "f1_test": f1_test,
        "fit_final": fit_final,
        "n_features": n_features,
        "runtime": runtime,
        "curve": fit_curve,
    }


def run_single_parallel_task(task: dict):
    data_split = task["data_split"]
    data = Data()
    data.set_train_test(
        X_train=data_split["X_train"],
        y_train=data_split["y_train"],
        X_test=data_split["X_test"],
        y_test=data_split["y_test"],
    )
    out = run_single(
        data,
        task["estimator"],
        task["method"],
        task["tf"],
        task["args"],
        task["seed"],
    )
    return task["run"], out


def execute_pending_runs(
    data: Data,
    estimator: str,
    method: str,
    tf: str,
    args: argparse.Namespace,
    pending_runs: List[int],
    on_run_complete=None,
):
    if args.parallel != "yes" or len(pending_runs) <= 1:
        completed = []
        for run in pending_runs:
            item = (run, run_single(data, estimator, method, tf, args, args.seed_base + run))
            if on_run_complete is not None:
                on_run_complete(*item)
            completed.append(item)
        return completed

    data_split = {
        "X_train": data.X_train,
        "y_train": data.y_train,
        "X_test": data.X_test,
        "y_test": data.y_test,
    }
    max_workers = min(args.n_workers, len(pending_runs))
    tasks = [
        {
            "run": run,
            "data_split": data_split,
            "estimator": estimator,
            "method": method,
            "tf": tf,
            "args": args,
            "seed": args.seed_base + run,
        }
        for run in pending_runs
    ]
    completed = []
    completed_by_run = {}
    next_run = min(pending_runs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_parallel_task, task) for task in tasks]
        for future in as_completed(futures):
            run, out = future.result()
            completed_by_run[run] = out
            while next_run in completed_by_run:
                item = (next_run, completed_by_run.pop(next_run))
                if on_run_complete is not None:
                    on_run_complete(*item)
                completed.append(item)
                next_run += 1
    return completed


def pad_mean_curves(curves: List[np.ndarray], target_len: int) -> np.ndarray:
    if not curves:
        return np.array([])
    mat = np.full((len(curves), target_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        c = np.asarray(curve, dtype=float).ravel()
        ln = min(target_len, c.size)
        mat[i, :ln] = c[:ln]
    return np.nanmean(mat, axis=0)

def build_label_payload(
    estimator: str,
    acc_runs: List[float],
    ps_runs: List[float],
    rs_runs: List[float],
    f1_runs: List[float],
    fit_runs: List[float],
    feat_runs: List[float],
    time_runs: List[float],
    curves: List[np.ndarray],
    epochs: int,
):
    curve_mean = pad_mean_curves(curves, epochs)
    return {
        "Estimator": estimator,
        "AccMean": float(np.nanmean(acc_runs)),
        "F1Mean": float(np.nanmean(f1_runs)),
        "PSMean": float(np.nanmean(ps_runs)),
        "RSMean": float(np.nanmean(rs_runs)),
        "FitMean": float(np.nanmean(fit_runs)),
        "FeatMean": float(np.nanmean(feat_runs)),
        "TimeMean": float(np.nanmean(time_runs)),
        "AccBest": float(np.nanmax(acc_runs)),
        "AccRuns": np.array(acc_runs, dtype=float),
        "F1Runs": np.array(f1_runs, dtype=float),
        "PSRuns": np.array(ps_runs, dtype=float),
        "RSRuns": np.array(rs_runs, dtype=float),
        "FitRuns": np.array(fit_runs, dtype=float),
        "FeatRuns": np.array(feat_runs, dtype=float),
        "TimeRuns": np.array(time_runs, dtype=float),
        "Curve": curve_mean,
        "CurvesAll": curves,
        "CompletedRuns": len(acc_runs),
    }

def save_cache(path: str, payload: dict):
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_cache(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_cache_safe(path: str, label: str):
    if not os.path.exists(path):
        return None
    try:
        return load_cache(path)
    except Exception as exc:
        print(f"[cache-warning] Could not load {label} '{path}': {exc}")
        return None

def load_results_from_cache(paths: Paths, args: argparse.Namespace, dataset_names: List[str], cache_sig: str) -> Dict[str, Dict]:
    results_struct = {}
    missing = []
    for dataset_name in dataset_names:
        results_struct[dataset_name] = {}
        for estimator in args.estimators:
            cache_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_results.pkl",
            )
            progress_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_progress.pkl",
            )
            payload = load_cache_safe(cache_file, "final cache")
            if payload is None:
                payload = load_cache_safe(progress_file, "partial checkpoint")
            if payload is None:
                missing.append(f"{dataset_name}/{estimator}")
                continue
            results_struct[dataset_name].update(payload)

    if missing:
        raise FileNotFoundError(
            "No cache files were found for: "
            + ", ".join(missing)
            + ". Run the full experiment or verify that the parameters match the existing cache."
        )
    return results_struct

def payload_completed_runs(payload: dict) -> int:
    total = 0
    for row in payload.values():
        if not isinstance(row, dict):
            continue
        total += int(row.get("CompletedRuns", len(row.get("AccRuns", []))))
    return total


def parse_result_label(label: str, args: argparse.Namespace) -> dict:
    label_upper = str(label).upper()
    optimizer_candidates = []
    for opt in args.optimizers:
        resolved = resolve_optimizer_name(opt)
        optimizer_candidates.extend([str(opt), optimizer_acronym(opt), resolved, optimizer_acronym(resolved)])
    ordered_opts = sorted(
        list(dict.fromkeys(optimizer_candidates)),
        key=len,
        reverse=True,
    )
    method = next(
        (
            opt
            for opt in ordered_opts
            if label_upper == opt.upper() or label_upper.startswith(f"{opt.upper()}_")
        ),
        str(label),
    )
    rest = label_upper[len(method):].lstrip("_") if method != str(label) else ""
    try:
        method = optimizer_acronym(resolve_optimizer_name(method))
    except ValueError:
        method = optimizer_acronym(method)

    estimator = ""
    for est in sorted([str(e) for e in args.estimators], key=len, reverse=True):
        est_upper = est.upper()
        if rest == est_upper:
            estimator = est.lower()
            rest = ""
            break
        suffix = f"_{est_upper}"
        if rest.endswith(suffix):
            estimator = est.lower()
            rest = rest[: -len(suffix)]
            break

    transfer_function = ""
    for tf in sorted(SUPPORTED_TRANSFER_FUNCTIONS, key=len, reverse=True):
        tf_upper = tf.upper()
        if rest == tf_upper or rest.startswith(f"{tf_upper}_") or f"_{tf_upper}" in rest:
            transfer_function = tf.lower()
            break

    return {"method": method, "transfer_function": transfer_function, "estimator": estimator}


def optimizer_display_label(name: str) -> str:
    return optimizer_acronym(str(name))

def is_dsade_method(name: str) -> bool:
    return str(name).upper() in {"MACRO-DE", "DSA-DE", "DSADE", "DSADE_AWAD", "DSADE-AWAD"}

def is_exact_dsade_method(name: str) -> bool:
    return str(name).upper() in {"DSA-DE", "DSADE"}

def optimizer_order_from_config(opt_order: List[str]) -> List[str]:
    ordered = []
    for name in opt_order:
        display_name = optimizer_acronym(name)
        if display_name not in ordered:
            ordered.append(display_name)
    return ordered

def prepare_plot_groups(df: pd.DataFrame, opt_order: List[str]) -> tuple[pd.DataFrame, List[str], Dict[str, str], Dict[str, str]]:
    if df.empty:
        return df.copy(), [], {}, {}

    plot_df = df.copy()
    if "TransferFunction" not in plot_df.columns:
        plot_df["TransferFunction"] = ""
    plot_df["TransferFunction"] = plot_df["TransferFunction"].fillna("").astype(str).str.lower()

    tf_counts = plot_df[plot_df["TransferFunction"] != ""].groupby("Optimizer")["TransferFunction"].nunique()
    variant_methods = set(tf_counts[tf_counts > 1].index)

    def make_group(row):
        opt = str(row["Optimizer"])
        tf = str(row["TransferFunction"]).lower()
        return f"{opt}_{tf.upper()}" if opt in variant_methods and tf else opt

    plot_df["PlotGroup"] = plot_df.apply(make_group, axis=1)
    group_meta = (
        plot_df[["PlotGroup", "Optimizer", "TransferFunction"]]
        .drop_duplicates()
        .set_index("PlotGroup")
        .to_dict("index")
    )

    present_methods = [str(meta["Optimizer"]) for meta in group_meta.values()]
    configured_order = optimizer_order_from_config(opt_order)
    method_order = [opt for opt in configured_order if opt in set(present_methods)]
    method_order.extend(opt for opt in present_methods if opt not in set(method_order))

    opts = []
    for opt in method_order:
        opt_groups = sorted(
            [g for g, meta in group_meta.items() if meta["Optimizer"] == opt],
            key=lambda g: (str(group_meta[g]["TransferFunction"]), g),
        )
        opts.extend(opt_groups)
    opts.extend(g for g in group_meta if g not in set(opts))

    colors = muted_color_palette(len(opts))
    color_map = {}
    label_map = {}
    for i, group in enumerate(opts):
        meta = group_meta[group]
        method = meta["Optimizer"]
        tf = meta["TransferFunction"]
        color_map[group] = colors[i]
        base_label = optimizer_display_label(method)
        label_map[group] = f"{base_label} {tf.upper()}" if tf and method in variant_methods else base_label

    return plot_df, opts, color_map, label_map

def plot_bar(values: np.ndarray, labels: List[str], ylabel: str, title: str, out_path: str):
    colors = muted_color_palette(len(labels))
    plt.figure(figsize=(10, 5), facecolor="white")
    bars = plt.bar(np.arange(len(labels)), values)
    for i, b in enumerate(bars):
        b.set_color(colors[i])
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _force_white_background(plt.gcf())
    plt.savefig(out_path, dpi=600, facecolor="white")
    plt.close()

def plot_lines(curves_by_label: Dict[str, np.ndarray], title: str, ylabel: str, out_path: str):
    styles = ["-", "--", ":", "-."]
    labels = list(curves_by_label.keys())
    colors = muted_color_palette(len(labels))
    plt.figure(figsize=(10, 5), facecolor="white")
    for i, label in enumerate(labels):
        curve = curves_by_label[label]
        if curve.size == 0:
            continue
        plt.plot(curve, linestyle=styles[i % len(styles)], color=colors[i], linewidth=2.4, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(4, max(1, len(labels))), frameon=False)
    plt.tight_layout()
    _force_white_background(plt.gcf())
    plt.savefig(out_path, dpi=600, facecolor="white")
    plt.close()

def export_global_excel(results_struct: Dict[str, Dict], dataset_names: List[str], out_path: str):
    all_labels = sorted(set().union(*[set(v.keys()) for v in results_struct.values()])) if results_struct else []
    if not all_labels:
        return []
    idx = pd.Index(dataset_names, name="Dataset")
    acc = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    ps = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    rs = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    f1 = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    fit = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    feat = pd.DataFrame(np.nan, index=idx, columns=all_labels)
    tim = pd.DataFrame(np.nan, index=idx, columns=all_labels)

    for ds, alg_data in results_struct.items():
        for lbl, row in alg_data.items():
            acc.loc[ds, lbl] = row.get("AccMean", np.nan)
            ps.loc[ds, lbl] = row.get("PSMean", np.nan)
            rs.loc[ds, lbl] = row.get("RSMean", np.nan)
            f1.loc[ds, lbl] = row.get("F1Mean", np.nan)
            fit.loc[ds, lbl] = row.get("FitMean", np.nan)
            feat.loc[ds, lbl] = row.get("FeatMean", np.nan)
            tim.loc[ds, lbl] = row.get("TimeMean", np.nan)

    try:
        with pd.ExcelWriter(out_path) as writer:
            acc.to_excel(writer, sheet_name="Accuracy")
            ps.to_excel(writer, sheet_name="Precision")
            rs.to_excel(writer, sheet_name="Recall")
            f1.to_excel(writer, sheet_name="F1Score")
            fit.to_excel(writer, sheet_name="Fitness")
            feat.to_excel(writer, sheet_name="Features")
            tim.to_excel(writer, sheet_name="Time")
        return [out_path]
    except ModuleNotFoundError:
        base = os.path.splitext(out_path)[0]
        paths = [
            f"{base}_Accuracy.csv",
            f"{base}_Precision.csv",
            f"{base}_Recall.csv",
            f"{base}_F1Score.csv",
            f"{base}_Fitness.csv",
            f"{base}_Features.csv",
            f"{base}_Time.csv",
        ]
        acc.to_csv(paths[0])
        ps.to_csv(paths[1])
        rs.to_csv(paths[2])
        f1.to_csv(paths[3])
        fit.to_csv(paths[4])
        feat.to_csv(paths[5])
        tim.to_csv(paths[6])
        return paths

def _run_stats(values, best_mode: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"Best": np.nan, "Worst": np.nan, "Mean": np.nan, "Std": np.nan}

    if best_mode == "min":
        best = np.nanmin(arr)
        worst = np.nanmax(arr)
    else:
        best = np.nanmax(arr)
        worst = np.nanmin(arr)

    std = 0.0 if arr.size == 1 else float(np.nanstd(arr, ddof=1))
    return {
        "Best": float(best),
        "Worst": float(worst),
        "Mean": float(np.nanmean(arr)),
        "Std": std,
    }

def export_statistical_excel(
    results_struct: Dict[str, Dict],
    dataset_names: List[str],
    optimizer_order: List[str],
    args: argparse.Namespace,
    out_path: str,
):
    metrics = {
        "Accuracy": ("AccRuns", "max"),
        "Precision": ("PSRuns", "max"),
        "Recall": ("RSRuns", "max"),
        "F1Score": ("F1Runs", "max"),
        "Fitness": ("FitRuns", "min"),
        "Features": ("FeatRuns", "min"),
        "Time": ("TimeRuns", "min"),
    }
    stats = ["Best", "Worst", "Mean", "Std"]
    optimizers = optimizer_order_from_config(optimizer_order)
    index = pd.MultiIndex.from_product([optimizers, stats], names=["Optimizer", "Statistic"])
    sheets = {
        sheet_name: pd.DataFrame(np.nan, index=index, columns=dataset_names)
        for sheet_name in metrics
    }

    for dataset_name in dataset_names:
        alg_data = results_struct.get(dataset_name, {})
        runs_by_optimizer = {
            sheet_name: {optimizer: [] for optimizer in optimizers}
            for sheet_name in metrics
        }
        for label, row in alg_data.items():
            parsed = parse_result_label(label, args)
            optimizer = optimizer_acronym(parsed["method"])
            if optimizer not in set(optimizers):
                continue
            for sheet_name, (run_key, _) in metrics.items():
                values = np.asarray(row.get(run_key, []), dtype=float).ravel()
                if values.size:
                    runs_by_optimizer[sheet_name][optimizer].append(values)

        for sheet_name, (_, best_mode) in metrics.items():
            for optimizer in optimizers:
                chunks = runs_by_optimizer[sheet_name][optimizer]
                values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
                stat_values = _run_stats(values, best_mode)
                for stat in stats:
                    sheets[sheet_name].loc[(optimizer, stat), dataset_name] = stat_values[stat]

    with pd.ExcelWriter(out_path) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name)
    return out_path

def generate_summary_dataframe(results_struct: Dict[str, Dict], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for dataset_name, alg_data in results_struct.items():
        for label, row in alg_data.items():
            parsed = parse_result_label(label, args)
            method = parsed["method"]
            estimator = parsed["estimator"] or None
            estimator = estimator or (row.get("Estimator") if isinstance(row, dict) else None) or (
                args.estimators[0] if len(args.estimators) == 1 else ""
            )
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Estimator": estimator,
                    "Optimizer": method,
                    "TransferFunction": parsed["transfer_function"],
                    "Configuracion": label,
                    "F1_test": float(row.get("F1Mean", np.nan)),
                    "AS_test": float(row.get("AccMean", np.nan)) / 100.0,
                    "PS_test": float(row.get("PSMean", np.nan)),
                    "RS_test": float(row.get("RSMean", np.nan)),
                    "N_Features_Selected": float(row.get("FeatMean", np.nan)),
                    "Runtime": float(row.get("TimeMean", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _plot_legend_patches(opts: List[str], color_map: Dict[str, str], label_map: Dict[str, str]) -> List[mpatches.Patch]:
    return [mpatches.Patch(color=color_map.get(o, "#888"), label=label_map.get(o, o)) for o in opts]


def _force_white_background(fig):
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for ax in fig.get_axes():
        ax.set_facecolor("white")


def _save_chart(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    _force_white_background(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_classifier_metric_grid_chart(df: pd.DataFrame, out_dir: str, opt_order: List[str]):
    if df.empty:
        return None

    plot_df = df.copy()
    plot_df["Estimator"] = plot_df["Estimator"].astype(str).str.lower()
    plot_df, opts, color_map, label_map = prepare_plot_groups(plot_df, opt_order)
    if not opts:
        return None
    method_by_group = plot_df.drop_duplicates("PlotGroup").set_index("PlotGroup")["Optimizer"].to_dict()

    metric_cols = ["AS_test", "PS_test", "RS_test", "F1_test"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_header_styles = [
        ("#d8e8f3", "#b8d3e6"),
        ("#d2efee", "#abd9d7"),
        ("#f7efd8", "#ead9ad"),
        ("#f9d5d9", "#edaeb8"),
    ]

    present_estimators = [str(e).lower() for e in plot_df["Estimator"].dropna().unique()]
    required_estimators = [e for e in ESTIMATORS if e in SUPPORTED_ESTIMATORS]
    estimators = [e for e in SUPPORTED_ESTIMATORS if e in set(required_estimators + present_estimators)]
    estimators += sorted(e for e in present_estimators if e not in set(estimators))
    if not estimators:
        return None

    grouped = plot_df.groupby(["Estimator", "PlotGroup"])[metric_cols].mean()
    n_rows = len(estimators)
    n_cols = len(metric_cols)
    fig_w = max(16.0, 4.2 * n_cols)
    fig_h = max(4.5, 2.75 * n_rows + 2.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, facecolor="#f7f9fc")
    x = np.arange(len(opts))
    colors = [color_map.get(opt, "#888888") for opt in opts]
    xlabels = [label_map.get(opt, opt) for opt in opts]

    for r, estimator in enumerate(estimators):
        for c, (metric, metric_label) in enumerate(zip(metric_cols, metric_labels)):
            ax = axes[r, c]
            ax.set_facecolor("#f3f6fa")
            vals = [
                float(grouped.loc[(estimator, opt), metric])
                if (estimator, opt) in grouped.index
                else np.nan
                for opt in opts
            ]
            edges = ["black" if is_dsade_method(method_by_group.get(opt)) else "none" for opt in opts]
            widths = [1.8 if is_dsade_method(method_by_group.get(opt)) else 0.0 for opt in opts]
            bars = ax.bar(x, vals, color=colors, edgecolor=edges, linewidth=widths, width=0.68)

            mean_val = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            if np.isfinite(mean_val):
                ax.axhline(mean_val, color="#d76c6c", linestyle="--", linewidth=0.9, alpha=0.8)

            for bar, value in zip(bars, vals):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.006,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                    color="#333333",
                )

            if not np.isfinite(vals).any():
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#777777",
                )

            ax.set_ylim(0.0, 1.10)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.24, linewidth=0.8)
            ax.set_axisbelow(True)

            if c == 0:
                ax.set_ylabel(estimator.upper(), fontsize=12, fontweight="bold", color="#19365f")
            if r == 0:
                face, edge = metric_header_styles[c]
                ax.set_title(
                    metric_label,
                    fontsize=12,
                    fontweight="bold",
                    color="#19365f",
                    pad=12,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor=face, edgecolor=edge),
                )

    legend = _plot_legend_patches(opts, color_map, label_map)
    fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 6), fontsize=9, framealpha=0.95)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    filename = "09_resultados_clasificador_metrica_todos_datasets.png"
    _save_chart(fig, out_dir, filename)
    return filename


def generate_notebook_style_charts(df: pd.DataFrame, out_dir: str, opt_order: List[str]):
    if df.empty:
        return []
    os.makedirs(out_dir, exist_ok=True)
    plot_df, opts, color_map, label_map = prepare_plot_groups(df, opt_order)
    if not opts:
        return []
    method_by_group = plot_df.drop_duplicates("PlotGroup").set_index("PlotGroup")["Optimizer"].to_dict()

    saved = []
    metrics = ["F1_test", "AS_test", "PS_test", "RS_test"]
    met_labels = ["F1-score", "Accuracy", "Precision", "Recall"]
    means = plot_df.groupby("PlotGroup")[metrics].mean()

    fig, ax = plt.subplots(figsize=(max(12, 1.25 * len(opts) + 7), 6))
    x = np.arange(len(metrics))
    n = len(opts)
    w = min(0.75 / max(1, n), 0.13)
    for i, opt in enumerate(opts):
        offset = (i - n / 2 + 0.5) * w
        vals = [means.loc[opt, m] for m in metrics]
        is_dsade = is_dsade_method(method_by_group.get(opt))
        bars = ax.bar(
            x + offset,
            vals,
            w,
            color=color_map.get(opt, "#888"),
            alpha=0.95 if is_dsade else 0.70,
            linewidth=2 if is_dsade else 0.5,
            edgecolor=color_map.get(opt, "#888"),
        )
        if is_dsade:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(met_labels, fontsize=12)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Average value", fontsize=11)
    ax.legend(handles=_plot_legend_patches(opts, color_map, label_map), loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_chart(fig, out_dir, "01_metricas_globales.png")
    saved.append("01_metricas_globales.png")

    smells = sorted(plot_df["Dataset"].unique())
    pivot_smell = plot_df.groupby(["Dataset", "PlotGroup"])["F1_test"].mean().unstack()
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(smells) * max(1, len(opts)) + 5), 6))
    x = np.arange(len(smells))
    w = 0.75 / max(1, len(opts))
    for i, opt in enumerate(opts):
        vals = [pivot_smell.loc[s, opt] if (s in pivot_smell.index and opt in pivot_smell.columns) else np.nan for s in smells]
        ax.bar(x + (i - len(opts) / 2 + 0.5) * w, vals, w, color=color_map.get(opt, "#888"), alpha=0.85, label=label_map.get(opt, opt))
    ax.set_xticks(x)
    ax.set_xticklabels(smells, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Average F1-test")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "02_f1_por_smell.png")
    saved.append("02_f1_por_smell.png")

    ests = sorted(plot_df["Estimator"].unique())
    pivot = plot_df.groupby(["Estimator", "PlotGroup"])["F1_test"].mean().unstack()
    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(opts) + 5), 6))
    x = np.arange(len(ests))
    n = len(opts)
    w = min(0.75 / max(1, n), 0.13)
    for i, opt in enumerate(opts):
        offset = (i - n / 2 + 0.5) * w
        vals = [pivot.loc[e, opt] if (e in pivot.index and opt in pivot.columns) else np.nan for e in ests]
        ax.bar(x + offset, vals, w, color=color_map.get(opt, "#888"), alpha=0.95 if is_dsade_method(method_by_group.get(opt)) else 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels([e.upper() for e in ests], fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Average F1-test", fontsize=11)
    ax.legend(handles=_plot_legend_patches(opts, color_map, label_map), loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "03_f1_por_estimador.png")
    saved.append("03_f1_por_estimador.png")

    pivot = plot_df.groupby(["Dataset", "PlotGroup"])["F1_test"].mean().unstack()
    mat = pivot.reindex(index=smells, columns=opts).values
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(opts) + 4), max(5, 0.35 * len(smells) + 2)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1-test", shrink=0.8)
    ax.set_xticks(range(len(opts)))
    ax.set_xticklabels([label_map.get(o, o) for o in opts], fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(smells)))
    ax.set_yticklabels(smells, fontsize=10)
    for i in range(len(smells)):
        for j, opt in enumerate(opts):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if v > 0.80 else "#333"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold" if is_dsade_method(method_by_group.get(opt)) else "normal")
    fig.tight_layout()
    _save_chart(fig, out_dir, "04_heatmap.png")
    saved.append("04_heatmap.png")

    data_box = [plot_df[plot_df["PlotGroup"] == o]["F1_test"].values for o in opts]
    fig, ax = plt.subplots(figsize=(max(11, 0.65 * len(opts) + 5), 6))
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5)
    for patch, opt in zip(bp["boxes"], opts):
        patch.set_facecolor(color_map.get(opt, "#888"))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(opts) + 1))
    ax.set_xticklabels([label_map.get(o, o) for o in opts], fontsize=9, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("F1-test", fontsize=11)
    fig.tight_layout()
    _save_chart(fig, out_dir, "05_boxplot_f1.png")
    saved.append("05_boxplot_f1.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    for opt in opts:
        sub = plot_df[plot_df["PlotGroup"] == opt]
        is_dsade = is_dsade_method(method_by_group.get(opt))
        ax.scatter(
            sub["N_Features_Selected"],
            sub["F1_test"],
            color=color_map.get(opt, "#888"),
            s=120 if is_dsade else 60,
            alpha=0.90 if is_dsade else 0.65,
            marker="*" if is_dsade else "o",
            label=label_map.get(opt, opt),
        )
    ax.set_xlabel("Number of selected features", fontsize=11)
    ax.set_ylabel("F1-test", fontsize=11)
    ax.set_ylim(0.0, 1.06)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "06_scatter_features_f1.png")
    saved.append("06_scatter_features_f1.png")

    means = plot_df.groupby("PlotGroup")[["F1_test", "AS_test", "PS_test", "RS_test", "N_Features_Selected"]].mean()
    max_feat = max(float(means["N_Features_Selected"].max()), 1.0)
    categories = ["F1-test", "Accuracy", "Precision", "Recall", "Feat.\\nEfficiency"]
    n_cat = len(categories)
    angles = [n / float(n_cat) * 2 * np.pi for n in range(n_cat)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for opt in opts:
        row = means.loc[opt]
        vals = [row["F1_test"], row["AS_test"], row["PS_test"], row["RS_test"], 1 - row["N_Features_Selected"] / max_feat]
        vals += vals[:1]
        is_dsade = is_dsade_method(method_by_group.get(opt))
        ax.plot(angles, vals, color=color_map.get(opt, "#888"), linewidth=2.5 if is_dsade else 1.2, linestyle="-" if is_dsade else "--", label=label_map.get(opt, opt))
        ax.fill(angles, vals, color=color_map.get(opt, "#888"), alpha=0.15 if is_dsade else 0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "07_radar.png")
    saved.append("07_radar.png")

    feat_med = plot_df.groupby("PlotGroup")["N_Features_Selected"].mean()
    rt_med = plot_df.groupby("PlotGroup")["Runtime"].mean()
    feat_vals = [feat_med[o] for o in opts]
    rt_vals = [rt_med[o] for o in opts]
    colors = [color_map.get(o, "#888") for o in opts]
    x = np.arange(len(opts))
    w = 0.38
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.bar(x - w / 2, feat_vals, w, color=colors, alpha=0.85, zorder=3)
    ax2.bar(x + w / 2, rt_vals, w, color=colors, alpha=0.45, hatch="///", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map.get(o, o) for o in opts], fontsize=9, rotation=35, ha="right")
    ax1.set_ylabel("Average selected features", fontsize=11)
    ax2.set_ylabel("Average runtime (seconds)", fontsize=11, color="#555")
    legend_elems = [
        mpatches.Patch(facecolor="#666", alpha=0.85, label="Selected features"),
        mpatches.Patch(facecolor="#666", alpha=0.45, hatch="///", label="Runtime (sec)"),
    ]
    ax1.legend(handles=legend_elems, loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "08_features_runtime.png")
    saved.append("08_features_runtime.png")

    grid_chart = generate_classifier_metric_grid_chart(df, out_dir, opt_order)
    if grid_chart:
        saved.append(grid_chart)
    return saved

def build_run_level_dataframe(results_struct: Dict[str, Dict], args: argparse.Namespace, estimator_filter: str = "knn") -> pd.DataFrame:
    rows = []
    for dataset_name, alg_data in results_struct.items():
        for label, row in alg_data.items():
            parsed = parse_result_label(label, args)
            estimator = parsed["estimator"] or row.get("Estimator", "")
            if str(estimator).lower() != estimator_filter.lower():
                continue
            runs_by_metric = {
                "AS_test": np.asarray(row.get("AccRuns", []), dtype=float) / 100.0,
                "F1_test": np.asarray(row.get("F1Runs", []), dtype=float),
                "PS_test": np.asarray(row.get("PSRuns", []), dtype=float),
                "RS_test": np.asarray(row.get("RSRuns", []), dtype=float),
                "N_Features_Selected": np.asarray(row.get("FeatRuns", []), dtype=float),
                "Runtime": np.asarray(row.get("TimeRuns", []), dtype=float),
            }
            n_runs = max((values.size for values in runs_by_metric.values()), default=0)
            for run_idx in range(n_runs):
                out = {
                    "Dataset": dataset_name,
                    "Estimator": estimator_filter.lower(),
                    "Optimizer": parsed["method"],
                    "TransferFunction": parsed["transfer_function"],
                    "Configuracion": label,
                    "Run": run_idx + 1,
                }
                for metric, values in runs_by_metric.items():
                    out[metric] = float(values[run_idx]) if run_idx < values.size else np.nan
                rows.append(out)
    return pd.DataFrame(rows)


def build_curve_dataframe(results_struct: Dict[str, Dict], args: argparse.Namespace, estimator_filter: str = "svm") -> pd.DataFrame:
    rows = []
    for dataset_name, alg_data in results_struct.items():
        for label, row in alg_data.items():
            parsed = parse_result_label(label, args)
            estimator = parsed["estimator"] or row.get("Estimator", "")
            if str(estimator).lower() != estimator_filter.lower():
                continue
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Estimator": estimator_filter.lower(),
                    "Optimizer": parsed["method"],
                    "TransferFunction": parsed["transfer_function"],
                    "Configuracion": label,
                    "Curve": np.asarray(row.get("Curve", []), dtype=float),
                }
            )
    return pd.DataFrame(rows)


def _grid_shape(n_items: int) -> tuple[int, int]:
    n_cols = min(4, max(1, int(np.ceil(np.sqrt(max(1, n_items))))))
    n_rows = int(np.ceil(max(1, n_items) / n_cols))
    return n_rows, n_cols


def generate_seven_global_charts(
    df: pd.DataFrame,
    results_struct: Dict[str, Dict],
    out_dir: str,
    opt_order: List[str],
    args: argparse.Namespace,
    estimator_filter: str = "svm", # Change here for knn
):
    if df.empty:
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    chart1 = generate_classifier_metric_grid_chart(df, out_dir, opt_order)
    if chart1:
        new_chart1 = "01_resultados_clasificador_todos_datasets.png"
        os.replace(os.path.join(out_dir, chart1), os.path.join(out_dir, new_chart1))
        saved.append(new_chart1)

    knn_df = df[df["Estimator"].astype(str).str.lower() == estimator_filter.lower()].copy()
    if knn_df.empty:
        return saved
    plot_df, opts, color_map, label_map = prepare_plot_groups(knn_df, opt_order)
    if not opts:
        return saved
    method_by_group = plot_df.drop_duplicates("PlotGroup").set_index("PlotGroup")["Optimizer"].to_dict()
    datasets = sorted(plot_df["Dataset"].dropna().unique())
    n_rows, n_cols = _grid_shape(len(datasets))

    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "Feat.\nEfficiency"]
    angles = [n / 5.0 * 2 * np.pi for n in range(5)]
    angles += angles[:1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.8 * n_rows),
        subplot_kw=dict(polar=True),
        squeeze=False,
    )
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = plot_df[plot_df["Dataset"] == dataset]
        means = sub.groupby("PlotGroup")[["AS_test", "PS_test", "RS_test", "F1_test", "N_Features_Selected"]].mean()
        max_feat = max(float(means["N_Features_Selected"].max()), 1.0)
        for opt in opts:
            if opt not in means.index:
                continue
            row = means.loc[opt]
            vals = [row["AS_test"], row["PS_test"], row["RS_test"], row["F1_test"], 1 - row["N_Features_Selected"] / max_feat]
            vals += vals[:1]
            is_dsade = is_dsade_method(method_by_group.get(opt))
            is_macro = method_by_group.get(opt) == "MaCRO-DE"

            ax.plot(
                angles,
                vals,
                color=color_map.get(opt, "#888"),
                linewidth=4.0 if is_macro else (2.4 if is_dsade else 1.1),
                linestyle="-" if is_macro else ("-" if is_dsade else "--"),
                zorder=10 if is_macro else 2
            )

            ax.fill(
                angles,
                vals,
                color=color_map.get(opt, "#888"),
                alpha=0.20 if is_macro else (0.12 if is_dsade else 0.04)
            )
            # ax.plot(angles, vals, color=color_map.get(opt, "#888"), linewidth=2.4 if is_dsade else 1.1, linestyle="-" if is_dsade else "--")
            # ax.fill(angles, vals, color=color_map.get(opt, "#888"), alpha=0.12 if is_dsade else 0.04)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(dataset, fontsize=11, fontweight="bold", pad=14)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.legend(handles=_plot_legend_patches(opts, color_map, label_map), loc="lower center", ncol=min(len(opts), 6), fontsize=9)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    _save_chart(fig, out_dir, "02_radar_por_dataset_knn.png")
    saved.append("02_radar_por_dataset_knn.png")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax1 = axes[idx // n_cols, idx % n_cols]
        ax2 = ax1.twinx()
        sub = plot_df[plot_df["Dataset"] == dataset].groupby("PlotGroup")[["N_Features_Selected", "Runtime"]].mean()
        x = np.arange(len(opts))
        feat_vals = [sub.loc[o, "N_Features_Selected"] if o in sub.index else np.nan for o in opts]
        rt_vals = [sub.loc[o, "Runtime"] if o in sub.index else np.nan for o in opts]
        colors = [color_map.get(o, "#888") for o in opts]
        ax1.bar(x - 0.18, feat_vals, 0.36, color=colors, alpha=0.85)
        ax2.bar(x + 0.18, rt_vals, 0.36, color=colors, alpha=0.40, hatch="///")
        ax1.set_xticks(x)
        ax1.set_xticklabels([label_map.get(o, o) for o in opts], rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("Features", fontsize=9)
        ax2.set_ylabel("Runtime (s)", fontsize=9)
        ax1.set_title(dataset, fontsize=11, fontweight="bold")
        ax1.grid(axis="y", alpha=0.25)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
    _save_chart(fig, out_dir, "03_features_runtime_por_dataset_knn.png")
    saved.append("03_features_runtime_por_dataset_knn.png")

    run_df = build_run_level_dataframe(results_struct, args, estimator_filter)
    run_source = run_df if not run_df.empty else plot_df
    run_plot_df, run_opts, run_color_map, run_label_map = prepare_plot_groups(run_source, opt_order)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = run_plot_df[run_plot_df["Dataset"] == dataset]
        data_box = [sub[sub["PlotGroup"] == opt]["AS_test"].dropna().values for opt in run_opts]
        bp = ax.boxplot(data_box, patch_artist=True, widths=0.55, showmeans=True)
        for patch, opt in zip(bp["boxes"], run_opts):
            patch.set_facecolor(run_color_map.get(opt, "#888"))
            patch.set_alpha(0.60)
        ax.set_xticks(range(1, len(run_opts) + 1))
        ax.set_xticklabels([run_label_map.get(o, o) for o in run_opts], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Accuracy (test)", fontsize=9)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
    _save_chart(fig, out_dir, "04_boxplot_accuracy_por_dataset_knn.png")
    saved.append("04_boxplot_accuracy_por_dataset_knn.png")

    curve_df = build_curve_dataframe(results_struct, args, estimator_filter)
    if curve_df.empty:
        curve_plot_df = pd.DataFrame()
        curve_opts, curve_color_map, curve_label_map = opts, color_map, label_map
    else:
        curve_plot_df, curve_opts, curve_color_map, curve_label_map = prepare_plot_groups(curve_df, opt_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.4 * n_rows), squeeze=False)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = curve_plot_df[curve_plot_df["Dataset"] == dataset] if not curve_plot_df.empty else pd.DataFrame()
        plotted = False
        for opt in curve_opts:
            rows_opt = sub[sub["PlotGroup"] == opt] if not sub.empty else pd.DataFrame()
            if rows_opt.empty:
                continue
            curve = np.asarray(rows_opt.iloc[0]["Curve"], dtype=float)
            if curve.size == 0:
                continue
            is_dsade = is_dsade_method(rows_opt.iloc[0]["Optimizer"])
            is_macro = str(rows_opt.iloc[0]["Optimizer"]).upper() == "MACRO-DE"
            ax.plot(curve, color=curve_color_map.get(opt, "#888"), linewidth=2.4 if is_macro else (2.4 if is_dsade else 1.4), linestyle="-")
            #ax.plot(curve, color=curve_color_map.get(opt, "#888"), linewidth=2.4 if is_dsade else 1.4, linestyle="-" if is_dsade else "--")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "Sin curvas", transform=ax.transAxes, ha="center", va="center", color="#777")
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Fitness", fontsize=9)
        ax.grid(alpha=0.25)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.legend(handles=_plot_legend_patches(curve_opts, curve_color_map, curve_label_map), loc="lower center", ncol=min(len(curve_opts), 6), fontsize=9)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    _save_chart(fig, out_dir, "05_convergence_por_dataset_knn.png")
    saved.append("05_convergence_por_dataset_knn.png")

    pivot = plot_df.groupby(["PlotGroup", "Dataset"])["F1_test"].mean().unstack()
    mat = pivot.reindex(index=opts, columns=datasets).values
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(datasets) + 4), max(5, 0.45 * len(opts) + 2)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1-Score (test)", shrink=0.8)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right")
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels([label_map.get(o, o) for o in opts])
    # for tick, opt in zip(ax.get_yticklabels(), opts):
    #     if str(method_by_group.get(opt)).upper() == "MACRO-DE":
    #         tick.set_color("red")
    #         tick.set_fontweight("bold")
    macro_idx = next(
        (
            i for i, opt in enumerate(opts)
            if str(method_by_group.get(opt)).upper() == "MACRO-DE"
        ),
        None,
    )

    if macro_idx is not None:
        rect = plt.Rectangle((-0.5, macro_idx - 0.5), len(datasets),1, fill=False, edgecolor="black", linewidth=2.5, zorder=100)
        ax.add_patch(rect)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Metaheuristics")
    for i in range(len(opts)):
        for j in range(len(datasets)):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="white" if value > 0.80 else "#222", fontsize=8)
    fig.tight_layout()
    _save_chart(fig, out_dir, "06_heatmap_f1_knn.png")
    saved.append("06_heatmap_f1_knn.png")

    data_violin = [run_plot_df[run_plot_df["PlotGroup"] == opt]["RS_test"].dropna().values for opt in run_opts]
    fig, ax = plt.subplots(figsize=(max(12, 0.85 * len(run_opts) + 5), 6.5))
    parts = ax.violinplot(data_violin, showmeans=False, showmedians=False, widths=0.78)
    for body, opt in zip(parts["bodies"], run_opts):
        body.set_facecolor(run_color_map.get(opt, "#888"))
        body.set_edgecolor(run_color_map.get(opt, "#888"))
        body.set_alpha(0.22)
    for i, (opt, values) in enumerate(zip(run_opts, data_violin), start=1):
        if values.size == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(np.full(values.size, i) + jitter, values, color=run_color_map.get(opt, "#888"), edgecolor="white", linewidth=0.5, s=35, zorder=3)
        mean_val = float(np.nanmean(values))
        median_val = float(np.nanmedian(values))
        ax.scatter(i, mean_val, marker="D", color="black", edgecolor="white", linewidth=1.2, s=140, zorder=4)
        ax.hlines(median_val, i - 0.25, i + 0.25, colors="black", linestyles="--", linewidth=1.2)
        ax.text(i, mean_val + 0.018, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(range(1, len(run_opts) + 1))
    ax.set_xticklabels([run_label_map.get(o, o) for o in run_opts], rotation=35, ha="right")
    ax.set_ylabel("Recall (test)")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#555", label="Mean"),
            plt.Line2D([0], [0], color="#555", linestyle="--", label="Median"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", label="Value per dataset/run"),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    fig.tight_layout()
    _save_chart(fig, out_dir, "07_violin_recall_knn.png")
    saved.append("07_violin_recall_knn.png")

    generate_global_accuracy_boxplot(
        run_plot_df,
        out_dir,
        opt_order
    )
    saved.append("08_global_accuracy_distribution.png")

    generate_global_features_runtime(
        plot_df,
        out_dir,
        opt_order
    )
    saved.append("09_global_features_runtime_tradeoff.png")
    return saved

def generate_global_accuracy_boxplot(df, out_dir, opt_order):

    plot_df, opts, color_map, label_map = prepare_plot_groups(df, opt_order)

    fig, ax = plt.subplots(
        figsize=(max(12, 0.8 * len(opts) + 5), 6)
    )

    data_box = [
        plot_df[plot_df["PlotGroup"] == opt]["AS_test"].values
        for opt in opts
    ]

    bp = ax.boxplot(
        data_box,
        patch_artist=True,
        widths=0.55,
        showmeans=True
    )

    for patch, opt in zip(bp["boxes"], opts):

        patch.set_facecolor(color_map.get(opt, "#888"))
        patch.set_alpha(0.70)

        if label_map.get(opt) == "MaCRO-DE":
            patch.set_edgecolor("black")
            patch.set_linewidth(3.0)

    for i, opt in enumerate(opts):

        vals = plot_df[
            plot_df["PlotGroup"] == opt
        ]["AS_test"]

        if len(vals) > 0:
            ax.text(
                i + 1,
                np.mean(vals) + 0.01,
                f"{np.mean(vals):.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold"
            )

    ax.set_ylabel("Accuracy (test)")
    ax.set_xlabel("Metaheuristics")
    ax.set_ylim(0.50, 1.05)

    ax.set_xticks(range(1, len(opts)+1))
    ax.set_xticklabels(
        [label_map[o] for o in opts],
        rotation=45,
        ha="right"
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    _save_chart(
        fig,
        out_dir,
        "08_global_accuracy_distribution.png"
    )

def generate_global_features_runtime(df, out_dir, opt_order):

    plot_df, opts, color_map, label_map = prepare_plot_groups(df, opt_order)

    feat_med = (
        plot_df.groupby("PlotGroup")
        ["N_Features_Selected"]
        .mean()
    )

    rt_med = (
        plot_df.groupby("PlotGroup")
        ["Runtime"]
        .mean()
    )

    feat_vals = [feat_med[o] for o in opts]
    rt_vals   = [rt_med[o] for o in opts]

    x = np.arange(len(opts))
    w = 0.38

    fig, ax1 = plt.subplots(figsize=(12,6))

    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - w/2,
        feat_vals,
        w,
        alpha=0.85
    )

    bars2 = ax2.bar(
        x + w/2,
        rt_vals,
        w,
        alpha=0.45,
        hatch="///"
    )

    for bar, opt in zip(bars1, opts):

        bar.set_color(color_map.get(opt, "#888"))

        if label_map.get(opt) == "MaCRO-DE":
            bar.set_edgecolor("black")
            bar.set_linewidth(3)

    for bar, opt in zip(bars2, opts):

        bar.set_color(color_map.get(opt, "#888"))

        if label_map.get(opt) == "MaCRO-DE":
            bar.set_edgecolor("black")
            bar.set_linewidth(3)

    for i, v in enumerate(feat_vals):

        ax1.text(
            i - w/2,
            v + 0.2,
            f"{v:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    for i, v in enumerate(rt_vals):

        ax2.text(
            i + w/2,
            v + 0.5,
            f"{v:.1f}s",
            ha="center",
            fontsize=9
        )

    ax1.set_ylabel("Average selected features")
    ax2.set_ylabel("Average runtime (sec)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [label_map[o] for o in opts],
        rotation=45,
        ha="right"
    )

    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    _save_chart(
        fig,
        out_dir,
        "09_global_features_runtime_tradeoff.png"
    )

def regenerate_figures_from_cache(paths: Paths, args: argparse.Namespace, dataset_names: List[str], cache_sig: str):
    results_struct = load_results_from_cache(paths, args, dataset_names, cache_sig)
    statistical_excel = os.path.join(paths.res_dir, f"Statistical_Results_{paths.exp_tag}.xlsx")
    export_statistical_excel(results_struct, dataset_names, list(args.optimizers), args, statistical_excel)
    summary_df = generate_summary_dataframe(results_struct, args)
    summary_csv = os.path.join(paths.res_dir, f"RESUMEN_GRAFICAS_{paths.exp_tag}.csv")
    summary_df.to_csv(summary_csv, index=False)
    generated_charts = generate_seven_global_charts(
        summary_df,
        results_struct,
        paths.fig_dir,
        list(args.optimizers),
        args,
    )
    return summary_csv, generated_charts, statistical_excel

def main():
    args = parse_args()
    logging.disable(logging.INFO)
    logging.getLogger("mealpy").setLevel(logging.WARNING)
    if args.list_optimizers:
        print_available_optimizers()
        return

    validate_selection_options(args)
    args.optimizers = resolve_optimizers(args)
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.n_workers < 1:
        raise ValueError("--n-workers must be >= 1")

    paths = make_paths(args)
    cache_sig = build_cache_signature(args)
    show_tf = len(args.transfer_functions) > 1
    show_cls = len(args.estimators) > 1

    dataset_specs = resolve_dataset_specs(args)
    dataset_names = [spec.name for spec in dataset_specs]

    print(f"Experiment: {paths.exp_tag}")
    print_dataset_summary(args, dataset_specs)
    print(f"Cache signature: {cache_sig}")

    if args.figures_only:
        summary_csv, generated_charts, statistical_excel = regenerate_figures_from_cache(paths, args, dataset_names, cache_sig)
        print("Completed figures-only.")
        print(f"Cache dir: {paths.cache_dir}")
        print(f"Figures dir: {paths.fig_dir}")
        print(f"Charts summary CSV: {summary_csv}")
        print(f"Statistical results: {statistical_excel}")
        if generated_charts:
            print("Charts:")
            for name in generated_charts:
                print(f"  - {os.path.join(paths.fig_dir, name)}")
        return

    results_struct = {}
    for spec in dataset_specs:
        dataset_name, X, y = load_dataset(spec)
        results_struct[dataset_name] = {}
        data = Data(X, y)
        try:
            data.split_train_test(test_size=args.test_size, random_state=args.random_state, stratify=y)
        except ValueError:
            data.split_train_test(test_size=args.test_size, random_state=args.random_state)

        for estimator in args.estimators:
            cache_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_results.pkl",
            )
            progress_file = os.path.join(
                paths.cache_dir,
                f"{paths.exp_tag}_{dataset_name}_{estimator.lower()}_{cache_sig}_progress.pkl",
            )
            cache_payload = load_cache_safe(cache_file, "final cache") if args.reuse_cache else None
            progress_payload = load_cache_safe(progress_file, "partial checkpoint")
            if cache_payload is not None and (
                progress_payload is None
                or payload_completed_runs(cache_payload) >= payload_completed_runs(progress_payload)
            ):
                print(f"[cache] {dataset_name} / {estimator}")
                cls_payload = cache_payload
            else:
                cls_payload = progress_payload or {}
                if progress_payload is not None:
                    print(f"[resume] Resuming {dataset_name} / {estimator} from partial checkpoint")
            for method in args.optimizers:
                for tf in args.transfer_functions:
                    label = build_alg_label(method, tf, estimator, show_tf, show_cls)
                    legacy_label = build_legacy_alg_label(method, tf, estimator, show_tf, show_cls)
                    if label not in cls_payload and legacy_label in cls_payload:
                        cls_payload[label] = cls_payload.pop(legacy_label)
                    prev = cls_payload.get(label, {})
                    acc_runs = list(np.asarray(prev.get("AccRuns", []), dtype=float))
                    ps_runs = list(np.asarray(prev.get("PSRuns", []), dtype=float))
                    rs_runs = list(np.asarray(prev.get("RSRuns", []), dtype=float))
                    f1_runs = list(np.asarray(prev.get("F1Runs", []), dtype=float))
                    fit_runs = list(np.asarray(prev.get("FitRuns", []), dtype=float))
                    feat_runs = list(np.asarray(prev.get("FeatRuns", []), dtype=float))
                    time_runs = list(np.asarray(prev.get("TimeRuns", []), dtype=float))
                    curves = list(prev.get("CurvesAll", []))

                    done = len(acc_runs)
                    if done >= args.runs:
                        print(f"Running {dataset_name} | {label} | runs={args.runs} (already complete)")
                        continue
                    print(f"Running {dataset_name} | {label} | runs={args.runs} (resume from {done})")

                    pending_runs = list(range(done, args.runs))
                    def checkpoint_run(run, out):
                        acc_runs.append(out["as_test"])
                        ps_runs.append(out["ps_test"])
                        rs_runs.append(out["rs_test"])
                        f1_runs.append(out["f1_test"])
                        fit_runs.append(out["fit_final"])
                        feat_runs.append(out["n_features"])
                        time_runs.append(out["runtime"])
                        curves.append(out["curve"])
                        print(
                            f"  Run {run + 1:02d} | Acc={acc_runs[-1]:.2f}% | F1={f1_runs[-1]:.4f} | "
                            f"Fit={fit_runs[-1]:.4f} | Feat={feat_runs[-1]} | Time={time_runs[-1]:.2f}s"
                        )

                        cls_payload[label] = build_label_payload(
                            estimator,
                            acc_runs,
                            ps_runs,
                            rs_runs,
                            f1_runs,
                            fit_runs,
                            feat_runs,
                            time_runs,
                            curves,
                            args.epochs,
                        )
                        save_cache(progress_file, cls_payload)
                        save_cache(cache_file, cls_payload)

                    if args.parallel == "yes" and len(pending_runs) > 1:
                        print(f"  Parallel: yes | workers={min(args.n_workers, len(pending_runs))}")
                        execute_pending_runs(
                            data,
                            estimator,
                            method,
                            tf,
                            args,
                            pending_runs,
                            on_run_complete=checkpoint_run,
                        )
                    else:
                        for run in pending_runs:
                            checkpoint_run(
                                run,
                                run_single(data, estimator, method, tf, args, args.seed_base + run),
                            )

                    cls_payload[label] = build_label_payload(
                        estimator,
                        acc_runs,
                        ps_runs,
                        rs_runs,
                        f1_runs,
                        fit_runs,
                        feat_runs,
                        time_runs,
                        curves,
                        args.epochs,
                    )
                    save_cache(progress_file, cls_payload)
                    save_cache(cache_file, cls_payload)
            save_cache(cache_file, cls_payload)

            results_struct[dataset_name].update(cls_payload)

    excel_path = os.path.join(paths.res_dir, f"Global_Results_{paths.exp_tag}.xlsx")
    exported = export_global_excel(results_struct, dataset_names, excel_path)
    statistical_excel = os.path.join(paths.res_dir, f"Statistical_Results_{paths.exp_tag}.xlsx")
    export_statistical_excel(results_struct, dataset_names, list(args.optimizers), args, statistical_excel)
    summary_df = generate_summary_dataframe(results_struct, args)
    summary_csv = os.path.join(paths.res_dir, f"RESUMEN_GRAFICAS_{paths.exp_tag}.csv")
    summary_df.to_csv(summary_csv, index=False)
    chart_dir = paths.fig_dir
    generated_charts = generate_seven_global_charts(summary_df, results_struct, chart_dir, list(args.optimizers), args)

    print("Completed.")
    print(f"Cache dir: {paths.cache_dir}")
    print(f"Figures dir: {paths.fig_dir}")
    print(f"Charts summary CSV: {summary_csv}")
    print(f"Charts dir: {chart_dir}")
    print(f"Statistical results: {statistical_excel}")
    if generated_charts:
        print("Notebook-style charts:")
        for name in generated_charts:
            print(f"  - {os.path.join(chart_dir, name)}")
    print("Global results:")
    for p in exported:
        print(f"  - {p}")

if __name__ == "__main__":
    main()

