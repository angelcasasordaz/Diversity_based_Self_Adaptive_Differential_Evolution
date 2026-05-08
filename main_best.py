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
from typing import Dict, List

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

from dsade_optimizer import DSADE

# DEFAULT_OPTIMIZERS = [
#     "OriginalPSO",
#     "OriginalGWO",
#     "OriginalWOA",
#     "OriginalDE",
#     "JADE",
#     "SADE",
#     "OriginalBRO",
#     "OriginalDMOA",
#     "OriginalFOX",
#     "OriginalRIME",
#     "DSADE",
# ]

#Test
DEFAULT_OPTIMIZERS = ["DSADE"]
DEFAULT_ESTIMATORS = ["knn"]
# DEFAULT_TRANSFER_FUNCTIONS = ["vstf_01"]
DEFAULT_TRANSFER_FUNCTIONS = [
    "vstf_01",
    "vstf_02",
    "vstf_03",
    "vstf_04",
    "sstf_01",
    "sstf_02",
    "sstf_03",
    "sstf_04",
]

TEST_datasets_clasific_14 = [
    "Blood",
    "BreastCancer",
    "BreastEW",
    "Glass",
    "HeartEW",
    "Ionosphere",
    "Iris",
    "Lymphography",
    "Sonar",
    "Tic-tac-toe",
    "WaveformEW",
    "Wine",
    "Zoo",
]
CHART_PALETTE = {
    "DSADE": "#3266ad",
    "OriginalGWO": "#e06c00",
    "OriginalWOA": "#2a9d5c",
    "OriginalCA": "#c44569",
    "OriginalPSO": "#9b59b6",
    "OriginalDE": "#6a4c93",
    "JADE": "#2d6a4f",
    "SADE": "#f4a261",
    "OriginalSHADE": "#264653",
    "OriginalFOX": "#1b9aaa",
    "OriginalRIME": "#e76f51",
    "OriginalBRO": "#577590",
    "OriginalDMOA": "#90be6d",
    "OriginalMGO": "#f9844a",
    "OriginalHHO": "#4d4d4d",
    "OriginalGOA": "#8a5a44",
}
CHART_LABELS = {
    "DSADE": "DSADE",
    "OriginalGWO": "GWO",
    "OriginalWOA": "WOA",
    "OriginalCA": "CA",
    "OriginalPSO": "PSO",
    "OriginalDE": "DE",
    "JADE": "JADE",
    "SADE": "SADE",
    "OriginalSHADE": "SHADE",
    "OriginalFOX": "FOX",
    "OriginalRIME": "RIME",
    "OriginalBRO": "BRO",
    "OriginalDMOA": "DMOA",
    "OriginalMGO": "MGO",
    "OriginalHHO": "HHO",
    "OriginalGOA": "GOA",
}
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Framework de comparacion FS (multi-dataset, multi-run, cache)")
    parser.add_argument("--exp-id", type=int, default=626, help="ID numerico del experimento")
    parser.add_argument("--dataset-source", default="mafese", choices=["mafese"], help="Origen de datasets")
    parser.add_argument("--dataset-suite", default="test14", choices=["test14"], help="Suite de datasets")
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS), help="Lista de optimizadores")
    parser.add_argument("--estimators", nargs="+", default=DEFAULT_ESTIMATORS, help="Lista de clasificadores")
    parser.add_argument("--transfer-functions", nargs="+", default=DEFAULT_TRANSFER_FUNCTIONS, help="Lista de transfer functions")
    parser.add_argument("--runs", type=int, default=20, help="Ejecuciones independientes por combinacion")
    parser.add_argument("--epochs", type=int, default=50, help="Iteraciones del optimizador")
    parser.add_argument("--pop-size", type=int, default=50, help="Tamano de poblacion")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio")
    parser.add_argument("--random-state", type=int, default=2, help="Semilla de split")
    parser.add_argument("--seed-base", type=int, default=1234, help="Semilla base por run")
    parser.add_argument("--output-root", default=".", help="Raiz para Figures/Results")
    parser.add_argument("--reuse-cache", action="store_true", help="Usar cache si existe")
    parser.add_argument("--parallel", default="yes", choices=["yes", "no"], help="Ejecutar runs en paralelo: yes/no")
    parser.add_argument("--n-workers", type=int, default=4, help="Numero de procesos paralelos si --parallel yes")
    parser.add_argument("--dsade-beta-min", type=float, default=0.2)
    parser.add_argument("--dsade-beta-max", type=float, default=0.8)
    parser.add_argument("--dsade-pcr", type=float, default=0.2)
    parser.add_argument("--dsade-mahal-q", type=float, default=0.68)
    return parser.parse_args()

def resolve_optimizers(args: argparse.Namespace) -> List[str]:
    return list(dict.fromkeys(args.optimizers))

def validate_selection_options(args: argparse.Namespace) -> None:
    invalid_estimators = [e for e in args.estimators if e not in SUPPORTED_ESTIMATORS]
    if invalid_estimators:
        raise ValueError(
            f"Clasificadores no soportados: {invalid_estimators}. "
            f"Validos: {', '.join(SUPPORTED_ESTIMATORS)}"
        )
    invalid_tf = [tf for tf in args.transfer_functions if tf not in SUPPORTED_TRANSFER_FUNCTIONS]
    if invalid_tf:
        raise ValueError(
            f"Transfer functions no soportadas: {invalid_tf}. "
            f"Validas: {', '.join(SUPPORTED_TRANSFER_FUNCTIONS)}"
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
        return list(TEST_datasets_clasific_14)
    raise ValueError(f"Suite de datasets no soportada: {args.dataset_suite}")


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


def build_optimizer(name: str, args: argparse.Namespace):
    if name.upper() == "DSADE":
        return DSADE(
            epoch=args.epochs,
            pop_size=args.pop_size,
            beta_min=args.dsade_beta_min,
            beta_max=args.dsade_beta_max,
            pcr=args.dsade_pcr,
            mahalanobis_q=args.dsade_mahal_q,
        )
    if name.upper() == "ORIGINALDMOA":
        return SafeOriginalDMOA(epoch=args.epochs, pop_size=args.pop_size)
    return name

def build_cache_signature(args: argparse.Namespace) -> str:
    payload = {
        "optimizers": list(args.optimizers),
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
    parts = [method.upper()]
    if show_tf:
        parts.append(str(transfer_function).upper())
    if show_cls:
        parts.append(classifier.upper())
    return "_".join(parts)

def muted_color_palette(n: int) -> np.ndarray:
    cmap = plt.get_cmap("turbo", max(n, 1))
    colors = cmap(np.arange(max(n, 1)))[:, :3]
    colors = 0.8 * colors + 0.2
    return np.clip(colors, 0.0, 1.0)


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
):
    if args.parallel != "yes" or len(pending_runs) <= 1:
        return [
            (run, run_single(data, estimator, method, tf, args, args.seed_base + run))
            for run in pending_runs
        ]

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
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_parallel_task, task) for task in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
    return sorted(completed, key=lambda item: item[0])


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


def parse_result_label(label: str, args: argparse.Namespace) -> dict:
    label_upper = str(label).upper()
    ordered_opts = sorted([str(o) for o in args.optimizers], key=len, reverse=True)
    method = next(
        (
            opt
            for opt in ordered_opts
            if label_upper == opt.upper() or label_upper.startswith(f"{opt.upper()}_")
        ),
        str(label),
    )
    rest = label_upper[len(method):].lstrip("_") if method != str(label) else ""

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


def prepare_plot_groups(df: pd.DataFrame, opt_order: List[str]) -> tuple[pd.DataFrame, List[str], Dict[str, str], Dict[str, str]]:
    if df.empty:
        return df.copy(), [], {}, {}

    plot_df = df.copy()
    if "FuncionTransferencia" not in plot_df.columns:
        plot_df["FuncionTransferencia"] = ""
    plot_df["FuncionTransferencia"] = plot_df["FuncionTransferencia"].fillna("").astype(str).str.lower()

    tf_counts = plot_df[plot_df["FuncionTransferencia"] != ""].groupby("Optimizador")["FuncionTransferencia"].nunique()
    variant_methods = set(tf_counts[tf_counts > 1].index)

    def make_group(row):
        opt = str(row["Optimizador"])
        tf = str(row["FuncionTransferencia"]).lower()
        return f"{opt}_{tf.upper()}" if opt in variant_methods and tf else opt

    plot_df["GrupoGrafica"] = plot_df.apply(make_group, axis=1)
    group_meta = (
        plot_df[["GrupoGrafica", "Optimizador", "FuncionTransferencia"]]
        .drop_duplicates()
        .set_index("GrupoGrafica")
        .to_dict("index")
    )

    opts = []
    for opt in opt_order:
        opt_groups = sorted(
            [g for g, meta in group_meta.items() if meta["Optimizador"] == opt],
            key=lambda g: (str(group_meta[g]["FuncionTransferencia"]), g),
        )
        opts.extend(opt_groups)
    opts.extend(sorted(g for g in group_meta if g not in set(opts)))

    colors = muted_color_palette(len(opts))
    color_map = {}
    label_map = {}
    for i, group in enumerate(opts):
        meta = group_meta[group]
        method = meta["Optimizador"]
        tf = meta["FuncionTransferencia"]
        if method not in variant_methods and method in CHART_PALETTE:
            color_map[group] = CHART_PALETTE[method]
        else:
            color_map[group] = colors[i]
        base_label = CHART_LABELS.get(method, method)
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
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
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
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(4, max(1, len(labels))), frameon=False)
    plt.tight_layout()
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
                    "Archivo": dataset_name,
                    "Estimador": estimator,
                    "Optimizador": method,
                    "FuncionTransferencia": parsed["transfer_function"],
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


def _legend_patches(opts: List[str]) -> List[mpatches.Patch]:
    return [mpatches.Patch(color=CHART_PALETTE.get(o, "#888"), label=CHART_LABELS.get(o, o)) for o in opts]


def _plot_legend_patches(opts: List[str], color_map: Dict[str, str], label_map: Dict[str, str]) -> List[mpatches.Patch]:
    return [mpatches.Patch(color=color_map.get(o, "#888"), label=label_map.get(o, o)) for o in opts]


def _save_chart(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_classifier_metric_grid_chart(df: pd.DataFrame, out_dir: str, opt_order: List[str]):
    if df.empty:
        return None

    plot_df = df.copy()
    plot_df["Estimador"] = plot_df["Estimador"].astype(str).str.lower()
    plot_df, opts, color_map, label_map = prepare_plot_groups(plot_df, opt_order)
    if not opts:
        return None
    method_by_group = plot_df.drop_duplicates("GrupoGrafica").set_index("GrupoGrafica")["Optimizador"].to_dict()

    metric_cols = ["AS_test", "PS_test", "RS_test", "F1_test"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_header_styles = [
        ("#d8e8f3", "#b8d3e6"),
        ("#d2efee", "#abd9d7"),
        ("#f7efd8", "#ead9ad"),
        ("#f9d5d9", "#edaeb8"),
    ]

    present_estimators = [str(e).lower() for e in plot_df["Estimador"].dropna().unique()]
    required_estimators = [e for e in DEFAULT_ESTIMATORS if e in SUPPORTED_ESTIMATORS]
    estimators = [e for e in SUPPORTED_ESTIMATORS if e in set(required_estimators + present_estimators)]
    estimators += sorted(e for e in present_estimators if e not in set(estimators))
    if not estimators:
        return None

    grouped = plot_df.groupby(["Estimador", "GrupoGrafica"])[metric_cols].mean()
    n_rows = len(estimators)
    n_cols = len(metric_cols)
    fig_w = max(16.0, 4.2 * n_cols)
    fig_h = max(4.5, 2.75 * n_rows + 2.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, facecolor="#f7f9fc")
    fig.suptitle(
        "Resultados por Clasificador y Metrica - Todos los datasets",
        fontsize=18,
        fontweight="bold",
        color="#19365f",
        y=0.995,
    )

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
            edges = ["black" if method_by_group.get(opt) == "DSADE" else "none" for opt in opts]
            widths = [1.8 if method_by_group.get(opt) == "DSADE" else 0.0 for opt in opts]
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
                    "Sin datos",
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
    if any(method_by_group.get(opt) == "DSADE" for opt in opts):
        legend.append(mpatches.Patch(facecolor="#333333", edgecolor="black", label="DSADE: borde negro"))
    fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 6), fontsize=9, framealpha=0.95)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.97])
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
    method_by_group = plot_df.drop_duplicates("GrupoGrafica").set_index("GrupoGrafica")["Optimizador"].to_dict()

    saved = []
    metricas = ["F1_test", "AS_test", "PS_test", "RS_test"]
    met_labels = ["F1-score", "Accuracy", "Precision", "Recall"]
    medias = plot_df.groupby("GrupoGrafica")[metricas].mean()

    fig, ax = plt.subplots(figsize=(max(12, 1.25 * len(opts) + 7), 6))
    x = np.arange(len(metricas))
    n = len(opts)
    w = min(0.75 / max(1, n), 0.13)
    for i, opt in enumerate(opts):
        offset = (i - n / 2 + 0.5) * w
        vals = [medias.loc[opt, m] for m in metricas]
        is_dsade = method_by_group.get(opt) == "DSADE"
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
    ax.set_title("Average detection metrics per optimizer", fontsize=13, fontweight="bold", pad=12)
    ax.legend(handles=_plot_legend_patches(opts, color_map, label_map), loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save_chart(fig, out_dir, "01_metricas_globales.png")
    saved.append("01_metricas_globales.png")

    smells = sorted(plot_df["Archivo"].unique())
    pivot_smell = plot_df.groupby(["Archivo", "GrupoGrafica"])["F1_test"].mean().unstack()
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
    ax.set_title("Average F1-score per code smell and optimizer", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "02_f1_por_smell.png")
    saved.append("02_f1_por_smell.png")

    ests = sorted(plot_df["Estimador"].unique())
    pivot = plot_df.groupby(["Estimador", "GrupoGrafica"])["F1_test"].mean().unstack()
    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(opts) + 5), 6))
    x = np.arange(len(ests))
    n = len(opts)
    w = min(0.75 / max(1, n), 0.13)
    for i, opt in enumerate(opts):
        offset = (i - n / 2 + 0.5) * w
        vals = [pivot.loc[e, opt] if (e in pivot.index and opt in pivot.columns) else np.nan for e in ests]
        ax.bar(x + offset, vals, w, color=color_map.get(opt, "#888"), alpha=0.95 if method_by_group.get(opt) == "DSADE" else 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels([e.upper() for e in ests], fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Average F1-test", fontsize=11)
    ax.set_title("Average F1-score per classifier and optimizer", fontsize=13, fontweight="bold", pad=12)
    ax.legend(handles=_plot_legend_patches(opts, color_map, label_map), loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "03_f1_por_estimador.png")
    saved.append("03_f1_por_estimador.png")

    pivot = plot_df.groupby(["Archivo", "GrupoGrafica"])["F1_test"].mean().unstack()
    mat = pivot.reindex(index=smells, columns=opts).values
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(opts) + 4), max(5, 0.35 * len(smells) + 2)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1-test", shrink=0.8)
    ax.set_xticks(range(len(opts)))
    ax.set_xticklabels([label_map.get(o, o) for o in opts], fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(smells)))
    ax.set_yticklabels(smells, fontsize=10)
    ax.set_title("Heatmap F1-test: code smell x optimizer", fontsize=13, fontweight="bold", pad=12)
    for i in range(len(smells)):
        for j, opt in enumerate(opts):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if v > 0.80 else "#333"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold" if method_by_group.get(opt) == "DSADE" else "normal")
    fig.tight_layout()
    _save_chart(fig, out_dir, "04_heatmap.png")
    saved.append("04_heatmap.png")

    data_box = [plot_df[plot_df["GrupoGrafica"] == o]["F1_test"].values for o in opts]
    fig, ax = plt.subplots(figsize=(max(11, 0.65 * len(opts) + 5), 6))
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5)
    for patch, opt in zip(bp["boxes"], opts):
        patch.set_facecolor(color_map.get(opt, "#888"))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(opts) + 1))
    ax.set_xticklabels([label_map.get(o, o) for o in opts], fontsize=9, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("F1-test", fontsize=11)
    ax.set_title("F1-test distribution per optimizer")
    fig.tight_layout()
    _save_chart(fig, out_dir, "05_boxplot_f1.png")
    saved.append("05_boxplot_f1.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    for opt in opts:
        sub = plot_df[plot_df["GrupoGrafica"] == opt]
        is_dsade = method_by_group.get(opt) == "DSADE"
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
    ax.set_title("Selected features vs F1-test per optimizer", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "06_scatter_features_f1.png")
    saved.append("06_scatter_features_f1.png")

    medias = plot_df.groupby("GrupoGrafica")[["F1_test", "AS_test", "PS_test", "RS_test", "N_Features_Selected"]].mean()
    max_feat = max(float(medias["N_Features_Selected"].max()), 1.0)
    categories = ["F1-test", "Accuracy", "Precision", "Recall", "Feat.\\nEfficiency"]
    n_cat = len(categories)
    angles = [n / float(n_cat) * 2 * np.pi for n in range(n_cat)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for opt in opts:
        row = medias.loc[opt]
        vals = [row["F1_test"], row["AS_test"], row["PS_test"], row["RS_test"], 1 - row["N_Features_Selected"] / max_feat]
        vals += vals[:1]
        is_dsade = method_by_group.get(opt) == "DSADE"
        ax.plot(angles, vals, color=color_map.get(opt, "#888"), linewidth=2.5 if is_dsade else 1.2, linestyle="-" if is_dsade else "--", label=label_map.get(opt, opt))
        ax.fill(angles, vals, color=color_map.get(opt, "#888"), alpha=0.15 if is_dsade else 0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Multidimensional performance profile", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_chart(fig, out_dir, "07_radar.png")
    saved.append("07_radar.png")

    feat_med = plot_df.groupby("GrupoGrafica")["N_Features_Selected"].mean()
    rt_med = plot_df.groupby("GrupoGrafica")["Runtime"].mean()
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
    ax1.set_title("Selected features and runtime per optimizer", fontsize=13, fontweight="bold", pad=12)
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

def main():
    args = parse_args()
    logging.disable(logging.INFO)
    logging.getLogger("mealpy").setLevel(logging.WARNING)
    validate_selection_options(args)
    args.optimizers = resolve_optimizers(args)
    if args.runs < 1:
        raise ValueError("--runs debe ser >= 1")
    if args.n_workers < 1:
        raise ValueError("--n-workers debe ser >= 1")

    paths = make_paths(args)
    cache_sig = build_cache_signature(args)
    show_tf = len(args.transfer_functions) > 1
    show_cls = len(args.estimators) > 1

    dataset_names = resolve_mafese_dataset_names(args)

    print(f"Experiment: {paths.exp_tag}")
    print(f"Dataset source: {args.dataset_source}")
    print(f"Dataset suite: {args.dataset_suite} ({len(dataset_names)} datasets)")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Cache signature: {cache_sig}")

    results_struct = {}
    for dataset_name in dataset_names:
        results_struct[dataset_name] = {}
        mafese_data = get_dataset(dataset_name)
        if mafese_data is None:
            raise ValueError(
                f"mafese no pudo cargar '{dataset_name}'. "
                "Verifica que exista en la suite 'test14' de mafese."
            )
        X = np.asarray(mafese_data.X, dtype=np.float64)
        y = np.asarray(mafese_data.y).astype(np.int32)
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
            if args.reuse_cache and os.path.exists(cache_file):
                print(f"[cache] {dataset_name} / {estimator}")
                cls_payload = load_cache(cache_file)
            else:
                cls_payload = {}
                if os.path.exists(progress_file):
                    cls_payload = load_cache(progress_file)
                    print(f"[resume] Reanudando {dataset_name} / {estimator} desde checkpoint parcial")
                for method in args.optimizers:
                    for tf in args.transfer_functions:
                        label = build_alg_label(method, tf, estimator, show_tf, show_cls)
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
                        if args.parallel == "yes" and len(pending_runs) > 1:
                            print(f"  Parallel: yes | workers={min(args.n_workers, len(pending_runs))}")
                            run_outputs = execute_pending_runs(data, estimator, method, tf, args, pending_runs)
                        else:
                            run_outputs = []
                            for run in pending_runs:
                                run_outputs.append((run, run_single(data, estimator, method, tf, args, args.seed_base + run)))

                        for run, out in run_outputs:
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
                if os.path.exists(progress_file):
                    os.remove(progress_file)

            results_struct[dataset_name].update(cls_payload)

            labels_this = list(cls_payload.keys())
            if not labels_this:
                continue

            acc_vec = np.array([cls_payload[l]["AccMean"] for l in labels_this], dtype=float)
            f1_vec = np.array([cls_payload[l]["F1Mean"] for l in labels_this], dtype=float)
            curves_by_label = {l: cls_payload[l]["Curve"] for l in labels_this}

            plot_bar(
                acc_vec,
                labels_this,
                "Mean Accuracy (%)",
                f"Mean Accuracy - {dataset_name} ({estimator.upper()}) [{paths.exp_tag}]",
                os.path.join(paths.fig_dir, f"{paths.exp_tag}_{dataset_name}_Accuracy_{estimator.lower()}.svg"),
            )
            plot_bar(
                f1_vec,
                labels_this,
                "Mean F1-score",
                f"Mean F1-score - {dataset_name} ({estimator.upper()}) [{paths.exp_tag}]",
                os.path.join(paths.fig_dir, f"{paths.exp_tag}_{dataset_name}_F1_{estimator.lower()}.svg"),
            )
            plot_lines(
                curves_by_label,
                f"Mean Convergence - {dataset_name} ({estimator.upper()}) [{paths.exp_tag}]",
                "Objective Function Value (minimized)",
                os.path.join(paths.fig_dir, f"{paths.exp_tag}_{dataset_name}_Convergence_{estimator.lower()}.svg"),
            )

    excel_path = os.path.join(paths.res_dir, f"Global_Results_{paths.exp_tag}.xlsx")
    exported = export_global_excel(results_struct, dataset_names, excel_path)
    summary_df = generate_summary_dataframe(results_struct, args)
    summary_csv = os.path.join(paths.res_dir, f"RESUMEN_GRAFICAS_{paths.exp_tag}.csv")
    summary_df.to_csv(summary_csv, index=False)
    chart_dir = paths.fig_dir
    generated_charts = generate_notebook_style_charts(summary_df, chart_dir, list(args.optimizers))

    print("Completed.")
    print(f"Cache dir: {paths.cache_dir}")
    print(f"Figures dir: {paths.fig_dir}")
    print(f"Charts summary CSV: {summary_csv}")
    print(f"Charts dir: {chart_dir}")
    if generated_charts:
        print("Notebook-style charts:")
        for name in generated_charts:
            print(f"  - {os.path.join(chart_dir, name)}")
    print("Global results:")
    for p in exported:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
