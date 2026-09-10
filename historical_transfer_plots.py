"""Blue-only plotting definitions based on main_best.py at 9fc6150.

Commit: 9fc6150e1d0b0c1d120f68983990b44ab446f434 (2026-09-07).
The navy override originated in bcc0cb9; d3f1889 replaced it for transfer plots.
The blue-only plots use dark SSTF and lighter VSTF shades and show only KNN in the grid.
The adapter redirects output names and isolates historical Matplotlib defaults.
The command-line adapter reads existing caches; it never performs optimization
or writes scientific exports.
"""
from pathlib import Path
from typing import Dict, List, Optional
import os
import pickle
import tempfile

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# This resolver is unchanged between the historical commit and current HEAD.
from optimizer_factory import optimizer_acronym

METRICS_FILENAME = "TransferFunctions_ClassificationMetrics_Blue.png"
TRADEOFF_FILENAME = "TransferFunctions_FeaturesRuntimeTradeoff_Blue.png"

ESTIMATORS = [
    "knn",
    "svm",
    # "rf",
]

SUPPORTED_ESTIMATORS = ["knn", "svm", "rf", "adaboost", "xgb", "tree", "ann"]

CHART_CMAP = "Dark2"

MACRO_DE_COLOR = "#19365f"

def muted_color_palette(n: int) -> np.ndarray:
    cmap = plt.get_cmap(CHART_CMAP, max(n, 1))
    return cmap(np.arange(max(n, 1)))[:, :3]

def optimizer_display_label(name: str) -> str:
    return optimizer_acronym(str(name))

def is_exact_dsade_method(name: str) -> bool:
    return str(name).upper() in {"DSA-DE", "DSADE"}

def is_dsade_plot_group(opt: str, method_by_group: Optional[Dict[str, str]] = None) -> bool:
    method = method_by_group.get(opt, opt) if method_by_group else opt
    return is_exact_dsade_method(method)

def apply_dsade_patch_highlight(patch, opt: str, method_by_group: Optional[Dict[str, str]] = None, linewidth: float = 2.4) -> None:
    if is_dsade_plot_group(opt, method_by_group):
        patch.set_edgecolor("black")
        patch.set_linewidth(linewidth)

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

    blue = plt.get_cmap("Blues")(0.75)
    transfer_colors = dict(zip(
        [f"{family}_{index:02d}" for family in ("sstf", "vstf") for index in range(1, 5)],
        plt.get_cmap("Blues")(np.linspace(0.90, 0.45, 8)),
    ))
    color_map = {}
    label_map = {}
    for group in opts:
        meta = group_meta[group]
        method = meta["Optimizer"]
        tf = meta["TransferFunction"]
        color_map[group] = transfer_colors.get(tf, blue)
        base_label = optimizer_display_label(method)
        label_map[group] = f"{base_label} {tf.upper()}" if tf and method in variant_methods else base_label

    return plot_df, opts, color_map, label_map

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

    estimators = ["knn"]

    grouped = plot_df.groupby(["Estimator", "PlotGroup"])[metric_cols].mean()
    n_rows = len(estimators)
    n_cols = len(metric_cols)
    fig_w = max(16.0, 4.2 * n_cols)
    fig_h = max(4.5, 2.75 * n_rows + 2.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, facecolor="#f7f9fc")
    x = np.arange(len(opts))
    colors = [color_map[opt] for opt in opts]
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
            edges = ["black" if is_dsade_plot_group(opt, method_by_group) else "none" for opt in opts]
            widths = [2.2 if is_dsade_plot_group(opt, method_by_group) else 0.0 for opt in opts]
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

def generate_global_features_runtime(df, out_dir, opt_order):

    plot_df, opts, color_map, label_map = prepare_plot_groups(df, opt_order)
    method_by_group = plot_df.drop_duplicates("PlotGroup").set_index("PlotGroup")["Optimizer"].to_dict() if not plot_df.empty else {}

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
        apply_dsade_patch_highlight(bar, opt, method_by_group, linewidth=2.8)

    for bar, opt in zip(bars2, opts):

        bar.set_color(color_map.get(opt, "#888"))

        if label_map.get(opt) == "MaCRO-DE":
            bar.set_edgecolor("black")
            bar.set_linewidth(3)
        apply_dsade_patch_highlight(bar, opt, method_by_group, linewidth=2.8)

    # Point offsets stay readable across the two different axis scales.
    value_labels = []
    for ax, values, shift, fmt, weight in (
        (ax1, feat_vals, -w / 2, "{:.2f}", "bold"),
        (ax2, rt_vals, w / 2, "{:.1f}s", "normal"),
    ):
        for i, value in enumerate(values):
            if np.isfinite(value):
                value_labels.append(ax.annotate(
                    fmt.format(value), (i + shift, value),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                    fontweight=weight,
                ))

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

    # Resolve collisions in display coordinates, including labels on twin axes.
    # Repeat after adding headroom because changing limits moves the bar tops.
    for _ in range(12):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        gap = renderer.points_to_pixels(4)
        placed = []
        for label in value_labels:
            label.set_position((0, 6))
            bbox = label.get_window_extent(renderer)
            for previous in sorted(placed, key=lambda box: box.y0):
                if (bbox.x0 < previous.x1 + gap and bbox.x1 > previous.x0 - gap
                        and bbox.y0 < previous.y1 + gap and bbox.y1 > previous.y0 - gap):
                    offset = (previous.y1 + gap - bbox.y0) * 72 / fig.dpi
                    label.set_position((0, label.get_position()[1] + offset))
                    bbox = label.get_window_extent(renderer)
            placed.append(bbox)
        overflow = max((box.y1 + gap - ax1.bbox.y1 for box in placed), default=0)
        if overflow <= 0:
            break
        for ax in (ax1, ax2):
            lower, upper = ax.get_ylim()
            ax.set_ylim(lower, lower + (upper - lower) * (1.08 + overflow / ax.bbox.height))

    _save_chart(
        fig,
        out_dir,
        "09_global_features_runtime_tradeoff.png"
    )


def _render_extra(generator, original_filename, df, out_dir, opt_order, filename):
    """Run the blue-only plotting code, publishing only the extra PNG."""
    if df.empty:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 9fc6150 used default figure DPI (100) and white backgrounds. Current
    # main_best uses 600 DPI, which changes collision placement before saving.
    with plt.rc_context(rc=plt.rcParamsDefault), tempfile.TemporaryDirectory() as staging:
        generator(df, staging, opt_order)
        # Copy bytes because the temporary directory can be on another filesystem.
        (out_dir / filename).write_bytes((Path(staging) / original_filename).read_bytes())
    return filename


def render_classifier_metric_grid(df, out_dir, opt_order, filename=METRICS_FILENAME):
    return _render_extra(
        generate_classifier_metric_grid_chart,
        "09_resultados_clasificador_metrica_todos_datasets.png",
        df, out_dir, opt_order, filename,
    )


def render_features_runtime(df, out_dir, opt_order, filename=TRADEOFF_FILENAME):
    return _render_extra(
        generate_global_features_runtime,
        "09_global_features_runtime_tradeoff.png",
        df, out_dir, opt_order, filename,
    )


def load_exp626_plot_summary(root):
    """Read saved means independently for each classifier, without cache writes.

    The existing CSV contains only KNN. Both classifiers' complete caches are
    present under signature 0fbda34b93. Accuracy uses the same percent-to-fraction
    conversion as generate_summary_dataframe in the historical study.
    """
    cache_dir = Path(root) / "Results/EXP626/transfer_functions/cache"
    records = []
    for dataset in ("BreastCancer", "Ionosphere", "Tic-tac-toe", "Wine", "Zoo"):
        for estimator in ESTIMATORS:
            path = cache_dir / f"EXP626_{dataset}_{estimator}_0fbda34b93_results.pkl"
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            expected_labels = {
                f"MACRO-DE_{family}_{index:02d}"
                for family in ("SSTF", "VSTF") for index in range(1, 5)
            }
            if set(payload) != expected_labels:
                raise ValueError(f"Unexpected transfer-function coverage in {path}")
            for label, row in payload.items():
                if row["Estimator"] != estimator or row["CompletedRuns"] != 30:
                    raise ValueError(f"Incomplete or mismatched saved results in {path}: {label}")
                records.append({
                    "Dataset": dataset,
                    "Estimator": estimator,
                    "Optimizer": "MaCRO-DE",
                    "TransferFunction": label.removeprefix("MACRO-DE_").lower(),
                    "AS_test": float(row["AccMean"]) / 100.0,
                    "PS_test": float(row["PSMean"]),
                    "RS_test": float(row["RSMean"]),
                    "F1_test": float(row["F1Mean"]),
                    "N_Features_Selected": float(row["FeatMean"]),
                    "Runtime": float(row["TimeMean"]),
                })
    return pd.DataFrame(records)


if __name__ == "__main__":
    # The study's figures-only entry point also rewrites scientific exports.
    # Use the dedicated read-only loader and only the two historical renderers.
    root = Path(__file__).resolve().parent
    output_path = root / "Figures/EXP626/transfer_functions"
    summary = load_exp626_plot_summary(root)
    svm = summary[summary["Estimator"].str.lower() == "svm"]
    if summary.empty or svm.empty:
        raise ValueError("Historical EXP626 rendering requires saved summary data and SVM results.")
    if set(summary["Optimizer"].str.upper()) != {"MACRO-DE"}:
        raise ValueError("Expected the saved MaCRO-DE transfer-function experiment.")
    for filename in (
        render_classifier_metric_grid(summary, output_path, ["MaCRO-DE"]),
        render_features_runtime(svm, output_path, ["MaCRO-DE"]),
    ):
        print(output_path / filename)
