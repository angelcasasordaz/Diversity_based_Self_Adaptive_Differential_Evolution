"""Cache-only EXP627 publication figures. Run with python -B full_rep1_report.py."""
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import t

from full_replica_report import DATASETS, framework, load_completed_full, report_guard, sha256
from full_replica_report import validate_replica_destination, REPLICA_DIRECTORIES
import shutil
from PIL import Image


PUBLICATION_COLORS = {
    "DSA-DE": "#6E0D1B", "DE": "#009E73", "JADE": "#E69F00",
    "SHADE": "#CC79A7", "PSO": "#56B4E9", "WOA": "#D55E00",
    "HHO": "#6A3D9A", "GOA": "#8DAA00", "SA": "#F0C808",
    "BRO": "#A65628", "RUN": "#4D4D4D", "FOX": "#999999",
}
ORDER = tuple(PUBLICATION_COLORS)
CLASSIFIERS = ("svm", "knn", "rf")
DARK_RED = "#6E0D1B"
STEMS = ("grafica_resumen_general", "radar_6smells_grid_svm", "heatmap_f1score",
         "ranking_precision", "boxplot_accuracy_general", "violin_recall",
         "convergence_curve", "features_runtime_per_optimizer")
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
         "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 9,
         "axes.titlesize": 11, "figure.facecolor": "white", "axes.facecolor": "white",
         "savefig.facecolor": "white", "savefig.dpi": 600, "figure.dpi": 100,
         "pdf.fonttype": 42, "ps.fonttype": 42}


def validate_palette():
    assert len(ORDER) == 12
    assert len({to_rgba(color) for color in PUBLICATION_COLORS.values()}) == 12
    assert ORDER == ("DSA-DE", "DE", "JADE", "SHADE", "PSO", "WOA", "HHO", "GOA", "SA", "BRO", "RUN", "FOX")


def color(optimizer):
    return PUBLICATION_COLORS[optimizer]


def style_axes(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color="#D9D9D9", linestyle="--", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#777777")


def optimizer_ticks(ax, horizontal=False):
    if horizontal:
        ax.set_yticks(np.arange(12), ORDER)
        labels = ax.get_yticklabels()
    else:
        ax.set_xticks(np.arange(12), ORDER, rotation=45, ha="right")
        labels = ax.get_xticklabels()
    labels[0].set_color(color("DSA-DE"))
    labels[0].set_fontweight("bold")


def styled_bar(bar, opt):
    bar.set_facecolor(color(opt))
    bar.set_edgecolor(DARK_RED if opt == "DSA-DE" else color(opt))
    bar.set_linewidth(2.2 if opt == "DSA-DE" else 0.6)


def metric_values(df, metric, classifier):
    sub = df[df.Estimator == classifier]
    return np.stack([sub[sub.Optimizer == opt].set_index("Dataset").loc[list(DATASETS), metric].to_numpy(float)
                     for opt in ORDER])


def precision_figure(values):
    fig, ax = plt.subplots(figsize=(10, 6.2), layout="constrained")
    means = values.mean(axis=1)
    ci = t.ppf(0.975, 5) * values.std(axis=1, ddof=1) / np.sqrt(6)
    for i, opt in enumerate(ORDER):
        emphasis = i == 0
        ax.errorbar(means[i], i, xerr=ci[i], fmt="D" if emphasis else "o", color=color(opt),
                    markersize=8 if emphasis else 5.5, elinewidth=2.8 if emphasis else 1.5,
                    capsize=5 if emphasis else 3.5, markeredgecolor=DARK_RED if emphasis else "white",
                    markeredgewidth=1.2 if emphasis else 0.6)
        ax.text(1.02, i, f"{means[i]:.4f}", transform=ax.get_yaxis_transform(), va="center",
                color=color(opt) if emphasis else "#333333", fontweight="bold" if emphasis else "normal")
    optimizer_ticks(ax, horizontal=True)
    ax.invert_yaxis()
    ax.set_xlabel("Average Precision (test) ± 95% CI")
    style_axes(ax, horizontal=True)
    return fig


def observations(ax, values, means=True):
    for i, opt in enumerate(ORDER):
        ax.scatter(i + np.linspace(-0.13, 0.13, 6), values[i], color=color(opt),
                   edgecolor=DARK_RED if i == 0 else "white", linewidth=1 if i == 0 else 0.6,
                   s=48 if i == 0 else 30, zorder=4)
        mean = values[i].mean()
        if means:
            ax.scatter(i, mean, marker="D", s=100 if i == 0 else 70,
                       facecolor=color(opt), edgecolor=DARK_RED if i == 0 else "black", zorder=6)
        ax.annotate(f"{mean:.4f}", (i, values[i].max()), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=color(opt) if i == 0 else "#333333",
                    fontweight="bold" if i == 0 else "normal")
    optimizer_ticks(ax)
    ax.set_ylim(max(0, values.min() - 0.07), 1.07)
    style_axes(ax)


def violin_figure(values):
    fig, ax = plt.subplots(figsize=(13, 5.8), layout="constrained")
    parts = ax.violinplot(values.T, positions=np.arange(12), widths=0.78, showextrema=False)
    for i, (body, opt) in enumerate(zip(parts["bodies"], ORDER)):
        body.set_facecolor(color(opt))
        body.set_edgecolor(DARK_RED if i == 0 else color(opt))
        body.set_alpha(0.28)
        body.set_linewidth(2.8 if i == 0 else 1)
        ax.hlines(np.median(values[i]), i-0.3, i+0.3, colors="black", linestyles="--", linewidth=1.3)
    observations(ax, values)
    ax.set_ylabel("Recall (test)")
    ax.legend(handles=[Line2D([], [], marker="D", color="none", markerfacecolor="#666666", markeredgecolor="black", label="Mean"),
                       Line2D([], [], color="black", linestyle="--", label="Median"),
                       Line2D([], [], marker="o", color="none", markerfacecolor="#666666", label="Value by code smell")],
              loc="lower left", ncol=3, frameon=False)
    return fig


def boxplot_figure(values, ylabel):
    fig, ax = plt.subplots(figsize=(13, 5.8), layout="constrained")
    boxes = ax.boxplot(values.T, positions=np.arange(12), widths=0.58, patch_artist=True,
                       showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    for i, (box, opt) in enumerate(zip(boxes["boxes"], ORDER)):
        styled_bar(box, opt)
        box.set_facecolor(to_rgba(color(opt), 0.32))
        for segment in boxes["whiskers"][i*2:i*2+2] + boxes["caps"][i*2:i*2+2]:
            segment.set_color(DARK_RED if i == 0 else color(opt))
            segment.set_linewidth(2.2 if i == 0 else 1.2)
    observations(ax, values, means=False)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#666666", label="Value by code smell"),
                       Line2D([], [], color="black", label="Median")], loc="lower left", frameon=False, ncol=2)
    return fig


def general_figure(df):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), layout="constrained")
    for row, classifier in enumerate(CLASSIFIERS):
        for col, (metric, title) in enumerate((("AS_test", "Accuracy"), ("PS_test", "Precision"),
                                              ("RS_test", "Recall"), ("F1_test", "F1-Score"))):
            ax = axes[row, col]
            values = metric_values(df, metric, classifier).mean(axis=1)
            bars = ax.bar(np.arange(12), values, width=0.72)
            for bar, opt, value in zip(bars, ORDER, values):
                styled_bar(bar, opt)
                ax.annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", rotation=90,
                            fontsize=8, color=color(opt) if opt == "DSA-DE" else "#333333")
            ax.axhline(values.mean(), color="#666666", linestyle="--", linewidth=0.8)
            optimizer_ticks(ax)
            ax.tick_params(axis="x", labelsize=8)
            ax.set_ylim(0, 1.12)
            if row == 0:
                ax.set_title(title, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(classifier.upper(), fontweight="bold")
            style_axes(ax)
    fig.legend(handles=[Line2D([], [], color="#666666", linestyle="--", label="Mean across metaheuristics")],
               loc="outside lower center", frameon=False)
    return fig


def tradeoff_figure(df, classifier):
    features = metric_values(df, "N_Features_Selected", classifier).mean(axis=1)
    runtime = metric_values(df, "Runtime", classifier).mean(axis=1)
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    twin = ax.twinx()
    for axis, values, shift, hatch, alpha in ((ax, features, -0.2, None, 1), (twin, runtime, 0.2, "///", 0.45)):
        bars = axis.bar(np.arange(12)+shift, values, width=0.36, hatch=hatch, alpha=alpha)
        for bar, opt, value in zip(bars, ORDER, values):
            styled_bar(bar, opt)
            axis.annotate(f"{value:.2f}" if axis is ax else f"{value:.1f}",
                          (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5), textcoords="offset points",
                          ha="center", fontsize=9, rotation=90,
                          color=color(opt) if opt == "DSA-DE" else "#333333")
        axis.set_ylim(0, max(values)*1.25)
    optimizer_ticks(ax)
    ax.set_ylabel("Average selected features")
    optimizer_ticks(twin)
    twin.set_ylabel("Average runtime (s)")
    style_axes(ax)
    twin.spines["top"].set_visible(False)
    ax.legend(handles=[Patch(facecolor="#777777", label="Selected features"),
                       Patch(facecolor="#777777", hatch="///", alpha=0.45, label="Runtime")],
              loc="upper left", ncol=2, frameon=False)
    return fig


def line_style(opt):
    i = ORDER.index(opt)
    return dict(color=color(opt), linewidth=2.8 if i == 0 else 1.2,
                linestyle=("-", "--", ":", "-.")[i % 4],
                marker=("D", "o", "s", "^", "v", "P", "X", "*", "+", "x", "<", ">")[i],
                markersize=5 if i == 0 else 3, zorder=10 if i == 0 else 2)


def line_legend(fig):
    legend = fig.legend(handles=[Line2D([], [], label=opt, **line_style(opt)) for opt in ORDER],
                        loc="lower center", ncol=6, frameon=False)
    legend.get_texts()[0].set_color(color("DSA-DE"))
    legend.get_texts()[0].set_fontweight("bold")


def radar_figure(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"polar": True})
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles = np.r_[angles, angles[0]]
    for ax, dataset in zip(axes.flat, DATASETS):
        sub = df[(df.Estimator == "svm") & (df.Dataset == dataset)].set_index("Optimizer").loc[list(ORDER)]
        # Preserve the original radar's feature-efficiency normalization.
        max_feat = max(float(sub.N_Features_Selected.max()), 1.0)
        for opt in ORDER:
            row = sub.loc[opt]
            values = [row.AS_test, row.PS_test, row.RS_test, row.F1_test,
                      1 - row.N_Features_Selected / max_feat]
            ax.plot(angles, values + values[:1], **line_style(opt))
            ax.fill(angles, values + values[:1], color=color(opt), alpha=0.08 if opt == "DSA-DE" else 0.015)
        ax.set_xticks(angles[:-1], ["Accuracy", "Precision", "Recall", "F1-Score", "Feature\nefficiency"], fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(color="#D9D9D9", linewidth=0.65)
        ax.spines["polar"].set_color("#BBBBBB")
        ax.set_title(dataset, pad=22)
    line_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=3, w_pad=3)
    return fig


def heatmap_figure(df):
    values = metric_values(df, "F1_test", "svm")
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="F1-Score (test)", shrink=0.8)
    ax.set_xticks(np.arange(6), DATASETS, rotation=30, ha="right")
    optimizer_ticks(ax, horizontal=True)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 6, 1, fill=False,
                              edgecolor=color("DSA-DE"), linewidth=3, clip_on=False))
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center",
                color="white" if values[i, j] > 0.8 else "#222222",
                fontweight="bold" if i == 0 else "normal")
    ax.set_xlabel("Code smell")
    return fig


def convergence_figure(results, args):
    # Same helper and stored Curve selection as generate_dataset_convergence:
    # one cached mean over 30 runs per optimizer and dataset, no new aggregation.
    curves = framework().build_curve_dataframe(results, args, "svm")
    curves["Optimizer"] = curves.Optimizer.replace({"DSADE": "DSA-DE"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, dataset in zip(axes.flat, DATASETS):
        for opt in ORDER:
            rows = curves[(curves.Dataset == dataset) & (curves.Optimizer == opt)]
            assert len(rows) == 1
            curve = np.asarray(rows.iloc[0].Curve, dtype=float)
            ax.plot(np.arange(curve.size), curve, markevery=max(1, curve.size // 12), **line_style(opt))
        ax.set_title(dataset)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        style_axes(ax)
    line_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def validate_output_destination(root, destination):
    """Accept only the named reporting tree; never follow links or accept caches."""
    root = Path(root).resolve()
    destination = Path(destination).absolute()
    from full_rep1_statistics import STEMS as statistical_stems, EXPECTED_RES
    layouts = {
        root / "Figures/EXP627/full_rep1": {
            "": {f"{stem}.png" for stem in STEMS},
            "statistics": {f"{stem}.png" for stem in statistical_stems},
        },
        root / "Results/EXP627/full_rep1": {
            "": {"validation.json", "Paper_Tables_EXP627.xlsx"},
            "statistics": EXPECTED_RES,
            "latex_tables": {f"table_{table}_{variant}.tex"
                             for table in ("overall", "datasets_1", "datasets_2")
                             for variant in ("mean_std", "full_stats")} | {"latex_table_validation.txt"},
        },
    }
    if destination not in layouts:
        raise ValueError(f"Unexpected reporting destination: {destination}")
    if destination.resolve() != destination or destination.is_symlink():
        raise ValueError(f"Redirected reporting destination: {destination}")
    for path in (destination, *destination.parents):
        if path.is_symlink():
            raise ValueError(f"Symlink in reporting path: {path}")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Not a reporting directory: {destination}")
        layout = layouts[destination]
        for path in destination.rglob("*"):
            relative = path.relative_to(destination)
            if (path.is_symlink() or path.is_mount() or path.resolve() != path
                    or path.suffix.lower() in {".pkl", ".pickle", ".ckpt"}
                    or "cache" in path.name.lower() or "checkpoint" in path.name.lower()):
                raise ValueError(f"Unexpected reporting content: {path}")
            if path.is_dir():
                if relative.as_posix() not in layout:
                    raise ValueError(f"Unexpected reporting directory: {path}")
            else:
                parent = "" if relative.parent == Path(".") else relative.parent.as_posix()
                if not path.is_file() or relative.name not in layout.get(parent, set()):
                    raise ValueError(f"Unexpected reporting content: {path}")
    return destination


def protected_hashes(root):
    excluded = [root / relative for relative in REPLICA_DIRECTORIES]
    excluded += [root / f"{kind}/EXP627/full_rep1" for kind in ("Figures", "Results")]
    files = set(root.rglob("*.pkl"))
    for kind in ("Figures", "Results"):
        files.update(p for p in (root / kind / "EXP627").rglob("*")
                     if p.is_file() and not any(base in p.parents for base in excluded))
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


def cleanup_duplicates(root):
    # Preflight all four exact paths and all contents before any deletion.
    paths = [validate_replica_destination(root, root / relative) for relative in REPLICA_DIRECTORIES]
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Symlink-resistant cleanup is required")
    deleted = []
    for path in paths:
        path = validate_replica_destination(root, path)
        if path.exists():
            print(f"Deleting validated reporting directory: {path}", flush=True)
            shutil.rmtree(path)
            deleted.append(str(path.relative_to(root)))
    return deleted


def run(args, *, resume=False):
    validate_palette()
    m = framework()
    root = Path(args.output_root).resolve()
    destination = validate_output_destination(root, root / "Figures/EXP627/full_rep1")
    reports = validate_output_destination(root, root / "Results/EXP627/full_rep1")
    expected = {f"{stem}.png" for stem in STEMS}
    before = protected_hashes(root)
    latex = reports / "latex_tables"
    latex_before = {p.name: sha256(p) for p in latex.iterdir()} if latex.exists() else None
    allowed = (destination, reports)
    with report_guard(allowed) as guard, plt.rc_context(STYLE):
        args, results, indexed, sources = load_completed_full(args)
        df = m.generate_summary_dataframe(results, args)
        reference = pd.read_csv(root / "Results/EXP627/full/RESUMEN_GRAFICAS_EXP627.csv")
        keys = ["Dataset", "Estimator", "Optimizer"]
        metrics = ["AS_test", "PS_test", "RS_test", "F1_test", "N_Features_Selected", "Runtime"]
        pd.testing.assert_frame_equal(df.set_index(keys)[metrics].sort_index(), reference.set_index(keys)[metrics].sort_index(),
                                      check_exact=False, rtol=1e-12, atol=1e-12)
        assert len(indexed) == 216
        df["Optimizer"] = df.Optimizer.replace({"DSADE": "DSA-DE"})
        destination.mkdir(exist_ok=True)
        reports.mkdir(exist_ok=True)
        generators = (
            lambda: general_figure(df), lambda: radar_figure(df), lambda: heatmap_figure(df),
            lambda: precision_figure(metric_values(df, "PS_test", "svm")),
            lambda: boxplot_figure(metric_values(df, "AS_test", "svm"), "Accuracy (test)"),
            lambda: violin_figure(metric_values(df, "RS_test", "svm")),
            lambda: convergence_figure(results, args), lambda: tradeoff_figure(df, "svm"),
        )
        checks = {}
        for stem, generate in zip(STEMS, generators):
            fig = generate()
            try:
                for ax in fig.axes:
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        if label.get_text() == "DSA-DE":
                            assert label.get_color() == color("DSA-DE") and label.get_fontweight() == "bold"
                target = destination / f"{stem}.png"
                m._save_figure(fig, target, save_pdf=True, bbox_inches="tight")
                with Image.open(target) as png:
                    assert png.format == "PNG" and all(abs(dpi-600) < 0.1 for dpi in png.info["dpi"])
                checks[target.name] = sha256(target)
            finally:
                plt.close(fig)
            print(f"Created {target.name} (600 dpi PNG)", flush=True)
        assert {p.name for p in destination.iterdir() if p.is_file()} == expected
        from full_rep1_statistics import run as run_statistics
        run_statistics(args)
        validate_output_destination(root, destination)
        validate_output_destination(root, reports)
        assert ({p.name: sha256(p) for p in latex.iterdir()} if latex.exists() else None) == latex_before
        assert guard["optimization_calls"] == 0
        assert protected_hashes(root) == before
        manifest = {"source": "EXP627 FULL", "datasets": list(DATASETS), "classifiers": list(CLASSIFIERS),
                    "runs_per_combination": 30, "combinations": len(indexed), "optimizer_order": ORDER,
                    "colors": PUBLICATION_COLORS, "single_classifier_figures": "svm",
                    "distribution_observations": "Six code-smell means, each based on 30 cached runs",
                    "precision_ci": "Student-t 95% CI over six code-smell Precision means; df=5",
                    "convergence": "Original build_curve_dataframe; stored Curve per code smell/optimizer, mean of 30 runs",
                    "optimization_calls": 0, "standard_pdfs_generated": 0, "dpi": 600,
                    "deleted_duplicate_folders": [], "original_files_unchanged": True,
                    "statistical_pngs": 4, "statistical_pdfs": 0, "latex_tables_unchanged": True,
                    "original_sha256": before, "source_caches": sources, "outputs_sha256": checks}
        (reports / "validation.json").write_text(json.dumps(manifest, indent=2))
    assert protected_hashes(root) == before
    print("Validated: 8 standard PNGs; 4 statistical PNGs; 0 PDFs; 0 optimization calls.")
    print(f"SHA-256 verified {len(before)} original files unchanged.")


if __name__ == "__main__":
    run(framework().parse_args())
