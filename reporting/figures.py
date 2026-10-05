"""Publication figures derived from actual cached datasets, classifiers and algorithms."""
from pathlib import Path
import argparse
from dataclasses import replace
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import t

from reporting.core import framework, report_stage
import full_plot_style as base_style
from figure_layout import full_figure_directories, full_figure_path, STATISTICS, main_figure_names, filename_component, METRIC_TOKENS
from figure_text import localize_figure, visible_text


STYLE = base_style.STYLE
palette = base_style.palette
style_axes = base_style.style_axes

BASE_FIGURES = (
    'grafica_resumen_general.png', 'radar_6smells_grid_svm.png',
    'ranking_precision.png', 'boxplot_accuracy_general.png', 'heatmap_f1score.png',
    'violin_recall.png', 'convergence_curve.png', 'features_runtime_per_optimizer.png',
)
# Historical names above are retained for reference; new FULL exports are numbered.
INDIVIDUAL_DIRECTORY = 'individual'


def report_from_results(args, results):
    """Adapt completed in-memory results for plotting, without cache/science writes."""
    from reporting.core import CompletedReport
    from reporting.paper_tables import METRICS
    m = framework()
    plot_args = argparse.Namespace(**vars(args))
    indexed, algorithms, classifiers = {}, [], []
    for dataset, rows in results.items():
        for label, row in rows.items():
            parsed = m.parse_result_label(label, plot_args)
            classifier = str(row['Estimator']).lower()
            algorithm = m.build_alg_label(parsed['method'], parsed['transfer_function'] or args.transfer_functions[0],
                                           classifier, len(args.transfer_functions) > 1, False)
            key = dataset, classifier, algorithm
            if key in indexed:
                raise ValueError(f'Ambiguous plot observations: {key}')
            indexed[key] = row
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            if classifier not in classifiers:
                classifiers.append(classifier)
    metrics = [metric for metric in METRICS if all(metric.run_key in row for row in indexed.values())]
    return CompletedReport(plot_args, results, indexed, list(results), classifiers, algorithms, metrics, 'in-memory', {})


def metric_token(metric):
    return {'AccRuns': 'accuracy', 'PSRuns': 'precision', 'RSRuns': 'recall', 'F1Runs': 'f1',
            'FitRuns': 'fitness', 'FeatRuns': 'features', 'TimeRuns': 'runtime'}[metric.run_key]


def run_values(report, classifier, key, scale=1):
    """Preserve the established main distributions over observed cached runs."""
    return np.asarray([np.concatenate([np.asarray(report.indexed[ds, classifier, algorithm][key], dtype=float)
                                      for ds in report.datasets]) / scale for algorithm in report.algorithms])


def metric_values(df, metric, classifier, datasets, algorithms):
    """Preserve observed dataset order; absent/duplicate cells fail explicitly."""
    sub = df[df.Estimator == classifier]
    if sub.duplicated(['Dataset', 'Optimizer']).any():
        raise ValueError(f'Ambiguous metric observations for {classifier}')
    return np.stack([sub[sub.Optimizer == opt].set_index('Dataset').loc[list(datasets), metric].to_numpy(float)
                     for opt in algorithms])


def save_png(fig, path):
    try:
        target = Path(path).with_suffix('.png')
        base_style.neutral_text(fig)
        with report_stage(f'Save {target.name} (600 dpi PNG)'):
            framework()._save_figure(fig, target, save_pdf=False, bbox_inches='tight')
        if not target.is_file():
            raise ValueError(f'Missing generated figure: {target}')
    finally:
        plt.close(fig)


def metric_matrix(report, classifier, metric):
    """Algorithm × dataset run means, in stored units adjusted by Metric.scale."""
    return np.asarray([[np.mean(report.indexed[ds, classifier, opt][metric.run_key]) / metric.scale
                        for ds in report.datasets] for opt in report.algorithms])


def dataset_mean_ci(values):
    """Student-t 95% intervals across dataset means; never over flattened runs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Expected finite algorithm-by-dataset observations')
    n = values.shape[1]
    return values.mean(axis=1), (t.ppf(.975, n - 1) * values.std(axis=1, ddof=1) / np.sqrt(n)
                                 if n > 1 else None)


def panel_grid(count, *, width=5, height=4, polar=False):
    if count < 1:
        raise ValueError('A figure needs at least one observed panel')
    rows, columns = base_style.grid_shape(count)
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows),
                             squeeze=False,
                             subplot_kw={'polar': True} if polar else None)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def algorithm_ticks(ax, algorithms, *, horizontal=False):
    labels = [base_style.display_label(a, algorithms) for a in algorithms]
    if horizontal:
        ax.set_yticks(range(len(algorithms)), labels)
    else:
        ax.set_xticks(range(len(algorithms)), labels, rotation=45, ha='right')


def summary_figure(report, classifier=None):
    colors = palette(report.algorithms)
    available = {m.run_key: m for m in report.metrics}
    # Base overview keeps the numbered FULL classifier-by-classification-metric layout.
    metrics = ([available[k] for k in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs') if k in available]
               if classifier is None else report.metrics)
    classifiers = report.classifiers if classifier is None else [classifier]
    columns = min(4, len(metrics))
    rows_per_classifier = math.ceil(len(metrics) / columns)
    rows = rows_per_classifier * len(classifiers)
    fig, axes = plt.subplots(rows, columns, figsize=(max(5, 4.2*columns), 2.75*rows+2.2), squeeze=False)
    for ci, cls in enumerate(classifiers):
        for mi, metric in enumerate(metrics):
            r, c = ci*rows_per_classifier + mi//columns, mi % columns
            ax = axes[r, c]
            values = metric_matrix(report, cls, metric).mean(axis=1)
            base_style.metric_bars(ax, values, report.algorithms, colors)
            algorithm_ticks(ax, report.algorithms)
            ax.tick_params(labelsize=8)
            ax.set_ylim(min(0., float(values.min())*1.2),
                        1.10 if metric.unit == '0–1' else max(1., float(values.max())*1.2))
            ax.set_ylabel(cls.upper() if classifier is None else f'{metric.name} ({metric.unit})',
                          fontsize=12 if classifier is None else 10, fontweight='bold', color='black')
            if ci == 0 or classifier is not None:
                header_index = {'AccRuns': 0, 'PSRuns': 1, 'RSRuns': 2, 'F1Runs': 3}.get(metric.run_key, mi)
                base_style.metric_header(ax, metric.name, header_index)
            style_axes(ax)
        for mi in range(len(metrics), rows_per_classifier*columns):
            axes[ci*rows_per_classifier + mi//columns, mi % columns].set_visible(False)
    fig.legend(handles=[Patch(color=colors[a], label=base_style.display_label(a, report.algorithms)) for a in report.algorithms],
               loc='lower center', ncol=min(6, len(report.algorithms)), fontsize=9, framealpha=.95)
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig


def radar_values(report, classifier):
    """Classification metrics plus the established per-dataset feature efficiency."""
    available = {metric.run_key: metric for metric in report.metrics}
    metrics = [available[key] for key in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs') if key in available]
    labels = [metric.name for metric in metrics]
    values = [metric_matrix(report, classifier, metric) for metric in metrics]
    if 'FeatRuns' in available:
        features = metric_matrix(report, classifier, available['FeatRuns'])
        values.append(1 - features / np.maximum(features.max(axis=0), 1.0))
        labels.append('Feature\nefficiency')
    if not values:
        return labels, np.empty((len(report.algorithms), len(report.datasets), 0))
    return labels, np.stack(values, axis=-1)


def radar_figure(report, classifier, labels, values):
    colors = palette(report.algorithms)
    fig, axes = panel_grid(len(report.datasets), width=5, height=4.8, polar=True)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.r_[angles, angles[0]]
    for di, (ax, dataset) in enumerate(zip(axes, report.datasets)):
        for ai in curve_draw_order(report.algorithms):
            algorithm = report.algorithms[ai]
            observed = values[ai, di]
            highlighted = base_style.method_key(algorithm) == 'DSADE'
            ax.plot(angles, np.r_[observed, observed[0]], color=colors[algorithm],
                    label=base_style.display_label(algorithm, report.algorithms), **base_style.line_style(algorithm),
                    markersize=4, linewidth=2.1 if highlighted else 1.2,
                    zorder=3 if highlighted else 2,
                    markeredgecolor=colors[algorithm])
            ax.fill(angles, np.r_[observed, observed[0]], color=colors[algorithm], alpha=.04)
        ax.set_xticks(angles[:-1], labels, fontsize=8)
        ax.set_ylim(min(0., float(values.min())), max(1., float(values.max())))
        ax.set_title(f'{dataset} / {classifier.upper()}', fontsize=11, fontweight='bold', pad=14)
    handles, names = axes[0].get_legend_handles_labels()
    legend_order = np.argsort(curve_draw_order(report.algorithms))
    handles = [handles[i] for i in legend_order]
    names = [names[i] for i in legend_order]
    fig.legend(handles, names, loc='lower center', ncol=min(6, len(names)), fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    return fig


def heatmap_figure(report, classifier, metric):
    values = metric_matrix(report, classifier, metric)
    fig, ax = plt.subplots(figsize=base_style.figure_size('heatmap', len(report.algorithms), len(report.datasets)))
    normalized = metric.run_key in {'AccRuns', 'F1Runs', 'PSRuns', 'RSRuns'}
    im = ax.imshow(values, aspect='auto', cmap='Blues', vmin=0 if normalized else None, vmax=1 if normalized else None)
    ax.set_xticks(range(len(report.datasets)), report.datasets, rotation=35, ha='right')
    algorithm_ticks(ax, report.algorithms, horizontal=True)
    ax.set_title(f'{classifier.upper()} — {metric.name} ({metric.unit})')
    fig.colorbar(im, ax=ax, label=f'{metric.name} ({metric.unit}): cached run mean', shrink=.8)
    for i, algorithm in enumerate(report.algorithms):
        if base_style.method_key(algorithm) == 'DSADE':
            framework().add_heatmap_row_outline(ax, i, len(report.datasets))
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f'{values[i,j]:.4f}' if normalized else f'{values[i,j]:.4g}', ha='center', va='center', fontsize=8,
                color='white' if im.norm(values[i,j]) > .8 else 'black')
    fig.tight_layout()
    return fig


def precision_figure(values, algorithms, classifier):
    means, intervals = dataset_mean_ci(values)
    colors = palette(algorithms)
    fig, ax = plt.subplots(figsize=(max(7, max(map(len, algorithms)) * .12), max(3, len(algorithms)*.45)), layout='constrained')
    for i, algorithm in enumerate(algorithms):
        highlighted = base_style.method_key(algorithm) == 'DSADE'
        ax.errorbar(means[i], i, xerr=None if intervals is None else intervals[i], fmt='o',
                    color=colors[algorithm], markersize=8 if highlighted else 6, capsize=4,
                    markeredgecolor='black' if highlighted else colors[algorithm])
        ax.annotate(f'{means[i]:.4f}', (means[i], i), xytext=(8, 7), textcoords='offset points', fontsize=9)
    algorithm_ticks(ax, algorithms, horizontal=True)
    ax.invert_yaxis()
    ax.set_xlabel('Average precision (test)' + (' ± 95% CI' if intervals is not None else ' (CI unavailable)'))
    ax.set_title(classifier.upper())
    style_axes(ax, True)
    return fig


def observations(ax, values, algorithms, *, means=False, points=True):
    colors = palette(algorithms)
    for i, algorithm in enumerate(algorithms):
        highlighted = base_style.method_key(algorithm) == 'DSADE'
        if points:
            ax.scatter(i + np.linspace(-.08, .08, values.shape[1]), values[i], s=45 if highlighted else 35,
                       color=colors[algorithm], edgecolor='black' if highlighted else 'white',
                       linewidth=1.2 if highlighted else .5, zorder=4)
        if means:
            if points:
                ax.scatter(i, values[i].mean(), marker='D', s=140, color='black', edgecolor='white', zorder=5)
                ax.annotate(f'{values[i].mean():.3f}', (i, values[i].mean()), xytext=(5, 8),
                            textcoords='offset points', fontsize=8)
            else:
                ax.plot(i, values[i].mean(), marker='D', markersize=8, color='black',
                        markeredgecolor='white', linestyle='none', zorder=5)
    algorithm_ticks(ax, algorithms)
    style_axes(ax)


def distribution_points(ax, values, algorithms):
    """Small run observations with reproducible horizontal-only jitter."""
    rng = np.random.default_rng(0)
    for i, (algorithm, color) in enumerate(palette(algorithms).items()):
        jitter = rng.uniform(-.06, .06, len(values[i]))
        ax.scatter(i + jitter, values[i], s=12, alpha=.65, color=color,
                   edgecolor='white', linewidth=.25, zorder=3)


def mean_value_labels(ax, values):
    """Center mean numbers beside their markers, away from the median line."""
    for i, sample in enumerate(values):
        mean = float(np.mean(sample))
        above = mean >= float(np.median(sample))
        label = ax.annotate(f'{mean:.3f}', (i, mean),
                            xytext=(0, 9 if above else -9), textcoords='offset points',
                            ha='center', va='bottom' if above else 'top',
                            fontsize=8, color='black', fontweight='bold', zorder=6)
        label.set_in_layout(False)


def boxplot_figure(values, algorithms, classifier, *, show_points=True, metric_name='Accuracy'):
    fig, ax = plt.subplots(figsize=base_style.figure_size('boxplot', len(algorithms)))
    boxes = ax.boxplot(values.T, positions=np.arange(len(algorithms)), patch_artist=True, widths=.55,
                       showfliers=False, showmeans=True, medianprops={'color': 'black', 'linewidth': 1.5},
                       **({} if show_points else {'meanprops': {'marker': 'D', 'markersize': 8,
                           'markerfacecolor': 'black', 'markeredgecolor': 'white'}}))
    for box, algorithm, color in zip(boxes['boxes'], algorithms, palette(algorithms).values()):
        box.set_facecolor(color); box.set_alpha(.70)
        base_style.highlight_patch(box, algorithm, 2.8)
    observations(ax, values, algorithms, means=show_points, points=show_points)
    ax.set_ylabel(f'{metric_name} (test): one cached run mean per dataset')
    ax.set_title(classifier.upper())
    ax.legend(handles=[Line2D([], [], marker='D', color='none', markerfacecolor='black', label='Mean'),
                       Line2D([], [], color='black', label='Median')], loc='lower right')
    fig.tight_layout()
    return fig


def dataset_boxplot_figure(report, classifier, metric=None):
    if metric is None:
        metric = next(m for m in report.metrics if m.run_key == 'AccRuns')
    fig, axes = panel_grid(len(report.datasets), width=5.8, height=4.6)
    for ax, dataset in zip(axes, report.datasets):
        values = np.asarray([report.indexed[dataset, classifier, a][metric.run_key] for a in report.algorithms]) / metric.scale
        boxes = ax.boxplot(values.T, positions=np.arange(len(report.algorithms)), patch_artist=True,
                           widths=.55, showmeans=True, medianprops={'color': 'black'})
        for box, algorithm, color in zip(boxes['boxes'], report.algorithms, palette(report.algorithms).values()):
            box.set_facecolor(color); box.set_alpha(.6)
            base_style.highlight_patch(box, algorithm, 2.5)
        algorithm_ticks(ax, report.algorithms)
        ax.set_ylim(0, 1.08); ax.set_ylabel(f'{metric.name} (test)')
        ax.set_title(f'{dataset} / {classifier.upper()}')
        style_axes(ax)
    fig.tight_layout()
    return fig


def violin_figure(values, algorithms, classifier, *, show_points=True, metric_name='Recall'):
    """Densities use every supplied value; point visibility never changes the sample."""
    fig, ax = plt.subplots(figsize=base_style.figure_size('violin', len(algorithms)))
    for i, (algorithm, color) in enumerate(palette(algorithms).items()):
        if len(values[i]) > 1 and np.ptp(values[i]) > 0:
            parts = ax.violinplot([values[i]], positions=[i], widths=.78, showextrema=False)
            parts['bodies'][0].set_facecolor(color)
            parts['bodies'][0].set_alpha(.22)
            if base_style.method_key(algorithm) == 'DSADE':
                parts['bodies'][0].set_edgecolor('black')
                parts['bodies'][0].set_linewidth(2.4)
        ax.hlines(np.median(values[i]), i-.3, i+.3, colors='black', linestyles='--', linewidth=1.3)
    observations(ax, values, algorithms, means=True, points=show_points)
    ax.set_ylabel(f'{metric_name} (test): one cached run mean per dataset')
    ax.set_title(classifier.upper())
    ax.legend(handles=[Line2D([], [], marker='D', color='none', markerfacecolor='#777777', label='Mean'),
                       Line2D([], [], color='black', linestyle='--', label='Median')]
                       + ([Line2D([], [], marker='o', color='none', markerfacecolor='#777777', label='Dataset mean')]
                          if show_points else []),
              loc='lower right', framealpha=.9)
    fig.tight_layout()
    return fig


def curve_draw_order(algorithms):
    """Paint primary curves last without changing stored algorithm order."""
    return sorted(range(len(algorithms)), key=lambda i: base_style.method_key(algorithms[i]) == 'DSADE')


def final_stage_inset(ax, curves, algorithms, colors, language):
    """Magnify the lowest final-stage curves in the lower-right corner."""
    # These are child axes, so the six dataset panels and common legend stay intact.
    shortest = min(len(curve) for curve in curves)
    start = min(int(.75 * shortest), shortest - 2)
    inset = ax.inset_axes((.56, .16, .39, .30))
    inset.set_in_layout(False)
    for i in curve_draw_order(algorithms):
        algorithm, curve = algorithms[i], curves[i]
        highlighted = base_style.method_key(algorithm) == 'DSADE'
        inset.plot(np.arange(start, len(curve)), curve[start:], color=colors[algorithm],
                   label=base_style.display_label(algorithm, algorithms), **base_style.line_style(algorithm),
                   markersize=3, markevery=max(1, (len(curve)-start)//5),
                   linewidth=2.4 if highlighted else 1.3, zorder=3 if highlighted else 2)
    # Use the bottom band of final fitness values, capped at the lower half.
    # High methods remain plotted but cannot expand the zoom, including DSA-DE.
    finals = np.asarray([curve[-1] for curve in curves])
    cutoff = min(float(np.median(finals)), float(finals.min() + .1*np.ptp(finals)))
    tail = np.concatenate([curve[start:] for curve in curves if curve[-1] <= cutoff])
    padding = max(float(np.ptp(tail))*.12, float(np.max(np.abs(tail)))*.001, 1e-6)
    inset.set_xlim(start, max(len(curve)-1 for curve in curves))
    inset.set_ylim(float(tail.min())-padding, float(tail.max())+padding)
    inset.set_title(visible_text('Final stage', language), fontsize=8, fontweight='normal', pad=3)
    inset.tick_params(labelsize=6)
    inset.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    inset.yaxis.get_offset_text().set_fontsize(6)
    inset.grid(alpha=.2)
    return inset


def convergence_figure(report, classifier, datasets, *, skipped=None):
    fig, axes = panel_grid(len(datasets), width=5.8, height=4.4)
    colors = palette(report.algorithms)
    language = figure_language(report)
    for ax, ds in zip(axes, datasets):
        curves, missing = [], []
        for i in curve_draw_order(report.algorithms):
            algorithm = report.algorithms[i]
            curve = np.asarray(report.indexed[ds, classifier, algorithm].get('Curve', []), dtype=float)
            if curve.ndim != 1 or not curve.size or not np.isfinite(curve).all():
                missing.append(algorithm)
                continue
            curves.append((algorithm, curve))
            highlighted = base_style.method_key(algorithm) == 'DSADE'
            ax.plot(np.arange(len(curve)), curve, color=colors[algorithm],
                    label=base_style.display_label(algorithm, report.algorithms),
                    **base_style.line_style(algorithm), markersize=4, markevery=max(1, len(curve)//12),
                    linewidth=2.4 if highlighted else 1.3, zorder=3 if highlighted else 2)
        ax.set_title(f'{ds} / {classifier.upper()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9); ax.set_ylabel('Fitness', fontsize=9)
        style_axes(ax)
        if not missing and min((len(curve) for _, curve in curves), default=0) >= 3:
            # Supply stored curves in the same order as their algorithm names.
            stored_curves = dict(curves)
            final_stage_inset(ax, [stored_curves[a] for a in report.algorithms],
                              report.algorithms, colors, language)
        else:
            reason = (f'Final-stage inset unavailable: missing/invalid stored curves for {missing}'
                      if missing else 'Final-stage inset unavailable: fewer than three stored iterations')
            if skipped is not None:
                skipped.append({'output': f'Convergence inset {ds}/{classifier}', 'reason': reason})
            if not curves:
                ax.text(.5, .5, visible_text('No stored curves', language), transform=ax.transAxes, ha='center')
    handles = [Line2D([], [], color=colors[a], label=base_style.display_label(a, report.algorithms),
                     **base_style.line_style(a), markersize=4,
                     linewidth=2.4 if base_style.method_key(a) == 'DSADE' else 1.3) for a in report.algorithms]
    fig.legend(handles=handles, loc='lower center', ncol=min(6, len(handles)), fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    return fig


def tradeoff_figure(features, runtime, algorithms, classifier):
    fig, ax = plt.subplots(figsize=base_style.figure_size('features_runtime', len(algorithms)))
    draw_tradeoff_axes(ax, features.mean(axis=1), runtime.mean(axis=1), algorithms)
    ax.set_title(classifier.upper())
    fig.tight_layout()
    return fig


def draw_tradeoff_axes(ax, features, runtime, algorithms, *, compact_labels=False):
    twin = ax.twinx()
    colors = list(palette(algorithms).values())
    for axis, values, shift, hatch, alpha in ((ax, features, -.19, None, .85),
                                             (twin, runtime, .19, '///', .45)):
        bars = axis.bar(np.arange(len(algorithms))+shift, values, width=.38, color=colors, hatch=hatch, alpha=alpha)
        for bar, algorithm, value in zip(bars, algorithms, values):
            base_style.highlight_patch(bar, algorithm, 2.8)
            axis.annotate(f'{value:.1f}s' if hatch else f'{value:.2f}',
                          (bar.get_x() + bar.get_width()/2, value), xytext=(0, 3 if compact_labels else 6),
                          textcoords='offset points', ha='center', va='bottom',
                          fontsize=6 if compact_labels else 8, rotation=90 if compact_labels else 0,
                          fontweight='bold' if base_style.method_key(algorithm) == 'DSADE' else 'normal')
        axis.set_ylim(0, max(1., float(values.max())*(1.4 if compact_labels else 1.2)))
        if compact_labels:
            axis.tick_params(axis='y', labelsize=8)
    algorithm_ticks(ax, algorithms)
    ax.set_ylabel('Average selected features'); twin.set_ylabel('Average runtime (s)')
    style_axes(ax)
    if compact_labels:
        ax.tick_params(axis='x', labelsize=7)
        ax.yaxis.label.set_size(9)
        twin.yaxis.label.set_size(9)
    else:
        ax.legend(handles=[Patch(facecolor='#777777', label='Selected features'),
                           Patch(facecolor='#777777', hatch='///', alpha=.45, label='Runtime')], framealpha=.95)


def dataset_tradeoff_figure(report, classifier, *, compact_labels=False):
    available = {m.run_key: m for m in report.metrics}
    features = metric_matrix(report, classifier, available['FeatRuns'])
    runtime = metric_matrix(report, classifier, available['TimeRuns'])
    fig, axes = panel_grid(len(report.datasets), width=5.8, height=4.6)
    for i, (ax, dataset) in enumerate(zip(axes, report.datasets)):
        draw_tradeoff_axes(ax, features[:, i], runtime[:, i], report.algorithms, compact_labels=compact_labels)
        ax.set_title(f'{dataset} / {classifier.upper()}')
    if compact_labels:
        fig.legend(handles=[Patch(facecolor='#777777', label='Selected features'),
                            Patch(facecolor='#777777', hatch='///', alpha=.45, label='Runtime')],
                   loc='lower center', ncol=2, fontsize=9, framealpha=.95)
    fig.tight_layout(rect=(0, .06, 1, 1) if compact_labels else (0, 0, 1, 1))
    return fig


def publication_figures(report, skipped):
    """Generic views for every classifier; FULL reserves metric heatmaps for 06."""
    available = {metric.run_key: metric for metric in report.metrics}
    for classifier in report.classifiers:
        suffix = filename_component(classifier)
        yield f'generic_summary_{suffix}', summary_figure(report, classifier)
        labels, radar = radar_values(report, classifier)
        if len(labels) >= 3:
            yield f'generic_radar_{suffix}', radar_figure(report, classifier, labels, radar)
        else:
            skipped.append({'output': f'Radar/{classifier}', 'reason': 'Fewer than three available classification/feature-efficiency axes'})
        if report.args.experiment_mode != 'full':
            for metric in report.metrics:
                yield f'generic_heatmap_{suffix}_{metric_token(metric)}', heatmap_figure(report, classifier, metric)
        for key, stem, generate_figure in (('PSRuns', 'precision', precision_figure),
                                           ('AccRuns', 'accuracy_boxplot', boxplot_figure),
                                           ('RSRuns', 'recall_violin', violin_figure)):
            if key not in available:
                skipped.append({'output': f'{stem}/{classifier}', 'reason': f'{key} unavailable'})
                continue
            values = metric_matrix(report, classifier, available[key])
            if key == 'PSRuns' and values.shape[1] < 2:
                skipped.append({'output': f'Precision CI/{classifier}', 'reason': 'At least two dataset means required; only observed means are plotted'})
            if key == 'RSRuns':
                for i, algorithm in enumerate(report.algorithms):
                    if values.shape[1] < 2 or np.ptp(values[i]) == 0:
                        skipped.append({'output': f'Violin density/{classifier}/{algorithm}',
                                        'reason': 'Insufficient or constant dataset means; observations, mean and median retained'})
            yield f'generic_{stem}_{suffix}', generate_figure(values, report.algorithms, classifier)
        complete = []
        for ds in report.datasets:
            missing = [a for a in report.algorithms if not np.asarray(report.indexed[ds, classifier, a].get('Curve', [])).size]
            if missing:
                skipped.append({'output': f'Convergence {ds}/{classifier}', 'reason': f'No stored mean curve for {missing}'})
            else:
                complete.append(ds)
        if complete:
            yield f'generic_convergence_{suffix}', convergence_figure(report, classifier, complete)
        if {'FeatRuns', 'TimeRuns'} <= available.keys():
            yield f'generic_features_runtime_{suffix}', tradeoff_figure(
                metric_matrix(report, classifier, available['FeatRuns']),
                metric_matrix(report, classifier, available['TimeRuns']), report.algorithms, classifier)
        else:
            skipped.append({'output': f'Features/runtime/{classifier}', 'reason': 'Requires both FeatRuns and TimeRuns'})


def base_classifier(report):
    """One presentation-only main classifier, shared by FULL and replicas."""
    configured = str(getattr(report.args, 'plot_global_estimator', framework().PLOT_GLOBAL_ESTIMATOR)).lower()
    if configured not in report.classifiers and hasattr(report.args, 'plot_global_estimator'):
        raise ValueError(f'Selected publication classifier {configured!r} has no stored results; '
                         f'available classifiers: {report.classifiers}')
    return configured if configured in report.classifiers else report.classifiers[0]


def base_metric_token(report):
    configured = str(getattr(report.args, 'plot_global_metric', framework().PLOT_GLOBAL_METRIC)).lower()
    if configured not in METRIC_TOKENS:
        raise ValueError(f'Unsupported publication metric: {configured}')
    return configured


def figure_language(report):
    language = getattr(report.args, 'figure_language', framework().FIGURE_LANGUAGE)
    if language not in ('en', 'es'):
        raise ValueError(f'Unsupported figure language: {language}')
    return language


def base_figure_names(report):
    return main_figure_names(base_classifier(report), base_metric_token(report))


def base_publication_figures(report, skipped):
    """Localize root figure text only; filenames and stored identities are unchanged."""
    language = figure_language(report)
    protected = [*report.datasets, *report.algorithms,
                 *(base_style.display_label(a, report.algorithms) for a in report.algorithms)]
    for stem, fig in _base_publication_figures(report, skipped):
        localize_figure(fig, language, protected=protected)
        if language == 'es':
            bottom = {'01': .06, '02': .08, '03': .06, '05': .08}.get(stem[:2], 0)
            fig.tight_layout(rect=(0, bottom, 1, 1))
        yield stem, fig


def _base_publication_figures(report, skipped):
    """The same nine publication views for normal FULL and cache-only reports."""
    names = base_figure_names(report)
    classifier = base_classifier(report)
    available = {m.run_key: m for m in report.metrics}
    metric = next((m for m in report.metrics if metric_token(m) == base_metric_token(report)), None)
    if any(k in available for k in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs')):
        yield Path(names[0]).stem, summary_figure(report)
    else:
        skipped.append({'output': names[0], 'reason': 'No classification metrics'})
    labels, values = radar_values(report, classifier)
    if len(labels) >= 3:
        yield Path(names[1]).stem, radar_figure(report, classifier, labels, values)
    else:
        skipped.append({'output': names[1], 'reason': 'Fewer than three radar axes'})
    if {'FeatRuns', 'TimeRuns'} <= available.keys():
        yield Path(names[2]).stem, dataset_tradeoff_figure(report, classifier, compact_labels=True)
    else:
        skipped.append({'output': names[2], 'reason': 'Requires FeatRuns and TimeRuns'})
    if metric is not None:
        yield Path(names[3]).stem, dataset_boxplot_figure(report, classifier, metric)
    else:
        skipped.append({'output': names[3], 'reason': f'{base_metric_token(report)} unavailable'})
    if metric is not None:
        fig = heatmap_figure(report, classifier, metric)
        fig.axes[0].set_xlabel('Dataset'); fig.axes[0].set_ylabel('Metaheuristics')
        yield Path(names[5]).stem, fig
    else:
        skipped.append({'output': names[5], 'reason': f'{base_metric_token(report)} unavailable'})
    yield Path(names[4]).stem, convergence_figure(report, classifier, report.datasets, skipped=skipped)
    if metric is not None:
        values = run_values(report, classifier, metric.run_key, metric.scale)
        fig = violin_figure(values, report.algorithms,
                           classifier, show_points=False, metric_name=metric.name)
        fig.axes[0].set_ylabel(f'{metric.name} (test): cached runs across datasets')
        mean_value_labels(fig.axes[0], values)
        yield Path(names[6]).stem, fig
    else:
        skipped.append({'output': names[6], 'reason': f'{base_metric_token(report)} unavailable'})
    if metric is not None:
        values = run_values(report, classifier, metric.run_key, metric.scale)
        fig = boxplot_figure(values, report.algorithms,
                            classifier, show_points=False, metric_name=metric.name)
        fig.axes[0].set_ylabel(f'{metric.name} (test): cached runs across datasets')
        mean_value_labels(fig.axes[0], values)
        yield Path(names[7]).stem, fig
    else:
        skipped.append({'output': names[7], 'reason': f'{base_metric_token(report)} unavailable'})
    if {'FeatRuns', 'TimeRuns'} <= available.keys():
        yield Path(names[8]).stem, tradeoff_figure(metric_matrix(report, classifier, available['FeatRuns']),
            metric_matrix(report, classifier, available['TimeRuns']), report.algorithms, classifier)
    else:
        skipped.append({'output': names[8], 'reason': 'Requires FeatRuns and TimeRuns'})


def per_dataset_figures(report, skipped):
    """Extra single-smell panels always identify the smell and classifier."""
    available = {m.run_key: m for m in report.metrics}
    for dataset in report.datasets:
        single = replace(report, datasets=[dataset])
        for classifier in report.classifiers:
            suffix = f'{filename_component(dataset)}_{filename_component(classifier)}'
            labels, values = radar_values(single, classifier)
            if len(labels) >= 3:
                yield f'radar_{suffix}', radar_figure(single, classifier, labels, values)
            if all(np.asarray(single.indexed[dataset, classifier, a].get('Curve', [])).size for a in report.algorithms):
                yield f'convergence_{suffix}', convergence_figure(single, classifier, [dataset])
            if report.args.experiment_mode != 'full':
                for metric in report.metrics:
                    yield f'heatmap_{suffix}_{metric_token(metric)}', heatmap_figure(single, classifier, metric)
            if {'FeatRuns', 'TimeRuns'} <= available.keys():
                yield f'features_runtime_{suffix}', dataset_tradeoff_figure(single, classifier)


def individual_destination(report, destination):
    return Path(destination) / INDIVIDUAL_DIRECTORY if report.args.experiment_mode == 'full' else Path(destination)


def statistics_destination(report, destination):
    return Path(destination) / STATISTICS


def generate(report, destination, *, generated=None):
    skipped = []
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    individual = individual_destination(report, destination)
    individual.mkdir(parents=True, exist_ok=True)
    if report.args.experiment_mode == 'full':
        full_figure_directories(destination)
    with plt.rc_context(STYLE):
        if report.args.experiment_mode == 'full':
            for stem, fig in base_publication_figures(report, skipped):
                target = full_figure_path(destination, f'{stem}.png')
                save_png(fig, target)
                if generated is not None:
                    generated.append(str(target.relative_to(destination)))
        for stem, fig in publication_figures(report, skipped):
            target = (full_figure_path(destination, f'{stem}.png')
                      if report.args.experiment_mode == 'full' else individual / f'{stem}.png')
            save_png(fig, target)
            if generated is not None:
                generated.append(str(target.relative_to(destination)))
        if report.args.experiment_mode == 'full':
            for stem, fig in per_dataset_figures(report, skipped):
                target = individual / f'{stem}.png'
                save_png(fig, target)
                if generated is not None:
                    generated.append(str(target.relative_to(destination)))
    return skipped


def statistical_figures(analysis, algorithms, metric):
    x, ranked, ranks = analysis['x'], analysis['ranked'], analysis['mean_ranks']
    labels = [algorithms[i] for i in ranked]
    k, n = len(algorithms), len(x)
    fig, ax = plt.subplots(figsize=(max(6, max(map(len, algorithms)) * .12), max(3, k * .45)), layout='constrained')
    bars = ax.barh(range(k), ranks[ranked], color=list(palette(labels).values()))
    for bar, label in zip(bars, labels):
        base_style.highlight_patch(bar, label)
    ax.set_yticks(range(k), [base_style.display_label(a, algorithms) for a in labels]); ax.invert_yaxis()
    ax.set_xlabel('Average rank (1 = best)')
    style_axes(ax, True)
    yield 'generic_average_rank', fig

    reference = algorithms[0]
    pairs = analysis['pairs']
    comparisons = pairs[(pairs.Algorithm_A == reference) & np.isfinite(pairs.Holm_adjusted_p)]
    if len(comparisons):
        fig, ax = plt.subplots(figsize=(8, max(3, len(comparisons)*.5)), layout='constrained')
        values = comparisons.Holm_adjusted_p.to_numpy()
        # Keep zero p-values visible without assigning a made-up positive value.
        ax.scatter(values, np.arange(len(values)), color=palette([reference])[reference])
        for i, value in enumerate(values):
            ax.annotate(f'{value:.5g}', (value, i), xytext=(5, 5), textcoords='offset points')
        ax.axvline(.05, linestyle='--', color='#777777')
        ax.set_yticks(range(len(values)), [base_style.display_label(a, algorithms) for a in comparisons.Algorithm_B]); ax.invert_yaxis()
        ax.set_xlim(-.02, 1.08)
        ax.set_xlabel(f'Holm-adjusted p; {len(pairs)}-pair family')
        ax.set_title(f'{base_style.display_label(reference, algorithms)} versus other algorithms (configured reference)')
        style_axes(ax, True)
        yield 'generic_reference_comparisons', fig

    fig, ax = plt.subplots(figsize=(max(5, k * .6), max(4, k * .5)), layout='constrained')
    im = ax.imshow(np.ma.masked_invalid(analysis['matrix']), vmin=0, vmax=1, cmap='Greys_r')
    labels_all = [base_style.display_label(a, algorithms) for a in algorithms]
    ax.set_xticks(range(k), labels_all, rotation=45, ha='right'); ax.set_yticks(range(k), labels_all)
    for i, j in np.ndindex((k, k)):
        p = analysis['matrix'][i, j]
        ax.text(j, i, f'{p:.3g}' if np.isfinite(p) else 'N/A', ha='center', va='center', fontsize=8,
                color='white' if p < .5 else 'black')
    fig.colorbar(im, ax=ax, label='Holm-adjusted p (all algorithm pairs)')
    yield 'generic_holm_heatmap', fig

    fig, ax = plt.subplots(figsize=(max(6, k * .65), 4), layout='constrained')
    boxes = ax.boxplot(x[:, ranked], positions=np.arange(k), showfliers=False, patch_artist=True, widths=.55)
    for box, label in zip(boxes['boxes'], labels):
        box.set_facecolor(palette(labels)[label]); box.set_alpha(.7)
        base_style.highlight_patch(box, label, 2.8)
    for pos, i in enumerate(ranked):
        ax.scatter(pos + np.linspace(-.18, .18, n), x[:, i], s=15, color=palette(algorithms)[algorithms[i]])
    ax.set_xticks(range(k), [base_style.display_label(a, algorithms) for a in labels], rotation=45, ha='right')
    ax.set_ylabel(f'{metric}: cached run mean per matched block')
    style_axes(ax)
    yield 'generic_block_distribution', fig
