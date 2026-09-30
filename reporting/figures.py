"""Publication figures derived from actual cached datasets, classifiers and algorithms."""
from pathlib import Path
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


STYLE = base_style.STYLE
palette = base_style.palette
style_axes = base_style.style_axes

BASE_FIGURES = (
    'grafica_resumen_general.png', 'radar_6smells_grid_svm.png',
    'ranking_precision.png', 'boxplot_accuracy_general.png', 'heatmap_f1score.png',
    'violin_recall.png', 'convergence_curve.png', 'features_runtime_per_optimizer.png',
)
INDIVIDUAL_DIRECTORY = 'individual'


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
    labels = [base_style.display_label(a) for a in algorithms]
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
    fig.legend(handles=[Patch(color=colors[a], label=base_style.display_label(a)) for a in report.algorithms],
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
        for ai, algorithm in enumerate(report.algorithms):
            observed = values[ai, di]
            highlighted = base_style.method_key(algorithm) == 'DSADE'
            ax.plot(angles, np.r_[observed, observed[0]], color=colors[algorithm],
                    label=base_style.display_label(algorithm), **base_style.line_style(algorithm),
                    markersize=4, linewidth=2.4 if highlighted else 1.1,
                    markeredgecolor='black' if highlighted else colors[algorithm])
            ax.fill(angles, np.r_[observed, observed[0]], color=colors[algorithm], alpha=.12 if highlighted else .04)
        ax.set_xticks(angles[:-1], labels, fontsize=8)
        ax.set_ylim(min(0., float(values.min())), max(1., float(values.max())))
        ax.set_title(f'{dataset} / {classifier.upper()}', fontsize=11, fontweight='bold', pad=14)
    handles, names = axes[0].get_legend_handles_labels()
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


def observations(ax, values, algorithms, *, means=False):
    colors = palette(algorithms)
    for i, algorithm in enumerate(algorithms):
        highlighted = base_style.method_key(algorithm) == 'DSADE'
        ax.scatter(i + np.linspace(-.08, .08, values.shape[1]), values[i], s=45 if highlighted else 35,
                   color=colors[algorithm], edgecolor='black' if highlighted else 'white',
                   linewidth=1.2 if highlighted else .5, zorder=4)
        if means:
            ax.scatter(i, values[i].mean(), marker='D', s=140, color='black', edgecolor='white', zorder=5)
    algorithm_ticks(ax, algorithms)
    style_axes(ax)


def boxplot_figure(values, algorithms, classifier):
    fig, ax = plt.subplots(figsize=base_style.figure_size('boxplot', len(algorithms)))
    boxes = ax.boxplot(values.T, positions=np.arange(len(algorithms)), patch_artist=True, widths=.55,
                       showfliers=False, showmeans=True, medianprops={'color': 'black', 'linewidth': 1.5})
    for box, algorithm, color in zip(boxes['boxes'], algorithms, palette(algorithms).values()):
        box.set_facecolor(color); box.set_alpha(.70)
        base_style.highlight_patch(box, algorithm, 2.8)
    observations(ax, values, algorithms)
    ax.set_ylabel('Accuracy (test): one cached run mean per dataset')
    ax.set_title(classifier.upper())
    fig.tight_layout()
    return fig


def violin_figure(values, algorithms, classifier):
    """Draw densities only for nonconstant samples, retaining every observation."""
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
    observations(ax, values, algorithms, means=True)
    ax.set_ylabel('Recall (test): one cached run mean per dataset')
    ax.set_title(classifier.upper())
    ax.legend(handles=[Line2D([], [], marker='D', color='none', markerfacecolor='#777777', label='Mean'),
                       Line2D([], [], color='black', linestyle='--', label='Median'),
                       Line2D([], [], marker='o', color='none', markerfacecolor='#777777', label='Dataset mean')],
              loc='lower right', framealpha=.9)
    fig.tight_layout()
    return fig


def convergence_figure(report, classifier, datasets):
    fig, axes = panel_grid(len(datasets), width=5.8, height=4.4)
    colors = palette(report.algorithms)
    for ax, ds in zip(axes, datasets):
        for i, algorithm in enumerate(report.algorithms):
            curve = np.asarray(report.indexed[ds, classifier, algorithm]['Curve'])
            ax.plot(np.arange(len(curve)), curve, label=base_style.display_label(algorithm), color=colors[algorithm],
                    **base_style.line_style(algorithm), markersize=4, markevery=max(1, len(curve)//12),
                    linewidth=2.4 if base_style.method_key(algorithm) == 'DSADE' else 1.4)
        ax.set_title(f'{ds} / {classifier.upper()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9); ax.set_ylabel('Fitness', fontsize=9)
        style_axes(ax)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=min(6, len(names)), fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    return fig


def tradeoff_figure(features, runtime, algorithms, classifier):
    fig, ax = plt.subplots(figsize=base_style.figure_size('features_runtime', len(algorithms)))
    twin = ax.twinx()
    colors = list(palette(algorithms).values())
    for axis, values, shift, hatch, alpha in ((ax, features.mean(axis=1), -.19, None, .85),
                                             (twin, runtime.mean(axis=1), .19, '///', .45)):
        bars = axis.bar(np.arange(len(algorithms))+shift, values, width=.38, color=colors, hatch=hatch, alpha=alpha)
        for bar, algorithm in zip(bars, algorithms):
            base_style.highlight_patch(bar, algorithm, 2.8)
        axis.set_ylim(0, max(1., float(values.max())*1.2))
    algorithm_ticks(ax, algorithms)
    ax.set_ylabel('Average selected features'); twin.set_ylabel('Average runtime (s)')
    ax.set_title(classifier.upper())
    style_axes(ax)
    ax.legend(handles=[Patch(facecolor='#777777', label='Selected features'),
                       Patch(facecolor='#777777', hatch='///', alpha=.45, label='Runtime')], framealpha=.95)
    fig.tight_layout()
    return fig


def publication_figures(report, skipped):
    """Eight figure types, applied uniformly to every actual classifier."""
    available = {metric.run_key: metric for metric in report.metrics}
    for ci, classifier in enumerate(report.classifiers, 1):
        suffix = f'c{ci}'
        yield f'generic_summary_{suffix}', summary_figure(report, classifier)
        labels, radar = radar_values(report, classifier)
        if len(labels) >= 3:
            yield f'generic_radar_{suffix}', radar_figure(report, classifier, labels, radar)
        else:
            skipped.append({'output': f'Radar/{classifier}', 'reason': 'Fewer than three available classification/feature-efficiency axes'})
        for mi, metric in enumerate(report.metrics, 1):
            yield f'generic_heatmap_{suffix}_m{mi}', heatmap_figure(report, classifier, metric)
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
    """Preserve the historical base views' SVM; use observed data on other suites."""
    return 'svm' if 'svm' in report.classifiers else report.classifiers[0]


def base_figure_names(report):
    names = list(BASE_FIGURES)
    classifier = base_classifier(report)
    if len(report.datasets) != 6 or classifier != 'svm':
        names[1] = f'radar_{len(report.datasets)}datasets_grid_{classifier}.png'
    return tuple(names)


def base_publication_figures(report, skipped):
    """Eight existing base identities, sharing the individual figure builders."""
    names = base_figure_names(report)
    classifier = base_classifier(report)
    available = {m.run_key: m for m in report.metrics}
    if any(k in available for k in ('AccRuns', 'PSRuns', 'RSRuns', 'F1Runs')):
        yield Path(names[0]).stem, summary_figure(report)
    else:
        skipped.append({'output': names[0], 'reason': 'No classification metrics'})
    labels, values = radar_values(report, classifier)
    if len(labels) >= 3:
        yield Path(names[1]).stem, radar_figure(report, classifier, labels, values)
    else:
        skipped.append({'output': names[1], 'reason': 'Fewer than three radar axes'})
    for index, key, builder in ((2, 'PSRuns', precision_figure),
                                (3, 'AccRuns', boxplot_figure), (5, 'RSRuns', violin_figure)):
        if key in available:
            yield Path(names[index]).stem, builder(metric_matrix(report, classifier, available[key]),
                                                   report.algorithms, classifier)
        else:
            skipped.append({'output': names[index], 'reason': f'{key} unavailable'})
    if 'F1Runs' in available:
        yield Path(names[4]).stem, heatmap_figure(report, classifier, available['F1Runs'])
    else:
        skipped.append({'output': names[4], 'reason': 'F1Runs unavailable'})
    complete = [ds for ds in report.datasets if all(
        np.asarray(report.indexed[ds, classifier, a].get('Curve', [])).size for a in report.algorithms)]
    if complete:
        yield Path(names[6]).stem, convergence_figure(report, classifier, complete)
    else:
        skipped.append({'output': names[6], 'reason': 'No complete stored curves'})
    if {'FeatRuns', 'TimeRuns'} <= available.keys():
        yield Path(names[7]).stem, tradeoff_figure(metric_matrix(report, classifier, available['FeatRuns']),
            metric_matrix(report, classifier, available['TimeRuns']), report.algorithms, classifier)
    else:
        skipped.append({'output': names[7], 'reason': 'Requires FeatRuns and TimeRuns'})


def individual_destination(report, destination):
    return Path(destination) / INDIVIDUAL_DIRECTORY if report.args.experiment_mode == 'full' else Path(destination)


def generate(report, destination):
    skipped = []
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    individual = individual_destination(report, destination)
    individual.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        if report.args.experiment_mode == 'full':
            for stem, fig in base_publication_figures(report, skipped):
                save_png(fig, destination / f'{stem}.png')
        for stem, fig in publication_figures(report, skipped):
            save_png(fig, individual / f'{stem}.png')
    return skipped


def statistical_figures(analysis, algorithms, metric):
    x, ranked, ranks = analysis['x'], analysis['ranked'], analysis['mean_ranks']
    labels = [algorithms[i] for i in ranked]
    k, n = len(algorithms), len(x)
    fig, ax = plt.subplots(figsize=(max(6, max(map(len, algorithms)) * .12), max(3, k * .45)), layout='constrained')
    bars = ax.barh(range(k), ranks[ranked], color=list(palette(labels).values()))
    for bar, label in zip(bars, labels):
        base_style.highlight_patch(bar, label)
    ax.set_yticks(range(k), [base_style.display_label(a) for a in labels]); ax.invert_yaxis()
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
        ax.set_yticks(range(len(values)), [base_style.display_label(a) for a in comparisons.Algorithm_B]); ax.invert_yaxis()
        ax.set_xlim(-.02, 1.08)
        ax.set_xlabel(f'Holm-adjusted p; {len(pairs)}-pair family')
        ax.set_title(f'{reference} versus other algorithms (configured reference)')
        style_axes(ax, True)
        yield 'generic_reference_comparisons', fig

    fig, ax = plt.subplots(figsize=(max(5, k * .6), max(4, k * .5)), layout='constrained')
    im = ax.imshow(np.ma.masked_invalid(analysis['matrix']), vmin=0, vmax=1, cmap='Greys_r')
    labels_all = [base_style.display_label(a) for a in algorithms]
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
    ax.set_xticks(range(k), [base_style.display_label(a) for a in labels], rotation=45, ha='right')
    ax.set_ylabel(f'{metric}: cached run mean per matched block')
    style_axes(ax)
    yield 'generic_block_distribution', fig
