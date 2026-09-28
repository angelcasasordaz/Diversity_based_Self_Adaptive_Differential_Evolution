"""EXP627 reporting only. No experiment entry point is called."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from PIL import Image
from reporting.exp627_core import framework, load_completed_full, report_guard, sha256, DATASETS, CLASSIFIERS, CONFIG_ORDER

ORDER = ('DSA-DE', 'DE', 'JADE', 'SHADE', 'PSO', 'WOA', 'HHO', 'GOA', 'SA', 'BRO', 'RUN', 'FOX')
STEMS = ('stat_fig1_average_rank', 'stat_fig2_dsade_vs_others', 'stat_fig3_holm_heatmap', 'stat_fig4_f1_boxplot')
IDS = ('average_rank', 'dsade_pairwise_holm', 'adjusted_pvalue_heatmap', 'f1_distribution_by_algorithm')
EXPECTED_RES = {'statistical_summary.csv', 'pairwise_wilcoxon_holm.csv', 'statistical_report.txt'}
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9,
         'ytick.labelsize': 10, 'figure.dpi': 100, 'savefig.dpi': 600, 'figure.facecolor': 'white',
         'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42}


def snapshot(ROOT, FIG, RES):
    files = set(ROOT.glob('Results/**/*.pkl'))
    for kind in ('Results', 'Figures'):
        files.update(p for p in (ROOT / kind / 'EXP627').rglob('*')
                     if p.is_file() and FIG not in p.parents and RES not in p.parents)
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def validate_dest(path, expected):
    assert path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents))
    assert path.parent.is_dir()
    if path.exists():
        assert path.is_dir()
        assert all(p.is_file() and not p.is_symlink() and p.name in expected for p in path.iterdir())


def axes_style(ax, horizontal=False):
    ax.set_axisbelow(True)
    ax.grid(axis='x' if horizontal else 'y', color='#dddddd', linewidth=.6)
    ax.spines[['top', 'right']].set_visible(False)


def emphasize(labels):
    for label in labels:
        if label.get_text() == 'DSA-DE':
            label.set_fontweight('bold')


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'


def exact_signed_rank_check(d):
    # Independent subset-sum enumeration of the exact signed-rank distribution.
    assert np.all(d != 0) and len(np.unique(abs(d))) == len(d)
    ranks = rankdata(abs(d)).astype(int)
    positive = int(ranks[d > 0].sum())
    total = int(ranks.sum())
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for rank in ranks:
        counts[rank:] += counts[:-rank].copy()
    w = min(positive, total-positive)
    return w, min(1., 2*counts[:w+1].sum() / (2**len(d)))


def figures(x, mean_ranks, ranked, pairs, matrix):
    labels = [ORDER[i] for i in ranked]
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    for y, i in enumerate(ranked):
        dsade = i == 0
        ax.plot([1, mean_ranks[i]], [y, y], color='#222222' if dsade else '#aaaaaa', linewidth=2.4 if dsade else 1)
        ax.scatter(mean_ranks[i], y, marker='D' if dsade else 'o', s=75 if dsade else 35, color='#111111' if dsade else '#666666', zorder=3)
        ax.annotate(f'{mean_ranks[i]:.3f}', (mean_ranks[i], y), xytext=(9, 0), textcoords='offset points', va='center', fontweight='bold' if dsade else 'normal')
    ax.set_yticks(range(12), labels)
    ax.invert_yaxis()
    ax.set_xlim(.8, 12.8)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('Average rank (1 = best; lower is better)')
    emphasize(ax.get_yticklabels())
    axes_style(ax, True)
    yield fig

    dsade = pairs[pairs.Algorithm_A == 'DSA-DE'].set_index('Algorithm_B').loc[list(ORDER[1:])]
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    lower = min(.001, float(dsade.Holm_adjusted_p.min()) / 3)
    for y, (name, row) in enumerate(dsade.iterrows()):
        p = float(row.Holm_adjusted_p)
        sig = p < .05
        ax.hlines(y, lower, p, color='#aaaaaa', linewidth=1)
        ax.scatter(p, y, marker='D' if sig else 'o', facecolor='#222222' if sig else 'white', edgecolor='#222222', s=55, zorder=3)
        ax.text(1.03, y, f'{p:.8g}  {stars(p)}', transform=ax.get_yaxis_transform(), va='center', fontsize=10, fontweight='bold' if sig else 'normal')
    ax.axvline(.05, color='#555555', linestyle='--', linewidth=1.2)
    ax.text(.05, 1.02, r'$\alpha = 0.05$', transform=ax.get_xaxis_transform(), ha='center')
    ax.set_xscale('log')
    ax.set_xlim(lower, 1.3)
    ax.set_yticks(range(11), list(dsade.index))
    ax.invert_yaxis()
    ax.set_xlabel('Holm-adjusted p-value (log scale; DSA-DE versus each algorithm)')
    axes_style(ax, True)
    fig.supxlabel('* p < 0.05; ** p < 0.01; *** p < 0.001; ns: not significant. Holm family: all 66 pairs.', fontsize=9)
    yield fig

    fig, ax = plt.subplots(figsize=(11, 9), layout='constrained')
    display = np.ma.array((matrix < .05).astype(float), mask=np.triu(np.ones((12, 12), dtype=bool)))
    cmap = ListedColormap(['#f2f2f2', '#bacbd5'])
    cmap.set_bad('white')
    ax.imshow(display, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5], 2))
    for i in range(12):
        for j in range(i):
            p = matrix[i, j]
            ax.text(j, i, f'{p:.4g}', ha='center', va='center', fontsize=8.5, fontweight='bold' if p < .05 else 'normal')
    ax.set_xticks(range(12), ORDER, rotation=45, ha='right')
    ax.set_yticks(range(12), ORDER)
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    emphasize(ax.get_xticklabels()+ax.get_yticklabels())
    ax.legend(handles=[Patch(facecolor='#bacbd5', label='Holm-adjusted p < 0.05'), Patch(facecolor='#f2f2f2', edgecolor='#bbbbbb', label='p ≥ 0.05')], loc='upper right', frameon=False)
    yield fig

    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    boxes = ax.boxplot(x[:, ranked], positions=np.arange(12), patch_artist=True, widths=.58,
                       showfliers=False, medianprops={'color':'black', 'linewidth':1.5})
    for pos, i in enumerate(ranked):
        box = boxes['boxes'][pos]
        box.set_facecolor('#c2c2c2' if i == 0 else '#eeeeee')
        box.set_edgecolor('black' if i == 0 else '#777777')
        box.set_linewidth(2.4 if i == 0 else 1)
        # Fixed offsets preserve all 18 observations without random jitter or new seeds.
        ax.scatter(pos + np.linspace(-.18, .18, 18), x[:, i], marker='D' if i == 0 else 'o', s=20 if i == 0 else 14,
                   color='#222222' if i == 0 else '#666666', edgecolor='white', linewidth=.4, zorder=3)
    ax.set_xticks(range(12), labels, rotation=40, ha='right')
    emphasize(ax.get_xticklabels())
    ax.set_ylabel('F1_test (mean of 30 runs per matched block)')
    ax.set_xlabel('Algorithms ordered by average rank (best to worst)')
    axes_style(ax)
    yield fig


def run(args):
    """Regenerate only the established FULL_REP1 statistical reporting outputs."""
    ROOT = Path(args.output_root).resolve()
    FIG = ROOT / 'Figures/EXP627/full_rep1/statistics'
    RES = ROOT / 'Results/EXP627/full_rep1/statistics'
    from reporting.exp627_figures import validate_output_destination
    validate_output_destination(ROOT, FIG.parent)
    validate_output_destination(ROOT, RES.parent)
    m = framework()  # Import plotting dependencies before the strict write guard.
    validate_dest(FIG, {f'{s}.png' for s in STEMS})
    validate_dest(RES, EXPECTED_RES)
    before = snapshot(ROOT, FIG, RES)
    with report_guard((FIG, RES)) as guard, plt.rc_context(STYLE):
        args, results, indexed, sources = load_completed_full(args)
        blocks = tuple(itertools.product(DATASETS, CLASSIFIERS))
        assert len(blocks) == len(set(blocks)) == 18
        assert set(indexed) == {(d, c, o) for d, c in blocks for o in CONFIG_ORDER}
        x = np.array([[np.mean(np.asarray(indexed[d,c,o]['F1Runs'], dtype=float)) for o in CONFIG_ORDER] for d,c in blocks])
        assert x.shape == (18, 12) and np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))
        assert all(len(indexed[d,c,o]['F1Runs']) == 30 for d,c in blocks for o in CONFIG_ORDER)
        ranks = rankdata(-x, axis=1, method='average')
        assert np.allclose(ranks.sum(axis=1), 78)
        mean_ranks = ranks.mean(axis=0)
        ranked = np.argsort(mean_ranks, kind='stable')
        friedman = friedmanchisquare(*x.T)
        tie_sum = sum(sum(n**3-n for n in np.unique(row, return_counts=True)[1]) for row in x)
        tie_correction = 1-tie_sum/(18*(12**3-12))
        independent_q = (12*18/(12*13)*np.sum(mean_ranks**2) - 3*18*13)/tie_correction
        assert np.isclose(friedman.statistic, independent_q, rtol=1e-12)
        summary = pd.DataFrame({'Algorithm': ORDER, 'Mean_F1': x.mean(axis=0), 'Std_F1': x.std(axis=0, ddof=1),
                                'Median_F1': np.median(x, axis=0), 'Mean_Rank': mean_ranks}).iloc[ranked].reset_index(drop=True)
        rows = []
        for i,j in itertools.combinations(range(12), 2):
            d = x[:, i] - x[:, j]  # Exactly the same ordered block keys on both sides.
            assert len(d) == 18
            test = wilcoxon(d, alternative='two-sided', zero_method='wilcox', method='exact', correction=False)
            w_check, p_check = exact_signed_rank_check(d)
            assert test.statistic == w_check and np.isclose(test.pvalue, p_check, rtol=0, atol=1e-15)
            rows.append([ORDER[i], ORDER[j], float(test.statistic), float(test.pvalue)])
        pairs = pd.DataFrame(rows, columns=['Algorithm_A', 'Algorithm_B', 'Wilcoxon_statistic', 'Raw_p'])
        assert len(pairs) == 66
        raw = pairs.Raw_p.to_numpy()
        sort = np.argsort(raw, kind='stable')
        adjusted = np.empty(66)
        adjusted[sort] = np.minimum(1., np.maximum.accumulate(raw[sort] * np.arange(66, 0, -1)))
        # Independently check the defining Holm step-down maximum for every pair.
        for k, idx in enumerate(sort):
            assert np.isclose(adjusted[idx], min(1., max((66-j)*raw[sort[j]] for j in range(k+1))), rtol=0, atol=1e-15)
        pairs['Holm_adjusted_p'] = adjusted
        pairs['Significant_0.05'] = adjusted < .05
        matrix = np.ones((12, 12))
        for row in pairs.itertuples():
            i,j = ORDER.index(row.Algorithm_A), ORDER.index(row.Algorithm_B)
            matrix[i,j] = matrix[j,i] = row.Holm_adjusted_p
        assert np.array_equal(matrix, matrix.T)
        FIG.mkdir(exist_ok=True)
        RES.mkdir(exist_ok=True)
        summary.to_csv(RES / 'statistical_summary.csv', index=False, float_format='%.17g')
        pairs.to_csv(RES / 'pairwise_wilcoxon_holm.csv', index=False, float_format='%.17g')
        for name, expected in [('statistical_summary.csv', summary), ('pairwise_wilcoxon_holm.csv', pairs)]:
            pd.testing.assert_frame_equal(pd.read_csv(RES/name), expected, check_exact=False, check_dtype=False, rtol=1e-14, atol=1e-15)
        for stem, figure_id, fig in zip(STEMS, IDS, figures(x, mean_ranks, ranked, pairs, matrix)):
            try:
                m._save_statistical_figure(fig, FIG / (stem+'.png'), statistical_figure=figure_id, bbox_inches='tight')
                with Image.open(FIG/(stem+'.png')) as png:
                    assert png.format == 'PNG' and all(abs(v-600)<.1 for v in png.info['dpi'])
                assert not (FIG/(stem+'.pdf')).exists()
            finally:
                plt.close(fig)
            print('Created', stem, 'PNG (600 dpi)', flush=True)
        assert snapshot(ROOT, FIG, RES) == before
        assert guard['optimization_calls'] == 0
        report = ['EXP627 FULL — matched-block F1 statistical analysis', '='*58,
                  'Source: Results/EXP627/full/cache/; raw F1Runs only.',
                  'Hierarchy: arithmetic mean of the 30 independent cached F1 runs within each dataset-classifier-algorithm combination.',
                  'Matched blocks: 18 = 6 code smells × 3 classifiers. Algorithms: 12. Observations per algorithm: exactly 18.',
                  'No raw run flattening, missing blocks, imputation, or experiment execution.',
                  'Mean_F1, Median_F1 and sample Std_F1 (ddof=1) describe the 18 block means; Std_F1 is NOT within-block run variability.',
                  'Higher F1 is better. Per-block ranks use descending F1, average ranks for ties, and rank 1 as best.',
                  f'Friedman statistic (tie-corrected): {friedman.statistic:.17g}',
                  f'Friedman p-value (chi-square approximation, df=11): {friedman.pvalue:.17g}',
                  'Friedman omnibus result: '+('reject equal performance across algorithms at alpha=0.05.' if friedman.pvalue < .05 else 'do not reject equal performance at alpha=0.05.'),
                  'The omnibus result alone does not establish superiority of any individual algorithm.',
                  'Pairwise tests: exact two-sided Wilcoxon signed-rank on 18 paired block differences; zero_method=wilcox; no continuity correction.',
                  'All 66 difference vectors were checked: no zero differences and no tied absolute differences.',
                  'Every exact Wilcoxon statistic and p-value was independently checked by subset-sum enumeration of the signed-rank null distribution.',
                  'Multiplicity: one Holm step-down Bonferroni family containing ALL 66 unordered algorithm pairs (not only the 11 DSA-DE pairs).',
                  'Adjusted p_(i) = min(1, max_{j<=i} (66-j+1) p_(j)); significance uses p_adj < 0.05.',
                  'Stars: * p_adj < 0.05; ** < 0.01; *** < 0.001; ns otherwise.', '', 'AVERAGE RANKS (best to worst)']
        report += [f'{ORDER[i]}: {mean_ranks[i]:.12g}' for i in ranked]
        report += ['', 'DSA-DE PAIRWISE COMPARISONS (same 18 matched blocks; Holm family of 66)']
        for row in pairs[pairs.Algorithm_A == 'DSA-DE'].itertuples():
            j = ORDER.index(row.Algorithm_B)
            delta = float(np.mean(x[:,0]-x[:,j]))
            conclusion = ('Statistically significant difference; DSA-DE has '+('higher' if delta > 0 else 'lower')+' observed mean F1.'
                          if row.Holm_adjusted_p < .05 else 'No statistically significant difference after Holm correction; no superiority claim.')
            report.append(f'DSA-DE vs {row.Algorithm_B}: W={row.Wilcoxon_statistic:.12g}, raw p={row.Raw_p:.17g}, adjusted p={row.Holm_adjusted_p:.17g}, mean paired difference={delta:.12g}. {conclusion}')
        report += ['', 'INTERPRETATION AND SCOPE',
                   'Ranks summarize relative performance, not effect magnitude or pairwise significance. Non-significance does not prove equivalence.',
                   'Inference uses the requested dataset-classifier block design. Classifiers on the same dataset share data; these are not 18 independent datasets.',
                   'Wilcoxon interpretation assumes a symmetric distribution of block-level paired differences. Results concern the specified blocks and aggregation hierarchy.',
                   'Figure 1 shows average ranks only; it is not a critical-difference diagram and contains no inferred non-significance groups.',
                   'Figures 1 and 4 and the summary CSV are ordered by ascending average rank (stable configured order breaks any rank ties).',
                   'Figure 2 uses configured order excluding DSA-DE; Figure 3 uses configured order on both axes and only the lower triangle.',
                   'Figure 3 uses two discrete fills (significant / not significant); annotations are rounded for display, full precision is in the CSV.',
                   '', 'MATCHED BLOCK MEANS (auditable input matrix; 30 runs per cell)', 'Dataset,Classifier,'+','.join(ORDER)]
        report += [d+','+c+','+','.join(format(v,'.17g') for v in x[k]) for k,(d,c) in enumerate(blocks)]
        report += ['', 'VALIDATION', 'Exactly 18 finite block-level F1 observations per algorithm: PASS.',
                   'Every Friedman and Wilcoxon input uses identical ordered dataset-classifier keys: PASS.',
                   'Mean ranks independently reconcile with the Friedman statistic: PASS.',
                   'Holm adjustment independently checked for all 66 pairs: PASS.',
                   'CSV serialization checked against computed values: PASS.',
                   'Four PNGs checked for 600 dpi; no PDFs generated: PASS.',
                   'Optimization calls = 0. No experiment runs, optimizer calls, cache writes, or scientific configuration changes.',
                   f'Original caches/results/figures and previous reporting outputs unchanged: SHA-256 verified for {len(before)} files.',
                   'Protected-file manifest SHA-256: '+hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
                   'Only the specified statistical outputs were written within the project.',
                   f'Software: NumPy {np.__version__}, SciPy {scipy.__version__}, pandas {pd.__version__}, Matplotlib {matplotlib.__version__}.',
                   '', 'SOURCE CACHE SHA-256']
        report += [f'{source}: {before[source]}' for source in sources]
        report += ['', 'METHOD REFERENCES',
                   'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html',
                   'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html',
                   'https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html']
        (RES/'statistical_report.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
        assert {p.name for p in FIG.iterdir()} == {f'{s}.png' for s in STEMS}
        assert {p.name for p in RES.iterdir()} == EXPECTED_RES
    assert snapshot(ROOT, FIG, RES) == before
    print(f'Friedman: statistic={friedman.statistic:.12g}, p={friedman.pvalue:.12g}')
    print(summary.to_string(index=False))
    print(pairs[pairs.Algorithm_A=='DSA-DE'].to_string(index=False))
    print(f'VALIDATED: 18 matched blocks per algorithm; 66 exact Wilcoxon tests; 4 PNG; 0 PDF; 2 CSV + 1 report; {len(before)} original files unchanged; optimization calls = 0.')


if __name__ == "__main__":
    run(framework().parse_args())
