# FULL replica layout and style audit

Scope: the currently selected EXP627, with six cached datasets, twelve methods,
three classifiers, and seven metrics. Historical `full`, `full_rep1`, and
`full_rep2` files remain in place. This is cache-only reporting, not a new
scientific repetition or an optimization run.

## Existing figure inventory

`Figures/EXP627/full/` contains nine PNGs:

```text
01_resultados_clasificador_todos_datasets.png
02_radar_por_dataset_knn.png
03_features_runtime_por_dataset_knn.png
04_boxplot_accuracy_por_dataset_knn.png
05_convergence_por_dataset_knn.png
06_heatmap_f1_knn.png
07_violin_recall_knn.png
08_global_accuracy_distribution.png
09_global_features_runtime_tradeoff.png
```

The numbered renderer is the visual source: identity-based colors (DSA-DE blue
`#0072B2`), black method outlines, stable line/marker combinations, metric header
boxes, white backgrounds, full axes frames, and light grids. Its legacy `_knn`
filenames do not themselves identify the selected classifier; the renderer
accepts `estimator_filter` and currently defaults to SVM.

`full_rep1/` contains the eight base identities listed below and four PNGs under
`statistics/`: `stat_fig1_average_rank.png`, `stat_fig2_dsade_vs_others.png`,
`stat_fig3_holm_heatmap.png`, and `stat_fig4_f1_boxplot.png`. Its validation
manifest specifies SVM for the single-classifier base views and all classifiers
for the summary. It used burgundy DSA-DE data marks and some burgundy text.

`full_rep2/` contains 42 root-level `generic_*.png` files: seven non-heatmap
types × three classifiers, plus seven heatmaps × three classifiers. The exact
names are those in the `individual/` manifest below. Its `statistics/` contains
the same four `generic_*.png` names now routed under `individual/statistics/`.
It used a separate burgundy-first positional palette, different panel layout,
hidden top/right spines, and lighter distribution fills.

## Exact full_rep3 figure manifest

Root: `Figures/EXP627/full_rep3/`, **eight PNGs only**:

| Filename | Content |
| --- | --- |
| `grafica_resumen_general.png` | KNN/SVM/RF × Accuracy/Precision/Recall/F1; means across the six datasets |
| `radar_6smells_grid_svm.png` | Six SVM dataset panels; four classification metrics plus feature efficiency |
| `ranking_precision.png` | SVM precision means and Student-t 95% intervals across dataset means |
| `boxplot_accuracy_general.png` | SVM accuracy distribution across the six cached dataset means |
| `heatmap_f1score.png` | SVM F1 method × dataset matrix |
| `violin_recall.png` | SVM recall distribution across dataset means, with observations/mean/median |
| `convergence_curve.png` | Six SVM panels using stored mean fitness curves |
| `features_runtime_per_optimizer.png` | SVM mean selected features and runtime by method |

`individual/`: **42 PNGs**, with these exact expansions:

```text
generic_summary_c1.png
generic_summary_c2.png
generic_summary_c3.png
generic_radar_c1.png
generic_radar_c2.png
generic_radar_c3.png
generic_precision_c1.png
generic_precision_c2.png
generic_precision_c3.png
generic_accuracy_boxplot_c1.png
generic_accuracy_boxplot_c2.png
generic_accuracy_boxplot_c3.png
generic_recall_violin_c1.png
generic_recall_violin_c2.png
generic_recall_violin_c3.png
generic_convergence_c1.png
generic_convergence_c2.png
generic_convergence_c3.png
generic_features_runtime_c1.png
generic_features_runtime_c2.png
generic_features_runtime_c3.png
generic_heatmap_c1_m1.png
generic_heatmap_c1_m2.png
generic_heatmap_c1_m3.png
generic_heatmap_c1_m4.png
generic_heatmap_c1_m5.png
generic_heatmap_c1_m6.png
generic_heatmap_c1_m7.png
generic_heatmap_c2_m1.png
generic_heatmap_c2_m2.png
generic_heatmap_c2_m3.png
generic_heatmap_c2_m4.png
generic_heatmap_c2_m5.png
generic_heatmap_c2_m6.png
generic_heatmap_c2_m7.png
generic_heatmap_c3_m1.png
generic_heatmap_c3_m2.png
generic_heatmap_c3_m3.png
generic_heatmap_c3_m4.png
generic_heatmap_c3_m5.png
generic_heatmap_c3_m6.png
generic_heatmap_c3_m7.png
```

Classifier mapping: `c1=knn`, `c2=svm`, `c3=rf`.
Metric mapping: `m1=Accuracy`, `m2=F1`, `m3=Precision`, `m4=Recall`,
`m5=Fitness`, `m6=Selected features`, `m7=Runtime`.

`individual/statistics/`: **four PNGs**:

```text
generic_average_rank.png
generic_reference_comparisons.png
generic_holm_heatmap.png
generic_block_distribution.png
```

Total: **54 PNGs**, all at 600 dpi; no PDFs. Statistical calculations and tables
stay under `Results/EXP627/full_rep3/statistics/`. The append-only report allocator
chooses version 3 because matching versions 1 and 2 already exist. Later reports
use the same layout in the next available version; no existing version is
overwritten. For other suites the radar filename reflects the actual dataset
count/classifier, and unavailable base metrics are explicitly recorded as
skipped rather than fabricated.

## One visual source, preserved scientific data

`full_plot_style.py` holds the existing numbered-FULL palette, line styles,
panel-grid logic, figure dimensions, metric headers, bar formatting, and neutral
text policy. The numbered renderer and replica renderer import that same source.
The eight base figures reuse the individual figure builders, so there is no
separate replica-1 or replica-2 visual theme.

DSA-DE's plotted bars, curves, points, and distributions retain blue method
identity and black outlines where applicable. Axis labels (including method
names), titles, legend text, and ordinary annotations are black. White heatmap
annotations remain for contrast on dark cells. Existing heatmap color maps and
the summary's red mean guide retain their distinct data meaning.

Unchanged: cached run arrays, run-to-dataset aggregation, dataset means,
Student-t intervals, feature-efficiency calculations, stored convergence curves,
matched-block statistics, Wilcoxon/Holm calculations, exported numerical tables,
datasets, experiment settings, and optimizer behavior. Generic distributions
continue to show dataset means; they are not replaced with the numbered FULL
renderer’s run-level observations merely to copy its appearance.

`Results/EXP627/full_rep3/validation.json` records the exact generated paths,
style identity, base classifier, source/output hashes, zero optimization calls,
and verification that historical files are unchanged.

## Focused validation

`tests/test_full_rep3_layout.py` validates the exact 8/42/4 split, shared style
objects and palette identity independent of selection order, blue data marks
with neutral text, reuse of numbered-summary drawing logic, unchanged plotted
values, scoped statistical styles, and actual version-3 orchestration without
optimization. Existing reporting tests cover saved scientific values, workbook
cells, statistical results, PNG/DPI policy, report-only dispatch, historical
hashes, and atomic append-only publication.

Completed validation: the 53-test reporting regression suite passed, followed
by 15 final layout/style/PNG/visualization checks (the two runs overlap). The
actual cache-only report published all 54 PNGs with zero optimizer calls and
verified historical hashes. Every new workbook cell equals its `full_rep2`
counterpart, all three statistical CSVs are byte-identical, and every output
hash matches `validation.json`. Representative summary and convergence PNGs
were also visually inspected. `git diff --check` passed.
