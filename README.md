# Diversity_based_Self_Adaptive_Differential_Evolution
The Diversity Self-Adaptive Differential Evolution (DSA-DE) algorithm implements an examination of the diversity as a self-adaptive process for configuring its internal parameters.

## Cache-only reporting

A plain IDE Run of `main_best.py` continues EXP629 using compatible current
EXP629 checkpoints first, then imports missing compatible rows from read-only
EXP627. Its defaults are `EXP_ID=629`, `REUSE_CACHE_FROM_EXP_ID=627`,
`REUSE_CACHE=True`, `COMPUTE_DEVICE="gpu"`, `RF_BACKEND_POLICY="sklearn"`, and
`RF_CPU_FALLBACK=True`. RF stays compatible with historical sklearn results even
when cuML is installed; custom optimizer kernels can still use GPU. Scientific
run counts, budgets, datasets, optimizer parameters and seeds are unchanged.
EXP630 is not the default. `FULL_REPLICA_REPORT_ONLY=False` continues the real
experiment in `EXP629/full`; incomplete combinations may execute new runs.

Use `--report-only` after the selected EXP629 caches are complete to create the
next presentation version without optimization:

```bash
python -B main_best.py --report-only --exp-id 629 --experiment-mode full
```

The console prints flushed startup, stage, and per-PNG progress with elapsed
times. Full 600-dpi reporting can take several minutes; publication occurs only
after every artifact is validated. To run the complete workload in a temporary
destination while reading the original caches directly, add
`--report-output-root /tmp/exp627-report-check`. This changes only where the next
report version is created; `--output-root` still selects the source tree.

`--full-replica-report-only` and `--full-rep1-report-only` remain aliases of
`--report-only`; neither fixes the destination at `full_rep1`. The plural
`--experiment-modes` option supports multiple modes. All selected caches must
pass validation before any version is allocated. Non-default scientific settings
must match the selected cache identity; reporting never repairs incompatible or
incomplete caches, uses progress checkpoints, migrates caches from another EXP,
or creates replacement experiments. `--reuse-cache-from-exp-id` does not select
report sources: cache-only sources always belong to `--exp-id`.

The next version is `max(N) + 1` across **both**
`Figures/EXP<ID>/full_repN` and `Results/EXP<ID>/full_repN`, starting at 1.
A report version is a new presentation of existing runs, not an independent
scientific repetition. Older versions are never rewritten. Unpaired versions,
conflicting manifest identities, symlinks, and unsafe destinations cause an error.
A temporary allocation lock prevents competing reporters. Both trees are staged,
validated, and published with atomic no-replace directory renames. If the second
publication fails, the first is rolled back. An interrupted process may leave a
lock or unpaired version; reporting stops for manual inspection and never deletes
or overwrites these automatically. Staging directories do not consume version
numbers. Atomic publication currently requires Linux `renameat2`.

Each version contains:

- Publication PNGs for every available classifier/metric and stored convergence
  curve, plus statistical PNGs; all at 600 dpi. No PDFs are generated.
- `Global_Results_EXP<ID>.xlsx`, `Statistical_Results_EXP<ID>.xlsx`, and
  `Paper_Tables_EXP<ID>.xlsx`.
- The existing `<Mode>_Friedman_Analysis_EXP<ID>.xlsx` for FULL/ABLATION when
  cached fitness observations are available.
- `statistics/statistical_summary.csv`, `pairwise_wilcoxon_holm.csv`,
  `matched_block_means.csv`, and `statistical_report.txt`.
- `validation.json`, recording the selected EXP, version, cache signatures and
  hashes, datasets, algorithms/variants, classifiers, completed counts, metrics,
  destinations, output hashes, and explicit reasons for unavailable outputs.

When several modes/studies are selected, each has its own subdirectory within the
same version; sensitivity studies also include the parameter name. In FULL and
ABLATION, the original Friedman workbook retains its original fitness aggregation
and conditional post-hoc definition. The separate matched-block analysis uses
cached F1 run means per dataset/classifier (or an available metric if F1 is absent),
average ranks for ties, paired two-sided Wilcoxon tests, and one Holm family over
all algorithm pairs. Missing blocks are explicitly listed, never imputed.
Insufficient blocks and all-tied Friedman inputs have no invented test result.
Wilcoxon zeros/ties and exact versus approximate methods are recorded per pair.

## Generic publication figures

New FULL exports use the same layout for normal runs and cache-only reports:

```text
Figures/EXP629/full/             # completed experiment exports
  01_...png through 09_...png    # main publication figures
  individual/                   # additional generic and single-smell views
  statistics/                   # average ranks, Holm/posthoc, p-value heatmaps
Figures/EXP629/full_repN/        # new cache-only presentation version
  01_...png through 09_...png    # same main publication names/renderers
  individual/
  statistics/
Results/EXP629/full/             # existing experiment workbooks, summary CSV, cache/
Results/EXP629/full_repN/        # existing versioned report workbook convention
  statistics/                   # statistical CSV/text outputs, as before
```

Root holds nine main figures: all-classifier overview (01), radar grid (02),
dataset feature/runtime grid (03), dataset accuracy boxplots (04), convergence
grid (05), F1 heatmap (06), recall violin (07), global accuracy distribution (08),
and global feature/runtime tradeoff (09). Both FULL paths share one renderer.
`PLOT_GLOBAL_ESTIMATOR="knn"` (or `--plot-global-estimator`) selects the main
classifier; the first observed classifier is used if that choice is absent.
Classifier-specific root names reflect the actual classifier. Plot settings do
not affect scientific/cache identity. Main distributions retain observed cached
runs; generic confidence/distribution views retain their existing dataset means.

Extra multi-dataset views use names such as `generic_convergence_knn.png`
and `generic_features_runtime_svm.png`.
Single-smell panels always name their smell, for example
`convergence_DataClass_knn.png`, `radar_GodClass_svm.png`,
and `features_runtime_SwitchStatements_knn.png`.
Metric tokens are accuracy, precision, recall, f1, fitness, features, and runtime;
opaque `cN/mN` indices are no longer exported. Unsafe filename characters are
percent-encoded without changing dataset keys. All these extra views stay in
`individual/`, including additional generic boxplots/violins. FULL generates
only the main metric heatmap (06); no extra heatmaps are generated in `individual/`.

The eight historical report filenames (`grafica_resumen_general`,
`radar_6smells_grid_svm`, `ranking_precision`, `boxplot_accuracy_general`,
`violin_recall`, `heatmap_f1score`, `convergence_curve`, and
`features_runtime_per_optimizer`) are not renamed on disk. New exports use the
numbered main set and readable individual names instead.
Average-rank, reference-comparison, Holm heatmap and matched-block distribution
PNGs are siblings of `individual/`, rather than inside it. Report manifests list
root, individual and statistical figures separately. Existing figures are never
moved or deleted by this layout change; older root files can remain in place.
Workbooks retain their existing Results destinations. The plot/report alias
`MaCRO-DE-t` → `DSA-DE` does not change cache labels or scientific identity.
Its visual identity uses the established primary blue, solid line/circle marker,
strong black curve/heatmap outlines, and emphasized bars. Numeric bar labels,
runtime hatching, distribution mean/median markers and cell annotations remain.

Every selected experiment uses the same reporting implementation. There are no
experiment-specific modules, presets, signatures, layouts, or classifier choices.
All eight publication figure types are available for each actual classifier:
metric summary, radar panels, metric heatmaps, precision with Student-t 95% CIs,
accuracy boxplots, recall violins, stored convergence curves, and features/runtime
tradeoffs. Dataset and optimizer counts determine panel sizes, ticks, legends,
observations, confidence-interval degrees of freedom, and colors. Missing inputs,
insufficient CI observations, or constant violin samples receive explicit reasons
in the manifest; observations are never fabricated. Radar feature efficiency uses
`1 - features / max(max_features, 1)` independently within each dataset.

Statistical figures show average ranks, comparisons with the first configured
algorithm, the full Holm-adjusted matrix, and matched-block distributions. The
Holm correction includes all algorithm pairs, regardless of the reference shown.
Paper Tables have one shared dynamic layout for all experiments.

EXP627 is used only as a read-only regression reference in tests. Its saved means,
statistics, and workbook values are compared with generic calculations without
regenerating historical figures or modifying historical files.

## Paper Tables presentation and calculations

All newly generated Paper Tables use white/default backgrounds, normal black
Calibri text, visible Excel gridlines, no decorative borders or fills, wrapped
headings, and content-sized columns and rows. Their full used range is printable
at normal scale across multiple pages, without a fixed A3 or one-page layout.
The actual grouping and merged labels are retained; no columns or values are
removed to make tables narrower. Workbooks are read back to check every numeric
cell and presentation constraints before publication.

Generic workbooks include `Overall` and one dataset sheet per available
classifier; there is no fixed sheet count. Available Accuracy, F1, Precision,
Recall, Fitness, Features, and Time arrays are selected. Accuracy is converted
from percent to 0–1; other units are unchanged. Overall Best/Worst/Mean/Std are
calculated **after** averaging matching run positions equally across datasets.
Std uses the existing sample definition (`ddof=1`, zero for one finite run).
Per-dataset tables preserve their original run arrays and metric directions.
Missing combinations, partial metric grids, empty arrays and ambiguous identities
fail safely. A metric absent from the entire cache is explicitly marked unavailable.

Normal FULL continues to add the generic Paper Tables workbook through
`reporting.paper_tables.export_paper_tables`. Global Results, Statistical Results,
and Friedman exporters and their presentation are unchanged. Saved historical
workbooks are never restyled by cache-only reporting.

## Normal scientific execution

An explicit mode without a report-only flag retains the existing normal dispatch:

```bash
python -B main_best.py --exp-id 627 --experiment-mode full
```

This uses the original `full/` directories and normal cache-reuse and scientific
execution rules. It is **not** a report-only command and is never redirected into
`full_repN`. All other existing experiment modes remain available.

## Optimizer-local FULL cache reuse

FULL saves each optimizer/transfer-function checkpoint under its local signature:
`Results/EXP<ID>/full/cache/EXP<ID>_<dataset>_<classifier>_<local-signature>_{results,progress}.pkl`.
Each local file contains one row with a stable optimizer/TF/classifier label and
its own `CacheIdentity`. Its identity
includes mode, dataset source/name, classifier, transfer function, canonical
optimizer, runs/epochs/population size, test split, random state, seed base,
fitness definition and weights, and only that optimizer's scientific constructor
parameters and implementation revision. The optimizer comparison list, unrelated
optimizer parameters, device, GPU resource settings, and worker count are excluded.
The optimizer-local signature is available through `build_cache_signature(args,
optimizer_name, dataset_name, estimator, transfer_function)`.
Shared comparison files with a common-settings signature are also retained for
existing reporting. Changing one optimizer's scientific settings creates new
local files while preserving its previous scientific configurations.

Normal execution searches `Results/EXP<ID>/full/cache/EXP<ID>_<dataset>_<classifier>_*_{results,progress}.pkl`
in the current EXP and the configured `--reuse-cache-from-exp-id` source. Each row
must independently pass scientific identity and run-array validation. The longest
compatible prefix is selected per optimizer and transfer function. Imported runs
are saved under optimizer-local signatures only in the current EXP, together with
the reporting comparison files; source caches are read-only. Filename signatures
need not match the current signature when complete scientific metadata matches.
Logs identify
`CACHE IMPORTED`, `CACHE HIT`, `CACHE MISS`, and `CACHE SOURCE INCOMPATIBLE`.

Historical FULL rows without `CacheIdentity` require a verified reconstruction of
their complete legacy filename digest, including the source comparison list and
any encoded implementation revisions. Because that schema omitted dataset source
and FULL fitness weights, legacy imports are limited to the historical code-smell
source and default fitness weights. Unknown signatures and conflicting metadata
are rejected. Historical reporting retains exact legacy lookup and never imports.
Interrupted shared files may contain only an ordered prefix of the configured
comparison. That prefix is accepted only when the full configured legacy digest
matches the filename. Lookup prioritizes current optimizer-local files, then
current shared/legacy files, then the configured source EXP. Compatible current
legacy rows are materialized as local checkpoints before testing completion.
FULL writes append immutable snapshots instead of replacing existing cache files.

RF cache hits and imports never construct an RF estimator. Explicit backend
policies do not import cuML during lookup. For missing RF runs, GPU mode uses
cuML when `RF_BACKEND_POLICY="auto"` is selected. The EXP629 default is
`RF_BACKEND_POLICY="sklearn"` to retain historical RF cache compatibility.
Set `RF_CPU_FALLBACK = True` or pass `--rf-cpu-fallback` to allow
the historical sklearn/MAFESE RF path when cuML cannot be imported. Optimizer
kernels remain on GPU; only RF fitness/evaluation uses CPU. The log says
`cuML unavailable; using sklearn RF on CPU while optimizer backend remains GPU.`
With fallback enabled, pending-run validation skips the mandatory cuML preflight.
Estimator construction returns a sklearn RF instance when cuML is unavailable;
evaluation recovery uses that same backend selection. Both the CLI flag and the
configured default are preserved in mode/worker arguments.
Fallback does not hide optimizer, fitting, or cuML constructor errors.

RF backend is scientific identity. Configure `RF_BACKEND_POLICY` or pass
`--rf-backend-policy auto|sklearn|cuml`:

| Policy | GPU mode | CPU/hybrid mode |
| --- | --- | --- |
| `auto` | cuML; sklearn if cuML cannot be imported and fallback is enabled | sklearn |
| `sklearn` | sklearn RF with GPU optimizer kernels | sklearn |
| `cuml` | cuML required for new RF runs; CPU fallback is forbidden | sklearn |

The choice is resolved once and shared by every optimizer, mode variant, and
worker. FULL optimizer-local signatures include `CacheIdentity.rf_backend`.
RF shared files and other modes use a backend suffix plus checked backend
metadata. KNN/SVM identities and filenames are unchanged. Recorded sklearn and
cuML RF rows are never mutually compatible. Workers cannot silently substitute
sklearn after a cuML cache identity has been selected.

Legacy RF rows without provenance are eligible only for sklearn/unknown reuse,
after the complete existing identity or legacy filename digest is verified.
Imports log `rf_backend=sklearn | provenance=legacy_unknown_assumed_sklearn`.
For a cuML selection these rows log `CACHE SOURCE INCOMPATIBLE` with
`expected=cuml, actual=sklearn/unknown`. There is no option to bypass this check.
Older rows with an explicit consistent cuML backend can be recovered after all
other scientific fields pass. Mixed, conflicting, or incomplete backend traces
are rejected. Existing files remain intact; FULL migration writes new snapshots.

## Optional cuML RF GPU setup

The validated stable setup is RAPIDS/cuML 26.08 with CUDA 13, CuPy 14.2.0,
and Python 3.13.16 on Ubuntu 26.04. RAPIDS 26.08 supports Python 3.11–3.14
and CUDA 13.0–13.3 on Linux with glibc 2.28+. CUDA 13 requires an NVIDIA
driver at least 580.65.06 and a Turing/SM75 or newer GPU. Check the
[official release matrix](https://docs.rapids.ai/platform-support/#rapids-26-08)
and [installation guide](https://docs.rapids.ai/install/) on each future PC;
`nvidia-smi` reports driver capability, not an installed toolkit.

`requirements-gpu.txt` contains optional RF/CuPy dependencies with validated
RAPIDS patch versions. `requirements-linux-gpu.txt` includes both the base
requirements and this GPU file. CPU requirements never include cuML.
Use a separate venv on another PC if its existing dependencies would need major
changes. On an already compatible project venv, inspect the plan before installing:

```bash
.venv/bin/python -m pip install --dry-run --only-binary=:all: -r requirements-gpu.txt
.venv/bin/python -m pip install --only-binary=:all: -r requirements-gpu.txt
.venv/bin/python -c "from cuml.ensemble import RandomForestClassifier; print('cuML RF OK')"
RUN_CUML_RF_SMOKE=1 .venv/bin/python -B -m unittest tests.test_cuml_rf_smoke -v
```

The smoke check uses 128 synthetic rows, fits small native cuML forests, and
constructs no optimizers or experiment outputs. Prebuilt wheels do not require
`nvcc` on PATH; their matching CUDA/NVRTC libraries are resolved as dependencies.
These pins are CUDA 13 specific; choose the official matching package family
for a CUDA 12 environment rather than mixing wheel families.

The current EXP629 default uses GPU optimizer kernels with sklearn RF. For future
experiments selecting `--rf-backend-policy auto`, native cuML RF is preferred
even if `RF_CPU_FALLBACK=True`; the fallback
flag only permits sklearn when cuML cannot be imported under `auto`. Use
`--rf-backend-policy cuml` to require GPU RF and its distinct compatible caches.
CPU mode always uses sklearn RF. sklearn RF keeps historical
sklearn defaults (`n_jobs=None`); independent CPU runs can still run in parallel.

Native RF training and prediction execute GPU kernels. Dataset preparation,
host/device transfers, Python orchestration and metric calculation still involve
CPU work. cuML uses quantile splits, which can produce different results from
sklearn's exact splits despite matching common parameters; GPU RF must not be
described as numerically identical to historical sklearn RF.

GPU RF run groups execute one run at a time because forest/RMM allocations are
outside the optimizer CuPy memory limit and VRAM estimate. Each forest uses
`n_streams=1` for reproducible seeded training, while tree/node calculations
remain parallel on GPU. This prevents multiple forest workspaces from competing
on a 6 GB GPU; a single oversized forest can still exhaust VRAM. KNN/SVM retain
their existing worker policy. Scientific forest defaults (100 trees, unlimited
depth in cuML 26.08, `max_features="sqrt"`, 128 bins) are retained.

New RF observations record backend, library version and estimator parameters in
`RFExecutionRuns`. Resumed legacy observations lacking this trace remain marked
`unknown` and can continue only with sklearn. Mixed legacy rows are rejected;
new runs must match the selected scientific backend. Historical sklearn results
remain reusable under the sklearn selection. Reusing those results does not
recompute them on GPU or convert their provenance into cuML results.

With EXP629 configured to read EXP627, unchanged DE, JADE, SHADE, PSO, WOA, HHO,
GOA, SA, BRO, RUN, and FOX rows can be reused while missing MaCRO-DE-t runs execute.
Adding/removing comparison optimizers does not change FULL scientific identity.
Other modes retain their historical scientific signatures with the added RF-only
backend filename suffix and compatibility check.

Focused validation (mocked optimization and exports, temporary output only):

```bash
python -B -m unittest tests.test_full_cache_reuse
```

## Inexpensive reporting validation

Main FULL/full_repN publication figures have three presentation-only settings in
`main_best.py`: `PLOT_GLOBAL_ESTIMATOR = "knn"`,
`PLOT_GLOBAL_METRIC = "accuracy"`, and `FIGURE_LANGUAGE = "en"`. The equivalent
CLI options are `--plot-global-estimator {knn,svm,rf}`,
`--plot-global-metric {accuracy,f1,precision,recall}`, and
`--figure-language {en,es}`. For example, render a new cache-only report using
stored SVM recall results with Spanish root-figure text:

```bash
.venv/bin/python main_best.py --report-only --exp-id 629 --experiment-mode full \
  --plot-global-estimator svm --plot-global-metric recall --figure-language es
```

The selected classifier must have stored results. Figure 01 remains the
all-classifier overview. Figures 02–05 use the selected classifier; 04 also uses
the selected metric. Figure 06 is the selected metric's **dataset run-mean
heatmap**. Figures 07 (violin) and 08 (boxplot) use **all cached runs pooled across
datasets for the selected classifier**, not dataset means or pooled classifiers.
Accuracy is displayed on a 0–1 scale using the existing percentage conversion;
F1, precision and recall retain their stored 0–1 units. Figure 09 also uses the
selected classifier's features/runtime data. Distribution views retain mean and
median indicators without individual scatter points.

Root filenames use English tokens, including
`04_boxplot_<metric>_por_dataset_<classifier>.png`,
`06_heatmap_<metric>_<classifier>.png`,
`07_violin_<metric>_<classifier>.png`, and
`08_global_<metric>_distribution.png`. Figure 08's title identifies its selected
classifier. Spanish changes only visible root-figure text (recall is
“Sensibilidad”); datasets and algorithm names are preserved. Individual and
statistical destinations and names remain unchanged. Presentation selections are
recorded in new report manifests, never in cache signatures or scientific
identities. Existing report directories are not rewritten by report-only runs.

Convergence panels include an inset of the final quarter of stored iterations,
titled “Final stage” / “Etapa final”, with all algorithms and palette-colored
curves. Insets stay in the lower-right corner without changing main-axis limits.
Their vertical zoom uses the lowest final-fitness band (the bottom 10% of the
final-value range, capped at the lower half), so high curves need not be visible,
including DSA-DE when it converges high. A missing,
invalid, or shorter-than-three-iteration curve prevents that panel's inset and is
reported explicitly in `skipped_outputs`; available main curves remain visible.

```bash
python -B -m unittest tests.test_dispatch tests.test_full_rep1_validation tests.test_replica_safety tests.test_paper_tables tests.test_generic_reporting tests.test_generic_figures tests.test_reporting_png tests.test_mode_datasets tests.test_statistical_transfer_exports tests.test_reporting_reference
```

These tests use temporary outputs, small synthetic caches, mocked scientific
execution, and tiny PNGs. Saved transfer caches, when present, are read-only inputs
for existing exporter regressions. No full-resolution historical publication
figures or optimization runs are generated by this validation suite.
