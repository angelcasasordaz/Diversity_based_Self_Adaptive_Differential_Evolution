# Diversity_based_Self_Adaptive_Differential_Evolution
The Diversity Self-Adaptive Differential Evolution (DSA-DE) algorithm implements an examination of the diversity as a self-adaptive process for configuring its internal parameters.

## Cache-only reporting

A plain IDE Run of `main_best.py` uses the configured `EXP_ID` and experiment
modes, reads completed caches, and creates the next report version. It never
executes optimization or changes the experiment ID. For an explicit selection:

```bash
python -B main_best.py --report-only --exp-id 627 --experiment-mode full
```

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

## Inexpensive reporting validation

```bash
python -B -m unittest tests.test_dispatch tests.test_full_rep1_validation tests.test_replica_safety tests.test_paper_tables tests.test_generic_reporting tests.test_generic_figures tests.test_reporting_png tests.test_mode_datasets tests.test_statistical_transfer_exports tests.test_reporting_reference
```

These tests use temporary outputs, small synthetic caches, mocked scientific
execution, and tiny PNGs. Saved transfer caches, when present, are read-only inputs
for existing exporter regressions. No full-resolution historical publication
figures or optimization runs are generated by this validation suite.
