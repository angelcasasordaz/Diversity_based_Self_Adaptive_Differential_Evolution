# Diversity_based_Self_Adaptive_Differential_Evolution
The Diversity Self-Adaptive Differential Evolution (DSA-DE) algorithm implements an examination of the diversity as a self-adaptive process for configuring its internal parameters.

Normal FULL reporting also writes `Results/EXP<ID>/full/Paper_Tables_EXP<ID>.xlsx`
through `reporting.paper_tables.export_paper_tables`. The workbook includes an
overall table and a dataset table for every classifier present in the results.
It selects available Accuracy, F1-Score, Precision, Recall, Fitness, Features,
and Time run arrays; Accuracy is converted from percent to 0–1 and the other
metrics retain their stored units. Existing Global Results, Statistical Results,
and Friedman exports are unchanged.

Overall Best/Worst/Mean/Std are calculated after averaging matching run positions
equally across datasets. Std uses `ddof=1` (zero for one finite run). Run counts
may vary between algorithm/classifier/metric groups, but must match across
datasets within each group. Missing combinations, empty arrays, and ambiguous
algorithm/classifier identities are rejected rather than silently aggregated.
Other experiment modes and the EXP627 report-only path do not invoke this exporter.

The historical three-sheet EXP627 manuscript layout remains available explicitly
with `python -m reporting.exp627_excel_tables`, retaining its cache identity and
reference checks and its `Results/EXP627/full_rep1/Paper_Tables_EXP627.xlsx` destination.
