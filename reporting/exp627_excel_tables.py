"""Generate only the three EXP627 paper tables; never invoke experiment execution.

Run: python -B -m reporting.exp627_excel_tables
Only Results/EXP627/full_rep1/Paper_Tables_EXP627.xlsx is replaced.
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.exp627_core import (
    CLASSIFIERS, DATASETS, TABLE_ORDER, framework, load_completed_full,
    report_guard, sha256,
)

from reporting import paper_tables


METRICS = (
    ("Accuracy", "AccRuns", "AS_test", "Accuracy", 100.0),
    ("F1-Score", "F1Runs", "F1_test", "F1Score", 1.0),
    ("Precision", "PSRuns", "PS_test", "Precision", 1.0),
    ("Recall", "RSRuns", "RS_test", "Recall", 1.0),
)
CLASSIFIER_ORDER = ("knn", "rf", "svm")
DATASET_LABELS = ("Data Class", "Feature Envy", "God Class", "Long Method",
                  "Long Parameter List", "Switch Statement")
STATS = paper_tables.STATS
SHEETS = ("Table2_Overall", "Table3_Datasets", "Table4_Datasets")


def protected_hashes(root, destination):
    files = set(root.rglob("*.pkl"))
    for kind in ("Results", "Figures"):
        files.update(p for p in (root / kind / "EXP627").rglob("*")
                     if p.is_file() and destination not in p.parents)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


def require_close(actual, reference, label):
    if not np.allclose(actual, reference, rtol=1e-12, atol=1e-12):
        raise ValueError(f"EXP627 reference-mean validation failed: {label}")


def prepare_tables(args):
    """Validate report means first; calculate statistics only from original runs."""
    m = framework()
    args, results, indexed, _ = load_completed_full(args)
    classifier = inspect.signature(m._generate_seven_global_charts).parameters["estimator_filter"].default
    if classifier != "svm":
        raise ValueError("EXP627 per-dataset report classifier changed; audit required")
    source = Path(args.output_root) / "Results/EXP627/full"
    summary = pd.read_csv(source / "RESUMEN_GRAFICAS_EXP627.csv")
    key_columns = ["Dataset", "Estimator", "Optimizer"]
    if summary.duplicated(key_columns).any():
        raise ValueError("Ambiguous identities in original EXP627 report")
    summary = summary.set_index(key_columns)
    expected_keys = {(ds, cls, opt) for ds in DATASETS for cls in CLASSIFIERS for opt in TABLE_ORDER}
    if set(summary.index) != expected_keys:
        raise ValueError("Original EXP627 report must have all 216 unique combinations")
    existing = pd.read_excel(source / "Global_Results_EXP627.xlsx", sheet_name=None, index_col=0)
    # Reproduce every cached mean against two independently serialized original reports.
    for (dataset, cls, opt), row in indexed.items():
        for metric, run_key, csv_key, excel_sheet, scale in METRICS:
            mean = np.mean(row[run_key]) / scale
            require_close(mean, summary.loc[(dataset, cls, opt), csv_key], f"CSV/{dataset}/{cls}/{opt}/{metric}")
            require_close(mean, existing[excel_sheet].loc[dataset, f"{opt}_{cls.upper()}"] / scale,
                          f"Excel/{dataset}/{cls}/{opt}/{metric}")
    _, layouts = manuscript_layout(classifier)
    tables = paper_tables.calculate_tables(indexed, TABLE_ORDER, layouts, m._run_stats)
    # Independent serialized means remain authoritative for this historical preset.
    for name, columns in layouts.items():
        reference = [[np.mean([summary.loc[(ds, cls, opt),
                         next(item[2] for item in METRICS if item[1] == metric.run_key)]
                         for ds in datasets])
                      for _, _, datasets, cls, metric in columns] for opt in TABLE_ORDER]
        require_close(tables[name][1], reference, f"{name}/Mean rows")
    return tables, classifier


def manuscript_layout(classifier):
    metrics = tuple(paper_tables.Metric(name, key, scale=scale)
                    for name, key, _, _, scale in METRICS)
    layouts = {SHEETS[0]: [(metric.name, cls.upper(), DATASETS, cls, metric)
                          for metric in metrics for cls in CLASSIFIER_ORDER]}
    for index, name in enumerate(SHEETS[1:]):
        offset = index * 3
        layouts[name] = [(label, metric.name, [dataset], classifier, metric)
                         for dataset, label in zip(DATASETS[offset:offset+3], DATASET_LABELS[offset:offset+3])
                         for metric in metrics]
    return metrics, layouts


def make_workbook(tables, classifier):
    _, layouts = manuscript_layout(classifier)
    return paper_tables.make_workbook(
        tables, TABLE_ORDER, layouts,
        title="EXP627 FULL — Paper Tables 2, 3 and 4", dataset_heading="Code Smell",
        algorithm_labels={"DSADE": "DSA-DE"},
        description=(
            "30 completed runs; sample Std (ddof=1); all metrics 0–1. "
            "Overall statistics use 30 per-run six-dataset averages. Dataset tables use SVM. "
            "Mean rows verified against original EXP627 Global Results and report CSV; "
            "manuscript screenshots are not available for direct comparison."
        ),
    )


def run_paper_tables(args):
    root = Path(args.output_root).resolve()
    destination = root / "Results/EXP627/full_rep1"
    target = destination / "Paper_Tables_EXP627.xlsx"
    from reporting.exp627_figures import validate_output_destination
    validate_output_destination(root, destination)
    before = protected_hashes(root, destination)
    # Reuse the report-only write/execution guard. No figure functions are called.
    with report_guard((destination,)) as guard:
        tables, classifier = prepare_tables(args)
        wb = make_workbook(tables, classifier)
        print("Validated: 6 datasets / 3 classifiers / 12 algorithms / 30 runs; SVM dataset tables.")
        print("All 432 Mean cells reproduce the original EXP627 report values (tolerance 1e-12).")
        destination.mkdir(exist_ok=True)
        payload = paper_tables.workbook_bytes(wb, tables, temp_dir=destination)
        if protected_hashes(root, destination) != before:
            raise AssertionError("Protected original files changed before saving")
        target.write_bytes(payload)
        assert guard["optimization_calls"] == 0
    if protected_hashes(root, destination) != before:
        raise AssertionError("Protected original files changed")
    print(f"Created: {target}")
    print(f"Verified 3 worksheets / 1,728 numeric cells; optimization calls = 0; {len(before)} protected files unchanged (SHA-256).")


if __name__ == "__main__":
    run_paper_tables(framework().parse_args())
