"""Validated cache-only reporting and append-only report version publication."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np


def framework():
    # main_best.py may be running as __main__; reuse that exact module.
    main = sys.modules.get("__main__")
    if getattr(main, "__file__", "").endswith("main_best.py"):
        return main
    import main_best
    return main_best


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def report_stage(label):
    """Make long cache-only operations visible even in a buffered IDE console."""
    started = time.perf_counter()
    print(f'[report] {label} ...', flush=True)
    try:
        yield
    except BaseException:
        print(f'[report] {label} failed after {time.perf_counter() - started:.1f}s', flush=True)
        raise
    else:
        print(f'[report] {label} complete ({time.perf_counter() - started:.1f}s)', flush=True)


@contextmanager
def report_guard(destinations):
    """Reject scientific execution and filesystem mutations outside replica folders."""
    state = {"active": True, "optimization_calls": 0}
    forbidden = {"_run_single", "execute_pending_runs", "build_optimizer",
                 "configure_compute_backend", "save_cache", "start_gpu_request_service",
                 "load_dataset", "load_codesmell_dataset", "load_mafese_dataset"}
    blocked_codes = {}
    def profile(frame, event, arg):
        if event != 'call':
            return
        # Code metadata is immutable. Inspect it once instead of re-triggering
        # object.__getattr__ audit events on every library function call.
        code = frame.f_code
        blocked = blocked_codes.get(code)
        if blocked is None:
            name = code.co_name
            blocked = (name in forbidden or (name == 'solve' and 'mealpy' in code.co_filename)
                       or (name == 'fit' and 'mafese' in code.co_filename))
            blocked_codes[code] = blocked
        if blocked:
            state["optimization_calls"] += 1
            raise RuntimeError(f"REPORT-ONLY blocked scientific execution: {code.co_name}")
    def allowed(path):
        if isinstance(path, int):
            return False
        resolved = Path(os.fsdecode(path)).resolve()
        return any(resolved == base or base in resolved.parents for base in destinations) and resolved.suffix.lower() not in {'.pkl', '.pickle', '.ckpt', '.pdf'}
    mutation_events = frozenset({'open', 'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod',
                                'os.utime', 'os.truncate', 'shutil.rmtree', 'os.rename',
                                'os.link', 'os.symlink', 'subprocess.Popen', 'os.system', 'os.fork'})
    def audit(event, args):
        if not state["active"] or event not in mutation_events:
            return
        targets = []
        def at(path, dir_fd):
            # shutil.rmtree uses descriptor-relative unlink/rmdir on Linux.
            if dir_fd is not None and dir_fd != -1 and not os.path.isabs(path):
                return Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / os.fsdecode(path)
            return path
        if event == "open":
            path, mode, flags = args
            if (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                targets = [path]
        elif event in {"os.remove", "os.rmdir"}:
            targets = [at(args[0], args[1])]
        elif event in {"os.mkdir", "os.chmod"}:
            targets = [at(args[0], args[2])]
        elif event in {"os.utime", "os.truncate", "shutil.rmtree"}:
            targets = [args[0]]
        elif event in {"os.rename", "os.link", "os.symlink"}:
            targets = list(args[:2])
        elif event in {"subprocess.Popen", "os.system", "os.fork"}:
            raise RuntimeError(f"REPORT-ONLY blocked process execution: {event}")
        if any(not allowed(path) for path in targets):
            raise PermissionError(f"REPORT-ONLY blocked mutation outside replica output: {event} {targets}")
    previous = sys.getprofile()
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    sys.setprofile(profile)
    try:
        yield state
    finally:
        state["active"] = False
        sys.setprofile(previous)
        sys.dont_write_bytecode = previous_bytecode


@dataclass
class CompletedReport:
    args: argparse.Namespace
    results: dict
    indexed: dict
    datasets: list
    classifiers: list
    algorithms: list
    metrics: list
    signature: str
    sources: dict

    @property
    def exp_tag(self):
        return f"EXP{self.args.exp_id:03d}"


def safe_path(path):
    """Reject redirects, traversal and non-directory ancestors before any write."""
    path = Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Unsafe reporting path: {path}")
    if any(p.exists() and not p.is_dir() for p in path.parents):
        raise ValueError(f"Non-directory reporting ancestor: {path}")
    return path


def load_completed_cache(args):
    """Read exact final-cache identities; never use progress or migration readers.

    Configuration determines the required grid, and cache labels/records supply
    its data. All framework run fields are required by the numerical exporters.
    Unavailable metrics/curves are explicitly omitted; partial metric grids fail.
    """
    from reporting.paper_tables import METRICS
    m = framework()
    args = argparse.Namespace(**vars(args))
    if not isinstance(args.exp_id, int) or args.exp_id < 0 or args.runs < 1:
        raise ValueError("An explicit nonnegative EXP ID and positive run count are required")
    m.apply_experiment_mode(args)  # Local Namespace only; no scientific globals change.
    signature = m.build_cache_signature(args)
    root = safe_path(args.output_root)
    tag = f"EXP{args.exp_id:03d}"
    cache = safe_path(root / 'Results' / tag / args.experiment_mode / 'cache')
    if not cache.is_dir():
        raise FileNotFoundError(f"Missing completed cache: {cache}; no recomputation allowed")
    configured = m.configured_dataset_names(args)
    classifiers = list(dict.fromkeys(str(c).lower() for c in args.estimators))
    if not classifiers:
        raise ValueError("No configured classifiers")
    if configured is None:
        # Resolve the configured source list without reading dataset contents.
        # This detects a whole missing dataset, as well as missing classifiers.
        configured = [spec.name for spec in m.resolve_dataset_specs(args)]
    datasets = list(dict.fromkeys(Path(str(d)).stem if str(d).endswith('.csv') else str(d)
                                  for d in configured))
    if not datasets:
        raise ValueError("No configured datasets")
    if any('/' in d or '\\' in d or d in {'.', '..'} for d in datasets):
        raise ValueError("Unsafe dataset identity")
    results, indexed, sources, algorithms = {}, {}, {}, []
    show_tf, show_cls = len(args.transfer_functions) > 1, len(classifiers) > 1
    for dataset in datasets:
        results[dataset] = {}
        for classifier in classifiers:
            legacy_full = False
            classifier_signature = m.classifier_cache_signature(args, signature, classifier)
            path = safe_path(cache / f'{tag}_{dataset}_{classifier}_{classifier_signature}_results.pkl')
            if not path.is_file() and classifier == 'rf':
                path = safe_path(cache / f'{tag}_{dataset}_{classifier}_{signature}_results.pkl')
            if not path.is_file() and args.experiment_mode == 'full':
                # Historical reporting stays exact and read-only; wildcard
                # migration belongs exclusively to normal scientific execution.
                legacy_signature = m.build_legacy_cache_signature(args)
                path = safe_path(cache / f'{tag}_{dataset}_{classifier}_{legacy_signature}_results.pkl')
                legacy_full = True
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed cache: {path}; no recomputation allowed")
            before = sha256(path)
            payload = m.load_cache(str(path))
            expected = m.expected_result_labels(args, classifier, show_tf, show_cls)
            canonical = m.expected_result_labels(args, classifier, show_tf, True)
            groups = m.expected_result_labels(args, classifier, show_tf, False)
            if args.experiment_mode == 'full' and not legacy_full:
                # FULL updates append snapshots. Keep reporting read-only and
                # final-only, selecting a complete scientifically matching grid.
                snapshot_signature = classifier_signature if path.name.startswith(f'{tag}_{dataset}_{classifier}_{classifier_signature}_') else signature
                snapshots = sorted(cache.glob(f'{tag}_{dataset}_{classifier}_{snapshot_signature}_snapshot_*_results.pkl'))
                for candidate in (path, *snapshots):
                    candidate = safe_path(candidate)
                    stored = m.load_cache(str(candidate))
                    selected = {}
                    for (label, legacy), (output, output_legacy), (group, group_legacy) in zip(expected, canonical, groups):
                        keys = [key for key in dict.fromkeys((label, legacy, output, output_legacy, group, group_legacy))
                                if isinstance(stored, dict) and key in stored]
                        if len(keys) != 1:
                            break
                        row = stored[keys[0]]
                        parsed = m.parse_result_label(output, args)
                        metadata = m.full_checkpoint_metadata(args, parsed['method'], dataset, classifier,
                            parsed['transfer_function'] or args.transfer_functions[0])
                        values, reason = m.validate_source_label_runs(row, classifier, args.runs, metadata)
                        if reason or len(values['AccRuns']) != args.runs:
                            break
                        selected[keys[0]] = row
                    else:
                        path, payload, before = candidate, selected, sha256(candidate)
                        break
            if not isinstance(payload, dict) or len(payload) != len(expected):
                raise ValueError(f"Incomplete/extra optimizer rows: {path}")
            used = set()
            for (label, legacy), (output_label, output_legacy), (algorithm, group_legacy) in zip(expected, canonical, groups):
                # Classifier-specific legacy caches can have unqualified labels.
                candidates = (label, legacy, output_label, output_legacy, algorithm, group_legacy)
                matches = [key for key in dict.fromkeys(candidates) if key in payload]
                if len(matches) != 1 or matches[0] in used:
                    raise ValueError(f"Missing/ambiguous result label: {path}/{label}")
                used.add(matches[0])
                row = payload[matches[0]]
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid cache row: {path}/{label}")
                if classifier == 'rf':
                    backend_reason = m.rf_backend_compatibility(row, m.resolve_rf_backend(args))
                    if backend_reason:
                        raise ValueError(f'{path.name}/{label}: {backend_reason}')
                fields = [metric.run_key for metric in METRICS] + ['CurvesAll']
                if all(field in row for field in fields):
                    values, reason = m.validate_source_label_runs(row, classifier, args.runs)
                else:
                    # The framework validator requires its full current schema.
                    # Older caches may omit an entire metric: validate only actual
                    # observations, and never fabricate fields to pass that validator.
                    values = {field: row[field] for field in fields if field in row}
                    reason = None if str(row.get('Estimator', '')).lower() == classifier else 'classifier metadata mismatch'
                if reason or int(row.get('CompletedRuns', 0)) != args.runs:
                    raise ValueError(f"Incomplete cache {path}/{label}: {reason or 'completed run count mismatch'}")
                parsed = m.parse_result_label(output_label, args)
                if args.experiment_mode == 'full' and (not legacy_full or 'CacheIdentity' in row):
                    expected_metadata = m.full_checkpoint_metadata(
                        args, parsed['method'], dataset, classifier,
                        parsed['transfer_function'] or args.transfer_functions[0],
                    )
                    if not m.checkpoint_metadata_matches(row, expected_metadata):
                        raise ValueError(f'Scientific cache identity mismatch: {path}/{label}')
                if parsed['estimator'] != classifier:
                    raise ValueError(f"Classifier identity mismatch: {output_label}")
                if row.get('ExperimentMode', args.experiment_mode) != args.experiment_mode:
                    raise ValueError(f"Mode identity mismatch: {output_label}")
                for metric in METRICS:
                    if metric.run_key not in values:
                        continue
                    arr = np.asarray(values[metric.run_key], dtype=float)
                    if arr.shape != (args.runs,) or not np.isfinite(arr).all():
                        raise ValueError(f"Invalid run values: {path}/{label}/{metric.run_key}")
                    mean_key = metric.run_key.replace('Runs', 'Mean')
                    if not np.isclose(row.get(mean_key, np.nan), arr.mean(), rtol=1e-12, atol=1e-12):
                        raise ValueError(f"Cached summary disagrees with runs: {path}/{label}/{mean_key}")
                curves = [np.asarray(c, dtype=float) for c in values.get('CurvesAll', [])]
                if 'CurvesAll' in values and len(curves) != args.runs:
                    raise ValueError(f'Incomplete convergence run array: {label}')
                if any(c.ndim != 1 or not np.isfinite(c).all() for c in curves):
                    raise ValueError(f"Invalid convergence curves: {label}")
                mean_curve = np.asarray(row.get('Curve', []), dtype=float)
                if mean_curve.size:
                    if (mean_curve.shape != (args.epochs,) or not np.isfinite(mean_curve).all()
                            or (curves and (not all(c.size for c in curves) or not np.allclose(
                                mean_curve, m.pad_mean_curves(curves, args.epochs), rtol=1e-12, atol=1e-12)))):
                        raise ValueError(f"Invalid cached mean convergence: {label}")
                key = dataset, classifier, algorithm
                if key in indexed:
                    raise ValueError(f"Ambiguous scientific identity: {key}")
                indexed[key] = row
                results[dataset][output_label] = row
                if algorithm not in algorithms:
                    algorithms.append(algorithm)
            if set(payload) != used or sha256(path) != before:
                raise ValueError(f"Cache identity/content changed during read: {path}")
            sources[str(path.relative_to(root))] = before
    metrics = [metric for metric in METRICS if any(metric.run_key in row for row in indexed.values())]
    if not metrics:
        raise ValueError('No available run metrics in completed cache')
    for metric in metrics:
        if any(metric.run_key not in row for row in indexed.values()):
            raise ValueError(f'Incomplete metric grid: {metric.run_key}')
    return CompletedReport(args, results, indexed, datasets, classifiers, algorithms,
                           metrics, signature, sources)


def report_versions(root, exp_id):
    """Inspect both trees. Unpaired, redirected or malformed versions are errors."""
    if not isinstance(exp_id, int) or exp_id < 0:
        raise ValueError('A nonnegative user-selected EXP ID is required')
    root = safe_path(root)
    versions = []
    for kind in ('Figures', 'Results'):
        parent = safe_path(root / kind / f'EXP{exp_id:03d}')
        found = set()
        if parent.exists():
            for path in parent.iterdir():
                match = re.fullmatch(r'full_rep([1-9][0-9]*)', path.name)
                if not match:
                    if re.fullmatch(r'full_rep[0-9]+', path.name):
                        raise ValueError(f"Noncanonical report version: {path}")
                    continue
                safe_path(path)
                if not path.is_dir():
                    raise ValueError(f"Conflicting report version: {path}")
                for child in path.rglob('*'):
                    if child.is_symlink():
                        raise ValueError(f"Symlink within report: {child}")
                manifest_path = path / 'validation.json'
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    if (manifest.get('experiment_id', exp_id) != exp_id
                            or manifest.get('report_version', int(match[1])) != int(match[1])):
                        raise ValueError(f'Conflicting report manifest identity: {manifest_path}')
                found.add(int(match[1]))
        versions.append(found)
    if versions[0] != versions[1]:
        raise ValueError(f"Incomplete Figures/Results report pairs: {versions}; existing files preserved")
    return sorted(versions[0])


def next_report_version(root, exp_id):
    return max(report_versions(root, exp_id), default=0) + 1


def _rename_new(source, destination):
    """Atomic, no-replace directory publication (Linux); fail closed elsewhere."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise RuntimeError('Atomic no-replace publication requires renameat2')
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def staged_version(root, exp_id):
    """Serialize allocators, stage both trees, then publish a matching pair.

    A crash between the two renames leaves an unpaired version: the next attempt
    fails safely for manual inspection. Temporary names never consume a number.
    """
    root = safe_path(root)
    next_report_version(root, exp_id)  # Preflight before directory creation.
    parents = [safe_path(root / k / f'EXP{exp_id:03d}') for k in ('Figures', 'Results')]
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
    lock = parents[1] / '.report-allocation.lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(f'Report allocation is locked; inspect before retrying: {lock}') from exc
    stages, published = [], []
    try:
        version = next_report_version(root, exp_id)
        finals = [p / f'full_rep{version}' for p in parents]
        for parent in parents:
            stages.append(Path(tempfile.mkdtemp(prefix='.report-staging-', dir=parent)))
        yield version, stages[0], stages[1], finals
        for source, target in zip(stages, finals):
            safe_path(target)
            inode = source.stat().st_ino
            _rename_new(source, target)
            published.append((source, target, inode))
    except BaseException:
        for source, target, inode in reversed(published):
            if not target.is_symlink() and target.stat().st_ino == inode:
                _rename_new(target, source)
        raise
    finally:
        for stage in stages:
            if stage.exists() and not stage.is_symlink():
                shutil.rmtree(stage)
        lock.rmdir()


def _validate_artifacts(figures, results, required):
    from openpyxl import load_workbook
    from PIL import Image
    for relative in required:
        if not (results / relative).is_file():
            raise ValueError(f'Missing required report output: {relative}')
    hashes = {}
    for base in (figures, results):
        for path in base.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Redirected generated output: {path}')
            if not path.is_file():
                continue
            if path.suffix.lower() in {'.pdf', '.pkl', '.pickle', '.ckpt'} or path.stat().st_size == 0:
                raise ValueError(f'Invalid generated output: {path}')
            if path.suffix == '.xlsx':
                wb = load_workbook(path, data_only=False)
                try:
                    if not wb.sheetnames or any(ws.max_row < 1 for ws in wb):
                        raise ValueError(f'Empty workbook: {path}')
                    if path.name.startswith('Paper_Tables'):
                        from reporting.paper_tables import validate_plain_workbook
                        validate_plain_workbook(wb)
                finally:
                    wb.close()
            if path.suffix == '.png':
                with Image.open(path) as png:
                    if png.format != 'PNG' or any(abs(v - 600) > .1 for v in png.info.get('dpi', (0, 0))):
                        raise ValueError(f'Expected 600 dpi PNG: {path}')
                    png.verify()
            hashes[('Figures/' if base == figures else 'Results/') + str(path.relative_to(base))] = sha256(path)
    return hashes


def generate_outputs(report, figures, results):
    """Reuse framework Excel exporters without calling its experiment dispatcher."""
    from reporting import figures as plotting, statistics, paper_tables
    m, args = framework(), report.args
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    tag = report.exp_tag
    required = [f'Global_Results_{tag}.xlsx', f'Statistical_Results_{tag}.xlsx', f'Paper_Tables_{tag}.xlsx',
                'statistics/statistical_summary.csv', 'statistics/pairwise_wilcoxon_holm.csv',
                'statistics/statistical_report.txt']
    with report_stage(f'{tag}: Global Results Excel'):
        m.export_global_excel(report.results, report.datasets, str(results / required[0]))
    with report_stage(f'{tag}: Statistical Results Excel'):
        m.export_statistical_excel(report.results, report.datasets, args.optimizers, args, str(results / required[1]))
    with report_stage(f'{tag}: paper tables (format, serialize, validate)'):
        paper_tables.export_indexed_tables(report.indexed, report.datasets, report.algorithms, report.classifiers,
                                          report.metrics, results / required[2], title=f'{tag} {args.experiment_mode.upper()}')
    skipped = [{'output': f'{metric.name} tables/figures', 'reason': f'{metric.run_key} unavailable in every cache row'}
               for metric in paper_tables.METRICS if metric not in report.metrics]
    if args.experiment_mode in {'full', 'ablation'} and any(metric.run_key == 'FitRuns' for metric in report.metrics):
        name = f'{args.experiment_mode.capitalize()}_Friedman_Analysis_{tag}.xlsx'
        with report_stage(f'{tag}: Friedman Excel'):
            m.export_friedman_analysis(report.results, report.datasets, args.optimizers, args, str(results / name))
        required.append(name)
    else:
        skipped.append({'output': 'Framework fitness Friedman Excel', 'reason': 'Requires full/ablation mode and available FitRuns'})
    with report_stage(f'{tag}: publication figures at 600 dpi'):
        skipped.extend(plotting.generate(report, figures))
    with report_stage(f'{tag}: statistical analysis and figures'):
        skipped.extend(statistics.export(report, plotting.individual_destination(report, figures) / 'statistics', results / 'statistics'))
    if args.experiment_mode == 'full':
        actual = {p.name for p in figures.iterdir() if p.is_file()}
        expected = set(plotting.base_figure_names(report))
        omitted = {item['output'] for item in skipped} & expected
        if actual != expected - omitted:
            raise ValueError(f'Unexpected root figure layout: expected {sorted(expected - omitted)}, got {sorted(actual)}')
    return required, skipped


def run_report(args):
    """Create one complete report version for the user-selected EXP and modes."""
    # Import exporters/dependencies before the strict mutation guard is active.
    started = time.perf_counter()
    print(f'[report] Starting cache-only EXP{args.exp_id:03d}: {", ".join(args.experiment_modes)}', flush=True)
    from reporting import figures, statistics, paper_tables
    m = framework()
    root = safe_path(args.output_root)
    modes = list(dict.fromkeys(args.experiment_modes))
    if not modes or any(mode not in {'full', 'ablation', 'sensitivity', 'sensitivity_weights', 'transfer_functions'} for mode in modes):
        raise ValueError('Select at least one supported experiment mode')
    reports = []
    with report_stage('Validate original completed caches'), report_guard(()) as preflight:
        for mode in modes:
            local = m.clone_args_for_mode(args, mode)
            studies = m.sensitivity_study_args(local) if mode == 'sensitivity' else [local]
            reports.extend(load_completed_cache(study) for study in studies)
    with report_stage('Hash protected historical outputs'):
        previous = {str(p.relative_to(root)): sha256(p) for kind in ('Figures', 'Results')
                    for p in (root / kind / f'EXP{args.exp_id:03d}').rglob('*') if p.is_file()}
    destination_root = safe_path(getattr(args, 'report_output_root', None) or root)
    print(f'[report] Source: {root}; destination: {destination_root}', flush=True)
    with staged_version(destination_root, args.exp_id) as (version, fig, res, finals):
        old_tempdir = tempfile.tempdir
        tempfile.tempdir = str(res)
        try:
            with report_guard((fig, res)) as guard:
                required, entries = [], []
                for report in reports:
                    sub = Path('.') if len(reports) == 1 else Path(report.args.experiment_mode)
                    if len(reports) > 1 and report.args.experiment_mode == 'sensitivity':
                        sub /= report.args.sensitivity_parameter
                    names, skipped = generate_outputs(report, fig / sub, res / sub)
                    required.extend(str(sub / name) for name in names)
                    entries.append({'mode': report.args.experiment_mode, 'subdirectory': str(sub),
                                    'datasets': report.datasets, 'classifiers': report.classifiers,
                                    'algorithms': report.algorithms, 'metrics': [m.run_key for m in report.metrics],
                                    'source_directory': str(root / 'Results' / report.exp_tag / report.args.experiment_mode / 'cache'),
                                    'completed_runs': sorted({r['CompletedRuns'] for r in report.indexed.values()}),
                                    'cache_identity': report.signature, 'source_cache_sha256': report.sources,
                                    'figure_style': figures.base_style.STYLE_ID,
                                    'base_classifier': figures.base_classifier(report) if report.args.experiment_mode == 'full' else None,
                                    'root_figures': sorted(p.name for p in (fig / sub).glob('*.png')),
                                    'individual_figures': sorted(str(p.relative_to(fig / sub)) for p in
                                        figures.individual_destination(report, fig / sub).rglob('*.png'))
                                        if report.args.experiment_mode == 'full' else [],
                                    'skipped_outputs': skipped})
                with report_stage('Validate every generated artifact'):
                    hashes = _validate_artifacts(fig, res, required)
                with report_stage('Verify protected historical hashes'):
                    if any(sha256(root / path) != digest for path, digest in previous.items()):
                        raise ValueError('Protected source/report changed during reporting')
                manifest = {'experiment_id': args.exp_id, 'report_version': version,
                            'report_version_is_scientific_repetition': False, 'reports': entries,
                            'figures_destination': str(finals[0]), 'results_destination': str(finals[1]),
                            'outputs_sha256': hashes, 'required_results': required,
                            'optimization_calls': guard['optimization_calls'] + preflight['optimization_calls'],
                            'dpi': 600, 'pdfs_generated': 0, 'protected_files_unchanged': True}
                (res / 'validation.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        finally:
            tempfile.tempdir = old_tempdir
    print(f'[report] Published EXP{args.exp_id:03d} full_rep{version} in {time.perf_counter() - started:.1f}s: '
          f'{finals[0]} and {finals[1]}', flush=True)
    return manifest
