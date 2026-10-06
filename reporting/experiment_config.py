"""Experiment configuration manifests; checking is strictly read-only."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

FIELDS = ('dataset_source', 'datasets', 'optimizers', 'estimators', 'transfer_functions',
          'runs', 'epochs', 'pop_size', 'test_size', 'random_state', 'seed_base',
          'fitness_alpha', 'fitness_beta', 'dsade_beta_min', 'dsade_beta_max',
          'dsade_pcr', 'dsade_mahal_q')
SELECTIONS = frozenset(('datasets', 'optimizers', 'estimators', 'transfer_functions'))
EXTRAS = ('sensitivity_parameter', 'sensitivity_values', 'sensitivity_weight_pairs')


def manifest_path(args):
    from reporting.core import safe_path
    return safe_path(Path(args.output_root) / 'Results' / f'EXP{args.exp_id:03d}' /
                     args.experiment_mode / 'experiment_config.json')


def read_manifest(args):
    path = manifest_path(args)
    if not path.exists():
        return None
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
        if (config['schema_version'] != 1 or config['experiment_id'] != args.exp_id
                or config['experiment_mode'] != args.experiment_mode):
            raise ValueError('manifest experiment identity/schema MISMATCH')
        for field in FIELDS:
            if field not in config:
                raise ValueError(f'missing field {field}')
        for field in SELECTIONS:
            if not isinstance(config[field], list) or not config[field]:
                raise ValueError(f'invalid selection {field}')
        for field in FIELDS:
            if field not in SELECTIONS and field != 'dataset_source':
                if not isinstance(config[field], (int, float)) or not math.isfinite(config[field]):
                    raise ValueError(f'invalid numeric setting {field}')
        if config['runs'] < 1 or config['epochs'] < 1 or config['pop_size'] < 1:
            raise ValueError('nonpositive runs/epochs/pop_size')
        if not isinstance(config['cache_signatures'], list) or not config['cache_signatures']:
            raise ValueError('missing real cache signatures')
        for entry in config['cache_signatures']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'cache':
                raise ValueError(f'unsafe cache path: {relative}')
            if not isinstance(entry['signature'], str) or not isinstance(entry['sha256'], str):
                raise ValueError('invalid cache signature/hash')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Manifest MISMATCH {path}: {exc}') from exc
    return config


def _canonical(field, value):
    from reporting.core import framework
    if field == 'optimizers':
        return sorted(framework().resolve_optimizer_name(name) for name in value)
    if field in SELECTIONS:
        return sorted(value)
    return json.loads(json.dumps(value))


def apply_manifest(args, config):
    """Use saved defaults and preserve every explicit user selection/setting."""
    from reporting.core import _audit_cache
    local = argparse.Namespace(**vars(args))
    explicit = getattr(args, 'report_explicit_options', ())
    for field in FIELDS + EXTRAS:
        if field not in config:
            continue
        flag = '--' + field.replace('_', '-')
        if flag in explicit:
            requested = _canonical(field, getattr(local, field))
            stored = _canonical(field, config[field])
            compatible = (set(requested) <= set(stored) if field in SELECTIONS and field != 'transfer_functions'
                          else requested == stored)
            if not compatible:
                _audit_cache('MISMATCH', field, getattr(local, field), config[field], 'experiment_config.json')
                raise ValueError(f'Manifest MISMATCH {field}: requested={getattr(local, field)!r}; stored={config[field]!r}')
        else:
            setattr(local, field, config[field])
    # Saved selections and source now constrain the cache grid as well.
    local.report_explicit_options = frozenset(explicit) | {'--datasets', '--estimators', '--optimizers', '--dataset-source'}
    local.report_manifest = config
    print(f'[config] Using {manifest_path(args)}', flush=True)
    return local


def config_from_report(report):
    from reporting.core import framework, _report_algorithm, sha256
    m, args = framework(), report.args
    config = {field: getattr(args, field) for field in FIELDS}
    config.update(schema_version=1, experiment_id=args.exp_id, experiment_mode=args.experiment_mode,
                  datasets=report.datasets, estimators=report.classifiers,
                  optimizers=list(dict.fromkeys(_report_algorithm(a, '')[1] for a in report.algorithms)))
    for field in EXTRAS:
        if hasattr(args, field):
            config[field] = getattr(args, field)
    legacy = getattr(args, 'report_legacy_parameters', None)
    if legacy:
        config.update(legacy)
        # This legacy schema used the fixed default fitness contract. Dataset
        # source is inferred only from the named catalog, never from a run.
        config['fitness_alpha'], config['fitness_beta'] = .9, .1
        if set(report.datasets) <= set(m.TEST_DATASETS_CLASSIFICATION_14):
            config['dataset_source'] = 'mafese'
    identities = [r['CacheIdentity'] for r in report.indexed.values() if 'CacheIdentity' in r]
    parameters = {}
    for identity in identities:
        for field in FIELDS:
            if field in identity:
                config[field] = identity[field]
        method = identity['optimizer']
        params = identity.get('optimizer_parameters', {})
        if method in parameters and parameters[method] != params:
            raise ValueError(f'Inconsistent optimizer parameters for manifest: {method}')
        parameters[method] = params
    config['optimizer_parameters'] = parameters
    for field, parameter in (('dsade_beta_min', 'beta_min'), ('dsade_beta_max', 'beta_max'),
                             ('dsade_pcr', 'pcr'), ('dsade_mahal_q', 'mahalanobis_q')):
        values = {params[parameter] for params in parameters.values() if parameter in params}
        if len(values) == 1:
            config[field] = values.pop()
    config['configuration_evidence'] = 'CacheIdentity' if identities else 'verified legacy digest and historical fitness contract'
    mode_root = manifest_path(args).parent
    entries = []
    for relative, digest in report.sources.items():
        path = Path(args.output_root).absolute() / relative
        if sha256(path) != digest:
            raise ValueError(f'Cache changed before recording manifest: {path}')
        payload = m.load_cache(str(path))
        estimator = str(next(iter(payload.values()))['Estimator']).lower()
        prefix = f'{report.exp_tag}_'
        stem = path.stem[len(prefix):].rsplit('_', 1)[0]
        dataset, _, signature = stem.rpartition(f'_{estimator}_')
        entries.append(dict(dataset=dataset, estimator=estimator, signature=signature,
                            path=str(path.relative_to(mode_root)), sha256=digest))
        if sha256(path) != digest:
            raise ValueError(f'Cache changed while recording manifest: {path}')
    config['cache_signatures'] = entries
    return config


def write_manifest(args, config, *, replace=False):
    """Atomically publish metadata only; a report preserves an existing manifest."""
    path = manifest_path(args)
    if path.exists() and not replace:
        read_manifest(args)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return path
    descriptor, temporary = tempfile.mkstemp(prefix='.experiment-config-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'[config] Saved {path}', flush=True)
    return path


def save_report_config(report):
    config = config_from_report(report)
    previous = read_manifest(report.args)
    # A subset report preserves the complete experiment selection. A complete
    # report can refresh compatible cache locations (e.g. results -> progress).
    complete = previous is not None and all(_canonical(field, config[field]) == _canonical(field, previous[field])
                                           for field in SELECTIONS)
    return write_manifest(report.args, config, replace=complete)


def record_completed_config(args, datasets):
    """Record the selected completed science, including actual checkpoint filenames."""
    from reporting.core import load_cached_figure_report, report_guard
    local = argparse.Namespace(**vars(args))
    local.datasets = list(datasets)
    local.report_current_science = True
    local.report_explicit_options = {'--datasets', '--estimators', '--optimizers', '--dataset-source'}
    with report_guard(()):
        report = load_cached_figure_report(local, use_manifest=False)
    return write_manifest(local, config_from_report(report), replace=True)


def check_exp_config(args):
    """Compare current configuration against stored evidence without any writes."""
    from reporting.core import framework, report_guard, load_cached_figure_report, _audit_cache
    m, checks = framework(), []
    with report_guard(()):
        for mode in dict.fromkeys(args.experiment_modes):
            local = m.clone_args_for_mode(args, mode)
            m.apply_experiment_mode(local)
            try:
                config = read_manifest(local)
            except ValueError as exc:
                print(f'[config] MISMATCH {exc}', flush=True)
                checks.append(False)
                continue
            if config is None:
                try:
                    config = config_from_report(load_cached_figure_report(local, use_manifest=False))
                except (ValueError, FileNotFoundError) as exc:
                    print(f'[config] MISMATCH EXP{args.exp_id:03d}/{mode}: {exc}', flush=True)
                    checks.append(False)
                    continue
            print(f'[config] Check EXP{args.exp_id:03d}/{mode}', flush=True)
            current_datasets = local.datasets or m.configured_dataset_names(local)
            if current_datasets is None:
                try:
                    current_datasets = [spec.name for spec in m.resolve_dataset_specs(local)]
                except (ValueError, FileNotFoundError) as exc:
                    print(f'[config] MISMATCH current datasets: {exc}', flush=True)
                    current_datasets = []
            match = True
            for field in FIELDS + tuple(field for field in EXTRAS if field in config):
                current = current_datasets if field == 'datasets' else getattr(local, field)
                equal = _canonical(field, current) == _canonical(field, config[field])
                _audit_cache('MATCH' if equal else 'MISMATCH', field, current, config[field], 'stored experiment config')
                match &= equal
            for method, stored in config.get('optimizer_parameters', {}).items():
                if method not in {m.resolve_optimizer_name(o) for o in local.optimizers}:
                    continue
                current = m.full_cache_identity(local, method, config['datasets'][0],
                                               config['estimators'][0], config['transfer_functions'][0])['optimizer_parameters']
                equal = current == stored
                _audit_cache('MATCH' if equal else 'MISMATCH', f'optimizer_parameters/{method}', current, stored)
                match &= equal
            for entry in config['cache_signatures']:
                path = manifest_path(local).parent / entry['path']
                from reporting.core import safe_path, sha256
                path = safe_path(path)
                equal = path.is_file() and sha256(path) == entry['sha256']
                _audit_cache('MATCH' if equal else 'MISMATCH', 'cache_signatures', entry['signature'],
                             str(path), 'stored SHA256 ' + ('verified' if equal else 'missing/changed'))
                match &= equal
            if read_manifest(local) is not None:
                stored_args = argparse.Namespace(**vars(local))
                stored_args.report_explicit_options = frozenset()
                try:
                    load_cached_figure_report(stored_args)
                except (ValueError, FileNotFoundError) as exc:
                    print(f'[config] MISMATCH manifest/cache identity: {exc}', flush=True)
                    match = False
            checks.append(bool(match))
    matched = bool(checks) and all(checks)
    print(f'[config] {"MATCH" if matched else "MISMATCH"} overall EXP{args.exp_id:03d}', flush=True)
    return {'match': matched, 'optimization_calls': 0}
