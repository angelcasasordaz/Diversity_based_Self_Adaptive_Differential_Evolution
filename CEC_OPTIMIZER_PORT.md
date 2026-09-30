# CEC optimizer port audit

## DSADE-CEC GPU transport (2026-09-30)

DSADE-CEC now accepts `compute_device`, `gpu_device_id`, and
`gpu_memory_fraction` through the existing custom-optimizer adapter. It is
registered as `GENERIC_GPU`, using `DiversityMathBatcher` and the existing local
or remote GPU owner service. AWAD, regularized covariance/Cholesky inverse
Mahalanobis grouping, mutation, and crossover execute on the device. Agent
state, ordered RNG draws, adaptation scalars, MAFESE/scikit-learn fitness,
solution correction, and greedy fitness survivor selection remain on CPU.

The CPU AWAD and covariance equations remain intact. The GPU pool uses the same
inverse-based Cholesky geometry, chi-square cutoff, pool order, fallback, and
per-target population reconstruction. No frozen-generation change was applied
to CEC DSADE: its single-mode population can change after each greedy survivor,
and swarm-mode selection remains deferred. The scientific revision is unchanged
because CPU behavior and scientific parameters are unchanged.

GPU requests still raise on unavailable CUDA or failed service operations;
there is no CPU fallback for explicit GPU mode. Existing factory and comparison
validation code required no changes. Tests confirm five of five configured
optimizers validate with an available GPU backend, and unavailable backends
remain rejected. Main DSADE and all three MaCRO implementations are unchanged.

Validation: **25 focused tests run, 23 passed, two real-CUDA tests skipped**
(`cudaErrorNoDevice`). Seeded CPU trajectories match the authoritative CEC
source exactly; the emulated GPU service path matches CPU trajectories/control
histories in single and swarm modes (one and six dimensions), retaining RNG
state. Source AST checks normalize only added GPU dispatch/equations. Factory
device propagation, required kernel dispatch, strict failure behavior, aliases,
five-optimizer strategy/validation, and main DSADE preservation passed.
Compile/import checks and `git diff --check` passed. Actual CUDA parity and
runtime five-of-five validation remain unverified without a CUDA device.
No full EXP comparison or experiment artifact writes were performed.

Earlier statements below describing DSADE-CEC as CPU-only are historical and
superseded by this transport integration.

## Final active v2: MaCRO D-scaled coordinates and adaptive pcr (2026-09-30)

The active revision is `macro-d-scaled-coordinate-adaptive-pcr-v5`:

```text
F_j = clip(U(beta_min, beta_max) * clip(1.5 - D, 0.5, 1.5), 0.1, 1.5)
pcr = 0.1 + 0.25 * (1 - dM)
```

Uniform draws are independent per coordinate and mutation. D is the same
delayed diversity value used inline by `MaCRO_DE.evolve`:
`clip(div_norm_for_update, 0, 1)`. V2 reuses its inherited AWAD helper and
cumulative-maximum normalization, matching MaCRO-DE's definition. The completed
generation updates diversity as `clip(AWAD / (div_max_seen + EPSILON), 0, 1)`
for the next generation. Initial D is 1. D is frozen for the generation and
recorded in `d_hist`; F histories continue to record mean coordinate scales.
No DSADE additive AWAD-based F formula or scalar `rand*(.60+1-dM)` is used.

The separate target `dM = sqrt(dist2) / max(sqrt(dist2))` still comes from the
frozen generation's Mahalanobis distances (zero for a collapsed population).
Close/far groups still compare squared distances to the chi-square threshold
using regularized covariance/Cholesky geometry. AWAD does not compute the
groups: it supplies MaCRO D for scale adaptation and routes donor-pool sampling.
Frozen-generation mutation, target exclusion, forced crossover, and greedy
fitness survivor selection remain unchanged.

The adapter already forwards beta bounds and q without fixed pcr and needs no
further change. The revised scientific identity prevents reuse of v4 caches
without modifying them. Main DSADE still maps to `dsade_awad_optimizer.py`.
MaCRO-DE, MaCRO-DE-t, DSADE, and DSADE-CEC were not changed. MAFESE integration,
transfer/fitness functions, datasets, classifiers, reporting, and historical
comments are retained.

Final v5 validation: **19 focused tests run, 18 passed, one CUDA test skipped**
(`cudaErrorNoDevice`). Tests verify seeded coordinate draws with MaCRO D
scaling/clipping, the reused diversity definition, adaptive pcr from dM,
chi-square grouping, frozen mutation and D, target exclusion, greedy survivors
in single/swarm modes, both aliases, beta forwarding, cache revision isolation,
tiny MAFESE/KNN smoke runs, and unchanged comparison variants/main DSADE.
Compile/import checks and `git diff --check` passed. No full EXP comparison
was run. Ready for a new CPU EXP rerun; GPU execution remains unverified here.

## Historical v4: unscaled coordinate draws (superseded, 2026-09-30)

MaCRO-DE-t-v2 restores the reference v2 differential scale behavior:
`F_j ~ U(beta_min, beta_max)`, independently per coordinate and mutation, with
no dM scaling or clipping. Direct defaults are beta `.10/.60`; the FS adapter
again forwards `dsade_beta_min`/`dsade_beta_max` (currently `.40/.80` in
`main_best.py`). Only crossover remains adaptive:
`pcr = 0.1 + 0.25 * (1 - dM)`.

The target's `dM = sqrt(dist2) / max(sqrt(dist2))` still uses the frozen
generation's Mahalanobis distances from the population mean; a collapsed
population uses zero. Close/far remains regularized covariance/Cholesky
chi-square threshold classification, with pseudoinverse fallback. AWAD only
routes donor sampling. Frozen-generation mutation, target exclusion (including
fallback pools), forced crossover, and greedy fitness selection are retained.
`f_hist` records per-target mean coordinate scales; `fmean_hist` records the
generation mean. No coordinate-scale history tensor is stored.

Revision `coordinate-beta-adaptive-pcr-v4` includes beta bounds in scientific
cache identity and prevents reuse of the intermediate scalar-F revision.
Existing caches and experiment artifacts are untouched. Both
`MaCRO-DE-t-v2` and `MaCRO_DE_t_v2` resolve to the corrected class. MaCRO-DE,
MaCRO-DE-t, DSADE, and DSADE-CEC are unchanged; main DSADE still maps to
`dsade_awad_optimizer.py`. MAFESE integration, transfer/fitness functions,
datasets, classifiers, and reporting are retained.

Active v4 validation: **18 focused tests run, 17 passed, one CUDA test skipped**
(`cudaErrorNoDevice`). Tests cover exact seeded coordinate uniform draws,
beta-bound validation/forwarding and cache identity, adaptive pcr, normalized
distances, frozen-generation mutation, target exclusion, chi-square grouping,
greedy survivors in single/swarm modes, both aliases, tiny MAFESE/KNN smoke
runs, and unchanged source-equivalent comparison variants and main DSADE.
Compile/import checks and `git diff --check` passed. No full EXP comparison,
staging, commit, or push was performed. Ready for a new CPU EXP rerun; actual
GPU execution remains unverified in this environment.

## Historical adaptive v3 correction (superseded F behavior, 2026-09-30)

The active FS `MaCRO-DE-t-v2` intentionally supersedes the fixed/configurable-pcr
CEC v2 implementation. The supplied reference checkout
`../Adaptive_Mahalanobis-Cholesky_Differential_Evolution_MaCRO_DE` still contains
that older behavior, so v2 is no longer an exact source-equivalence comparison.
`DSADE-CEC` remains the exact CEC/greedy comparison.

For each target in the frozen generation, squared Mahalanobis distance is
computed from the population mean using regularized covariance (`1e-6`) and
Cholesky inverse geometry, with pseudoinverse fallback. Close/far classification
compares that squared distance with `chi2.ppf(mahalanobis_q, n_dims)`.
AWAD does **not** compute these groups; its existing delayed normalized value
only routes donor sampling between them.

The adaptive control distance is `dM = sqrt(dist2) / max(sqrt(dist2))` within
the frozen generation; all-zero distances give all-zero dM. Each target uses
one scalar `rand` draw in `[0, 1)`:

```text
F   = rand * (0.60 + (1 - dM))
pcr = 0.1 + 0.25 * (1 - dM)
```

V2 accepts epoch, population size, q, and device settings. The adapter does not
forward beta bounds or fixed pcr, and direct fixed-control arguments are rejected.
The revision `mahalanobis-adaptive-control-v3` distinguishes its cache identity
without changing existing caches. The MAFESE wrapper, transfer and fitness
functions, datasets, classifiers, reporting, and historical commented options
are retained. MaCRO-DE, MaCRO-DE-t, and v2 all use greedy fitness survivors.
Among these comparison variants, only main DSADE uses AWAD survivor selection
and still maps to `dsade_awad_optimizer.py`; the independent
`DE-DiversitySelection` ablation also has AWAD selection.

The earlier source audit below is historical wherever it describes fixed v2.

Correction validation: **17 focused tests run, 16 passed, one CUDA test skipped**
(`cudaErrorNoDevice`). This includes adaptive-control formula and generation
tests, both aliases, cache revision isolation, unchanged DSADE fingerprint and
AWAD selection, source equivalence for the remaining exact ports, and the
existing tiny MAFESE/KNN smoke runs. Compile/import checks passed without
writing bytecode; `git diff --check` passed. No full EXP comparison was run;
EXP627/EXP628 artifacts, datasets, reports, figures, and caches were untouched.
`optimizer_factory.py` was audited and required no change: its existing adapter
and revision hooks already consume the corrected v2 mapping and identity.

Audited on 2026-09-29 against the sibling project
`../Adaptive_Mahalanobis-Cholesky_DIfferential_Evolution`, commit
`9fd6895d2084760882b5b3325f667701a6964666`. The source files listed below were
unmodified in that checkout; its unrelated untracked validation script was not
used. Public identities were traced through `algorithm_acronym_list.py`, then
through the actual classes, inheritance, backend methods, and `main.py` parameter
mapping. No CEC benchmark, dataset, reporting, or experiment code was imported.

## Exact identities and the DSADE naming conflict

| Requested CEC name | CEC registry implementation | Feature-selection selection |
| --- | --- | --- |
| `DSADE`, `DSA-DE`, `DSA_DE` | `dsade_optimizer.DSADE` | Exact source available as **`DSADE-CEC`** (`dsade_cec_optimizer.DSADE`). Existing **`DSADE`** still resolves to `dsade_awad_optimizer.DSADE`, unchanged. |
| `MaCRO-DE`, `MACRO_DE`, `MACRODE` | `macro_de_optimizer.MaCRO_DE` | `MaCRO-DE` → `macro_de_optimizer.MaCRO_DE` |
| `MaCRO-DE-t`, `DE-MC-CF`, `DE_MC_CF` | `de_mc_cf_optimizer.DE_MC_CF` → `de_mc_optimizer.DE_MC` → `de_ablation_base.MahalanobisDEBase` | `MaCRO-DE-t` → existing `macro_de_t_optimizer.MaCRO_DE_t` (verified source-derived port) |
| `MaCRO-DE-t-v2`, `DE-MC-CF-v2`, `DE_MC_CF_V2` | `de_mc_cf_v2_optimizer.DE_MC_CF_V2` → the preceding chain | `MaCRO-DE-t-v2` → `macro_de_t_v2_optimizer.DE_MC_CF_V2` (also exported as `MaCRO_DE_t_v2`) |

CEC's public DSADE uses greedy fitness survivor selection. This project's DSADE
additionally accepts an offspring with a better local AWAD contribution even if
fitness is worse. Replacing it with CEC DSADE would violate the requirement to
keep existing DSADE unchanged. The extra `DSADE-CEC` entry preserves both exact
implementations without silently redefining an established alias. Source-local
historical aliases `DSADE`/`IMPDE` inside CEC's **MaCRO-DE module** are not the
CEC registry's public DSADE mapping and are not registered as such here.

## Scientific behavior

All variants use AWAD normalized against its cumulative maximum, delayed until
the next generation, to route toward the close pool at normalized diversity
`>= 0.5`, otherwise toward the far pool. They use a chi-square Mahalanobis
threshold, covariance regularization `1e-6`, and Cholesky-based inverse geometry
with pseudoinverse fallback. Forced binomial crossover and both calls to solution
correction are retained, including MAFESE's existing binary transfer behavior.

| FS name | Donor population | Mutation and crossover | Survivor selection |
| --- | --- | --- | --- |
| `DSADE` (unchanged) | Recomputed per target; can include target among donors | Coordinate-wise uniform beta, multiplied by `clip(1.5 - diversity, .5, 1.5)`, then clipped to `[.1, 1.5]`; crossover `clip(pcr + .25*(1-diversity), .1, .95)` | Better fitness, otherwise better local AWAD |
| `DSADE-CEC` | Recomputed per target; can include target among donors | Same adaptive scale/crossover formulas | CEC greedy fitness selection |
| `MaCRO-DE` | Frozen once per generation; can include target among donors | Same adaptive scale/crossover formulas | CEC greedy fitness selection |
| `MaCRO-DE-t` | Frozen once per generation; target excluded from all donor pools, including fallback | Fixed scalar `wf=.5`, fixed `cr=.9` | CEC greedy fitness selection |
| `MaCRO-DE-t-v2` | Same frozen population, target exclusion, routing, and geometry as `-t` | `F_j = clip(U(beta_min,beta_max)*clip(1.5-D,.5,1.5),.1,1.5)` with MaCRO D; adaptive `pcr = .1+.25*(1-dM)` | Greedy fitness selection |

The per-target population changes within a generation only in single mode; the
source's swarm/parallel mode performs deferred population selection. These
mode-specific semantics, RNG draw order, pool-size fallback, and tie handling
are preserved. Despite inheriting from `DE_MC`, CEC `DE_MC_CF` explicitly selects
the **inverse-based `cholesky`** backend path, not `cholesky_solve` whitening.
V2 retains that override.

## Integration boundary

- Replaced the old local MaCRO-DE implementation with the CEC source; only its
  backend constructor/import and scientific revision metadata were adapted.
- Initially copied v2's methods directly, importing the verified existing `-t`
  parent; the adaptive correction above now replaces its fixed control.
- Copied CEC DSADE directly; only documentation and revision metadata were added.
- Reused `macro_de_t_backend.CECCovarianceKernels` and `MaCRODETBackend`, which
  transport the CEC covariance equations through the existing NumPy/CuPy GPU
  owner service. No new numerical backend or CEC batch experiment scheduler is
  needed. MaCRO-DE, `-t`, and v2 retain CPU/GPU covariance support; their AWAD,
  RNG, and per-agent control stay on CPU as in the CEC classes. Local DSADE's
  existing GPU support is untouched. CEC DSADE is correctly declared CPU-only.
- Registered aliases in `optimizer_adapters.py`; the existing factory, resolver,
  capability analysis, and scientific cache identity hooks consume them directly.
  MaCRO-DE's new revision prevents old local MaCRO-DE caches from being treated
  as the new CEC implementation. Existing result files are not modified.
- All existing experiment settings remain in force: `dsade_beta_min`,
  `dsade_beta_max`, `dsade_pcr`, and `dsade_mahal_q` map to the corresponding
  parameters for DSADE, MaCRO-DE, and DSADE-CEC. `-t` receives only q and keeps
  fixed F/CR. V2 receives beta bounds and q, scales coordinate-wise uniform F
  with MaCRO D and clips it, and adapts pcr from target dM; direct v2
  construction defaults to beta `.10/.60`, q `.50`.
  The CEC experiment settings themselves were not copied.

## Files

Added:

- `macro_de_t_v2_optimizer.py`
- `dsade_cec_optimizer.py`
- `tests/test_cec_optimizer_port.py`
- `CEC_OPTIMIZER_PORT.md`

Modified:

- `macro_de_optimizer.py`: exact CEC generation logic and backend bridge.
- `optimizer_adapters.py`: independent aliases/classes and parameter mappings.
- `tests/test_macro_de_t.py`: correct sibling reference path and new alias assertion.
- `tests/test_dsade_canonical.py`: normalize Python 3.12/3.13 AST formatting to
  preserve its original Python 3.11 expected fingerprint, without changing it.

`main_best.py`, local DSADE, datasets, fitness, transfer functions, classifiers,
experiment settings, reporting, and existing results are unchanged.

## Validation

The focused suite tests resolution/construction of every registered candidate
alias, distinct scientific identities, parameter mappings, and tiny actual
`main_best._run_single` MAFESE runs (48 synthetic samples, six features, KNN,
`vstf_01`, ten individuals, two epochs). No full experiment runs.

Source-backed tests compare method ASTs and seeded CPU trajectories against the
unmodified CEC classes in single and swarm modes, in one and six dimensions.
They check final populations, fitness histories, AWAD/control histories, routing
counters, and final RNG state. Existing `-t` tests additionally cover donor
exclusion, forced crossover, close/far routing, degeneracy, covariance fallback,
and GPU service transport. CUDA tests skip explicitly when no device is present.
Set `CEC_SOURCE_DIR` if the reference checkout is elsewhere.

Validation result: **48 tests run, 46 passed, two CUDA tests skipped** because
the environment reports `cudaErrorNoDevice`. Actual GPU execution therefore
remains unverified here; CPU source equivalence and GPU service transport passed.
`git diff --check` passed, and local DSADE was verified byte-for-byte equal to HEAD.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cec-port-mpl \
  .venv/bin/python -m unittest \
  tests.test_cec_optimizer_port tests.test_macro_de_t tests.test_dsade_canonical \
  tests.test_sensitivity_optimizers tests.test_sensitivity_weights -v
```

To compare the four requested local names, replace only the selection block:

```python
OPTIMIZERS = [
    "DSADE",
    "MaCRO-DE",
    "MaCRO-DE-t",
    "MaCRO-DE-t-v2",
]
```

For the exact CEC DSADE comparison, use `"DSADE-CEC"` in the first slot. Active
v2 intentionally differs from the supplied CEC v2 source.
The existing report-only default also remains unchanged; an explicit
`--experiment-mode full` uses the existing experiment dispatch when a real run
is desired.

## Source SHA-256 manifest

```text
a240cb3b11dcaa4681605252e93302f0f95ce197402f8c73b6be9713b90a3691  algorithm_acronym_list.py
2fc3a1347a9abdd3ab039643f2fa3bd29c290802e6811d38cb453eb472844b21  dsade_optimizer.py
fffd33f956215812f4236945d8106e094810dd0ca290e172d8b62570a760c890  macro_de_optimizer.py
0501f80af33f3bf6d5926b45d88fa545afb261ab3c487697ca8d914c6c42f5eb  de_mc_cf_optimizer.py
cd62dbfabdebf2d3133e1fb067921c5f5d8ef26d49aea184086d1a2ac52968da  de_mc_cf_v2_optimizer.py
424907cd4c97bf84540851a45775e83583c6cc21706fd81cdce902dc4a1a6750  de_mc_optimizer.py
ae7ab5ef49eee6a5c536b56e5117ba7c6695562b9a590263e2477144a8d610f0  de_ablation_base.py
a79aa07a8a53a8a2aa7a149a09e7174fd89bd346c0e26664e33c08830f036850  compute_backend.py
```
