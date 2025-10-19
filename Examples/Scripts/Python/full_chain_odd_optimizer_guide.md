# `full_chain_odd_optimizer.py` User Guide

This document explains how to run the CKF Bayesian optimizer, what each configuration knob controls, and how to adapt the workflow to alternate detector geometries.

---

## 1. Environment Prerequisites

1. Load the ACTS toolchain from the LCG view:
   ```bash
   lsetup "views LCG_107 x86_64-el9-gcc13-opt"
   ```
2. Activate (or create) the overlay virtual environment that carries the extra Python packages listed in `venv_requirements.txt`:
   ```bash
   python3 -m venv ~/acts-opt          # run once
   source ~/acts-opt/bin/activate
   pip install -r venv_requirements.txt
   ```
   If the venv already exists, just `source` it after the `lsetup` step.
3. Execute all commands from the ACTS repository root (the directory containing `Examples/`).

---

## 2. Basic Usage

```bash
python3 Examples/Scripts/Python/full_chain_optimizer_fixed.py \
  --events-per-trial=1 \
  --n-seed-trials=2 \
  --n-guided-trials=2 \
  --output-dir=my_results
```

- Random **seed trials** explore parameter space uniformly; **guided trials** use Bayesian optimization.
- Outputs (CSV, JSON, plots) are written to the directory passed via `--output-dir`.
- The console prints the best parameter set at completion.

---

## 3. Key CLI Options

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--opt-vars` | Variables to optimize (`all`, `geo-independent`, or comma-separated list) | `all` |
| `--multi-objective` | Switch to MOBO (Pareto) optimization | Off |
| `--n-seed-trials` | Random seed trials (resume default = 0) | 10 fresh / 0 resume |
| `--n-guided-trials` | Bayesian-guided trials (resume default = 0) | 40 fresh / 0 resume |
| `--resume-from PATH` | Continue from an existing results directory | None |
| `--ttbar-pu` | Pile-up passed to the wrapped full-chain | 200 |
| `--k-value` | K factor in the single-objective score | 5.0 |
| `--events-per-trial` | Events simulated per optimizer evaluation | 10 |
| `--verbose` | Emit detailed per-trial logs | Off |

**Resume behaviour:** When `--resume-from` is provided without overriding `--n-seed-trials` or `--n-guided-trials`, no additional trials run (`0/0`). Supply explicit counts to extend a previous campaign.

---

## 4. CKF Parameter Space

The optimizer exposes 15 CKF seeding knobs. The preset `--opt-vars=geo-independent` activates only the first block; `--opt-vars=all` (default) or a custom list can include every entry.

**Geometry-independent variables**

| Name | Range | Description |
| ---- | ----- | ----------- |
| `maxSeedsPerSpM` | 1–7 (integer) | Seeds per space-point multiplet |
| `cotThetaMax` | 7–11 | Maximum polar-angle slope |
| `sigmaScattering` | 2–8 | Multiple-scattering uncertainty scale |
| `radLengthPerSeed` | 0.005–0.2 | Material budget per seed |
| `impactMax` | 3–25 | Transverse impact-parameter cut |
| `maxPtScattering` | 10–40 | Momentum scale for scattering estimation |
| `deltaRMin` | 1–15 | Inner radial search window (mm) |
| `deltaRMax` | 60–300 | Outer radial search window (mm) |

**Geometry-sensitive additions**

| Name | Range | Description |
| ---- | ----- | ----------- |
| `rMin` / `rMax` | 0–20 / 150–250 | Cylindrical ROI bounds (mm) |
| `zMin` / `zMax` | -2500 – -1500 / 1500 – 2500 | Longitudinal ROI bounds (mm) |
| `collisionZMin` / `collisionZMax` | -300 – -200 / 200 – 300 | Collision-vertex window (mm) |
| `minPt` | 0.3–1.0 | Minimum transverse momentum (GeV) |

Parameters not actively optimized revert to the defaults in `DEFAULT_CKF_PARAMS`.

---

## 5. Optimization Modes & Scoring

By default the script performs **single-objective** optimization. Each trial evaluates:

\[
score = -efficiency - \{fakerate} + duplicaterate/K + runtime/K
\]

- `K` (set via `--k-value`, default 5.0) trades off duplication and runtime penalties against efficiency.  
- The optimizer minimizes the score: higher efficiency and lower penalties yield smaller values.  
- Metrics are read from `odd_output/performance_finding_ckf.root` and `odd_output/timing.csv`; runtime falls back to wall-clock timing if no CSV data is found.

For **multi-objective** runs (`--multi-objective`), Xopt minimizes the vector (`-efficiency`, `fakerate`, `duplicaterate`, `runtime`) and writes a Pareto surface to `pareto_optimal_solutions.csv`.

---

## 6. Suggested Test Workflow

```bash
# Smoke test
python3 .../full_chain_optimizer_fixed.py --events-per-trial=1 --n-seed-trials=1 --n-guided-trials=1 --output-dir=out_smoke

# Resume (adds additional trials)
python3 .../full_chain_optimizer_fixed.py --resume-from=out_smoke --n-seed-trials=1 --n-guided-trials=1

# Multi-objective sample
python3 .../full_chain_optimizer_fixed.py --multi-objective --events-per-trial=1 --n-seed-trials=1 --n-guided-trials=1 --output-dir=out_multi
```

Inspect `optimization_history.csv`, `pareto_optimal_solutions.csv` (if MOBO), and the generated plots to validate the outcomes.

---

## 7. Adapting to a Different Geometry (`full_chain_theirgeometry.py`)

1. **Clone the script** (optional but recommended):
   ```bash
   cp Examples/Scripts/Python/full_chain_optimizer_fixed.py \
      Examples/Scripts/Python/full_chain_theirgeometry_optimizer.py
   ```
2. **Update `get_full_chain_path()`** to point to your geometry driver (e.g. `full_chain_theirgeometry.py`) and adjust the fallback search paths.
3. **Review defaults** such as `--ttbar-pu`, `DEFAULT_CKF_PARAMS`, and `VOCS_CKF` bounds to match the new setup.
4. **Ensure argument compatibility:** the wrapped geometry script must accept the `--sf_*` parameters generated in `evaluate_ckf_*`.
5. **Run a smoke test** (`--events-per-trial=1 --n-seed-trials=1 --n-guided-trials=1`) to verify the new chain before launching longer campaigns.

---

## 8. Failure Handling

- Exceptions inside a trial yield a fallback result (`fail=1`, pessimistic metrics) so the optimizer can continue.
- The failing command and exception message are printed to the console to aid debugging.
- Temporary working directories are always removed.

---

## 9. Contact

For questions or issues, reach out to chancel@andrew.cmu.edu or the maintainers of `Examples/Scripts/Python`.

Happy optimizing!
