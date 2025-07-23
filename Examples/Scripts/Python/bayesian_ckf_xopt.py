 
"""bayesian_ckf_xopt.py
Minimal self‑contained example of running a Bayesian optimisation of the
CKF (Combinatorial Kalman Filter) hyper‑parameters with Xopt ≥ 2.4
and pydantic ≥ 2.3.

The script:

1.  Defines the search‑space (VOCS) for seven CKF seeding parameters.
2.  Wraps the ACTS CKF performance evaluator (`ckf.py`) so that it can be
    called by Xopt, receiving a dict of hyper‑parameters and returning
    a dict of objective values.
3.  Builds an Upper‑Confidence‑Bound Gaussian‑process generator
    (from BoTorch) and runs a user‑defined number of optimisation
    iterations.
4.  Stores a CSV of all evaluated points plus the generator state in
    the output directory.

Usage
-----
python bayesian_ckf_xopt.py \
       --n-trials  100 \
       --output-dir  /path/to/outdir

Optional
--------
--acts-build   Path to your ACTS build (defaults to $ACTS_BUILD
               or ~/acts/build)
--sim-data     Directory with your simulated events (defaults to cwd)
"""  # noqa: D205,D400,E501
import sys
import tempfile, shutil
import argparse
import time
import re
import json
import math
from typing import Dict, List
import subprocess
import uproot
import pathlib
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pathlib import Path
import xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from mpi4py.futures import MPIPoolExecutor


# ---------------------------------------------------------------------------
# Globals & constants
# ---------------------------------------------------------------------------
COMPLETED_TRIALS: int = 0  # updated inside workers
VERBOSE: bool = False             # set from CLI
SIM_DATA_DIR: Path | None = None    # filled in main()
# ------------------------------------------------------------------
# Pythia-8 controls (set later by main())
# ------------------------------------------------------------------
USE_P8: bool = False      # enable with -P / --pythia8
TTBAR_PU: int = 0         # pile-up μ to pass to ckf.py

# ------------------------------------------------------------------
# CKF hyper‑parameter search space (bounds)
# ------------------------------------------------------------------
VOCS_CKF = VOCS(
    variables={
        # name                lower  upper
        "maxSeedsPerSpM": (1, 10),
        "cotThetaMax": (7, 10),
        "sigmaScattering": (2, 8),
        "radLengthPerSeed": (0.001, 0.5),
        "impactMax": (0.1, 25.0),
        "maxPtScattering": (10.0, 50.0),
        "deltaRMin": (1.0, 20.0),
        "deltaRMax": (50.0, 300.0),
    },
    objectives={"score": "MINIMIZE"},
)

# ------------------------------------------------------------------
# 1.  Hard-coded "standard" CKF parameter set
# ------------------------------------------------------------------
standard_point = dict(
    maxSeedsPerSpM   = 1,
    cotThetaMax      = 7.40627,
    sigmaScattering  = 5,
    radLengthPerSeed = 0.1,
    impactMax        = 3.0,
    maxPtScattering  = 10.0,
    deltaRMin        = 1.0,
    deltaRMax        = 60.0,
)


# ------------------------------------------------------------------
# Default values used whenever a variable *is not* optimised
# ------------------------------------------------------------------
DEFAULT_CKF_PARAMS: Dict[str, float] = {
    "maxSeedsPerSpM": 12,
    "cotThetaMax": 1.5,
    "sigmaScattering": 2.0,
    "radLengthPerSeed": 1.0,
    "impactMax": 1.5,
    "maxPtScattering": 10.0,
    "deltaRMin": 0.5,
    "deltaRMax": 100.0,
}

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and return command‑line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Optional list of CLI tokens (use `None` for `sys.argv[1:]`).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian optimisation of CKF hyper‑parameters with Xopt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n-seed-trials",
        type=int,
        default=5,
        metavar="N",
        help="Number of initial random evaluations (warm‑up).",
    )
    parser.add_argument(
        "--n-guided-trials",
        type=int,
        default=50,
        metavar="N",
        help="Number of Bayesian‑optimisation iterations after the seed phase.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("xopt_out"),
        metavar="DIR",
        help="Directory in which to store optimiser outputs.",
    )

    parser.add_argument(
        "--opt-vars",
        default="all",
        metavar="VARS",
        help=(
            "Comma‑separated list of CKF variables to optimise "
            f"(choices: {', '.join(VOCS_CKF.variables.keys())}; "
            "use 'all' to optimise every variable)."
        ),
    )

    parser.add_argument(
    "--indir",
    type=Path,
    default=None,
    metavar="DIR",
    help="Directory that contains the input ROOT files for ckf.py "
         "(omit to leave ckf.py’s own default unchanged).",
    )  

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed per‑trial logs and diagnostics.",
    )

    # --- Pythia-8 controls ----------------------------------------
    parser.add_argument(
        "-P", "--pythia8",
        action="store_true",
        help="Tell ckf.py to generate events on-the-fly with Pythia-8 tt̄bar",
    )
    parser.add_argument(
        "--ttbar-pu",
        type=int,
        default=0,
        metavar="MU",
        help="Pile-up ⟨μ⟩ to mix into the Pythia-8 tt̄bar sample",
    )

    return parser.parse_args(argv)

def vprint(*args, **kwargs) -> None:  # noqa: D401 – simple wrapper
    """Verbose print – emits only if global ``VERBOSE`` is *True*."""
    if VERBOSE:
        print(*args, **kwargs)

def _init_worker(verbose_flag: bool, indir_str: str, use_p8: bool, ttbar_pu: int):
    """Run once in every pool worker; copy flags sent from rank-0."""
    global VERBOSE, SIM_DATA_DIR, USE_P8, TTBAR_PU
    VERBOSE = verbose_flag
    SIM_DATA_DIR = Path(indir_str) if indir_str else None
    USE_P8    = use_p8          # <-- new
    TTBAR_PU  = ttbar_pu        # <-- new
    

# ------------------------------------------------------------------
# Wrapper around ckf.py: takes a *dict* of parameters, returns *dict*
# ------------------------------------------------------------------

def evaluate_ckf(params: dict):
    """Run a single full chain benchmark with the supplied hyper‑parameters."""

    # safe defaults in case full_chain_odd.py crashes
    score = 1.0
    eff = fake = dup = run_time = float("nan")

    global COMPLETED_TRIALS 

    # Generate a sequential index (best-effort) for logging
    trial_idx = COMPLETED_TRIALS + 1

    # ---------- BEGIN banner ---------------------------------
    rank        = MPI.COMM_WORLD.Get_rank()
    start_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
    vprint(f"{start_stamp}  R{rank} BEGIN  trial={trial_idx}  params={params}")

    # Fill in any parameters that were NOT optimised with defaults
    full_params = DEFAULT_CKF_PARAMS.copy()
    full_params.update(params)

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="full_chain_run_"))
    full_chain_args = [
        f"--sf_{k}={int(round(v))}" if k == "maxSeedsPerSpM" else f"--sf_{k}={v}"
        for k, v in full_params.items()
    ]

    cli = ["python", "Examples/Scripts/Python/full_chain_odd.py", "--events=1", *full_chain_args]

    if SIM_DATA_DIR is not None:
        cli.append(f"--indir={SIM_DATA_DIR}")  

    if USE_P8:
        cli += ["--ttbar", f"--ttbar-pu={TTBAR_PU}"]

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            cli, check=True, capture_output=True, text=True
        )

        # Parse performance metrics
        with uproot.open("odd_output/performance_finding_ckf.root") as rh:
            eff = rh["eff_particles"].member("fElements")[0] * 100  # Convert to percentage
            fake = rh["fakeratio_tracks"].member("fElements")[0] * 100  # Convert to percentage
            dup = rh["duplicateratio_tracks"].member("fElements")[0] * 100  # Convert to percentage

        # Parse timing information
        timing_csv = Path("odd_output/timing.csv")
        if timing_csv.exists():
            print("Timing CSV found. Attempting to read...")
            timing = pd.read_csv(timing_csv)
            #print(f"CSV read successfully. Columns: {timing.columns}")
            #print(f"First few rows:\n{timing.head()}")
            
            col = "identifier" if "identifier" in timing.columns else "name"
            #print(f"Using column '{col}' for identification")

            ckf_row = timing[timing[col] == "Algorithm:TrackFindingAlgorithm"]
            seed_row = timing[timing[col] == "Algorithm:GridTripletSeedingAlgorithm"]

            #print(f"CKF row: {ckf_row}")
            #print(f"Seed row: {seed_row}")

            if not ckf_row.empty and not seed_row.empty:
                ckf_time = ckf_row['time_perevent_s'].values[0]
                seeding_time = seed_row['time_perevent_s'].values[0] 
                run_time = ckf_time + seeding_time
                #print(f"Calculated run_time: {run_time}")
            else:
                print("Warning: CKF or seeding timing data not found. Using fallback timing method.")
                run_time = time.perf_counter() - t0
        else:
            print(f"Timing CSV not found at {timing_csv}. Using fallback timing method.")
            run_time = time.perf_counter() - t0


        # Calculate score
        if any(math.isnan(x) or math.isinf(x) for x in [eff, fake, dup, run_time]):
            score = 1.0
        else:
            k = 5  # As specified in the image
            penalty = fake + dup / k + run_time / k
            score = -(eff - penalty)  # Negate for minimization

    except Exception as exc:
        if VERBOSE:
            print(f"[WARN] An unexpected error occurred: {type(exc).__name__}: {exc}")
        return {
            "score": np.nan,
            "efficiency": np.nan,
            "fakerate": np.nan,
            "duplicaterate": np.nan,
            "runtime": np.nan,
            "fail": 1.0
        }

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        COMPLETED_TRIALS += 1
        end_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
        if "score" in locals():        # full_chain_odd.py succeeded
            vprint(
                f"{end_stamp}  R{rank} END    trial={trial_idx}  "
                f"score={score:.3f}  eff={eff:.3f}  fake={fake:.3f}  "
                f"dup={dup:.3f}  time={run_time:.3f}s"
            )
        else:                          # we bailed out in except
            vprint(f"{end_stamp}  R{rank} FAILED trial={trial_idx}")
    
        return {
            "score": score,
            "efficiency": eff,
            "fakerate": fake,
            "duplicaterate": dup,
            "runtime": run_time,
            "fail": 0.0
        }

def plot_score_vs_trial_dual_autozoom(
    df: pd.DataFrame,
    n_seed: int,
    outdir: Path,
    *,
    keep_pct: float = 0.96,          # fraction of non-zero scores to keep in frame
    overview_fname: str = "score_vs_trial_full.png",
    zoom_fname: str = "score_vs_trial_zoom.png",
) -> tuple[Path, Path]:
    """
    Save two plots:
    (A) overview — full y–range
    (B) zoomed — automatically chosen so that
        • y-min excludes all zeros (failures)
        • y-range keeps ~`keep_pct` of positive scores

    Returns
    -------
    (Path, Path)
        Paths to the overview PNG and the zoomed PNG.
    """
    outdir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ masks
    seed_mask   = df["trial"] < n_seed
    guided_mask = ~seed_mask
    best_so_far = df["inv_score"].cummax()

    # ------------------------------------------------------------ un-zoomed
    plt.figure(figsize=(8, 4))
    plt.scatter(df.loc[seed_mask,   "trial"], df.loc[seed_mask,   "inv_score"],
                marker="o", label="seed trials",   zorder=3)
    plt.scatter(df.loc[guided_mask, "trial"], df.loc[guided_mask, "inv_score"],
                marker="x", label="guided trials", zorder=2)
    plt.xlabel("Trial number")
    plt.ylabel("Inverted score (higher is better)")
    plt.title("Inverted score vs Trial number — overview")
    plt.legend()
    plt.tight_layout()
    overview_path = outdir / overview_fname
    plt.savefig(overview_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------- auto-zoom
    # 1. positive (non-zero) scores only
    pos_scores = df.loc[df["inv_score"] > 0, "inv_score"]
    if pos_scores.empty:
        # nothing to zoom on → just reuse overview range
        zoom_low, zoom_high = df["inv_score"].min(), df["inv_score"].max()
    else:
        # 2. keep central `keep_pct` of the distribution
        lo_q = (1 - keep_pct) / 2          # e.g. 0.02 for 96 %
        hi_q = 1 - lo_q                    # e.g. 0.98
        zoom_low = pos_scores.quantile(lo_q)
        zoom_high = pos_scores.quantile(hi_q)

        # add ±5 % padding
        pad = 0.05 * (zoom_high - zoom_low)
        zoom_low = max(zoom_low - pad, 0.0)
        zoom_high += pad

    best_val   = best_so_far.max()
    best_trial = df.loc[df["inv_score"] == best_val, "trial"].iloc[0]

    plt.figure(figsize=(8, 4))
    plt.scatter(df.loc[seed_mask,   "trial"], df.loc[seed_mask,   "inv_score"],
                marker="o", label="seed trials",   zorder=3)
    plt.scatter(df.loc[guided_mask, "trial"], df.loc[guided_mask, "inv_score"],
                marker="x", label="guided trials", zorder=2)
    plt.step(df["trial"], best_so_far, where="post",
             linestyle="--", linewidth=1.2, label="running best", zorder=4)
    plt.annotate(f"Best: {best_val:.4f} (trial {best_trial})",
                 xy=(best_trial, best_val),
                 xytext=(0.02, 0.97), textcoords="axes fraction",
                 va="top", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                 arrowprops=dict(arrowstyle="->", lw=1.0))
    plt.xlabel("Trial number")
    plt.ylabel("Inverted score (higher is better)")
    plt.title("Inverted score vs Trial number — zoomed (auto)")
    plt.ylim(zoom_low, zoom_high)
    plt.legend()
    plt.tight_layout()
    zoom_path = outdir / zoom_fname
    plt.savefig(zoom_path, dpi=150)
    plt.close()

    return overview_path, zoom_path

def _print_progress(current: int, total: int, width: int = 40) -> None:
    """Render a one-line text progress bar like  [#####-----] 12/50."""
    fraction = current / total
    filled   = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    # \r returns cursor to line start; flush=True forces immediate update
    print(f"\r[{bar}] {current}/{total} trials", end="", flush=True)
    if current == total:
        print()   # newline when done

# ------------------------------------------------------------------
# Main optimisation loop
# ------------------------------------------------------------------
def main(argv=None):
    
    args = parse_cli_args(argv) 

    global USE_P8, TTBAR_PU
    USE_P8  = args.pythia8
    TTBAR_PU = args.ttbar_pu

    if MPI.COMM_WORLD.Get_rank() == 0:
        if USE_P8:
            print(f"[rank-0] Pythia-8 ON → μ = {TTBAR_PU}")
        else:
            print("[rank-0] Pythia-8 OFF (ckf.py will look for particles.root)")

    # How many MPI ranks did the user launch?
    world_size = MPI.COMM_WORLD.Get_size()        # 1 in serial, N under mpiexec

    # Will the root rank (0) also run CKF jobs?
    root_is_worker = True   # stays True unless you set include_self=False below

    pool_size = world_size if root_is_worker else max(1, world_size - 1)
    
    # --- copy rank-0 flags into globals ---
    global VERBOSE, SIM_DATA_DIR, GEOMETRY
    VERBOSE = args.verbose
    SIM_DATA_DIR = args.indir           # may be None

    if args.opt_vars.lower() == "all":
        opt_vars = list(VOCS_CKF.variables.keys())
    else:
        opt_vars = [v.strip() for v in args.opt_vars.split(",") if v.strip()]
        bad = set(opt_vars) - VOCS_CKF.variables.keys()
        if bad:
            ap.error(f"Unknown CKF variable(s): {', '.join(sorted(bad))}")

    global OPT_VARS
    OPT_VARS = set(opt_vars)  # make visible to evaluate_ckf

    vocs_subset = VOCS(
        variables={k: VOCS_CKF.variables[k] for k in opt_vars},
        objectives=VOCS_CKF.objectives,
        constraints = {"fail": ["LESS_THAN", 0.5]}
    )

    n_seed     = args.n_seed_trials
    n_guided   = args.n_guided_trials

    outdir: pathlib.Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # -- main()  (remove the old broadcasts) ----------------------------
    pool = MPIPoolExecutor(
        initializer=_init_worker,
        initargs=(
            args.verbose,
            str(args.indir) if args.indir else "",
            args.pythia8,      # new
            args.ttbar_pu,     # new
        ),
     include_self=True  
    )

    evaluator = xopt.Evaluator(        
        function=evaluate_ckf,             # ←  replace old
        executor=pool,
        max_workers=MPI.COMM_WORLD.Get_size()  
    )
    generator = ExpectedImprovementGenerator(
        vocs=vocs_subset,
        n_candidates=MPI.COMM_WORLD.Get_size()  
    )

    X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)

    # ---------- rank-0 drives the optimisation ----------
    if MPI.COMM_WORLD.Get_rank() == 0:
        X.evaluate(standard_point) 
        X.random_evaluate(n_seed)

        # total guided evaluations = args.n_trials
        total_trials = n_seed + n_guided
        while len(X.data) < total_trials:
            X.step()
            if not VERBOSE:
                _print_progress(len(X.data), total_trials)
        # ---------- persist results --------------------------------
           # each step spawns 4 CKF runs
        # Prepare dataframe for CSV output
        df = X.data.copy().reset_index(drop=True)
        df["trial"] = df.index

        # Invert the score for the CSV output, maintaining the existing logic
        df["inv_score"] = np.where(df["score"] == 1.0, 0.0, -df["score"])

        # Ensure all metrics are present
        for metric in ["efficiency", "fakerate", "duplicaterate", "runtime"]:
            if metric not in df.columns:
                df[metric] = np.nan

        # Reorder columns to match Optuna-like output
        column_order = (
            ["trial"] +
            list(VOCS_CKF.variables.keys()) +
            ["efficiency", "fakerate", "duplicaterate", "runtime", "inv_score"]
        )
        df = df[column_order]

        # Save to CSV
        csv_path = outdir / "history.csv"
        df.to_csv(csv_path, index=False)

        with open(outdir / "xopt_state.json", "w") as fh:
            fh.write(X.model_dump_json(indent=2))
        
        # Prepare dataframe for plotting
        df = X.data.copy().reset_index(drop=True)
        df["trial"] = df.index
        df["inv_score"] = np.where(df["score"] == 1.0, 0.0, -df["score"])

        full_png, zoom_png = plot_score_vs_trial_dual_autozoom(df, n_seed, outdir)
        # ---------- console summary -------------------------------
        best_row = df.loc[df["inv_score"].idxmax()] 
        print(f"✅ Optimisation finished. Results in {outdir}")
        print(f"Best parameters →\n{best_row.to_dict()}")
        print(f"History saved to {csv_path}")
        print("Plots Saved:", full_png, "and", zoom_png)

if __name__ == "__main__":  # pragma: no cover
    main()