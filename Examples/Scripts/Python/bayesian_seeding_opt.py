import os
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


# Globals

VERBOSE: bool = False
SIM_DATA_DIR: Path | None = None
TTBAR_PU: int = 0
OPT_VARS: set = set()
COMPLETED_TRIALS: int = 0

# CKF hyper‑parameter search space (bounds)
VOCS_CKF = VOCS(
    variables={
        # name                lower  upper
        #geo-independent
        "maxSeedsPerSpM": (1, 7),
        "cotThetaMax": (7, 11),
        "sigmaScattering": (2, 8),
        "radLengthPerSeed": (0.005, 0.2),
        "impactMax": (3, 25.0),
        "maxPtScattering": (10.0, 40.0),
        "deltaRMin": (1.0, 15.0),
        "deltaRMax": (60.0, 300.0),
        
        #geo-sensitive
        "rMin": (0.0, 20.0),
        "rMax": (150.0, 250.0),
        "zMin": (-2500.0, -1500.0),
        "zMax": (1500.0, 2500.0),
        "collisionZMin": (-300.0, -200.0),
        "collisionZMax": (200.0, 300.0),
        "minPt": (0.3, 1),
    },
    objectives={"score": "MINIMIZE"},
)

# Define variable groups for easier reference
ALL_VARS = list(VOCS_CKF.variables.keys())
GEO_VARS = [
    "maxSeedsPerSpM", "cotThetaMax", "sigmaScattering", "radLengthPerSeed",
    "impactMax", "maxPtScattering", "deltaRMin", "deltaRMax"
]

# Hard-coded "standard" CKF parameter set
standard_point = {
    "maxSeedsPerSpM": 1,
    "cotThetaMax": 7.40627,
    "sigmaScattering": 5,
    "radLengthPerSeed": 0.1,
    "impactMax": 3.0,
    "maxPtScattering": 10.0,
    "deltaRMin": 1.0,
    "deltaRMax": 60.0,
    "rMin": 0.0,
    "rMax": 200.0,
    "zMin": -2000.0,
    "zMax": 2000.0,
    "collisionZMin": -250.0,
    "collisionZMax": 250.0,
    "minPt": 0.5,
}

# Default values used whenever a variable *is not* optimised
DEFAULT_CKF_PARAMS: Dict[str, float] = {
    "maxSeedsPerSpM": 12,
    "cotThetaMax": 1.5,
    "sigmaScattering": 2.0,
    "radLengthPerSeed": 1.0,
    "impactMax": 1.5,
    "maxPtScattering": 10.0,
    "deltaRMin": 0.5,
    "deltaRMax": 100.0,
    "rMin": 0.0,
    "rMax": 200.0,
    "zMin": -2000.0,
    "zMax": 2000.0,
    "collisionZMin": -250.0,
    "collisionZMax": 250.0,
    "minPt": 0.5,
}

# CLI helpers
def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Bayesian optimisation of seeding hyper‑parameters with Xopt.",
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
        help=(
            "'all' = all 15 variables, "
            "'geo-independent' = 7 geometry-independent variables, "
            "or a comma-separated list (choices: "
            f"{', '.join(ALL_VARS)})."
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

    parser.add_argument(
        "--ttbar-pu",
        type=int,
        default=0,
        metavar="MU",
        help="Pile-up ⟨μ⟩ to mix into the Pythia-8 tt̄bar sample",
    )

    return parser.parse_args(argv)

def get_full_chain_path():
    """Get the path to full_chain_tunable.py relative to this script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "full_chain_odd_tunable.py")

def vprint(*args, **kwargs) -> None:  # noqa: D401 – simple wrapper
    """Verbose print – emits only if global ``VERBOSE`` is *True*."""
    if VERBOSE:
        print(*args, **kwargs)

def _init_worker(verbose_flag: bool, indir_str: str, ttbar_pu: int):
    """Run once in every pool worker; copy flags sent from rank-0."""
    global VERBOSE, SIM_DATA_DIR, TTBAR_PU
    VERBOSE = verbose_flag
    SIM_DATA_DIR = Path(indir_str) if indir_str else None
    TTBAR_PU  = ttbar_pu        # <-- new
    
# Wrapper around full_chain_odd: takes a *dict* of parameters, returns *dict*
def evaluate_ckf(params: dict):
    """Run a single full chain benchmark with the supplied hyper‑parameters."""

    # safe defaults in case full_chain_odd.py crashes
    score = 1.0
    eff = fake = dup = run_time = float("nan")
    result = None

    global COMPLETED_TRIALS 

    # Get current progress from the main process (this is approximate)
    trial_idx = COMPLETED_TRIALS 

    # ---------- BEGIN banner ---------------------------------
    rank = MPI.COMM_WORLD.Get_rank()
    start_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
    vprint(f"{start_stamp}  R{rank} BEGIN  trial={trial_idx}  params={params}")

    # Fill in any parameters that were NOT optimised with defaults
    full_params = DEFAULT_CKF_PARAMS.copy()
    full_params.update(params)

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="full_chain_run_"))
    original_cwd = os.getcwd()

    try:
        # Change to working directory
        os.chdir(workdir)
        
        full_chain_args = [
            f"--sf_{k}={int(round(v))}" if k == "maxSeedsPerSpM" else f"--sf_{k}={v}"
            for k, v in full_params.items()
        ]

        # Get the path to full_chain_odd.py
        full_chain_path = get_full_chain_path()
        
        # Use sys.executable to ensure we're using the correct Python interpreter
        cli = ["python", full_chain_path, "--events=10", *full_chain_args]
        
        if SIM_DATA_DIR is not None:
            cli.append(f"--indir={SIM_DATA_DIR}")
        else:
            cli.extend(["--ttbar", f"--ttbar-pu={TTBAR_PU}"])

        t0 = time.perf_counter()
        proc = subprocess.run(
            cli, check=True, capture_output=True, text=True
        )

        # Parse performance metrics (files are now in current workdir)
        perf_file = Path("odd_output/performance_finding_ckf.root")
        with uproot.open(perf_file) as rh:
            eff = rh["eff_particles"].member("fElements")[0] * 100  # Convert to percentage
            fake = rh["fakeratio_tracks"].member("fElements")[0] * 100  # Convert to percentage
            dup = rh["duplicateratio_tracks"].member("fElements")[0] * 100  # Convert to percentage

        # Parse timing information
        timing_csv = Path("odd_output/timing.csv")
        if timing_csv.exists():
            vprint("Timing CSV found. Attempting to read...")
            timing = pd.read_csv(timing_csv)
            
            col = "identifier" if "identifier" in timing.columns else "name"

            ckf_row = timing[timing[col] == "Algorithm:TrackFindingAlgorithm"]
            seed_row = timing[timing[col] == "Algorithm:GridTripletSeedingAlgorithm"]

            if not ckf_row.empty and not seed_row.empty:
                ckf_time = ckf_row['time_perevent_s'].values[0]
                seeding_time = seed_row['time_perevent_s'].values[0] 
                run_time = ckf_time + seeding_time
            else:
                vprint("Warning: CKF or seeding timing data not found. Using fallback timing method.")
                run_time = time.perf_counter() - t0
        else:
            vprint(f"Timing CSV not found at {timing_csv}. Using fallback timing method.")
            run_time = time.perf_counter() - t0

        # Calculate score
        if any(math.isnan(x) or math.isinf(x) for x in [eff, fake, dup, run_time]):
            score = 1.0
        else:
            k = 5  # As specified in the image
            penalty = fake + dup / k + run_time / k
            score = -(eff - penalty)  # Negate for minimization

        result = {
            "score": score,
            "efficiency": eff,
            "fakerate": fake,
            "duplicaterate": dup,
            "runtime": run_time,
            "fail": 0.0
        }

    except Exception as exc:
        vprint(f"[WARN] An unexpected error occurred: {type(exc).__name__}: {exc}")
        result = {
            "score": 1.0,  # Bad score for minimization
            "efficiency": 0.0,
            "fakerate": 100.0,
            "duplicaterate": 100.0,
            "runtime": 999.0,
            "fail": 1.0
        }

    finally:
        # Return to original directory
        os.chdir(original_cwd)
        
        # Clean up working directory
        shutil.rmtree(workdir, ignore_errors=True)
        
        # Update trial counter
        COMPLETED_TRIALS += 1
        
        # Log results
        end_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
        if result and result["fail"] == 0.0:
            vprint(
                f"{end_stamp}  R{rank} END    trial={trial_idx}  "
                f"score={result['score']:.3f}  eff={result['efficiency']:.3f}  "
                f"fake={result['fakerate']:.3f}  dup={result['duplicaterate']:.3f}  "
                f"time={result['runtime']:.3f}s"
            )
        else:
            vprint(f"{end_stamp}  R{rank} FAILED trial={trial_idx}")
    
    return result

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

    global TTBAR_PU, VERBOSE, SIM_DATA_DIR, OPT_VARS, COMPLETED_TRIALS
    TTBAR_PU = args.ttbar_pu
    VERBOSE = args.verbose
    SIM_DATA_DIR = args.indir
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        if SIM_DATA_DIR is None:
            print(f"[rank-0] No input directory specified. Pythia-8 will be used with pile-up μ = {TTBAR_PU}")
        else:
            print(f"[rank-0] Using input directory: {SIM_DATA_DIR}")

    # Parse optimization variables
    if args.opt_vars.lower() == "all":
        opt_vars = ALL_VARS
    elif args.opt_vars.lower() == "geo-independent":
        opt_vars = GEO_VARS
    else:
        opt_vars = [v.strip() for v in args.opt_vars.split(",") if v.strip()]
        bad = set(opt_vars) - set(ALL_VARS)
        if bad:
            raise ValueError(f"Unknown CKF variable(s): {', '.join(sorted(bad))}")

    OPT_VARS = set(opt_vars)

    vocs_subset = VOCS(
        variables={k: VOCS_CKF.variables[k] for k in opt_vars},
        objectives=VOCS_CKF.objectives,
        constraints={"fail": ["LESS_THAN", 0.5]}
    )

    n_seed = args.n_seed_trials
    n_guided = args.n_guided_trials
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # Set up MPI pool and Xopt
    pool = MPIPoolExecutor(
        initializer=_init_worker,
        initargs=(
            args.verbose,
            str(args.indir) if args.indir else "",
            args.ttbar_pu,
        ),
        include_self=True  
    )

    evaluator = xopt.Evaluator(        
        function=evaluate_ckf,
        executor=pool,
        max_workers=MPI.COMM_WORLD.Get_size()  
    )
    generator = ExpectedImprovementGenerator(
        vocs=vocs_subset,
        n_candidates=MPI.COMM_WORLD.Get_size()  
    )

    X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)

    # Only rank-0 drives the optimization
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Create standard point with only the variables being optimized
        standard_subset = {k: standard_point[k] for k in opt_vars}
       
        X.evaluate(standard_subset)
        COMPLETED_TRIALS += 1
        X.random_evaluate(n_seed)
        COMPLETED_TRIALS += n_seed


        total_trials = n_seed + n_guided + 1  # +1 for standard point
        while len(X.data) < total_trials:
            X.step()
            if not VERBOSE:
                _print_progress(len(X.data), total_trials)

        # Prepare dataframe for CSV output
        df = X.data.copy().reset_index(drop=True)
        df["trial"] = df.index

        # Invert the score for the CSV output
        df["inv_score"] = np.where(df["score"] == 1.0, 0.0, -df["score"])

        # Ensure all metrics are present
        for metric in ["efficiency", "fakerate", "duplicaterate", "runtime"]:
            if metric not in df.columns:
                df[metric] = np.nan

        # Save to CSV
        csv_path = outdir / "history.csv"
        df.to_csv(csv_path, index=False)

        # Save Xopt state
        with open(outdir / "xopt_state.json", "w") as fh:
            fh.write(X.model_dump_json(indent=2))
        
        # Generate plots
        full_png, zoom_png = plot_score_vs_trial_dual_autozoom(df, n_seed, outdir)
        
        # Console summary
        best_row = df.loc[df["inv_score"].idxmax()] 
        print(f"✅ Optimisation finished. Results in {outdir}")
        print(f"Best parameters →\n{best_row.to_dict()}")
        print(f"History saved to {csv_path}")
        print("Plots Saved:", full_png, "and", zoom_png)

    # Close the pool
    pool.shutdown()


if __name__ == "__main__":
    main()