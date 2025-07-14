 
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
from functools import partial 
from pathlib import Path
import xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from mpi4py.futures import MPIPoolExecutor


# ---------------------------------------------------------------------------
# Globals & constants
# ---------------------------------------------------------------------------
DEFAULT_GEOMETRY = "generic"
COMPLETED_TRIALS: int = 0  # updated inside workers
GEOMETRY: str = DEFAULT_GEOMETRY  # broadcast from rank‑0 at runtime
VERBOSE: bool = False             # set from CLI

# ------------------------------------------------------------------
# CKF hyper‑parameter search space (bounds)
# ------------------------------------------------------------------
VOCS_CKF = VOCS(
    variables={
        # name                lower  upper
        "maxSeedsPerSpM": (5, 30),
        "cotThetaMax": (0.5, 3.0),
        "sigmaScattering": (0.1, 10.0),
        "radLengthPerSeed": (0.1, 4.0),
        "impactMax": (0.1, 3.0),
        "maxPtScattering": (1.0, 50.0),
        "deltaRMin": (0.0, 3.0),
        "deltaRMax": (3.0, 300.0),
    },
    objectives={"score": "MINIMIZE"},
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

    # Optimisation hyper‑parameters
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

    # File‑system options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("xopt_out"),
        metavar="DIR",
        help="Directory in which to store optimiser outputs.",
    )

    # Optimisation variable control
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

    # Reconstruction options
    parser.add_argument(
        "--geometry",
        choices=["generic", "odd"],
        default=DEFAULT_GEOMETRY,
        help="Detector geometry to reconstruct with.",
    )

    # Verbosity flag
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed per‑trial logs and diagnostics.",
    )

    return parser.parse_args(argv)

def vprint(*args, **kwargs) -> None:  # noqa: D401 – simple wrapper
    """Verbose print – emits only if global ``VERBOSE`` is *True*."""
    if VERBOSE:
        print(*args, **kwargs)

def _init_worker(verbose_flag: bool):
    """Run once in every pool worker; copy the verbosity flag."""
    global VERBOSE
    VERBOSE = verbose_flag

# ------------------------------------------------------------------
# Wrapper around ckf.py: takes a *dict* of parameters, returns *dict*
# ------------------------------------------------------------------

def evaluate_ckf(params: dict, verbose: bool = False):
    """Run a single CKF benchmark with the supplied hyper‑parameters."""

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


    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ckf_run_"))
    ckf_args = [
        f"--sf_{k}={int(round(v))}" if k == "maxSeedsPerSpM" else f"--sf_{k}={v}"
        for k, v in params.items()
    ]

    if GEOMETRY == "odd":
        ckf_args.append("--sf_minPt=1.0")
    cli = ["python", "ckf.py", "--nEvents=1",f"--geometry={GEOMETRY}",  f"--output={workdir}", *ckf_args]

    cmd = ["python", "ckf.py", "--nEvents=1",
           f"--output={workdir}", *ckf_args]

    try:
        proc = subprocess.run(
            cli, check=True, capture_output=True, text=True
        )

        timing_path = workdir / "timing.tsv"
        if timing_path.exists():
            timing = pd.read_csv(timing_path, sep="\t")
            seed   = timing.loc[
                timing["identifier"] == "Algorithm:SeedingAlgorithm",
                "time_perevent_s",
            ]
            tkfind = timing.loc[
                timing["identifier"] == "Algorithm:TrackFindingAlgorithm",
                "time_perevent_s",
            ]
            seed_time = seed.iat[0] if not seed.empty else 0.0
            ckf_time  = tkfind.iat[0] if not tkfind.empty else 0.0
            run_time  = float(seed_time + ckf_time)
        else:
            m = re.search(r"Average time per event:\s*([\d\.]+)\s*ms", proc.stdout)
            if m:
                run_time = float(m.group(1)) / 1000.0
            else:
                # fallback 2: wall-clock
                run_time = time.perf_counter() - t0
        # ---- pull metrics ----------------------------------------
        with uproot.open(workdir / "performance_seeding.root") as rh:
            eff   = rh["eff_tracks"].member("fElements")[0]
            fake  = rh["fakeratio_tracks"].member("fElements")[0]
            dup   = rh["duplicateratio_tracks"].member("fElements")[0]

        # composite score  (same as Optuna example)
        import math
        if any(math.isnan(x) or math.isinf(x) for x in [eff, fake, dup, run_time]):
            score = 1.0
        else:
            k_dup, k_time = 7, 7
            penalty = fake + dup / k_dup + run_time / k_time
            score = -(eff - penalty)   # negate → MINIMIZE

    except Exception as exc:
        if VERBOSE:
            print("[WARN] ckf.py failed:", exc, file=sys.stderr)
        return {"score": 1.0} 

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        COMPLETED_TRIALS += 1
        end_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
    if "score" in locals():        # ckf.py succeeded
        vprint(
            f"{end_stamp}  R{rank} END    trial={trial_idx}  "
            f"score={score:.3f}  eff={eff:.3f}  fake={fake:.3f}  "
            f"dup={dup:.3f}  time={run_time:.3f}s"
        )
    else:                          # we bailed out in except
        vprint(f"{end_stamp}  R{rank} FAILED trial={trial_idx}")
    return {"score": score}



def plot_score_vs_trial(df: pd.DataFrame, n_seed: int, outdir: Path) -> Path:
    """Create a PNG showing inverted score vs trial number.

    Parameters
    ----------
    df : pandas.DataFrame
        Optimisation history with at least the columns `trial` and `inv_score`.
    n_seed : int
        Number of initial random seed trials (highlighted in red).
    outdir : Path
        Directory in which to store the PNG.

    Returns
    -------
    Path
        Full path to the saved PNG file.
    """
    # Split masks for styling
    seed_mask = df["trial"] < n_seed
    success_mask = df["inv_score"] > 0.0

    plt.figure(figsize=(8, 4))
    plt.plot(df.loc[success_mask, "trial"], df.loc[success_mask, "inv_score"], marker="o", linewidth=1.5, label="successful trials")
    plt.scatter(df["trial"], df["inv_score"], zorder=4)
    plt.scatter(df.loc[seed_mask, "trial"], df.loc[seed_mask, "inv_score"], color="red", zorder=5, label="seed trials")

    plt.xlabel("Trial number")
    plt.ylabel("Inverted score (higher is better)")
    plt.title("Inverted score vs Trial number — CKF optimisation")
    plt.legend()
    plt.tight_layout()

    png_path = outdir / "score_vs_trial.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path

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

    # in main(), right after you set VERBOSE = args.verbose
    rank = MPI.COMM_WORLD.Get_rank()
    verb_flag = args.verbose if rank == 0 else None
    verb_flag = MPI.COMM_WORLD.bcast(verb_flag, root=0)

    # overwrite the module-level flag on *every* rank
    global VERBOSE
    VERBOSE = verb_flag

    # ---- share geometry string across all MPI ranks ----------------
    geom_str = args.geometry if MPI.COMM_WORLD.Get_rank() == 0 else None
    geom_str = MPI.COMM_WORLD.bcast(geom_str, root=0)

    global GEOMETRY
    GEOMETRY = geom_str

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
    )

    n_seed     = args.n_seed_trials
    n_guided   = args.n_guided_trials

    outdir: pathlib.Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    pool      = MPIPoolExecutor(initializer=_init_worker, initargs=(args.verbose,),)                    # ← ➋ NEW LINE
    evaluator = xopt.Evaluator(        
        function=evaluate_ckf,             # ←  replace old
        executor=pool
    )
    generator = UpperConfidenceBoundGenerator(
        vocs=vocs_subset,
        n_candidates=4
    )

    X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)

    # ---------- rank-0 drives the optimisation ----------
    if MPI.COMM_WORLD.Get_rank() == 0:
        X.random_evaluate(n_seed)

        # total guided evaluations = args.n_trials
        total_trials = n_seed + n_guided
        while len(X.data) < total_trials:
            X.step()
            if not VERBOSE:
                _print_progress(len(X.data), total_trials)
        # ---------- persist results --------------------------------
           # each step spawns 4 CKF runs
        csv_path = outdir / "history.csv"
        X.data.to_csv(csv_path, index=False)

        with open(outdir / "xopt_state.json", "w") as fh:
            fh.write(X.model_dump_json(indent=2))
        
        # Prepare dataframe for plotting
        df = X.data.copy().reset_index(drop=True)
        df["trial"] = df.index
        df["inv_score"] = np.where(df["score"] == 1.0, 0.0, -df["score"])

        png_path = plot_score_vs_trial(df, n_seed, outdir)

        # ---------- console summary -------------------------------
        best_row = df.loc[df["inv_score"].idxmax()]
        print(f"✅i Optimisation finished. Results in {outdir}")
        print(f"Best parameters →\n{best_row.to_dict()}")
        print(f"History saved to {csv_path}")
        print(f"Plot saved to   {png_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
