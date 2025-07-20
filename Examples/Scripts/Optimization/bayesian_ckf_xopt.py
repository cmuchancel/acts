 
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
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from mpi4py.futures import MPIPoolExecutor
import multiprocessing as mp 


# ---------------------------------------------------------------------------
# Globals & constants
# ---------------------------------------------------------------------------
DEFAULT_GEOMETRY = "generic"
COMPLETED_TRIALS: int = 0  # updated inside workers
GEOMETRY: str = DEFAULT_GEOMETRY  # broadcast from rank‑0 at runtime
VERBOSE: bool = False             # set from CLI
EVENT_Q: mp.Queue | None = None

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
    "maxSeedsPerSpM": 1.0,
    "cotThetaMax": 7.40627,
    "sigmaScattering": 5.0,
    "radLengthPerSeed": 0.1,
    "impactMax": 3.0,
    "maxPtScattering": 10.0,
    "deltaRMin": 1.0,
    "deltaRMax": 60.0,
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
        "--geometry",
        choices=["generic", "odd"],
        default=DEFAULT_GEOMETRY,
        help="Detector geometry to reconstruct with.",
    )

    parser.add_argument(
        "--sim-data", type=Path, default=Path("."),
        help="Directory that contains particles.root",
    )

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

def _init_worker(verbose_flag: bool, geometry: str):
    """Run once in every pool worker; copy flags sent from rank-0."""
    global VERBOSE,  GEOMETRY
    VERBOSE = verbose_flag
    GEOMETRY = geometry

# ------------------------------------------------------------------
# Wrapper around ckf.py: takes a *dict* of parameters, returns *dict*
# ------------------------------------------------------------------

METRICS_LOG = Path("metrics_sidecar.csv")


def evaluate_ckf(params: dict):
    """Run a single CKF benchmark with the supplied hyper‑parameters."""
    idx = EVENT_Q.get() 
    # safe defaults in case ckf.py crashes
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


    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ckf_run_"))
    ckf_args = [
        f"--sf_{k}={int(round(v))}" if k == "maxSeedsPerSpM" else f"--sf_{k}={v}"
        for k, v in params.items()
    ]

    if GEOMETRY == "odd":
        ckf_args.append("--sf_minPt=1.0")
    cli = ["python", "ckf.py", "--nEvents=1",f"--geometry={GEOMETRY}",  f"--output={workdir}",         f"--event-number={idx}",  *ckf_args]

    if SIM_DIR is not None:
        cli.insert(3, f"--indir={SIM_DIR}")  

    try:

        t0 = time.perf_counter()
        
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

        # ----- NEW: append one line to the side-car CSV -----------------
        # all ranks can open in append mode safely (each write ≤4096 B on POSIX)
        with METRICS_LOG.open("a") as fh:
            fh.write(f"{COMPLETED_TRIALS},{eff},{fake},{dup},{run_time}\n")
    # ----------------------------------------------------------------

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

# ------------------------------------------------------------------
# Convenience:  1-liner for any metric -----------------------------
# ------------------------------------------------------------------
def _plot_metric_vs_trial(
    df: pd.DataFrame,
    metric: str,
    n_seed: int,
    outdir: Path,
    *,
    fname: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    zoom: tuple[float, float] | None = None,   # (ymin,ymax)  – omit for full range
) -> Path:
    """
    Scatter metric (y) vs trial (x).  Colours: red = seed trials, blue = guided.

    Parameters
    ----------
    df : pd.DataFrame   – expects 'trial' and `<metric>` columns already present
    metric : str        – one of {'eff', 'fake', 'dup', 'run_time'}
    n_seed : int        – number of seed trials
    outdir : Path
    fname  : str | None – filename (PNG).  Defaults to f'{metric}_vs_trial.png'
    y_label, title : str | None – labels to override sensible defaults
    zoom   : (float,float)|None – y-range override (min,max)
    """
    outdir.mkdir(exist_ok=True)

    if fname is None:
        fname = f"{metric}_vs_trial.png"
    if y_label is None:
        y_label = metric.replace("_", " ").capitalize()
    if title is None:
        title = f"{y_label} vs Trial number"

    seed_mask   = df["trial"] < n_seed

    plt.figure(figsize=(8, 4))
    plt.scatter(df.loc[seed_mask,   "trial"], df.loc[seed_mask,   metric],
                marker="o",  color="red",  label="seed trials",   zorder=3)
    plt.scatter(df.loc[~seed_mask,  "trial"], df.loc[~seed_mask,  metric],
                marker="x",  color="royalblue", label="guided trials", zorder=2)
    plt.xlabel("Trial number")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    if zoom is not None:
        plt.ylim(*zoom)
    plt.tight_layout()
    path = outdir / fname
    plt.savefig(path, dpi=150)
    plt.close()
    return path

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

    global EVENT_Q, SIM_DIR
    
    args = parse_cli_args(argv) 

    # How many MPI ranks did the user launch?
    world_size = MPI.COMM_WORLD.Get_size()        # 1 in serial, N under mpiexec

    # Will the root rank (0) also run CKF jobs?
    root_is_worker = True   # stays True unless you set include_self=False below

    pool_size = world_size if root_is_worker else max(1, world_size - 1)
    
    # --- copy rank-0 flags into globals ---
    global VERBOSE,  GEOMETRY
    VERBOSE = args.verbose           # may be None
    GEOMETRY = args.geometry

    # initialise event queue ------------------------------------------------
    SIM_DIR = args.sim_data                             # >>> ADD >>> 7/8
    n_evt   = uproot.open(SIM_DIR / "particles.root")["events"].num_entries
    EVENT_Q = mp.Queue()
    for i in range(n_evt):
        EVENT_Q.put(i) 

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

    # -- main()  (remove the old broadcasts) ----------------------------
    pool = MPIPoolExecutor(
        initializer=_init_worker,
        initargs=(
            args.verbose,
            args.geometry,
        ),
     include_self=True  
    )

    evaluator = xopt.Evaluator(        
        function=evaluate_ckf,             # ←  replace old
        executor=pool,
        max_workers=MPI.COMM_WORLD.Get_size()  
    )
    generator = UpperConfidenceBoundGenerator(
        vocs=vocs_subset,
        n_candidates=MPI.COMM_WORLD.Get_size()  
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

        full_png, zoom_png = plot_score_vs_trial_dual_autozoom(df, n_seed, outdir)

        # --------------------------------------------
        # Load the side-car metrics and merge on trial
        # --------------------------------------------
        ##metrics_df = pd.read_csv(
        ##    METRICS_LOG,
        ##    names=["trial", "eff", "fake", "dup", "run_time"]
        ##)

        ##df = X.data.copy().reset_index(drop=True)
        ##df["trial"] = df.index                 # keep as 0 … N-1
        ##df = df.merge(metrics_df, on="trial")

       ##for m in ("eff", "fake", "dup", "run_time"):
            ##_plot_metric_vs_trial(df, m, n_seed, outdir)

        # ---------- console summary -------------------------------
        best_row = df.loc[df["inv_score"].idxmax()]
        print(f"✅ Optimisation finished. Results in {outdir}")
        print(f"Best parameters →\n{best_row.to_dict()}")
        print(f"History saved to {csv_path}")
        print("Plots Saved:", full_png, "and", zoom_png)

if __name__ == "__main__":  # pragma: no cover
    main()
