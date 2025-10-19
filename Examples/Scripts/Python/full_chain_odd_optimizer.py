import sys
import tempfile, shutil
import argparse
import time
import re
import json
import math
from typing import Dict
import subprocess
import uproot
import pathlib
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.mobo import MOBOGenerator
import os

COMPLETED_TRIALS: int = 0  
VERBOSE: bool = False        
TTBAR_PU: int = 0
K_VALUE: float = 5.0
EVENTS_PER_TRIAL: int = 10
OPT_VARS: set = set()

# CKF hyper‑parameter search space (bounds)
VOCS_CKF = VOCS(
    variables={
        # name                lower  upper
        # geo-independent variables
        "maxSeedsPerSpM": (1, 7),
        "cotThetaMax": (7, 11),
        "sigmaScattering": (2, 8),
        "radLengthPerSeed": (0.005, 0.2),
        "impactMax": (3, 25.0),
        "maxPtScattering": (10.0, 40.0),
        "deltaRMin": (1.0, 15.0),
        "deltaRMax": (60.0, 300.0),
        
        # geo-sensitive variables
        "rMin": (0.0, 20.0),
        "rMax": (150.0, 250.0),
        "zMin": (-2500.0, -1500.0),
        "zMax": (1500.0, 2500.0),
        "collisionZMin": (-300.0, -200.0),
        "collisionZMax": (200.0, 300.0),
        "minPt": (0.3, 1),
    },
    objectives={
        "efficiency": "MAXIMIZE",
        "fakerate": "MINIMIZE",
        "duplicaterate": "MINIMIZE",
        "runtime": "MINIMIZE"
    },
)


# Single-objective version for backwards compatibility
VOCS_CKF_SINGLE = VOCS(
    variables=VOCS_CKF.variables,
    objectives={"score": "MINIMIZE"},
    constraints={"fail": ["LESS_THAN", 0.5]}
)

# Default CKF parameter values
DEFAULT_CKF_PARAMS = {
    # geo-independent
    "maxSeedsPerSpM": 1,
    "cotThetaMax": 7.40627,
    "sigmaScattering": 5,
    "radLengthPerSeed": 0.1,
    "impactMax": 3.0,
    "maxPtScattering": 10.0,
    "deltaRMin": 1.0,
    "deltaRMax": 60.0,
    
    # geo-sensitive
    "rMin": 0.0,
    "rMax": 200.0,
    "zMin": (-2000.0),
    "zMax": 2000.0,
    "collisionZMin": (-250.0),
    "collisionZMax": 250.0,
    "minPt": 0.5,
}

# Predefined variable sets
ALL_VARS = list(VOCS_CKF.variables.keys())
GEO_VARS = ["maxSeedsPerSpM", "cotThetaMax", "sigmaScattering", "radLengthPerSeed", 
           "impactMax", "maxPtScattering", "deltaRMin", "deltaRMax"]


def vprint(*args, **kwargs):
    """Verbose print: only print if VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)

def parse_cli_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian optimization of CKF seeding parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core optimization parameters
    parser.add_argument(
        "--opt-vars", 
        type=str, 
        default="all",
        help=(
            "'all' = all 15 variables, "
            "'geo-independent' = 8 geometry-independent variables, "
            "or a comma-separated list (choices: "
            f"{', '.join(ALL_VARS)})."
        )
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume optimization from existing results directory"
    )

    parser.add_argument(
        "--n-seed-trials", 
        type=int, 
        default=10,
        help="Number of random seed trials (when resuming: additional trials to run)"
    )
    parser.add_argument(
        "--n-guided-trials", 
        type=int, 
        default=40,
        help="Number of Bayesian-guided trials (when resuming: additional trials to run)"
    )
    
    parser.add_argument(
        "--multi-objective",
        action="store_true",
        help="Use multi-objective optimization (MOBO) instead of single-objective"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("bayesian_ckf_results"),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--ttbar-pu", 
        type=int, 
        default=200,
        help="Pile-up level for ttbar events (when using Pythia8)"
    )
    
    parser.add_argument(
        "--k-value",
        type=float,
        default=5.0,
        metavar="K",
        help="K value for score calculation: score = -(eff - (fake + dup/K + time/K))",
    )

    parser.add_argument(
        "--events-per-trial",
        type=int,
        default=10,
        metavar="N",
        help="Number of events to simulate per trial",
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args(argv)

def get_full_chain_path():
    """Get the path to full_chain_odd.py script."""
    script_dir = pathlib.Path(__file__).parent.absolute()
    full_chain_path = script_dir / "full_chain_odd_tunable.py"
    
    if not full_chain_path.exists():
        # Try alternative locations
        alt_paths = [
            script_dir.parent / "Examples" / "Scripts" / "Python" / "full_chain_odd_tunable.py",
            pathlib.Path.cwd() / "full_chain_odd_tunable.py",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                full_chain_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"full_chain_odd_tunable.py not found in {script_dir} or alternative locations. "
                "Please ensure the script is in the correct location."
            )
    
    return str(full_chain_path)

def evaluate_ckf_single_objective(params: Dict[str, float]) -> Dict[str, float]:
    """
    Single-objective evaluation function that returns a combined score.
    This maintains backwards compatibility with the original single-objective approach.
    """
    global COMPLETED_TRIALS
    
    rank = 0
    trial_idx = COMPLETED_TRIALS + 1
    start_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
    
    vprint(f"{start_stamp}  R{rank} START  trial={trial_idx}")
    vprint(f"{start_stamp}  R{rank} PARAMS trial={trial_idx}  {params}")

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
        cli = [sys.executable, full_chain_path, f"--events={EVENTS_PER_TRIAL}", "--ttbar", f"--ttbar-pu={TTBAR_PU}", *full_chain_args]

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
            k = K_VALUE
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
        os.chdir(original_cwd)
        shutil.rmtree(workdir, ignore_errors=True)
        COMPLETED_TRIALS += 1
        end_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
        if result["efficiency"] > 0:  # Success case
            vprint(
                f"{end_stamp}  R{rank} END    trial={trial_idx}  "
                f"eff={result['efficiency']:.3f}  "
                f"fake={result['fakerate']:.3f}  dup={result['duplicaterate']:.3f}  "
                f"time={result['runtime']:.3f}s  score={result['score']:.3f}"
            )
        else:  # Failure case
            vprint(f"{end_stamp}  R{rank} FAILED trial={trial_idx}")

    return result

def evaluate_ckf_multi_objective(params: Dict[str, float]) -> Dict[str, float]:
    """
    Multi-objective evaluation function that returns individual metrics.
    This is for the multi-objective optimization approach.
    """
    global COMPLETED_TRIALS
    
    rank = 0
    trial_idx = COMPLETED_TRIALS + 1
    start_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
    
    vprint(f"{start_stamp}  R{rank} START  trial={trial_idx}")
    vprint(f"{start_stamp}  R{rank} PARAMS trial={trial_idx}  {params}")

    # Fill in any parameters that were NOT optimised with defaults
    full_params = DEFAULT_CKF_PARAMS.copy()
    full_params.update(params)

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="full_chain_run_"))
    original_cwd = os.getcwd()

    result = {
        "efficiency": 0.0,
        "fakerate": 100.0,
        "duplicaterate": 100.0,
        "runtime": 999.0,
    }

    try:
        # Change to working directory
        os.chdir(workdir)
        
        full_chain_args = [
            f"--sf_{k}={int(round(v))}" if k == "maxSeedsPerSpM" else f"--sf_{k}={v}"
            for k, v in full_params.items()
        ]

        # Get the path to full_chain_odd.py
        full_chain_path = get_full_chain_path()
        
        cli = [sys.executable, full_chain_path, f"--events={EVENTS_PER_TRIAL}", "--ttbar", f"--ttbar-pu={TTBAR_PU}", *full_chain_args]

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

        result = {
            "efficiency": eff,
            "fakerate": fake,
            "duplicaterate": dup,
            "runtime": run_time,
        }

    except Exception as exc:
        if VERBOSE:
            print(f"[WARN] An unexpected error occurred: {type(exc).__name__}: {exc}")

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(workdir, ignore_errors=True)
        COMPLETED_TRIALS += 1
        end_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
        if result["efficiency"] > 0:  # Success case
            vprint(
                f"{end_stamp}  R{rank} END    trial={trial_idx}  "
                f"eff={result['efficiency']:.3f}  "
                f"fake={result['fakerate']:.3f}  dup={result['duplicaterate']:.3f}  "
                f"time={result['runtime']:.3f}s"
            )
        else:  # Failure case
            vprint(f"{end_stamp}  R{rank} FAILED trial={trial_idx}")
    return result

def plot_score_vs_trial_dual_autozoom(df, n_seed, outdir):
    """Generate plots for single-objective optimization results."""
    # Create the dual plot with auto-zoom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full plot
    trials = df.index + 1
    ax1.scatter(trials[:n_seed], df["inv_score"][:n_seed], 
               color="red", alpha=0.7, s=50, label="Random seed")
    ax1.scatter(trials[n_seed:], df["inv_score"][n_seed:], 
               color="blue", alpha=0.7, s=50, label="Bayesian-guided")
    ax1.set_xlabel("Trial number")
    ax1.set_ylabel("Inverse score (higher = better)")
    ax1.set_title("CKF Parameter Optimization Progress (Full)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Auto-zoom plot (focus on Bayesian trials)
    if len(df) > n_seed:
        zoom_trials = trials[n_seed:]
        zoom_scores = df["inv_score"][n_seed:]
        ax2.scatter(zoom_trials, zoom_scores, color="blue", alpha=0.7, s=50)
        ax2.set_xlabel("Trial number")
        ax2.set_ylabel("Inverse score (higher = better)")
        ax2.set_title("CKF Parameter Optimization Progress (Bayesian-guided)")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No Bayesian trials yet", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Bayesian-guided trials (none yet)")
    
    plt.tight_layout()
    
    # Save plots
    full_png = outdir / "optimization_progress_full.png"
    zoom_png = outdir / "optimization_progress_zoom.png"
    plt.savefig(full_png, dpi=150, bbox_inches="tight")
    plt.close()
    
    return str(full_png), str(zoom_png)

def find_pareto_optimal(df):
    """Find Pareto-optimal solutions for multi-objective optimization."""
    # For maximization objectives, we negate them for the Pareto calculation
    # efficiency: maximize (so we use -efficiency)
    
    objectives = df[['efficiency', 'fakerate', 'duplicaterate', 'runtime']].values
    objectives[:, 0] = -objectives[:, 0]
    
    pareto_mask = np.zeros(len(objectives), dtype=bool)
    
    for i in range(len(objectives)):
        # Check if point i is dominated by any other point
        dominated = False
        for j in range(len(objectives)):
            if i != j:
                # Point j dominates point i if j is better or equal in all objectives
                # and strictly better in at least one
                if (np.all(objectives[j] <= objectives[i]) and 
                    np.any(objectives[j] < objectives[i])):
                    dominated = True
                    break
        pareto_mask[i] = not dominated
    
    return df[pareto_mask].copy()

def main():
    """Main optimization function."""
    global VERBOSE, TTBAR_PU, K_VALUE, EVENTS_PER_TRIAL, OPT_VARS, COMPLETED_TRIALS

    args = parse_cli_args()
    
    # Set global variables
    VERBOSE = args.verbose
    TTBAR_PU = args.ttbar_pu
    K_VALUE = args.k_value
    EVENTS_PER_TRIAL = args.events_per_trial

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

    # Choose VOCS and functions based on optimization mode
    if args.multi_objective:
        vocs_subset = VOCS(
            variables={k: VOCS_CKF.variables[k] for k in opt_vars},
            objectives=VOCS_CKF.objectives,
        )
        evaluate_function = evaluate_ckf_multi_objective
        generator_class = MOBOGenerator
        reference_point = {
            "efficiency": 0.0,      # Low efficiency is bad (we want to maximize)
            "fakerate": 100.0,      # High fake rate is bad (we want to minimize)
            "duplicaterate": 100.0, # High duplicate rate is bad (we want to minimize)
            "runtime": 1000.0       # High runtime is bad (we want to minimize)
        }
    else:
        vocs_subset = VOCS(
            variables={k: VOCS_CKF.variables[k] for k in opt_vars},
            objectives={"score": "MINIMIZE"},
            constraints={"fail": ["LESS_THAN", 0.5]}
        )
        evaluate_function = evaluate_ckf_single_objective
        generator_class = ExpectedImprovementGenerator
        reference_point = None

    # Handle output directory and resume logic
    if args.resume_from:
        # Convert to absolute path relative to current working directory
        outdir = Path(args.resume_from).resolve()
        if not outdir.exists():
            raise FileNotFoundError(f"Resume directory {outdir} does not exist")

        state_file = outdir / "xopt_state.json"
        csv_file = outdir / "optimization_history.csv"
        
        print(f"📂 Resuming optimization from {outdir}")
        print(f"Looking for files:")
        print(f"  State: {state_file}")
        print(f"  CSV: {csv_file}")
        print(f"  State exists: {state_file.exists()}")
        print(f"  CSV exists: {csv_file.exists()}")
    else:
        outdir = Path(args.output_dir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        state_file = None
        csv_file = None

    size = 1

    evaluator = xopt.Evaluator(
        function=evaluate_function,
        executor=None,
        max_workers=size,
    )

    if args.multi_objective:
        generator = generator_class(
            vocs=vocs_subset,
            n_candidates=size,
            reference_point=reference_point,
        )
    else:
        generator = generator_class(
            vocs=vocs_subset,
            n_candidates=size,
        )

    X = None

    n_seed = args.n_seed_trials
    n_guided = args.n_guided_trials

    if args.resume_from:
        try:
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                print(f"✅ CSV loaded - shape: {df.shape}")

                saved_vars = [col for col in df.columns if col in ALL_VARS]
                current_vars = opt_vars

                print(f"Saved variables: {saved_vars}")
                print(f"Current variables: {current_vars}")

                if set(saved_vars) != set(current_vars):
                    print("⚠️  Variable mismatch detected!")
                    print("🔧 Updating current optimization to match saved variables")
                    opt_vars = saved_vars
                    OPT_VARS = set(opt_vars)

                    if args.multi_objective:
                        vocs_subset = VOCS(
                            variables={k: VOCS_CKF.variables[k] for k in opt_vars},
                            objectives=VOCS_CKF.objectives,
                        )
                    else:
                        vocs_subset = VOCS(
                            variables={k: VOCS_CKF.variables[k] for k in opt_vars},
                            objectives={"score": "MINIMIZE"},
                            constraints={"fail": ["LESS_THAN", 0.5]},
                        )

                    if args.multi_objective:
                        generator = generator_class(
                            vocs=vocs_subset,
                            n_candidates=size,
                            reference_point=reference_point,
                        )
                    else:
                        generator = generator_class(
                            vocs=vocs_subset,
                            n_candidates=size,
                        )

                    evaluator = xopt.Evaluator(
                        function=evaluate_function,
                        executor=None,
                        max_workers=size,
                    )

                X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)
                X.data = df.copy()

                print("🔧 Adding existing data to generator...")
                X.generator.add_data(df)
            else:
                raise FileNotFoundError("CSV file not found")

        except Exception as e:
            print(f"⚠️  Failed to resume: {e}")
            print("🔄 Starting fresh optimization...")
            X = None
    else:
        X = None

    if X is None:
        X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)

    try:
        if hasattr(X.data, '__len__'):
            COMPLETED_TRIALS = len(X.data)
        elif hasattr(X.data, 'shape'):
            COMPLETED_TRIALS = X.data.shape[0]
        else:
            COMPLETED_TRIALS = 0
    except:
        COMPLETED_TRIALS = 0

    try:
        if hasattr(X.data, 'columns') and len(X.data) > 0 and 'trial' not in X.data.columns:
            X.data['trial'] = range(len(X.data))
    except:
        pass

    mode_str = 'multi-objective' if args.multi_objective else 'single-objective'
    print(f"🚀 Starting {mode_str} optimization")
    print(f"Variables to optimize: {opt_vars}")

    if args.resume_from:
        print(f"Running {n_seed} additional seed trials and {n_guided} additional guided trials")
        print(f"Already completed: {COMPLETED_TRIALS} trials")
    else:
        print(f"Target trials - Seed: {n_seed}, Guided: {n_guided}")

    print(f"Using Pythia-8 with pile-up μ = {TTBAR_PU}")
    print(f"Using K value = {K_VALUE} for score calculation")
    print(f"Simulating {EVENTS_PER_TRIAL} events per trial")

    if args.resume_from:
        trials_to_run_seed = n_seed
        trials_to_run_guided = n_guided
    else:
        trials_to_run_seed = n_seed
        trials_to_run_guided = n_guided

    total_trials_to_run = trials_to_run_seed + trials_to_run_guided

    if total_trials_to_run == 0:
        print("🎯 No trials requested!")
    else:
        if trials_to_run_seed > 0:
            print(f"🎲 Running {trials_to_run_seed} random seed trials...")
            for _ in range(trials_to_run_seed):
                X.random_evaluate(1)

        if trials_to_run_guided > 0:
            print(f"🧠 Running {trials_to_run_guided} Bayesian-guided trials...")

            for i in range(trials_to_run_guided):
                X.step()
    print()

    df = X.data
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if 'trial' not in df.columns:
        df['trial'] = range(1, len(df) + 1)

    if not args.multi_objective and 'inv_score' not in df.columns:
        df['inv_score'] = -df['score']

    csv_path = outdir / "optimization_history.csv"
    df_to_save = df.copy()
    if 'maxSeedsPerSpM' in df_to_save.columns:
        df_to_save['maxSeedsPerSpM'] = df_to_save['maxSeedsPerSpM'].round().astype(int)
    df_to_save.to_csv(csv_path, index=False)

    with open(outdir / "xopt_state.json", "w") as fh:
        fh.write(X.model_dump_json(indent=2))

    if args.multi_objective:
        pareto_optimal = find_pareto_optimal(df)
        pareto_csv_path = outdir / "pareto_optimal_solutions.csv"
        pareto_optimal.to_csv(pareto_csv_path, index=False)

        print(f"✅ Multi-objective optimization finished. Results in {outdir}")
        print(f"History saved to {csv_path}")
        print(f"Pareto-optimal solutions saved to: {pareto_csv_path}")
        print(f"Total trials: {len(df)}")
        print(f"Pareto-optimal solutions: {len(pareto_optimal)}")
    else:
        if args.resume_from:
            total_seed_trials = n_seed
        else:
            total_seed_trials = n_seed

        full_png, zoom_png = plot_score_vs_trial_dual_autozoom(df, total_seed_trials, outdir)

        best_row = df_to_save.loc[df_to_save["inv_score"].idxmax()]
        print(f"✅ Single-objective optimization finished. Results in {outdir}")
        print(f"Best parameters →\n{best_row.to_dict()}")
        print(f"History saved to {csv_path}")
        print("Plots Saved:", full_png, "and", zoom_png)

if __name__ == "__main__":
    main()
