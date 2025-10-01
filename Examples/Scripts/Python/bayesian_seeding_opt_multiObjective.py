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
from xopt.generators.bayesian.mobo import MOBOGenerator
from mpi4py.futures import MPIPoolExecutor
import os

COMPLETED_TRIALS: int = 0  
VERBOSE: bool = False        
SIM_DATA_DIR: Path | None = None    
USE_P8: bool = False     
TTBAR_PU: int = 0        

# CKF hyper‑parameter search space (bounds)
VOCS_CKF = VOCS(
    variables={
        # name                lower  upper
        "maxSeedsPerSpM": (1, 10),
        "cotThetaMax": (7, 12),
        "sigmaScattering": (1, 10),
        "radLengthPerSeed": (0.001, 0.5),
        "impactMax": (0.1, 25.0),
        "maxPtScattering": (10.0, 50.0),
        "deltaRMin": (1.0, 20.0),
        "deltaRMax": (50.0, 300.0),
    },
    objectives={
        "efficiency": "MAXIMIZE",
        "fakerate": "MINIMIZE",
        "duplicaterate": "MINIMIZE",
        "runtime": "MINIMIZE"
    },
)

# 1.  Hard-coded "standard" CKF parameter set
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
}

reference_point = {
    "efficiency": 0.0,  # We want to maximize efficiency, so set the reference to the worst possible value
    "fakerate": 100.0,  # We want to minimize fakerate, so set the reference to a high value
    "duplicaterate": 100.0,  # We want to minimize duplicaterate, so set the reference to a high value
    "runtime": 1000.0  # We want to minimize runtime, so set the reference to a high value
}


# CLI helpers
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
        "--resume-from",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing previous optimization state (xopt_state.json) to resume from.",
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
    USE_P8    = use_p8         
    TTBAR_PU  = ttbar_pu        

def get_full_chain_path():
    """Get the path to full_chain_tunable.py relative to this script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "full_chain_odd_tunable.py")
    
# Wrapper around ckf.py: takes a *dict* of parameters, returns *dict*
def evaluate_ckf(params: dict):
    """Run a single full chain benchmark with the supplied hyper‑parameters."""

    full_params = DEFAULT_CKF_PARAMS.copy()
    full_params.update(params)

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

    full_chain_args = [
        f"--sf_{k}={int(round(v))}" if k == "maxSeedsPerSpM" else f"--sf_{k}={v}"
        for k, v in full_params.items()
    ]

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="full_chain_run_"))
    original_cwd = os.getcwd()


    try:
        # Change to working directory
        os.chdir(workdir)
        
        # Get the path to full_chain_odd.py
        full_chain_path = get_full_chain_path()
        
        cli = ["python", full_chain_path, "--events=10", *full_chain_args]
        
        if SIM_DATA_DIR is not None:
            cli.append(f"--indir={SIM_DATA_DIR}")  

        if USE_P8:
            cli += ["--ttbar", f"--ttbar-pu={TTBAR_PU}"]

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
            #print("Timing CSV found. Attempting to read...")
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

    except Exception as exc:
        if VERBOSE:
            print(f"[WARN] An unexpected error occurred: {type(exc).__name__}: {exc}")
        return {
            "efficiency": 0,
            "fakerate": 100,
            "duplicaterate": 100,
            "runtime": 100,
        }

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(workdir, ignore_errors=True)
        COMPLETED_TRIALS += 1
        end_stamp = datetime.now().strftime("%H:%M:%S.%f")[:12]
    
        return {
            "efficiency": eff,
            "fakerate": fake,
            "duplicaterate": dup,
            "runtime": run_time,
        }

def get_pareto_optimal(df):
    objectives = ["efficiency", "fakerate", "duplicaterate", "runtime"]
    is_pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                dominates = True
                for obj in objectives:
                    if obj == "efficiency":
                        if df.iloc[i][obj] < df.iloc[j][obj]:
                            dominates = False
                            break
                    else:
                        if df.iloc[i][obj] > df.iloc[j][obj]:
                            dominates = False
                            break
                if dominates:
                    is_pareto[j] = False
    return df[is_pareto]

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

    global USE_P8, TTBAR_PU, VERBOSE, SIM_DATA_DIR, COMPLETED_TRIALS
    USE_P8 = args.pythia8
    TTBAR_PU = args.ttbar_pu
    VERBOSE = args.verbose
    SIM_DATA_DIR = args.indir

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"[rank-0] Pythia-8 {'ON' if USE_P8 else 'OFF'} → μ = {TTBAR_PU}")

    opt_vars = list(VOCS_CKF.variables.keys()) if args.opt_vars.lower() == "all" else [v.strip() for v in args.opt_vars.split(",") if v.strip()]

    if args.resume_from:
        # Try to resume from previous optimization
        state_file = args.resume_from / "xopt_state.json"
        csv_file = args.resume_from / "history.csv"
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Resuming optimization from {args.resume_from}")
        
        try:
            if state_file.exists():
                # Try loading from JSON state first
                X = xopt.Xopt.model_validate_json(state_file.read_text())
                if MPI.COMM_WORLD.Get_rank() == 0:
                    data_len = len(X.data) if hasattr(X.data, '__len__') else 0
                    print(f"✅ Loaded from JSON state: {data_len} previous trials")
            else:
                raise FileNotFoundError("JSON state not found")

        except (KeyError, xopt.errors.XoptError) as e:
            if MPI.COMM_WORLD.Get_rank() == 0:
                if "'name'" in str(e) or "No generator named" in str(e):
                    print("🔧 Fixing generator name in JSON state...")
                    # Auto-fix the JSON file with correct generator name
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    # Use the correct generator class name
                    state_data["generator"]["name"] = "mobo"  # Try lowercase
                    
                    with open(state_file, 'w') as f:
                        json.dump(state_data, f, indent=2)
                    
                    try:
                        # Try loading again
                        X = xopt.Xopt.model_validate_json(state_file.read_text())
                        data_len = len(X.data) if hasattr(X.data, '__len__') else 0
                        print(f"✅ Fixed and loaded from JSON state: {data_len} previous trials")
                    except Exception as e2:
                        print(f"⚠️  Still failed after fix: {e2}")
                        # Fall back to CSV
                        raise Exception("JSON fix failed, falling back to CSV")
                else:
                    raise
            else:
                raise
                    
        except Exception as e:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"⚠️  Failed to load JSON state ({e})")
                print("🔄 Attempting CSV fallback...")
            
            if not csv_file.exists():
                raise FileNotFoundError(f"Neither state file nor CSV found in {args.resume_from}")
            
            # Fallback: reconstruct from CSV
            vocs_subset = VOCS(
                variables={k: VOCS_CKF.variables[k] for k in opt_vars},
                objectives=VOCS_CKF.objectives
            )

            generator = MOBOGenerator(
                vocs=vocs_subset,
                n_candidates=1,
                reference_point=reference_point
            )

            evaluator = xopt.Evaluator(
                function=evaluate_ckf,
                executor=MPIPoolExecutor(
                    initializer=_init_worker,
                    initargs=(VERBOSE, str(SIM_DATA_DIR) if SIM_DATA_DIR else "", USE_P8, TTBAR_PU),
                )
            )

            X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)
            
            # Load previous data from CSV and keep as DataFrame
            if MPI.COMM_WORLD.Get_rank() == 0:
                df = pd.read_csv(csv_file)
                X.data = df  # Keep as DataFrame
                print(f"✅ Loaded from CSV fallback: {len(X.data)} previous trials")
        
        # Update evaluator with current settings (for all cases)
        X.evaluator = xopt.Evaluator(
            function=evaluate_ckf,
            executor=MPIPoolExecutor(
                initializer=_init_worker,
                initargs=(VERBOSE, str(SIM_DATA_DIR) if SIM_DATA_DIR else "", USE_P8, TTBAR_PU),
            )
        )
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            # Reset global counter
            try:
                if hasattr(X.data, '__len__'):
                    COMPLETED_TRIALS = len(X.data)
                elif hasattr(X.data, 'shape'):  # DataFrame
                    COMPLETED_TRIALS = X.data.shape[0]
                else:
                    COMPLETED_TRIALS = 0
            except:
                COMPLETED_TRIALS = 0
            
            # Add trial numbers if missing - but keep as DataFrame
            try:
                if hasattr(X.data, 'columns') and 'trial' not in X.data.columns:
                    # Add trial column to DataFrame
                    X.data['trial'] = range(len(X.data))
                    print(f"Added trial numbering to {len(X.data)} existing trials")
            except Exception as e:
                print(f"Warning: Could not add trial numbering: {e}")
            
    else:
        # Start fresh optimization
        vocs_subset = VOCS(
            variables={k: VOCS_CKF.variables[k] for k in opt_vars},
            objectives=VOCS_CKF.objectives
        )

        generator = MOBOGenerator(
            vocs=vocs_subset,
            n_candidates=1,
            reference_point=reference_point
        )

        evaluator = xopt.Evaluator(
            function=evaluate_ckf,
            executor=MPIPoolExecutor(
                initializer=_init_worker,
                initargs=(VERBOSE, str(SIM_DATA_DIR) if SIM_DATA_DIR else "", USE_P8, TTBAR_PU),
            )
        )

        X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)

    if MPI.COMM_WORLD.Get_rank() == 0:
        # Only do initial random sampling if starting fresh
        if not args.resume_from:
            X.random_evaluate(args.n_seed_trials)

        # Continue with guided optimization
        try:
            if hasattr(X.data, '__len__'):
                initial_trials = len(X.data)
            elif hasattr(X.data, 'shape'):
                initial_trials = X.data.shape[0]
            else:
                initial_trials = 0
        except:
            initial_trials = 0
            
        for i in range(args.n_guided_trials):
            result = X.step()
            
            try:
                if hasattr(X.data, '__len__'):
                    current_trial = len(X.data)
                elif hasattr(X.data, 'shape'):
                    current_trial = X.data.shape[0]
                else:
                    current_trial = initial_trials + i + 1
            except:
                current_trial = initial_trials + i + 1
            
            if VERBOSE:
                print(f"Trial {current_trial}: {result}")
            else:
                _print_progress(i+1, args.n_guided_trials)

        # Save results
        outdir = args.output_dir
        outdir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(X.data)
        csv_path = outdir / "history.csv"
        df.to_csv(csv_path, index=False)

        pareto_optimal = get_pareto_optimal(df)
        pareto_csv_path = outdir / "pareto_optimal.csv"
        pareto_optimal.to_csv(pareto_csv_path, index=False)

        with open(outdir / "xopt_state.json", "w") as fh:
            fh.write(X.model_dump_json(indent=2))

        print(f"✅ Optimisation finished. Results in {outdir}")
        print(f"History saved to {csv_path}")
        print(f"Pareto-optimal solutions saved to: {pareto_csv_path}")
        print(f"Total trials: {len(df)}")
        print(f"Pareto-optimal solutions: {len(pareto_optimal)}")

if __name__ == "__main__":
    main()
