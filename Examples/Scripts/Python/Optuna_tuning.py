#!/usr/bin/env python3
import sys
import re
import optuna
import logging
import uproot
import matplotlib
import subprocess
import json
import pandas as pd
import time
from pathlib import Path

matplotlib.use("pdf")

srcDir = Path(__file__).resolve().parent

import csv

def write_trial_to_csv(trial_number, params, eff, fakerate, duplicaterate, runtime, score, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if trial_number == 0:  # Write header for the first trial
            header = ['trial_number'] + list(params.keys()) + ['efficiency', 'fakerate', 'duplicaterate', 'runtime', 'score']
            writer.writerow(header)
        
        row = [trial_number] + list(params.values()) + [eff, fakerate, duplicaterate, runtime, score]
        writer.writerow(row)


def run_full_chain(params, names, outDir):
    if len(params) != len(names):
        raise Exception("Length of Params must equal names")

    full_chain_script = srcDir / "full_chain_odd.py"
    nevts = "--events=1"  # Changed from --nEvents to --events
    outdir = "--output=" + str(outDir)
    ttbar_flag = "--ttbar"  # Changed from -P to --ttbar
    pileup = "--ttbar-pu=200"

    ret = ["python"]
    ret.append(str(full_chain_script))  # Convert to string
    ret.append(nevts)
    ret.append(outdir)
    ret.append(ttbar_flag)
    ret.append(pileup)

    # Add the optimization parameters
    for i, param in enumerate(params):
        arg = "--sf_" + names[i] + "=" + str(param)
        ret.append(arg)

    # Run the full chain and capture the result
    try:
        proc = subprocess.run(ret, capture_output=True, text=True, timeout=300)  # 5 min timeout
    except subprocess.TimeoutExpired:
        logging.warning("Full chain script timed out")
        return False, None

    # Bail out early on non-zero exit; keep default penalties
    if proc.returncode != 0:
        logging.warning("Full chain failed (exit %s). Stderr: %s", 
                        proc.returncode, proc.stderr)
        return False, proc

    return True, proc


class Objective:
    def __init__(self, k_dup, k_time, csv_file):
        self.res = {
            "eff": [],
            "fakerate": [],
            "duplicaterate": [],
            "runtime": [],
        }
        self.csv_file = csv_file
        self.k_dup = k_dup
        self.k_time = k_time

    def __call__(self, trial, ckf_perf=True):
        params = {
            "maxSeedsPerSpM": trial.suggest_int("maxSeedsPerSpM", 0, 10),
            "cotThetaMax": trial.suggest_float("cotThetaMax", 5.0, 10.0),
            "sigmaScattering": trial.suggest_float("sigmaScattering", 0.2, 50),
            "radLengthPerSeed": trial.suggest_float("radLengthPerSeed", 0.001, 0.1),
            "impactMax": trial.suggest_float("impactMax", 0.1, 25),
            "maxPtScattering": trial.suggest_float("maxPtScattering", 1, 50),
            "deltaRMin": trial.suggest_float("deltaRMin", 0.25, 30),
            "deltaRMax": trial.suggest_float("deltaRMax", 50, 300),
        }


        get_tracking_perf(self, ckf_perf, list(params.values()), list(params.keys()))


        if self.res["eff"] and self.res["fakerate"] and self.res["duplicaterate"] and self.res["runtime"]:
            efficiency = self.res["eff"][-1]
            fakerate = self.res["fakerate"][-1]
            duplicaterate = self.res["duplicaterate"][-1]
            runtime = self.res["runtime"][-1]
            
            penalty = fakerate + duplicaterate / self.k_dup + runtime / (self.k_time)
            score = efficiency - penalty
            
            write_trial_to_csv(
                trial.number, 
                params, 
                efficiency, 
                fakerate, 
                duplicaterate, 
                runtime, 
                score, 
                self.csv_file
            )
            
            return score
        else:
            logging.warning("Could not extract valid metrics, returning default poor value")
            write_trial_to_csv(
                trial.number, 
                params, 
                None, None, None, None, -100.0, 
                self.csv_file
            )
        return -100.0


def get_tracking_perf(self, ckf_perf, params, keys):
    if ckf_perf:
        outDirName = "odd_output"  # Changed to match full_chain_odd.py default
        outputfile = srcDir / outDirName / "performance_finding_ckf.root"  # Updated path
        effContName = "particles"
        contName = "tracks"
    else:
        outDirName = "odd_output"  # Changed to match full_chain_odd.py default
        outputfile = srcDir / outDirName / "performance_seeding.root"
        effContName = "seeds"
        contName = "seeds"

    outputDir = Path(srcDir / outDirName)
    outputDir.mkdir(exist_ok=True)

    # defaults in case ANYTHING fails
    eff = 0.0
    fake = 1.0
    dup = 1.0
    run_time = 999.0

    t0 = time.perf_counter()  # Start timing
    ok, proc = run_full_chain(params, keys, outputDir)  # Changed function name

    # If the script failed, store pessimistic defaults
    if not ok:
        self.res['eff'].append(eff)
        self.res['fakerate'].append(fake)
        self.res['duplicaterate'].append(dup)
        self.res['runtime'].append(run_time)
        return

    # Check if output file exists
    if not Path(outputfile).is_file():
        logging.warning("No output ROOT file found at %s — keeping default penalties.", outputfile)
        self.res["eff"].append(eff)
        self.res["fakerate"].append(fake)
        self.res["duplicaterate"].append(dup)
        self.res["runtime"].append(run_time)
        return

    try:
        rootFile = uproot.open(outputfile)
        
        # Extract performance metrics
        self.res["eff"].append(rootFile["eff_" + effContName].member("fElements")[0] * 100)
        self.res["fakerate"].append(rootFile["fakeratio_" + contName].member("fElements")[0] * 100)
        self.res["duplicaterate"].append(
            rootFile["duplicateratio_" + contName].member("fElements")[0] * 100)
    
    except Exception as e:
        logging.warning("Failed to read ROOT file: %s", e)
        self.res["eff"].append(eff)
        self.res["fakerate"].append(fake)
        self.res["duplicaterate"].append(dup)
        self.res["runtime"].append(run_time)
        return

    # Extract timing information
    timing_csv = srcDir / outDirName / "timing.csv"
    if timing_csv.exists():
        #print("Timing CSV found. Attempting to read...")
        try:
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
                run_time = time.perf_counter() - t0  # Use seconds
        except Exception as e:
            logging.warning(f"Failed to read timing CSV: {e}")
            run_time = time.perf_counter() - t0  # Use seconds
    else:
        print(f"Timing CSV not found at {timing_csv}. Using fallback timing method.")
        run_time = time.perf_counter() - t0  # Use seconds

    # Store the result
    self.res["runtime"].append(run_time)
    return


def main():
    k_dup = 5
    k_time = 5

    # Create the CSV file for storing results
    csv_file = "optimization_results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['trial_number', 'maxSeedsPerSpM', 'cotThetaMax', 'sigmaScattering', 
                         'radLengthPerSeed', 'impactMax', 'maxPtScattering', 'deltaRMin', 
                         'deltaRMax', 'efficiency', 'fakerate', 'duplicaterate', 'runtime', 'score'])

    # Initializing the objective (score) function
    objective = Objective(k_dup, k_time, csv_file)  # Pass csv_file to Objective


    # Optuna logger
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "full_chain_study"  # Updated study name
    storage_name = "sqlite:///{}.db".format(study_name)

    # creating a new optuna study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    # Start Optimization
    study.optimize(objective, n_trials=100)

    # Printout the best trial values
    print("Best Trial until now", flush=True)
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}", flush=True)

    # After optimization is complete
    logging.info(f"Results have been saved to {csv_file}")

    outputDir = Path("OptunaResults")
    outputDir.mkdir(exist_ok=True)

    with open(outputDir / "results.json", "w") as fp:
        json.dump(study.best_params, fp)


if __name__ == "__main__":
    main()