 
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
import subprocess
import uproot
import pathlib
import pandas as pd
from datetime import datetime
from mpi4py import MPI

import xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from mpi4py.futures import MPIPoolExecutor

# ------------------------------------------------------------------
# CKF hyper‑parameter search space
# ------------------------------------------------------------------
VOCS_CKF = VOCS(
    variables={
        # name                lower  upper
        "maxSeedsPerSpM":   ( 5,     30),
        "cotThetaMax":      ( 0.5,   3.0),
        "sigmaScattering":  ( 0.1,   10.0),
        "radLengthPerSeed": ( 0.1,   4.0),
        "impactMax":        ( 0.1,  3.0),
        "maxPtScattering":  ( 1.0,  50.0),
        "deltaRMin":        ( 0.0,   3.0),
        "deltaRMax":        ( 3.0,  300.0),
    },
    objectives={"score": "MINIMIZE"},
)

# ------------------------------------------------------------------
# Wrapper around ckf.py: takes a *dict* of parameters, returns *dict*
# ------------------------------------------------------------------

def evaluate_ckf(params: dict) -> dict:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="ckf_run_"))
    ckf_args = [
        f"--sf_{k}={int(round(v))}" if k == "maxSeedsPerSpM" else f"--sf_{k}={v}"
        for k, v in params.items()
    ]
    cli = ["python", "ckf.py", "--nEvents=1", f"--output={workdir}", *ckf_args]

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
            eff   = rh["eff_particles"].member("fElements")[0]
            fake  = rh["fakeratio_tracks"].member("fElements")[0]
            dup   = rh["duplicateratio_tracks"].member("fElements")[0]

        print(f"[DEBUG] eff={eff} fake={fake} dup={dup} run={run_time}")

        # composite score  (same as Optuna example)
        import math
        if any(math.isnan(x) or math.isinf(x) for x in [eff, fake, dup, run_time]):
            score = 1.0
        else:
            k_dup, k_time = 7, 7
            penalty = fake + dup / k_dup + run_time / k_time
            score = -(eff - penalty)   # negate → MINIMIZE

        return {"score": score}

    except Exception as exc:
        print("[WARN] ckf.py failed:", exc, file=sys.stderr)
        return {"score": 1.0}

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

# ------------------------------------------------------------------
# Main optimisation loop
# ------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n-trials", type=int, default=50, help="Total optimisation iterations")
    ap.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("xopt_out"))
    args = ap.parse_args(argv)

    outdir: pathlib.Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    pool      = MPIPoolExecutor()                    # ← ➋ NEW LINE
    evaluator = xopt.Evaluator(                     # ←  replace old
        function=evaluate_ckf,
        executor=pool
    )
    generator = UpperConfidenceBoundGenerator(vocs=VOCS_CKF)
    X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=VOCS_CKF)

    # ---------- rank-0 drives the optimisation ----------
    if MPI.COMM_WORLD.Get_rank() == 0:
        X.random_evaluate(5)
        for _ in range(args.n_trials):
            X.step()

        csv_path = outdir / "history.csv"
        X.data.to_csv(csv_path, index=False)

        with open(outdir / "xopt_state.json", "w") as fh:
            fh.write(X.model_dump_json(indent=2))

        best = X.data.loc[X.data["score"].idxmin()]
        print(f"✅ Optimisation finished. Results in {outdir}")
        print(f"Best parameters →\n{best.to_dict()}")
        print(f"History saved to {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
