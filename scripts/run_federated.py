#!/usr/bin/env python
"""Run the federated experiment of the paper on a chosen backend.

Backends
--------
``local``   exact density-matrix simulation of the gate-level noise model with shot sampling
``aer``     noiseless Aer simulation of the same native-gate programs (checks the program path)
``nexus``   Quantinuum emulator or device through Quantinuum Nexus (requires ``qnx login``)

The protocol is identical on every backend.  Results are written to
``results/<name>/seed<seed>.npz`` together with a JSON summary of the configuration.

Examples
--------
    python scripts/run_federated.py --scenario shots_and_noise --backend local --seeds 0 1 2
    python scripts/run_federated.py --scenario shots_and_noise --backend nexus --seeds 0 --name reimei-E
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qfedqgt.experiment import build, make_methods                       # noqa: E402
from qfedqgt.federated import LocalExecutor, RunConfig, run_federated     # noqa: E402
from qfedqgt.scenarios import SCENARIOS, RUN, learning_rates             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="shots_and_noise", choices=sorted(SCENARIOS))
    ap.add_argument("--backend", default="local", choices=["local", "aer", "nexus"])
    ap.add_argument("--methods", nargs="+", default=["QFedAvg", "QFedFisher", "QFedQGT"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--name", default=None, help="results sub-directory (default: <scenario>-<backend>)")
    ap.add_argument("--device", default="reimei-E", help="Nexus device name")
    ap.add_argument("--project", default="QFedQGT", help="Nexus project name")
    ap.add_argument("--segments", type=int, default=132, help="circuits packed into one program")
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    cfg = RUN
    if args.rounds is not None:
        cfg = replace(cfg, rounds=args.rounds)
    if args.eval_every is not None:
        cfg = replace(cfg, eval_every=args.eval_every)
    name = args.name or f"{args.scenario}-{args.backend}"
    out_dir = os.path.join(ROOT, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    lrs = {m: learning_rates(args.scenario)[m] for m in args.methods}

    for seed in args.seeds:
        path = os.path.join(out_dir, f"seed{seed}.npz")
        if os.path.exists(path):
            print(f"[skip] {path} exists")
            continue
        device_factory, executor = None, LocalExecutor()
        if args.backend in ("aer", "nexus"):
            from qfedqgt.nexus import AerExecutor, DeviceProfile, NexusExecutor

            if args.backend == "aer":
                executor = AerExecutor(segments=args.segments, seed=seed)
                device_factory = lambda scale, role, k: DeviceProfile("aer", scale, noisy=False)
            else:
                cache = os.path.join(out_dir, f"nexus-cache-seed{seed}")
                executor = NexusExecutor(args.project, cache, segments=args.segments, run_name=f"{name}-s{seed}")
                device_factory = lambda scale, role, k: DeviceProfile(args.device, scale)
        model, clients, theta0, ds, eval_device = build(sc, seed, device_factory=device_factory)
        t0 = time.time()

        def on_round(r, theta, hist):
            # checkpoint: the parameters of every method after every round
            np.savez(os.path.join(out_dir, f"checkpoint-seed{seed}.npz"), round=r,
                     **{m: theta[m] for m in theta})
            acc = {m: hist[m]["acc_exact"][-1] for m in theta}
            print(f"  seed {seed} round {r:3d}  " + "  ".join(f"{m}={a:.3f}" for m, a in acc.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

        out = run_federated(model, clients, make_methods(lrs), replace(cfg, seed=seed), theta0,
                            ds.X_test, ds.y_test, executor, eval_device, on_round=on_round)
        usage = out.pop("_usage")
        flat = {}
        for m, h in out.items():
            for key, val in h.items():
                if isinstance(val, np.ndarray):
                    flat[f"{m}/{key}"] = val
        np.savez_compressed(path, **flat)
        summary = dict(scenario=asdict(sc), run=asdict(replace(cfg, seed=seed)), learning_rates=lrs,
                       backend=args.backend, device=args.device if args.backend == "nexus" else args.backend,
                       seed=seed, usage=usage, seconds=time.time() - t0,
                       n_programs=getattr(executor, "n_programs", None),
                       cost_estimate_hqc=getattr(executor, "cost_estimate", None),
                       cost_charged_hqc=getattr(executor, "cost_charged", None),
                       date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with open(os.path.join(out_dir, f"seed{seed}.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"[done] {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
