#!/usr/bin/env python
"""Simulation campaign of the paper on the local simulator (gate-level noise model).

Stages
------
tune    step-size grid of every method on the tuning seeds      -> results/tuning/<scenario>/
select  step size with the highest final test accuracy          -> results/tuning/learning_rates.json
main    reported runs with the selected step sizes              -> results/sim-<scenario>/
sens    every step size of the grid on the reported seeds       -> results/sens-<scenario>/
        (step-size sensitivity of the three compared methods; use with --scenarios)

    python scripts/run_campaign.py tune   --workers 12
    python scripts/run_campaign.py select
    python scripts/run_campaign.py main   --workers 12

Every (scenario, seed, method) is one task and one file, and finished tasks are skipped.
The tuning seeds and the reported seeds are disjoint (qfedqgt/scenarios.py).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                   # noqa: E402
import glob                                                       # noqa: E402
import json                                                       # noqa: E402
import sys                                                        # noqa: E402
import time                                                       # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np                                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt.experiment import simulate                           # noqa: E402
from qfedqgt import scenarios as S                                # noqa: E402

MAIN_METHODS = ("QFedAvg", "QFedFisher", "QFedQGT")
ABLATIONS = ("QFedAvg-shots", "QNG+data", "Bures+data", "Euclid+re", "QFedQGT-invvar")
WITH_ABLATIONS = ("shots_and_noise", "noise_only_x500")
TAIL = 10


def methods_of(scenario: str) -> tuple:
    return MAIN_METHODS + ABLATIONS if scenario in WITH_ABLATIONS else MAIN_METHODS


def _run(scenario: str, seed: int, label: str, lr: float, path: str) -> tuple:
    t0 = time.time()
    out = simulate(S.SCENARIOS[scenario], {label: lr}, S.RUN, seed)
    h = out[label]
    np.savez_compressed(path, **{f"{label}/{k}": v for k, v in h.items() if isinstance(v, np.ndarray)})
    return scenario, seed, label, float(np.mean(h["acc_exact"][-TAIL:])), time.time() - t0


def _tasks(stage: str, scenarios) -> list:
    tasks = []
    for sc in scenarios:
        if stage == "tune":
            d = os.path.join(ROOT, "results", "tuning", sc)
            for seed in S.TUNING_SEEDS:
                for m in methods_of(sc):
                    for lr in S.grid(m):
                        tasks.append((sc, seed, f"{m}@{lr:g}", lr, os.path.join(d, f"seed{seed}__{m}@{lr:g}.npz")))
        elif stage == "sens":
            # every step size of the grid on the reported seeds (the selected one is in sim-<scenario>)
            d = os.path.join(ROOT, "results", f"sens-{sc}")
            lrs = S.learning_rates(sc)
            for seed in S.REPORT_SEEDS:
                for m in MAIN_METHODS:
                    for lr in S.grid(m):
                        if lr != lrs[m]:
                            tasks.append((sc, seed, f"{m}@{lr:g}", lr, os.path.join(d, f"seed{seed:02d}__{m}@{lr:g}.npz")))
        else:
            d = os.path.join(ROOT, "results", f"sim-{sc}")
            lrs = S.learning_rates(sc)
            for seed in S.REPORT_SEEDS:
                for m in methods_of(sc):
                    tasks.append((sc, seed, m, lrs[m], os.path.join(d, f"seed{seed:02d}__{m}.npz")))
    return tasks


def run(stage: str, scenarios, workers: int) -> None:
    tasks = [t for t in _tasks(stage, scenarios) if not os.path.exists(t[-1])]
    for t in tasks:
        os.makedirs(os.path.dirname(t[-1]), exist_ok=True)
    print(f"{stage}: {len(tasks)} tasks on {workers} workers", flush=True)
    # long tasks first (many clients, natural-gradient methods)
    tasks.sort(key=lambda t: -S.SCENARIOS[t[0]].n_clients)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, *t) for t in tasks]
        for f in as_completed(futs):
            sc, seed, label, acc, dt = f.result()
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  last: {sc} seed {seed} {label} acc {acc:.3f} ({dt:.0f}s)"
                      f"  elapsed {time.time() - t0:.0f}s", flush=True)


def select(scenarios) -> None:
    """Step size with the highest final test accuracy on the reference device.

    The accuracy is the mean over the last rounds and over the tuning seeds; ties are broken by
    the lower test loss.  The report also records which value the lowest test loss would select.
    """
    path = os.path.join(ROOT, "results", "tuning", "learning_rates.json")
    table = {}
    if os.path.exists(path):
        with open(path) as f:
            table = json.load(f)
    report = {}
    for sc in scenarios:
        d = os.path.join(ROOT, "results", "tuning", sc)
        table[sc], report[sc] = {}, {}
        for m in methods_of(sc):
            rows = []
            for lr in S.grid(m):
                files = sorted(glob.glob(os.path.join(d, f"seed*__{m}@{lr:g}.npz")))
                if len(files) != len(S.TUNING_SEEDS):
                    raise SystemExit(f"tuning runs missing for {sc} {m}@{lr:g}: {len(files)} files")
                loss = [float(np.load(f)[f"{m}@{lr:g}/loss_exact"][-TAIL:].mean()) for f in files]
                acc = [float(np.load(f)[f"{m}@{lr:g}/acc_shots"][-TAIL:].mean()) for f in files]
                rows.append((lr, float(np.mean(loss)), float(np.mean(acc))))
            best = max(rows, key=lambda r: (r[2], -r[1]))
            by_loss = min(rows, key=lambda r: r[1])
            table[sc][m] = best[0]
            report[sc][m] = dict(selected=best[0], selected_by_loss=by_loss[0],
                                 grid=[dict(lr=r[0], loss=r[1], acc=r[2]) for r in rows])
            note = "" if by_loss[0] == best[0] else f"  (lowest loss: {by_loss[0]:g})"
            print(f"{sc:18s} {m:15s} lr={best[0]:<5g} " + "  ".join(f"{r[0]:g}:{r[1]:.3f}/{r[2]:.3f}" for r in rows) + note)
    with open(path, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
    with open(os.path.join(ROOT, "results", "tuning", "tuning_report.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["tune", "select", "main", "sens"])
    ap.add_argument("--scenarios", nargs="+", default=sorted(S.SCENARIOS), choices=sorted(S.SCENARIOS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    if a.stage == "select":
        select(a.scenarios)
    else:
        run(a.stage, a.scenarios, a.workers)
