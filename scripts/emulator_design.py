#!/usr/bin/env python
"""Candidate designs of the two-class experiment on the emulator, compared in the local gate-level model.

An experiment on the emulator is limited by the number of circuits times the number of shots.  Before
the runs on the emulator, this script compares candidate designs (number of noisy clients, minibatch
size, shots of the reference clients, number of rounds, Dirichlet parameter) in the local model whose
noisy clients have the multiplier that reproduces the echo survival of the emulator.

Seeds.  The step size of every method is selected on the tuning seeds (100 to 107) for every candidate
and every number of rounds.  The candidates are then compared on the design seeds (200 to 215).  Neither
set is used on the emulator, whose runs use the seeds 0, 1, ...

    python scripts/emulator_design.py --workers 13                       # all candidates
    python scripts/emulator_design.py --candidates 2+10_B4_M10_iid       # selected candidates
    python scripts/emulator_design.py --report                           # table from the stored results

Output: ``results/emulator-design/<candidate>.json`` (one file per candidate; finished candidates are
skipped) and a table on the screen.  The cost of a run on the emulator is estimated with the formula
of ``qfedqgt.nexus.program_cost``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                   # noqa: E402
import json                                                       # noqa: E402
import sys                                                        # noqa: E402
import time                                                       # noqa: E402
import warnings                                                   # noqa: E402
from concurrent.futures import ProcessPoolExecutor                # noqa: E402

import numpy as np                                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import scenarios as S                                # noqa: E402
from qfedqgt import sim                                           # noqa: E402
from qfedqgt.experiment import Scenario, simulate                 # noqa: E402
from qfedqgt.federated import RunConfig                           # noqa: E402

OUT = os.path.join(ROOT, "results", "emulator-design")
METHODS = ("QFedAvg", "QFedFisher", "QFedQGT")
TUNING_SEEDS = S.TUNING_SEEDS
DESIGN_SEEDS = tuple(range(200, 216))
ROUNDS = 40                    # every run has this many rounds; shorter trainings are read off at the checkpoints
CHECKPOINTS = (15, 20, 30, 40)
TAIL = 5                       # the accuracy at a checkpoint is the mean over this many rounds
EVAL_SHOTS = 100
ECHO_SHOTS = 5
BASE = dict(classes=(0, 1), n_qubits=4, layers=1, beta=4.0, angle_scale=0.5, n_per_client=40,
            n_train_per_class=400, n_test_per_class=100)


def candidate(n_ref: int, n_noisy: int, batch: int, shots_ref: int, alpha: float | None = None, shots_noisy: int = 1):
    """(scenario, minibatch size) of a candidate: reference clients and clients with the multiplier of the emulator."""
    sc = Scenario(name="candidate", **BASE, alpha=alpha,
                  gate_scales=(1.0,) * n_ref + (S.EMU_SCALE,) * n_noisy,
                  local_scales=(1.0,) * n_ref + (S.EMU_LOCAL_SCALE,) * n_noisy,
                  shots=(shots_ref,) * n_ref + (shots_noisy,) * n_noisy)
    return sc, batch


def _name(n_ref, n_noisy, batch, shots_ref, alpha):
    return f"{n_ref}+{n_noisy}_B{batch}_M{shots_ref}_" + ("iid" if alpha is None else f"dir{alpha:g}")


# (reference clients, noisy clients, minibatch, shots of the reference clients, Dirichlet parameter)
CANDIDATES = {_name(*c): candidate(*c) for c in S.EMULATOR2_DESIGN_CANDIDATES}


def cost(name: str, rounds: int) -> dict:
    """Circuits, circuit-shots and the cost formula (HQC) of the training of one run with the three methods."""
    sc, batch = CANDIDATES[name]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    per_image = 1 + 2 * t.n_params
    circuits = len(METHODS) * batch * per_image * sc.n_clients * rounds
    circuit_shots = len(METHODS) * batch * per_image * sum(sc.shots) * rounds
    # cost formula of a program without its constant: (N_1q + 10 N_2q + 5 N_m) / 5000 per circuit and shot
    n1 = sum(1 for g in t.gates if g.kind != "zz")
    per_circuit_shot = (n1 + 10 * t.count("zz") + 5 * 2 * t.n) / 5000.0
    echo = batch * sc.n_clients * rounds * ECHO_SHOTS * 2 * per_circuit_shot      # an echo circuit has twice the gates
    return dict(circuits=circuits, circuit_shots=circuit_shots, hqc=circuit_shots * per_circuit_shot + echo)


def _run(args):
    name, seed, method, lr = args
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sc, batch = CANDIDATES[name]
    cfg = RunConfig(rounds=ROUNDS, batch=batch, local_steps=1, echo_shots=ECHO_SHOTS, eval_shots=EVAL_SHOTS)
    h = simulate(sc, {method: lr}, cfg, seed)[method]
    dev, ex = np.asarray(h["acc_shots"]), np.asarray(h["acc_exact"])
    return (name, seed, method, lr, [float(dev[t - TAIL + 1:t + 1].mean()) for t in CHECKPOINTS],
            [float(ex[t - TAIL + 1:t + 1].mean()) for t in CHECKPOINTS])


def evaluate(names, workers: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in names if not os.path.exists(os.path.join(OUT, n + ".json"))]
    if not names:
        return
    t0 = time.time()
    tasks = [(n, s, m, lr) for n in names for s in TUNING_SEEDS for m in METHODS for lr in S.grid(m)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tune = list(pool.map(_run, tasks, chunksize=2))
    print(f"tuning seeds: {len(tasks)} runs in {time.time() - t0:.0f} s", flush=True)
    selected, need = {}, set()
    for n in names:
        for ci, T in enumerate(CHECKPOINTS):
            for m in METHODS:
                score = {lr: float(np.mean([r[4][ci] for r in tune if r[0] == n and r[2] == m and r[3] == lr])) for lr in S.grid(m)}
                selected[(n, T, m)] = dict(lr=max(score, key=score.get), grid=score)
                need |= {(n, s, m, selected[(n, T, m)]["lr"]) for s in DESIGN_SEEDS}
    need = sorted(need)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(_run, need, chunksize=2))
    print(f"design seeds: {len(need)} runs, total {time.time() - t0:.0f} s", flush=True)
    for n in names:
        sc, batch = CANDIDATES[n]
        out = dict(candidate=n, batch=batch, scenario={k: v for k, v in sc.__dict__.items()}, rounds_simulated=ROUNDS,
                   tail=TAIL, eval_shots=EVAL_SHOTS, tuning_seeds=list(TUNING_SEEDS), design_seeds=list(DESIGN_SEEDS),
                   checkpoints={})
        for ci, T in enumerate(CHECKPOINTS):
            row = dict(cost=cost(n, T), methods={})
            for m in METHODS:
                lr = selected[(n, T, m)]["lr"]
                rr = sorted([r for r in runs if r[0] == n and r[2] == m and r[3] == lr], key=lambda r: r[1])
                row["methods"][m] = dict(step_size=lr, tuning={f"{k:g}": v for k, v in selected[(n, T, m)]["grid"].items()},
                                         device=[r[4][ci] for r in rr], noiseless=[r[5][ci] for r in rr])
            out["checkpoints"][str(T)] = row
        with open(os.path.join(OUT, n + ".json"), "w") as f:
            json.dump(out, f, indent=1)


def summary(name: str) -> list:
    """Rows of the report for one candidate: one row per checkpoint."""
    with open(os.path.join(OUT, name + ".json")) as f:
        d = json.load(f)
    rows = []
    for T, row in d["checkpoints"].items():
        acc = {m: np.array(row["methods"][m]["device"]) for m in METHODS}
        n = len(acc["QFedQGT"])
        item = dict(candidate=name, rounds=int(T), hqc=row["cost"]["hqc"], circuit_shots=row["cost"]["circuit_shots"],
                    step_sizes=[row["methods"][m]["step_size"] for m in METHODS],
                    accuracy={m: float(acc[m].mean()) for m in METHODS},
                    accuracy_se={m: float(acc[m].std(ddof=1) / np.sqrt(n)) for m in METHODS}, difference={})
        for m in METHODS[:2]:
            diff = 100.0 * (acc["QFedQGT"] - acc[m])
            item["difference"][m] = dict(mean=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(n)),
                                         higher=int((diff > 0).sum()), n=n)
        rows.append(item)
    return rows


def report(names) -> None:
    print(f"{'candidate':22s} {'T':>3s} {'HQC/run':>8s}  step sizes          accuracy (QFedAvg QFedFisher QFedQGT)   "
          "QFedQGT - QFedAvg      QFedQGT - QFedFisher")
    for name in names:
        if not os.path.exists(os.path.join(OUT, name + ".json")):
            continue
        for r in summary(name):
            a, d = r["accuracy"], r["difference"]
            print(f"{name:22s} {r['rounds']:3d} {r['hqc']:8.0f}  {str(r['step_sizes']):19s} "
                  + " ".join(f"{a[m]:.3f}" for m in METHODS) + "                    "
                  + "   ".join(f"{d[m]['mean']:+5.1f} +/- {d[m]['se']:3.1f} ({d[m]['higher']:2d}/{d[m]['n']})" for m in METHODS[:2]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=sorted(CANDIDATES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--report", action="store_true", help="only print the table of the stored results")
    a = ap.parse_args(argv)
    if not a.report:
        evaluate(a.candidates, a.workers)
    report(a.candidates)


if __name__ == "__main__":
    main()
