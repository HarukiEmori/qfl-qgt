#!/usr/bin/env python
"""Sensitivity of the two-class emulator experiment to the multiplier of the local model.

The local gate-level model of the experiment uses a multiplier for the noisy clients that reproduces
an echo measurement of the circuit on the emulator (8 inputs, one parameter vector).  A pre-check run
on the emulator measures the echo frequency again, for the inputs and parameters of a training round.
This script

1. reads the echo frequency of the noisy clients from the shot tables of the pre-check run,
2. finds the multiplier of the local model that reproduces it (the model probability is averaged over
   the local images of the noisy clients of the pre-check seeds, at the initial parameters),
3. repeats the selection of the step sizes (tuning seeds) and the prediction for the seeds of the
   experiment at several multipliers of the local model.

    python scripts/emulator_multiplier_check.py --workers 13

Output: ``results/emulator-design/multiplier-sensitivity.json`` and a table on the screen.  The script
submits nothing and does not connect to Nexus.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                   # noqa: E402
import json                                                       # noqa: E402
import sys                                                        # noqa: E402
import warnings                                                   # noqa: E402
from concurrent.futures import ProcessPoolExecutor                # noqa: E402
from dataclasses import replace                                   # noqa: E402

import numpy as np                                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import scenarios as S                                # noqa: E402
from qfedqgt import sim                                           # noqa: E402
from qfedqgt.experiment import build, simulate                    # noqa: E402

SCENARIO = "emulator2_iid"
METHODS = ("QFedAvg", "QFedFisher", "QFedQGT")
OUT = os.path.join(ROOT, "results", "emulator-design", "multiplier-sensitivity.json")


def scenario_with(local: float):
    sc = S.SCENARIOS[SCENARIO]
    n_ref = sum(1 for g in sc.gate_scales if g == 1.0)
    return replace(sc, local_scales=(1.0,) * n_ref + (float(local),) * (sc.n_clients - n_ref))


def precheck_echo(name: str) -> dict:
    """All-zero frequency of the echo circuits of the noisy clients in the pre-check run ``results/<name>``."""
    sc, cfg = S.SCENARIOS[SCENARIO], S.run_config(SCENARIO)
    cache = os.path.join(ROOT, "results", name, "cache")
    hits = total = circuits = 0
    with open(os.path.join(cache, "jobs.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") != "submitted" or rec["device"]["gate_scale"] != max(sc.gate_scales):
                continue
            if rec["shots"] != [cfg.echo_shots]:
                continue                                           # the programs of the parameter-shift circuits
            for key in rec["keys"]:
                table = np.load(os.path.join(cache, key + ".npz"))["shots"]
                segments = table.shape[1] // sc.n_qubits
                zero = table.reshape(table.shape[0], segments, sc.n_qubits).sum(axis=2) == 0
                hits += int(zero.sum()); total += zero.size; circuits += segments
    freq = hits / total
    with open(os.path.join(ROOT, "results", name, "summary.json")) as f:
        seeds = sorted(r["seed"] for r in json.load(f)["runs"])
    return dict(run=name, seeds=seeds, circuits=circuits, samples=total, frequency=freq,
                standard_error=float(np.sqrt(freq * (1 - freq) / total)))


def model_echo(seeds, local: float) -> float:
    """All-zero probability of the echo circuit in the local model, averaged over the images of the noisy clients."""
    sc = S.SCENARIOS[SCENARIO]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    e = t.echo()
    noise = sim.NoiseModel.reimei().scaled(float(local))
    n_ref = sum(1 for g in sc.gate_scales if g == 1.0)
    values = []
    for seed in seeds:
        _, clients, theta0, _, _ = build(sc, seed)
        for c in clients[n_ref:]:
            values.append(float(sim.probabilities(sim.density(e, e.angles(c.X, theta0), noise), noise, t.n)[:, 0].mean()))
    return float(np.mean(values))


def matched(seeds, target: float, lo: float = 100.0, hi: float = 2000.0):
    if not model_echo(seeds, hi) < target < model_echo(seeds, lo):
        return None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if model_echo(seeds, mid) > target else (lo, mid)
    return float(0.5 * (lo + hi))


def _run(args):
    local, method, lr, seed = args
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    h = simulate(scenario_with(local), {method: lr}, S.run_config(SCENARIO), seed)[method]
    return local, method, lr, seed, float(np.mean(h["acc_shots"][-S.tail(SCENARIO):]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precheck", default="emulator2-smoke", help="results folder of the pre-check run")
    ap.add_argument("--multipliers", nargs="+", type=float, default=[430.0, 460.0, 490.0, 540.0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)

    echo = precheck_echo(a.precheck)
    f, se = echo["frequency"], echo["standard_error"]
    echo["matched_multiplier"] = matched(echo["seeds"], f)
    echo["matched_multiplier_range"] = [matched(echo["seeds"], f + se), matched(echo["seeds"], f - se)]
    echo["model_probability"] = {f"{m:g}": model_echo(echo["seeds"], m) for m in a.multipliers}
    print(f"pre-check run {a.precheck}: echo frequency of the noisy clients {f:.3f} +/- {se:.3f} ({echo['circuits']} circuits, "
          f"{echo['samples']} samples); matched multiplier of the local model {echo['matched_multiplier']:.0f} "
          f"({echo['matched_multiplier_range'][0]:.0f} to {echo['matched_multiplier_range'][1]:.0f})")

    plan = {m: S.learning_rates(SCENARIO)[m] for m in METHODS}
    tuning, report = S.tuning_seeds(SCENARIO), S.EMULATOR2_SEEDS
    tasks = [(l, m, lr, s) for l in a.multipliers for m in METHODS for lr in S.grid(m) for s in tuning]
    tasks += [(l, m, plan[m], s) for l in a.multipliers for m in METHODS for s in report]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(_run, tasks, chunksize=8))
    rows = []
    for l in a.multipliers:
        selected = {}
        for m in METHODS:
            score = {lr: float(np.mean([r[4] for r in res if r[0] == l and r[1] == m and r[2] == lr and r[3] in tuning]))
                     for lr in S.grid(m)}
            selected[m] = max(score, key=score.get)
        acc = {m: np.array([r[4] for r in sorted(res, key=lambda r: r[3])
                            if r[0] == l and r[1] == m and r[2] == plan[m] and r[3] in report]) for m in METHODS}
        row = dict(multiplier=l, selected_step_sizes=selected, same_as_experiment=selected == plan,
                   accuracy={m: float(acc[m].mean()) for m in METHODS}, difference={})
        for m in METHODS[:2]:
            d = 100.0 * (acc["QFedQGT"] - acc[m])
            row["difference"][m] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), higher=int((d > 0).sum()))
        rows.append(row)
        print(f"multiplier {l:g}: step sizes {[selected[m] for m in METHODS]} ({'as' if row['same_as_experiment'] else 'NOT as'} in the "
              f"experiment); accuracy " + " ".join(f"{row['accuracy'][m]:.3f}" for m in METHODS) + "; differences "
              + ", ".join(f"{row['difference'][m]['mean']:+.1f} +/- {row['difference'][m]['se']:.1f}" for m in METHODS[:2]))
    out = dict(scenario=SCENARIO, step_sizes_of_the_experiment=plan, tuning_seeds=list(tuning), report_seeds=list(report),
               precheck_echo=echo, multipliers=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
