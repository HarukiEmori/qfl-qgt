#!/usr/bin/env python
"""Does the error of a circuit depend on its position in a packed program?

The experiments pack many circuits into one program: every circuit is followed by a measurement
and a reset of all qubits, and the next circuit reuses the qubits.  This script checks that a
circuit late in a program is as reliable as an early one.  One program holds ``--circuits`` copies
of the same echo circuit (the circuit of ``--scenario`` followed by its inverse, whose ideal
outcome is all zeros), and the frequency of the all-zero outcome is recorded for every position.
Every error-rate multiplier is one job with one program; all jobs are submitted together.

    python scripts/nexus_position.py --backend aer --circuits 600 --shots 10           # noiseless check of the program path
    python scripts/nexus_position.py --device reimei-E --scales 1 300 --shots 100 20 --circuits 600
    python scripts/nexus_position.py --device reimei-E --scales 1 300 --shots 100 20 --circuits 600 --cached-only

Requires ``qnx login`` unless every result is already in the cache.  Output:
``results/nexus-position/<device>.json`` and the run record ``<device>-provenance.json``
(which program, when, on which device; shown when the script ends).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import data, provenance, sim                             # noqa: E402
from qfedqgt.federated import Request                                 # noqa: E402
from qfedqgt.scenarios import SCENARIOS                               # noqa: E402


def position_trend(hits: np.ndarray, shots: int) -> dict:
    """Survival frequency against the position of the circuit in the program.

    ``hits[i]`` is the number of shots of circuit ``i`` with the all-zero outcome.  The function
    returns the mean survival, the change of the survival from the first to the last circuit
    (slope of a least-squares line, with the standard error that follows from the binomial variance
    at the mean survival), and the survival in the four quarters of the program.
    """
    hits = np.asarray(hits, float)
    n = len(hits)
    y = hits / shots
    x = np.linspace(-0.5, 0.5, n)
    mean = float(y.mean())
    var = max(mean * (1.0 - mean), 1e-12) / shots          # variance of one position
    sxx = float((x ** 2).sum())
    change = float((x * (y - mean)).sum() / sxx)
    quarters = [float(q.mean()) for q in np.array_split(y, 4)]
    return dict(circuits=n, shots=int(shots), survival=mean, survival_se=float(np.sqrt(var / n)),
                change_first_to_last=change, change_se=float(np.sqrt(var / sxx)),
                quarters=quarters, quarter_se=float(np.sqrt(var / (n / 4.0))))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="nexus", choices=["nexus", "aer"])
    ap.add_argument("--device", default="reimei-E")
    ap.add_argument("--scenario", default="emulator_iid", choices=sorted(SCENARIOS),
                    help="scenario whose circuit and input are used")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 300.0], help="error-rate multipliers")
    ap.add_argument("--shots", nargs="+", type=int, default=[100, 20], help="shots per circuit, one value per multiplier")
    ap.add_argument("--circuits", type=int, default=600, help="copies of the echo circuit in the program")
    ap.add_argument("--project", default="QFedQGT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=4 * 3600.0)
    ap.add_argument("--stall-timeout", type=float, default=600.0)
    ap.add_argument("--cached-only", action="store_true", help="never submit a job; fail if a result is not in the cache")
    a = ap.parse_args(argv)
    if len(a.shots) == 1:
        a.shots = a.shots * len(a.scales)
    if len(a.shots) != len(a.scales):
        ap.error("--shots needs one value, or one value per multiplier")

    from qfedqgt.nexus import AerExecutor, DeviceProfile, NexusExecutor, program_cost

    sc = SCENARIOS[a.scenario]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    e = t.echo()
    ds = data.load_mnist(sc.classes, t.n_data, 200, 20, angle_scale=sc.angle_scale, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    x = ds.X_train[rng.choice(len(ds.y_train), size=1, replace=False)]
    theta = rng.uniform(-sc.init_width, sc.init_width, size=t.n_params)
    A = np.tile(e.angles(x, theta), (a.circuits, 1))            # the same circuit at every position
    stem = a.device if a.backend == "nexus" else "aer"
    out_dir = os.path.join(ROOT, "results", "nexus-position")
    if a.backend == "aer":
        a.scales = [1.0]
        a.shots = a.shots[:1]
        ex = AerExecutor(segments=a.circuits, seed=a.seed)
        devices = [DeviceProfile("aer", 1.0, noisy=False)]
    else:
        ex = NexusExecutor(a.project, os.path.join(out_dir, f"cache-{a.device}"), segments=a.circuits,
                           run_name=f"position-{a.device}", timeout=a.timeout, max_programs_per_job=1,
                           cached_only=a.cached_only, stall_timeout=None if a.cached_only else a.stall_timeout)
        devices = [DeviceProfile(a.device, s) for s in a.scales]
    reqs = [Request(d, e, A, m, tag=f"position-x{s:g}") for d, s, m in zip(devices, a.scales, a.shots)]
    estimate = [float(program_cost(e, a.circuits, m)) for m in a.shots]
    print("cost formula (HQC): " + ", ".join(f"x{s:g}: {c:.0f}" for s, c in zip(a.scales, estimate)), flush=True)

    with provenance.record_run("survival of an echo circuit against its position in a packed program",
                               entry="scripts/nexus_position.py", executor=ex,
                               parameters=dict(scenario=a.scenario, layers=sc.layers, circuits=a.circuits,
                                               scales=a.scales, shots=a.shots, seed=a.seed),
                               save_to=os.path.join(out_dir, f"{stem}-provenance.json"), replay=a.cached_only) as rec:
        ex.run(reqs)
        rows = []
        for s, m, req in zip(a.scales, a.shots, reqs):
            hits = req.counts[0, :, 0]
            noise = sim.NoiseModel.reimei().scaled(s)
            local = 1.0 if a.backend == "aer" else float(sim.probabilities(sim.density(e, A[:1], noise), noise, t.n)[0, 0])
            row = dict(scale=s, **position_trend(hits, m), survival_local_model=local, hits=[int(h) for h in hits])
            rows.append(row)
            z = row["change_first_to_last"] / row["change_se"] if row["change_se"] > 0 else 0.0
            print(f"x{s:<5g} survival {row['survival']:.4f}+/-{row['survival_se']:.4f} (local model {local:.4f})  "
                  f"change from the first to the last circuit {row['change_first_to_last']:+.4f}+/-{row['change_se']:.4f} "
                  f"({z:+.1f} standard errors)  quarters " + " ".join(f"{q:.4f}" for q in row["quarters"])
                  + f" (+/-{row['quarter_se']:.4f})", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        jobs = getattr(ex, "job_records", [])
        result = dict(device=a.device if a.backend == "nexus" else "aer", scenario=a.scenario, layers=sc.layers,
                      circuits=a.circuits, seed=a.seed, cost_formula_hqc=estimate,
                      cost_charged_hqc=float(sum(j["cost"] for j in jobs if isinstance(j.get("cost"), (int, float)))),
                      jobs_submitted_from=min((j["submitted"] for j in jobs if j.get("submitted")), default=""),
                      jobs_completed_by=max((j["completed"] for j in jobs if j.get("completed")), default=""),
                      rows=rows)
        path = os.path.join(out_dir, f"{stem}.json")
        provenance.write_json(path, result, replay=a.cached_only, ignore=("cost_charged_hqc",))
        rec.outputs.append(path)


if __name__ == "__main__":
    main()
