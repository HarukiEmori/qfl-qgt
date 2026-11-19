#!/usr/bin/env python
"""Smallest useful job on a Quantinuum backend through Nexus (run this before any campaign).

For every error-rate multiplier the script submits two programs, the echo circuits and the
forward circuits of a few inputs, and reports whether the backend accepted them, how long the
jobs took, what they cost, and the echo survival compared with the local noise model.

    python scripts/nexus_smoke.py --device reimei-SC                  # syntax checker, no cost
    python scripts/nexus_smoke.py --device reimei-E --scales 1 200    # emulator, with error scaling

Requires ``qnx login``.  Results and the job log go to ``results/nexus-tests/<device>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import data, geometry as geo, provenance, sim           # noqa: E402
from qfedqgt.federated import Request                                 # noqa: E402
from qfedqgt.nexus import DeviceProfile, NexusExecutor, program_cost   # noqa: E402
from qfedqgt.scenarios import SCENARIOS                               # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="reimei-SC")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0])
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--inputs", type=int, default=4)
    ap.add_argument("--project", default="QFedQGT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    a = ap.parse_args(argv)

    sc = SCENARIOS["shots_and_noise"]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    e = t.echo()
    ds = data.load_mnist(sc.classes, t.n_data, 200, 20, angle_scale=sc.angle_scale, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    X = ds.X_train[rng.choice(len(ds.y_train), size=a.inputs, replace=False)]
    theta = rng.uniform(-sc.init_width, sc.init_width, size=t.n_params)
    out_dir = os.path.join(ROOT, "results", "nexus-tests", a.device)
    report = []
    for scale in a.scales:
        device = DeviceProfile(a.device, scale)
        ex = NexusExecutor(a.project, os.path.join(out_dir, f"cache-x{scale:g}"), segments=a.inputs,
                           run_name=f"smoke-{a.device}-x{scale:g}", timeout=a.timeout)
        reqs = [Request(device, e, e.angles(X, theta), a.shots, tag="echo"),
                Request(device, t, t.angles(X, theta), a.shots, tag="forward")]
        t0 = time.time()
        with provenance.record_run("smallest job on a Quantinuum backend", entry="scripts/nexus_smoke.py", executor=ex,
                                   parameters=dict(scale=scale, shots=a.shots, inputs=a.inputs, seed=a.seed),
                                   save_to=os.path.join(out_dir, f"provenance-x{scale:g}.json")):
            ex.run(reqs)
        dt = time.time() - t0
        survival = float(reqs[0].counts[0, :, 0].sum() / (a.shots * a.inputs))
        # local gate-level model for comparison
        noise = sim.NoiseModel.reimei().scaled(scale)
        local = float(sim.probabilities(sim.density(e, e.angles(X, theta), noise), noise, t.n)[:, 0].mean())
        z = sim.z_observables(t.n, pairs=False)
        phi = reqs[1].counts[0] @ z / a.shots
        phi_local = sim.probabilities(sim.density(t, t.angles(X, theta), noise), noise, t.n) @ z
        row = dict(device=a.device, scale=scale, shots=a.shots, inputs=a.inputs, seconds=round(dt, 1),
                   programs=ex.n_programs, cost_formula=round(ex.cost_estimate, 2), cost_charged=ex.cost_charged,
                   echo_survival=survival, echo_survival_local=round(local, 4),
                   p_hat=float(geo.depolarizing_from_purity(max(survival, 1.0 / t.dim), t.dim)),
                   features_rms_difference=float(np.sqrt(np.mean((phi - phi_local) ** 2))),
                   counts_shape=list(reqs[1].counts.shape))
        report.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"smoke-{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
