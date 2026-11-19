#!/usr/bin/env python
"""Noise strength inferred from echo circuits on a Quantinuum backend, for several error-rate multipliers.

Every multiplier is one job with one program (the echo circuits of a few inputs).  All jobs are
submitted together and wait in the queue at the same time.  The script compares the survival
frequency and the inferred noise strength with the local gate-level noise model.

    python scripts/nexus_geometry.py --device reimei-E --scales 1 50 150 200 300 500 --shots 100

Requires ``qnx login`` unless every result is already in the cache.  Output:
``results/nexus-geometry/<device>.json`` and the run record ``<device>-provenance.json``
(which program, when, on which device; shown when the script ends).
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
from qfedqgt.nexus import DeviceProfile, NexusExecutor                # noqa: E402
from qfedqgt.scenarios import SCENARIOS                               # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="reimei-E")
    ap.add_argument("--scales", nargs="+", type=float, default=[1, 50, 150, 200, 300, 500])
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--inputs", type=int, default=8)
    ap.add_argument("--project", default="QFedQGT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=7200.0)
    a = ap.parse_args(argv)

    sc = SCENARIOS["shots_and_noise"]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    e = t.echo()
    N = t.dim
    ds = data.load_mnist(sc.classes, t.n_data, 200, 20, angle_scale=sc.angle_scale, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    X = ds.X_train[rng.choice(len(ds.y_train), size=a.inputs, replace=False)]
    theta = rng.uniform(-sc.init_width, sc.init_width, size=t.n_params)
    A = e.angles(X, theta)
    out_dir = os.path.join(ROOT, "results", "nexus-geometry")
    ex = NexusExecutor(a.project, os.path.join(out_dir, f"cache-{a.device}"), segments=a.inputs,
                       run_name=f"geometry-{a.device}", timeout=a.timeout)
    reqs = [Request(DeviceProfile(a.device, s), e, A, a.shots, tag=f"echo-x{s:g}") for s in a.scales]
    t0 = time.time()
    with provenance.record_run("noise strength from echo circuits", entry="scripts/nexus_geometry.py", executor=ex,
                               parameters=dict(scales=a.scales, shots=a.shots, inputs=a.inputs, seed=a.seed),
                               save_to=os.path.join(out_dir, f"{a.device}-provenance.json")) as rec:
        ex.run(reqs)
        rows = []
        for s, req in zip(a.scales, reqs):
            n = a.shots * a.inputs
            surv = float(req.counts[0, :, 0].sum() / n)
            err = float(np.sqrt(max(surv * (1 - surv), 1e-12) / n))
            noise = sim.NoiseModel.reimei().scaled(s)
            local = float(sim.probabilities(sim.density(e, A, noise), noise, t.n)[:, 0].mean())
            p_hat = float(geo.depolarizing_from_purity(max(surv, 1.0 / N), N))
            p_loc = float(geo.depolarizing_from_purity(max(local, 1.0 / N), N))
            rows.append(dict(scale=s, samples=n, survival=surv, survival_se=err, survival_local=local,
                             p_hat=p_hat, p_hat_local=p_loc, kappa=float(geo.kappa(min(p_hat, 0.999), N)),
                             kappa_local=float(geo.kappa(min(p_loc, 0.999), N))))
            print(f"x{s:<5g} survival {surv:.3f}+/-{err:.3f} (local model {local:.3f})  p_hat {p_hat:.3f} (local {p_loc:.3f})",
                  flush=True)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{a.device}.json")
        jobs = ex.job_records
        with open(path, "w") as f:
            json.dump(dict(device=a.device, shots=a.shots, inputs=a.inputs, seed=a.seed,
                           cost_charged_hqc=float(sum(j["cost"] for j in jobs if isinstance(j.get("cost"), (int, float)))),
                           cost_formula_hqc=ex.cost_estimate,
                           jobs_submitted_from=min((j["submitted"] for j in jobs if j.get("submitted")), default=""),
                           jobs_completed_by=max((j["completed"] for j in jobs if j.get("completed")), default=""),
                           rows=rows), f, indent=1)
        rec.outputs.append(path)

if __name__ == "__main__":
    main()
