#!/usr/bin/env python
"""Bures metric of one layer measured with two copies of the state (destructive SWAP test).

For every error-rate multiplier the script runs the two-copy circuits of ``qfedqgt.metric`` for
the parameters of one layer at one input, and compares the estimated block of the Bures metric
with (a) the exact Bures metric of the local gate-level noise model, (b) the value that the same
estimator returns in the local model, in which the SWAP test is also noisy, and (c) the
noiseless (Fubini-Study) metric.

    python scripts/nexus_metric.py --backend local                     # local model only, no cost
    python scripts/nexus_metric.py --backend nexus --device reimei-SC  # syntax check, no cost
    python scripts/nexus_metric.py --backend nexus --device reimei-E --scales 1 50 --shots 200

``--backend nexus`` requires ``qnx login``.  Output: ``results/metric/<device>.json`` and the
run record ``results/metric/<device>-provenance.json`` (shown when the script ends).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import data, geometry as geo, metric, provenance, sim      # noqa: E402
from qfedqgt.federated import LocalBackend, LocalExecutor, Request       # noqa: E402
from qfedqgt.scenarios import SCENARIOS                                  # noqa: E402


def overlap_errors(overlaps: np.ndarray, pairs, m: int, shots: int, N: int, p_hat: float) -> np.ndarray:
    """Standard errors of the entries of the Bures block (binomial errors of the overlaps, propagated)."""
    var = np.maximum(1.0 - overlaps ** 2, 0.0) / shots
    V = np.zeros((2 * m, 2 * m))
    for (u, v), x in zip(pairs, var[1:]):
        V[u, v] = V[v, u] = x
    # hs_ab = (1/4) sum_{s, s'} s s' O[(a, s), (b, s')]; on the diagonal the mixed overlap enters twice
    out = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                out[a, b] = (V[2 * a, 2 * a] + V[2 * a + 1, 2 * a + 1] + 4.0 * V[2 * a, 2 * a + 1]) / 16.0
            else:
                out[a, b] = sum(V[2 * a + i, 2 * b + j] for i in (0, 1) for j in (0, 1)) / 16.0
    return np.sqrt(out) / (2.0 * (1.0 - p_hat + 2.0 * p_hat / N))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("local", "nexus"), default="local")
    ap.add_argument("--device", default="reimei-E")
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 50.0])
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--layer", type=int, default=1, help="layer whose parameter block is measured (0-based)")
    ap.add_argument("--params", nargs="*", type=int, default=None, help="explicit parameter indices instead of a layer")
    ap.add_argument("--segments", type=int, default=8, help="two-copy circuits per program")
    ap.add_argument("--local-scale-factor", type=float, default=1.0,
                    help="multiplier of the local model = this factor times the multiplier of the device")
    ap.add_argument("--project", default="QFedQGT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=4 * 3600.0)
    a = ap.parse_args(argv)

    sc = SCENARIOS["shots_and_noise"]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    t2 = metric.two_copy_template(t)
    N = t.dim
    ds = data.load_mnist(sc.classes, t.n_data, 200, 20, angle_scale=sc.angle_scale, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    x = ds.X_train[rng.choice(len(ds.y_train))]
    theta = rng.uniform(-sc.init_width, sc.init_width, size=t.n_params)
    params = list(a.params) if a.params else list(t.blocks[a.layer])
    m = len(params)
    TA, TB, pairs = metric.overlap_plan(theta, params)
    A2 = t2.angles(np.tile(np.concatenate([x, x]), (len(TA), 1)), np.concatenate([TA, TB], axis=1))

    name = a.device if a.backend == "nexus" else "local"
    out_dir = os.path.join(ROOT, "results", "metric")
    os.makedirs(out_dir, exist_ok=True)
    if a.backend == "nexus":
        from qfedqgt.nexus import DeviceProfile, NexusExecutor

        ex = NexusExecutor(a.project, os.path.join(out_dir, f"cache-{a.device}"), segments=a.segments,
                           run_name=f"metric-{a.device}", timeout=a.timeout)
        devices = [DeviceProfile(a.device, s) for s in a.scales]
    else:
        ex = LocalExecutor()
        devices = [LocalBackend(sim.NoiseModel.reimei().scaled(s * a.local_scale_factor), seed=a.seed + i)
                   for i, s in enumerate(a.scales)]

    parameters = dict(scales=a.scales, shots=a.shots, parameters_measured=params, circuits_per_scale=len(TA),
                      qubits=t2.n, seed=a.seed, local_scale_factor=a.local_scale_factor)
    with provenance.record_run("two-copy measurement of the Bures metric", entry="scripts/nexus_metric.py",
                               executor=ex, parameters=parameters,
                               save_to=os.path.join(out_dir, f"{name}-provenance.json")) as rec:
        reqs = [Request(dev, t2, A2, a.shots, tag=f"two-copy-x{s:g}") for dev, s in zip(devices, a.scales)]
        ex.run(reqs)
        G_pure = np.real(sim.pure_qgt(t, t.angles(x, theta))[0])[np.ix_(params, params)]
        rows = []
        for s, req in zip(a.scales, reqs):
            overlaps = metric.overlaps_from_counts(req.counts[0], a.shots, t.n)
            est = metric.bures_block(overlaps, pairs, m, N)
            err = overlap_errors(overlaps, pairs, m, a.shots, N, est["p_hat"])
            # local gate-level model at the matched multiplier
            noise = sim.NoiseModel.reimei().scaled(s * a.local_scale_factor)
            rho, drho = sim.density_derivatives(t, t.angles(x, theta), noise)
            G_exact = geo.bures_metric(geo.sld_qgt(rho[0], drho[0]))[np.ix_(params, params)]
            purity_exact = float(np.real(np.trace(rho[0] @ rho[0])))
            p2 = sim.probabilities(sim.density(t2, A2, noise), noise, t2.n)
            model = metric.bures_block(p2 @ metric.swap_signs(t.n), pairs, m, N)
            dev_exact = est["bures"] - G_exact
            dev_model = est["bures"] - model["bures"]
            row = dict(scale=s, shots=a.shots, circuits=len(TA), purity=est["purity"], p_hat=est["p_hat"],
                       purity_local_exact=purity_exact, purity_local_estimator=model["purity"],
                       p_hat_local_estimator=model["p_hat"],
                       bures=est["bures"].tolist(), bures_se=err.tolist(), bures_local_exact=G_exact.tolist(),
                       bures_local_estimator=model["bures"].tolist(), metric_noiseless=G_pure.tolist(),
                       trace=float(np.trace(est["bures"])), trace_se=float(np.sqrt((np.diag(err) ** 2).sum())),
                       trace_local_exact=float(np.trace(G_exact)), trace_local_estimator=float(np.trace(model["bures"])),
                       trace_noiseless=float(np.trace(G_pure)),
                       rms_deviation_from_exact=float(np.sqrt(np.mean(dev_exact ** 2))),
                       rms_deviation_from_estimator=float(np.sqrt(np.mean(dev_model ** 2))),
                       rms_standard_error=float(np.sqrt(np.mean(err ** 2))),
                       chi2_per_entry_exact=float(np.mean((dev_exact / err) ** 2)),
                       chi2_per_entry_estimator=float(np.mean((dev_model / err) ** 2)))
            rows.append(row)
            print(f"x{s:<5g} purity {est['purity']:.3f} (local model {model['purity']:.3f}, state {purity_exact:.3f})  "
                  f"Tr G {row['trace']:.3f}+/-{row['trace_se']:.3f} (local estimator {row['trace_local_estimator']:.3f}, "
                  f"exact {row['trace_local_exact']:.3f}, noiseless {row['trace_noiseless']:.3f})  "
                  f"rms dev {row['rms_deviation_from_exact']:.3f} / se {row['rms_standard_error']:.3f}", flush=True)
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            jobs = getattr(ex, "job_records", [])
            json.dump(dict(backend=a.backend, device=name, shots=a.shots, seed=a.seed, parameters=params,
                           cost_charged_hqc=float(sum(j["cost"] for j in jobs if isinstance(j.get("cost"), (int, float)))),
                           cost_formula_hqc=getattr(ex, "cost_estimate", 0.0),
                           jobs_submitted_from=min((j["submitted"] for j in jobs if j.get("submitted")), default=""),
                           jobs_completed_by=max((j["completed"] for j in jobs if j.get("completed")), default=""),
                           rows=rows), f, indent=1)
        rec.outputs.append(path)


if __name__ == "__main__":
    main()
