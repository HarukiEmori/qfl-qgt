#!/usr/bin/env python
"""Two-copy estimate of the Bures metric in the local gate-level model, separated into its steps.

On a noisy device the two-copy estimate of a block of the Bures metric (``scripts/nexus_metric.py``) lies
below the Bures metric of the state.  Two things contribute, and the local model separates them.

1. The estimator divides the Hilbert-Schmidt block ``Tr[(d_i rho)(d_j rho)]`` by ``2 (1 - p + 2 p / N)`` with
   the noise strength ``p`` inferred from the purity.  This conversion is exact for global depolarizing noise
   and not for the gate-level noise of the model.
2. The destructive SWAP test is noisy as well: its gates have errors, and the readout has errors.  The phase
   flips that the model applies after the last gate of the SWAP test precede the measurement in the
   computational basis and do not change the outcome probabilities.

Before the SWAP test the two copies of the model are in the product of the single-copy states.  The first step
therefore involves the conversion only, and the second step the SWAP test only.  For the circuit, the input, the
parameter vector and the parameter block of ``scripts/nexus_metric.py`` the script computes four blocks at every
error-rate multiplier:

    state            the Bures metric of the state of the model,
    noiseless        the two-copy estimate from the exact overlaps ``Tr[rho_a rho_b]`` of the states of the
                     model, that is, with a noiseless two-copy measurement and without statistical error,
    perfect_readout  the two-copy estimate from the outcome probabilities of the noisy two-copy circuits of the
                     model with a perfect readout (gate errors of the SWAP test only),
    noisy            the two-copy estimate from the outcome probabilities of the noisy two-copy circuits of the
                     model (gate errors and readout errors), without statistical error,

and the attenuation ``c_G = <G_est, G> / <G, G>`` of each block with respect to the block ``G`` of the noiseless
(Fubini-Study) metric, with the Frobenius inner product.

    python scripts/metric_decomposition.py            # computes, prints, and writes the result file
    python scripts/metric_decomposition.py --check    # computes and compares with the stored result file

Output: text and ``results/metric/decomposition.json``.  The script submits nothing and does not connect to
Nexus.  It also verifies that its "state" and "noisy" blocks equal the blocks of the local model stored with
the measurement on the emulator (``results/metric/reimei-E.json``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import data, geometry as geo, metric, sim      # noqa: E402
from qfedqgt.scenarios import SCENARIOS                      # noqa: E402

OUT = os.path.join(ROOT, "results", "metric", "decomposition.json")
MEASURED = os.path.join(ROOT, "results", "metric", "reimei-E.json")


def attenuation(G_est: np.ndarray, G: np.ndarray) -> float:
    """Least-squares factor ``c`` of ``G_est ~ c G`` (Frobenius inner product)."""
    return float(np.sum(G_est * G) / np.sum(G * G))


def setup(seed: int = 0, layer: int = 1):
    """Circuit, input, parameters and parameter block of ``scripts/nexus_metric.py`` with its default arguments."""
    sc = SCENARIOS["shots_and_noise"]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    ds = data.load_mnist(sc.classes, t.n_data, 200, 20, angle_scale=sc.angle_scale, seed=seed)
    rng = np.random.default_rng(seed)
    x = ds.X_train[rng.choice(len(ds.y_train))]
    theta = rng.uniform(-sc.init_width, sc.init_width, size=t.n_params)
    return t, x, theta, list(t.blocks[layer])


def blocks(t, x, theta, params, noise: sim.NoiseModel) -> dict:
    """The four blocks of the docstring for one noise model, with the purities and the inferred noise strengths."""
    N, m = t.dim, len(params)
    TA, TB, pairs = metric.overlap_plan(theta, params)
    # Bures metric of the state
    rho, drho = sim.density_derivatives(t, t.angles(x, theta), noise)
    G_state = geo.bures_metric(geo.sld_qgt(rho[0], drho[0]))[np.ix_(params, params)]
    purity_state = float(np.real(np.trace(rho[0] @ rho[0])))
    # noiseless two-copy measurement: exact overlaps of the states of the single-copy circuit
    X = np.tile(x, (len(TA), 1))
    ra = sim.density(t, t.angles(X, TA), noise)
    rb = sim.density(t, t.angles(X, TB), noise)
    noiseless = metric.bures_block(np.real(np.einsum("cij,cji->c", ra, rb)), pairs, m, N)
    # noisy two-copy circuits of the model
    t2 = metric.two_copy_template(t)
    A2 = t2.angles(np.tile(np.concatenate([x, x]), (len(TA), 1)), np.concatenate([TA, TB], axis=1))
    rho2 = sim.density(t2, A2, noise)
    noisy = metric.bures_block(sim.probabilities(rho2, noise, t2.n) @ metric.swap_signs(t.n), pairs, m, N)
    # the same circuits read out without errors: the gate errors of the SWAP test alone
    gates = metric.bures_block(sim.probabilities(rho2, replace(noise, p_meas0=0.0, p_meas1=0.0), t2.n) @ metric.swap_signs(t.n),
                               pairs, m, N)
    return dict(state=dict(bures=G_state, purity=purity_state, p=float(geo.depolarizing_from_purity(purity_state, N))),
                noiseless=dict(bures=noiseless["bures"], purity=noiseless["purity"], p=noiseless["p_hat"]),
                perfect_readout=dict(bures=gates["bures"], purity=gates["purity"], p=gates["p_hat"]),
                noisy=dict(bures=noisy["bures"], purity=noisy["purity"], p=noisy["p_hat"]))


def compute(scales=(1.0, 50.0), seed: int = 0, layer: int = 1) -> dict:
    t, x, theta, params = setup(seed, layer)
    G = np.real(sim.pure_qgt(t, t.angles(x, theta))[0])[np.ix_(params, params)]
    rows = []
    for s in scales:
        b = blocks(t, x, theta, params, sim.NoiseModel.reimei().scaled(s))
        row = dict(scale=float(s))
        for key, v in b.items():
            row[key] = dict(attenuation=attenuation(v["bures"], G), trace=float(np.trace(v["bures"])),
                            purity=float(v["purity"]), noise_strength=float(v["p"]), bures=v["bures"].tolist())
        rows.append(row)
    return dict(seed=seed, parameters=params, qubits=t.n, metric_noiseless=G.tolist(),
                trace_noiseless=float(np.trace(G)),
                definition="attenuation = <G_est, G> / <G, G> with the Frobenius inner product and the block G of the "
                           "noiseless metric; state: Bures metric of the state of the local model; noiseless: two-copy "
                           "estimate from the exact overlaps of the states; perfect_readout: two-copy estimate from the "
                           "noisy two-copy circuits of the model read out without errors; noisy: two-copy estimate from "
                           "the noisy two-copy circuits of the model",
                rows=rows)


def compare_with_measurement(result: dict) -> None:
    """The "state" and "noisy" blocks must equal the blocks of the local model stored with the measurement."""
    with open(MEASURED) as f:
        stored = {r["scale"]: r for r in json.load(f)["rows"]}
    for row in result["rows"]:
        r = stored.get(row["scale"])
        if r is None:
            raise AssertionError(f"multiplier {row['scale']:g} has no row in results/metric/reimei-E.json")
        if not (np.allclose(row["state"]["bures"], r["bures_local_exact"], atol=1e-10)
                and np.allclose(row["noisy"]["bures"], r["bures_local_estimator"], atol=1e-10)
                and np.allclose(result["metric_noiseless"], r["metric_noiseless"], atol=1e-10)):
            raise AssertionError(f"multiplier {row['scale']:g}: the blocks differ from those of results/metric/reimei-E.json")
    print("the blocks of the state and of the noisy two-copy circuits equal those stored in results/metric/reimei-E.json")


def show(result: dict) -> None:
    print(f"{'multiplier':>10s} | {'state':>8s} {'noiseless':>10s} {'perfect readout':>16s} {'noisy':>8s} | purities (state, noisy circuits) "
          f"| p (state, noisy circuits)")
    for r in result["rows"]:
        print(f"{r['scale']:>10g} | {r['state']['attenuation']:8.3f} {r['noiseless']['attenuation']:10.3f} "
              f"{r['perfect_readout']['attenuation']:16.3f} {r['noisy']['attenuation']:8.3f} | {r['state']['purity']:.4f}, "
              f"{r['noisy']['purity']:.4f}                | {r['state']['noise_strength']:.4f}, {r['noisy']['noise_strength']:.4f}")


def _close(a, b, tol) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the stored result file instead of writing it")
    a = ap.parse_args(argv)
    result = compute()
    show(result)
    compare_with_measurement(result)
    text = json.dumps(result, indent=1)
    if a.check:
        with open(OUT) as f:
            stored = json.load(f)
        same = _close(stored, json.loads(text), 1e-9)
        print("results/metric/decomposition.json " + ("is reproduced" if same else "DIFFERS from the computed result"))
        if not same:
            sys.exit(1)
    else:
        with open(OUT, "w") as f:
            f.write(text)
        print("written: results/metric/decomposition.json")


if __name__ == "__main__":
    main()
