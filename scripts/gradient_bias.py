#!/usr/bin/env python
"""Bias of the gradient estimate at a finite number of shots, for the two-class experiment on the emulator.

The gradient estimate of one sample (``VQC.per_sample_gradients``) is a product of two factors that come from
independent circuits: the derivative ``u`` of the loss with respect to the features, evaluated at the features
measured with the unshifted circuit, and the parameter-shift estimate of the Jacobian ``J`` of the features.
Its mean over the measurement outcomes is therefore ``E[u(phi_hat)]^T J``, whereas the gradient of the loss is
``u(phi)^T J``.  The two agree for a loss that is affine in the features, and they differ for the cross-entropy
at a finite number of shots.

The script evaluates both vectors without sampling in the local gate-level model of the devices of the
experiment (reference clients: multiplier 1 and 16 shots per circuit; noisy clients: the matched multiplier of
the model and 1 shot per circuit).  The mean of ``u`` is a sum over all outcome histograms of the unshifted
circuit.  The vectors are means over the local data of a client, which is the mean of the minibatch estimate
over the measurement outcomes and over the draw of the minibatch.  They are evaluated for every client at the
parameters of the runs on the emulator (every method, every seed, the start of every round).  Reported are

    relative bias = |E[g_hat] - grad f| / |grad f|    and    cosine = E[g_hat] . grad f / (|E[g_hat]| |grad f|).

    python scripts/gradient_bias.py            # computes, prints, and writes the result file
    python scripts/gradient_bias.py --check    # computes and compares with the stored result file

Output: text and ``results/emulator2-gradient-bias.json``.  The script submits nothing and does not connect to
Nexus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from math import lgamma

import numpy as np
from scipy.special import xlogy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import scenarios as S          # noqa: E402
from qfedqgt import sim                     # noqa: E402
from qfedqgt.experiment import build        # noqa: E402

SCENARIO = "emulator2_iid"
CAMPAIGN = "emulator2-reimei-E"
METHODS = ("QFedAvg", "QFedFisher", "QFedQGT")
OUT = os.path.join(ROOT, "results", "emulator2-gradient-bias.json")
ROUND_RANGES = ((1, 5), (6, 15), (16, 25))       # first and last round of every range (1-based)
COSINE_THRESHOLD = 0.9


def histograms(shots: int, outcomes: int):
    """All vectors of ``outcomes`` non-negative counts that sum to ``shots``."""
    if outcomes == 1:
        yield (shots,)
        return
    for c in range(shots + 1):
        for rest in histograms(shots - c, outcomes - 1):
            yield (c,) + rest


@lru_cache(maxsize=None)
def histogram_table(shots: int, outcomes: int):
    """Count vectors (H, outcomes) and the logarithms of their multinomial coefficients (H,)."""
    H = np.array(list(histograms(shots, outcomes)), dtype=float)
    logc = lgamma(shots + 1) - np.array([sum(lgamma(c + 1) for c in h) for h in H])
    return H, logc


def expected_dphi(model, p: np.ndarray, y: np.ndarray, shots: int) -> np.ndarray:
    """Mean of the derivative of the loss with respect to the features, evaluated at measured features.

    ``p`` (B, D) holds the outcome probabilities of the unshifted circuit of every sample and ``y`` (B,) the
    labels.  The features are estimated from ``shots`` shots.  Returns an array (B, classes).
    """
    H, logc = histogram_table(shots, p.shape[1])
    # xlogy(0, 0) = 0: a histogram without counts on an impossible outcome keeps its weight
    w = np.exp(logc[None, :] + xlogy(H[None, :, :], p[:, None, :]).sum(axis=2))    # (B, H) multinomial probabilities
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-9):
        raise ArithmeticError("the multinomial probabilities do not sum to one")
    phi = H @ model.obs / shots                                    # (H, classes) features of every histogram
    out = np.empty((len(y), model.obs.shape[1]))
    for cls in np.unique(y):
        _, u = model.loss_and_dphi(phi, np.full(len(H), cls))
        out[y == cls] = w[y == cls] @ u
    return out


def gradient_and_mean_estimate(model, X, y, theta, noise: sim.NoiseModel, shots: int):
    """Gradient of the mean loss over ``(X, y)`` on the device and the mean of its estimate with ``shots`` shots."""
    t = model.template
    A = t.angles(X, theta)
    p0 = sim.probabilities(sim.density(t, A, noise), noise, t.n)   # (B, D)
    _, u = model.loss_and_dphi(p0 @ model.obs, y)
    shifted = sim.shifted_angles(t, A).reshape(len(y) * t.n_params * 2, -1)
    phi = (sim.probabilities(sim.density(t, shifted, noise), noise, t.n) @ model.obs).reshape(len(y), t.n_params, 2, -1)
    J = 0.5 * (phi[:, :, 0] - phi[:, :, 1])                         # (B, d, classes)
    grad = np.einsum("bdf,bf->d", J, u) / len(y)
    mean = np.einsum("bdf,bf->d", J, expected_dphi(model, p0, y, shots)) / len(y)
    return grad, mean


def relative_bias_and_cosine(grad: np.ndarray, mean: np.ndarray):
    ng, nm = float(np.linalg.norm(grad)), float(np.linalg.norm(mean))
    if ng == 0.0 or nm == 0.0:
        raise ArithmeticError("the gradient or the mean of its estimate vanishes; the relative bias or the cosine is not defined")
    return float(np.linalg.norm(mean - grad)) / ng, float(grad @ mean) / (ng * nm)


def _quantiles(x: np.ndarray) -> dict:
    return dict(median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))


def summarize(bias: np.ndarray, cosine: np.ndarray, rounds: np.ndarray) -> dict:
    """Statistics of the arrays (seeds, rounds, clients) of one role and one method."""
    def stats(sel) -> dict:
        b, c = bias[:, sel].ravel(), cosine[:, sel].ravel()
        return dict(n=int(b.size), relative_bias=_quantiles(b), cosine=_quantiles(c),
                    cosine_below_threshold=float(np.mean(c < COSINE_THRESHOLD)))
    out = dict(all=stats(np.ones(len(rounds), bool)))
    for lo, hi in ROUND_RANGES:
        out[f"rounds_{lo}_{hi}"] = stats((rounds >= lo) & (rounds <= hi))
    return out


def compute(seeds=S.EMULATOR2_SEEDS, methods=METHODS) -> dict:
    sc = S.SCENARIOS[SCENARIO]
    local = sc.local_scales or sc.gate_scales
    roles = ["reference" if s == 1.0 else "noisy" for s in sc.gate_scales]
    n_rounds = S.EMULATOR2_RUN.rounds
    rounds = np.arange(1, n_rounds + 1)
    bias = np.empty((len(methods), len(seeds), n_rounds, sc.n_clients))
    cosine = np.empty_like(bias)
    noise = [sim.NoiseModel.reimei().scaled(s) for s in local]
    for j, seed in enumerate(seeds):
        model, clients, _, _, _ = build(sc, seed)
        run = np.load(os.path.join(ROOT, "results", CAMPAIGN, f"{SCENARIO}-seed{seed}.npz"))
        for i, name in enumerate(methods):
            thetas = run[f"{name}/theta"]                            # (rounds + 1, d); row r: start of round r + 1
            if thetas.shape[0] != n_rounds + 1:
                raise ValueError(f"seed {seed}, {name}: {thetas.shape[0] - 1} rounds are stored, expected {n_rounds}")
            for r in range(n_rounds):
                for k, client in enumerate(clients):
                    grad, mean = gradient_and_mean_estimate(model, client.X, client.y, thetas[r], noise[k], sc.shots[k])
                    bias[i, j, r, k], cosine[i, j, r, k] = relative_bias_and_cosine(grad, mean)
    result = dict(scenario=SCENARIO, campaign=CAMPAIGN, seeds=list(seeds), rounds=int(n_rounds),
                  clients=[dict(role=roles[k], multiplier_emulator=sc.gate_scales[k], multiplier_model=local[k],
                                shots=sc.shots[k], samples=int(sc.n_per_client)) for k in range(sc.n_clients)],
                  round_ranges=[list(r) for r in ROUND_RANGES], cosine_threshold=COSINE_THRESHOLD,
                  definition="relative bias = |E[g_hat] - grad f| / |grad f| and cosine of the angle between E[g_hat] and "
                             "grad f, for the local data of a client at the parameters at the start of a round",
                  summary={}, relative_bias={}, cosine={})
    for i, name in enumerate(methods):
        result["summary"][name] = {}
        for role in ("reference", "noisy"):
            sel = [k for k in range(sc.n_clients) if roles[k] == role]
            result["summary"][name][role] = summarize(bias[i][:, :, sel], cosine[i][:, :, sel], rounds)
        # arrays (seeds, rounds, clients), rounded: enough to recompute every statistic of the summary
        result["relative_bias"][name] = np.round(bias[i], 6).tolist()
        result["cosine"][name] = np.round(cosine[i], 6).tolist()
    return result


def show(result: dict) -> None:
    for name, by_role in result["summary"].items():
        for role, s in by_role.items():
            print(f"{name:11s} {role:9s}", end="")
            for key, v in s.items():
                b, c = v["relative_bias"], v["cosine"]
                label = "all rounds" if key == "all" else "rounds " + key.split("_", 1)[1].replace("_", "-")
                print(f"  | {label}: bias {b['median']:.2f} ({b['q25']:.2f}, {b['q75']:.2f}), cosine {c['median']:.3f}, "
                      f"below {result['cosine_threshold']:g}: {v['cosine_below_threshold']:.2f}", end="")
            print(f"  [n = {s['all']['n']}]")


def _close(a, b, tol=1e-9) -> bool:
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
    text = json.dumps(result, indent=1)
    if a.check:
        with open(OUT) as f:
            stored = json.load(f)
        same = _close(stored, json.loads(text), tol=1e-6)
        print("results/emulator2-gradient-bias.json " + ("is reproduced" if same else "DIFFERS from the computed result"))
        if not same:
            sys.exit(1)
    else:
        with open(OUT, "w") as f:
            f.write(text)
        print("written: results/emulator2-gradient-bias.json")


if __name__ == "__main__":
    main()
