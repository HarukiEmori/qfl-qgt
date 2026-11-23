#!/usr/bin/env python
"""Two-class experiment on the emulator against the local gate-level model.

The script reads the finished campaign of the two-class experiment (``results/emulator2-reimei-E``),
the same runs in the local model with the procedure of the emulator (``results/emulator2-local``), and
the prediction of the plan (``results/sim-emulator2_iid``), and reports

1. the final accuracies of the three methods on the emulator and in the model, and their paired
   differences over the seeds (emulator minus model),
2. the differences QFedQGT minus baseline on the emulator and in the model, and the paired difference of
   these differences,
3. a check of the evaluation on the emulator: for the parameters trained on the emulator and the test
   images of the evaluation, the accuracy of the noiseless circuit, the expected accuracy when the
   features are estimated from the shots of the evaluation on the reference device of the local model,
   and the accuracy measured on the emulator,
4. the noiseless accuracy of the global models after every round, and the estimated noise strengths and
   aggregation weights of QFedQGT.

    python scripts/emulator_model_comparison.py

Output: text and ``results/emulator2-comparison.json``.  The script submits nothing and does not connect
to Nexus.  The expected accuracy of item 3 is exact: for two classes the prediction depends on the counts
only through the sign of a difference of two counts, whose distribution is computed from the binomial
distribution (``prediction_probability``).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.stats import binom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import scenarios as S                                # noqa: E402
from qfedqgt import sim                                           # noqa: E402
from qfedqgt.experiment import build                              # noqa: E402

SCENARIO = "emulator2_iid"
EMULATOR, LOCAL = "emulator2-reimei-E", "emulator2-local"
METHODS = ("QFedAvg", "QFedFisher", "QFedQGT")
OUT = os.path.join(ROOT, "results", "emulator2-comparison.json")
THRESHOLDS = (0.8, 0.85, 0.9)


def prediction_probability(p: np.ndarray, obs: np.ndarray, shots: int) -> np.ndarray:
    """Probability that a two-class model predicts class 0 when its features are estimated from ``shots`` shots.

    ``p`` has shape (images, outcomes) and holds the outcome probabilities, ``obs`` has shape (outcomes, 2) and
    holds the values of the two readout observables on every outcome.  The prediction is the class with the
    larger estimated feature, and class 0 for a tie (``numpy.argmax``).  The estimated difference of the
    features is ``sum_x (obs[x, 0] - obs[x, 1]) n_x / shots``; every outcome contributes a positive value, the
    opposite value, or zero, so the prediction is class 0 exactly if the positive outcomes are at least as
    many as the negative ones.
    """
    d = obs[:, 0] - obs[:, 1]
    values = np.unique(np.abs(d[d != 0]))
    if len(values) > 1:
        raise ValueError("the readout observables do not differ by a single value on the outcomes")
    p_plus, p_minus = p[:, d > 0].sum(axis=1), p[:, d < 0].sum(axis=1)
    p_nonzero = np.clip(p_plus + p_minus, 0.0, 1.0)
    q = np.divide(p_plus, p_nonzero, out=np.full_like(p_plus, 0.5), where=p_nonzero > 0)
    m = np.arange(shots + 1)
    out = np.zeros(len(p))
    for i in range(len(p)):
        weight = binom.pmf(m, shots, p_nonzero[i])                    # number of shots with a nonzero contribution
        # P(n_plus >= m - n_plus) = P(n_plus >= ceil(m / 2)) for n_plus ~ Binomial(m, q)
        at_least = binom.sf(np.ceil(m / 2.0) - 1, m, q[i])
        out[i] = float(weight @ at_least)
    return out


def _mean_se(x) -> dict:
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))))


def _paired(a, b) -> dict:
    d = 100.0 * (np.asarray(a, float) - np.asarray(b, float))
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), higher=int((d > 0).sum()), n=len(d))


def _load(name: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", name, f"{SCENARIO}-seed*.npz"))):
        out[int(os.path.basename(f).split("seed")[-1][:-4])] = np.load(f)
    return out


def compare() -> dict:
    sc, cfg, experiment = S.SCENARIOS[SCENARIO], S.run_config(SCENARIO), S.experiment_of(SCENARIO)
    tail = experiment["tail"]
    spec = importlib.util.spec_from_file_location("run_emulator", os.path.join(ROOT, "scripts", "run_emulator.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    emu, loc = _load(EMULATOR), _load(LOCAL)
    seeds = sorted(emu)
    if sorted(loc) != seeds:
        raise SystemExit("the emulator and the local model have different seeds")
    device = {n: {m: np.array([d[s][f"{m}/acc_device"].mean() for s in seeds]) for m in METHODS} for n, d in (("emulator", emu), ("model", loc))}
    exact = {n: {m: np.array([d[s][f"{m}/acc_exact"][-tail:].mean() for s in seeds]) for m in METHODS} for n, d in (("emulator", emu), ("model", loc))}
    prediction = {}
    for m in METHODS:
        files = sorted(glob.glob(os.path.join(ROOT, "results", f"sim-{SCENARIO}", f"seed*__{m}.npz")))
        prediction[m] = np.array([np.load(f)[f"{m}/acc_shots"][-tail:].mean() for f in files])
        if len(prediction[m]) != len(seeds):
            raise SystemExit(f"the prediction of the plan has {len(prediction[m])} seeds for {m}")

    out = dict(scenario=SCENARIO, seeds=seeds, tail=tail, accuracy={}, method_emulator_minus_model={}, difference={})
    out["accuracy"] = dict(
        emulator={m: dict(device=_mean_se(device["emulator"][m]), noiseless=_mean_se(exact["emulator"][m])) for m in METHODS},
        model_same_procedure={m: dict(device=_mean_se(device["model"][m]), noiseless=_mean_se(exact["model"][m])) for m in METHODS},
        model_prediction={m: _mean_se(prediction[m]) for m in METHODS})
    for m in METHODS:
        out["method_emulator_minus_model"][m] = dict(device=_paired(device["emulator"][m], device["model"][m]),
                                                     noiseless=_paired(exact["emulator"][m], exact["model"][m]),
                                                     device_minus_prediction=_paired(device["emulator"][m], prediction[m]))
    for m in METHODS[:2]:
        d_emu = device["emulator"]["QFedQGT"] - device["emulator"][m]
        d_loc = device["model"]["QFedQGT"] - device["model"][m]
        d_pre = prediction["QFedQGT"] - prediction[m]
        e_emu = exact["emulator"]["QFedQGT"] - exact["emulator"][m]
        e_loc = exact["model"]["QFedQGT"] - exact["model"][m]
        zero = np.zeros(len(seeds))
        out["difference"][m] = dict(
            emulator=_paired(d_emu, zero), model_same_procedure=_paired(d_loc, zero), model_prediction=_paired(d_pre, zero),
            emulator_minus_model_same_procedure=_paired(d_emu, d_loc), emulator_minus_model_prediction=_paired(d_emu, d_pre),
            noiseless=dict(emulator=_paired(e_emu, zero), model_same_procedure=_paired(e_loc, zero),
                           emulator_minus_model_same_procedure=_paired(e_emu, e_loc)))

    # -- evaluation on the emulator: noiseless, expected with the shots of the evaluation, measured
    rounds = list(range(cfg.rounds + 1 - tail, cfg.rounds + 1))
    n_classes = len(sc.classes)
    per_class = experiment["eval_inputs"] // n_classes
    noise = sim.NoiseModel.reimei()
    blocks = {m: [] for m in METHODS}
    expected = {m: [] for m in METHODS}
    for s in seeds:
        model, _, _, ds, _ = build(sc, s)
        t = model.template
        for m in METHODS:
            theta = emu[s][f"{m}/theta"]
            a, b = [], []
            for j, r in enumerate(rounds):
                idx = runner.eval_indices(ds.y_test, n_classes, per_class, j if experiment["eval_rotate"] else 0)
                X, y = ds.X_test[idx], ds.y_test[idx]
                a.append(model.accuracy_exact(X, y, theta[r]))
                p = sim.probabilities(sim.density(t, t.angles(X, theta[r]), noise), noise, t.n)
                p0 = prediction_probability(p, model.obs, cfg.eval_shots)
                b.append(float(np.where(y == 0, p0, 1.0 - p0).mean()))
            blocks[m].append(np.mean(a))
            expected[m].append(np.mean(b))
    out["evaluation"] = dict(shots=cfg.eval_shots, images=experiment["eval_inputs"] * tail, methods={})
    for m in METHODS:
        out["evaluation"]["methods"][m] = dict(noiseless_on_the_evaluation_images=float(np.mean(blocks[m])),
                                               expected_with_shots=float(np.mean(expected[m])),
                                               measured=float(device["emulator"][m].mean()),
                                               measured_minus_expected=_paired(device["emulator"][m], expected[m]))

    # -- noiseless accuracy after every round, mean over the seeds
    curves = {m: np.array([emu[s][f"{m}/acc_exact"] for s in seeds]).mean(axis=0) for m in METHODS}
    out["curves"] = {m: [float(x) for x in curves[m]] for m in METHODS}
    out["rounds_with_qfedqgt_highest"] = [int(r) for r in range(len(curves["QFedQGT"]))
                                          if curves["QFedQGT"][r] > max(curves["QFedAvg"][r], curves["QFedFisher"][r])]
    out["first_round_with_mean_at_least"] = {f"{thr:g}": {m: next((int(r) for r in range(len(curves[m])) if curves[m][r] >= thr), None)
                                                           for m in METHODS} for thr in THRESHOLDS}
    out["seeds_with_qfedqgt_highest"] = int(sum(device["emulator"]["QFedQGT"][i] > max(device["emulator"]["QFedAvg"][i], device["emulator"]["QFedFisher"][i])
                                                for i in range(len(seeds))))

    # -- estimated noise strength and aggregation weights of QFedQGT
    n_ref = sum(1 for g in sc.gate_scales if g == 1.0)
    out["qfedqgt"] = {}
    for n, d in (("emulator", emu), ("model", loc)):
        p_hat = np.array([d[s]["QFedQGT/p_hat"][-1] for s in seeds])
        w = np.array([d[s]["QFedQGT/weights"][-tail:] for s in seeds])[:, :, :n_ref].sum(axis=2)
        out["qfedqgt"][n] = dict(p_hat_reference=float(p_hat[:, :n_ref].mean()), p_hat_noisy=float(p_hat[:, n_ref:].mean()),
                                 p_hat_noisy_min=float(p_hat[:, n_ref:].min()), p_hat_noisy_max=float(p_hat[:, n_ref:].max()),
                                 weight_reference=float(w.mean()), weight_reference_min=float(w.min()))
    return out


def report(out: dict) -> None:
    acc, diff = out["accuracy"], out["difference"]
    print(f"{len(out['seeds'])} seeds; final accuracy as the mean over the last {out['tail']} rounds")
    for name, key in (("emulator", "emulator"), ("model, same procedure", "model_same_procedure")):
        print(f"  {name:22s} device    " + "  ".join(f"{m} {acc[key][m]['device']['mean']:.3f} +/- {acc[key][m]['device']['se']:.3f}" for m in METHODS))
        print(f"  {name:22s} noiseless " + "  ".join(f"{m} {acc[key][m]['noiseless']['mean']:.3f} +/- {acc[key][m]['noiseless']['se']:.3f}" for m in METHODS))
    print(f"  {'prediction of the plan':22s} device    " + "  ".join(f"{m} {acc['model_prediction'][m]['mean']:.3f} +/- {acc['model_prediction'][m]['se']:.3f}" for m in METHODS))
    print("emulator minus model for every method (percentage points, paired over the seeds)")
    for m in METHODS:
        r = out["method_emulator_minus_model"][m]
        print(f"  {m:11s} device {r['device']['mean']:+5.1f} +/- {r['device']['se']:.1f}   noiseless {r['noiseless']['mean']:+5.1f} +/- {r['noiseless']['se']:.1f}")
    print("QFedQGT minus baseline (percentage points)")
    for m in METHODS[:2]:
        d = diff[m]
        print(f"  {m:11s} emulator {d['emulator']['mean']:+5.1f} +/- {d['emulator']['se']:.1f} (higher in {d['emulator']['higher']} of {d['emulator']['n']});"
              f" model, same procedure {d['model_same_procedure']['mean']:+5.1f} +/- {d['model_same_procedure']['se']:.1f};"
              f" prediction {d['model_prediction']['mean']:+5.1f} +/- {d['model_prediction']['se']:.1f}")
        print(f"  {'':11s} emulator minus model, same procedure {d['emulator_minus_model_same_procedure']['mean']:+5.1f} +/- {d['emulator_minus_model_same_procedure']['se']:.1f};"
              f" emulator minus prediction {d['emulator_minus_model_prediction']['mean']:+5.1f} +/- {d['emulator_minus_model_prediction']['se']:.1f}")
    ev = out["evaluation"]
    print(f"evaluation on the emulator ({ev['images']} images, {ev['shots']} shots per image)")
    for m in METHODS:
        e = ev["methods"][m]
        print(f"  {m:11s} noiseless {e['noiseless_on_the_evaluation_images']:.3f}   expected with the shots {e['expected_with_shots']:.3f}   measured {e['measured']:.3f}"
              f"   measured minus expected {e['measured_minus_expected']['mean']:+5.1f} +/- {e['measured_minus_expected']['se']:.1f}")
    print("rounds in which QFedQGT has the highest mean noiseless accuracy:", out["rounds_with_qfedqgt_highest"])
    for thr, rows in out["first_round_with_mean_at_least"].items():
        print(f"  first round with a mean noiseless accuracy of at least {thr}: " + ", ".join(f"{m} {rows[m]}" for m in METHODS))
    print(f"seeds in which QFedQGT has the highest accuracy on the emulator: {out['seeds_with_qfedqgt_highest']} of {len(out['seeds'])}")
    for n, q in out["qfedqgt"].items():
        print(f"QFedQGT, {n}: estimated noise strength {q['p_hat_reference']:.3f} (reference), {q['p_hat_noisy']:.3f} (noisy, {q['p_hat_noisy_min']:.2f} to {q['p_hat_noisy_max']:.2f});"
              f" weight of the reference clients {q['weight_reference']:.3f} (minimum {q['weight_reference_min']:.3f})")


def main() -> None:
    out = compare()
    report(out)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
