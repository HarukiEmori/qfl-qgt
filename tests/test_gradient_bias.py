"""Bias of the gradient estimate at a finite number of shots (scripts/gradient_bias.py)."""
import importlib.util
import json
import os
from math import comb

import numpy as np

from qfedqgt import scenarios as S
from qfedqgt import sim
from qfedqgt.experiment import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gradient_bias", os.path.join(ROOT, "scripts", "gradient_bias.py"))
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)


def _trained_setup(seed=0):
    """Model and clients of the two-class experiment and the parameters of QFedQGT after the last round on the emulator."""
    sc = S.SCENARIOS[gb.SCENARIO]
    model, clients, _, _, _ = build(sc, seed)
    run = np.load(os.path.join(ROOT, "results", gb.CAMPAIGN, f"{gb.SCENARIO}-seed{seed}.npz"))
    return sc, model, clients, run["QFedQGT/theta"][-1]


def test_histograms_are_all_count_vectors():
    for shots, outcomes in ((1, 4), (3, 4), (16, 4), (5, 2)):
        H, logc = gb.histogram_table(shots, outcomes)
        assert H.shape == (comb(shots + outcomes - 1, outcomes - 1), outcomes)
        assert np.all(H.sum(axis=1) == shots) and len({tuple(h) for h in H}) == len(H)
        # the multinomial coefficients sum to outcomes ** shots
        assert np.isclose(np.exp(logc).sum(), float(outcomes) ** shots, rtol=1e-10)


def test_expected_derivative_is_the_derivative_for_one_certain_outcome():
    _, model, clients, theta = _trained_setup()
    y = clients[0].y[:6]
    p = np.zeros((len(y), model.template.dim))
    p[np.arange(len(y)), np.arange(len(y)) % model.template.dim] = 1.0     # every sample has one certain outcome
    _, u = model.loss_and_dphi(p @ model.obs, y)
    for shots in (1, 7):
        assert np.allclose(gb.expected_dphi(model, p, y, shots), u, atol=1e-12)


def test_mean_of_the_estimate_agrees_with_sampling():
    sc, model, clients, theta = _trained_setup()
    rng = np.random.default_rng(12345)
    local = sc.local_scales
    for k, reps in ((0, 2000), (2, 4000)):                  # a reference client (16 shots) and a noisy client (1 shot)
        X, y, shots = clients[k].X, clients[k].y, sc.shots[k]
        noise = sim.NoiseModel.reimei().scaled(local[k])
        p = sim.probabilities(sim.density(model.template, model.gradient_angles(X, theta), noise), noise, model.template.n)
        total, total2 = np.zeros(model.template.n_params), np.zeros(model.template.n_params)
        for _ in range(reps):
            g = model.per_sample_gradients(sim.sample_counts(p, shots, rng), y, shots)[0].mean(axis=0)
            total += g
            total2 += g * g
        mc = total / reps
        se = np.sqrt((total2 / reps - mc ** 2) / reps)
        grad, mean = gb.gradient_and_mean_estimate(model, X, y, theta, noise, shots)
        assert np.allclose(grad, model.gradient_exact(X, y, theta, noise), atol=1e-12)
        assert np.abs((mc - mean) / se).max() < 4.0         # the enumerated mean is the mean of the estimator
        if k == 0:
            # for the reference client the samples also resolve the difference from the gradient of the loss
            assert np.abs((mc - grad) / se).max() > 10.0


def test_bias_decreases_as_the_inverse_of_the_shots():
    sc, model, clients, theta = _trained_setup()
    X, y = clients[0].X, clients[0].y
    noise = sim.NoiseModel.reimei()
    bias = {}
    for shots in (32, 64):
        grad, mean = gb.gradient_and_mean_estimate(model, X, y, theta, noise, shots)
        bias[shots] = gb.relative_bias_and_cosine(grad, mean)[0]
    assert 1.7 < bias[32] / bias[64] < 2.1


def test_stored_result_follows_from_the_computation():
    with open(gb.OUT) as f:
        stored = json.load(f)
    sc = S.SCENARIOS[gb.SCENARIO]
    assert stored["seeds"] == list(S.EMULATOR2_SEEDS) and stored["rounds"] == S.EMULATOR2_RUN.rounds
    assert [c["shots"] for c in stored["clients"]] == list(sc.shots)
    assert [c["multiplier_model"] for c in stored["clients"]] == list(sc.local_scales)
    # one seed and one method are computed again
    part = gb.compute(seeds=(3,), methods=("QFedQGT",))
    assert np.allclose(part["relative_bias"]["QFedQGT"][0], stored["relative_bias"]["QFedQGT"][3], atol=2e-6)
    assert np.allclose(part["cosine"]["QFedQGT"][0], stored["cosine"]["QFedQGT"][3], atol=2e-6)
    # the summary follows from the stored arrays
    roles = np.array([c["role"] for c in stored["clients"]])
    rounds = np.arange(1, stored["rounds"] + 1)
    for name in gb.METHODS:
        b, c = np.array(stored["relative_bias"][name]), np.array(stored["cosine"][name])
        assert b.shape == (len(stored["seeds"]), stored["rounds"], len(roles))
        for role in ("reference", "noisy"):
            again = gb.summarize(b[:, :, roles == role], c[:, :, roles == role], rounds)
            for key, v in stored["summary"][name][role].items():
                assert v["n"] == again[key]["n"]
                assert abs(v["relative_bias"]["median"] - again[key]["relative_bias"]["median"]) < 2e-6
                assert abs(v["cosine"]["median"] - again[key]["cosine"]["median"]) < 2e-6
                assert abs(v["cosine_below_threshold"] - again[key]["cosine_below_threshold"]) < 1e-12


def test_relative_bias_is_not_defined_for_a_vanishing_gradient():
    import pytest

    with pytest.raises(ArithmeticError):
        gb.relative_bias_and_cosine(np.zeros(4), np.ones(4))
