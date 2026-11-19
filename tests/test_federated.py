"""Tests of the estimators, the aggregation rules, and the federated loop."""
import numpy as np
import pytest

from qfedqgt import geometry as geo
from qfedqgt import sim
from qfedqgt.experiment import Scenario, build, make_methods, simulate
from qfedqgt.federated import (LocalBackend, LocalExecutor, RunConfig, between_client_variance,
                               efficiency_ratio, fisher_aggregate, random_effects_weights, run_federated)
from qfedqgt.model import VQC


def test_random_effects_weights_limits():
    q = np.array([0.2, 0.3, 0.5])
    s2 = np.array([0.1, 1.0, 10.0])
    w0 = random_effects_weights(q, s2, 0.0)
    assert np.allclose(w0, (1 / s2) / (1 / s2).sum())            # inverse variance
    winf = random_effects_weights(q, s2, 1e12)
    assert np.allclose(winf, q, atol=1e-9)                        # data shares
    # homogeneous devices and equal shares: uniform weights for every heterogeneity
    assert np.allclose(random_effects_weights(np.full(4, 0.25), np.full(4, 2.0), 0.3), 0.25)
    assert efficiency_ratio(np.full(4, 0.25), np.full(4, 2.0)) == pytest.approx(1.0)
    assert efficiency_ratio(q, s2) > 1.0


def test_between_client_variance_recovers_planted_value():
    rng = np.random.default_rng(0)
    K, d, tau2 = 400, 5, 2.0
    sig = rng.uniform(0.5, 3.0, size=K)
    upd = rng.normal(size=(K, d)) * np.sqrt(tau2 / d) + rng.normal(size=(K, d)) * np.sqrt(sig / d)[:, None]
    assert between_client_variance(upd, sig) == pytest.approx(tau2, rel=0.15)
    assert between_client_variance(rng.normal(size=(K, d)) * np.sqrt(sig / d)[:, None], sig) < 0.2


def test_fisher_aggregate_mask_falls_back_to_data_shares():
    thetas = np.array([[1.0, 0.0], [3.0, 2.0]])
    grads = np.array([[1.0, 2.0], [1.0, 2.0]])            # min-max normalized Fisher: first parameter 0, second 1
    share = np.array([0.25, 0.75])
    out, Fn = fisher_aggregate(thetas, grads, share, [(0, 1)], delta=0.01)
    assert out[0] == pytest.approx(0.25 * 1.0 + 0.75 * 3.0)   # masked parameter: data-share average
    assert out[1] == pytest.approx(1.0)                        # equal Fisher weights
    assert np.allclose(Fn, [[0, 1], [0, 1]])


def test_gradient_estimator_is_unbiased_for_exact_gradient():
    """The shot-based parameter-shift estimator converges to the exact gradient."""
    t = sim.layered_ansatz(3, 1)
    model = VQC(t, 2, beta=4.0)
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(3, t.n_data))
    y = np.array([0, 1, 0])
    th = rng.uniform(-1, 1, size=t.n_params)
    noise = sim.NoiseModel(p1=0.003, p2=0.03)
    exact = model.gradient_exact(X, y, th, noise)
    est = model.estimate_gradient(LocalBackend(noise, seed=0), X, y, th, shots=200_000)
    assert np.abs(est.grad - exact).max() < 5e-3
    assert est.var.shape == exact.shape and (est.var > 0).all()


def test_purity_estimate_and_bures_attenuation():
    t = sim.layered_ansatz(4, 2)
    model = VQC(t, 2)
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(6, t.n_data))
    th = rng.uniform(-1, 1, size=t.n_params)
    noise = sim.NoiseModel(p1=0.002, p2=0.05)
    gamma, _ = model.estimate_purity(LocalBackend(noise, seed=3), X, th, shots=20_000)
    rho = sim.density(t, t.angles(X, th), noise)
    purity = np.real(np.einsum("cab,cba->c", rho, rho)).mean()
    assert gamma == pytest.approx(purity, abs=0.01)
    assert 0.0 < geo.depolarizing_from_purity(gamma, t.dim) < 1.0


def test_federated_loop_runs_and_homogeneous_weights_are_uniform():
    sc = Scenario(n_qubits=3, layers=1, n_per_client=12, n_test_per_class=10,
                  gate_scales=(1.0, 1.0, 1.0, 1.0), shots=(40, 40, 40, 40))
    cfg = RunConfig(rounds=4, batch=4, echo_shots=40)
    out = simulate(sc, {"QFedAvg": 0.5, "QFedFisher": 0.3, "QFedQGT": 0.1}, cfg, seed=0)
    for m in ("QFedAvg", "QFedFisher", "QFedQGT"):
        h = out[m]
        assert h["acc_exact"].shape == (5,) and np.isfinite(h["loss_exact"]).all()
        assert np.allclose(h["weights"].sum(axis=1), 1.0)
    assert np.allclose(out["QFedAvg"]["weights"], 0.25)
    # first round of QFedQGT is the calibration round: parameters unchanged
    assert np.allclose(out["QFedQGT"]["theta"][1], out["QFedQGT"]["theta"][0])
    assert out["_usage"]["n_circuits"] > 0


def test_noisy_clients_are_downweighted():
    sc = Scenario(n_qubits=3, layers=1, n_per_client=12, n_test_per_class=10,
                  gate_scales=(1.0, 1.0, 150.0, 150.0), shots=(60, 60, 4, 4))
    cfg = RunConfig(rounds=5, batch=6, echo_shots=40)
    h = simulate(sc, {"QFedQGT": 0.1}, cfg, seed=1, evaluate_on_device=False)["QFedQGT"]
    w = h["weights"][-1]
    assert w[:2].sum() > 0.85 and w[2:].max() < 0.1
    p_hat = np.asarray(h["p_hat"][-1], float)
    assert p_hat[:2].max() < 0.1 and p_hat[2:].min() > 0.2
