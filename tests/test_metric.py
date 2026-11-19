"""Two-copy measurement of the Bures metric: the circuits and the estimators against exact values."""
import numpy as np

from qfedqgt import geometry as geo
from qfedqgt import metric, sim


def _setup(n=2, layers=2, seed=0):
    t = sim.layered_ansatz(n, layers)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=t.n_data)
    theta = rng.uniform(-1, 1, size=t.n_params)
    return t, x, theta


def _two_copy_angles(t2, x, TA, TB):
    X = np.tile(np.concatenate([x, x]), (len(TA), 1))
    return t2.angles(X, np.concatenate([TA, TB], axis=1))


def test_swap_test_gives_the_overlap_of_pure_states():
    t, x, theta = _setup()
    t2 = metric.two_copy_template(t)
    rng = np.random.default_rng(1)
    TA = theta + rng.uniform(-1, 1, size=(6, t.n_params))
    TB = theta + rng.uniform(-1, 1, size=(6, t.n_params))
    p = sim.probabilities(sim.density(t2, _two_copy_angles(t2, x, TA, TB)))
    measured = p @ metric.swap_signs(t.n)
    X = np.tile(x, (6, 1))
    a = sim.statevector(t, t.angles(X, TA))
    b = sim.statevector(t, t.angles(X, TB))
    exact = np.abs(np.einsum("ci,ci->c", a.conj(), b)) ** 2
    assert np.allclose(measured, exact, atol=1e-10)


def test_bures_block_matches_the_exact_metric_without_noise():
    t, x, theta = _setup()
    t2 = metric.two_copy_template(t)
    params = [0, 1, 4, 7]
    TA, TB, pairs = metric.overlap_plan(theta, params)
    assert len(pairs) == 2 * len(params) * (2 * len(params) + 1) // 2 and len(TA) == len(pairs) + 1
    p = sim.probabilities(sim.density(t2, _two_copy_angles(t2, x, TA, TB)))
    out = metric.bures_block(p @ metric.swap_signs(t.n), pairs, len(params), t.dim)
    Q = sim.pure_qgt(t, t.angles(x, theta))[0]
    assert abs(out["purity"] - 1.0) < 1e-10 and abs(out["p_hat"]) < 1e-6
    assert np.allclose(out["bures"], Q.real[np.ix_(params, params)], atol=1e-8)


def test_bures_block_matches_the_exact_metric_with_global_depolarizing_noise():
    t, x, theta = _setup()
    N, p_dep = t.dim, 0.3
    noise = sim.NoiseModel(global_p=p_dep)
    params = [2, 3, 5]
    TA, TB, pairs = metric.overlap_plan(theta, params)
    X = np.tile(x, (len(TA), 1))
    ra = sim.density(t, t.angles(X, TA), noise)
    rb = sim.density(t, t.angles(X, TB), noise)
    overlaps = np.real(np.einsum("cij,cji->c", ra, rb))        # what a noiseless SWAP test returns
    out = metric.bures_block(overlaps, pairs, len(params), N)
    assert abs(out["p_hat"] - p_dep) < 1e-8
    rho, drho = sim.density_derivatives(t, t.angles(x, theta), noise)
    G = geo.bures_metric(geo.sld_qgt(rho[0], drho[0]))
    assert np.allclose(out["bures"], G[np.ix_(params, params)], atol=1e-8)


def test_estimator_from_counts_is_unbiased_within_the_statistical_error():
    t, x, theta = _setup()
    t2 = metric.two_copy_template(t)
    params = [0, 1]
    TA, TB, pairs = metric.overlap_plan(theta, params)
    p = sim.probabilities(sim.density(t2, _two_copy_angles(t2, x, TA, TB)))
    shots = 4000
    counts = sim.sample_counts(p, shots, np.random.default_rng(3))
    est = metric.overlaps_from_counts(counts, shots, t.n)
    assert np.abs(est - p @ metric.swap_signs(t.n)).max() < 5.0 / np.sqrt(shots)
