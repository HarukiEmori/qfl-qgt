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


def _decomposition():
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("metric_decomposition", os.path.join(root, "scripts", "metric_decomposition.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decomposition_noiseless_measurement_is_exact_for_global_depolarizing_noise_and_not_for_gate_level_noise():
    md = _decomposition()
    t, x, theta = _setup()
    params = [0, 1, 4, 7]
    G = np.real(sim.pure_qgt(t, t.angles(x, theta))[0])[np.ix_(params, params)]
    # global depolarizing noise: the conversion of the Hilbert-Schmidt block is exact
    b = md.blocks(t, x, theta, params, sim.NoiseModel(global_p=0.3))
    assert np.allclose(b["noiseless"]["bures"], b["state"]["bures"], atol=1e-8)
    assert abs(b["noiseless"]["p"] - 0.3) < 1e-8 and abs(b["state"]["p"] - 0.3) < 1e-8
    assert abs(md.attenuation(b["state"]["bures"], G) - geo.kappa(0.3, t.dim)) < 1e-8
    # gate-level noise: the conversion is not exact, and the noisy two-copy circuits change the estimate again
    b = md.blocks(t, x, theta, params, sim.NoiseModel.reimei().scaled(100.0))
    c = {key: md.attenuation(v["bures"], G) for key, v in b.items()}
    assert abs(c["noiseless"] - c["state"]) > 1e-3 and abs(c["noisy"] - c["noiseless"]) > 1e-3
    assert abs(b["noiseless"]["purity"] - b["state"]["purity"]) < 1e-12


def test_stored_decomposition_matches_the_blocks_stored_with_the_measurement():
    import json

    md = _decomposition()
    with open(md.OUT) as f:
        stored = json.load(f)
    with open(md.MEASURED) as f:
        measured = {r["scale"]: r for r in json.load(f)["rows"]}
    G = np.array(stored["metric_noiseless"])
    assert [r["scale"] for r in stored["rows"]] == sorted(measured)
    for row in stored["rows"]:
        r = measured[row["scale"]]
        assert np.allclose(row["state"]["bures"], r["bures_local_exact"], atol=1e-10)
        assert np.allclose(row["noisy"]["bures"], r["bures_local_estimator"], atol=1e-10)
        assert np.allclose(G, r["metric_noiseless"], atol=1e-10)
        for key in ("state", "noiseless", "perfect_readout", "noisy"):
            assert abs(row[key]["attenuation"] - md.attenuation(np.array(row[key]["bures"]), G)) < 1e-12
        # the gate errors of the SWAP test lower the estimate, and the readout errors lower it further
        assert row["noiseless"]["attenuation"] > row["perfect_readout"]["attenuation"] > row["noisy"]["attenuation"]
        # the purity of the noiseless measurement is the purity of the state
        assert abs(row["noiseless"]["purity"] - row["state"]["purity"]) < 1e-12
        assert abs(row["noiseless"]["purity"] - r["purity_local_exact"]) < 1e-10


def test_two_copy_preparation_is_the_product_of_the_single_copy_states():
    # before the SWAP test the model prepares the two copies independently, with the noise of the single-copy circuit
    t, x, theta = _setup()
    t2 = metric.two_copy_template(t)
    n_prep = 2 * len(t.gates)
    prep = sim.Template(t2.n, tuple(t2.gates[:n_prep]), t2.n_params, t2.n_data, (),
                        tuple(e for e in t2.entangling_layers if e < n_prep))
    rng = np.random.default_rng(5)
    TA = theta + rng.uniform(-1, 1, size=(4, t.n_params))
    TB = theta + rng.uniform(-1, 1, size=(4, t.n_params))
    noise = sim.NoiseModel.reimei().scaled(100.0)
    rho2 = sim.density(prep, _two_copy_angles(prep, x, TA, TB), noise)
    X = np.tile(x, (4, 1))
    ra, rb = sim.density(t, t.angles(X, TA), noise), sim.density(t, t.angles(X, TB), noise)
    for i in range(4):
        assert np.allclose(rho2[i], np.kron(ra[i], rb[i]), atol=1e-12)


def test_phase_flips_after_the_swap_test_do_not_change_the_outcome_probabilities():
    from dataclasses import replace

    t, x, theta = _setup()
    t2 = metric.two_copy_template(t)
    assert t2.entangling_layers[-1] == len(t2.gates) - 1          # the last idle layer follows the last gate
    without = replace(t2, entangling_layers=t2.entangling_layers[:-1])
    TA, TB, _ = metric.overlap_plan(theta, [0, 3])
    A2 = _two_copy_angles(t2, x, TA, TB)
    noise = sim.NoiseModel.reimei().scaled(100.0)
    p = sim.probabilities(sim.density(t2, A2, noise), noise, t2.n)
    q = sim.probabilities(sim.density(without, A2, noise), noise, t2.n)
    assert np.array_equal(p, q)
    # an idle layer in the middle of the circuit does change them
    middle = replace(t2, entangling_layers=t2.entangling_layers[1:])
    assert np.abs(p - sim.probabilities(sim.density(middle, A2, noise), noise, t2.n)).max() > 1e-6
