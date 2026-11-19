"""Tests of the simulators and of the geometric identities stated in the paper."""
import numpy as np
import pytest
from scipy.linalg import expm

from qfedqgt import geometry as geo
from qfedqgt import sim

RNG = np.random.default_rng(11)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)


def kron(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def dense_unitary(template, a):
    """Reference construction of the circuit unitary by explicit Kronecker products."""
    n = template.n
    U = np.eye(2 ** n, dtype=complex)
    P = {"rx": X, "ry": Y, "rz": Z}
    for g, gate in enumerate(template.gates):
        ops = [I2] * n
        if gate.kind == "zz":
            ops[gate.qubits[0]] = Z
            ops[gate.qubits[1]] = Z
        else:
            ops[gate.qubits[0]] = P[gate.kind]
        U = expm(-0.5j * a[g] * kron(ops)) @ U
    return U


@pytest.mark.parametrize("n,layers", [(2, 2), (3, 2), (4, 2)])
def test_statevector_and_density_match_dense(n, layers):
    t = sim.layered_ansatz(n, layers)
    x = RNG.uniform(-1, 1, size=(3, t.n_data))
    th = RNG.uniform(-np.pi, np.pi, size=t.n_params)
    A = t.angles(x, th)
    psi = sim.statevector(t, A)
    for c in range(3):
        ref = dense_unitary(t, A[c])[:, 0]
        assert np.allclose(psi[c], ref, atol=1e-12)
    rho = sim.density(t, A)
    assert np.allclose(rho, np.einsum("ca,cb->cab", psi, psi.conj()), atol=1e-12)


def test_depolarizing_channels_against_pauli_sum():
    n = 3
    D = 2 ** n
    M = RNG.normal(size=(D, D)) + 1j * RNG.normal(size=(D, D))
    rho = M @ M.conj().T
    rho /= np.trace(rho)
    p = 0.23
    paulis = [I2, X, Y, Z]
    # one qubit
    q = 1
    ref = (1 - p) * rho
    for P in paulis[1:]:
        ops = [I2] * n
        ops[q] = P
        K = kron(ops)
        ref = ref + p / 3 * K @ rho @ K.conj().T
    out = sim._depolarize(rho[None], (q,), p, n)[0]
    assert np.allclose(out, ref, atol=1e-12)
    # two qubits
    q, r = 0, 2
    ref = (1 - p) * rho
    for a in range(4):
        for b in range(4):
            if a == 0 and b == 0:
                continue
            ops = [I2] * n
            ops[q] = paulis[a]
            ops[r] = paulis[b]
            K = kron(ops)
            ref = ref + p / 15 * K @ rho @ K.conj().T
    out = sim._depolarize(rho[None], (q, r), p, n)[0]
    assert np.allclose(out, ref, atol=1e-12)


def test_readout_flips():
    n = 2
    noise = sim.NoiseModel(p_meas0=0.1, p_meas1=0.3)
    rho = np.zeros((1, 4, 4), complex)
    rho[0, 1, 1] = 1.0     # state |01>: qubit 0 in 0, qubit 1 in 1
    p = sim.probabilities(rho, noise, n)[0]
    expected = np.array([0.9 * 0.3, 0.9 * 0.7, 0.1 * 0.3, 0.1 * 0.7])   # 00, 01, 10, 11
    assert np.allclose(p, expected)


def test_parameter_shift_derivative_matches_finite_difference():
    t = sim.layered_ansatz(3, 2)
    noise = sim.NoiseModel(p1=0.01, p2=0.05, p_idle=0.02)
    x = RNG.uniform(-1, 1, size=(1, t.n_data))
    th = RNG.uniform(-np.pi, np.pi, size=t.n_params)
    rho, drho = sim.density_derivatives(t, t.angles(x, th), noise)
    eps = 1e-5
    for i in (0, 3, t.n_params - 1):
        tp, tm = th.copy(), th.copy()
        tp[i] += eps
        tm[i] -= eps
        fd = (sim.density(t, t.angles(x, tp), noise) - sim.density(t, t.angles(x, tm), noise)) / (2 * eps)
        assert np.allclose(drho[0, i], fd[0], atol=1e-8)


def test_pure_qgt_matches_finite_difference():
    t = sim.layered_ansatz(3, 2)
    x = RNG.uniform(-1, 1, size=(1, t.n_data))
    th = RNG.uniform(-np.pi, np.pi, size=t.n_params)
    Q = sim.pure_qgt(t, t.angles(x, th))[0]
    psi = sim.statevector(t, t.angles(x, th))[0]
    eps = 1e-6
    dpsi = []
    for i in range(t.n_params):
        tp, tm = th.copy(), th.copy()
        tp[i] += eps
        tm[i] -= eps
        dpsi.append((sim.statevector(t, t.angles(x, tp))[0] - sim.statevector(t, t.angles(x, tm))[0]) / (2 * eps))
    dpsi = np.array(dpsi)
    ref = dpsi.conj() @ dpsi.T - np.outer(dpsi.conj() @ psi, psi.conj() @ dpsi.T)
    assert np.allclose(Q, ref, atol=1e-7)
    assert np.allclose(Q, Q.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(Q).min() > -1e-10


@pytest.mark.parametrize("p", [1e-9, 0.1, 0.4, 0.8])
def test_global_depolarizing_closed_forms(p):
    """Pure-state limit and the attenuation factors kappa, kappa_omega, chi."""
    t = sim.layered_ansatz(3, 2)
    N = t.dim
    x = RNG.uniform(-1, 1, size=(1, t.n_data))
    th = RNG.uniform(-np.pi, np.pi, size=t.n_params)
    A = t.angles(x, th)
    Qp = sim.pure_qgt(t, A)[0]
    rho, drho = sim.density_derivatives(t, A, sim.NoiseModel(global_p=p))
    Q = geo.sld_qgt(rho[0], drho[0])
    g, om = geo.bures_metric(Q), geo.uhlmann_curvature(Q)
    g0, om0 = np.real(Qp), -2 * np.imag(Qp)
    assert np.allclose(g, geo.kappa(p, N) * g0, atol=1e-7)
    assert np.allclose(om, geo.kappa_omega(p, N) * om0, atol=1e-7)
    assert np.abs(om0).max() > 1e-3            # the Berry curvature of this ansatz is not zero
    # incompatibility ratio (restricted to the support of the metric)
    R, R0 = geo.incompatibility_ratio(Q), geo.incompatibility_ratio(Qp)
    assert 0.0 <= R <= 1.0 + 1e-9
    assert R == pytest.approx(geo.chi(p, N) * R0, abs=1e-6)
    # purity and its inversion
    gamma = np.real(np.trace(rho[0] @ rho[0]))
    assert gamma == pytest.approx(geo.purity_of_depolarized(p, N), abs=1e-10)
    assert geo.depolarizing_from_purity(gamma, N) == pytest.approx(p, abs=1e-6)
    # the tensor is positive semidefinite, hence g >= +- (i/2) omega
    assert np.linalg.eigvalsh(Q).min() > -1e-9
    # C <= R F^{-1} on the support
    C = geo.incompatibility_matrix(Q)
    assert np.linalg.eigvalsh(C).min() > -1e-8


def test_echo_survival_equals_purity_for_gate_depolarizing():
    """For depolarizing faults attached to the gates, the echo survival probability is the purity."""
    t = sim.layered_ansatz(4, 2)
    noise = sim.NoiseModel(p1=0.004, p2=0.06)
    x = RNG.uniform(-1, 1, size=(4, t.n_data))
    th = RNG.uniform(-np.pi, np.pi, size=t.n_params)
    rho = sim.density(t, t.angles(x, th), noise)
    purity = np.real(np.einsum("cab,cba->c", rho, rho))
    e = t.echo()
    surv = sim.probabilities(sim.density(e, e.angles(x, th), noise))[:, 0]
    assert np.allclose(surv, purity, atol=1e-10)


def test_block_diagonal_and_sampling():
    t = sim.layered_ansatz(4, 2)
    G = RNG.normal(size=(t.n_params, t.n_params))
    B = geo.block_diagonal(G, t.blocks)
    assert B[0, t.blocks[1][0]] == 0 and B[1, 0] == G[1, 0]
    p = np.array([[0.2, 0.8], [0.5, 0.5]])
    c = sim.sample_counts(p, [100, 50], np.random.default_rng(0))
    assert c.sum(axis=1).tolist() == [100, 50]
