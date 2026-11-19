"""Measurement of the Bures metric with two copies of the state.

For a depolarized state ``rho = (1 - p) |psi><psi| + p 1/N`` the Bures metric is

    g_ij = Tr[(d_i rho)(d_j rho)] / [2 (1 - p + 2 p / N)],

and the Hilbert-Schmidt overlaps on the right-hand side are measurable with two copies of the
state (Koczor and Benjamin, Phys. Rev. A 106, 062416).  This module builds the two-copy circuits,
which end with a destructive SWAP test, and turns their counts into the overlaps
``Tr[rho_a rho_b]``, the purity, the noise strength, and a block of the Bures metric.

The derivative of the state follows from the parameter-shift rule,
``d_i rho = [rho(theta + pi/2 e_i) - rho(theta - pi/2 e_i)] / 2``, so that

    Tr[(d_i rho)(d_j rho)] = (1/4) sum_{s, s' = +-1} s s' Tr[rho(theta + s pi/2 e_i) rho(theta + s' pi/2 e_j)].
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import geometry as geo
from . import sim

__all__ = ["two_copy_template", "swap_signs", "overlap_plan", "overlaps_from_counts", "bures_block"]


def two_copy_template(t: sim.Template) -> sim.Template:
    """Template on ``2 n`` qubits: two copies of ``t`` prepared side by side, then a destructive SWAP test.

    Copy A uses the qubits ``0 .. n-1`` and the parameters ``0 .. d-1``; copy B uses the qubits
    ``n .. 2n-1`` and the parameters ``d .. 2d-1`` (likewise for the input components).  The SWAP
    test applies ``CNOT(q -> q + n)`` and a Hadamard gate on ``q`` for every qubit ``q`` of copy A,
    written with the native rotations (``H = Ry(pi/2) Rz(pi)`` and ``CZ = Rz(-pi/2) Rz(-pi/2) ZZ(pi/2)``
    up to global phases).
    """
    n = t.n
    gates, ent = [], []
    for g, gate in enumerate(t.gates):
        gates.append(gate)
        shift = dict(qubits=tuple(q + n for q in gate.qubits))
        if gate.src == "param":
            shift["index"] = gate.index + t.n_params
        elif gate.src == "data":
            shift["index"] = gate.index + t.n_data
        gates.append(replace(gate, **shift))
        if g in t.entangling_layers:
            ent.append(len(gates) - 1)

    def hadamard(q):
        return [sim.Gate("rz", (q,), "fixed", value=np.pi), sim.Gate("ry", (q,), "fixed", value=np.pi / 2)]

    for q in range(n):
        c, b = q, q + n
        gates += hadamard(b)
        gates += [sim.Gate("zz", (c, b), "fixed", value=np.pi / 2),
                  sim.Gate("rz", (c,), "fixed", value=-np.pi / 2), sim.Gate("rz", (b,), "fixed", value=-np.pi / 2)]
        gates += hadamard(b)
        gates += hadamard(c)
    ent.append(len(gates) - 1)
    return sim.Template(2 * n, tuple(gates), 2 * t.n_params, 2 * t.n_data, (), tuple(ent))


def swap_signs(n: int) -> np.ndarray:
    """Eigenvalue of the SWAP operator for every outcome of the ``2 n`` measured qubits.

    Qubit 0 is the most significant bit of the outcome index.  The outcome ``(a_q, b_q) = (1, 1)``
    of a pair is the singlet, which has the eigenvalue -1.
    """
    idx = np.arange(4 ** n)
    bits = (idx[:, None] >> (2 * n - 1 - np.arange(2 * n))[None, :]) & 1
    return 1 - 2 * ((bits[:, :n] * bits[:, n:]).sum(axis=1) % 2)


def overlap_plan(theta: np.ndarray, params) -> tuple:
    """Parameter vectors of the two copies for all overlaps needed for the metric block ``params``.

    Returns ``(TA, TB, pairs)``.  Row ``r`` of ``TA`` and ``TB`` holds the parameters of copy A and
    copy B of circuit ``r``.  The first circuit has both copies at ``theta`` (purity).  The states
    are ``theta + s pi/2 e_i`` for ``i`` in ``params`` and ``s = +-1``, numbered ``2 a`` for
    ``(params[a], +)`` and ``2 a + 1`` for ``(params[a], -)``, and ``pairs[r - 1] = (u, v)`` with
    ``u <= v`` lists the pair of states of circuit ``r``.
    """
    theta = np.asarray(theta, float)
    states = []
    for i in params:
        for s in (+1, -1):
            v = theta.copy()
            v[i] += s * np.pi / 2
            states.append(v)
    pairs = [(u, v) for u in range(len(states)) for v in range(u, len(states))]
    TA = np.array([theta] + [states[u] for u, _ in pairs])
    TB = np.array([theta] + [states[v] for _, v in pairs])
    return TA, TB, pairs


def overlaps_from_counts(counts: np.ndarray, shots: int, n: int) -> np.ndarray:
    """Estimates of ``Tr[rho_a rho_b]`` from the counts (C, 4**n) of the two-copy circuits."""
    return counts @ swap_signs(n) / shots


def bures_block(overlaps: np.ndarray, pairs, m: int, N: int) -> dict:
    """Purity, noise strength, Hilbert-Schmidt block and Bures block from the overlaps of :func:`overlap_plan`.

    ``overlaps[0]`` is the purity.  The Bures block uses the depolarizing relation with the noise
    strength inferred from the purity.
    """
    O = np.zeros((2 * m, 2 * m))
    for (u, v), o in zip(pairs, overlaps[1:]):
        O[u, v] = O[v, u] = o
    sign = np.array([+1, -1] * m)
    W = sign[:, None] * sign[None, :] * O
    hs = 0.25 * W.reshape(m, 2, m, 2).sum(axis=(1, 3))
    purity = float(overlaps[0])
    p = float(geo.depolarizing_from_purity(min(max(purity, 1.0 / N), 1.0), N))
    return dict(purity=purity, p_hat=p, hilbert_schmidt=hs, bures=hs / (2.0 * (1.0 - p + 2.0 * p / N)))
