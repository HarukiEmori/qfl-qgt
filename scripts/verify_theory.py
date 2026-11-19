#!/usr/bin/env python
"""Numerical verification of the statements of the paper.

Each check compares an analytical statement with an independent numerical evaluation
on random instances and prints PASS or FAIL.  Run:  python scripts/verify_theory.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qfedqgt import geometry as geo, sim                                   # noqa: E402
from qfedqgt.federated import between_client_variance, random_effects_weights   # noqa: E402

RNG = np.random.default_rng(2026)
RESULTS = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def random_instance(n=3, layers=2):
    t = sim.layered_ansatz(n, layers)
    x = RNG.uniform(-1, 1, size=(1, t.n_data))
    th = RNG.uniform(-np.pi, np.pi, size=t.n_params)
    return t, t.angles(x, th), x, th


# ---------------------------------------------------------------- Appendix A
def check_qgt_identities():
    t, A, _, _ = random_instance()
    noise = sim.NoiseModel(p1=0.01, p2=0.07, p_idle=0.02)
    rho, drho = sim.density_derivatives(t, A, noise)
    rho, drho = rho[0], drho[0]
    Q = geo.sld_qgt(rho, drho)
    lam, V = np.linalg.eigh(rho)
    D = np.einsum("am,iab,bn->imn", V.conj(), drho, V)
    d = len(drho)
    g = np.zeros((d, d)); om = np.zeros((d, d)); comm = np.zeros((d, d), complex)
    L = np.zeros_like(D)
    for m in range(len(lam)):
        for n in range(len(lam)):
            s = lam[m] + lam[n]
            if s > 1e-12:
                L[:, m, n] = 2 * D[:, m, n] / s
                a = D[:, m, n][:, None] * D[:, n, m][None, :]
                g += 0.5 * a.real / s
                om += -(lam[m] - lam[n]) * a.imag / s ** 2
    Lop = np.einsum("am,imn,bn->iab", V, L, V.conj())
    for i in range(d):
        for j in range(d):
            comm[i, j] = 0.25j * np.trace(rho @ (Lop[i] @ Lop[j] - Lop[j] @ Lop[i]))
    e1 = np.abs(Q.real - g).max(); e2 = np.abs(-2 * Q.imag - om).max(); e3 = np.abs(om - comm.real).max() + np.abs(comm.imag).max()
    report("Eqs. (bures), (uhl): eigenbasis forms and (i/4)Tr(rho[L_i,L_j])", max(e1, e2, e3) < 1e-9, f"max dev {max(e1, e2, e3):.1e}")
    report("positivity of Q^Mix and g_ii g_jj - g_ij^2 >= omega_ij^2 / 4", np.linalg.eigvalsh(Q).min() > -1e-10 and
           all(g[i, i] * g[j, j] - g[i, j] ** 2 >= om[i, j] ** 2 / 4 - 1e-12 for i in range(d) for j in range(d)))


def check_pure_limit_and_depolarizing():
    t, A, _, _ = random_instance()
    N = t.dim
    Qp = sim.pure_qgt(t, A)[0]
    dev = 0.0
    for p in (1e-9, 0.2, 0.5, 0.8):
        rho, drho = sim.density_derivatives(t, A, sim.NoiseModel(global_p=p))
        Q = geo.sld_qgt(rho[0], drho[0])
        dev = max(dev, np.abs(Q.real - geo.kappa(p, N) * Qp.real).max(), np.abs(Q.imag - geo.kappa_omega(p, N) * Qp.imag).max())
        dev = max(dev, abs(geo.incompatibility_ratio(Q) - geo.chi(p, N) * geo.incompatibility_ratio(Qp)))
        ab = 1 - p + 2 * p / N
        Q3 = np.array([[np.trace(rho[0] @ drho[0][i] @ drho[0][j]) for j in range(len(drho[0]))] for i in range(len(drho[0]))]) / ab ** 2
        dev = max(dev, np.abs(Q3 - Q).max())
    report("Appendix A.4 and Proposition (depolarizing): pure limit, kappa, kappa_omega, ratio", dev < 1e-6, f"max dev {dev:.1e}")
    report("Berry curvature of the ansatz is nonzero (pure-state mean Uhlmann curvature)", np.abs(Qp.imag).max() > 1e-3,
           f"max |Im Q| = {np.abs(Qp.imag).max():.3f}")


def check_invariance():
    t, A, x, th = random_instance()
    N = t.dim
    obs = sim.z_observables(t.n, pairs=False)
    u = RNG.normal(size=obs.shape[1])
    p, eps = 0.35, 0.05
    S = sim.shifted_angles(t, A)[0].reshape(-1, A.shape[1])
    phi = (sim.probabilities(sim.density(t, S, sim.NoiseModel(global_p=p))) @ obs).reshape(t.n_params, 2, -1)
    grad_k = 0.5 * (phi[:, 0] - phi[:, 1]) @ u
    phi0 = ((np.abs(sim.statevector(t, S)) ** 2) @ obs).reshape(t.n_params, 2, -1)
    JTu = 0.5 * (phi0[:, 0] - phi0[:, 1]) @ u
    G = np.real(sim.pure_qgt(t, A)[0])
    Gk = geo.kappa(p, N) * G
    Pk = np.linalg.inv(Gk + eps * np.trace(Gk) / np.trace(G) * np.eye(t.n_params))
    lhs = Pk @ grad_k
    rhs = (1 + 2 * p / (N * (1 - p))) * np.linalg.solve(G + eps * np.eye(t.n_params), JTu)
    report("Proposition (device-invariant update)", np.abs(lhs - rhs).max() < 1e-9 and np.abs(grad_k - (1 - p) * JTu).max() < 1e-9,
           f"max dev {np.abs(lhs - rhs).max():.1e}")


def check_echo():
    t, A, x, th = random_instance(4, 2)
    noise = sim.NoiseModel(p1=0.004, p2=0.08)
    rho = sim.density(t, A, noise)
    e = t.echo()
    surv = sim.probabilities(sim.density(e, e.angles(x, th), noise))[0, 0]
    report("Appendix B: echo survival = purity for gate-level depolarizing faults",
           abs(surv - np.real(np.trace(rho[0] @ rho[0]))) < 1e-10)


# ---------------------------------------------------------------- Appendix C
def check_holevo_sandwich():
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex)
    basis = [I2, X, Y, Z]
    ok = True; worst = 0.0
    for _ in range(5):
        r = RNG.uniform(0.3, 1.0); v0 = np.array([RNG.uniform(0.4, 2.6), RNG.uniform(0, 6.28)])
        def rho_of(v):
            nv = r * np.array([np.sin(v[0]) * np.cos(v[1]), np.sin(v[0]) * np.sin(v[1]), np.cos(v[0])])
            return 0.5 * (I2 + nv[0] * X + nv[1] * Y + nv[2] * Z)
        eps = 1e-6
        rho = rho_of(v0)
        drho = np.array([(rho_of(v0 + eps * np.eye(2)[i]) - rho_of(v0 - eps * np.eye(2)[i])) / (2 * eps) for i in range(2)])
        Q = geo.sld_qgt(rho, drho)
        F, D = 4 * Q.real, 4 * Q.imag
        Wm = RNG.normal(size=(2, 2)); W = Wm @ Wm.T + 0.3 * np.eye(2); sW = sqrtm(W).real
        CS = np.trace(W @ np.linalg.inv(F))
        C = geo.incompatibility_matrix(Q); R = geo.incompatibility_ratio(Q)
        mid = CS + np.trace(W @ C)
        rows, rhs = [], []
        for i in range(2):
            row = np.zeros(8); row[4 * i:4 * i + 4] = [np.trace(rho @ b).real for b in basis]; rows.append(row); rhs.append(0.0)
            for j in range(2):
                row = np.zeros(8); row[4 * i:4 * i + 4] = [np.trace(drho[j] @ b).real for b in basis]; rows.append(row); rhs.append(float(i == j))
        Am, bm = np.array(rows), np.array(rhs)
        c0 = np.linalg.lstsq(Am, bm, rcond=None)[0]
        ns = np.linalg.svd(Am)[2][np.linalg.matrix_rank(Am):].T
        def obj(z):
            c = (c0 + ns @ z).reshape(2, 4)
            Xs = [sum(c[i, k] * basis[k] for k in range(4)) for i in range(2)]
            Zm = np.array([[np.trace(rho @ Xs[i] @ Xs[j]) for j in range(2)] for i in range(2)])
            return np.trace(W @ Zm.real) + np.abs(np.linalg.eigvals(sW @ Zm.imag @ sW)).sum()
        CH = min(minimize(obj, RNG.normal(size=ns.shape[1]), method="Nelder-Mead",
                          options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000)).fun for _ in range(3))
        ok &= (CS - 1e-6 <= CH <= mid + 1e-5 <= (1 + R) * CS + 1e-5) and 0 <= R <= 1 + 1e-9
        worst = max(worst, CH / CS)
    report("Proposition (attainable precision): C^S <= C^H <= Tr[W(F^-1 + C)] <= (1+R) C^S", ok, f"largest C^H/C^S = {worst:.3f}")


# ---------------------------------------------------------------- Appendix D
def check_ceiling():
    ok = True
    for trial in range(4):
        t, A, x, th = random_instance(3, 1 + trial % 2)
        d, N = t.n_params, t.dim
        noise = sim.NoiseModel(global_p=0.4) if trial < 2 else sim.NoiseModel(p1=0.01, p2=0.08)
        obs = sim.z_observables(t.n, pairs=False); proj = obs @ RNG.normal(size=obs.shape[1])
        f = lambda theta: float(sim.probabilities(sim.density(t, t.angles(x, theta), noise))[0] @ proj)
        S = sim.shifted_angles(t, A)[0]; M = 50
        P = sim.probabilities(sim.density(t, S.reshape(2 * d, -1), noise)).reshape(d, 2, N)
        V = np.diag(((P @ proj ** 2) - (P @ proj) ** 2).sum(axis=1) / (4 * M))
        Ftot = np.zeros((d, d))
        for i in range(d):
            for s in range(2):
                rho, drho = sim.density_derivatives(t, S[i, s][None], noise)
                Ftot += M * 4 * np.real(geo.sld_qgt(rho[0], drho[0]))
        e = np.eye(d) * np.pi / 2
        H = np.array([[(f(th + e[i] + e[j]) - f(th + e[i] - e[j]) - f(th - e[i] + e[j]) + f(th - e[i] - e[j])) / 4
                       for j in range(d)] for i in range(d)])
        ok &= np.linalg.eigvalsh(V - H @ np.linalg.inv(Ftot) @ H).min() > -1e-12
    report("Proposition (precision ceiling): Cov >= D F_tot^-1 D^T for the parameter-shift estimator", ok)


# ---------------------------------------------------------------- Appendix E
def check_aggregation():
    K, d = 5, 4
    Sig = []
    for _ in range(K):
        M = RNG.normal(size=(d, d)); Sig.append(M @ M.T + 0.2 * np.eye(d))
    Lam = [np.linalg.inv(S) for S in Sig]
    S0 = np.linalg.inv(sum(Lam))
    ok = True
    for _ in range(20):
        Aw = [RNG.normal(size=(d, d)) for _ in range(K - 1)]
        Aw.append(np.eye(d) - sum(Aw))
        ok &= np.linalg.eigvalsh(sum(a @ s @ a.T for a, s in zip(Aw, Sig)) - S0).min() > -1e-9
    opt = [S0 @ L for L in Lam]
    ok &= np.abs(sum(a @ s @ a.T for a, s in zip(opt, Sig)) - S0).max() < 1e-9
    report("Proposition (matrix weights): sum A S A^T >= (sum Lambda)^-1 with equality at A^opt", ok)

    q = RNG.dirichlet(np.ones(K) * 3); s2 = RNG.uniform(0.01, 5, size=K); z2 = RNG.uniform(0.01, 2)
    B = lambda w: z2 * ((w - q) ** 2 / q).sum() + (w ** 2 * s2).sum()
    w = random_effects_weights(q, s2, z2)
    num = minimize(B, np.ones(K) / K, constraints=[dict(type="eq", fun=lambda w: w.sum() - 1)], method="SLSQP",
                   options=dict(ftol=1e-14, maxiter=500))
    closed = 1.0 / (q / (z2 + q * s2)).sum() - z2
    report("Proposition (scalar weights): minimizer and minimum of the bound",
           np.abs(num.x - w).max() < 1e-6 and abs(B(w) - closed) < 1e-10 and B(w) <= B(q) + 1e-12,
           f"B(w*) = {B(w):.4f} <= B(q) = {B(q):.4f}")

    tau2, sig = 0.7, RNG.uniform(0.2, 3.0, size=8)
    est = []
    for _ in range(4000):
        mu = RNG.normal(size=6)
        upd = mu + RNG.normal(size=(8, 6)) * np.sqrt(tau2 / 6) + RNG.normal(size=(8, 6)) * np.sqrt(sig / 6)[:, None]
        a = 1 / sig; Aa = a.sum(); mean = (a[:, None] * upd).sum(0) / Aa
        est.append(((a * ((upd - mean) ** 2).sum(1)).sum() - 7) / (Aa - (a ** 2).sum() / Aa))
    report("Appendix E.3: moment estimator of the between-client variance is unbiased before truncation",
           abs(np.mean(est) - tau2) < 0.05, f"mean {np.mean(est):.3f} vs {tau2}")
    assert between_client_variance(np.zeros((3, 2)), np.ones(3)) == 0.0


def check_efficiency():
    """Appendix E.4: efficiency of the data-share (and shot-weighted) average of plain gradients."""
    ok, ok_s, worst = True, True, 1.0
    for _ in range(200):
        K, d = int(RNG.integers(2, 12)), 4
        q = RNG.dirichlet(np.ones(K)); M = RNG.integers(1, 60, size=K).astype(float); a = RNG.uniform(0.02, 1.0, size=K)
        A0 = RNG.normal(size=(d, d)); V0 = A0 @ A0.T + np.eye(d)
        cov_k = [V0 / (M[k] * a[k] ** 2) for k in range(K)]            # covariance of h + xi_k / a_k

        def cov_of(w):                                                   # covariance of the normalized weighted sum
            w = w / w.sum()
            return sum(w[k] ** 2 * cov_k[k] for k in range(K))
        opt = np.linalg.inv(sum(np.linalg.inv(c) for c in cov_k))        # Proposition 5
        eff = (q * a).sum() ** 2 / ((q ** 2 / M).sum() * (M * a ** 2).sum())
        eff_s = (q * M * a).sum() ** 2 / ((q ** 2 * M).sum() * (M * a ** 2).sum())
        ok &= np.allclose(opt, eff * cov_of(q * a)) and eff <= 1 + 1e-12
        ok_s &= np.allclose(opt, eff_s * cov_of(q * M * a)) and eff_s <= 1 + 1e-12
        worst = min(worst, eff)
    report("Appendix E.4: efficiency of the data-share average of plain gradients, Eq. (efficiency)", ok,
           f"smallest efficiency in the sample {worst:.3f}")
    report("Appendix E.4: efficiency of the shot-weighted average of plain gradients", ok_s)
    q = np.full(5, 0.2)
    report("Appendix E.4: efficiency equals one if and only if M_k a_k / q_k is constant",
           abs((q * np.array([1, .5, .25, .2, .1])).sum() ** 2
               / ((q ** 2 / np.array([1, 2, 4, 5, 10.])).sum() * (np.array([1, 2, 4, 5, 10.]) * np.array([1, .5, .25, .2, .1]) ** 2).sum()) - 1) < 1e-12)


if __name__ == "__main__":
    check_qgt_identities()
    check_pure_limit_and_depolarizing()
    check_invariance()
    check_echo()
    check_holevo_sandwich()
    check_ceiling()
    check_aggregation()
    check_efficiency()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
