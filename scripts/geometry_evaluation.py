#!/usr/bin/env python
"""Direct evaluation of the mixed-state geometry of the client states.

For a range of device-noise scales the script computes, from the exact density matrices
of the gate-level noise model, the Bures metric, the mean Uhlmann curvature, and the
incompatibility ratio of the states prepared by a client, and compares them with the
closed forms of global depolarizing noise evaluated at the noise strength that the
client would infer from the purity (echo) of its states.

Output: ``results/geometry/geometry.json``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qfedqgt import data, geometry as geo, sim          # noqa: E402
from qfedqgt.scenarios import SCENARIOS                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def frob_fit(A: np.ndarray, B: np.ndarray):
    """Best scalar c with A ~ c B and the relative residual |A - c B| / |A|."""
    c = float(np.sum(A * B) / np.sum(B * B))
    return c, float(np.linalg.norm(A - c * B) / max(np.linalg.norm(A), 1e-300))


def main(n_inputs: int = 12, n_thetas: int = 3, seed: int = 0) -> None:
    sc = SCENARIOS["shots_and_noise"]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    N, d = t.dim, t.n_params
    ds = data.load_mnist(sc.classes, t.n_data, 200, 20, angle_scale=sc.angle_scale, seed=seed)
    rng = np.random.default_rng(seed)
    X = ds.X_train[rng.choice(len(ds.y_train), size=n_inputs, replace=False)]
    thetas = rng.uniform(-sc.init_width, sc.init_width, size=(n_thetas, d))
    base = sim.NoiseModel.reimei()
    scales = [1, 10, 20, 40, 50, 60, 80, 120, 150, 160, 200, 300, 500]
    rows = []
    e = t.echo()
    for model_name in ("gate", "global"):
        for s in scales:
            acc = {k: [] for k in ("purity", "echo", "g_fit", "g_res", "u_fit", "u_res", "g_res_hat", "u_res_hat",
                                   "R", "R0", "r_old", "r_old0")}
            for th in thetas:
                A = t.angles(X, th)
                Qp = sim.pure_qgt(t, A)                              # (C, d, d)
                if model_name == "gate":
                    noise = base.scaled(s)
                else:
                    # global depolarizing with the strength that reproduces the purity of the gate model
                    rho_g = sim.density(t, A, base.scaled(s))
                    pg = geo.depolarizing_from_purity(np.real(np.einsum("cab,cba->c", rho_g, rho_g)).mean(), N)
                    noise = sim.NoiseModel(global_p=float(pg))
                rho, drho = sim.density_derivatives(t, A, noise)
                purity = np.real(np.einsum("cab,cba->c", rho, rho))
                echo = sim.probabilities(sim.density(e, e.angles(X, th), noise), noise, t.n)[:, 0]
                p_hat = geo.depolarizing_from_purity(echo.mean(), N)
                for c in range(len(X)):
                    Q = geo.sld_qgt(rho[c], drho[c])
                    G, U = geo.bures_metric(Q), geo.uhlmann_curvature(Q)
                    G0, U0 = np.real(Qp[c]), -2 * np.imag(Qp[c])
                    gf, gr = frob_fit(G, G0)
                    uf, ur = frob_fit(U, U0)
                    acc["g_fit"].append(gf); acc["g_res"].append(gr)
                    acc["u_fit"].append(uf); acc["u_res"].append(ur)
                    acc["g_res_hat"].append(np.linalg.norm(G - geo.kappa(p_hat, N) * G0) / np.linalg.norm(G))
                    acc["u_res_hat"].append(np.linalg.norm(U - geo.kappa_omega(p_hat, N) * U0) / np.linalg.norm(U))
                    acc["R"].append(geo.incompatibility_ratio(Q)); acc["R0"].append(geo.incompatibility_ratio(Qp[c]))
                    acc["r_old"].append(np.sum(U ** 2) / np.sum(G ** 2)); acc["r_old0"].append(np.sum(U0 ** 2) / np.sum(G0 ** 2))
                acc["purity"] += list(purity); acc["echo"] += list(echo)
            row = dict(model=model_name, scale=s, p2=base.scaled(s).p2)
            for k, v in acc.items():
                row[k] = float(np.mean(v))
            row["p_purity"] = float(geo.depolarizing_from_purity(row["purity"], N))
            row["p_echo"] = float(geo.depolarizing_from_purity(row["echo"], N))
            row["kappa_pred"] = float(geo.kappa(row["p_echo"], N))
            row["kappa_omega_pred"] = float(geo.kappa_omega(row["p_echo"], N))
            row["R_pred"] = float(geo.chi(row["p_echo"], N) * row["R0"])
            row["r_old_pred"] = float(geo.chi(row["p_echo"], N) ** 2 * row["r_old0"])
            rows.append(row)
            print(f"{model_name:6s} scale {s:4d}: p_pur={row['p_purity']:.3f} p_echo={row['p_echo']:.3f} | G: fit {row['g_fit']:.3f} "
                  f"kappa {row['kappa_pred']:.3f} res {row['g_res']:.3f} (vs kappa G: {row['g_res_hat']:.3f}) | U: fit {row['u_fit']:.3f} "
                  f"kappa_w {row['kappa_omega_pred']:.3f} res {row['u_res']:.3f} | R {row['R']:.3f} pred {row['R_pred']:.3f} | "
                  f"r_old {row['r_old']:.3f} pred {row['r_old_pred']:.3f} p^2 {row['p_echo'] ** 2:.3f}", flush=True)
    out = os.path.join(ROOT, "results", "geometry")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "geometry.json"), "w") as f:
        json.dump(dict(n_qubits=sc.n_qubits, layers=sc.layers, n_params=d, n_inputs=n_inputs, n_thetas=n_thetas,
                       seed=seed, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
