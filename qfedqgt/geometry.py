"""Mixed-state quantum geometric tensor: Bures metric, mean Uhlmann curvature, and
the quantities derived from them.

Conventions (those of the paper)
--------------------------------
* Symmetric logarithmic derivative (SLD): ``d_i rho = (L_i rho + rho L_i) / 2``.
* Mixed-state quantum geometric tensor: ``Q_ij = Tr[rho L_i L_j] / 4 = g_ij - (i/2) omega_ij``.
  ``g`` is the Bures metric, ``F = 4 g`` the SLD quantum Fisher information matrix and
  ``omega = -2 Im Q = (i/4) Tr(rho [L_i, L_j])`` the mean Uhlmann curvature.
  For a pure state ``Q`` is the pure-state tensor, ``g`` the Fubini-Study metric and
  ``omega`` the Berry curvature.
* Incompatibility ratio: ``R = || F^{-1/2} D F^{-1/2} ||_inf`` with ``D = Im Tr[rho L_i L_j] = -2 omega``,
  which lies in [0, 1].
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "sld_qgt", "bures_metric", "uhlmann_curvature", "kappa", "kappa_omega", "chi",
    "depolarizing_from_purity", "purity_of_depolarized", "incompatibility_ratio",
    "incompatibility_matrix", "block_diagonal", "regularized_inverse",
]


def sld_qgt(rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Mixed-state geometric tensor ``Q_ij = Tr[rho L_i L_j]/4`` from ``rho`` (D, D) and ``drho`` (d, D, D).

    The SLD is defined on the pairs of eigenvectors with ``lambda_m + lambda_n > tol``;
    its undetermined block on the kernel of ``rho`` does not enter ``Tr[rho L_i L_j]``.
    """
    lam, V = np.linalg.eigh(rho)
    lam = np.clip(lam, 0.0, None)
    Dm = np.einsum("am,iab,bn->imn", V.conj(), drho, V)          # <m| d_i rho |n>
    s = lam[:, None] + lam[None, :]
    inv = np.where(s > tol, 1.0 / np.where(s > tol, s, 1.0), 0.0)
    L = 2.0 * Dm * inv[None, :, :]
    return 0.25 * np.einsum("m,imn,jnm->ij", lam, L, L)


def bures_metric(Q: np.ndarray) -> np.ndarray:
    return np.real(Q)


def uhlmann_curvature(Q: np.ndarray) -> np.ndarray:
    return -2.0 * np.imag(Q)


# ------------------------------------------------------------------ global depolarizing closed forms
def kappa(p, N):
    """Attenuation of the Bures metric: g -> kappa(p) g."""
    p = np.asarray(p, float)
    return (1 - p) ** 2 / (1 - p + 2 * p / N)


def kappa_omega(p, N):
    """Attenuation of the mean Uhlmann curvature: omega -> kappa_omega(p) omega."""
    p = np.asarray(p, float)
    return (1 - p) ** 3 / (1 - p + 2 * p / N) ** 2


def chi(p, N):
    """Attenuation of the incompatibility ratio: R -> chi(p) R, chi = kappa_omega / kappa."""
    p = np.asarray(p, float)
    return (1 - p) / (1 - p + 2 * p / N)


def purity_of_depolarized(p, N):
    p = np.asarray(p, float)
    return (1 - p) ** 2 + (1 - (1 - p) ** 2) / N


def depolarizing_from_purity(gamma, N):
    """Depolarizing strength p of (1-p)|psi><psi| + p 1/N having purity ``gamma``."""
    gamma = np.clip(np.asarray(gamma, float), 1.0 / N, 1.0)
    return 1.0 - np.sqrt((N * gamma - 1.0) / (N - 1.0))


# ------------------------------------------------------------------ incompatibility
def _inv_sqrt_psd(F: np.ndarray, rtol: float = 1e-10):
    w, U = np.linalg.eigh(0.5 * (F + F.T))
    keep = w > rtol * max(w.max(), 1e-300)
    return (U[:, keep] / np.sqrt(w[keep])) @ U[:, keep].T


def incompatibility_ratio(Q: np.ndarray) -> float:
    """``R = ||F^{-1/2} D F^{-1/2}||_inf`` on the support of ``F`` (0 <= R <= 1)."""
    F, D = 4.0 * np.real(Q), 4.0 * np.imag(Q)
    Fih = _inv_sqrt_psd(F)
    A = Fih @ D @ Fih                      # real antisymmetric
    return float(np.abs(np.linalg.eigvalsh(1j * A)).max())


def incompatibility_matrix(Q: np.ndarray) -> np.ndarray:
    """``C = F^{-1/2} |F^{-1/2} D F^{-1/2}| F^{-1/2}`` (positive semidefinite, C <= R F^{-1})."""
    F, D = 4.0 * np.real(Q), 4.0 * np.imag(Q)
    Fih = _inv_sqrt_psd(F)
    w, U = np.linalg.eigh(1j * (Fih @ D @ Fih))
    absA = np.real((U * np.abs(w)) @ U.conj().T)
    return Fih @ absA @ Fih


# ------------------------------------------------------------------ helpers for the preconditioner
def block_diagonal(G: np.ndarray, blocks) -> np.ndarray:
    """Keep the intra-block elements of ``G`` (layerwise approximation of the metric)."""
    out = np.zeros_like(G)
    for b in blocks:
        ix = np.ix_(list(b), list(b))
        out[ix] = G[ix]
    return out


def regularized_inverse(G: np.ndarray, eps: float) -> np.ndarray:
    return np.linalg.inv(G + eps * np.eye(G.shape[0]))
