"""Variational quantum classifier with a Pauli-Z readout and its shot-based estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sim

__all__ = ["VQC", "softmax", "GradientEstimate"]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class GradientEstimate:
    grad: np.ndarray          # (d,) minibatch gradient of the loss
    var: np.ndarray           # (d,) squared standard error of each gradient component
    loss: float               # minibatch loss evaluated with the measured features
    n_circuits: int           # number of circuits executed
    n_shots: int              # total number of shots


@dataclass
class VQC:
    """Classifier whose class scores are ``beta <Z_c>`` on the first ``n_classes`` qubits."""
    template: sim.Template
    n_classes: int
    beta: float = 4.0

    def __post_init__(self):
        if self.n_classes > self.template.n:
            raise ValueError("one readout qubit per class is required")
        self.obs = sim.z_observables(self.template.n, pairs=False)[:, : self.n_classes]   # (D, C)

    # ------------------------------------------------------------------ loss
    def loss_and_dphi(self, phi: np.ndarray, y: np.ndarray):
        """Cross-entropy of every sample and its gradient with respect to the features."""
        p = softmax(self.beta * phi)
        loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
        dphi = self.beta * p
        dphi[np.arange(len(y)), y] -= self.beta
        return loss, dphi

    def predict(self, phi: np.ndarray) -> np.ndarray:
        return np.argmax(phi, axis=-1)

    # ------------------------------------------------------------------ exact quantities
    def features_exact(self, X, theta, noise: sim.NoiseModel | None = None) -> np.ndarray:
        A = self.template.angles(X, theta)
        if noise is None or noise.is_noiseless:
            psi = sim.statevector(self.template, A)
            p = np.abs(psi) ** 2
        else:
            p = sim.probabilities(sim.density(self.template, A, noise), noise, self.template.n)
        return p @ self.obs

    def loss_exact(self, X, y, theta, noise=None) -> float:
        return float(self.loss_and_dphi(self.features_exact(X, theta, noise), y)[0].mean())

    def gradient_exact(self, X, y, theta, noise=None) -> np.ndarray:
        """Exact gradient of the mean loss (parameter-shift rule with exact expectation values)."""
        t = self.template
        A = t.angles(X, theta)
        B, d = A.shape[0], t.n_params
        _, dphi = self.loss_and_dphi(self.features_exact(X, theta, noise), y)
        S = sim.shifted_angles(t, A).reshape(B * d * 2, -1)
        if noise is None or noise.is_noiseless:
            p = np.abs(sim.statevector(t, S)) ** 2
        else:
            p = sim.probabilities(sim.density(t, S, noise), noise, t.n)
        phi = (p @ self.obs).reshape(B, d, 2, -1)
        J = 0.5 * (phi[:, :, 0] - phi[:, :, 1])
        return np.einsum("bdf,bf->d", J, dphi) / B

    def accuracy_exact(self, X, y, theta, noise=None) -> float:
        return float((self.predict(self.features_exact(X, theta, noise)) == y).mean())

    # ------------------------------------------------------------------ shot-based estimators
    def gradient_angles(self, X, theta) -> np.ndarray:
        """Angles of the unshifted circuits followed by the parameter-shifted circuits of every sample."""
        t = self.template
        A = t.angles(X, theta)
        S = sim.shifted_angles(t, A).reshape(A.shape[0] * t.n_params * 2, -1)
        return np.concatenate([A, S], axis=0)

    def per_sample_gradients(self, counts: np.ndarray, y: np.ndarray, shots: int):
        """Parameter-shift gradient of every sample from the counts of :meth:`gradient_angles`.

        ``counts`` has shape (B + 2 d B, D).  Returns the gradients (B, d) and the losses (B,).
        The loss and its gradient with respect to the features are evaluated at the features
        measured with the unshifted circuit.
        """
        t = self.template
        B, d, D = len(y), t.n_params, t.dim
        c0 = counts[:B]
        cs = counts[B:].reshape(B, d, 2, D)
        phi0 = c0 @ self.obs / shots
        loss, u = self.loss_and_dphi(phi0, y)                  # u: (B, C)
        proj = u @ self.obs.T                                   # (B, D) value of u.Z on each outcome
        m1 = np.einsum("bdsz,bz->bds", cs, proj) / shots
        return 0.5 * (m1[:, :, 0] - m1[:, :, 1]), loss

    def estimate_gradient(self, backend, X, y, theta, shots: int) -> GradientEstimate:
        """Minibatch gradient from measured counts and the squared standard error of each component."""
        counts = backend.counts(self.template, self.gradient_angles(X, theta), shots)[0]
        gb, loss = self.per_sample_gradients(counts, y, shots)
        B = len(y)
        var = gb.var(axis=0, ddof=1) / B if B > 1 else np.full(gb.shape[1], np.nan)
        n_circ = B * (1 + 2 * self.template.n_params)
        return GradientEstimate(gb.mean(axis=0), var, float(loss.mean()), n_circ, n_circ * shots)

    def estimate_purity(self, backend, X, theta, shots: int):
        """Echo estimate of the purity: survival frequency of ``U^dagger U |0>`` averaged over inputs."""
        e = self.template.echo()
        counts = backend.counts(e, e.angles(X, theta), shots)[0]
        return float(counts[:, 0].sum() / (shots * counts.shape[0])), counts.shape[0]

    def evaluate(self, backend, X, y, theta, shots: int) -> float:
        counts = backend.counts(self.template, self.template.angles(X, theta), shots)[0]
        return float((self.predict(counts @ self.obs / shots) == y).mean())
