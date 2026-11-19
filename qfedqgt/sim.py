"""Circuit templates and exact simulators.

A circuit is described by a :class:`Template` (an ordered list of gates) and an
array of rotation angles, one row per circuit.  The same description is executed
by the local simulators in this module and, through :mod:`qfedqgt.nexus`, by the
Quantinuum emulator, so that the federated protocol is identical on both.

Conventions
-----------
* Qubit 0 is the most significant bit of a computational-basis index.
* ``rx``, ``ry``, ``rz`` are :math:`e^{-i a P/2}`; ``zz`` is :math:`e^{-i a Z\\otimes Z/2}`
  (``a = pi/2`` is the maximally entangling ZZMax gate of the trapped-ion gate set).
* A depolarizing fault with Pauli-error probability ``p`` on ``m`` qubits is
  :math:`\\rho \\mapsto (1-p)\\rho + \\frac{p}{4^m-1}\\sum_{P\\neq I} P\\rho P`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

__all__ = [
    "Gate", "Template", "NoiseModel", "layered_ansatz", "statevector", "density",
    "probabilities", "sample_counts", "shifted_angles", "z_observables", "pure_qgt",
    "density_derivatives",
]


# --------------------------------------------------------------------------- template
@dataclass(frozen=True)
class Gate:
    """One gate of a template.

    ``src`` tells where the angle comes from: ``"fixed"`` (``value``), ``"data"``
    (``scale * x[index]``) or ``"param"`` (``scale * theta[index]``).
    """
    kind: str
    qubits: tuple
    src: str
    index: int = -1
    value: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class Template:
    n: int
    gates: tuple
    n_params: int
    n_data: int
    blocks: tuple = ()          # parameter indices of each layer (block-diagonal metric)
    entangling_layers: tuple = ()  # gate positions after which idle dephasing is applied
    global_points: tuple = ()   # gate positions after which the global depolarizing channel acts (default: the end)

    @property
    def dim(self) -> int:
        return 2 ** self.n

    def angles(self, X, theta) -> np.ndarray:
        """Angle array of shape (C, G) for inputs ``X`` (C, n_data) and parameters ``theta``."""
        X = np.atleast_2d(np.asarray(X, float))
        theta = np.asarray(theta, float)
        C = X.shape[0]
        th = np.broadcast_to(theta, (C, self.n_params)) if theta.ndim == 1 else theta
        A = np.empty((C, len(self.gates)))
        for g, gate in enumerate(self.gates):
            if gate.src == "fixed":
                A[:, g] = gate.value
            elif gate.src == "data":
                A[:, g] = gate.scale * X[:, gate.index]
            else:
                A[:, g] = gate.scale * th[:, gate.index]
        return A

    def param_gates(self) -> np.ndarray:
        """Position of the (unique) gate carrying each trainable parameter."""
        pos = np.full(self.n_params, -1)
        for g, gate in enumerate(self.gates):
            if gate.src == "param":
                if pos[gate.index] != -1:
                    raise ValueError("a parameter enters more than one gate")
                pos[gate.index] = g
        if (pos < 0).any():
            raise ValueError("unused parameter")
        return pos

    def inverse(self) -> "Template":
        """Template of the inverse circuit (gates reversed, angles negated)."""
        inv = tuple(replace(g, value=-g.value, scale=-g.scale) for g in reversed(self.gates))
        return Template(self.n, inv, self.n_params, self.n_data, self.blocks, ())

    def echo(self) -> "Template":
        """Template of the echo circuit ``U`` followed by ``U^dagger``.

        In the global depolarizing model the channel acts once after each half.
        """
        m = len(self.gates)
        return Template(self.n, self.gates + self.inverse().gates, self.n_params, self.n_data,
                        self.blocks, (), (m - 1, 2 * m - 1))

    def count(self, kind: str) -> int:
        return sum(1 for g in self.gates if g.kind == kind)


def layered_ansatz(n: int, layers: int, entangler: str = "linear") -> Template:
    """Data re-uploading ansatz.

    Layer ``l`` applies ``ry(x)`` on every qubit (``n`` new input components), a chain
    of ZZMax gates, and the trainable block ``rz(theta) ry(theta)`` on every qubit.
    The trainable block comes last so that the final rotations fix the measured basis.
    """
    gates, blocks, ent = [], [], []
    k = 0
    for l in range(layers):
        for q in range(n):
            gates.append(Gate("ry", (q,), "data", l * n + q))
        pairs = [(q, q + 1) for q in range(n - 1)]
        if entangler == "ring" and n > 2:
            pairs.append((n - 1, 0))
        for (q, r) in pairs:
            gates.append(Gate("zz", (q, r), "fixed", value=np.pi / 2))
        ent.append(len(gates) - 1)
        block = []
        for q in range(n):
            gates.append(Gate("rz", (q,), "param", k)); block.append(k); k += 1
            gates.append(Gate("ry", (q,), "param", k)); block.append(k); k += 1
        blocks.append(tuple(block))
    return Template(n, tuple(gates), k, layers * n, tuple(blocks), tuple(ent))


# --------------------------------------------------------------------------- noise
@dataclass(frozen=True)
class NoiseModel:
    """Gate-level noise used by the local density-matrix simulator.

    ``p1`` and ``p2`` are Pauli-error probabilities of the one- and two-qubit
    depolarizing faults.  The two-qubit value refers to the ZZMax angle and is scaled
    with the gate angle as ``(a (|angle|/pi)^power + b) p2`` (the parametrization of
    the trapped-ion emulator, normalized to one at ``pi/2``).  ``p_idle`` is a
    dephasing probability applied to every qubit after each entangling layer.
    ``global_p`` adds a global depolarizing channel at the end of the circuit (the
    exactly solvable model of the paper).
    """
    p1: float = 0.0
    p2: float = 0.0
    przz_a: float = 1.65
    przz_b: float = 0.175
    przz_power: float = 1.0
    p_meas0: float = 0.0      # probability of reading 1 when the qubit is in 0
    p_meas1: float = 0.0      # probability of reading 0 when the qubit is in 1
    p_init: float = 0.0
    p_idle: float = 0.0
    global_p: float = 0.0

    def scaled(self, gate: float, spam: float = 1.0) -> "NoiseModel":
        """Gate and idling faults multiplied by ``gate``; preparation and readout faults by ``spam``.

        This mirrors the scale factors of the emulator (``p1_scale``, ``p2_scale``,
        ``memory_scale`` on one side, ``meas_scale`` and ``init_scale`` on the other).
        """
        c = lambda v, s, top: float(min(max(v * s, 0.0), top))
        return replace(self, p1=c(self.p1, gate, 0.75), p2=c(self.p2, gate, 0.9375),
                       p_idle=c(self.p_idle, gate, 0.5), p_meas0=c(self.p_meas0, spam, 0.5),
                       p_meas1=c(self.p_meas1, spam, 0.5), p_init=c(self.p_init, spam, 0.5))

    def p2_at(self, angle) -> np.ndarray:
        a = np.abs(np.asarray(angle, float)) / np.pi
        return np.clip((self.przz_a * a ** self.przz_power + self.przz_b) * self.p2, 0.0, 0.9375)

    @property
    def is_noiseless(self) -> bool:
        return not any((self.p1, self.p2, self.p_meas0, self.p_meas1, self.p_init, self.p_idle, self.global_p))

    @staticmethod
    def reimei() -> "NoiseModel":
        """Published error parameters of the reimei emulator (Nexus backend page, read on 2026-11-18)."""
        return NoiseModel(p1=2.8e-5, p2=1.2e-3, p_meas0=7.0e-4, p_meas1=4.0e-3, p_init=4.0e-5, p_idle=2.8e-4)


# --------------------------------------------------------------------------- operators
@lru_cache(maxsize=None)
def _tables(n: int):
    D = 2 ** n
    idx = np.arange(D)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1      # (D, n)
    z = 1 - 2 * bits                                                   # eigenvalues of Z_q
    flip = np.stack([idx ^ (1 << (n - 1 - q)) for q in range(n)], axis=1)  # (D, n)
    return bits, z, flip


def _left(arr, kind, q, n, axis):
    """Multiply by the Pauli ``kind`` on qubit ``q`` along ``axis`` (ket index)."""
    _, z, flip = _tables(n)
    out = np.take(arr, flip[:, q], axis=axis)
    if kind == "ry":       # (Y v)[b] = -i z_q[b] v[flip(b)]
        shape = [1] * arr.ndim
        shape[axis] = -1
        out = out * (-1j * z[:, q]).reshape(shape)
    return out


def _right(arr, kind, q, n, axis):
    """Multiply by the Pauli on the right along ``axis`` (bra index): (A P)[.., a] = A[.., flip(a)] P[flip(a), a]."""
    _, z, flip = _tables(n)
    out = np.take(arr, flip[:, q], axis=axis)
    if kind == "ry":       # Y[flip(a), a] = +i z_q[a]
        shape = [1] * arr.ndim
        shape[axis] = -1
        out = out * (1j * z[:, q]).reshape(shape)
    return out


def _apply_psi(psi, kind, qubits, angle, n):
    """Apply a gate to state vectors ``psi`` of shape (..., D); ``angle`` broadcasts over the batch."""
    _, z, _ = _tables(n)
    a = np.asarray(angle, float)[..., None]
    if kind == "rz":
        return psi * np.exp(-0.5j * a * z[:, qubits[0]])
    if kind == "zz":
        return psi * np.exp(-0.5j * a * z[:, qubits[0]] * z[:, qubits[1]])
    if kind in ("rx", "ry"):
        return np.cos(a / 2) * psi - 1j * np.sin(a / 2) * _left(psi, kind, qubits[0], n, psi.ndim - 1)
    raise ValueError(kind)


def _apply_rho(rho, kind, qubits, angle, n):
    """Apply a gate to density matrices ``rho`` of shape (..., D, D)."""
    _, z, _ = _tables(n)
    a = np.asarray(angle, float)[..., None]
    if kind in ("rz", "zz"):
        d = z[:, qubits[0]] if kind == "rz" else z[:, qubits[0]] * z[:, qubits[1]]
        ph = np.exp(-0.5j * a * d)
        return rho * ph[..., :, None] * np.conj(ph)[..., None, :]
    if kind in ("rx", "ry"):
        c, s = np.cos(a / 2)[..., None], np.sin(a / 2)[..., None]
        q = qubits[0]
        left = c * rho - 1j * s * _left(rho, kind, q, n, rho.ndim - 2)
        return c * left + 1j * s * _right(left, kind, q, n, rho.ndim - 1)
    raise ValueError(kind)


def _depolarize(rho, qubits, p, n):
    """Depolarizing fault with Pauli-error probability ``p`` (broadcast over the batch)."""
    bits, _, flip = _tables(n)
    m = len(qubits)
    lam = np.asarray(p, float) * (4 ** m) / (4 ** m - 1)
    if not np.any(lam):
        return rho
    lam = lam[..., None, None] if np.ndim(lam) else lam
    if m == 1:
        q = qubits[0]
        f = flip[:, q]
        mixed = 0.5 * (rho + rho[..., f, :][..., :, f])
        mask = (bits[:, q][:, None] == bits[:, q][None, :])
    else:
        q, r = qubits
        fq, fr = flip[:, q], flip[:, r]
        fqr = fq[fr]
        mixed = 0.25 * (rho + rho[..., fq, :][..., :, fq] + rho[..., fr, :][..., :, fr]
                        + rho[..., fqr, :][..., :, fqr])
        mask = ((bits[:, q][:, None] == bits[:, q][None, :]) & (bits[:, r][:, None] == bits[:, r][None, :]))
    return (1 - lam) * rho + lam * mixed * mask


def _dephase(rho, q, p, n):
    bits, _, _ = _tables(n)
    if not p:
        return rho
    same = (bits[:, q][:, None] == bits[:, q][None, :])
    return rho * np.where(same, 1.0, 1.0 - 2.0 * p)


# --------------------------------------------------------------------------- simulators
def statevector(template: Template, A: np.ndarray) -> np.ndarray:
    """Noiseless output states, shape (C, D)."""
    A = np.atleast_2d(A)
    psi = np.zeros((A.shape[0], template.dim), complex)
    psi[:, 0] = 1.0
    for g, gate in enumerate(template.gates):
        psi = _apply_psi(psi, gate.kind, gate.qubits, A[:, g], template.n)
    return psi


def density(template: Template, A: np.ndarray, noise: NoiseModel | None = None) -> np.ndarray:
    """Output density matrices under the gate-level noise model, shape (C, D, D)."""
    A = np.atleast_2d(A)
    n, D = template.n, template.dim
    noise = noise or NoiseModel()
    rho = np.zeros((A.shape[0], D, D), complex)
    rho[:, 0, 0] = 1.0
    if noise.p_init:  # each qubit starts in |1> with probability p_init
        for q in range(n):
            f = _tables(n)[2][:, q]
            rho = (1 - noise.p_init) * rho + noise.p_init * rho[:, f, :][:, :, f]
    ent = set(template.entangling_layers)
    glob = set(template.global_points) if template.global_points else {len(template.gates) - 1}
    eye = np.eye(D) / D
    for g, gate in enumerate(template.gates):
        rho = _apply_rho(rho, gate.kind, gate.qubits, A[:, g], n)
        if gate.kind == "zz":
            if noise.p2:
                rho = _depolarize(rho, gate.qubits, noise.p2_at(A[:, g]), n)
        elif noise.p1:
            rho = _depolarize(rho, gate.qubits, noise.p1, n)
        if g in ent and noise.p_idle:
            for q in range(n):
                rho = _dephase(rho, q, noise.p_idle, n)
        if noise.global_p and g in glob:
            tr = np.real(np.einsum("cii->c", rho))[:, None, None]
            rho = (1 - noise.global_p) * rho + noise.global_p * tr * eye
    return rho


def probabilities(rho: np.ndarray, noise: NoiseModel | None = None, n: int | None = None) -> np.ndarray:
    """Outcome distribution of a computational-basis measurement, including readout flips."""
    p = np.real(np.einsum("...ii->...i", rho)).copy()
    if noise is not None and (noise.p_meas0 or noise.p_meas1):
        n = n or int(np.log2(p.shape[-1]))
        bits, _, flip = _tables(n)
        for q in range(n):
            stay = np.where(bits[:, q] == 0, 1 - noise.p_meas0, 1 - noise.p_meas1)   # read bit = true bit
            leak = np.where(bits[:, q] == 0, noise.p_meas1, noise.p_meas0)           # read bit b from the other
            p = stay * p + leak * p[..., flip[:, q]]
    p = np.clip(p, 0.0, None)
    return p / p.sum(axis=-1, keepdims=True)


def sample_counts(p: np.ndarray, shots, rng: np.random.Generator) -> np.ndarray:
    """Multinomial counts for each circuit; ``shots`` is an integer or an array of length C."""
    return rng.multinomial(np.broadcast_to(np.asarray(shots), p.shape[:-1]), p)


def shifted_angles(template: Template, A: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
    """Angles of the parameter-shifted circuits, shape (C, d, 2, G) with order (+, -)."""
    A = np.atleast_2d(A)
    pos = template.param_gates()
    out = np.repeat(A[:, None, None, :], template.n_params, axis=1)
    out = np.repeat(out, 2, axis=2)
    for i, g in enumerate(pos):
        s = shift * template.gates[g].scale
        out[:, i, 0, g] += s
        out[:, i, 1, g] -= s
    return out


def z_observables(n: int, pairs: bool = True) -> np.ndarray:
    """Sign table (D, F) of the diagonal observables Z_q and, optionally, Z_q Z_r (q < r)."""
    _, z, _ = _tables(n)
    cols = [z[:, q] for q in range(n)]
    if pairs:
        cols += [z[:, q] * z[:, r] for q in range(n) for r in range(q + 1, n)]
    return np.stack(cols, axis=1).astype(float)


# --------------------------------------------------------------------------- geometry inputs
def pure_qgt(template: Template, A: np.ndarray) -> np.ndarray:
    """Pure-state quantum geometric tensor Q_ij = <d_i psi|(1 - |psi><psi|)|d_j psi>, shape (C, d, d)."""
    A = np.atleast_2d(A)
    n, d = template.n, template.n_params
    pos = template.param_gates()
    owner = {int(g): i for i, g in enumerate(pos)}
    psi = np.zeros((A.shape[0], template.dim), complex)
    psi[:, 0] = 1.0
    dpsi = np.zeros((A.shape[0], d, template.dim), complex)
    for g, gate in enumerate(template.gates):
        psi = _apply_psi(psi, gate.kind, gate.qubits, A[:, g], n)
        dpsi = _apply_psi(dpsi, gate.kind, gate.qubits, A[:, g][:, None], n)
        if g in owner:
            i = owner[g]
            if gate.kind in ("rx", "ry"):
                gen = _left(psi, gate.kind, gate.qubits[0], n, 1)
            else:
                gen = psi * _tables(n)[1][:, gate.qubits[0]]
            dpsi[:, i, :] = -0.5j * gate.scale * gen
    overlap = np.einsum("cia,cja->cij", dpsi.conj(), dpsi)
    proj = np.einsum("cia,ca->ci", dpsi.conj(), psi)
    return overlap - proj[:, :, None] * proj.conj()[:, None, :]


def density_derivatives(template: Template, A: np.ndarray, noise: NoiseModel | None = None):
    """Density matrices and their exact parameter derivatives (parameter-shift rule).

    Returns ``rho`` (C, D, D) and ``drho`` (C, d, D, D).
    """
    A = np.atleast_2d(A)
    C, d, D = A.shape[0], template.n_params, template.dim
    rho = density(template, A, noise)
    S = shifted_angles(template, A).reshape(C * d * 2, -1)
    r = density(template, S, noise).reshape(C, d, 2, D, D)
    # shifted_angles moves each gate angle by +-(pi/2) * scale with scale = +-1, so the
    # difference below is already the derivative with respect to the parameter itself.
    drho = 0.5 * (r[:, :, 0] - r[:, :, 1])
    return rho, drho
