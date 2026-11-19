"""Experiment definitions shared by the local simulations and the emulator runs."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np

from . import data, sim
from .federated import Client, LocalBackend, LocalExecutor, Method, METHODS, RunConfig, run_federated
from .model import VQC

__all__ = ["Scenario", "build", "make_methods", "simulate"]


@dataclass(frozen=True)
class Scenario:
    """A federated task: model, data partition, and the device profile of every client."""
    name: str = "default"
    classes: tuple = (0, 1)
    n_qubits: int = 4
    layers: int = 2
    beta: float = 4.0
    angle_scale: float = 0.5
    n_train_per_class: int = 400
    n_test_per_class: int = 100
    n_per_client: int = 40
    alpha: float | None = None            # Dirichlet concentration of the label partition (None: IID)
    gate_scales: tuple = (1.0, 1.0) + (120.0,) * 8      # noise scale of each client's device
    shots: tuple = (50, 50) + (2,) * 8                  # shots per circuit of each client
    eval_scale: float = 1.0               # noise scale of the reference device used for evaluation
    init_width: float = 0.5               # initial parameters are uniform in [-init_width, init_width]

    @property
    def n_clients(self) -> int:
        return len(self.gate_scales)


def build(sc: Scenario, seed: int, base_noise: sim.NoiseModel | None = None, device_factory=None):
    """Model, clients, initial parameters, data, and the evaluation device.

    By default every client owns a :class:`LocalBackend` with the gate-level noise model
    scaled by its ``gate_scale``.  ``device_factory(gate_scale, role, index)`` can supply
    other device objects, for example emulator profiles for :class:`qfedqgt.nexus.NexusExecutor`.
    """
    base = base_noise or sim.NoiseModel.reimei()
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    model = VQC(t, len(sc.classes), beta=sc.beta)
    ds = data.load_mnist(sc.classes, t.n_data, sc.n_train_per_class, sc.n_test_per_class,
                         angle_scale=sc.angle_scale, seed=seed)
    parts = data.partition(ds.y_train, sc.n_clients, sc.n_per_client, sc.alpha, seed=seed)
    clients = []
    for k, idx in enumerate(parts):
        if device_factory is None:
            device = LocalBackend(base.scaled(sc.gate_scales[k]), seed=10_000 * (seed + 1) + k)
        else:
            device = device_factory(sc.gate_scales[k], "client", k)
        clients.append(Client(ds.X_train[idx], ds.y_train[idx], device, sc.shots[k], name=f"client{k}"))
    theta0 = np.random.default_rng(1000 + seed).uniform(-sc.init_width, sc.init_width, size=t.n_params)
    if device_factory is None:
        eval_device = LocalBackend(base.scaled(sc.eval_scale), seed=777 + seed)
    else:
        eval_device = device_factory(sc.eval_scale, "eval", -1)
    return model, clients, theta0, ds, eval_device


def make_methods(lrs: dict, eps: float = 0.05) -> list:
    """``{name: learning rate}`` -> list of :class:`Method`."""
    return [Method(name=name, lr=lr, eps=eps, **METHODS[name.split("@")[0]]) for name, lr in lrs.items()]


def simulate(sc: Scenario, lrs: dict, cfg: RunConfig, seed: int, eps: float = 0.05,
             base_noise: sim.NoiseModel | None = None, evaluate_on_device: bool = True) -> dict:
    """One federated run of several methods on the local simulator (methods run in lockstep)."""
    model, clients, theta0, ds, eval_backend = build(sc, seed, base_noise)
    cfg = replace(cfg, seed=seed)
    return run_federated(model, clients, make_methods(lrs, eps), cfg, theta0, ds.X_test, ds.y_test,
                         LocalExecutor(), eval_backend if evaluate_on_device else None)
