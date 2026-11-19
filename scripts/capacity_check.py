#!/usr/bin/env python
"""Capacity of the noiseless circuit: full-batch training with exact gradients.

The script trains the classifier of a scenario without device noise and without shot
noise on 200 training images (gradient descent with momentum) and reports the test
accuracy.  It gives the reference value quoted for the four-class task in the paper.

    python scripts/capacity_check.py            # writes results/capacity.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qfedqgt import data, sim                 # noqa: E402
from qfedqgt.model import VQC                 # noqa: E402
from qfedqgt.scenarios import SCENARIOS       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def capacity(name: str, seed: int = 0, n_train: int = 200, steps: int = 150, lr: float = 0.3, momentum: float = 0.9) -> dict:
    sc = SCENARIOS[name]
    t = sim.layered_ansatz(sc.n_qubits, sc.layers)
    model = VQC(t, len(sc.classes), beta=sc.beta)
    ds = data.load_mnist(sc.classes, t.n_data, sc.n_train_per_class, sc.n_test_per_class,
                         angle_scale=sc.angle_scale, seed=seed)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-sc.init_width, sc.init_width, size=t.n_params)
    idx = rng.choice(len(ds.y_train), n_train, replace=False)
    X, y = ds.X_train[idx], ds.y_train[idx]
    velocity = np.zeros_like(theta)
    for _ in range(steps):
        velocity = momentum * velocity + model.gradient_exact(X, y, theta)
        theta = theta - lr * velocity
    return dict(scenario=name, seed=seed, n_params=int(t.n_params), n_train=n_train, steps=steps,
                train_accuracy=model.accuracy_exact(X, y, theta),
                test_accuracy=model.accuracy_exact(ds.X_test, ds.y_test, theta),
                test_loss=model.loss_exact(ds.X_test, ds.y_test, theta))


if __name__ == "__main__":
    out = [capacity(name, seed) for name in ("shots_and_noise", "shots_and_noise_4c") for seed in (0, 1, 2)]
    for r in out:
        print(f"{r['scenario']:20s} seed {r['seed']}  d={r['n_params']}  train {r['train_accuracy']:.3f}  test {r['test_accuracy']:.3f}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "capacity.json"), "w") as f:
        json.dump(out, f, indent=1)
