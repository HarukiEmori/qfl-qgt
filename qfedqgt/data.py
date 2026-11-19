"""MNIST loading, feature reduction, and federated partitions."""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist.npz"),
    os.path.expanduser("~/.keras/datasets/mnist.npz"),
)


def mnist_path(download: bool = False) -> str:
    for path in _CANDIDATES:
        if os.path.exists(path):
            return path
    if download:
        os.makedirs(os.path.dirname(_CANDIDATES[0]), exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, _CANDIDATES[0])
        return _CANDIDATES[0]
    raise FileNotFoundError("mnist.npz not found; run scripts/download_mnist.py")


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: tuple

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def load_mnist(classes, n_features: int, n_train_per_class: int, n_test_per_class: int,
               angle_scale: float = 1.0, seed: int = 0) -> Dataset:
    """Principal components of the selected digits, standardized and scaled to rotation angles.

    The PCA basis and the standardization are fitted on the training split only.
    """
    raw = np.load(mnist_path())
    rng = np.random.default_rng(seed)

    def pick(x, y, per_class):
        xs, ys = [], []
        for c_new, c in enumerate(classes):
            idx = np.flatnonzero(y == c)
            idx = rng.choice(idx, size=per_class, replace=False)
            xs.append(x[idx].reshape(per_class, -1).astype(float) / 255.0)
            ys.append(np.full(per_class, c_new))
        return np.concatenate(xs), np.concatenate(ys)

    Xtr, ytr = pick(raw["x_train"], raw["y_train"], n_train_per_class)
    Xte, yte = pick(raw["x_test"], raw["y_test"], n_test_per_class)
    mean = Xtr.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xtr - mean, full_matrices=False)
    W = Vt[:n_features].T
    Ztr, Zte = (Xtr - mean) @ W, (Xte - mean) @ W
    mu, sd = Ztr.mean(axis=0), Ztr.std(axis=0)
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    perm = rng.permutation(len(ytr))
    perm_te = rng.permutation(len(yte))
    return Dataset(angle_scale * Ztr[perm], ytr[perm], angle_scale * Zte[perm_te], yte[perm_te], tuple(classes))


def partition(y: np.ndarray, n_clients: int, n_per_client: int, alpha: float | None, seed: int = 0):
    """Indices of the local datasets.

    ``alpha=None`` gives an IID split.  Otherwise the class proportions of each client
    are drawn from a symmetric Dirichlet distribution with concentration ``alpha``
    (a smaller ``alpha`` is more heterogeneous).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    pools = {c: list(rng.permutation(np.flatnonzero(y == c))) for c in classes}
    out = []
    for _ in range(n_clients):
        if alpha is None:
            props = np.full(len(classes), 1.0 / len(classes))
        else:
            props = rng.dirichlet(np.full(len(classes), alpha))
        counts = np.floor(props * n_per_client).astype(int)
        for _ in range(n_per_client - counts.sum()):
            counts[rng.choice(len(classes), p=props)] += 1
        idx = []
        for c, k in zip(classes, counts):
            take = min(k, len(pools[c]))
            idx += [pools[c].pop() for _ in range(take)]
        # top up from the largest remaining pools if a class ran out
        while len(idx) < n_per_client:
            c = max(pools, key=lambda c: len(pools[c]))
            idx.append(pools[c].pop())
        out.append(np.array(idx))
    return out
