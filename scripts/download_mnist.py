#!/usr/bin/env python
"""Download the MNIST dataset (11 MB) to ``data/mnist.npz``.

    python scripts/download_mnist.py

The file is the NumPy archive distributed with Keras (arrays x_train, y_train, x_test, y_test).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qfedqgt.data import mnist_path   # noqa: E402

if __name__ == "__main__":
    print("MNIST:", mnist_path(download=True))
