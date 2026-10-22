# qfl-qgt

Code accompanying the paper

> **Quantum Federated Learning Based on Bures--Uhlmann Geometry for Heterogeneous Noisy Clients**
> Haruki Emori, Masaki Uchihara, Yuuki Tokunaga
> arXiv / DOI: not yet assigned (to be added after submission)

## Status of this repository (2026-10-21)

This repository was created on 2026-10-21 by copying the notebook from the author's local virtual-environment folder
(`venv/pytket-dev/qfl/`), where it was not under version control.

**Important:** `QFL-QGT.ipynb` is a *prototype* of the method (QFL with the full quantum geometric tensor).
Its saved outputs correspond to a small statevector study, **not** to the experiment reported in the paper:

| Item | `QFL-QGT.ipynb` (this repo) | Paper (Sec. "Numerical experiments") |
| :-- | :-- | :-- |
| Data | scikit-learn `load_digits`, digits 0 vs 1; synthetic "ADNI" (`make_classification`) | MNIST-10 (ten classes, 3000 training images per class) |
| Model | 4 qubits, 3 layers | 6 qubits, 3 layers, dressed readout head |
| Clients | 5 (shot budgets inf, 4096, 512, 128, 16) | 9 (Table I conditions (a)-(d)) |
| Rounds | 20 | 60 (23 evaluated rounds) |
| Backend | Qiskit Aer statevector | Quantinuum `reimei-E` emulator via Quantinuum Nexus, 4000 shots |

The code that produced the `reimei-E` results, the depth sweep, and the shot-accuracy curve of the paper has not yet been located
and will be added to this repository when found.

## Contents

| File | Role |
| :-- | :-- |
| `QFL-QGT.ipynb` | Prototype implementation: `FastVQC` (statevector VQC with full QGT), clients, server aggregation for `qfl-avg`, `qfl-fisher`, `qfl-qgt`; runs 20 rounds on two datasets and plots accuracy |
| `qfl-qgt_res_final.pdf` | Figure written by the notebook (`plt.savefig("qfl-qgt_res_final.pdf")`) |
| `qfl-qgt_res.pdf` | Earlier version of the same figure |
| `requirements.txt` | Package versions read from the local environment (see the comment in the file) |

## Environment

- Python 3.12 (notebook metadata: 3.12.12)
- Package versions: see `requirements.txt` (read from the local `pytket-dev` environment on 2026-10-21; the exact versions at the time of the last run are unverified)

## Reproduction

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab QFL-QGT.ipynb   # run all cells (about 10 minutes on a laptop, according to the saved timings)
```

## License

MIT License (see `LICENSE`).
