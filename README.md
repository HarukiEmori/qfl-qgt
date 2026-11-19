# qfl-qgt

Code and data for the paper

> **Quantum Federated Learning Based on Bures--Uhlmann Geometry for Heterogeneous Noisy Clients**
> Haruki Emori, Masaki Uchihara, Yuuki Tokunaga
> arXiv:2608.28379

The repository implements QFedQGT, a quantum federated learning scheme in which every client
preconditions its gradient with the Bures metric of the mixed states that its own noisy device
prepares, and the server weights the client updates by their precision.  It contains the simulators,
the federated protocol, the interface to the Quantinuum emulator `reimei-E`, the records of the runs on
that emulator, the numerical checks of the statements of the paper, and the scripts that produce its
figures and tables.

## Layout

| Path | Content |
| :-- | :-- |
| `qfedqgt/sim.py` | Circuit templates, state-vector and density-matrix simulators, gate-level noise model, parameter-shift circuits |
| `qfedqgt/geometry.py` | Mixed-state quantum geometric tensor: Bures metric, mean Uhlmann curvature, incompatibility ratio, closed forms for global depolarizing noise |
| `qfedqgt/model.py` | Variational classifier with Pauli-Z readout and its estimators from measurement counts (gradient, purity) |
| `qfedqgt/federated.py` | Client update, aggregation rules (QFedAvg, QFedFisher, QFedQGT, ablations), federated loop, lockstep execution of several runs |
| `qfedqgt/nexus.py` | Native-gate programs (`PhasedX`, `Rz`, `ZZPhase`), packing of circuits by qubit reuse, execution through Quantinuum Nexus or a local Aer simulator, cache and re-attachment of jobs |
| `qfedqgt/metric.py` | Two-copy measurement of the Bures metric (destructive SWAP test, parameter-shift overlaps) |
| `qfedqgt/provenance.py` | Run records: which program, when, and on which device (shown automatically and saved next to the results) |
| `qfedqgt/scenarios.py` | Scenarios, run configuration, and learning rates of the paper (simulations and emulator experiment) |
| `scripts/verify_theory.py` | Numerical verification of the propositions (prints PASS/FAIL) |
| `scripts/geometry_evaluation.py` | Direct evaluation of the Bures metric and the mean Uhlmann curvature of the client states |
| `scripts/run_campaign.py` | Simulation campaign: step-size selection on tuning seeds and the reported runs |
| `scripts/run_federated.py` | One federated run on a chosen backend (`local`, `aer`, `nexus`) |
| `scripts/run_emulator.py` | Federated training on the emulator: several scenarios and seeds in lockstep, final evaluation on the device |
| `scripts/nexus_geometry.py` | Noise strength from echo circuits on the emulator, compared with the local noise model |
| `scripts/nexus_metric.py` | Two-copy measurement of a block of the Bures metric on the emulator |
| `scripts/nexus_smoke.py` | Smallest job on a Quantinuum backend (run before a campaign) |
| `scripts/capacity_check.py` | Accuracy of the noiseless circuits trained with exact gradients |
| `scripts/download_mnist.py` | Downloads the MNIST archive to `data/` |
| `scripts/make_figures.py` | Figures and table rows from `results/` |
| `notebooks/reimei_E_experiments.ipynb` | The emulator experiments as a notebook; every cell that runs circuits shows its run record |
| `tests/` | Unit tests (`pytest`) |
| `results/` | Results of the runs reported in the paper |
| `legacy/` | Prototype notebook of 2026-03 and its requirements (superseded; kept for the record) |

## Method in brief

1. **Device geometry.**  A client runs echo circuits `U†U` and estimates the purity of its states from the
   frequency of the all-zero outcome.  The purity gives a depolarizing strength `p_k` and the attenuation
   `kappa(p_k) = (1-p_k)^2 / (1-p_k+2p_k/N)` of its Bures metric.
2. **Local update.**  The client estimates the gradient with the parameter-shift rule and takes the
   natural-gradient step `Delta_k = -eta P_k grad`, `P_k = (G_k + eps_k 1)^-1`, with its own Bures metric `G_k`.
3. **Reported statistics.**  The client reports `Delta_k` and the squared standard error `sigma_k^2` of the
   step, computed from the spread of the per-sample gradients.
4. **Aggregation.**  The server uses the weights `w_k ∝ q_k / (zeta^2 + q_k sigma_k^2)`, where `q_k` is the
   data share and `zeta^2` the between-client variance estimated from the reported updates.  The weights of
   a round are computed from the statistics of earlier rounds.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_mnist.py      # writes data/mnist.npz (11 MB)
```

The simulations need only `numpy`, `scipy` and `matplotlib`.  The emulator runs need `pytket`, `qnexus`
and `quantinuum-schemas`; the offline check of the program path needs `pytket-qiskit` and `qiskit-aer`.

## Reproduction

```bash
python -m pytest tests -q                    # unit tests
python scripts/verify_theory.py              # numerical checks of the propositions
python scripts/geometry_evaluation.py        # Bures metric and mean Uhlmann curvature against noise
python scripts/run_campaign.py tune          # step-size grid on the tuning seeds
python scripts/run_campaign.py select        # writes results/tuning/learning_rates.json
python scripts/run_campaign.py main          # reported runs (16 seeds per scenario and method)
python scripts/run_campaign.py sens --scenarios shots_and_noise noise_only_x500   # every step size on the reported seeds
python scripts/make_figures.py all           # figures/geometry.pdf, training.pdf, sweep.pdf
```

The scenarios are defined in `qfedqgt/scenarios.py`.

| Name | Clients | Shots per circuit |
| :-- | :-- | :-- |
| `shots_and_noise` | 2 reference devices, 12 devices with 200-fold error rates | 50 and 2 |
| `noise_only_x<s>`, s = 1, 50, 150, 300, 500 | 2 reference devices, 18 devices with s-fold error rates | 8 |
| `shots_and_noise_4c` | as `shots_and_noise`, four classes, three layers | 50 and 2 |

Every (scenario, seed, method) is one file under `results/`, and finished tasks are skipped when a stage is
run again.  The step sizes are selected on eight tuning seeds (100 to 107) by the final test accuracy, and the
reported runs use the seeds 0 to 15.  The complete campaign takes about three hours on 14 cores.

A single federated run with all methods in lockstep, on the local simulator or on the Quantinuum emulator
(the latter requires an account with access to `reimei-E` and `qnx login`):

```bash
python scripts/run_federated.py --scenario shots_and_noise --backend local --seeds 0
python scripts/run_federated.py --scenario shots_and_noise --backend nexus --device reimei-E --seeds 0
```

The same command with `--device reimei` submits the experiment to the processor.  Finished jobs are cached
under `results/<name>/nexus-cache-seed<seed>/`, and an interrupted run resumes without resubmitting them.
The Nexus interface has been checked offline against a noiseless Aer simulation (`tests/test_programs.py`)
and was run on the syntax checker `reimei-SC` and on the emulator `reimei-E` on 2026-11-19:

```bash
python scripts/nexus_smoke.py --device reimei-SC                       # smallest job, no cost
python scripts/nexus_geometry.py --device reimei-E --shots 100         # echo circuits for six error-rate multipliers
```

`results/nexus-geometry/reimei-E.json` holds the echo measurements of the paper (noise strength inferred on the
emulator against the gate-level noise model).  A complete federated run on the emulator has not been made.  At the
throughput measured on that day (a queue wait of five to ten minutes per batch of jobs and 30 to 45 circuit shots
per second and job) one round of the first scenario with three methods takes about half an hour.
A classical register may hold at most 64 bits on these backends, and the packed programs therefore spread their
bits over several registers.

## Results

Final test accuracy on the reference device (mean over 16 seeds of the average over the last ten rounds).
The last column is the accuracy of QFedQGT minus that of QFedAvg and of QFedFisher in percentage points,
with the standard error of the paired difference.

| Scenario | QFedAvg | QFedFisher | QFedQGT | Difference |
| :-- | :-: | :-: | :-: | :-: |
| `shots_and_noise` | 0.776 | 0.815 | 0.916 | +14.0 ± 2.4, +10.1 ± 1.6 |
| `noise_only_x500` | 0.805 | 0.788 | 0.856 | +5.1 ± 1.7, +6.8 ± 2.2 |
| `noise_only_x300` | 0.784 | 0.809 | 0.844 | +5.9 ± 1.5, +3.4 ± 1.6 |
| `noise_only_x150` | 0.873 | 0.905 | 0.897 | +2.4 ± 2.1, -0.8 ± 1.4 |
| `noise_only_x50` | 0.921 | 0.954 | 0.941 | +2.0 ± 1.7, -1.3 ± 0.8 |
| `noise_only_x1` | 0.957 | 0.966 | 0.957 | 0.0 ± 0.5, -0.9 ± 0.7 |
| `shots_and_noise_4c` | 0.431 | 0.414 | 0.594 | +16.3 ± 1.8, +18.0 ± 2.1 |

The complete tables, including the ablation of the preconditioner and of the weights, are in the paper.
The accuracy of the baselines depends on the step size.  `results/sens-<scenario>/` contains the runs for every
step size of the grids on the reported seeds, which give the step-size sensitivity table of the paper.
The numbers come from the density-matrix simulation of a gate-level noise model (`--backend local`).

## License

MIT License (see `LICENSE`).
