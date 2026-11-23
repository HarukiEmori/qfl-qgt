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
| `qfedqgt/nexus.py` | Native-gate programs (`PhasedX`, `Rz`, `ZZPhase`), packing of circuits by qubit reuse, execution through Quantinuum Nexus or a local Aer simulator, cache and re-attachment of jobs, check of the number of shots of every downloaded result |
| `qfedqgt/metric.py` | Two-copy measurement of the Bures metric (destructive SWAP test, parameter-shift overlaps) |
| `qfedqgt/provenance.py` | Run records: which program, when, and on which device (shown automatically and saved next to the results) |
| `qfedqgt/scenarios.py` | Scenarios, run configuration, and learning rates of the paper (simulations and emulator experiment) |
| `scripts/verify_theory.py` | Numerical verification of the propositions (prints PASS/FAIL) |
| `scripts/geometry_evaluation.py` | Direct evaluation of the Bures metric and the mean Uhlmann curvature of the client states |
| `scripts/run_campaign.py` | Simulation campaign: step-size selection on tuning seeds and the reported runs |
| `scripts/run_federated.py` | One federated run on a chosen backend (`local`, `aer`, `nexus`) |
| `scripts/run_emulator.py` | Federated training on the emulator: several scenarios and seeds in lockstep, final evaluation on the device |
| `scripts/emulator_campaign.sh` | Runs a campaign of `run_emulator.py` until it has finished and restarts it after an interruption |
| `scripts/emulator_status.py` | Progress of a campaign (rounds, jobs, cost), read from its results folder |
| `scripts/emulator_precheck.py`, `scripts/emulator_campaign_after_precheck.sh` | Tests of a short pre-check run on the emulator (jobs completed, finite results, expected noise strengths), and a script that starts a campaign only after these tests pass |
| `scripts/emulator_design.py` | Candidate designs of the two-class emulator experiment, compared in the local model before the runs |
| `scripts/emulator_multiplier_check.py` | Echo frequency of the noisy clients in the pre-check run, the matched multiplier of the local model, and the step sizes and prediction of the two-class experiment at several multipliers |
| `scripts/emulator_summary.py` | Final accuracies and paired differences of the emulator experiments |
| `scripts/emulator_model_comparison.py` | Two-class experiment on the emulator against the local model: accuracies, paired differences, and a check of every measured evaluation against its expectation |
| `scripts/nexus_redownload.py` | Finds stored results with fewer shots than their jobs requested and downloads them again (submits nothing) |
| `scripts/nexus_geometry.py` | Noise strength from echo circuits on the emulator, compared with the local noise model |
| `scripts/nexus_metric.py` | Two-copy measurement of a block of the Bures metric on the emulator |
| `scripts/nexus_position.py` | Survival of an echo circuit against its position in a packed program |
| `scripts/nexus_job_times.py` | Waiting and execution times of the jobs of a campaign |
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
python scripts/make_figures.py all           # figures/geometry.pdf, training.pdf, sweep.pdf, emulator.pdf, emulator2.pdf
```

The scenarios are defined in `qfedqgt/scenarios.py`.

| Name | Clients | Shots per circuit |
| :-- | :-- | :-- |
| `shots_and_noise` | 2 reference devices, 12 devices with 200-fold error rates | 50 and 2 |
| `noise_only_x<s>`, s = 1, 50, 150, 300, 500 | 2 reference devices, 18 devices with s-fold error rates | 8 |
| `shots_and_noise_4c` | as `shots_and_noise`, four classes, three layers | 50 and 2 |

Every (scenario, seed, method) is one file under `results/`, and finished tasks are skipped when a stage is
run again.  The step sizes are selected on eight tuning seeds (100 to 107) by the final test accuracy, and the
reported runs use the seeds 0 to 15.  The two-class emulator experiment (`emulator2_iid`), whose runs are short,
uses 32 tuning seeds (100 to 131).  The complete campaign takes about three hours on 14 cores.

A single federated run with all methods in lockstep, on the local simulator or on the Quantinuum emulator
(the latter requires an account with access to `reimei-E` and `qnx login`):

```bash
python scripts/run_federated.py --scenario shots_and_noise --backend local --seeds 0
python scripts/run_federated.py --scenario shots_and_noise --backend nexus --device reimei-E --seeds 0
```

## Experiments on the emulator `reimei-E`

The emulator experiments of the paper were run from 2026-11-19 to 2026-11-21 (UTC) through Quantinuum Nexus.  Their shot tables,
job logs and run records are in `results/`, so that every command below reproduces its result from the cache
without a login.  With `--cached-only` a script never submits a job and stops if a result is missing.  Such a
replay recomputes the results from the stored shot tables, compares them with the stored result files, and
reports whether they are reproduced; it does not overwrite the stored results or the stored run records.  To
repeat a measurement on the emulator, remove its cache folder, log in with `qnx login`, and run the command
without `--cached-only`.

```bash
python scripts/nexus_geometry.py --device reimei-E --shots 100 --cached-only            # echo circuits, two-layer circuit
python scripts/nexus_geometry.py --device reimei-E --scenario emulator_iid --scales 1 100 200 300 400 500 --cached-only
python scripts/nexus_metric.py --backend nexus --device reimei-E --scales 1 50 --shots 200 --layer 1 --cached-only
python scripts/nexus_position.py --device reimei-E --scales 1 300 --shots 60 20 --circuits 600 --cached-only  # position in a program
python scripts/run_emulator.py --backend nexus --device reimei-E --name emulator-reimei-E --cached-only
python scripts/run_emulator.py --backend nexus --device reimei-E --name emulator2-reimei-E --cached-only   # two-class experiment
python scripts/emulator_summary.py                                                      # accuracies and paired differences
python scripts/emulator_model_comparison.py                                             # two-class experiment against the local model
python scripts/nexus_redownload.py --all                                                # shots of every stored result against its job
python scripts/make_figures.py emulator
python scripts/make_figures.py emulator2
```

The federated training is organized in campaigns (`EMULATOR_CAMPAIGNS` in `qfedqgt/scenarios.py`).  The runs of
a campaign advance round by round together, and one results folder holds one campaign.  The seeds of a campaign
and the packing of its circuits into programs are stored in `packing.json` of its folder and are reused by the
command above.  The four-class experiment has two campaigns with the same protocol: `results/emulator-reimei-E/`
(seeds 0 to 5; the packing changed after the fourth round from 528 to 816 circuits per program) and
`results/emulator-reimei-E-2/` (seeds 6 to 15, run after the first campaign had been evaluated, to reach the 16 seeds
of the simulations).  All 16 seeds are reported, and `scripts/emulator_summary.py` also lists the two campaigns
separately.  The second campaign was interrupted for about ten hours while the client computer slept; no job was
submitted twice.

The same steps are in `notebooks/reimei_E_experiments.ipynb`, which is stored with its output.

**Two experiments.**  The first experiment (`emulator_iid`, `emulator_dirichlet`; folders `emulator-reimei-E*`) has four
classes and a one-layer circuit on four qubits.  The noiseless accuracy of that circuit on the four-class task is about
0.72, so all accuracies of this experiment are low (between 0.27 and 0.42 on the emulator).  The second experiment (`emulator2_iid`; folder
`emulator2-reimei-E`) uses the two-class task of the simulations and a one-layer circuit on two qubits, which solves
the task (0.99 without noise) and costs about a quarter of the four-qubit circuit per image and shot: two reference
clients with 16 shots per circuit, six clients with 400-fold error rates and one shot, 8 images per client and round,
25 rounds, IID data, 16 seeds.  Its design, step sizes, number of seeds and analysis were fixed before the runs on the
emulator (`plans/emulator2.md`); the candidate designs were compared in the local model with
`scripts/emulator_design.py` (`results/emulator-design/`), and the multiplier of the local model was chosen so that it
reproduces echo measurements of the same circuit on the emulator (`results/nexus-geometry/reimei-E-calibration_n2.json`;
the dependence of the prediction on this multiplier is in `results/emulator-design/multiplier-sensitivity.json`).

**Running a campaign.**  A campaign needs one submission per round, so a process has to stay alive for hours:

```bash
# one round of two runs with other seeds, tests of their results, and then the campaign
PYTHON=python scripts/emulator_campaign_after_precheck.sh emulator2-smoke emulator2-reimei-E \
    --scenarios emulator2_iid --seeds 900 901 --rounds 1 --no-eval
PYTHON=python scripts/emulator_campaign.sh emulator2-reimei-E      # the campaign alone; continues it after a stop
python scripts/emulator_status.py --name emulator2-reimei-E        # progress: rounds, jobs, recorded cost
python scripts/emulator_status.py --name emulator2-reimei-E --nexus   # also asks Nexus for the open jobs
```

`emulator_campaign.sh` starts `run_emulator.py` and starts it again whenever it stops before the end.  Every start
reads the finished rounds from the cache and re-attaches to the submitted jobs, so nothing is submitted twice.
If a request to Nexus does not return (for example after the computer slept), the client ends its process after ten
minutes (`--stall-timeout`), and the script restarts it.  On macOS the script keeps the computer awake.  A closed lid
still puts a laptop to sleep; the submitted jobs keep running on the emulator, and the campaign continues after the
computer wakes up.  The output is appended to `results/<name>/campaign.log`.

To stop a campaign, end the script and the client (`kill $(cat results/<name>/campaign.pid)`, then the process of
`run_emulator.py`).  The jobs of the current round that are already submitted still run on the emulator unless they
are cancelled in the web interface of Nexus.  After a cancelled job, and after three failed jobs of the same programs,
the client ends with the exit status 76, and the script does not restart it; the same command continues the campaign
later.

| Result | Content |
| :-- | :-- |
| `results/nexus-geometry/reimei-E.json`, `reimei-E-emulator_iid.json` | Frequency of the all-zero outcome of echo circuits for several error-rate multipliers, the inferred noise strength, and the multiplier at which the local gate-level model has the same frequency |
| `results/metric/reimei-E.json` | Two-copy measurement of the 8 x 8 block of the Bures metric of one layer (137 circuits on 8 qubits, 200 shots) |
| `results/nexus-position/reimei-E.json` | Frequency of the all-zero outcome of 600 copies of one echo circuit in a single program, for every position: the change from the first to the last circuit is -0.0024 ± 0.0018 at the reference error rates and +0.003 ± 0.010 at 300-fold error rates |
| `results/emulator-design/` | Candidate designs of the two-class experiment in the local model: circuits on two, three and four qubits, numbers of noisy clients, minibatch sizes, shots, and Dirichlet partitions (step sizes on the tuning seeds 100 to 131, comparison on the seeds 200 to 215) |
| `results/nexus-geometry/reimei-E-calibration_n2.json`, `...-calibration_n3.json` | Echo survival of the one-layer circuits on two and three qubits for error-rate multipliers from 1 to 700, with the matched multipliers of the local model.  For the two-qubit circuit the survival is 0.314 ± 0.016 at the multiplier 400 and falls below 1/4, the value of the maximally mixed state, at 600 and 700 |
| `results/emulator-reimei-E/`, `results/emulator-reimei-E-2/` | Federated training on the emulator: four clients (two reference clients with 30 shots per circuit, two clients with 300-fold error rates and one shot), digits 0 to 3, one layer, 15 rounds, IID data and a Dirichlet label partition, 16 seeds each in two campaigns (seeds 0 to 5 and 6 to 15) |
| `results/emulator2-smoke/` | Pre-check of the two-class experiment: one round of two runs with the seeds 900 and 901 (5 jobs, among them a program with 2448 two-qubit circuits), and the tests of `scripts/emulator_precheck.py` (`precheck.json`, `precheck.log`) |
| `results/emulator-local*/`, `results/sim-emulator_*`, `results/sens-emulator_*` | The same experiment in the local gate-level model: the same seeds with the same grouping into campaigns, 16 reported seeds, and every step size of the grids |
| `results/emulator2-reimei-E/` | Two-class experiment on the emulator: eight clients (two reference clients with 16 shots per circuit, six clients with 400-fold error rates and one shot), digits 0 and 1, one layer on two qubits, 25 rounds, IID data, seeds 0 to 15 in one campaign of 352 jobs.  `redownload.log` and `provenance-rewrite.json` record the result that was downloaded again (see below) |
| `results/emulator2-local/`, `results/sim-emulator2_iid/`, `results/sens-emulator2_iid/` | The two-class experiment in the local gate-level model: the same runs with the evaluation procedure of the emulator, the prediction of `plans/emulator2.md`, and every step size of the grids |
| `results/emulator2-summary.json`, `results/emulator2-comparison.json` | Accuracies and paired differences of the two-class experiment, and its comparison with the local model (`scripts/emulator_summary.py`, `scripts/emulator_model_comparison.py`) |

**Run records.**  Every script that executes circuits ends by showing a run record and saves it next to its
results (`provenance.json`): which program produced the result (repository, commit, state of the working tree,
entry point), when (start and end, submission and completion of every job), and on which device (device name,
error-rate multipliers, job identifiers, shots, cost).  In a notebook the record is displayed as a table when
the cell finishes.  The commit is logged when a job is submitted, and results read from the cache keep the
record of the jobs that produced them.

```python
from qfedqgt.provenance import record_run
with record_run("my experiment", executor=executor, save_to="results/my-experiment"):
    executor.run(requests)
```

**Completeness of downloaded results.**  The executor compares the number of shots of every downloaded result with
the request.  It repeats the download of a result with too few shots and never stores such a result; if the result
stays incomplete, the run stops and the job remains re-attachable.  A stored table with a wrong number of shots is
refused when it is read.  This check was added after the two-class campaign.  The result of the first of its two
evaluation jobs, read at the moment the job was reported as completed, held the first 12 of its 100 shots.
`scripts/nexus_redownload.py` found the table, downloaded the complete result of the same job again, and kept the
incomplete table in `results/emulator2-reimei-E/cache/incomplete/`; nothing was submitted.  The result files were then
written anew from the cache (`scripts/run_emulator.py --cached-only --rewrite`; see `redownload.log` and
`provenance-rewrite.json` in the results folder), and only the accuracies measured on the emulator for the seeds 0 to 8
changed.  The incomplete result was noticed because the accuracies measured on the emulator deviated from their
expectations under the model of the reference device by much more than their statistical spread
(`scripts/emulator_model_comparison.py`; the numbers are in the Results section).  No other stored table of the
repository has fewer shots than its job requested (`python scripts/nexus_redownload.py --all`).

**What limits an experiment on the emulator.**  Measured on 2026-11-19:

- A job waits between 3 and 30 minutes before it starts.  This wait occurs in every round of a training.
- The jobs of one account ran almost one after another (on average 1.3 at a time), so the duration of a round is the
  wait plus the summed run time of its jobs.  Circuits times shots therefore determine both the duration and the
  cost (about 16 HQC per 1000 circuit-shots of the four-qubit one-layer circuit).
- The run time of a job with a single program grows with the number of circuits times the number of shots:
  about 6 s plus 0.011 s per circuit and shot for four-qubit one-layer circuits (528 circuits took 13 s with one
  shot and about 3 minutes with 30 shots).  Jobs with several programs were much slower (2 to 4 minutes per
  program), so every program is submitted as its own job.
- Programs of 528, 816 and 1224 four-qubit circuits were accepted; a program of 2112 four-qubit circuits was not
  accepted within 20 minutes.  A program of 2448 two-qubit circuits (the same number of measured bits as 1224
  four-qubit circuits) was accepted and executed.  A classical register may hold at most 64 bits, and the packed
  programs spread their bits over several registers.
- For strong multipliers the frequency of the all-zero outcome of an echo circuit falls below the value of the
  maximally mixed state (one-layer circuits: from the multiplier 500 on four and three qubits and from 600 on two
  qubits).  The error probabilities are then saturated, a depolarizing model no longer describes the device, and the
  local model has no matching multiplier; the experiments use multipliers below this range (300 and 400).
- The default waiting of the client (`qnx.jobs.wait_for`) polls with intervals of up to 20 minutes; the
  executor asks for the job status every 20 s instead.

- In the two-class campaign of 2026-11-20 (352 jobs with one program each, 14 jobs per round) a job waited between
  0.3 and 8.3 minutes (median 5.4), and a round took between 18 and 39 minutes.
- A result that is read at the moment its job is reported as completed can hold only the first shots (once among
  the results of the 722 jobs stored in this repository, see above).  The executor checks the number of shots of every result.

One round of the four-class experiment (12 runs, 4992 circuits) therefore took about 20 to 30 minutes, and the
experiment is much smaller than the simulations.  The cost formula gives 0.279 HQC per image and shot for the
one-layer circuit on four qubits (17 circuits per image) and 0.065 HQC for the one on two qubits (9 circuits per
image), which is why the two-class experiment uses the two-qubit circuit.  Interrupted campaigns resume from the cache and re-attach to
the jobs that are still running (`qfedqgt/nexus.py`).

## Results

### Emulator `reimei-E`

Final test accuracy of the models trained on the emulator, as the mean over 16 seeds of the average over the last
5 rounds.  `Device` is measured on the emulator (60 test images, 20 shots each), and `Noiseless` is the accuracy
of the noiseless circuit with the same parameters on all 400 test images.  The differences are the accuracy of
QFedQGT minus that of QFedAvg and of QFedFisher in percentage points with the standard error of the paired
difference, followed by the number of seeds in which QFedQGT is higher.

| Data | Accuracy | QFedAvg | QFedFisher | QFedQGT | Difference | Seeds |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: |
| IID | Device | 0.310 | 0.274 | 0.393 | +8.3 ± 2.8, +11.9 ± 2.9 | 11/16, 14/16 |
| IID | Noiseless | 0.307 | 0.287 | 0.398 | +9.1 ± 2.6, +11.0 ± 3.0 | 13/16, 13/16 |
| Dirichlet partition | Device | 0.300 | 0.314 | 0.375 | +7.5 ± 2.1, +6.1 ± 1.8 | 15/16, 12/16 |
| Dirichlet partition | Noiseless | 0.281 | 0.285 | 0.371 | +9.0 ± 3.0, +8.6 ± 2.5 | 13/16, 14/16 |

Averaged over the two partitions, with one value per seed, the differences are +7.9 ± 2.0 and +9.0 ± 1.8 points on the device and
+9.0 ± 2.2 and +9.8 ± 2.0 points for the noiseless circuit.  All four differences measured on the device exceed twice their standard errors.
The same experiment in the local gate-level model with 16 seeds (`results/sim-emulator_*`) gives differences of
+12.4 ± 2.4 and +13.7 ± 2.0 points (IID) and +9.8 ± 1.8 and +10.2 ± 2.3 points (Dirichlet partition).
The 32 runs comprised 228,480 circuits and 3,630,720 shots in 337 jobs.  `python scripts/emulator_summary.py` prints these numbers.

### Emulator `reimei-E`: two-class experiment

Final test accuracy of the models trained on the emulator, as the mean over 16 seeds of the average over the last
5 rounds (`results/emulator2-reimei-E/`).  The design, the step sizes, the seeds and the analysis were fixed before
the runs (`plans/emulator2.md`).  `Device` is measured on the emulator (100 test images, 100 shots each), and
`Noiseless` is the accuracy of the noiseless circuit with the same parameters on all 200 test images.

| Accuracy | QFedAvg | QFedFisher | QFedQGT | Difference | Seeds |
| :-- | :-: | :-: | :-: | :-: | :-: |
| Device | 0.872 | 0.812 | 0.918 | +4.6 ± 2.2, +10.6 ± 2.6 | 11/16, 13/16 |
| Noiseless | 0.868 | 0.796 | 0.926 | +5.8 ± 2.8, +13.0 ± 2.7 | 10/16, 14/16 |

The differences measured on the device are 2.1 and 4.0 times their standard errors.  QFedQGT and QFedAvg have the same accuracy in one seed on the device.
The prediction of the local gate-level model, made before the runs, was 0.796, 0.779 and 0.925 with
differences of +12.9 ± 2.3 and +14.6 ± 2.4 points.  The differences on the emulator are smaller than this
prediction by 8.3 ± 3.5 and 4.0 ± 3.7 points (paired over the seeds).  Against the model with the evaluation
procedure of the emulator (`results/emulator2-local/`), the accuracy on the emulator minus that of the model is
-0.6 ± 1.6 points for QFedQGT, +7.0 ± 3.1 points for QFedAvg and +1.8 ± 3.1 points for QFedFisher: QFedQGT agrees
with the model, and the baselines reach higher accuracies on the emulator than in the model.
`python scripts/emulator_model_comparison.py` prints this comparison (`results/emulator2-comparison.json`).

Check of the evaluation: every accuracy measured on the emulator (one per seed, method and evaluated round, 240 in all)
is compared with its expectation for the same parameters and images under the model of the reference device.  The
root mean square of the deviation in units of its standard deviation is 1.06.
With the result that was first downloaded with 12 of its 100 shots (see above) it was 2.73, and the accuracies
were 0.834, 0.803 and 0.891 with differences of +5.7 ± 2.0 and +8.8 ± 2.9 points.  These values are superseded by
the table above, which uses the complete result.
The 16 runs comprised 721,600 circuits and 3,891,200 shots in 352 jobs within 11.6 hours
(28,579 HQC charged; the cost formula gives 30,186 HQC).

### Simulations with the gate-level noise model

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
