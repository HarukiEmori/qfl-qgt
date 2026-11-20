# Plan of the two-class experiment on the emulator `reimei-E`

This file fixes the design, the step sizes, the number of seeds and the analysis of the experiment
`emulator2_iid` (results folder `results/emulator2-reimei-E/`).  It was committed before the first job of
the experiment was submitted.  The commit that contains this file is the record of what was decided in
advance; later changes of this file are visible in the history of the repository.

## Question

Does QFedQGT reach a higher test accuracy than QFedAvg and QFedFisher when the clients differ only in
their devices and in their shot budgets, at accuracies that are not limited by the circuit?

The first experiment on the emulator (four classes, one-layer circuit on four qubits) cannot answer the
second part: the noiseless accuracy of that circuit on that task is about 0.72, and all accuracies of
that experiment are between 0.27 and 0.42.  The second experiment uses the two-class task (digits 0 and
1) and a one-layer circuit on two qubits, which reaches 0.99 without noise (`scripts/capacity_check.py`).

## Design (fixed)

| Item | Value |
| :-- | :-- |
| Task | MNIST digits 0 and 1, 400 training and 100 test images per class and seed |
| Model | 2 qubits, 1 layer, 4 parameters, 9 circuits per image |
| Clients | 8 clients with 40 images each, IID data, equal data shares |
| Reference clients | 2 clients on the emulator with its own error parameters, 16 shots per circuit |
| Noisy clients | 6 clients with the one-qubit, two-qubit and memory error parameters multiplied by 400, 1 shot per circuit |
| Training | minibatch of 8 images per client and round, 25 rounds, 5 shots per echo circuit |
| Methods | QFedAvg, QFedFisher, QFedQGT with the same circuits, minibatches and shots |
| Step sizes | QFedAvg 0.6, QFedFisher 0.3, QFedQGT 0.1 (selected on the tuning seeds 100 to 131 in the local model) |
| Seeds on the emulator | 0 to 15 (16 seeds), all in one campaign |
| Evaluation on the emulator | the models of the last 5 rounds, each on its own class-balanced block of 20 test images, 100 shots per image, reference error parameters |

The scenario is `emulator2_iid` in `qfedqgt/scenarios.py`; the command of the campaign is

```bash
PYTHON=python scripts/emulator_campaign.sh emulator2-reimei-E
```

It is started after a pre-check on the emulator (see below).

## How the design was chosen

All choices were made in the local gate-level model.  In this model the noisy clients have the
multiplier that reproduces the echo survival measured on the emulator for the same circuit
(`scripts/nexus_geometry.py`, `results/nexus-geometry/`).

1. Noise of the noisy clients.  Echo circuits of the two-qubit circuit were run on the emulator with
   the multipliers 1, 300, 400, 500, 600 and 700 (`reimei-E-calibration_n2.json`).  At 400 the survival
   is 0.314 ± 0.016, the inferred noise strength is 0.71, and the matched multiplier of the local model
   is 537 (540 is used).  At 600 and 700 the survival is below 1/4, the value of the maximally mixed
   state; these multipliers are not used.
2. Step sizes: selected for every method on the tuning seeds 100 to 131 (`scripts/run_campaign.py tune`
   and `select`), by the final test accuracy.  The runs of the two-qubit circuit are short, and 32
   tuning seeds are used (8 in the other experiments), because the selection with 8 seeds was not
   stable for the baselines.
3. Design: candidates that differ in the number of qubits, the number of noisy clients, the minibatch
   size, the shots of the reference clients, the number of rounds and the data partition were compared
   on the design seeds 200 to 215 (`scripts/emulator_design.py`, `results/emulator-design/`).  The
   cost of a run on the emulator is proportional to the number of circuits times the number of shots,
   so the candidates are compared at their cost.
4. The seeds 0 to 15 of the emulator were used for neither choice.

Observations of the design study that determined the design:

- The accuracy of QFedQGT is set by the images and shots that the reference clients process per
  round.  The reference clients dominate the cost.
- With the one-layer circuit on four qubits (8 parameters, 17 circuits per image, 0.279 HQC per image
  and shot) the accuracy of QFedQGT after 25 rounds is 0.845 with 5 shots of the reference clients
  (3,400 HQC per run with the three methods) and 0.871 with 20 shots (8,400 HQC per run).  The circuit
  on two qubits (4 parameters, 9 circuits per image, 0.065 HQC per image and shot) reaches a noiseless
  accuracy of 0.99 on this task, and the chosen design reaches 0.923 at 1,800 HQC per run.
- Noisy clients with one shot per circuit add little to the cost and lower the accuracy of the
  baselines.
- With a Dirichlet partition of the labels the advantage of QFedQGT decreases with the label skew:
  QFedQGT puts most of the weight on the two reference clients, whose data then do not represent the
  two classes equally.  The experiment on the emulator uses IID data, and the dependence on the label
  skew is reported from the local model.

The scenario `emulator2_iid` was first defined with four qubits, 5 shots of the reference clients and
15 rounds (commit 5abf550).  That design was replaced by the present one before any job of the
experiment was submitted, for the reason given in the second observation.

## Prediction of the local model (made before the runs)

For the seeds 0 to 15 with the selected step sizes (`results/sim-emulator2_iid/`, evaluation on the
reference device with 100 shots on all 200 test images, mean over the last 5 rounds):

| Method | Accuracy | QFedQGT minus the method (points) | Seeds with QFedQGT higher |
| :-- | :-: | :-: | :-: |
| QFedAvg | 0.796 ± 0.025 | +12.9 ± 2.3 | 16 of 16 |
| QFedFisher | 0.779 ± 0.024 | +14.6 ± 2.4 | 15 of 16 |
| QFedQGT | 0.925 ± 0.008 | | |

On these seeds the selected step size of every method is also the value of its grid with the highest
accuracy (`results/sens-emulator2_iid/`).

The same 48 runs with the procedure of the emulator (runs in lockstep, evaluation of the last 5 rounds
on blocks of 20 test images; `results/emulator2-local/`) give 0.802 ± 0.027, 0.794 ± 0.024 and
0.924 ± 0.011, with differences of +12.2 ± 3.0 and +12.9 ± 2.6 points (QFedQGT higher in 14 of 16
seeds for both baselines).

In the first experiment the differences measured on the emulator agreed with those of the local model
within their standard errors, which were large (6 seeds).  The differences on the emulator may be
smaller than the values above.

Two variants of QFedAvg were run in the local model on the same seeds and are not run on the emulator
(`scripts/run_campaign.py`, methods `QFedAvg-shots` and `Euclid+re`).  The average of plain gradients
with weights proportional to the shots reaches 0.886 ± 0.021, and QFedQGT is +3.9 ± 2.3 points above
it.  With the weights of QFedQGT on plain gradients the accuracy is 0.879 ± 0.017, and QFedQGT is
+4.6 ± 1.7 points above it.  Most of the difference between QFedQGT and QFedAvg in this design is
therefore due to the weights.  The experiment on the emulator compares QFedQGT with QFedAvg and
QFedFisher, as stated above.

## Analysis (fixed)

- Primary outcome: the final test accuracy measured on the emulator, that is, the mean over the
  evaluated rounds, for every seed and method.
- Secondary outcome: the accuracy of the noiseless circuit with the parameters trained on the
  emulator, on all 200 test images, as the mean over the last 5 rounds.
- Comparison: for each baseline, the mean over the 16 seeds of the paired difference
  QFedQGT minus baseline, its standard error, and the number of seeds in which QFedQGT is higher.
- All 16 seeds are reported.  No seed is added or removed after the results are known.  Jobs that end
  with an error of the service are submitted again; an interrupted campaign continues from the cache.
- The results are reported whatever their sign and size.

## Cost and duration (estimate)

- Training: 16 runs x 3 methods x 25 rounds x 8 images x 9 circuits x (2 x 16 + 6 x 1) shots
  = 3.28 million circuit-shots in 691,200 circuits, about 23,600 HQC by the cost formula.
- Echo circuits of QFedQGT: about 1,300 HQC.
- Evaluation: 16 runs x 3 methods x 100 image evaluations x 100 shots = 480,000 circuit-shots, about
  3,500 HQC.
- Constant of 5 HQC per program: 14 programs per round with 2,448 circuits per program, about 1,800 HQC.
- Total: about 30,000 HQC.  Duration: about half an hour per round, 12 to 17 hours in total.

## Pre-check on the emulator

The campaign is started by `scripts/emulator_campaign_after_precheck.sh`.  It first completes one round of
two runs with other seeds (900 and 901; folder `results/emulator2-smoke/`), which use the circuits, the
programs and the devices of the experiment, and tests their results with `scripts/emulator_precheck.py`:
every job completed, finite results, and the noise strengths estimated from the echo circuits as expected
(at most 0.1 on average for the reference clients, and within 0.2 of the echo measurement 0.71 for the
noisy clients).  The campaign starts only if these tests pass.  The runs of the pre-check are not part of
the experiment, and their accuracies are not used.

```bash
PYTHON=python scripts/emulator_campaign_after_precheck.sh emulator2-smoke emulator2-reimei-E \
    --scenarios emulator2_iid --seeds 900 901 --rounds 1 --no-eval
```

A job that is cancelled, or programs whose jobs fail three times, stop the campaign for good
(`scripts/emulator_campaign.sh`); the same command continues it later.

## Optional second condition (not run unless decided separately)

A Dirichlet partition of the labels with the same design and the same seeds would be run as a separate
campaign with its own results folder, scenario and plan.  The dependence of the differences on the
Dirichlet parameter in the local model is in `results/emulator-design/` (candidates `..._dir5`,
`..._dir2`, `..._dir1`, `..._dir0.5`).
