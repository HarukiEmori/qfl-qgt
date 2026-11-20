"""Scenarios, run configuration, and learning rates of the experiments in the paper.

The scenarios use the two-class MNIST task (digits 0 and 1) on four qubits with two
re-uploading layers (16 trainable parameters); one additional scenario uses four classes
and three layers.  The clients differ only in the noise scale of their device and in their
shots per circuit; the data are IID and uncorrupted.
"""
from __future__ import annotations

import json
import os

from .experiment import Scenario
from .federated import RunConfig

_BASE = dict(classes=(0, 1), n_qubits=4, layers=2, beta=4.0, angle_scale=0.5, n_per_client=40,
             n_train_per_class=400, n_test_per_class=100)

SWEEP_SCALES = (1, 50, 150, 300, 500)

# emulator experiment (see below): multiplier on the emulator, matched multiplier of the local model,
# and shots per circuit of the reference clients
EMU_SCALE, EMU_LOCAL_SCALE, EMU_SHOTS_REFERENCE = 300.0, 370.0, 30


def _noise_only(scale: float) -> Scenario:
    """Two reference clients and eighteen clients whose error rates are ``scale`` times larger; equal shots."""
    return Scenario(name=f"noise_only_x{scale:g}", **_BASE,
                    gate_scales=(1.0,) * 2 + (float(scale),) * 18, shots=(8,) * 20)


SCENARIOS = {
    # two reference clients with 50 shots, twelve clients with 200-fold error rates and 2 shots
    "shots_and_noise": Scenario(name="shots_and_noise", **_BASE,
                                gate_scales=(1.0,) * 2 + (200.0,) * 12, shots=(50,) * 2 + (2,) * 12),
    # equal shot budgets, heterogeneity in the device noise only (x1 is the homogeneous control)
    **{f"noise_only_x{s}": _noise_only(s) for s in SWEEP_SCALES},
    # the first scenario with four classes (digits 0 to 3) and three layers (24 trainable parameters)
    "shots_and_noise_4c": Scenario(name="shots_and_noise_4c", **{**_BASE, "classes": (0, 1, 2, 3), "layers": 3},
                                   gate_scales=(1.0,) * 2 + (200.0,) * 12, shots=(50,) * 2 + (2,) * 12),
}

RUN = RunConfig(rounds=40, batch=8, local_steps=1, echo_shots=20, eval_shots=100, eval_every=1)

# --------------------------------------------------------------------------- emulator experiment
# Training on the emulator reimei-E (scripts/run_emulator.py).  On the emulator a job waits several
# minutes in the queue and every distinct circuit costs about 0.1 to 0.2 seconds whatever its shot
# count, so the experiment is designed around the number of distinct circuits: four clients, two
# images per client and round, and one layer (8 trainable parameters, 17 circuits per
# parameter-shift gradient).  The task has four classes (digits 0 to 3, one readout qubit per class).
# Two clients are reference clients with 30 shots per circuit, and two clients have 300-fold error
# rates and one shot per circuit.  ``gate_scales`` are the error-rate multipliers passed to the
# emulator; ``local_scales`` are the multipliers of the local gate-level model that reproduce the
# echo survival measured on the emulator (results/nexus-geometry), and they are used only for the
# step-size selection and for the prediction of the local model.
_EMU = dict(classes=(0, 1, 2, 3), n_qubits=4, layers=1, beta=4.0, angle_scale=0.5, n_per_client=40,
            n_train_per_class=400, n_test_per_class=100,
            gate_scales=(1.0,) * 2 + (EMU_SCALE,) * 2, local_scales=(1.0,) * 2 + (EMU_LOCAL_SCALE,) * 2,
            shots=(EMU_SHOTS_REFERENCE,) * 2 + (1,) * 2)
SCENARIOS["emulator_iid"] = Scenario(name="emulator_iid", **_EMU)
# the same devices with a Dirichlet label partition (concentration 0.5)
SCENARIOS["emulator_dirichlet"] = Scenario(name="emulator_dirichlet", alpha=0.5, **_EMU)

EMULATOR_SCENARIOS = ("emulator_iid", "emulator_dirichlet")
# The emulator experiment was executed in two campaigns with the same protocol: the first six seeds,
# then the remaining seeds up to the number of seeds of the simulations.  The local model follows the
# same grouping (the runs of a campaign advance in lockstep).
EMULATOR_CAMPAIGNS = {"emulator-reimei-E": (0, 1, 2, 3, 4, 5),
                      "emulator-reimei-E-2": (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)}
EMULATOR_LOCAL = {"emulator-local": (0, 1, 2, 3, 4, 5),
                  "emulator-local-2": (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)}
EMULATOR_SEEDS = (0, 1, 2, 3, 4, 5)        # default seeds of scripts/run_emulator.py (the first campaign)
EMULATOR_RUN = RunConfig(rounds=15, batch=2, local_steps=1, echo_shots=5, eval_shots=20, eval_every=1)
EMULATOR_TAIL = 5              # the reported accuracy is the mean over the last rounds
EMULATOR_SEGMENTS = 528        # circuits packed into one program (a program of 2112 circuits was not accepted in time)
EMULATOR_PROGRAMS_PER_JOB = 3  # the emulator runs two jobs at a time, so the programs of a round are spread over several jobs
EMULATOR_EVAL_INPUTS = 60      # test images of the evaluation on the emulator (the same number from every class)



# --------------------------------------------------------------------------- two-class experiment on the emulator
# The first experiment on the emulator has four classes and a one-layer circuit, whose noiseless accuracy on
# that task is about 0.72, so that all accuracies are low.  The second experiment uses the two-class task of the
# simulations, which the one-layer circuit solves (0.99 without noise).  Its design was chosen in the local
# gate-level model before any run on the emulator (scripts/emulator_design.py, results/emulator-design/):
# two reference clients with 5 shots per circuit, six clients with 300-fold error rates and one shot,
# eight images per client and round, 15 rounds, IID data, 16 seeds.  The final models are evaluated on the
# emulator with 100 shots per image, each of the last five rounds on its own block of 20 test images.
EMULATOR2_SHOTS_REFERENCE, EMULATOR2_NOISY_CLIENTS, EMULATOR2_BATCH = 5, 6, 8
_EMU2 = dict(classes=(0, 1), n_qubits=4, layers=1, beta=4.0, angle_scale=0.5, n_per_client=40,
             n_train_per_class=400, n_test_per_class=100,
             gate_scales=(1.0,) * 2 + (EMU_SCALE,) * EMULATOR2_NOISY_CLIENTS,
             local_scales=(1.0,) * 2 + (EMU_LOCAL_SCALE,) * EMULATOR2_NOISY_CLIENTS,
             shots=(EMULATOR2_SHOTS_REFERENCE,) * 2 + (1,) * EMULATOR2_NOISY_CLIENTS)
SCENARIOS["emulator2_iid"] = Scenario(name="emulator2_iid", **_EMU2)
EMULATOR2_SCENARIOS = ("emulator2_iid",)
# The same task on fewer qubits: circuits of candidate designs, used for echo measurements on the emulator
# (scripts/nexus_geometry.py --scenario calibration_n2) and for comparisons in the local model.
SCENARIOS["calibration_n2"] = Scenario(name="calibration_n2", **{**_EMU2, "n_qubits": 2})
SCENARIOS["calibration_n3"] = Scenario(name="calibration_n3", **{**_EMU2, "n_qubits": 3})
CALIBRATION_SCENARIOS = ("calibration_n2", "calibration_n3")
EMULATOR2_RUN = RunConfig(rounds=15, batch=EMULATOR2_BATCH, local_steps=1, echo_shots=5, eval_shots=100, eval_every=1)
EMULATOR2_SEEDS = tuple(range(16))
# candidates of the design study (scripts/emulator_design.py):
# (reference clients, noisy clients, minibatch, shots of the reference clients, Dirichlet parameter or None)
EMULATOR2_DESIGN_CANDIDATES = (
    (2, 2, 8, 5, None), (2, 4, 8, 5, None), (2, 6, 8, 5, None), (2, 10, 8, 5, None),
    (2, 6, 8, 10, None), (2, 6, 4, 10, None), (2, 6, 4, 5, None),
    (2, 6, 8, 5, 0.5), (2, 6, 8, 5, 1.0), (2, 6, 8, 5, 2.0), (2, 6, 8, 5, 5.0),
)

# Every experiment on the emulator: its scenarios, run configuration, number of final rounds of the reported
# accuracy, and number of test images of the evaluation on the emulator.
# ``eval_inputs`` is the number of test images of one evaluated round; with ``eval_rotate`` every evaluated round
# uses its own block of test images.  ``segments`` and ``programs_per_job`` are the default packing of the circuits.
EMULATOR_EXPERIMENTS = {
    "four_class": dict(scenarios=EMULATOR_SCENARIOS, run=EMULATOR_RUN, tail=EMULATOR_TAIL, eval_inputs=EMULATOR_EVAL_INPUTS,
                       eval_rotate=False, segments=EMULATOR_SEGMENTS, programs_per_job=EMULATOR_PROGRAMS_PER_JOB),
    "two_class": dict(scenarios=EMULATOR2_SCENARIOS, run=EMULATOR2_RUN, tail=5, eval_inputs=20,
                      eval_rotate=True, segments=1224, programs_per_job=1),
}
# experiment of every campaign folder (results/<name>), on the emulator and in the local model
EMULATOR2_CAMPAIGNS = {"emulator2-reimei-E": EMULATOR2_SEEDS}       # on the emulator
EMULATOR2_LOCAL = {"emulator2-local": EMULATOR2_SEEDS}              # the same runs in the local model
CAMPAIGN_EXPERIMENT = {**{name: "four_class" for name in EMULATOR_CAMPAIGNS}, **{name: "four_class" for name in EMULATOR_LOCAL},
                       **{name: "two_class" for name in EMULATOR2_CAMPAIGNS}, **{name: "two_class" for name in EMULATOR2_LOCAL}}


def _experiment_of(scenario: str) -> dict | None:
    return next((e for e in EMULATOR_EXPERIMENTS.values() if scenario in e["scenarios"]), None)


def experiment_of(scenario: str) -> dict:
    """Experiment on the emulator that ``scenario`` belongs to (the four-class experiment for other scenarios)."""
    return _experiment_of(scenario) or EMULATOR_EXPERIMENTS["four_class"]


def emulator_scenarios() -> tuple:
    """Names of all scenarios that belong to an experiment on the emulator."""
    return tuple(sc for e in EMULATOR_EXPERIMENTS.values() for sc in e["scenarios"])


def campaign_experiment(name: str) -> dict:
    """Experiment of the campaign folder ``name`` (the four-class experiment for unknown names)."""
    return EMULATOR_EXPERIMENTS[CAMPAIGN_EXPERIMENT.get(name, "four_class")]


def campaign_scenarios(name: str) -> tuple:
    return tuple(campaign_experiment(name)["scenarios"])


def campaign_seeds(name: str) -> tuple:
    """Seeds of the campaign folder ``name`` (those of the first campaign for unknown names)."""
    for table in (EMULATOR_CAMPAIGNS, EMULATOR_LOCAL, EMULATOR2_CAMPAIGNS, EMULATOR2_LOCAL):
        if name in table:
            return tuple(table[name])
    return tuple(EMULATOR_SEEDS)


def run_config(scenario: str) -> RunConfig:
    e = _experiment_of(scenario)
    return e["run"] if e else RUN


def tail(scenario: str) -> int:
    """Number of final rounds over which the reported accuracy is averaged."""
    e = _experiment_of(scenario)
    return e["tail"] if e else 10


def eval_inputs(scenario: str) -> int:
    """Number of test images of the evaluation on the emulator."""
    e = _experiment_of(scenario)
    return e["eval_inputs"] if e else EMULATOR_EVAL_INPUTS


def simulation_scenarios() -> tuple:
    """Scenarios of the simulations with the gate-level model (all scenarios outside the emulator experiments)."""
    other = emulator_scenarios() + CALIBRATION_SCENARIOS
    return tuple(k for k in SCENARIOS if k not in other)

# step-size grids of scripts/run_campaign.py (stage "tune")
GRID_EUCLID = (0.15, 0.3, 0.6, 1.0, 2.0, 4.0)        # plain-gradient methods
GRID_NATURAL = (0.025, 0.05, 0.1, 0.2, 0.3, 0.5)    # natural-gradient methods
NATURAL = ("QFedQGT", "QFedQGT-invvar", "QNG+data", "Bures+data")
TUNING_SEEDS = tuple(range(100, 108))          # disjoint from the reported seeds
REPORT_SEEDS = tuple(range(16))

# fallback values, used only when results/tuning/learning_rates.json does not exist
_DEFAULT_LR = {"QFedAvg": 0.6, "QFedFisher": 0.3, "QFedAvg-shots": 0.6, "QFedQGT": 0.1,
               "QFedQGT-invvar": 0.1, "QNG+data": 0.1, "Bures+data": 0.1, "Euclid+re": 0.6}
_LR_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "tuning", "learning_rates.json")


def grid(method: str) -> tuple:
    return GRID_NATURAL if method in NATURAL else GRID_EUCLID


def learning_rates(scenario: str) -> dict:
    """Step sizes selected on the tuning seeds for ``scenario`` (see scripts/run_campaign.py)."""
    lrs = dict(_DEFAULT_LR)
    if os.path.exists(_LR_FILE):
        with open(_LR_FILE) as f:
            lrs.update(json.load(f).get(scenario, {}))
    return lrs
