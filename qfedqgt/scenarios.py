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
EMU_SCALE, EMU_LOCAL_SCALE, EMU_SHOTS_REFERENCE = 300.0, 370.0, 10


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
# Training on the emulator reimei-E (scripts/run_emulator.py).  The emulator executes a few thousand
# gate-shots per second, so the circuits have one layer (8 trainable parameters, 17 circuits per
# parameter-shift gradient) and the shot budgets are small.  The task has four classes (digits 0 to 3,
# one readout qubit per class).  ``gate_scales`` are the error-rate multipliers passed to the
# emulator; ``local_scales`` are the multipliers of the local gate-level model that reproduce the
# echo survival measured on the emulator (results/nexus-geometry), and they are used only for the
# step-size selection and for the prediction of the local model.
_EMU = dict(classes=(0, 1, 2, 3), n_qubits=4, layers=1, beta=4.0, angle_scale=0.5, n_per_client=40,
            n_train_per_class=400, n_test_per_class=100,
            gate_scales=(1.0,) * 2 + (EMU_SCALE,) * 12, local_scales=(1.0,) * 2 + (EMU_LOCAL_SCALE,) * 12,
            shots=(EMU_SHOTS_REFERENCE,) * 2 + (1,) * 12)
SCENARIOS["emulator_iid"] = Scenario(name="emulator_iid", **_EMU)
# the same devices with a Dirichlet label partition (concentration 0.5)
SCENARIOS["emulator_dirichlet"] = Scenario(name="emulator_dirichlet", alpha=0.5, **_EMU)

EMULATOR_SCENARIOS = ("emulator_iid", "emulator_dirichlet")
SIMULATION_SCENARIOS = tuple(k for k in SCENARIOS if k not in EMULATOR_SCENARIOS)
EMULATOR_SEEDS = (0, 1, 2)
EMULATOR_RUN = RunConfig(rounds=30, batch=8, local_steps=1, echo_shots=5, eval_shots=10, eval_every=1)
EMULATOR_TAIL = 5              # the reported accuracy is the mean over the last rounds
EMULATOR_SEGMENTS = 132        # circuits packed into one program


def run_config(scenario: str) -> RunConfig:
    return EMULATOR_RUN if scenario in EMULATOR_SCENARIOS else RUN


def tail(scenario: str) -> int:
    """Number of final rounds over which the reported accuracy is averaged."""
    return EMULATOR_TAIL if scenario in EMULATOR_SCENARIOS else 10

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
