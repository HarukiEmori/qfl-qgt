"""Definitions of the two-class experiment on the emulator and of the helper tables of the campaigns."""
import importlib.util
import os

from qfedqgt import scenarios as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_two_class_scenario_and_experiment_table():
    sc = S.SCENARIOS["emulator2_iid"]
    scale, local = S.EMULATOR2_NOISE[S.EMULATOR2_QUBITS]
    assert sc.classes == (0, 1) and sc.layers == 1 and sc.alpha is None and sc.n_qubits == S.EMULATOR2_QUBITS
    assert sc.gate_scales == (1.0, 1.0) + (scale,) * S.EMULATOR2_NOISY_CLIENTS
    assert sc.local_scales == (1.0, 1.0) + (local,) * S.EMULATOR2_NOISY_CLIENTS
    assert sc.shots == (S.EMULATOR2_SHOTS_REFERENCE,) * 2 + (1,) * S.EMULATOR2_NOISY_CLIENTS
    e = S.experiment_of("emulator2_iid")
    assert e is S.EMULATOR_EXPERIMENTS["two_class"] and e["eval_rotate"] and e["programs_per_job"] == 1
    assert S.run_config("emulator2_iid") is S.EMULATOR2_RUN and S.tail("emulator2_iid") == 5
    assert S.eval_inputs("emulator2_iid") == 20 and S.EMULATOR2_RUN.batch == S.EMULATOR2_BATCH
    assert S.EMULATOR2_RUN.rounds == S.EMULATOR2_ROUNDS
    # the four-class experiment keeps its settings
    assert S.run_config("emulator_iid") is S.EMULATOR_RUN and S.tail("emulator_dirichlet") == S.EMULATOR_TAIL
    assert S.eval_inputs("emulator_iid") == S.EMULATOR_EVAL_INPUTS and not S.experiment_of("emulator_iid")["eval_rotate"]
    assert S.SCENARIOS["emulator_iid"].n_qubits == 4 and S.EMULATOR2_NOISE[4] == (S.EMU_SCALE, S.EMU_LOCAL_SCALE)
    # scenarios of the simulations use the configuration of the simulations
    assert S.run_config("shots_and_noise") is S.RUN and S.tail("shots_and_noise") == 10


def test_tuning_seeds_are_disjoint_from_the_other_seeds():
    assert S.tuning_seeds("shots_and_noise") == S.TUNING_SEEDS == S.tuning_seeds("emulator_iid")
    assert S.tuning_seeds("emulator2_iid") == S.EMULATOR2_TUNING_SEEDS and len(S.EMULATOR2_TUNING_SEEDS) == 32
    assert not set(S.EMULATOR2_TUNING_SEEDS) & set(S.EMULATOR2_SEEDS) and not set(S.TUNING_SEEDS) & set(S.REPORT_SEEDS)


def test_calibration_scenarios_keep_the_circuits_of_the_echo_measurements():
    for name, n in (("calibration_n2", 2), ("calibration_n3", 3)):
        sc = S.SCENARIOS[name]
        assert sc.n_qubits == n and sc.layers == 1 and sc.classes == (0, 1) and sc.angle_scale == 0.5
        assert name not in S.simulation_scenarios()


def test_campaign_tables():
    assert S.campaign_scenarios("emulator2-reimei-E") == ("emulator2_iid",)
    assert S.campaign_seeds("emulator2-reimei-E") == tuple(range(16)) == S.campaign_seeds("emulator2-local")
    assert S.campaign_scenarios("emulator-reimei-E-2") == S.EMULATOR_SCENARIOS
    assert S.campaign_seeds("emulator-reimei-E-2") == tuple(range(6, 16))
    assert S.campaign_seeds("emulator-reimei-E") == S.EMULATOR_SEEDS == S.campaign_seeds("some-other-name")
    assert "emulator2_iid" not in S.simulation_scenarios() and "emulator_iid" not in S.simulation_scenarios()
    assert "shots_and_noise" in S.simulation_scenarios()


def test_design_study_contains_the_chosen_design_and_its_cost():
    spec = importlib.util.spec_from_file_location("emulator_design", os.path.join(ROOT, "scripts", "emulator_design.py"))
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    chosen = design.name_of(S.EMULATOR2_QUBITS, S.EMULATOR2_NOISY_CLIENTS, S.EMULATOR2_BATCH, S.EMULATOR2_SHOTS_REFERENCE)
    assert chosen in design.CANDIDATES
    sc, batch = design.CANDIDATES[chosen]
    ref = S.SCENARIOS["emulator2_iid"]
    assert batch == S.EMULATOR2_BATCH and sc.shots == ref.shots
    assert sc.gate_scales == ref.gate_scales and sc.local_scales == ref.local_scales and sc.n_qubits == ref.n_qubits
    assert design.TUNING_SEEDS == S.EMULATOR2_TUNING_SEEDS and S.EMULATOR2_ROUNDS in design.CHECKPOINTS
    assert not set(design.DESIGN_SEEDS) & (set(design.TUNING_SEEDS) | set(S.EMULATOR2_SEEDS))
    rounds, clients = S.EMULATOR2_RUN.rounds, ref.n_clients
    cost = design.cost(chosen, rounds)
    # three methods, 9 circuits per image of the two-qubit circuit (unshifted and two shifts for each of 4 parameters)
    assert cost["circuits"] == 3 * clients * batch * 9 * rounds
    assert cost["circuit_shots"] == 3 * batch * 9 * sum(ref.shots) * rounds
    # cost formula per circuit and shot: (6 one-qubit gates + 10 x 1 two-qubit gate + 5 x 2 x 2 qubits) / 5000
    assert abs(cost["hqc_per_image_shot"] - 9 * 36 / 5000) < 1e-12
    assert abs(cost["hqc_evaluation"] - 3 * 100 * 100 * 36 / 5000) < 1e-9
    # every candidate has a matched multiplier for its number of qubits
    for name, (c, _) in design.CANDIDATES.items():
        assert (max(c.gate_scales), max(c.local_scales)) == S.EMULATOR2_NOISE[c.n_qubits], name


def test_prediction_probability_matches_the_enumeration_of_all_counts():
    import itertools
    import math

    import numpy as np

    from qfedqgt import sim

    spec = importlib.util.spec_from_file_location("emulator_model_comparison", os.path.join(ROOT, "scripts", "emulator_model_comparison.py"))
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    obs = sim.z_observables(2, pairs=False)[:, :2]            # the readout of the two-class model on two qubits
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(4), size=5)
    p[0] = [0.0, 1.0, 0.0, 0.0]                               # a deterministic outcome
    p[1] = [0.5, 0.0, 0.0, 0.5]                               # only outcomes on which the two observables agree: always a tie
    for shots in (1, 4, 7):
        exact = np.zeros(len(p))
        for counts in itertools.product(range(shots + 1), repeat=4):
            if sum(counts) != shots:
                continue
            coefficient = math.factorial(shots) / math.prod(math.factorial(c) for c in counts)
            phi = np.array(counts) @ obs / shots
            if int(np.argmax(phi)) == 0:                      # the prediction of the model, with class 0 for a tie
                exact += coefficient * np.prod(p ** np.array(counts), axis=1)
        assert np.allclose(comparison.prediction_probability(p, obs, shots), exact, atol=1e-12)
    assert comparison.prediction_probability(p[1:2], obs, 50)[0] == 1.0
