"""Definitions of the two-class experiment on the emulator and of the helper tables of the campaigns."""
import importlib.util
import os

from qfedqgt import scenarios as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_two_class_scenario_and_experiment_table():
    sc = S.SCENARIOS["emulator2_iid"]
    assert sc.classes == (0, 1) and sc.layers == 1 and sc.alpha is None
    assert sc.gate_scales == (1.0, 1.0) + (S.EMU_SCALE,) * S.EMULATOR2_NOISY_CLIENTS
    assert sc.local_scales == (1.0, 1.0) + (S.EMU_LOCAL_SCALE,) * S.EMULATOR2_NOISY_CLIENTS
    assert sc.shots == (S.EMULATOR2_SHOTS_REFERENCE,) * 2 + (1,) * S.EMULATOR2_NOISY_CLIENTS
    e = S.experiment_of("emulator2_iid")
    assert e is S.EMULATOR_EXPERIMENTS["two_class"] and e["eval_rotate"] and e["programs_per_job"] == 1
    assert S.run_config("emulator2_iid") is S.EMULATOR2_RUN and S.tail("emulator2_iid") == 5
    assert S.eval_inputs("emulator2_iid") == 20 and S.EMULATOR2_RUN.batch == S.EMULATOR2_BATCH
    # the four-class experiment keeps its settings
    assert S.run_config("emulator_iid") is S.EMULATOR_RUN and S.tail("emulator_dirichlet") == S.EMULATOR_TAIL
    assert S.eval_inputs("emulator_iid") == S.EMULATOR_EVAL_INPUTS and not S.experiment_of("emulator_iid")["eval_rotate"]
    # scenarios of the simulations use the configuration of the simulations
    assert S.run_config("shots_and_noise") is S.RUN and S.tail("shots_and_noise") == 10


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
    chosen = f"2+{S.EMULATOR2_NOISY_CLIENTS}_B{S.EMULATOR2_BATCH}_M{S.EMULATOR2_SHOTS_REFERENCE}_iid"
    assert chosen in design.CANDIDATES
    sc, batch = design.CANDIDATES[chosen]
    assert batch == S.EMULATOR2_BATCH and sc.shots == S.SCENARIOS["emulator2_iid"].shots
    cost = design.cost(chosen, S.EMULATOR2_RUN.rounds)
    # three methods, eight clients, eight images, 17 circuits, 15 rounds
    assert cost["circuits"] == 3 * 8 * 8 * 17 * S.EMULATOR2_RUN.rounds
    assert cost["circuit_shots"] == 3 * 8 * 17 * (2 * 5 + 6 * 1) * S.EMULATOR2_RUN.rounds
    assert 0.016 < cost["hqc"] / cost["circuit_shots"] < 0.02
