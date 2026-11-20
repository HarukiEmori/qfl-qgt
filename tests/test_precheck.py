"""Tests of a pre-check run before a campaign (scripts/emulator_precheck.py), on synthetic result folders."""
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("emulator_precheck", os.path.join(ROOT, "scripts", "emulator_precheck.py"))
precheck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precheck)


def _folder(tmp_path, noisy=0.7, reference=0.01, failed=0, open_jobs=0, nan=False, echo=0.708):
    """A results folder of a pre-check run with two runs, two reference clients and two noisy clients."""
    out = tmp_path / "results" / "pre"
    (out / "cache").mkdir(parents=True)
    scales = [1.0, 1.0, 400.0, 400.0]
    runs = []
    for seed in (900, 901):
        runs.append(dict(scenario="emulator2_iid", seed=seed, scenario_definition=dict(n_qubits=2, gate_scales=scales),
                         run=dict(p_max=0.9)))
        arrays = {}
        for m in ("QFedAvg", "QFedQGT"):
            arrays.update({f"{m}/acc_exact": np.array([0.5, np.nan if nan else 0.6]), f"{m}/loss_exact": np.array([0.7, 0.6]),
                           f"{m}/theta": np.zeros((2, 4)), f"{m}/train_loss": np.array([0.7]), f"{m}/weights": np.full((1, 4), 0.25),
                           # undefined by design: heterogeneity of the first round, accuracy on the device per round
                           f"{m}/zeta2": np.array([np.nan]), f"{m}/acc_shots": np.array([np.nan, np.nan])})
        arrays["QFedAvg/p_hat"] = np.full((1, 4), np.nan)              # a method that does not estimate the noise strength
        arrays.update({"QFedQGT/p_hat": np.array([[reference, reference, noisy, noisy]]),
                       "QFedQGT/kappa_hat": np.ones((1, 4)), "QFedQGT/sigma2": np.ones((1, 4))})
        np.savez(out / f"emulator2_iid-seed{seed}.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(dict(device="reimei-E", runs=runs)))
    events = [dict(event="submitted", job_id=str(i)) for i in range(3 + open_jobs)]
    events += [dict(event="completed", job_id=str(i), cost=10.0) for i in range(3)]
    events += [dict(event="failed", job_id="x")] * failed
    (out / "cache" / "jobs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    geo = tmp_path / "results" / "nexus-geometry"
    geo.mkdir()
    (geo / "reimei-E-calibration_n2.json").write_text(json.dumps(dict(rows=[dict(scale=1.0, p_hat=0.002),
                                                                             dict(scale=400.0, p_hat=echo)])))
    return str(tmp_path)


def test_a_complete_run_with_the_expected_noise_passes(tmp_path):
    r = precheck.check("pre", root=_folder(tmp_path))
    assert r["passed"] and r["jobs"] == dict(submitted=3, completed=3, failed=0, cost_hqc=30.0)
    assert r["expected_noise_strength"]["value"] == 0.708 and r["expected_noise_strength"]["multiplier"] == 400.0


def test_an_unfinished_or_failed_run_does_not_pass(tmp_path):
    assert not precheck.check("pre", root=str(tmp_path))["passed"]                        # no results at all
    assert not precheck.check("pre", root=_folder(tmp_path / "a", open_jobs=1))["passed"]  # a job is not completed
    assert not precheck.check("pre", root=_folder(tmp_path / "b", failed=1))["passed"]     # a job has failed
    assert not precheck.check("pre", root=_folder(tmp_path / "c", nan=True))["passed"]     # a stored array is not finite


def test_wrong_noise_strengths_do_not_pass(tmp_path):
    # noisy clients that look like reference clients: the multiplier did not reach the device
    assert not precheck.check("pre", root=_folder(tmp_path / "a", noisy=0.02))["passed"]
    # reference clients that look noisy
    assert not precheck.check("pre", root=_folder(tmp_path / "b", reference=0.5))["passed"]
    # noisy clients far from the echo measurement of the circuit
    assert not precheck.check("pre", root=_folder(tmp_path / "c", noisy=0.45, echo=0.708))["passed"]
    # the estimate is compared with the capped echo value
    assert precheck.check("pre", root=_folder(tmp_path / "d", noisy=0.9, echo=1.0))["passed"]


def _stub_repository(tmp_path, precheck_status=0, run_status=0):
    """A folder with the driver script and stand-ins for the scripts that it calls; the stand-ins record their calls."""
    import shutil
    import stat

    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, "scripts", "emulator_campaign_after_precheck.sh"), scripts)
    (scripts / "emulator_campaign.sh").write_text(
        "#!/bin/bash\necho \"campaign $*\" >> \"$(dirname \"$0\")/../calls.txt\"\n"
        f"if [ \"$1\" = pre ]; then exit {run_status}; fi\nexit 0\n")
    (scripts / "emulator_precheck.py").write_text(
        "import sys\nopen(sys.argv[0].replace('scripts/emulator_precheck.py', 'calls.txt'), 'a').write('precheck ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        f"sys.exit({precheck_status})\n")
    for f in scripts.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def _drive(folder):
    import subprocess
    import sys

    env = dict(os.environ, PYTHON=sys.executable)
    r = subprocess.run(["bash", str(folder / "scripts" / "emulator_campaign_after_precheck.sh"), "pre", "main",
                        "--seeds", "900", "--rounds", "1"], env=env, capture_output=True, text=True, timeout=60)
    calls = (folder / "calls.txt").read_text().splitlines() if (folder / "calls.txt").exists() else []
    return r.returncode, calls


def test_campaign_starts_only_after_a_passed_pre_check(tmp_path):
    # passed: pre-check run with its options, then the tests, then the campaign
    folder = _stub_repository(tmp_path / "ok")
    status, calls = _drive(folder)
    assert status == 0 and calls == ["campaign pre --seeds 900 --rounds 1", "precheck --name pre", "campaign main"]
    assert not (folder / "results" / "pre" / "precheck.failed").exists()
    # failed tests: the campaign is not started
    folder = _stub_repository(tmp_path / "failed", precheck_status=1)
    status, calls = _drive(folder)
    assert status == 1 and calls == ["campaign pre --seeds 900 --rounds 1", "precheck --name pre"]
    assert (folder / "results" / "pre" / "precheck.failed").exists()
    # the pre-check run itself did not finish: neither tests nor campaign
    folder = _stub_repository(tmp_path / "unfinished", run_status=1)
    status, calls = _drive(folder)
    assert status == 1 and calls == ["campaign pre --seeds 900 --rounds 1"]
    assert (folder / "results" / "pre" / "precheck.failed").exists()


def test_multiplier_check_changes_only_the_local_multiplier_of_the_noisy_clients():
    spec3 = importlib.util.spec_from_file_location("emulator_multiplier_check", os.path.join(ROOT, "scripts", "emulator_multiplier_check.py"))
    check = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(check)
    from qfedqgt import scenarios as S

    ref = S.SCENARIOS["emulator2_iid"]
    sc = check.scenario_with(430.0)
    assert sc.local_scales == (1.0, 1.0) + (430.0,) * 6 and sc.gate_scales == ref.gate_scales and sc.shots == ref.shots
    path = os.path.join(ROOT, "results", "emulator-design", "multiplier-sensitivity.json")
    if os.path.exists(path):          # the stored result: the step sizes of the experiment are selected at every multiplier
        with open(path) as f:
            stored = json.load(f)
        assert all(row["same_as_experiment"] for row in stored["multipliers"])
        assert stored["step_sizes_of_the_experiment"] == {m: S.learning_rates("emulator2_iid")[m] for m in check.METHODS}
