#!/usr/bin/env python
"""Tests of a short pre-check run on the emulator, made before a campaign is started.

A pre-check run executes one round of a few runs whose seeds are not part of the experiment
(``scripts/run_emulator.py --rounds 1 --no-eval --seeds ...``).  It uses the circuits, the programs
and the devices of the campaign.  This script reads its results and tests that

1. every submitted job is completed and none has failed,
2. the stored results of every run are finite (accuracy and loss of the noiseless circuit, parameters,
   training loss and weights of every method, and the estimated noise strength, metric attenuation and
   variance of QFedQGT),
3. the noise strength that QFedQGT estimates from the echo circuits is small for the reference
   clients and, for the noisy clients, close to the echo measurement of the same circuit
   (``results/nexus-geometry/<device>-calibration_n<qubits>.json`` or ``...-<scenario>.json``).

    python scripts/emulator_precheck.py --name emulator2-smoke

The report is printed and written to ``results/<name>/precheck.json``.  The exit status is 0 if all
tests pass and 1 otherwise; ``scripts/emulator_campaign_after_precheck.sh`` starts a campaign only
after the status 0.  The script submits nothing and does not connect to Nexus.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Arrays that must be finite.  Other stored arrays are undefined by design in some rounds or for some methods:
# the heterogeneity estimate of the first round, the per-round accuracy on the device (not evaluated on the
# emulator), and the noise strength of the methods that do not estimate it.
FINITE = ("acc_exact", "loss_exact", "theta", "train_loss", "weights")
FINITE_QFEDQGT = ("p_hat", "kappa_hat", "sigma2")
REFERENCE_MAX = 0.1        # largest accepted mean noise strength of the reference clients
NOISY_TOLERANCE = 0.2      # accepted distance of the mean noise strength of the noisy clients from the echo measurement
NOISY_RANGE = (0.4, 0.95)  # accepted range when no echo measurement of the circuit is stored


def job_events(path: str) -> dict:
    """Number of submitted, completed and failed jobs in a job log, and the recorded cost."""
    out = dict(submitted=0, completed=0, failed=0, cost_hqc=0.0)
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = rec.get("event")
            if event == "submitted":
                out["submitted"] += 1
            elif event == "completed":
                out["completed"] += 1
                if isinstance(rec.get("cost"), (int, float)) and np.isfinite(rec["cost"]):
                    out["cost_hqc"] += float(rec["cost"])
            elif event in ("failed", "reattach-failed"):
                out["failed"] += 1
    return out


def expected_noise(device: str, scenario: str, qubits: int, scale: float, p_max: float, root: str = ROOT):
    """Noise strength of the echo measurement of the circuit at the multiplier ``scale`` (None if not stored)."""
    folder = os.path.join(root, "results", "nexus-geometry")
    for stem in (f"{device}-calibration_n{qubits}", f"{device}-{scenario}"):
        path = os.path.join(folder, stem + ".json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = json.load(f)["rows"]
        for r in rows:
            if float(r["scale"]) == float(scale):
                return min(float(r["p_hat"]), p_max), os.path.relpath(path, root)
    return None, None


def check(name: str, root: str = ROOT) -> dict:
    """Tests of the pre-check run ``results/<name>`` under ``root``; ``passed`` is True if all tests pass."""
    out_dir = os.path.join(root, "results", name)
    tests, report = [], dict(name=name)

    def test(text: str, ok: bool, detail: str = "") -> None:
        tests.append(dict(test=text, passed=bool(ok), detail=detail))

    summary_path = os.path.join(out_dir, "summary.json")
    test("the run has finished (summary.json exists)", os.path.exists(summary_path))
    if not os.path.exists(summary_path):
        return dict(report, tests=tests, passed=False)
    with open(summary_path) as f:
        summary = json.load(f)
    jobs = job_events(os.path.join(out_dir, "cache", "jobs.jsonl"))
    report["jobs"] = jobs
    test("every submitted job is completed", jobs["submitted"] > 0 and jobs["completed"] == jobs["submitted"],
         f"{jobs['completed']} of {jobs['submitted']}")
    test("no job has failed", jobs["failed"] == 0, f"{jobs['failed']} failed")

    ref, noisy, finite, runs, not_finite = [], [], True, 0, []
    scale = qubits = scenario = p_max = None
    for row in summary["runs"]:
        sc, cfg = row["scenario_definition"], row["run"]
        scenario, qubits, p_max = row["scenario"], int(sc["n_qubits"]), float(cfg["p_max"])
        scales = np.asarray(sc["gate_scales"], dtype=float)
        scale = float(scales.max())
        files = glob.glob(os.path.join(out_dir, f"{row['scenario']}-seed{row['seed']}.npz"))
        if len(files) != 1:
            finite = False
            continue
        data = np.load(files[0])
        runs += 1
        methods = sorted({key.split("/")[0] for key in data.files})
        required = [f"{m}/{k}" for m in methods for k in FINITE] + [f"QFedQGT/{k}" for k in FINITE_QFEDQGT]
        for key in required:
            if key not in data.files or not np.all(np.isfinite(np.asarray(data[key], dtype=float))):
                finite = False
                not_finite.append(f"{row['scenario']}-seed{row['seed']}: {key}")
        p_hat = np.asarray(data["QFedQGT/p_hat"], dtype=float)[-1]        # estimate of every client after the last round
        ref += list(p_hat[scales == 1.0])
        noisy += list(p_hat[scales != 1.0])
    test("every run has its result file", runs == len(summary["runs"]) and runs > 0, f"{runs} of {len(summary['runs'])}")
    test("the accuracies, losses, parameters, weights and noise estimates are finite", finite, ", ".join(not_finite[:4]))
    report["noise_strength"] = dict(reference=[float(x) for x in ref], noisy=[float(x) for x in noisy])
    if ref and noisy:
        mean_ref, mean_noisy = float(np.mean(ref)), float(np.mean(noisy))
        test(f"mean noise strength of the reference clients is at most {REFERENCE_MAX:g}", mean_ref <= REFERENCE_MAX, f"{mean_ref:.3f}")
        expected, source = expected_noise(summary.get("device", ""), scenario, qubits, scale, p_max, root)
        report["expected_noise_strength"] = dict(value=expected, source=source, multiplier=scale)
        if expected is None:
            test(f"mean noise strength of the noisy clients is between {NOISY_RANGE[0]:g} and {NOISY_RANGE[1]:g}",
                 NOISY_RANGE[0] <= mean_noisy <= NOISY_RANGE[1], f"{mean_noisy:.3f} (no echo measurement of this circuit is stored)")
        else:
            test(f"mean noise strength of the noisy clients is within {NOISY_TOLERANCE:g} of the echo measurement",
                 abs(mean_noisy - expected) <= NOISY_TOLERANCE, f"{mean_noisy:.3f}, echo measurement {expected:.3f} ({source})")
        test("the noisy clients are noisier than the reference clients", mean_noisy > mean_ref + 0.2, f"{mean_noisy:.3f} against {mean_ref:.3f}")
    else:
        test("noise strengths of reference and noisy clients are available", False)
    return dict(report, tests=tests, passed=all(t["passed"] for t in tests))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="results sub-directory of the pre-check run")
    a = ap.parse_args(argv)
    result = check(a.name)
    for t in result["tests"]:
        print(f"[{'OK ' if t['passed'] else 'NO '}] {t['test']}" + (f"  ({t['detail']})" if t["detail"] else ""))
    if "jobs" in result:
        print(f"jobs: {result['jobs']['submitted']} submitted, recorded cost {result['jobs']['cost_hqc']:.1f} HQC")
    print("PRE-CHECK PASSED" if result["passed"] else "PRE-CHECK FAILED")
    out_dir = os.path.join(ROOT, "results", a.name)
    if os.path.isdir(out_dir):
        with open(os.path.join(out_dir, "precheck.json"), "w") as f:
            json.dump(result, f, indent=1)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
