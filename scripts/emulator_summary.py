#!/usr/bin/env python
"""Summary of the federated training on the emulator: final accuracies and paired differences.

The script reads the finished campaigns of the emulator experiment from ``results/`` and prints, for
every data partition,

* the final test accuracy measured on the emulator (``device``) and the accuracy of the noiseless
  circuit with the same parameters (``noiseless``), as the mean and the standard error over the seeds,
* the accuracy of QFedQGT minus that of every baseline in percentage points, with the standard error of
  the paired difference and the number of seeds in which QFedQGT is higher,
* the same differences for every campaign separately and for both partitions together (one value per
  seed, the mean over the partitions).

    python scripts/emulator_summary.py                      # every finished campaign on the emulator
    python scripts/emulator_summary.py emulator-local       # other result folders of scripts/run_emulator.py

Output: text, and ``results/emulator-summary.json`` when the campaigns on the emulator are summarized.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import scenarios as S          # noqa: E402

PARTITIONS = (("emulator_iid", "IID data"), ("emulator_dirichlet", "Dirichlet partition"))
KEYS = (("device", "accuracy_device"), ("noiseless", "accuracy_noiseless"))


def load(directories) -> list:
    """Runs of the given result folders; every run remembers its campaign."""
    runs = []
    for d in directories:
        with open(os.path.join(ROOT, "results", d, "summary.json")) as f:
            summary = json.load(f)
        for r in summary["runs"]:
            r["campaign"] = d
            runs.append(r)
    return runs


def mean_se(x) -> tuple:
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def paired(runs, method, key, reference="QFedQGT") -> dict:
    """Difference reference - method in percentage points over the given runs."""
    d = 100.0 * np.array([r["final"][reference][key] - r["final"][method][key] for r in runs])
    m, se = mean_se(d)
    return dict(mean=m, se=se, higher=int((d > 0).sum()), n=len(d))


def per_seed(runs, method, key, reference="QFedQGT") -> dict:
    """The same difference with one value per seed: the mean over the runs of the seed (the partitions)."""
    seeds = {}
    for r in runs:
        seeds.setdefault(r["seed"], []).append(100.0 * (r["final"][reference][key] - r["final"][method][key]))
    m, se = mean_se([np.mean(v) for v in seeds.values()])
    return dict(mean=m, se=se, n=len(seeds))


def summarize(directories) -> dict:
    runs = load(directories)
    methods = [m for m in ("QFedAvg", "QFedFisher", "QFedQGT") if m in runs[0]["final"]]
    baselines = [m for m in methods if m != "QFedQGT"]
    out = dict(campaigns=list(directories), partitions={}, both_partitions={})
    for scenario, title in PARTITIONS:
        sub = sorted([r for r in runs if r["scenario"] == scenario], key=lambda r: r["seed"])
        if not sub:
            continue
        print(f"{title}: {len(sub)} seeds")
        row = dict(seeds=[r["seed"] for r in sub], accuracy={}, difference={})
        for m in methods:
            row["accuracy"][m] = {name: dict(zip(("mean", "se"), mean_se([r["final"][m][key] for r in sub]))) for name, key in KEYS}
            a = row["accuracy"][m]
            print(f"  {m:11s} device {a['device']['mean']:.3f} +/- {a['device']['se']:.3f}"
                  f"   noiseless {a['noiseless']['mean']:.3f} +/- {a['noiseless']['se']:.3f}")
        for m in baselines:
            row["difference"][m] = {}
            for name, key in KEYS:
                p = paired(sub, m, key)
                row["difference"][m][name] = p
                line = (f"  QFedQGT - {m:11s} ({name:9s}) {p['mean']:+5.1f} +/- {p['se']:4.1f} points, "
                        f"higher in {p['higher']:2d} of {p['n']} seeds")
                if len(directories) > 1:
                    parts = []
                    for d in directories:
                        c = [r for r in sub if r["campaign"] == d]
                        if len(c) > 1:
                            q = paired(c, m, key)
                            row["difference"][m].setdefault(name + "_by_campaign", {})[d] = q
                            parts.append(f"{q['mean']:+.1f} +/- {q['se']:.1f} ({q['n']} seeds)")
                    line += "   by campaign: " + ", ".join(parts)
                print(line)
        out["partitions"][scenario] = row
    if len(out["partitions"]) > 1:
        print("Both partitions (one value per seed, standard error over the seeds)")
        for m in baselines:
            out["both_partitions"][m] = {}
            for name, key in KEYS:
                p = per_seed(runs, m, key)
                q = paired(runs, m, key)
                p["higher_runs"], p["runs"] = q["higher"], q["n"]
                out["both_partitions"][m][name] = p
                print(f"  QFedQGT - {m:11s} ({name:9s}) {p['mean']:+5.1f} +/- {p['se']:4.1f} points over {p['n']} seeds, "
                      f"higher in {q['higher']} of {q['n']} runs")
    return out


def main(argv=None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    on_emulator = not args
    directories = args or [d for d in S.EMULATOR_CAMPAIGNS if os.path.exists(os.path.join(ROOT, "results", d, "summary.json"))]
    if not directories:
        raise SystemExit("no finished campaign in results/")
    out = summarize(directories)
    if on_emulator:
        path = os.path.join(ROOT, "results", "emulator-summary.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote results/emulator-summary.json")


if __name__ == "__main__":
    main()
