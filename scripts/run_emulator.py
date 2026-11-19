#!/usr/bin/env python
"""Federated training on the Quantinuum emulator: several scenarios and seeds in lockstep.

Every (scenario, seed) is one federated run of the compared methods.  The runs advance round by
round together, and the circuits of all runs, clients and methods of a round are submitted at
once, so that they wait in the queue of the emulator once per round.  After the last round the
models of the last rounds are evaluated on the reference profile of the emulator.

Backends
--------
``local``   the gate-level noise model of this repository (the multipliers ``local_scales`` of the scenario)
``aer``     noiseless Aer simulation of the same packed programs (checks the program path)
``nexus``   Quantinuum emulator through Quantinuum Nexus (requires ``qnx login``)

    python scripts/run_emulator.py --backend local --name emulator-local             # prediction of the local model
    python scripts/run_emulator.py --backend nexus --device reimei-SC --rounds 1 --name emulator-syntax-check
    python scripts/run_emulator.py --backend nexus --device reimei-E --name emulator-reimei-E

The run is resumable: finished jobs are read from the cache and unfinished jobs are re-attached, so
the same command continues an interrupted campaign.  Output in ``results/<name>/``: one ``.npz`` per
run, ``summary.json``, the job log and the shot tables (``cache/``), and the run record
``provenance.json`` (which program, when, on which device; shown when the script ends).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt import provenance                                              # noqa: E402
from qfedqgt import scenarios as S                                          # noqa: E402
from qfedqgt.experiment import build, make_methods                          # noqa: E402
from qfedqgt.federated import LocalExecutor, Request, SharedExecutor, run_federated   # noqa: E402

METHODS = ("QFedAvg", "QFedFisher", "QFedQGT")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(S.EMULATOR_SCENARIOS), choices=sorted(S.SCENARIOS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(S.EMULATOR_SEEDS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--backend", default="local", choices=["local", "aer", "nexus"])
    ap.add_argument("--device", default="reimei-E", help="Nexus device name")
    ap.add_argument("--project", default="QFedQGT", help="Nexus project name")
    ap.add_argument("--name", default=None, help="results sub-directory (default: emulator-<backend or device>)")
    ap.add_argument("--rounds", type=int, default=None, help="override the number of rounds (for tests)")
    ap.add_argument("--segments", type=int, default=S.EMULATOR_SEGMENTS, help="circuits packed into one program")
    ap.add_argument("--no-eval", action="store_true", help="skip the evaluation of the final models on the device")
    ap.add_argument("--eval-inputs", type=int, default=S.EMULATOR_EVAL_INPUTS,
                    help="test images used for the evaluation on the device (the same number from every class)")
    ap.add_argument("--programs-per-job", type=int, default=S.EMULATOR_PROGRAMS_PER_JOB, help="programs of one Nexus job")
    ap.add_argument("--segments-schedule", nargs="*", default=[], metavar="CALL:SEGMENTS",
                    help="change of the packing during a campaign: from the given execution step on (1-based; one step "
                         "per round, then the evaluation), programs hold SEGMENTS circuits")
    ap.add_argument("--separate-runs", action="store_true",
                    help="never pack circuits of different runs into one program (runs can then be added later)")
    ap.add_argument("--timeout", type=float, default=12 * 3600.0, help="seconds to wait for one job")
    a = ap.parse_args(argv)

    name = a.name or ("emulator-" + (a.device if a.backend == "nexus" else a.backend))
    out_dir = os.path.join(ROOT, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    runs = [(sc, seed) for sc in a.scenarios for seed in a.seeds]

    schedule = tuple((int(x.split(":")[0]), int(x.split(":")[1])) for x in a.segments_schedule)
    device_factory, executor = None, LocalExecutor()
    if a.backend in ("aer", "nexus"):
        from qfedqgt.nexus import AerExecutor, DeviceProfile, NexusExecutor

        if a.backend == "aer":
            executor = AerExecutor(segments=a.segments, seed=0, segments_schedule=schedule)
            device_factory = lambda scale, role, k: DeviceProfile("aer", scale, noisy=False)      # noqa: E731
        else:
            executor = NexusExecutor(a.project, os.path.join(out_dir, "cache"), segments=a.segments,
                                     run_name=name, timeout=a.timeout, max_programs_per_job=a.programs_per_job,
                                     segments_schedule=schedule)
            device_factory = lambda scale, role, k: DeviceProfile(a.device, scale)               # noqa: E731

    # -- setup of every run
    setups = []
    for sc_name, seed in runs:
        sc = S.SCENARIOS[sc_name]
        cfg = replace(S.run_config(sc_name), seed=seed)
        if a.rounds is not None:
            cfg = replace(cfg, rounds=a.rounds)
        lrs = {m: S.learning_rates(sc_name)[m] for m in a.methods}
        model, clients, theta0, ds, eval_device = build(sc, seed, device_factory=device_factory)
        setups.append(dict(scenario=sc_name, seed=seed, sc=sc, cfg=cfg, lrs=lrs, model=model, clients=clients,
                           theta0=theta0, ds=ds, eval_device=eval_device))
    tail = S.tail(a.scenarios[0])
    parameters = dict(scenarios=a.scenarios, seeds=a.seeds, methods=a.methods,
                      rounds=setups[0]["cfg"].rounds, batch=setups[0]["cfg"].batch,
                      echo_shots=setups[0]["cfg"].echo_shots, eval_shots=setups[0]["cfg"].eval_shots,
                      eval_inputs=a.eval_inputs,
                      shots_per_circuit={s: sorted(set(S.SCENARIOS[s].shots)) for s in a.scenarios},
                      error_rate_multipliers={s: sorted(set(S.SCENARIOS[s].gate_scales)) for s in a.scenarios},
                      learning_rates={s: {m: S.learning_rates(s)[m] for m in a.methods} for s in a.scenarios},
                      segments=a.segments, programs_per_job=a.programs_per_job,
                      segments_schedule=[f"{c}:{v}" for c, v in schedule])
    backend = {}
    if a.backend == "local":
        backend = dict(local_multipliers={s: sorted(set(S.SCENARIOS[s].local_scales or S.SCENARIOS[s].gate_scales))
                                          for s in a.scenarios})

    with provenance.record_run("federated training of several runs in lockstep", entry="scripts/run_emulator.py",
                               executor=executor, backend=backend, parameters=parameters, save_to=out_dir) as rec:
        shared = SharedExecutor(executor, parties=len(setups), separate=a.separate_runs)
        results, errors = [None] * len(setups), [None] * len(setups)
        t0 = time.time()
        lock = threading.Lock()
        progress = {}

        def work(i: int) -> None:
            s = setups[i]

            def on_round(r, theta, hist):
                with lock:
                    progress[(i, r)] = {m: hist[m]["acc_exact"][-1] for m in theta}
                    if all((j, r) in progress for j in range(len(setups))):
                        mean = {m: float(np.mean([progress[(j, r)][m] for j in range(len(setups))])) for m in theta}
                        print(f"round {r:3d}  noiseless accuracy, mean over the runs: "
                              + "  ".join(f"{m}={v:.3f}" for m, v in mean.items())
                              + f"  ({time.time() - t0:.0f} s)", flush=True)

            try:
                results[i] = run_federated(s["model"], s["clients"], make_methods(s["lrs"]), s["cfg"], s["theta0"],
                                           s["ds"].X_test, s["ds"].y_test, shared.view(i), None, on_round=on_round)
            except BaseException as exc:          # reported after all threads have stopped
                errors[i] = exc
                shared._barrier.abort()

        threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(setups))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        failed = [e for e in errors if e is not None]
        if failed:
            first = shared.error or next(e for e in failed if "shared execution stopped" not in str(e))
            raise first

        # -- evaluation of the models of the last rounds on the reference profile of the device,
        #    on a class-balanced subset of the test images (the noiseless accuracy uses all of them)
        evals = {}
        for s in setups:
            y = s["ds"].y_test
            per_class = max(1, a.eval_inputs // len(s["sc"].classes))
            s["eval_idx"] = np.sort(np.concatenate([np.flatnonzero(y == c)[:per_class] for c in range(len(s["sc"].classes))]))
        if not a.no_eval:
            reqs = []
            for i, s in enumerate(setups):
                t = s["model"].template
                for m in a.methods:
                    thetas = results[i][m]["theta"]
                    for r in range(max(1, len(thetas) - tail), len(thetas)):
                        req = Request(s["eval_device"], t, t.angles(s["ds"].X_test[s["eval_idx"]], thetas[r]),
                                      s["cfg"].eval_shots, tag="eval", group=i if a.separate_runs else 0)
                        evals[(i, m, r)] = req
                        reqs.append(req)
            executor.run(reqs)

        # -- output
        summary = dict(name=name, backend=a.backend, device=a.device if a.backend == "nexus" else a.backend,
                       tail=tail, methods=a.methods, runs=[], seconds=time.time() - t0)
        for i, s in enumerate(setups):
            flat = {}
            row = dict(scenario=s["scenario"], seed=s["seed"], learning_rates=s["lrs"], final={})
            for m in a.methods:
                h = results[i][m]
                for key, val in h.items():
                    if isinstance(val, np.ndarray):
                        flat[f"{m}/{key}"] = val
                rounds = sorted(r for (j, mm, r) in evals if j == i and mm == m)
                acc_dev = []
                for r in rounds:
                    phi = evals[(i, m, r)].counts[0] @ s["model"].obs / s["cfg"].eval_shots
                    acc_dev.append(float((s["model"].predict(phi) == s["ds"].y_test[s["eval_idx"]]).mean()))
                flat[f"{m}/eval_rounds"] = np.array(rounds)
                flat[f"{m}/acc_device"] = np.array(acc_dev)
                row["final"][m] = dict(accuracy_device=float(np.mean(acc_dev)) if acc_dev else None,
                                       accuracy_noiseless=float(np.mean(h["acc_exact"][-tail:])),
                                       loss_noiseless=float(np.mean(h["loss_exact"][-tail:])))
            path = os.path.join(out_dir, f"{s['scenario']}-seed{s['seed']}.npz")
            np.savez_compressed(path, **flat)
            rec.outputs.append(path)
            row["scenario_definition"] = asdict(s["sc"])
            row["run"] = asdict(s["cfg"])
            summary["runs"].append(row)
            print(f"{s['scenario']:22s} seed {s['seed']}  " + "  ".join(
                f"{m}: device {row['final'][m]['accuracy_device'] if row['final'][m]['accuracy_device'] is not None else float('nan'):.3f}"
                f" noiseless {row['final'][m]['accuracy_noiseless']:.3f}" for m in a.methods), flush=True)
        summary["usage"] = dict(n_circuits=getattr(executor, "n_circuits", None), n_shots=getattr(executor, "n_shots", None),
                                n_programs=getattr(executor, "n_programs", None),
                                cost_formula_hqc=getattr(executor, "cost_estimate", None),
                                cost_charged_hqc=getattr(executor, "cost_charged", None))
        path = os.path.join(out_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=1, default=float)
        rec.outputs.append(path)


if __name__ == "__main__":
    main()
