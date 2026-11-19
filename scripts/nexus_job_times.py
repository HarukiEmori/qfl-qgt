#!/usr/bin/env python
"""Waiting and execution times of the jobs of a campaign, from the status records of Quantinuum Nexus.

The job log of a campaign (``results/<name>/cache/jobs.jsonl``) holds the identifiers of its jobs.
This script asks Nexus for the times at which every job was queued, started and completed, and
groups the jobs into execution steps (one step per round of a training, then the evaluation).

    python scripts/nexus_job_times.py --name emulator-reimei-E

Requires ``qnx login``.  Output: ``results/<name>/job_times.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="emulator-reimei-E", help="results sub-directory of the campaign")
    ap.add_argument("--gap", type=float, default=180.0, help="seconds without a submission that separate two steps")
    a = ap.parse_args(argv)

    import qnexus as qnx

    out_dir = os.path.join(ROOT, "results", a.name)
    with open(os.path.join(out_dir, "cache", "jobs.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    submitted = [r for r in records if r.get("event") == "submitted"]
    utc = dt.timezone.utc

    def iso(t):
        return None if t is None else t.astimezone(utc).isoformat(timespec="seconds")

    jobs = []
    for r in submitted:
        st = qnx.jobs.status(qnx.jobs.get(id=r["job_id"]))
        local = dt.datetime.strptime(r["time"], "%Y-%m-%dT%H:%M:%S%z")
        q, run, done = st.queued_time, st.running_time, st.completed_time
        jobs.append(dict(name=r["name"], job_id=r["job_id"], device=r["device"]["device_name"],
                         error_scale=r["device"]["gate_scale"], programs=r["programs"],
                         circuit_shots=r.get("circuit_shots"), status=str(st.status).split(".")[-1],
                         submitted=iso(local), queued=iso(q), running=iso(run), completed=iso(done),
                         wait_s=None if (q is None or run is None) else (run - q).total_seconds(),
                         run_s=None if (run is None or done is None) else (done - run).total_seconds()))
    # execution steps: bursts of submissions
    steps, current = [], []
    for job in jobs:
        t = dt.datetime.fromisoformat(job["submitted"])
        if current and (t - dt.datetime.fromisoformat(current[-1]["submitted"])).total_seconds() > a.gap:
            steps.append(current)
            current = []
        current.append(job)
    if current:
        steps.append(current)
    step_rows = []
    for i, group in enumerate(steps, start=1):
        start = min(dt.datetime.fromisoformat(j["submitted"]) for j in group)
        ends = [dt.datetime.fromisoformat(j["completed"]) for j in group if j["completed"]]
        end = max(ends) if len(ends) == len(group) else None
        step_rows.append(dict(step=i, jobs=len(group), programs=int(sum(j["programs"] for j in group)),
                              start=start.isoformat(timespec="seconds"), end=None if end is None else end.isoformat(timespec="seconds"),
                              minutes=None if end is None else round((end - start).total_seconds() / 60.0, 1)))
    waits = np.array([j["wait_s"] for j in jobs if j["wait_s"] is not None]) / 60.0
    runs = np.array([j["run_s"] for j in jobs if j["run_s"] is not None]) / 60.0
    summary = dict(jobs=len(jobs), steps=len(step_rows),
                   wait_minutes=dict(min=float(waits.min()), median=float(np.median(waits)), max=float(waits.max())),
                   run_minutes=dict(min=float(runs.min()), median=float(np.median(runs)), max=float(runs.max())))
    with open(os.path.join(out_dir, "job_times.json"), "w") as f:
        json.dump(dict(campaign=a.name, summary=summary, steps=step_rows, jobs=jobs), f, indent=1)
    print(json.dumps(summary, indent=1))
    for row in step_rows:
        print(f"step {row['step']:2d}: {row['jobs']:2d} jobs, {row['programs']:3d} programs, {row['minutes']} min")


if __name__ == "__main__":
    main()
