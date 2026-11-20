#!/usr/bin/env python
"""Progress of a campaign of scripts/run_emulator.py, read from the files of its results folder.

    python scripts/emulator_status.py --name emulator-reimei-E-3
    python scripts/emulator_status.py --name emulator-reimei-E-3 --nexus     # also asks Nexus for the open jobs

The script submits nothing.  Without ``--nexus`` it does not connect to Nexus.  It prints the state
of the campaign (finished, running, or stopped), the finished rounds, the jobs and their recorded
cost, and the last lines of the log.  The exit status is 0 when the campaign has finished and 1
otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def read_jobs(path: str) -> tuple:
    """(submitted, completed, failed) from ``jobs.jsonl``: dictionaries {job id: record}."""
    submitted, completed, failed = {}, {}, {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event, job_id = rec.get("event"), rec.get("job_id")
                if event == "submitted":
                    submitted[job_id] = rec
                elif event == "completed":
                    completed[job_id] = rec
                elif event in ("failed", "reattach-failed"):
                    failed[job_id] = rec
    return submitted, completed, failed


def process_alive(pid_file: str, name: str) -> int | None:
    """Process identifier of the campaign script, or of a run_emulator.py process of the campaign, if one is running."""
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(["pgrep", "-f", f"run_emulator.py .*--name {re.escape(name)}( |$)"],
                             capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError):
        return None


def status(name: str, nexus: bool = False, log_lines: int = 6) -> dict:
    from qfedqgt import scenarios as S

    out_dir = os.path.join(ROOT, "results", name)
    info = dict(name=name, exists=os.path.isdir(out_dir))
    if not info["exists"]:
        return info
    packing = {}
    if os.path.exists(os.path.join(out_dir, "packing.json")):
        with open(os.path.join(out_dir, "packing.json")) as f:
            packing = json.load(f)
    scenarios = packing.get("scenarios") or list(S.campaign_scenarios(name))
    seeds = packing.get("seeds") or []
    rounds = S.run_config(scenarios[0]).rounds
    info.update(scenarios=scenarios, seeds=seeds, rounds_total=rounds, runs=len(scenarios) * len(seeds))

    log = os.path.join(out_dir, "campaign.log")
    lines = []
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            lines = f.read().splitlines()
    done_rounds = [int(m.group(1)) for line in lines for m in [re.match(r"round\s+(\d+)\s+noiseless accuracy", line)] if m]
    info["rounds_done"] = max(done_rounds) if done_rounds else (None if not lines else 0)
    info["last_round_line"] = next((line for line in reversed(lines) if line.startswith("round ")), None)
    info["log_tail"] = lines[-log_lines:]

    submitted, completed, failed = read_jobs(os.path.join(out_dir, "cache", "jobs.jsonl"))
    open_jobs = {j: r for j, r in submitted.items() if j not in completed and j not in failed}
    cost = sum(r["cost"] for r in completed.values() if isinstance(r.get("cost"), (int, float)) and r["cost"] == r["cost"])
    times = [r.get("time") for r in list(submitted.values()) + list(completed.values()) if r.get("time")]
    info.update(jobs_submitted=len(submitted), jobs_completed=len(completed), jobs_failed=len(failed),
                jobs_open=len(open_jobs), cost_hqc=cost, last_job_event=max(times) if times else None)
    if times:
        last = time.mktime(time.strptime(max(times)[:19], "%Y-%m-%dT%H:%M:%S"))
        info["minutes_since_last_job_event"] = (time.time() - last) / 60.0

    info["finished"] = os.path.exists(os.path.join(out_dir, "summary.json"))
    if info["finished"]:
        info["rounds_done"] = rounds
    info["driver_pid"] = process_alive(os.path.join(out_dir, "campaign.pid"), name)
    info["state"] = "finished" if info["finished"] else ("running" if info["driver_pid"] else "stopped")

    if nexus and open_jobs:
        import warnings

        warnings.filterwarnings("ignore")
        import qnexus as qnx

        info["open_jobs"] = []
        for job_id, rec in open_jobs.items():
            try:
                st = qnx.jobs.status(qnx.jobs.get(id=job_id))
                state = str(st.status).split(".")[-1]
            except Exception as exc:
                state = f"unavailable ({type(exc).__name__})"
            info["open_jobs"].append(dict(name=rec.get("name"), status=state, submitted=rec.get("time")))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="results sub-directory of the campaign")
    ap.add_argument("--nexus", action="store_true", help="ask Nexus for the status of the open jobs (requires qnx login)")
    ap.add_argument("--json", action="store_true", help="print the status as JSON")
    a = ap.parse_args(argv)
    info = status(a.name, a.nexus)
    if a.json:
        print(json.dumps(info, indent=1))
        return 0 if info.get("finished") else 1
    if not info["exists"]:
        print(f"campaign {a.name}: no results folder")
        return 1
    label = {"finished": "FINISHED (summary.json is written)",
             "running": f"RUNNING (process {info['driver_pid']})",
             "stopped": "STOPPED (no campaign script is running and the campaign has not finished)"}[info["state"]]
    print(f"campaign {a.name}: {label}")
    print(f"  runs: {info['runs']} ({', '.join(info['scenarios'])}; seeds {info['seeds']})")
    if info["rounds_done"] is None:
        print(f"  rounds finished: unknown (no campaign.log; the campaign has {info['rounds_total']} rounds)")
    else:
        stage = "evaluation or output" if info["rounds_done"] >= info["rounds_total"] and not info["finished"] else "training"
        print(f"  rounds finished: {info['rounds_done']} of {info['rounds_total']}"
              + ("" if info["finished"] else f" (stage: {stage})"))
    print(f"  jobs: {info['jobs_submitted']} submitted, {info['jobs_completed']} completed, {info['jobs_open']} open, "
          f"{info['jobs_failed']} failed; recorded cost {info['cost_hqc']:.0f} HQC")
    if info.get("last_job_event"):
        print(f"  last job event: {info['last_job_event']} ({info['minutes_since_last_job_event']:.0f} min ago)")
    for job in info.get("open_jobs", []):
        print(f"    open job {job['name']}: {job['status']} (submitted {job['submitted']})")
    if info["state"] == "stopped":
        print(f"  to continue: scripts/emulator_campaign.sh {a.name}")
    if info["log_tail"]:
        print("  last lines of campaign.log:")
        for line in info["log_tail"]:
            print("    " + line[:160])
    return 0 if info["finished"] else 1


if __name__ == "__main__":
    sys.exit(main())
