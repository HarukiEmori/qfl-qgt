#!/usr/bin/env python
"""Results of the cache that hold fewer shots than their jobs were submitted with.

A result that is read right after its job is reported as completed can hold only the first shots,
because the service is still assembling it.  The executor now checks the number of shots of every
result and repeats the download; this script finds and repairs tables that were stored before that
check existed.

    python scripts/nexus_redownload.py --name emulator2-reimei-E            # list the incomplete tables (no connection)
    python scripts/nexus_redownload.py --name emulator2-reimei-E --repair   # download their results again from Nexus
    python scripts/nexus_redownload.py --all                                # list for every cache under results/

``--repair`` reads the results of the recorded jobs from Nexus (requires ``qnx login``).  It submits
nothing.  An incomplete table is kept in ``<cache>/incomplete/``, and the replacement is recorded in
``jobs.jsonl`` (event ``redownloaded``).  After a repair, run the script that produced the results
again (for a campaign: ``scripts/run_emulator.py --backend nexus --name <name>``); it reads everything
from the cache and writes the result files anew.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qfedqgt.nexus import incomplete_tables, redownload_incomplete, redownloaded     # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="results sub-directory of a campaign (its cache is results/<name>/cache)")
    ap.add_argument("--cache", help="a cache directory (alternative to --name)")
    ap.add_argument("--all", action="store_true", help="every cache directory under results/")
    ap.add_argument("--repair", action="store_true", help="download the incomplete results again from Nexus")
    a = ap.parse_args(argv)
    if a.all:
        caches = sorted(glob.glob(os.path.join(ROOT, "results", "*", "cache*")))
    elif a.cache:
        caches = [os.path.abspath(a.cache)]
    elif a.name:
        caches = [os.path.join(ROOT, "results", a.name, "cache")]
    else:
        ap.error("one of --name, --cache and --all is required")
    left = 0
    for cache in caches:
        rel = os.path.relpath(cache, ROOT)
        todo = incomplete_tables(cache)
        for r in redownloaded(cache):
            print(f"{rel}: downloaded again earlier: {r['name']}: {r['shots_before']} -> {r['shots_after']} shots ({r['time']})")
        if not todo:
            print(f"{rel}: every stored table has the number of shots of its job")
            continue
        for item in todo:
            want = "unknown" if item["shots_requested"] is None else item["shots_requested"]
            print(f"{rel}: job {item['name']}: stored {item['shots_stored']} shots, requested {want}")
        if a.repair:
            redownload_incomplete(cache)
            left += len(incomplete_tables(cache))
        else:
            left += len(todo)
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
