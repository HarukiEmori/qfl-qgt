"""Record of a run: which program produced a result, when, and on which device.

Every script of this repository that executes circuits wraps its work in :func:`record_run`.
When the block ends, the record is

* shown automatically (a table in a Jupyter notebook, plain text in a terminal), and
* saved as JSON next to the results.

The record lists the program (repository, commit, whether the working tree was modified, entry
point), the time (start and end of the run, submission and completion of every job), and the
device (backend, device name, error-rate multipliers, job identifiers, shots, cost).  The
executors log the commit at the moment a job is submitted, so that results read back from the
cache are still attributed to the program and the time that produced them.

The record contains no host name, user name or absolute path.

    from qfedqgt.provenance import record_run

    with record_run("echo circuits", entry="scripts/nexus_geometry.py", executor=ex,
                    parameters=dict(shots=100), save_to="results/nexus-geometry") as rec:
        ex.run(requests)
"""
from __future__ import annotations

import contextlib
import html
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

__all__ = ["ROOT", "program_info", "environment_info", "RunRecord", "record_run", "load_record", "now"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACKAGES = ("numpy", "scipy", "pytket", "qnexus", "quantinuum-schemas", "pytket-qiskit", "matplotlib")


def now() -> str:
    """Local time with the UTC offset, for example ``2026-11-19T10:15:00+0900``."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", "-C", ROOT) + args, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.rstrip("\n") if out.returncode == 0 else None


_IGNORED = ("results/", "figures/", "data/")


def program_info() -> dict:
    """Repository, commit and state of the working tree of the running program.

    ``modified`` lists the files outside the result folders that differ from the commit, including
    files that are not tracked yet; a run with a non-empty list is not reproducible from the commit alone.
    """
    commit = _git("rev-parse", "HEAD")
    url = _git("remote", "get-url", "origin")
    if url:
        url = re.sub(r"//[^/@]+@", "//", url)                      # never keep credentials of a remote URL
        url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
        url = re.sub(r"\.git$", "", url)
    status = _git("status", "--porcelain")
    modified = []
    for line in (status or "").splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')      # "XY path" (the two status letters, a space)
        if path and not path.startswith(_IGNORED):
            modified.append(path)
    return dict(repository=url, commit=commit, branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
                modified=sorted(modified))


def environment_info() -> dict:
    """Versions of Python and of the packages that execute or build circuits."""
    from importlib import metadata

    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return dict(python=platform.python_version(), system=f"{platform.system()} {platform.machine()}", packages=versions)


def _relative(path: str) -> str:
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else os.path.basename(path)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


@dataclass
class RunRecord:
    """What ran, when, and where.  Filled by :func:`record_run`."""
    task: str
    entry: str = ""
    program: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    backend: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    started: str = ""
    finished: str = ""
    duration_s: float = 0.0
    jobs: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    # -- content
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)
        return path

    def _job_summary(self) -> dict:
        jobs = self.jobs
        if not jobs:
            return {}
        submitted = sorted(j["submitted"] for j in jobs if j.get("submitted"))
        completed = sorted(j["completed"] for j in jobs if j.get("completed"))
        commits = sorted({j["commit"][:10] if j.get("commit") else "not recorded" for j in jobs})
        cost = [j["cost"] for j in jobs if isinstance(j.get("cost"), (int, float)) and j["cost"] == j["cost"]]
        devices = sorted({f"{j.get('device', '?')} x{j.get('error_scale', 1):g}" for j in jobs})
        return dict(count=len(jobs), devices=devices, first_submitted=submitted[0] if submitted else "",
                    last_completed=completed[-1] if completed else "", commits=commits,
                    programs=sum(int(j.get("programs", 0)) for j in jobs), cost_charged=float(sum(cost)),
                    from_cache=sum(1 for j in jobs if j.get("source") == "cache"))

    def rows(self) -> list:
        """(section, label, value) triples of the displayed table."""
        p, e, b = self.program, self.environment, self.backend
        commit = p.get("commit") or "unknown (not a git checkout)"
        state = "clean" if not p.get("modified") else "modified: " + ", ".join(p["modified"])
        out = [("Program", "task", self.task), ("Program", "entry point", self.entry),
               ("Program", "repository", p.get("repository") or "unknown"),
               ("Program", "commit", commit), ("Program", "working tree", state),
               ("When", "started", self.started), ("When", "finished", self.finished),
               ("When", "duration", f"{self.duration_s:.0f} s")]
        js = self._job_summary()
        if js:
            out += [("When", "jobs submitted from", js["first_submitted"]),
                    ("When", "last job completed", js["last_completed"])]
        for key, value in b.items():
            out.append(("Device", key.replace("_", " "), _fmt(value)))
        if js:
            out += [("Device", "jobs", f"{js['count']} ({js['from_cache']} read from the cache), {js['programs']} programs"),
                    ("Device", "devices of the jobs", ", ".join(js["devices"])),
                    ("Device", "commits that submitted the jobs", ", ".join(js["commits"])),
                    ("Device", "cost charged (HQC)", f"{js['cost_charged']:.1f}")]
            ids = [str(j.get("job_id")) for j in self.jobs]
            shown = ", ".join(ids[:4]) + (f", ... ({len(ids)} in total, all in the saved record)" if len(ids) > 4 else "")
            out.append(("Device", "job identifiers", shown))
        for key, value in self.usage.items():
            out.append(("Usage", key.replace("_", " "), _fmt(value)))
        for key, value in self.parameters.items():
            out.append(("Parameters", key.replace("_", " "), _fmt(value)))
        out.append(("Environment", "python", f"{e.get('python', '?')} on {e.get('system', '?')}"))
        out.append(("Environment", "packages", ", ".join(f"{k} {v}" for k, v in e.get("packages", {}).items())))
        for path in self.outputs:
            out.append(("Outputs", "file", path))
        for note in self.notes:
            out.append(("Notes", "", note))
        return out

    # -- display
    def text(self) -> str:
        lines, last = ["=" * 78, "RUN RECORD"], None
        for section, label, value in self.rows():
            if section != last:
                lines.append(f"-- {section}")
                last = section
            lines.append(f"   {label:<34}{value}" if label else f"   {value}")
        lines.append("=" * 78)
        return "\n".join(lines)

    __str__ = text

    def _repr_html_(self) -> str:
        cells, last = [], None
        for section, label, value in self.rows():
            head = f"<th style='text-align:left;vertical-align:top'>{html.escape(section)}</th>" if section != last else "<td></td>"
            last = section
            cells.append(f"<tr>{head}<td style='text-align:left'>{html.escape(label)}</td>"
                         f"<td style='text-align:left'><code>{html.escape(str(value))}</code></td></tr>")
        return ("<table><caption style='text-align:left'><b>Run record</b> (which program, when, on which device)"
                "</caption>" + "".join(cells) + "</table>")

    def show(self) -> None:
        """Display the record: a table in a notebook, text elsewhere."""
        if _in_notebook():
            from IPython.display import display

            display(self)
        else:
            print(self.text(), flush=True)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_fmt(v)}" for k, v in value.items())
    return str(value)


def _executor_backend(executor) -> dict:
    describe = getattr(executor, "describe", None)
    return dict(describe()) if callable(describe) else {}


@contextlib.contextmanager
def record_run(task: str, entry: str = "", executor=None, backend: dict | None = None,
               parameters: dict | None = None, save_to: str | None = None, show: bool = True):
    """Context manager that records a run and shows the record when the block ends.

    ``executor`` is the object that runs the circuits; its ``describe()`` gives the backend
    entry and its ``job_records`` the jobs.  ``save_to`` is a directory (the record is written
    to ``<save_to>/provenance.json``) or a path ending in ``.json``.  The record is shown and
    saved even if the block raises, with the error noted, so that an interrupted run leaves a
    trace of the jobs it submitted.
    """
    rec = RunRecord(task=task, entry=entry or _relative(sys.argv[0] or "interactive"),
                    program=program_info(), environment=environment_info(),
                    backend=dict(backend or {}), parameters=dict(parameters or {}), started=now())
    t0 = time.time()
    try:
        yield rec
    except BaseException as exc:
        rec.notes.append(f"the run ended with an error: {type(exc).__name__}: {str(exc)[:200]}")
        raise
    finally:
        rec.finished = now()
        rec.duration_s = round(time.time() - t0, 1)
        if executor is not None:
            rec.backend = {**_executor_backend(executor), **rec.backend}
            rec.jobs = list(getattr(executor, "job_records", []))
            for key in ("n_circuits", "n_shots", "n_programs", "cost_estimate"):
                value = getattr(executor, key, None)
                if value:
                    rec.usage.setdefault({"cost_estimate": "cost_formula_hqc"}.get(key, key), value)
        if rec.program.get("modified"):
            rec.notes.append("the working tree differed from the commit; the commit alone does not reproduce this run")
        if save_to:
            path = save_to if save_to.endswith(".json") else os.path.join(save_to, "provenance.json")
            rec.outputs = [_relative(p) for p in rec.outputs]
            rec.save(path)
            if _relative(path) not in rec.outputs:
                rec.outputs.append(_relative(path))
                rec.save(path)
        if show:
            rec.show()


def load_record(path: str) -> RunRecord:
    """Read a saved record (for display in a notebook)."""
    with open(path) as f:
        return RunRecord(**json.load(f))
