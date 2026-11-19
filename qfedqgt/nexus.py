"""Execution of circuit requests as programs in the native trapped-ion gate set.

The same program builder serves two runners:

* :class:`NexusExecutor` submits the programs to a Quantinuum device or emulator
  (``reimei-E`` by default) through Quantinuum Nexus (``qnexus``);
* :class:`AerExecutor` runs them on a local noiseless simulator and is used to check
  the program path offline.

Programs
--------
Several circuits on the same qubits are packed into one program: every circuit is
followed by a measurement of all qubits into its own classical bits and, except for
the last one, by a reset of all qubits (qubit reuse).  One program therefore returns,
shot by shot, one outcome for each of its circuits, which keeps the number of
programs, compilations and stored results small.  Circuits are written directly in
the native gates ``PhasedX``, ``Rz`` and ``ZZPhase``, so that no compilation job is needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import provenance, sim

__all__ = ["DeviceProfile", "build_program", "program_cost", "AerExecutor", "NexusExecutor"]


@dataclass(frozen=True)
class DeviceProfile:
    """Emulated device of one client: the emulator with rescaled gate and memory errors."""
    device_name: str = "reimei-E"
    gate_scale: float = 1.0      # multiplies the one-qubit, two-qubit and memory error rates
    noisy: bool = True

    def error_params(self) -> dict | None:
        if not self.noisy or self.gate_scale == 1.0:
            return None
        s = float(self.gate_scale)
        return dict(p1_scale=s, p2_scale=s, memory_scale=s)

    @property
    def tag(self) -> str:
        return f"{self.device_name}-x{self.gate_scale:g}" + ("" if self.noisy else "-noiseless")


# --------------------------------------------------------------------------- programs
MAX_REGISTER_WIDTH = 64      # the backend rejects wider classical registers (error CHI-1002)


def program_bits(n: int, segments: int) -> list:
    """Classical bits of a packed program in measurement order (segment by segment, qubit by qubit).

    The bits are spread over registers ``m0, m1, ...`` of at most ``MAX_REGISTER_WIDTH`` bits.
    """
    from pytket import Bit

    return [Bit(f"m{i // MAX_REGISTER_WIDTH}", i % MAX_REGISTER_WIDTH) for i in range(n * segments)]


def build_program(template: sim.Template, A: np.ndarray, name: str):
    """pytket circuit running the rows of ``A`` one after another on the same qubits."""
    from pytket import Circuit, OpType, Qubit

    n, S = template.n, A.shape[0]
    c = Circuit(n, name=name)
    total = n * S
    for r in range((total + MAX_REGISTER_WIDTH - 1) // MAX_REGISTER_WIDTH):
        c.add_c_register(f"m{r}", min(MAX_REGISTER_WIDTH, total - r * MAX_REGISTER_WIDTH))
    bits = program_bits(n, S)
    for s in range(S):
        for g, gate in enumerate(template.gates):
            a = float(A[s, g]) / np.pi                       # pytket angles are in half-turns
            if gate.kind == "ry":
                c.add_gate(OpType.PhasedX, [a, 0.5], [gate.qubits[0]])
            elif gate.kind == "rx":
                c.add_gate(OpType.PhasedX, [a, 0.0], [gate.qubits[0]])
            elif gate.kind == "rz":
                c.Rz(a, gate.qubits[0])
            elif gate.kind == "zz":
                c.ZZPhase(a, gate.qubits[0], gate.qubits[1])
            else:
                raise ValueError(gate.kind)
        for q in range(n):
            c.Measure(Qubit(q), bits[s * n + q])
        if s < S - 1:
            for q in range(n):
                c.add_gate(OpType.Reset, [q])
    return c


def program_cost(template: sim.Template, segments: int, shots: int) -> float:
    """Hardware-credit estimate ``5 + C (N_1q + 10 N_2q + 5 N_m) / 5000`` of one program."""
    n1 = sum(1 for g in template.gates if g.kind != "zz") * segments
    n2 = template.count("zz") * segments
    nm = template.n * (2 * segments)      # initial preparation, measurements, and resets
    return 5.0 + shots * (n1 + 10 * n2 + 5 * nm) / 5000.0


def _shots_to_counts(shot_table: np.ndarray, n: int, segments: int, split: int) -> np.ndarray:
    """(shots, n * segments) bit table -> counts of shape (split, segments, 2^n)."""
    M = shot_table.shape[0]
    weights = 1 << (n - 1 - np.arange(n))
    idx = shot_table.reshape(M, segments, n) @ weights          # (M, segments)
    out = np.zeros((split, segments, 2 ** n), dtype=np.int64)
    per = M // split
    for h in range(split):
        block = idx[h * per:(h + 1) * per]
        for s in range(segments):
            out[h, s] = np.bincount(block[:, s], minlength=2 ** n)
    return out


class _ProgramExecutor:
    """Groups requests by (device, shots), packs them into programs, and parses the shots."""

    def __init__(self, segments: int = 32):
        self.segments = segments
        self.n_circuits = 0
        self.n_shots = 0
        self.n_programs = 0
        self.cost_estimate = 0.0

    # -- to be provided by the runner
    def _submit(self, device, template, chunks, shots, label):
        raise NotImplementedError

    def _collect(self, handle):
        raise NotImplementedError

    def run(self, requests: list) -> None:
        groups = {}
        for i, req in enumerate(requests):
            if req.shots % req.split:
                raise ValueError("shots must be divisible by split")
            groups.setdefault((req.device, req.shots, req.template), []).append(i)
        handles = []
        for (device, shots, _), members in groups.items():
            template = requests[members[0]].template
            rows = np.concatenate([requests[i].A for i in members], axis=0)
            chunks = [rows[s:s + self.segments] for s in range(0, rows.shape[0], self.segments)]
            label = f"{getattr(device, 'tag', 'local')}-s{shots}-c{rows.shape[0]}"
            handles.append((members, template, chunks, shots, self._submit(device, template, chunks, shots, label)))
            self.n_programs += len(chunks)
            self.cost_estimate += sum(program_cost(template, len(ch), shots) for ch in chunks)
        for members, template, chunks, shots, handle in handles:
            tables = self._collect(handle)                       # one (shots, n * S) bit table per program
            split = max(requests[i].split for i in members)
            parts = [_shots_to_counts(tb, template.n, len(ch), split) for tb, ch in zip(tables, chunks)]
            counts = np.concatenate(parts, axis=1)               # (split, total circuits, D)
            pos = 0
            for i in members:
                req = requests[i]
                c = counts[:, pos:pos + req.n_circuits]
                req.counts = c if req.split == split else c.sum(axis=0, keepdims=True)
                pos += req.n_circuits
                self.n_circuits += req.n_circuits
                self.n_shots += req.n_circuits * req.shots


# --------------------------------------------------------------------------- local runner
class AerExecutor(_ProgramExecutor):
    """Runs the programs on the noiseless Aer simulator (offline check of the program path)."""

    def __init__(self, segments: int = 32, seed: int = 0):
        super().__init__(segments)
        from pytket.extensions.qiskit import AerBackend

        self.backend = AerBackend()
        self.seed = seed

    def describe(self) -> dict:
        return dict(kind="local noiseless simulation of the packed programs (Aer)")

    def _submit(self, device, template, chunks, shots, label):
        tables = []
        for j, ch in enumerate(chunks):
            circ = self.backend.get_compiled_circuit(build_program(template, ch, f"{label}-{j}"), optimisation_level=0)
            self.seed += 1
            res = self.backend.run_circuit(circ, n_shots=shots, seed=self.seed)
            bits = program_bits(template.n, len(ch))
            tables.append(np.asarray(res.get_shots(cbits=bits), dtype=np.int64))
        return tables

    def _collect(self, handle):
        return handle


# --------------------------------------------------------------------------- Nexus runner
class NexusExecutor(_ProgramExecutor):
    """Runs the programs on a Quantinuum backend through Quantinuum Nexus.

    Every group of programs is one execute job.  The shot tables of finished jobs are
    cached in ``cache_dir`` under a key derived from the device profile, the shot count
    and the circuit angles, so that an interrupted campaign can be resumed without
    resubmitting completed work.  A log of the jobs (identifier, programs, shots, cost,
    and the commit of the program that submitted them) is appended to ``cache_dir/jobs.jsonl``.
    A job that was submitted but not collected before an interruption is found again
    through this log and is re-attached by its identifier instead of being submitted a
    second time.

    The connection to Nexus is opened only when something has to be submitted, so that a
    run whose results are all in the cache needs neither a login nor a network connection.
    ``job_records`` lists the jobs behind the results of this session, including the jobs
    whose results were read from the cache (see :mod:`qfedqgt.provenance`).
    """

    def __init__(self, project: str, cache_dir: str, segments: int = 32, run_name: str = "qfedqgt",
                 max_programs_per_job: int = 300, timeout: float | None = None, cost_margin: float = 2.0):
        super().__init__(segments)
        self.project_name = project
        self.qnx = None
        self.project = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.run_name = run_name
        self.max_programs_per_job = max_programs_per_job
        self.timeout = timeout
        self.cost_margin = cost_margin
        self.cost_charged = 0.0
        self.job_records = []
        self._counter = 0
        self._program = None
        self._devices = set()

    def _connect(self):
        """Open the connection to Nexus (requires ``qnx login``) the first time it is needed."""
        if self.qnx is None:
            import qnexus as qnx

            self.qnx = qnx
            self.project = qnx.projects.get_or_create(name=self.project_name)
            qnx.context.set_active_project(self.project)
        return self.qnx

    def describe(self) -> dict:
        return dict(kind="Quantinuum Nexus", project=self.project_name,
                    device_profiles=sorted(self._devices), cache=provenance._relative(self.cache_dir))

    # -- helpers
    def _config(self, device: DeviceProfile):
        kwargs = dict(device_name=device.device_name)
        if not device.noisy:
            kwargs["noisy_simulation"] = False
        params = device.error_params()
        if params:
            from quantinuum_schemas.models.quantinuum_systems_noise import UserErrorParams

            kwargs["error_params"] = UserErrorParams(**params)
        return self._connect().QuantinuumConfig(**kwargs)

    def _key(self, device, template, chunk, shots) -> str:
        h = hashlib.sha1()
        h.update(json.dumps(asdict(device), sort_keys=True).encode())
        h.update(str(shots).encode())
        h.update(repr([(g.kind, g.qubits) for g in template.gates]).encode())
        h.update(np.ascontiguousarray(np.round(chunk, 12)).tobytes())
        return h.hexdigest()

    def _log(self, record: dict) -> None:
        with open(os.path.join(self.cache_dir, "jobs.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

    def _pending(self) -> dict:
        """{cache key: (job id, circuit id, job name)} of the jobs that were submitted and not collected."""
        path = os.path.join(self.cache_dir, "jobs.jsonl")
        if not os.path.exists(path):
            return {}
        submitted, done = {}, set()
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "submitted" and "keys" in rec:
                    submitted[rec["job_id"]] = rec
                elif rec.get("event") in ("completed", "reattach-failed", "failed"):
                    done.add(rec.get("job_id"))
        out = {}
        for job_id, rec in submitted.items():
            if job_id not in done:
                for key, ref_id in zip(rec["keys"], rec["ref_ids"]):
                    out[key] = (job_id, ref_id, rec["name"])
        return out

    def _job_index(self) -> tuple:
        """({job id: merged log record}, {cache key: job id}) from ``jobs.jsonl``."""
        path = os.path.join(self.cache_dir, "jobs.jsonl")
        jobs, by_key = {}, {}
        if not os.path.exists(path):
            return jobs, by_key
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                job_id, event = rec.get("job_id"), rec.get("event")
                if event == "submitted":
                    device = rec.get("device", {})
                    jobs[job_id] = dict(job_id=job_id, name=rec.get("name"), device=device.get("device_name"),
                                        error_scale=device.get("gate_scale", 1.0), noisy=device.get("noisy", True),
                                        shots=rec.get("shots"), programs=rec.get("programs"),
                                        submitted=rec.get("time"), commit=rec.get("commit"),
                                        modified=rec.get("modified"), completed=None, cost=None)
                    for key in rec.get("keys", []):
                        by_key[key] = job_id
                elif event == "completed" and job_id in jobs:
                    jobs[job_id].update(completed=rec.get("time"), cost=rec.get("cost"))
        return jobs, by_key

    def _note_jobs(self, keys, source: str) -> None:
        """Add the jobs that produced the shot tables ``keys`` to ``job_records`` (once each)."""
        if not keys:
            return
        jobs, by_key = self._job_index()
        seen = {r["job_id"] for r in self.job_records}
        for key in keys:
            job_id = by_key.get(key)
            if job_id is not None and job_id not in seen:
                seen.add(job_id)
                self.job_records.append(dict(jobs[job_id], source=source))

    # -- runner interface
    def _submit(self, device, template, chunks, shots, label):
        self._devices.add(device.tag)
        keys = [self._key(device, template, ch, shots) for ch in chunks]
        todo = [j for j, k in enumerate(keys) if not os.path.exists(os.path.join(self.cache_dir, k + ".npz"))]
        self._note_jobs([k for j, k in enumerate(keys) if j not in set(todo)], "cache")
        jobs = []
        qnx = self._connect() if todo else None
        if todo and self._program is None:
            self._program = provenance.program_info()
        # re-attach the jobs of an interrupted run before anything is submitted
        pending = self._pending() if todo else {}
        attach = {}
        for j in todo:
            if keys[j] in pending:
                job_id, ref_id, name = pending[keys[j]]
                attach.setdefault(job_id, dict(part=[], ref_ids=[], name=name))
                attach[job_id]["part"].append(j)
                attach[job_id]["ref_ids"].append(ref_id)
        for job_id, item in attach.items():
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            try:
                job = qnx.jobs.get(id=job_id)
            except Exception as exc:        # the job is gone: submit its programs again
                self._log(dict(event="reattach-failed", job_id=job_id, error=repr(exc)[:200], time=stamp))
                continue
            jobs.append(dict(job=job, source="reattached", **item))
            self._log(dict(event="reattached", name=item["name"], job_id=job_id, programs=len(item["part"]), time=stamp))
            todo = [j for j in todo if j not in set(item["part"])]
        cfg = self._config(device) if todo else None
        for start in range(0, len(todo), self.max_programs_per_job):
            part = todo[start:start + self.max_programs_per_job]
            self._counter += 1
            name = f"{self.run_name}-{self._counter:04d}-{label}-{time.strftime('%H%M%S')}"
            refs = [qnx.circuits.upload(circuit=build_program(template, chunks[j], f"{name}-p{j}"),
                                        name=f"{name}-p{j}", project=self.project) for j in part]
            cost = [float(np.ceil(self.cost_margin * program_cost(template, len(chunks[j]), shots))) for j in part]
            job = qnx.start_execute_job(programs=refs, n_shots=[shots] * len(refs), backend_config=cfg,
                                        name=name, project=self.project, max_cost=cost)
            ref_ids = [str(r.id) for r in refs]
            jobs.append(dict(job=job, part=part, ref_ids=ref_ids, name=name, source="submitted"))
            self._log(dict(event="submitted", name=name, job_id=str(job.id), device=asdict(device), shots=shots,
                           programs=len(part), keys=[keys[j] for j in part], ref_ids=ref_ids,
                           commit=self._program.get("commit"), modified=bool(self._program.get("modified")),
                           time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        return dict(keys=keys, jobs=jobs, template=template, chunks=chunks)

    def _collect(self, handle):
        qnx = self.qnx
        n = handle["template"].n
        for item in handle["jobs"]:
            try:
                qnx.jobs.wait_for(item["job"], timeout=self.timeout)
            except Exception as exc:
                # an errored job is recorded so that a later run submits its programs again
                # instead of re-attaching to it
                self._log(dict(event="failed", name=item["name"], job_id=str(item["job"].id),
                               error=repr(exc)[:300], time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
                raise
            by_input = {str(r.get_input().id): r for r in qnx.jobs.results(item["job"])}
            for j, ref_id in zip(item["part"], item["ref_ids"]):
                res = by_input[ref_id].download_result()
                bits = program_bits(n, len(handle["chunks"][j]))
                table = np.asarray(res.get_shots(cbits=bits), dtype=np.uint8)
                np.savez_compressed(os.path.join(self.cache_dir, handle["keys"][j] + ".npz"), shots=table)
            try:
                cost = float(qnx.jobs.cost(item["job"]))
            except Exception:           # the cost is informative only
                cost = float("nan")
            self.cost_charged += 0.0 if np.isnan(cost) else cost
            self._log(dict(event="completed", name=item["name"], job_id=str(item["job"].id), cost=cost,
                           time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            self._note_jobs([handle["keys"][j] for j in item["part"]], item.get("source", "submitted"))
        return [np.load(os.path.join(self.cache_dir, k + ".npz"))["shots"].astype(np.int64) for k in handle["keys"]]
