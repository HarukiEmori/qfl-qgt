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
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import numpy as np

from . import provenance, sim

__all__ = ["DeviceProfile", "build_program", "program_cost", "AerExecutor", "NexusExecutor", "NexusUnreachable",
           "STALL_EXIT_STATUS", "login_status"]

STALL_EXIT_STATUS = 75     # exit status of a process that the stall guard of NexusExecutor ended


class NexusUnreachable(RuntimeError):
    """Nexus did not answer.  The submitted jobs are not affected, and a later run re-attaches to them."""


def login_status() -> str:
    """Whether the stored Nexus login works, as a sentence.  The account is not shown.

    Call this at the start of every task that submits jobs.  If the login has expired, run
    ``qnx login`` in a terminal (browser authentication) and call the function again.
    """
    try:
        import qnexus as qnx

        qnx.projects.get_all().count()
    except Exception as exc:
        return f"Nexus login: NOT valid ({type(exc).__name__}).  Run `qnx login` in a terminal."
    return "Nexus login: valid"


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

    def __init__(self, segments: int = 32, segments_schedule=()):
        self.segments = segments
        # ((first call, segments), ...): from the given call of run() on (1-based), programs hold this many
        # circuits.  The schedule lets a campaign change its packing between rounds without changing the
        # programs of the rounds that are already in the cache.
        self.segments_schedule = tuple(sorted(segments_schedule))
        self.n_calls = 0
        self.n_circuits = 0
        self.n_shots = 0
        self.n_programs = 0
        self.cost_estimate = 0.0

    # -- to be provided by the runner
    def _submit(self, device, specs, label):
        """Start the programs ``specs = [(template, angle rows, shots), ...]`` on one device."""
        raise NotImplementedError

    def _collect(self, handle):
        """Shot tables of the programs of ``_submit``, in the same order."""
        raise NotImplementedError

    def _segments_now(self) -> int:
        seg = self.segments
        for first, value in self.segments_schedule:
            if self.n_calls >= first:
                seg = value
        return seg

    def run(self, requests: list) -> None:
        # requests with the same device, shot count and template share programs;
        # all programs of one device form one submission (one job on an emulator)
        self.n_calls += 1
        segments = self._segments_now()
        groups, devices = {}, {}
        for i, req in enumerate(requests):
            if req.shots % req.split:
                raise ValueError("shots must be divisible by split")
            key = (req.device, req.shots, req.template)
            if key not in groups:
                devices.setdefault(req.device, []).append(key)
            groups.setdefault(key, []).append(i)
        handles = []
        for device, keys in devices.items():
            specs, layout = [], []
            for key in keys:
                _, shots, template = key
                by_group = {}
                for i in groups[key]:
                    by_group.setdefault(getattr(requests[i], "group", 0), []).append(i)
                for members in by_group.values():        # programs never mix the groups (runs)
                    rows = np.concatenate([requests[i].A for i in members], axis=0)
                    chunks = [rows[s:s + segments] for s in range(0, rows.shape[0], segments)]
                    layout.append((members, template, shots, len(specs), len(chunks)))
                    specs += [(template, ch, shots) for ch in chunks]
            circuits = sum(len(ch) for _, ch, _ in specs)
            label = f"{getattr(device, 'tag', 'local')}-p{len(specs)}-c{circuits}"
            handles.append((layout, specs, self._submit(device, specs, label)))
            self.n_programs += len(specs)
            self.cost_estimate += sum(program_cost(t, len(ch), m) for t, ch, m in specs)
        for layout, specs, handle in handles:
            tables = self._collect(handle)                       # one (shots, n * S) bit table per program
            for members, template, shots, first, count in layout:
                split = max(requests[i].split for i in members)
                parts = [_shots_to_counts(tables[j], template.n, len(specs[j][1]), split)
                         for j in range(first, first + count)]
                counts = np.concatenate(parts, axis=1)           # (split, total circuits, D)
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

    def __init__(self, segments: int = 32, seed: int = 0, segments_schedule=()):
        super().__init__(segments, segments_schedule)
        from pytket.extensions.qiskit import AerBackend

        self.backend = AerBackend()
        self.seed = seed

    def describe(self) -> dict:
        return dict(kind="local noiseless simulation of the packed programs (Aer)")

    def _submit(self, device, specs, label):
        tables = []
        for j, (template, ch, shots) in enumerate(specs):
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
    With ``cached_only=True`` the executor never submits; it raises an error if a result is
    missing, which protects a reproduction from the cache against accidental jobs.
    ``job_records`` lists the jobs behind the results of this session, including the jobs
    whose results were read from the cache (see :mod:`qfedqgt.provenance`).

    A request to Nexus can block for ever on a dead connection, for example after the
    computer slept or changed its network.  With ``stall_timeout`` set, a guard thread ends
    the process with status ``STALL_EXIT_STATUS`` when a request has not returned within that
    many seconds; a caller such as ``scripts/emulator_campaign.sh`` then starts the process
    again, and the new process re-attaches to the submitted jobs.
    """

    def __init__(self, project: str, cache_dir: str, segments: int = 32, run_name: str = "qfedqgt",
                 max_programs_per_job: int = 300, timeout: float | None = None, cost_margin: float = 2.0,
                 io_threads: int = 4, poll_interval: float = 20.0, max_poll_failures: int = 90,
                 segments_schedule=(), cached_only: bool = False, stall_timeout: float | None = None):
        super().__init__(segments, segments_schedule)
        self.stall_timeout = stall_timeout          # seconds after which a request without answer ends the process
        self._active, self._active_lock, self._guard = {}, threading.Lock(), None
        self.cached_only = cached_only              # never submit: fail if a result is not in the cache
        self.io_threads = io_threads
        self.poll_interval = poll_interval          # seconds between two status requests of a job
        self.max_poll_failures = max_poll_failures  # consecutive failed status requests before giving up
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
            with self._request("project"):
                self.project = qnx.projects.get_or_create(name=self.project_name)
            qnx.context.set_active_project(self.project)
        return self.qnx

    @contextmanager
    def _request(self, label: str):
        """Marks a request to Nexus as in progress while the block runs (used by the stall guard)."""
        if self.stall_timeout is None:
            yield
            return
        self._start_guard()
        token = object()
        with self._active_lock:
            self._active[id(token)] = (label, time.monotonic())
        try:
            yield
        finally:
            with self._active_lock:
                self._active.pop(id(token), None)

    def _start_guard(self) -> None:
        """Start the thread that ends the process when a request has not returned for ``stall_timeout`` seconds.

        The age of a request is measured with the monotonic clock, which does not advance while the
        computer sleeps, so that a sleep alone does not trigger the guard.
        """
        if self._guard is not None:
            return

        def watch():
            while True:
                time.sleep(min(5.0, self.stall_timeout / 4))
                now = time.monotonic()
                with self._active_lock:
                    late = [(now - t0, label) for label, t0 in self._active.values() if now - t0 > self.stall_timeout]
                if late:
                    age, label = max(late)
                    print(f"[stall guard] the request '{label}' to Nexus has not returned for {age:.0f} s; the process "
                          f"ends with status {STALL_EXIT_STATUS} and can be started again (submitted jobs are re-attached)",
                          file=sys.stderr, flush=True)
                    os._exit(STALL_EXIT_STATUS)

        self._guard = threading.Thread(target=watch, daemon=True, name="nexus-stall-guard")
        self._guard.start()

    def _map(self, fn, items) -> list:
        """Apply ``fn`` to the items with a few threads (uploads and downloads wait on the network); order is kept."""
        items = list(items)
        if self.io_threads <= 1 or len(items) <= 1:
            return [fn(x) for x in items]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.io_threads) as pool:
            return list(pool.map(fn, items))

    def _wait(self, job, name: str) -> None:
        """Wait until a job is completed, by asking for its status at a fixed interval.

        The default waiting of the client keeps a websocket open and then polls with intervals of
        up to twenty minutes; a dropped connection or a slow poll would stall a campaign of many
        short jobs.  Failed status requests (network, token refresh) are retried; if they keep
        failing, :class:`NexusUnreachable` is raised and the job stays re-attachable.
        """
        t0, failures = time.time(), 0
        while True:
            try:
                with self._request(f"status of {name}"):
                    state = str(self.qnx.jobs.status(job).status).split(".")[-1]
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= self.max_poll_failures:
                    raise NexusUnreachable(f"status of job {name} unavailable after {failures} requests: {exc!r}") from exc
                state = "UNKNOWN"
            if state == "COMPLETED":
                return
            if state in ("ERROR", "CANCELLED", "CANCELLING", "TERMINATED", "DEPLETED"):
                raise RuntimeError(f"job {name} ended with status {state}")
            if self.timeout is not None and time.time() - t0 > self.timeout:
                raise TimeoutError(f"job {name} not completed after {self.timeout:.0f} s (status {state})")
            time.sleep(self.poll_interval)

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
    def _submit(self, device, specs, label):
        self._devices.add(device.tag)
        keys = [self._key(device, template, ch, shots) for template, ch, shots in specs]
        todo = [j for j, k in enumerate(keys) if not os.path.exists(os.path.join(self.cache_dir, k + ".npz"))]
        self._note_jobs([k for j, k in enumerate(keys) if j not in set(todo)], "cache")
        jobs = []
        if todo and self.cached_only:
            raise RuntimeError(f"{len(todo)} of {len(keys)} programs for {device.tag} are not in the cache "
                               f"({provenance._relative(self.cache_dir)}); nothing was submitted because cached_only is set")
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
                with self._request(f"job {item['name']}"):
                    job = qnx.jobs.get(id=job_id)
            except Exception as exc:
                if getattr(exc, "status_code", None) not in (404, 410):
                    # no answer or a server error: the job may well exist, so nothing is submitted in its place
                    raise NexusUnreachable(f"job {item['name']} could not be retrieved: {exc!r}") from exc
                # the service no longer knows the job: submit its programs again
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
            def upload(j, _name=name):
                circuit = build_program(specs[j][0], specs[j][1], f"{_name}-p{j}")
                with self._request(f"upload of {_name}-p{j}"):
                    return qnx.circuits.upload(circuit=circuit, name=f"{_name}-p{j}", project=self.project)

            refs = self._map(upload, part)
            shots = [int(specs[j][2]) for j in part]
            cost = [float(np.ceil(self.cost_margin * program_cost(specs[j][0], len(specs[j][1]), specs[j][2]))) for j in part]
            with self._request(f"submission of {name}"):
                job = qnx.start_execute_job(programs=refs, n_shots=shots, backend_config=cfg,
                                            name=name, project=self.project, max_cost=cost)
            ref_ids = [str(r.id) for r in refs]
            jobs.append(dict(job=job, part=part, ref_ids=ref_ids, name=name, source="submitted"))
            self._log(dict(event="submitted", name=name, job_id=str(job.id), device=asdict(device),
                           shots=sorted(set(shots)), circuit_shots=int(sum(len(specs[j][1]) * specs[j][2] for j in part)),
                           programs=len(part), keys=[keys[j] for j in part], ref_ids=ref_ids,
                           commit=self._program.get("commit"), modified=bool(self._program.get("modified")),
                           time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        return dict(keys=keys, jobs=jobs, specs=specs)

    def _collect(self, handle):
        qnx = self.qnx
        specs = handle["specs"]
        for item in handle["jobs"]:
            try:
                self._wait(item["job"], item["name"])
            except NexusUnreachable:
                raise                   # the job itself is fine: the next run re-attaches to it
            except Exception as exc:
                # an errored job is recorded so that a later run submits its programs again
                # instead of re-attaching to it
                self._log(dict(event="failed", name=item["name"], job_id=str(item["job"].id),
                               error=repr(exc)[:300], time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
                raise
            with self._request(f"results of {item['name']}"):
                by_input = {str(r.get_input().id): r for r in qnx.jobs.results(item["job"])}

            def fetch(pair, _name=item["name"]):
                j, ref_id = pair
                with self._request(f"download of {_name}"):
                    res = by_input[ref_id].download_result()
                bits = program_bits(specs[j][0].n, len(specs[j][1]))
                table = np.asarray(res.get_shots(cbits=bits), dtype=np.uint8)
                np.savez_compressed(os.path.join(self.cache_dir, handle["keys"][j] + ".npz"), shots=table)

            self._map(fetch, zip(item["part"], item["ref_ids"]))
            try:
                with self._request(f"cost of {item['name']}"):
                    cost = float(qnx.jobs.cost(item["job"]))
            except Exception:           # the cost is informative only
                cost = float("nan")
            self.cost_charged += 0.0 if np.isnan(cost) else cost
            self._log(dict(event="completed", name=item["name"], job_id=str(item["job"].id), cost=cost,
                           time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            self._note_jobs([handle["keys"][j] for j in item["part"]], item.get("source", "submitted"))
        return [np.load(os.path.join(self.cache_dir, k + ".npz"))["shots"].astype(np.int64) for k in handle["keys"]]
