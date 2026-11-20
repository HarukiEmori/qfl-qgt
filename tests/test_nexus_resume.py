"""Submission, caching and re-attachment logic of the Nexus executor, exercised with a stand-in for qnexus.

The stand-in records the calls and returns all-zero shot tables.  It checks the bookkeeping of
``NexusExecutor`` (which jobs are submitted, cached, or re-attached); it does not emulate the service.
"""
import json
import os
import subprocess
import sys
import types
import uuid

import numpy as np
import pytest

from qfedqgt import sim
from qfedqgt.federated import Request

pytest.importorskip("pytket")
pytest.importorskip("quantinuum_schemas")


class _Ref:
    def __init__(self, **kw):
        self.id = uuid.uuid4()
        self.__dict__.update(kw)


class NotFound(Exception):
    """The answer of the service for an unknown job (HTTP status 404)."""
    status_code = 404


class FakeQnx(types.ModuleType):
    """Minimal stand-in for the parts of qnexus that NexusExecutor calls."""

    def __init__(self):
        super().__init__("qnexus")
        self.store = {}            # job id -> job
        self.started, self.fetched, self.configs = [], [], []
        self.broken = set()        # ids of jobs that end in an error
        self.default_state = "COMPLETED"   # final status of the jobs that are not in ``broken``
        self.offline = False       # the service does not answer
        fake = self

        class projects:
            @staticmethod
            def get_or_create(name):
                return _Ref(name=name)

        class context:
            @staticmethod
            def set_active_project(project):
                fake.project = project

        class circuits:
            @staticmethod
            def upload(circuit, name, project):
                return _Ref(circuit=circuit, name=name)

        class jobs:
            @staticmethod
            def status(job):
                if fake.offline:
                    raise ConnectionError("no route to host")
                state = "ERROR" if str(job.id) in fake.broken else fake.default_state
                return types.SimpleNamespace(status=f"JobStatusEnum.{state}")

            @staticmethod
            def results(job):
                out = []
                for ref, shots in zip(job.programs, job.n_shots):
                    def get_shots(cbits, _shots=shots):
                        return np.zeros((_shots, len(cbits)), dtype=int)
                    result = types.SimpleNamespace(get_shots=get_shots)
                    out.append(types.SimpleNamespace(get_input=lambda _r=ref: _r, download_result=lambda _x=result: _x))
                return out

            @staticmethod
            def cost(job):
                return 1.5

            @staticmethod
            def get(id=None, **kw):
                fake.fetched.append(str(id))
                if fake.offline:
                    raise ConnectionError("no route to host")
                if str(id) not in fake.store:
                    raise NotFound(f"job {id} not found")
                return fake.store[str(id)]

        self.projects, self.context, self.circuits, self.jobs = projects, context, circuits, jobs

    def QuantinuumConfig(self, **kw):
        self.configs.append(kw)
        return types.SimpleNamespace(**kw)

    def start_execute_job(self, programs, n_shots, backend_config, name, project, max_cost):
        job = _Ref(programs=programs, n_shots=n_shots, name=name, max_cost=max_cost)
        self.store[str(job.id)] = job
        self.started.append(name)
        return job


def _requests(device, shots=6):
    t = sim.layered_ansatz(2, 1)
    rng = np.random.default_rng(0)
    A = t.angles(rng.uniform(-1, 1, size=(5, t.n_data)), rng.uniform(-1, 1, size=t.n_params))
    return t, [Request(device, t, A[:3], shots, tag="a"), Request(device, t, A[3:], shots, tag="b")]


def test_submit_cache_and_reattach(tmp_path, monkeypatch):
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor

    device = DeviceProfile("reimei-E", 40.0)
    cache = str(tmp_path / "cache")

    # 1. a complete run: one job with three programs (five circuits, two per program)
    ex = NexusExecutor("test", cache, segments=2)
    t, reqs = _requests(device)
    ex.run(reqs)
    assert len(fake.started) == 1 and ex.n_programs == 3
    assert reqs[0].counts.shape == (1, 3, t.dim) and reqs[1].counts.shape == (1, 2, t.dim)
    assert reqs[0].counts[0, :, 0].tolist() == [6, 6, 6]            # all shots in the all-zero outcome
    assert ex.cost_charged == pytest.approx(1.5)
    # the noisy profile sends the documented scale factors to the backend configuration
    params = fake.configs[0]["error_params"]
    assert params.p1_scale == 40.0 and params.p2_scale == 40.0 and params.memory_scale == 40.0

    # the job is recorded with the commit of the program that submitted it
    assert len(ex.job_records) == 1 and ex.job_records[0]["source"] == "submitted"
    assert ex.job_records[0]["device"] == "reimei-E" and ex.job_records[0]["error_scale"] == 40.0
    assert "commit" in ex.job_records[0] and ex.job_records[0]["completed"]

    # 2. a second run with the same inputs uses the cache, submits nothing, and does not connect to Nexus
    ex2 = NexusExecutor("test", cache, segments=2)
    _, reqs2 = _requests(device)
    ex2.run(reqs2)
    assert len(fake.started) == 1 and fake.fetched == [] and ex2.qnx is None
    assert np.array_equal(reqs2[0].counts, reqs[0].counts)
    # ... and still names the job that produced the cached results
    assert [r["job_id"] for r in ex2.job_records] == [ex.job_records[0]["job_id"]]
    assert ex2.job_records[0]["source"] == "cache"

    # 3. an interrupted run: the job is submitted but never collected
    device3 = DeviceProfile("reimei-E", 80.0)
    ex3 = NexusExecutor("test", cache, segments=2)
    t3, reqs3 = _requests(device3)
    rows = np.concatenate([r.A for r in reqs3], axis=0)
    chunks = [rows[s:s + 2] for s in range(0, rows.shape[0], 2)]
    ex3._submit(device3, [(t3, ch, reqs3[0].shots) for ch in chunks], "interrupted")
    assert len(fake.started) == 2

    # 4. the restarted run finds the job in jobs.jsonl and re-attaches it instead of submitting again
    ex4 = NexusExecutor("test", cache, segments=2)
    _, reqs4 = _requests(device3)
    ex4.run(reqs4)
    assert len(fake.started) == 2 and len(fake.fetched) == 1
    assert reqs4[0].counts.shape == (1, 3, t3.dim)

    # 5. if the job cannot be retrieved, its programs are submitted again
    device5 = DeviceProfile("reimei-E", 120.0)
    ex5 = NexusExecutor("test", cache, segments=2)
    t5, reqs5 = _requests(device5)
    ex5._submit(device5, [(t5, ch, reqs5[0].shots) for ch in chunks], "lost")
    assert len(fake.started) == 3
    fake.store.clear()                                              # the service no longer knows the job
    ex6 = NexusExecutor("test", cache, segments=2)
    _, reqs6 = _requests(device5)
    ex6.run(reqs6)
    assert len(fake.started) == 4 and reqs6[1].counts.shape == (1, 2, t5.dim)


    # 6. a job that ends in an error is recorded as failed and is not re-attached by the next run
    device7 = DeviceProfile("reimei-E", 160.0)
    ex7 = NexusExecutor("test", cache, segments=2)
    t7, reqs7 = _requests(device7)
    handle = ex7._submit(device7, [(t7, ch, reqs7[0].shots) for ch in chunks], "broken")
    fake.broken.add(str(handle["jobs"][0]["job"].id))
    with pytest.raises(RuntimeError):
        ex7._collect(handle)
    n_started, n_fetched = len(fake.started), len(fake.fetched)
    ex8 = NexusExecutor("test", cache, segments=2)
    _, reqs8 = _requests(device7)
    ex8.run(reqs8)
    assert len(fake.started) == n_started + 1 and len(fake.fetched) == n_fetched


def test_requests_of_one_device_form_one_job_with_their_own_shots(tmp_path, monkeypatch):
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor

    t = sim.layered_ansatz(2, 1)
    e = t.echo()
    rng = np.random.default_rng(0)
    A = t.angles(rng.uniform(-1, 1, size=(5, t.n_data)), rng.uniform(-1, 1, size=t.n_params))
    E = e.angles(rng.uniform(-1, 1, size=(3, t.n_data)), rng.uniform(-1, 1, size=t.n_params))
    clean, noisy = DeviceProfile("reimei-E", 1.0), DeviceProfile("reimei-E", 300.0)
    reqs = [Request(clean, t, A[:3], 10, tag="grad"), Request(noisy, t, A[3:], 1, tag="grad"),
            Request(clean, e, E[:1], 5, tag="echo"), Request(noisy, e, E[1:], 5, tag="echo")]
    ex = NexusExecutor("test", str(tmp_path / "cache"), segments=2)
    ex.run(reqs)
    assert len(fake.started) == 2                                  # one job per device profile
    shots = sorted(sorted(job.n_shots) for job in fake.store.values())
    assert shots == [[1, 5], [5, 10, 10]]                           # every program keeps its own shot count
    assert [r.counts.shape for r in reqs] == [(1, 3, t.dim), (1, 2, t.dim), (1, 1, t.dim), (1, 2, t.dim)]
    assert reqs[0].counts[0, :, 0].tolist() == [10, 10, 10] and reqs[3].counts[0, :, 0].tolist() == [5, 5]


def test_programs_of_a_group_do_not_depend_on_the_other_groups(tmp_path, monkeypatch):
    """Adding a second run to a job leaves the cache keys of the first run unchanged."""
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor

    t = sim.layered_ansatz(2, 1)
    rng = np.random.default_rng(0)
    A = t.angles(rng.uniform(-1, 1, size=(7, t.n_data)), rng.uniform(-1, 1, size=t.n_params))
    device = DeviceProfile("reimei-E", 300.0)
    cache = str(tmp_path / "cache")
    ex = NexusExecutor("test", cache, segments=4)
    ex.run([Request(device, t, A[:3], 2, group=0)])
    assert len(fake.started) == 1
    ex2 = NexusExecutor("test", cache, segments=4)
    first, second = Request(device, t, A[:3], 2, group=0), Request(device, t, A[3:], 2, group=1)
    ex2.run([first, second])
    assert len(fake.started) == 2 and ex2.n_programs == 2          # only the new group was submitted
    assert fake.store[max(fake.store, key=lambda k: fake.started.index(fake.store[k].name))].n_shots == [2]
    assert first.counts.shape == (1, 3, t.dim) and second.counts.shape == (1, 4, t.dim)


def test_segments_schedule_keeps_the_programs_of_earlier_calls(tmp_path, monkeypatch):
    """A packing that changes at a later call leaves the cache keys of the earlier calls unchanged."""
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor

    t = sim.layered_ansatz(2, 1)
    rng = np.random.default_rng(0)
    A = t.angles(rng.uniform(-1, 1, size=(8, t.n_data)), rng.uniform(-1, 1, size=t.n_params))
    device = DeviceProfile("reimei-E", 300.0)
    cache = str(tmp_path / "cache")
    ex = NexusExecutor("test", cache, segments=2)
    ex.run([Request(device, t, A, 2)])                      # call 1: four programs of two circuits
    assert ex.n_programs == 4 and len(fake.started) == 1
    ex2 = NexusExecutor("test", cache, segments=2, segments_schedule=((2, 8),))
    ex2.run([Request(device, t, A, 2)])                     # call 1 again: the same four programs, all cached
    assert len(fake.started) == 1
    ex2.run([Request(device, t, A + 0.1, 2)])               # call 2: one program of eight circuits
    assert len(fake.started) == 2 and ex2.n_programs == 5


def test_cached_only_never_submits(tmp_path, monkeypatch):
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor

    device = DeviceProfile("reimei-E", 300.0)
    cache = str(tmp_path / "cache")
    t, reqs = _requests(device)
    with pytest.raises(RuntimeError, match="not in the cache"):
        NexusExecutor("test", cache, segments=2, cached_only=True).run(reqs)
    assert fake.started == []
    NexusExecutor("test", cache, segments=2).run(reqs)               # fills the cache
    _, again = _requests(device)
    ex = NexusExecutor("test", cache, segments=2, cached_only=True)
    ex.run(again)                                                     # now served from the cache
    assert len(fake.started) == 1 and ex.qnx is None and np.array_equal(again[0].counts, reqs[0].counts)


def test_unreachable_service_keeps_the_jobs_reattachable(tmp_path, monkeypatch):
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor, NexusUnreachable

    device = DeviceProfile("reimei-E", 40.0)
    cache = str(tmp_path / "cache")
    ex = NexusExecutor("test", cache, segments=2, poll_interval=0.0, max_poll_failures=2)
    t, reqs = _requests(device)
    rows = np.concatenate([r.A for r in reqs], axis=0)
    specs = [(t, rows[s:s + 2], reqs[0].shots) for s in range(0, rows.shape[0], 2)]
    handle = ex._submit(device, specs, "offline")
    assert len(fake.started) == 1

    # the service stops answering while the job is awaited: the job is not recorded as failed
    fake.offline = True
    with pytest.raises(NexusUnreachable):
        ex._collect(handle)
    # ... and a restarted run that cannot retrieve the job submits nothing in its place
    ex2 = NexusExecutor("test", cache, segments=2, poll_interval=0.0, max_poll_failures=2)
    with pytest.raises(NexusUnreachable):
        ex2.run(_requests(device)[1])
    with open(os.path.join(cache, "jobs.jsonl")) as f:
        events = [json.loads(line)["event"] for line in f]
    assert events == ["submitted"] and len(fake.started) == 1

    # when the service answers again, the run re-attaches to the job and submits nothing
    fake.offline = False
    ex3 = NexusExecutor("test", cache, segments=2)
    _, reqs3 = _requests(device)
    ex3.run(reqs3)
    assert len(fake.started) == 1 and reqs3[0].counts.shape == (1, 3, t.dim)


def test_programs_that_fail_repeatedly_stop_the_run_for_good(tmp_path, monkeypatch):
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor, NexusJobFatal

    device = DeviceProfile("reimei-E", 200.0)
    cache = str(tmp_path / "cache")
    fake.default_state = "ERROR"                       # every job of these programs ends in an error
    kinds = []
    for attempt in range(3):
        ex = NexusExecutor("test", cache, segments=2, max_job_failures=3)
        _, reqs = _requests(device)
        with pytest.raises(RuntimeError) as info:
            ex.run(reqs)
        kinds.append(type(info.value).__name__)
    # the first two failures allow another submission, the third one does not
    assert kinds == ["RuntimeError", "RuntimeError", "NexusJobFatal"] and len(fake.started) == 3
    assert issubclass(NexusJobFatal, RuntimeError)
    # other programs are not affected by the failures of these programs
    fake.default_state = "COMPLETED"
    ex = NexusExecutor("test", cache, segments=2, max_job_failures=3)
    _, other = _requests(DeviceProfile("reimei-E", 210.0))
    ex.run(other)
    assert other[0].counts.shape[1] == 3


def test_a_cancelled_job_stops_the_run_for_good(tmp_path, monkeypatch):
    fake = FakeQnx()
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    from qfedqgt.nexus import DeviceProfile, NexusExecutor, NexusJobFatal

    for state in ("CANCELLED", "TERMINATED", "DEPLETED"):
        fake.default_state = state
        ex = NexusExecutor("test", str(tmp_path / state), segments=2)
        _, reqs = _requests(DeviceProfile("reimei-E", 220.0))
        with pytest.raises(NexusJobFatal):
            ex.run(reqs)                                # the first cancelled job already ends the run
    assert len(fake.started) == 3


def test_fatal_stop_has_its_own_exit_status():
    from qfedqgt.nexus import FATAL_EXIT_STATUS, STALL_EXIT_STATUS

    assert FATAL_EXIT_STATUS == 76 and FATAL_EXIT_STATUS != STALL_EXIT_STATUS
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = open(os.path.join(root, "scripts", "emulator_campaign.sh")).read()
    # the restart loop of the campaign script ends on this status instead of starting the run again
    assert '"$status" -eq 76' in script and script.index('"$status" -eq 76') < script.index("restart $n in")


def test_runner_turns_a_fatal_stop_into_its_exit_status(monkeypatch):
    import importlib.util
    from qfedqgt.nexus import FATAL_EXIT_STATUS, NexusJobFatal

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("run_emulator_cli", os.path.join(root, "scripts", "run_emulator.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def cancelled(argv=None):
        raise NexusJobFatal("job x was ended with the status CANCELLED")

    def broken(argv=None):
        raise RuntimeError("job x ended with status ERROR")

    monkeypatch.setattr(runner, "main", cancelled)
    with pytest.raises(SystemExit) as info:
        runner.cli([])
    assert info.value.code == FATAL_EXIT_STATUS
    monkeypatch.setattr(runner, "main", broken)       # an ordinary failure keeps its exception: the campaign restarts
    with pytest.raises(RuntimeError):
        runner.cli([])


def test_stall_guard_ends_a_process_whose_request_does_not_return(tmp_path):
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})\n"
        "from qfedqgt.nexus import NexusExecutor\n"
        f"ex = NexusExecutor('test', {str(tmp_path / 'cache')!r}, stall_timeout=0.4)\n"
        "with ex._request('fast request'):\n"
        "    pass\n"
        "print('first request returned', flush=True)\n"
        "with ex._request('blocked request'):\n"
        "    time.sleep(30)\n"
        "print('not reached', flush=True)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=25)
    from qfedqgt.nexus import STALL_EXIT_STATUS

    assert out.returncode == STALL_EXIT_STATUS
    assert "first request returned" in out.stdout and "not reached" not in out.stdout
    assert "blocked request" in out.stderr

