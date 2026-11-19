"""Submission, caching and re-attachment logic of the Nexus executor, exercised with a stand-in for qnexus.

The stand-in records the calls and returns all-zero shot tables.  It checks the bookkeeping of
``NexusExecutor`` (which jobs are submitted, cached, or re-attached); it does not emulate the service.
"""
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


class FakeQnx(types.ModuleType):
    """Minimal stand-in for the parts of qnexus that NexusExecutor calls."""

    def __init__(self):
        super().__init__("qnexus")
        self.store = {}            # job id -> job
        self.started, self.fetched, self.configs = [], [], []
        self.broken = set()        # ids of jobs that end in an error
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
            def wait_for(job, timeout=None):
                if str(job.id) in fake.broken:
                    raise RuntimeError("job errored")
                return None

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
    ex3._submit(device3, t3, chunks, reqs3[0].shots, "interrupted")
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
    ex5._submit(device5, t5, chunks, reqs5[0].shots, "lost")
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
    handle = ex7._submit(device7, t7, chunks, reqs7[0].shots, "broken")
    fake.broken.add(str(handle["jobs"][0]["job"].id))
    with pytest.raises(RuntimeError):
        ex7._collect(handle)
    n_started, n_fetched = len(fake.started), len(fake.fetched)
    ex8 = NexusExecutor("test", cache, segments=2)
    _, reqs8 = _requests(device7)
    ex8.run(reqs8)
    assert len(fake.started) == n_started + 1 and len(fake.fetched) == n_fetched
