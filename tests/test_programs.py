"""The native-gate programs reproduce the simulator, including the packing of several circuits."""
import numpy as np
import pytest

from qfedqgt import sim
from qfedqgt.federated import Request
from qfedqgt.nexus import AerExecutor, DeviceProfile, _shots_to_counts, build_program, program_cost

pytket = pytest.importorskip("pytket")


def test_single_circuit_unitary_matches_simulator():
    from pytket import Circuit, OpType

    t = sim.layered_ansatz(3, 2)
    rng = np.random.default_rng(3)
    A = t.angles(rng.uniform(-1, 1, size=(1, t.n_data)), rng.uniform(-np.pi, np.pi, size=t.n_params))
    prog = build_program(t, A, "check")
    unitary_part = Circuit(t.n)
    for cmd in prog.get_commands():
        if cmd.op.type in (OpType.Measure, OpType.Reset):
            continue
        unitary_part.add_gate(cmd.op, cmd.qubits)
    psi = unitary_part.get_statevector()
    ref = sim.statevector(t, A)[0]
    assert abs(np.vdot(ref, psi)) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(np.abs(psi) ** 2, np.abs(ref) ** 2, atol=1e-9)


def test_shot_table_parsing():
    n, segments = 2, 3
    table = np.array([[0, 1, 1, 0, 1, 1],      # outcomes 01, 10, 11
                      [0, 1, 0, 0, 1, 1],      # 01, 00, 11
                      [1, 1, 1, 0, 0, 0],      # 11, 10, 00
                      [0, 0, 1, 0, 0, 1]])     # 00, 10, 01
    c = _shots_to_counts(table, n, segments, split=2)
    assert c.shape == (2, 3, 4)
    assert c[0, 0].tolist() == [0, 2, 0, 0] and c[1, 0].tolist() == [1, 0, 0, 1]
    assert c[0, 1].tolist() == [1, 0, 1, 0] and c[1, 1].tolist() == [0, 0, 2, 0]
    assert c.sum() == 12


def test_packed_programs_on_aer_match_exact_distribution():
    """Circuits packed with measurement and reset give the same statistics as separate circuits."""
    pytest.importorskip("pytket.extensions.qiskit")
    t = sim.layered_ansatz(3, 2)
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(7, t.n_data))
    th = rng.uniform(-np.pi, np.pi, size=t.n_params)
    A = t.angles(X, th)
    shots = 4000
    req = Request(DeviceProfile(noisy=False), t, A, shots, split=2)
    ex = AerExecutor(segments=3, seed=1)
    ex.run([req])
    assert req.counts.shape == (2, 7, t.dim)
    assert req.counts.sum(axis=(0, 2)).tolist() == [shots] * 7
    p = np.abs(sim.statevector(t, A)) ** 2
    freq = req.counts.sum(axis=0) / shots
    assert np.abs(freq - p).max() < 5 * np.sqrt(0.25 / shots)
    assert ex.n_programs == 3 and ex.n_circuits == 7
    # echo circuits return to |0...0> without noise
    e = t.echo()
    req2 = Request(DeviceProfile(noisy=False), e, e.angles(X[:3], th), 50)
    ex.run([req2])
    assert (req2.counts[0, :, 0] == 50).all()


def test_long_programs_use_registers_of_at_most_64_bits():
    """A program with more than 64 classical bits is split over several registers and parsed in order."""
    pytest.importorskip("pytket.extensions.qiskit")
    from qfedqgt.nexus import MAX_REGISTER_WIDTH, program_bits

    t = sim.layered_ansatz(4, 1)
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(20, t.n_data))
    th = rng.uniform(-np.pi, np.pi, size=t.n_params)
    A = t.angles(X, th)
    prog = build_program(t, A, "long")
    widths = sorted(reg.size for reg in prog.c_registers)
    assert widths == [16, 64] and max(widths) <= MAX_REGISTER_WIDTH
    assert len(program_bits(t.n, 20)) == 80 and len(set(program_bits(t.n, 20))) == 80
    shots = 3000
    req = Request(DeviceProfile(noisy=False), t, A, shots)
    ex = AerExecutor(segments=20, seed=2)
    ex.run([req])
    assert ex.n_programs == 1 and req.counts.shape == (1, 20, t.dim)
    p = np.abs(sim.statevector(t, A)) ** 2
    # every segment, including those stored in the second register, has its own distribution
    assert np.abs(req.counts[0] / shots - p).max() < 5 * np.sqrt(0.25 / shots)


def test_cost_formula():
    t = sim.layered_ansatz(4, 2)
    # 24 one-qubit gates and 6 ZZ gates per circuit; 4 preparations, 4 measurements per segment
    assert program_cost(t, 1, 100) == pytest.approx(5 + 100 * (24 + 60 + 5 * 8) / 5000)
    assert DeviceProfile(gate_scale=40.0).error_params() == dict(p1_scale=40.0, p2_scale=40.0, memory_scale=40.0)
    assert DeviceProfile().error_params() is None
