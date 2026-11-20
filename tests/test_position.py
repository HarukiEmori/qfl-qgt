"""Statistics of the survival against the position of a circuit in a packed program (scripts/nexus_position.py)."""
import importlib.util
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("nexus_position", os.path.join(ROOT, "scripts", "nexus_position.py"))
nexus_position = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nexus_position)


def test_flat_survival_gives_no_trend():
    rng = np.random.default_rng(1)
    changes = []
    for _ in range(200):
        r = nexus_position.position_trend(rng.binomial(40, 0.8, size=300), 40)
        changes.append(r["change_first_to_last"] / r["change_se"])
    changes = np.array(changes)
    # the standardized change has mean zero and unit variance when the survival does not depend on the position
    assert abs(changes.mean()) < 0.25 and 0.8 < changes.std() < 1.2
    assert abs(r["survival"] - 0.8) < 5 * r["survival_se"]


def test_injected_trend_is_recovered():
    rng = np.random.default_rng(2)
    n, shots, drop = 400, 200, 0.10
    p = 0.85 - drop * np.linspace(0.0, 1.0, n)              # the survival falls by 0.10 along the program
    r = nexus_position.position_trend(rng.binomial(shots, p), shots)
    assert abs(r["change_first_to_last"] + drop) < 4 * r["change_se"]
    assert r["change_first_to_last"] / r["change_se"] < -5
    assert r["quarters"][0] > r["quarters"][-1]
    assert abs(r["quarters"][0] - (0.85 - drop / 8)) < 5 * r["quarter_se"]


def test_evaluation_blocks_are_class_balanced_and_disjoint():
    spec2 = importlib.util.spec_from_file_location("run_emulator", os.path.join(ROOT, "scripts", "run_emulator.py"))
    run_emulator = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(run_emulator)
    y = np.array([0, 1] * 50)
    blocks = [run_emulator.eval_indices(y, 2, 10, b) for b in range(5)]
    for idx in blocks:
        assert len(idx) == 20 and (y[idx] == 0).sum() == 10 and (y[idx] == 1).sum() == 10
    assert len(np.unique(np.concatenate(blocks))) == 100           # five blocks cover all images once
    # block 0 is the first images of every class (the evaluation of the four-class experiment)
    assert np.array_equal(blocks[0], np.sort(np.concatenate([np.flatnonzero(y == c)[:10] for c in (0, 1)])))
