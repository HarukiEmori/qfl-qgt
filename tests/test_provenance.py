"""Run records: content, automatic display, and the saved file."""
import json

import pytest

from qfedqgt import provenance
from qfedqgt.federated import LocalExecutor


def test_record_is_shown_and_saved(tmp_path, capsys):
    ex = LocalExecutor()
    with provenance.record_run("unit test", entry="tests/test_provenance.py", executor=ex,
                               parameters=dict(shots=10), save_to=str(tmp_path)) as rec:
        ex.n_circuits, ex.n_shots = 3, 30
    shown = capsys.readouterr().out
    assert "RUN RECORD" in shown and "unit test" in shown and "local simulation" in shown
    assert rec.started and rec.finished and rec.usage["n_shots"] == 30
    saved = json.load(open(tmp_path / "provenance.json"))
    assert saved["task"] == "unit test" and saved["parameters"] == {"shots": 10}
    assert set(saved["program"]) == {"repository", "commit", "branch", "modified"}
    again = provenance.load_record(str(tmp_path / "provenance.json"))
    assert again.text() == rec.text() and "<table>" in again._repr_html_()


def test_record_survives_an_error_and_has_no_local_paths(tmp_path, capsys):
    with pytest.raises(RuntimeError):
        with provenance.record_run("failing run", save_to=str(tmp_path / "out.json")) as rec:
            raise RuntimeError("stop")
    text = capsys.readouterr().out
    assert "ended with an error" in text
    saved = open(tmp_path / "out.json").read()
    assert "/Users/" not in saved and "/home/" not in saved
    assert "@" not in (rec.program.get("repository") or "")


def test_job_summary_reports_devices_times_and_commits():
    rec = provenance.RunRecord(task="t", jobs=[
        dict(job_id="a", device="reimei-E", error_scale=200.0, programs=2, submitted="2026-11-19T10:00:00+0900",
             completed="2026-11-19T10:05:00+0900", cost=12.5, commit="0123456789abcdef", source="cache"),
        dict(job_id="b", device="reimei-E", error_scale=1.0, programs=1, submitted="2026-11-19T09:00:00+0900",
             completed="2026-11-19T09:02:00+0900", cost=float("nan"), commit=None, source="submitted")])
    text = rec.text()
    assert "reimei-E x1, reimei-E x200" in text and "2026-11-19T09:00:00+0900" in text
    assert "0123456789" in text and "not recorded" in text and "12.5" in text


def test_error_notes_do_not_contain_absolute_paths(tmp_path, capsys):
    import os

    with pytest.raises(RuntimeError):
        with provenance.record_run("failing run", save_to=str(tmp_path / "out.json")) as rec:
            raise RuntimeError(f"missing file {os.path.join(provenance.ROOT, 'results', 'x.npz')} in {os.path.expanduser('~')}")
    capsys.readouterr()
    assert provenance.ROOT not in rec.notes[0] and os.path.expanduser("~") not in rec.notes[0]
    assert "results/x.npz" in rec.notes[0]


def test_replay_keeps_stored_results(tmp_path, capsys):
    """A replay from the cache compares with stored files and never overwrites them."""
    import numpy as np

    from qfedqgt import provenance as pv

    path = str(tmp_path / "result.json")
    assert pv.write_json(path, dict(a=1.0, seconds=3.0)) == "written"
    assert pv.write_json(path, dict(a=1.0, seconds=9.0), replay=True, ignore=("seconds",)) == "reproduced"
    assert pv.write_json(path, dict(a=2.0, seconds=9.0), replay=True, ignore=("seconds",)) == "differs"
    with open(path) as f:
        assert json.load(f) == dict(a=1.0, seconds=3.0)                      # the stored file is untouched
    arrays = str(tmp_path / "run.npz")
    assert pv.write_arrays(arrays, {"m/x": np.arange(4.0)}) == "written"
    assert pv.write_arrays(arrays, {"m/x": np.arange(4.0)}, replay=True) == "reproduced"
    assert pv.write_arrays(arrays, {"m/x": np.arange(4.0) + 1}, replay=True) == "differs"
    assert np.array_equal(np.load(arrays)["m/x"], np.arange(4.0))
    record = str(tmp_path / "provenance.json")
    with pv.record_run("first", save_to=record, show=False):
        pass
    with pv.record_run("second", save_to=record, show=False, replay=True) as rec:
        pass
    assert pv.load_record(record).task == "first"
    assert any("stored record" in note for note in rec.notes)
