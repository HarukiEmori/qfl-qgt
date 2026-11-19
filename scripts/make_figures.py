#!/usr/bin/env python
"""Figures and table rows of the paper from the files in ``results/``.

    python scripts/make_figures.py geometry            # direct evaluation of the geometry
    python scripts/make_figures.py training            # accuracy and loss against the round
    python scripts/make_figures.py sweep               # final accuracy against the noise of the clients
    python scripts/make_figures.py emulator [dir ...]  # training on the emulator (default: every finished campaign)
    python scripts/make_figures.py table <dir> ...     # LaTeX rows of a results table
    python scripts/make_figures.py all

A results directory is a sub-directory of ``results/`` written by ``run_campaign.py``
or ``run_federated.py``.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.ticker           # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qfedqgt import geometry as geo   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
PDF_METADATA = {"CreationDate": None}      # no time stamp, so that an unchanged figure gives an unchanged file
STYLE = {
    "QFedAvg": dict(color="#1f77b4", ls="--", marker="s", label="QFedAvg"),
    "QFedFisher": dict(color="#2ca02c", ls="-.", marker="^", label="QFedFisher"),
    "QFedQGT": dict(color="#d62728", ls="-", marker="o", label="QFedQGT"),
    "QFedAvg-shots": dict(color="#7f7f7f", ls=":", marker="v", label="QFedAvg (shot weights)"),
    "QFedQGT-invvar": dict(color="#ff7f0e", ls="-", marker="D", label="QFedQGT ($\\zeta=0$)"),
    "QNG+data": dict(color="#9467bd", ls="--", marker="x", label="Pure-state QNG, data shares"),
    "Bures+data": dict(color="#8c564b", ls=":", marker="+", label="Bures QNG, data shares"),
    "Euclid+re": dict(color="#17becf", ls="-.", marker="*", label="Plain gradient, precision weights"),
}
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42, "figure.dpi": 150})


def load(directory: str) -> dict:
    """{method: {key: array (seeds, ...)}} from the seed files of a results directory."""
    path = directory if os.path.isdir(directory) else os.path.join(ROOT, "results", directory)
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "seed*.npz"))):
        z = np.load(f)          # numeric arrays only; pickled objects are not accepted
        for key in z.files:
            m, k = key.split("/")
            out.setdefault(m, {}).setdefault(k, []).append(z[key])
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in out.items()}


def geometry() -> None:
    with open(os.path.join(ROOT, "results", "geometry", "geometry.json")) as f:
        g = json.load(f)
    rows = [r for r in g["rows"] if r["model"] == "gate"]
    N = 2 ** g["n_qubits"]
    p = np.array([r["p_echo"] for r in rows])
    pp = np.linspace(0, max(0.9, p.max()), 200)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
    ax[0].plot(pp, geo.kappa(pp, N), "k-", lw=1, label=r"$\kappa(p)$")
    ax[0].plot(p, [r["g_fit"] for r in rows], "o", color="#d62728", ms=4, label="Gate-level noise")
    ax[0].set_xlabel(r"Estimated noise strength $\widehat{p}$"); ax[0].set_ylabel("Attenuation of the Bures metric")
    ax[0].legend(frameon=False, loc="upper right", handlelength=1.5); ax[0].set_title("(a)", loc="left", fontsize=9)
    ax[1].plot(pp, geo.kappa_omega(pp, N), "k-", lw=1, label=r"$\kappa_\omega(p)$")
    ax[1].plot(p, [r["u_fit"] for r in rows], "s", color="#1f77b4", ms=4, label="Gate-level noise")
    ax[1].set_xlabel(r"Estimated noise strength $\widehat{p}$"); ax[1].set_ylabel("Attenuation of the curvature")
    ax[1].legend(frameon=False, loc="upper right", handlelength=1.5); ax[1].set_title("(b)", loc="left", fontsize=9)
    r0 = np.mean([r["R0"] for r in rows])
    ax[2].plot(pp, geo.chi(pp, N) * r0, "k-", lw=1, label=r"$[\kappa_\omega/\kappa]\,\mathcal{R}$")
    ax[2].plot(p, [r["R"] for r in rows], "^", color="#2ca02c", ms=4, label=r"Exact $\mathcal{R}_k$")
    ax[2].plot(pp, pp ** 2, "k:", lw=1, label=r"$p^2$")
    ax[2].set_xlabel(r"Estimated noise strength $\widehat{p}$"); ax[2].set_ylabel("Incompatibility ratio")
    ax[2].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42), handlelength=1.5); ax[2].set_title("(c)", loc="left", fontsize=9)
    for a in ax:
        a.set_xlim(-0.02, pp.max() + 0.02); a.set_ylim(-0.03, 1.05)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "geometry.pdf"), metadata=PDF_METADATA)
    print("wrote figures/geometry.pdf")


PANELS = (("sim-shots_and_noise", "(a) Scenario I"),
          ("sim-noise_only_x500", "(b) Scenario II, $s=500$"),
          ("sim-noise_only_x1", "(c) Scenario II, $s=1$"))
SWEEP = (1, 50, 150, 300, 500)
TAIL = 10


def training(panels=PANELS, key_acc="acc_shots", out="training.pdf", methods=("QFedAvg", "QFedFisher", "QFedQGT")) -> None:
    """Accuracy on the reference device and exact loss against the round (mean and standard error over seeds)."""
    fig, ax = plt.subplots(2, len(panels), figsize=(2.35 * len(panels), 4.0), squeeze=False, sharex=True)
    for j, (d, title) in enumerate(panels):
        res = load(d)
        for m in methods:
            if m not in res:
                continue
            h = res[m]
            rounds = h["round"][0]
            for i, (key, ylabel) in enumerate(((key_acc, "Test accuracy"), ("loss_exact", "Test loss"))):
                y = np.asarray(h[key], float)
                mean = np.nanmean(y, axis=0)
                st = STYLE[m]
                ax[i, j].plot(rounds, mean, color=st["color"], ls=st["ls"], lw=1.3, label=st["label"])
                if y.shape[0] > 1:
                    se = np.nanstd(y, axis=0, ddof=1) / np.sqrt(y.shape[0])
                    ax[i, j].fill_between(rounds, mean - se, mean + se, color=st["color"], alpha=0.18, lw=0)
                if j == 0:
                    ax[i, j].set_ylabel(ylabel)
        ax[1, j].set_xlabel("Communication round")
        ax[0, j].set_title(title, loc="left", fontsize=9)
        ax[0, j].set_ylim(0.45, 1.0)
        ax[1, j].set_ylim(0.15, 0.75)
    ax[0, 0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, out), metadata=PDF_METADATA)
    print(f"wrote figures/{out}")


def sweep(scales=SWEEP, out="sweep.pdf", methods=("QFedAvg", "QFedFisher", "QFedQGT")) -> None:
    """Final accuracy in Scenario II against the error-rate multiplier of the noisy clients."""
    with open(os.path.join(ROOT, "results", "geometry", "geometry.json")) as f:
        rows = {r["scale"]: r for r in json.load(f)["rows"] if r["model"] == "gate"}
    x = np.arange(len(scales))
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for m in methods:
        mean, se = [], []
        for s in scales:
            y = np.asarray(load(f"sim-noise_only_x{s}")[m]["acc_shots"], float)[:, -TAIL:].mean(axis=1)
            mean.append(y.mean()); se.append(y.std(ddof=1) / np.sqrt(len(y)))
        st = STYLE[m]
        ax.errorbar(x, mean, yerr=se, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, lw=1.2,
                    capsize=2, label=st["label"])
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in scales])
    ax.set_xlabel("Error-rate multiplier $s$ of the noisy clients"); ax.set_ylabel("Final test accuracy")
    ax.set_ylim(0.6, 1.0)
    top = ax.secondary_xaxis("top")
    top.set_xticks(x); top.set_xticklabels([f"{rows[s]['p_echo']:.2f}" for s in scales])
    top.set_xlabel(r"Noise strength $\widehat{p}$")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, out), metadata=PDF_METADATA)
    print(f"wrote figures/{out}")


EMULATOR_PANELS = (("emulator_iid", "(a) IID data"), ("emulator_dirichlet", "(b) Dirichlet partition"))


def campaign_dirs() -> list:
    """Finished campaigns of the emulator experiment (folders of ``results/`` with a summary)."""
    from qfedqgt.scenarios import EMULATOR_CAMPAIGNS
    return [d for d in EMULATOR_CAMPAIGNS if os.path.exists(os.path.join(ROOT, "results", d, "summary.json"))]


def load_runs(directories, scenario: str) -> dict:
    """{method: {key: array (seeds, ...)}} from the run files ``<scenario>-seed<k>.npz`` of scripts/run_emulator.py.

    ``directories`` is one results folder or a list of them (the campaigns of an experiment).
    """
    if isinstance(directories, str):
        directories = [directories]
    out = {}
    for directory in directories:
        path = directory if os.path.isdir(directory) else os.path.join(ROOT, "results", directory)
        for f in sorted(glob.glob(os.path.join(path, f"{scenario}-seed*.npz"))):
            z = np.load(f)
            for key in z.files:
                m, k = key.split("/")
                out.setdefault(m, {}).setdefault(k, []).append(z[key])
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in out.items()}


def emulator(directories=None, out="emulator.pdf", methods=("QFedAvg", "QFedFisher", "QFedQGT")) -> None:
    """Training on the emulator: accuracy of the global model against the round.

    ``directories``: results folders of the campaigns (default: every finished campaign of
    ``qfedqgt.scenarios.EMULATOR_CAMPAIGNS``).

    Lines: accuracy of the noiseless circuit with the parameters trained on the emulator (mean over the
    seeds, standard error as a band).  Markers: accuracy measured on the emulator for the models of the
    last rounds (mean over the seeds, standard error as a bar).
    """
    fig, ax = plt.subplots(1, len(EMULATOR_PANELS), figsize=(3.3 * len(EMULATOR_PANELS), 2.7), squeeze=False, sharey=True)
    for j, (scenario, title) in enumerate(EMULATOR_PANELS):
        res = load_runs(directories or campaign_dirs(), scenario)
        for m in methods:
            if m not in res:
                continue
            h, st = res[m], STYLE[m]
            y = np.asarray(h["acc_exact"], float)
            rounds = np.arange(y.shape[1])
            mean = y.mean(axis=0)
            ax[0, j].plot(rounds, mean, color=st["color"], ls=st["ls"], lw=1.3, label=st["label"])
            if y.shape[0] > 1:
                se = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
                ax[0, j].fill_between(rounds, mean - se, mean + se, color=st["color"], alpha=0.18, lw=0)
            if "acc_device" in h and np.asarray(h["acc_device"]).size:
                d = np.asarray(h["acc_device"], float)
                x = np.asarray(h["eval_rounds"][0], float)
                err = d.std(axis=0, ddof=1) / np.sqrt(d.shape[0]) if d.shape[0] > 1 else None
                ax[0, j].errorbar(x, d.mean(axis=0), yerr=err, color=st["color"], marker=st["marker"], ms=3.5, ls="none",
                                  capsize=1.5, lw=0.8, mfc="white")
        ax[0, j].axhline(0.25, color="0.6", lw=0.6, ls=":")
        ax[0, j].set_xlabel("Communication round")
        ax[0, j].set_title(title, loc="left", fontsize=9)
        ax[0, j].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax[0, 0].set_ylabel("Test accuracy")
    ax[0, 0].legend(frameon=False, loc="upper left")
    ax[0, 0].set_ylim(0.15, 0.6)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, out), metadata=PDF_METADATA)
    fig.savefig(os.path.join(FIG, os.path.splitext(out)[0] + ".png"), dpi=200)      # shown by the notebook
    print(f"wrote figures/{out} and figures/{os.path.splitext(out)[0]}.png")


def table(dirs, tail=TAIL, methods=("QFedAvg", "QFedFisher", "QFedQGT")) -> None:
    """Final accuracy on the reference device, exact accuracy and exact loss (mean and standard error)."""
    for d in dirs:
        res = load(d)
        print(f"% {d}")
        for m in methods:
            if m not in res:
                continue
            h = res[m]

            def stat(key):
                y = np.asarray(h[key], float)[:, -tail:].mean(axis=1)
                se = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else float("nan")
                return y.mean(), se
            a, ase = stat("acc_shots"); e, ese = stat("acc_exact"); l, lse = stat("loss_exact")
            print(f"{m} & ${a:.3f}\\pm{ase:.3f}$ & ${e:.3f}\\pm{ese:.3f}$ & ${l:.3f}\\pm{lse:.3f}$ \\\\"
                  f"  % seeds={np.asarray(h['acc_exact']).shape[0]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "geometry":
        geometry()
    elif cmd == "training":
        training()
    elif cmd == "sweep":
        sweep()
    elif cmd == "emulator":
        emulator(sys.argv[2:] or None)
    elif cmd == "table":
        table(sys.argv[2:])
    elif cmd == "all":
        geometry(); training(); sweep()
        if campaign_dirs():
            emulator()
