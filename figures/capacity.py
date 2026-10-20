"""Compare exact-diameter GRU widths on identical held-out circles."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboscore.diagnostics import circle_quality
from roboscore.kinematics import Arm


def main():
    root = Path(os.environ.get("ROBOSCORE_CAPACITY", "outputs/diameter_capacity"))
    reports, paths, curves = {}, {}, {}
    reference = None
    for hidden in (64, 128, 256):
        directory = root / f"gru{hidden}"
        report = json.loads((directory / "report.json").read_text())
        reports[str(hidden)] = report
        if reference is None:
            reference = report["data"]["sha256"]
        if reference != report["data"]["sha256"]:
            raise ValueError("capacity models used different source files")
        with np.load(directory / "test_rollouts.npz", allow_pickle=False) as data:
            diameter = data["diameter"].copy()
            target = data["q_target"].copy()
            paths[hidden] = data["q_pred"].copy()
        if hidden == 64:
            d0, target0 = diameter.copy(), target.copy()
        elif not np.array_equal(diameter, d0) or not np.array_equal(target, target0):
            raise ValueError("test examples differ across widths")
        curves[hidden] = np.genfromtxt(directory / "history.csv", delimiter=",", names=True)
    diagnostics = {str(h): circle_quality(q, d0) for h, q in paths.items()}
    comparison = {
        "data_sha256": reference,
        "models": reports,
        "circle_diagnostics": diagnostics,
        "limits": "Single seed, matched epoch rules not FLOPs; fits are descriptive, not success metrics.",
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for hidden, history in curves.items():
        ax.plot(history["epoch"], history["val_nrmse"], label=f"GRU {hidden}")
    ax.set(xlabel="Epoch", ylabel="Validation mean NRMSE", yscale="log")
    ax.legend(frameon=False)
    fig.savefig(root / "validation_curves.png", dpi=160)
    fig.savefig(root / "validation_curves.pdf")
    plt.close(fig)
    target = Arm().fk(target0)
    target -= target[:, :1].copy()
    actual = {h: Arm().fk(q) for h, q in paths.items()}
    for xy in actual.values():
        xy -= xy[:, :1].copy()
    indices = np.argsort(d0)[np.linspace(0, len(d0) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, i in zip(axes.flat, indices):
        ax.plot(*target[i].T, color="black", lw=1.8, label="Target")
        for hidden, xy in actual.items():
            ax.plot(*xy[i].T, "--", lw=1.1, label=f"GRU {hidden}")
        ax.set(title=f"Diameter = {d0[i]:.3f}", xlabel="Relative x", ylabel="Relative y")
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span = max(x1 - x0, y1 - y0)
        ax.set_xlim((x0 + x1 - span) / 2, (x0 + x1 + span) / 2)
        ax.set_ylim((y0 + y1 - span) / 2, (y0 + y1 + span) / 2)
        ax.set_aspect("equal", adjustable="box")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.savefig(root / "capacity_inference.png", dpi=160)
    fig.savefig(root / "capacity_inference.pdf")
    plt.close(fig)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
