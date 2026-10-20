"""Plot preselected first-per-family learned rollouts and validation history."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboscore.kinematics import Arm

OUT = Path(__file__).resolve().parent


def main():
    source = Path(os.environ.get("ROBOSCORE_RUN", "outputs/fixed32"))
    with np.load(source / "test_rollouts.npz", allow_pickle=False) as data:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
        for ax, family in zip(axes.flat, dict.fromkeys(data["family"])):
            i = np.flatnonzero(data["family"] == family)[0]
            for key, label, style in [
                ("q_target", "Target", "k-"),
                ("q_pred", "Learned B", "C0--"),
            ]:
                p = Arm().fk(data[key][i])
                p -= p[0].copy()
                ax.plot(*p.T, style, label=label)
            length = np.sum((data["scores"][i] >= 0) & (data["scores"][i] < 8))
            ax.set(title=f"{family}: tokens={length}", xlabel="Relative x", ylabel="Relative y")
            # Expand the shorter coordinate span so narrow curves keep readable axes.
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            span = max(x1 - x0, y1 - y0)
            ax.set_xlim((x0 + x1 - span) / 2, (x0 + x1 + span) / 2)
            ax.set_ylim((y0 + y1 - span) / 2, (y0 + y1 + span) / 2)
            ax.set_aspect("equal", adjustable="box")
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.savefig(source / "drawings.png", dpi=150)
        fig.savefig(source / "drawings.pdf")
        plt.close(fig)
    history = np.genfromtxt(source / "history.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(np.atleast_1d(history["epoch"]), np.atleast_1d(history["val_nrmse"]))
    ax.set(xlabel="Epoch", ylabel="Validation mean NRMSE")
    fig.savefig(source / "learning_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
