"""Render one fixed benchmark run; override input with ROBOSCORE_RUN."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboscore.kinematics import Arm

OUT = Path(__file__).resolve().parent


def main():
    source = Path(os.environ.get("ROBOSCORE_RUN", "outputs/baseline"))
    with np.load(source / "trajectories.npz", allow_pickle=False) as data:
        arm = Arm()
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
        for ax, family in zip(axes.flat, dict.fromkeys(data["family"])):
            indices = np.flatnonzero(data["family"] == family)
            # Prefer a feasible joint-copy example; never plot rejected execution as valid.
            good = [i for i in indices if np.isfinite(data["joint_copy"][i]).all()]
            i = good[0] if good else indices[0]
            target = data["relative"][i]
            ax.plot(*target.T, color="black", lw=2, label="A target")
            for method, label, style in [
                ("ik_oracle", "B: full-curve IK", "--"),
                ("joint_copy", "B: joint copy", ":"),
            ]:
                q = data[method][i]
                if np.isfinite(q).all():
                    p = arm.fk(q)
                    p -= p[0].copy()
                    ax.plot(*p.T, style, lw=1.5, label=label)
            ax.set(
                title=str(family),
                xlabel="Relative x (link units)",
                ylabel="Relative y (link units)",
            )
            ax.set_aspect("equal", adjustable="datalim")
        axes.flat[0].legend(fontsize=8, frameon=False)
        fig.savefig(OUT / "baseline.png", dpi=180)
        fig.savefig(OUT / "baseline.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        for ax, key, title in zip(axes, ["q_a", "ik_oracle"], ["Robot A", "Robot B (IK oracle)"]):
            q = data[key][0]
            p = arm.fk(q)
            ax.plot(*p.T, label="Pen path")
            for idx, style in [(0, "-"), (-1, "--")]:
                elbow = np.array([np.cos(q[idx, 0]), np.sin(q[idx, 0])])
                chain = np.vstack([[0, 0], elbow, p[idx]])
                ax.plot(*chain.T, style, marker="o", label="Start" if idx == 0 else "End")
            ax.set(
                title=title,
                xlabel="World x (link units)",
                ylabel="World y (link units)",
                xlim=(-2.1, 2.1),
                ylim=(-2.1, 2.1),
            )
            ax.set_aspect("equal")
            ax.legend(frameon=False, fontsize=8)
        fig.savefig(OUT / "arms.png", dpi=180)
        fig.savefig(OUT / "arms.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
