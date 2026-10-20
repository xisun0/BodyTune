"""Animate actual sampled arm configurations, not just the pen trajectory."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from roboscore.kinematics import Arm

OUT = Path(__file__).resolve().parent


def main():
    source = Path(os.environ.get("ROBOSCORE_RUN", "outputs/baseline"))
    report = json.loads((source / "report.json").read_text())
    arm = Arm(**report["config"]["arm"])
    dt = report["config"]["dt"]
    with np.load(source / "trajectories.npz", allow_pickle=False) as data:
        # A circle makes continuous posture changes easy to see.
        indices = np.flatnonzero(data["family"] == "circle")
        i = next(int(i) for i in indices if np.isfinite(data["joint_copy"][i]).all())
        qs = [data[key][i].copy() for key in ("q_a", "ik_oracle", "joint_copy")]
        relative = data["relative"][i].copy()
    chains = [arm.joints(q) for q in qs]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    artists = []
    radius = sum(arm.lengths) + 0.1
    for ax, chain, title in zip(
        axes, chains, ["A: demonstration", "B: full-curve IK", "B: joint copy"]
    ):
        target = relative + chain[0, -1]
        ax.plot(*target.T, color=".7", lw=1, label="Target drawing")
        (trail,) = ax.plot([], [], color="#1678a5", lw=1.5, label="Executed pen path")
        (links,) = ax.plot([], [], "o-", color="#d77918", lw=4, markersize=7, label="Arm + joints")
        label = ax.text(0.03, 0.03, "", transform=ax.transAxes, fontsize=9)
        ax.set(
            title=title,
            xlabel="World x (link units)",
            ylabel="World y (link units)",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )
        ax.set_aspect("equal")
        artists.append((trail, links, label))
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    clock = fig.suptitle("")

    def update(t):
        clock.set_text(f"Circle | sample {t + 1}/{len(qs[0])} | t = {t * dt:.2f} s")
        for chain, q, (trail, links, label) in zip(chains, qs, artists):
            links.set_data(*chain[t].T)
            trail.set_data(*chain[: t + 1, -1].T)
            label.set_text(f"q1 = {q[t, 0]:+.2f}, q2 = {q[t, 1]:+.2f} rad")
        return [clock, *[a for group in artists for a in group]]

    update(len(qs[0]) // 2)
    fig.savefig(OUT / "posture_midpoint.png", dpi=150)
    animation = FuncAnimation(fig, update, frames=len(qs[0]), interval=dt * 1000, blit=False)
    animation.save(OUT / "arm_motion.gif", writer=PillowWriter(fps=round(1 / dt)), dpi=100)
    # Self-contained browser playback includes pause, scrub and speed controls.
    (OUT / "arm_motion.html").write_text(
        animation.to_jshtml(fps=round(1 / dt), default_mode="loop")
    )
    plt.close(fig)
    print(f"Rendered pair {i}, {len(qs[0])} sampled postures, dt={dt}s")


if __name__ == "__main__":
    main()
