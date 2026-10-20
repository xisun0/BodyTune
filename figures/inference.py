"""Fresh checkpoint inference on the first circle in the saved test split."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from roboscore.kinematics import Arm
from roboscore.models import ModelConfig, MotionModel, features
from roboscore.train import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "inference"


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "outputs/fixed32/test_rollouts.npz", allow_pickle=False) as data:
        index = int(np.flatnonzero(data["family"] == "circle")[0])
        a = torch.from_numpy(data["q_a"][index : index + 1].copy())
        target = torch.from_numpy(data["q_target"][index : index + 1].copy())
    results, trajectories = {}, [a[0].numpy()]
    for mode in ("fixed32", "eos32"):
        checkpoint = torch.load(ROOT / f"outputs/{mode}/best.pt", weights_only=True)
        config = ModelConfig(**checkpoint["config"])
        model = MotionModel(config)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.inference_mode():
            _, score = model.notator(features(a, config.dt))
            # Receiver input is exclusively integer score + its initial state.
            q = model.decode(score, target[:, 0])
        message = score[0][score[0] >= 0].tolist()
        results[mode] = {"message": message, "metrics": evaluate(q, target, score, config)}
        trajectories.append(q[0].numpy())
    report = {
        "test_index": index,
        "shape": "circle",
        "selection": "first circle, not best-error selection",
        "a_initial": a[0, 0].tolist(),
        "b_initial": target[0, 0].tolist(),
        "models": results,
        "note": "Fresh hard inference. No optimizer. Target used only for evaluation/plots.",
    }
    (OUT / "example.json").write_text(json.dumps(report, indent=2) + "\n")
    arm = Arm()
    target_xy = arm.fk(a[0].numpy())
    relative = target_xy - target_xy[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, mode, q in zip(axes, ("fixed32", "eos32"), trajectories[1:]):
        xy = arm.fk(q)
        xy -= xy[0].copy()
        ax.plot(*relative.T, "k-", label="A: target circle")
        ax.plot(*xy.T, "C0--", label="B: learned execution")
        error = results[mode]["metrics"]["mean_nrmse"]
        ax.set(title=f"{mode}: NRMSE={error:.4f}", xlabel="Relative x", ylabel="Relative y")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "inference.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    chains = [arm.joints(q) for q in trajectories]
    artists = []
    for ax, chain, title in zip(
        axes, chains, ["A: demonstrated circle", "B: fixed32 inference", "B: EOS inference"]
    ):
        ax.plot(*(relative + chain[0, -1]).T, color=".7", label="Target")
        (line,) = ax.plot([], [], "o-", color="#cf731b", lw=3)
        (trail,) = ax.plot([], [], color="#1976a5", lw=1.5)
        ax.set(title=title, xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), xlabel="World x", ylabel="World y")
        ax.set_aspect("equal")
        artists.append((line, trail))
    clock = fig.suptitle("")

    def update(t):
        clock.set_text(
            f"Learned inference | t={t * 0.05:.2f} s | orange: full arm, blue: pen trace"
        )
        for chain, (line, trail) in zip(chains, artists):
            line.set_data(*chain[t].T)
            trail.set_data(*chain[: t + 1, -1].T)
        return [clock, *[a for group in artists for a in group]]

    animation = FuncAnimation(fig, update, frames=len(a[0]), interval=50)
    animation.save(OUT / "inference.gif", writer=PillowWriter(fps=20), dpi=90)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
