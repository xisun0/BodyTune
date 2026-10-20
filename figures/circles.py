"""Preview circle demonstrations and visualize single-token inference."""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from roboscore.kinematics import Arm
from roboscore.models import ModelConfig, MotionModel


def main():
    source = Path(os.environ.get("ROBOSCORE_RUN", "outputs/circles1"))
    with np.load(source / "train_demonstrations.npz", allow_pickle=False) as data:
        diameter = data["diameter"].copy()
        order = np.argsort(diameter)
        chosen = order[np.linspace(0, len(order) - 1, 8).astype(int)]
        fig, axes = plt.subplots(2, 4, figsize=(11, 6), layout="constrained")
        for ax, i in zip(axes.flat, chosen):
            ax.plot(*data["relative"][i].T)
            ax.set(
                title=f"Demo {i}: diameter={diameter[i]:.3f}",
                xlabel="Relative x",
                ylabel="Relative y",
                xlim=(-0.75, 0.05),
                ylim=(-0.4, 0.4),
            )
            ax.set_aspect("equal")
        fig.savefig(source / "training_circles.png", dpi=150)
        plt.close(fig)
        with (source / "training_index.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["demo", "diameter", "a_q1", "a_q2", "b_q1", "b_q2"])
            for i in range(len(diameter)):
                writer.writerow([i, diameter[i], *data["q_a"][i, 0], *data["q_target"][i, 0]])
    with np.load(source / "test_demonstrations.npz", allow_pickle=False) as data:
        diameter = data["diameter"].copy()
    with np.load(source / "test_rollouts.npz", allow_pickle=False) as data:
        ids = data["scores"][:, 0].copy()
        arm = Arm()
        target = arm.fk(data["q_target"])
        actual = arm.fk(data["q_pred"])
        target -= target[:, :1].copy()
        actual -= actual[:, :1].copy()
    error = np.linalg.norm(target - actual, axis=-1)
    length = np.linalg.norm(np.diff(target, axis=1), axis=-1).sum(-1)
    nrmse = np.sqrt(np.mean(error**2, axis=1)) / length
    order = np.argsort(diameter)
    chosen = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, i in zip(axes.flat, chosen):
        ax.plot(*target[i].T, "k-", label="Target")
        ax.plot(*actual[i].T, "C0--", label="Learned B")
        ax.set(
            title=f"d={diameter[i]:.3f}, token={ids[i]}, NRMSE={nrmse[i]:.3f}",
            xlabel="Relative x",
            ylabel="Relative y",
            xlim=(-0.75, 0.05),
            ylim=(-0.4, 0.4),
        )
        ax.set_aspect("equal")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(source / "circle_inference.png", dpi=160)
    fig.savefig(source / "circle_inference.pdf")
    plt.close(fig)
    history = np.genfromtxt(source / "history.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].scatter(diameter, ids, s=12, alpha=0.6)
    axes[0].set(
        xlabel="True circle diameter",
        ylabel="Predicted token ID",
        yticks=range(8),
        ylim=(-0.5, 7.5),
        title="Hard messages on held-out examples",
    )
    axes[1].plot(history["epoch"], history["val_nrmse"])
    axes[1].set(xlabel="Epoch", ylabel="Validation mean NRMSE", title="Training progress")
    fig.savefig(source / "circle_tokens.png", dpi=160)
    plt.close(fig)
    groups = []
    for token in sorted(np.unique(ids).tolist()):
        d = diameter[ids == token]
        groups.append(
            {
                "token": token,
                "count": len(d),
                "diameter_min": float(d.min()),
                "diameter_max": float(d.max()),
                "diameter_mean": float(d.mean()),
                "mean_nrmse": float(nrmse[ids == token].mean()),
            }
        )
    (source / "diameter_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    print(json.dumps(groups, indent=2))
    checkpoint = torch.load(source / "best.pt", weights_only=True)
    model = MotionModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.set_num_threads(2)
    # All eight commands share the same feasible initial state; no demonstration supplied.
    q0 = arm.ik(np.array([[1.0, 0.5], [1.0, 0.5]]), branch=1)[0]
    scores = torch.arange(8, dtype=torch.long).reshape(8, 1)
    initial = torch.tensor(np.tile(q0, (8, 1)), dtype=torch.float32)
    with torch.inference_mode():
        q = model.decode(scores, initial).numpy()
    p = arm.fk(q)
    p -= p[:, :1].copy()
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), layout="constrained")
    for token, ax in enumerate(axes.flat):
        ax.plot(*p[token].T)
        used = "used" if token in ids else "unused in test"
        ax.set(
            title=f"Token {token} ({used})",
            xlabel="Relative x",
            ylabel="Relative y",
            xlim=(-0.75, 0.05),
            ylim=(-0.4, 0.4),
        )
        ax.set_aspect("equal")
    fig.savefig(source / "token_prototypes.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(
        source / "token_prototypes.npz", q=q, scores=scores.numpy(), q0=initial.numpy()
    )


if __name__ == "__main__":
    main()
