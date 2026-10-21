"""Inspect integer VQ scores and executed circles, with earlier local baseline."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from roboscore.aligned import load_splits
from roboscore.kinematics import Arm
from roboscore.vq import VQConfig, VQModel


def main():
    root = Path("outputs/tcn_vq")
    report = json.loads((root / "report.json").read_text())
    with np.load(root / "test_rollouts.npz") as data:
        motion, ids = data["motion"], data["scores"]
        target_q, q = data["q_target"], data["q_pred"]
    with np.load("outputs/circles1/test_demonstrations.npz") as data:
        diameter = data["diameter"].copy()
    # Export a stable integer corpus for a future language-model stage.
    splits, hashes = load_splits(report["data"]["source"], "local")
    if hashes != report["data"]["sha256"]:
        raise ValueError("source hashes changed")
    checkpoint = torch.load(root / "best.pt", weights_only=True)
    torch.set_num_threads(4)
    model = VQModel(VQConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        for name, (observed, _) in splits.items():
            scores = torch.cat([model.encode(batch) for batch in observed.split(64)]).numpy()
            np.savez_compressed(root / f"{name}_scores.npz", scores=scores)
            if name == "test":
                assert np.array_equal(scores, ids)
    baseline = Path("outputs/aligned128/local95")
    old_report = json.loads((baseline / "report.json").read_text())
    if old_report["data"]["sha256"] != report["data"]["sha256"]:
        raise ValueError("source hashes differ")
    with np.load(baseline / "test_rollouts.npz") as data:
        assert np.array_equal(target_q, data["q_target"])
        old_q = data["q_pred"].copy()
    paths = [Arm().fk(values) for values in (target_q, old_q, q)]
    for path in paths:
        path -= path[:, :1].copy()
    selected = np.argsort(diameter)[np.linspace(0, len(diameter) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, index in zip(axes.flat, selected):
        for path, style, label in zip(
            paths, ("k-", "--", "--"), ("Target", "MLP local95", "TCN + VQ")
        ):
            ax.plot(*path[index].T, style, label=label, lw=1.5)
        ax.set(title=f"Diameter {diameter[index]:.3f}", xlabel="Relative x", ylabel="Relative y")
        ax.set_aspect("equal", adjustable="box")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(root / "inference.png", dpi=160)
    fig.savefig(root / "inference.pdf")
    plt.close(fig)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:8])
    norm = BoundaryNorm(np.arange(-0.5, 8.5), 8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].scatter(*motion.reshape(-1, 2).T, c=ids.flatten(), s=3, cmap=cmap, norm=norm, alpha=0.5)
    axes[0].set(
        title="TCN/VQ IDs (15-step causal context)",
        xlabel="Endpoint dx per step",
        ylabel="Endpoint dy per step",
    )
    axes[0].set_aspect("equal")
    image = axes[1].imshow(
        ids[selected], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm
    )
    axes[1].set(
        xlabel="Action index (0–94)",
        ylabel="Target diameter",
        yticks=range(6),
        yticklabels=[f"{diameter[i]:.3f}" for i in selected],
    )
    fig.colorbar(image, ax=axes, ticks=range(8), label="VQ code ID")
    fig.savefig(root / "symbols.png", dpi=160)
    plt.close(fig)
    history = np.genfromtxt(root / "history.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot(history["epoch"], history["val_nrmse"])
    axes[0].set(xlabel="Joint epoch", ylabel="Validation NRMSE", yscale="log")
    axes[1].plot(history["epoch"], history["val_codes"], label="Used codes")
    axes[1].plot(history["epoch"], history["val_perplexity"], label="Code perplexity")
    axes[1].set(xlabel="Joint epoch", ylim=(0, 8.5))
    axes[1].legend(frameon=False)
    fig.savefig(root / "training.png", dpi=160)
    plt.close(fig)
    statistics = []
    for code in range(8):
        values = motion[ids == code]
        statistics.append(
            {
                "id": code,
                "count": len(values),
                "mean_dx_dy": values.mean(0).tolist() if len(values) else None,
                "mean_step_length": float(np.linalg.norm(values, axis=-1).mean())
                if len(values)
                else None,
            }
        )
    comparison = {
        "vq": report,
        "previous_mlp_local95": old_report["summary"],
        "observed_displacement_groups": statistics,
        "limits": "TCN codes depend on 15-step history; scatter is descriptive, not a memoryless symbol definition. "
        "Model parameters, initialization, pretraining and objectives differ.",
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "code_usage": report["code_usage"]}, indent=2))


if __name__ == "__main__":
    main()
