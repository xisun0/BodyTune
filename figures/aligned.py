"""Compare global and causal local scores; describe learned endpoint-motion symbols."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from roboscore.diagnostics import circle_quality
from roboscore.kinematics import Arm


def main():
    root = Path(os.environ.get("ROBOSCORE_ALIGNMENT", "outputs/aligned128"))
    reports, data = {}, {}
    for mode, folder in (("global", "global1"), ("local", "local95")):
        directory = root / folder
        reports[mode] = json.loads((directory / "report.json").read_text())
        with np.load(directory / "test_rollouts.npz", allow_pickle=False) as archive:
            data[mode] = {key: archive[key].copy() for key in archive.files}
    g, l = reports["global"], reports["local"]
    for key in ("parameters", "performer_parameters", "initial_weights_sha256"):
        if g["training"][key] != l["training"][key]:
            raise ValueError(f"unmatched models: {key}")
    if g["data"]["sha256"] != l["data"]["sha256"]:
        raise ValueError("different source files")
    for key in ("motion", "q_target"):
        if not np.array_equal(data["global"][key], data["local"][key]):
            raise ValueError("different ordered test examples")
    if g["summary"]["permute_steps"] != g["summary"]["test"]:
        raise ValueError("one-token temporal identity control changed results")
    with np.load(
        Path(g["data"]["source"]) / "test_demonstrations.npz", allow_pickle=False
    ) as archive:
        diameter = archive["diameter"].copy()
    target = Arm().fk(data["global"]["q_target"])
    target -= target[:, :1].copy()
    actual = {mode: Arm().fk(values["q_pred"]) for mode, values in data.items()}
    for xy in actual.values():
        xy -= xy[:, :1].copy()
    diagnostics = {
        mode: circle_quality(values["q_pred"], diameter) for mode, values in data.items()
    }
    selected = np.argsort(diameter)[np.linspace(0, len(diameter) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, i in zip(axes.flat, selected):
        ax.plot(*target[i].T, "k-", lw=1.8, label="Target")
        for mode, label in (("global", "Global: 1 token"), ("local", "Local: 95 tokens")):
            ax.plot(*actual[mode][i].T, "--", lw=1.3, label=label)
        ax.set(title=f"Diameter = {diameter[i]:.3f}", xlabel="Relative x", ylabel="Relative y")
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span = max(x1 - x0, y1 - y0)
        ax.set_xlim((x0 + x1 - span) / 2, (x0 + x1 + span) / 2)
        ax.set_ylim((y0 + y1 - span) / 2, (y0 + y1 + span) / 2)
        ax.set_aspect("equal", adjustable="box")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(root / "aligned_inference.png", dpi=160)
    fig.savefig(root / "aligned_inference.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for folder, label in (("global1", "Global: 1 token"), ("local95", "Local: 95 tokens")):
        h = np.genfromtxt(root / folder / "history.csv", delimiter=",", names=True)
        ax.plot(h["epoch"], h["val_nrmse"], label=label)
    ax.set(xlabel="Epoch", ylabel="Validation mean NRMSE", yscale="log")
    ax.legend(frameon=False)
    fig.savefig(root / "alignment_curves.png", dpi=160)
    plt.close(fig)
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:8])
    norm = BoundaryNorm(np.arange(-0.5, 8.5), 8)
    motion = data["local"]["motion"].reshape(-1, 2)
    ids = data["local"]["scores"].reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    points = axes[0].scatter(*motion.T, c=ids, cmap=cmap, norm=norm, s=3, alpha=0.5)
    axes[0].set(
        xlabel="Endpoint dx per step",
        ylabel="Endpoint dy per step",
        title="Local encoder inputs and hard symbols",
    )
    axes[0].set_aspect("equal")
    axes[1].scatter(diameter, data["global"]["scores"][:, 0], s=12)
    axes[1].set(
        xlabel="True diameter (evaluation metadata)",
        ylabel="Global token ID",
        yticks=range(8),
        ylim=(-0.5, 7.5),
    )
    fig.colorbar(points, ax=axes[0], ticks=range(8), label="Token ID")
    fig.savefig(root / "local_symbols.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
    image = ax.imshow(
        data["local"]["scores"][selected],
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        extent=(0.5, 95.5, 5.5, -0.5),
    )
    ax.set(
        xlabel="Aligned action step",
        ylabel="Target diameter",
        yticks=range(6),
        yticklabels=[f"{diameter[i]:.3f}" for i in selected],
    )
    fig.colorbar(image, ax=ax, ticks=range(8), label="Token ID")
    fig.savefig(root / "local_scores.png", dpi=160)
    plt.close(fig)
    code_stats = []
    for code in np.unique(ids):
        vectors = motion[ids == code]
        mean = vectors.mean(0)
        average_norm = np.linalg.norm(vectors, axis=-1).mean()
        code_stats.append(
            {
                "token": int(code),
                "count": len(vectors),
                "mean_dx": float(mean[0]),
                "mean_dy": float(mean[1]),
                "mean_step_norm": float(average_norm),
                "direction_coherence": float(np.linalg.norm(mean) / average_norm),
            }
        )
    groups = []
    global_ids = data["global"]["scores"][:, 0]
    for code in np.unique(global_ids):
        values = diameter[global_ids == code]
        groups.append(
            {
                "token": int(code),
                "count": len(values),
                "diameter_mean": float(values.mean()),
                "diameter_min": float(values.min()),
                "diameter_max": float(values.max()),
            }
        )
    comparison = {
        "models": reports,
        "circle_diagnostics": diagnostics,
        "local_code_statistics": code_stats,
        "global_code_groups": groups,
        "limits": "Parameter/init matched, information budgets differ by 95x; one seed and circle-only data.",
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        json.dumps(
            {
                "circle_diagnostics": diagnostics,
                "local_code_statistics": code_stats,
                "global_code_groups": groups,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
