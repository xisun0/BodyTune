"""Matched global/local TCN VQ scores: task accuracy, scale and circularity."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from roboscore.aligned import load_splits
from roboscore.kinematics import Arm
from roboscore.vq import VQConfig, VQModel


def main():
    root = Path("outputs/vq_tokens")
    reports, archives, xy = {}, {}, {}
    for mode in ("global1", "local95"):
        reports[mode] = json.loads((root / mode / "report.json").read_text())
        with np.load(root / mode / "test_rollouts.npz") as data:
            archives[mode] = {k: data[k].copy() for k in data.files}
        xy[mode] = Arm().fk(archives[mode]["q_pred"])
        xy[mode] -= xy[mode][:, :1].copy()
    g, l = reports["global1"], reports["local95"]
    for key in (
        "parameters",
        "performer_parameters",
        "initial_weights_before_kmeans_sha256",
        "seed",
        "batch_size",
        "epochs",
        "pretrain_epochs",
        "min_epochs",
        "patience",
        "optimizer",
        "learning_rate",
        "gradient_clip",
        "threads",
    ):
        if g["training"][key] != l["training"][key]:
            raise ValueError(f"unmatched training/model field {key}")
    if g["data"]["sha256"] != l["data"]["sha256"]:
        raise ValueError("different source files")
    for key in ("motion", "q_target"):
        assert np.array_equal(archives["global1"][key], archives["local95"][key])
    assert g["summary"]["test"] == g["summary"]["permute_steps"]
    with np.load(Path(g["data"]["source"]) / "test_demonstrations.npz") as data:
        diameter = data["diameter"].copy()
    target = Arm().fk(archives["global1"]["q_target"])
    target -= target[:, :1].copy()
    selected = np.argsort(diameter)[np.linspace(0, len(diameter) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, i in zip(axes.flat, selected):
        ax.plot(*target[i].T, "k-", lw=1.8, label="Target")
        for mode, label in (("global1", "VQ: 1 token"), ("local95", "VQ: 95 tokens")):
            ax.plot(*xy[mode][i].T, "--", lw=1.3, label=label)
        ax.set(title=f"Diameter {diameter[i]:.3f}", xlabel="Relative x", ylabel="Relative y")
        ax.set_aspect("equal", adjustable="box")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(root / "inference.png", dpi=160)
    fig.savefig(root / "inference.pdf")
    plt.close(fig)
    fields = (
        ("Success (%)", lambda r: r["summary"]["test"]["success_rate"] * 100),
        ("Mean NRMSE", lambda r: r["summary"]["test"]["mean_nrmse"]),
        (
            "Diameter error (%)",
            lambda r: r["circle_diagnostics"]["mean_relative_diameter_error"] * 100,
        ),
        (
            "Radial RMSE / radius (%)",
            lambda r: r["circle_diagnostics"]["mean_radial_error_over_radius"] * 100,
        ),
    )
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
    for ax, (title, extract) in zip(axes, fields):
        values = [extract(reports[mode]) for mode in ("global1", "local95")]
        ax.bar(["1 token", "95 tokens"], values, color=["tab:blue", "tab:orange"])
        ax.set(title=title, ylim=(0, max(values) * 1.2))
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.4g}", ha="center", va="bottom")
    fig.savefig(root / "metrics.png", dpi=160)
    fig.savefig(root / "metrics.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for mode, label in (("global1", "1 token"), ("local95", "95 tokens")):
        history = np.genfromtxt(root / mode / "history.csv", delimiter=",", names=True)
        axes[0].plot(history["epoch"], history["val_nrmse"], label=label)
        axes[1].plot(history["epoch"], history["val_perplexity"], label=label)
    axes[0].set(xlabel="Joint epoch", ylabel="Validation NRMSE", yscale="log")
    axes[1].set(xlabel="Joint epoch", ylabel="Validation code perplexity", ylim=(0, 8.5))
    for ax in axes:
        ax.legend(frameon=False)
    fig.savefig(root / "training.png", dpi=160)
    plt.close(fig)
    torch.set_num_threads(4)
    splits, hashes = load_splits(g["data"]["source"], "local")
    if hashes != g["data"]["sha256"]:
        raise ValueError("source hashes changed")
    for mode in ("global1", "local95"):
        checkpoint = torch.load(root / mode / "best.pt", weights_only=True)
        model = VQModel(VQConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.inference_mode():
            for name, (motion, _) in splits.items():
                ids = torch.cat([model.encode(batch) for batch in motion.split(64)]).numpy()
                np.savez_compressed(root / mode / f"{name}_scores.npz", scores=ids)
                if name == "test":
                    assert np.array_equal(ids, archives[mode]["scores"])
    rows = {}
    for mode, report in reports.items():
        rows[mode] = {title: extract(report) for title, extract in fields}
    comparison = {
        "models": reports,
        "metrics": rows,
        "matched": "Dataset files/order, parameter shapes/count/init before kmeans, optimizer, losses, pretraining and epoch/stopping budgets.",
        "differences": "Global temporal pooling before VQ vs 95 local codes; 3 vs 285 nominal bits. Kmeans operates on each arm's different latent distribution.",
        "limits": "Both new arms use a clock-conditioned auxiliary decoder. Legacy 50.42% is not the matched local control. One seed and circle-only fixed phase; no isolated locality or grammar claim.",
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
