"""Fixed diameter-label control: execution overlays and quantization-only reference."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboscore.diagnostics import circle_quality
from roboscore.kinematics import Arm


def main():
    root = Path("outputs/aligned128")
    run = root / "binned8"
    report = json.loads((run / "report.json").read_text())
    global_report = json.loads((root / "global1/report.json").read_text())
    if report["data"]["sha256"] != global_report["data"]["sha256"]:
        raise ValueError("source dataset mismatch")
    with np.load(run / "test_rollouts.npz") as data:
        target_q, predicted_q, ids = data["q_target"], data["q_pred"], data["scores"][:, 0]
    with np.load(root / "global1/test_rollouts.npz") as data:
        assert np.array_equal(target_q, data["q_target"])
        global_q = data["q_pred"].copy()
    with np.load("outputs/circles1/test_demonstrations.npz") as data:
        diameter = data["diameter"].copy()
    edges = np.array(report["diameter_bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    target = Arm().fk(target_q)
    target -= target[:, :1].copy()
    # Exact circles at each fixed bin midpoint: endpoint-only reference, no controller/IK.
    reference = target * (centers[ids] / diameter)[:, None, None]
    length = np.linalg.norm(np.diff(target, axis=1), axis=-1).sum(-1)
    error = np.sqrt(np.mean(np.sum((reference - target) ** 2, axis=-1), axis=1)) / length
    groups = []
    for code in range(8):
        mask = ids == code
        groups.append(
            {
                "id": code,
                "test_count": int(mask.sum()),
                "lower": edges[code],
                "upper": edges[code + 1],
            }
        )
    comparison = {
        "report": report,
        "circle_diagnostics": circle_quality(predicted_q, diameter),
        "bin_counts": groups,
        "midpoint_circle_reference": {
            "mean_nrmse": float(error.mean()),
            "geometric_success_rate": float(np.mean(error <= 0.01)),
            "success_count": int(np.sum(error <= 0.01)),
            "limits": "Ideal midpoint circles; no kinematic feasibility check. Not an upper bound: "
            "other representatives, state-dependent outputs or different bins may score differently.",
        },
    }
    paths = [Arm().fk(q) for q in (global_q, predicted_q)]
    for path in paths:
        path -= path[:, :1].copy()
    selected = np.argsort(diameter)[np.linspace(0, len(diameter) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for ax, i in zip(axes.flat, selected):
        ax.plot(*target[i].T, "k-", label="Target", lw=1.8)
        ax.plot(*paths[0][i].T, "--", label="Learned global token", lw=1.2)
        ax.plot(*paths[1][i].T, "--", label="Fixed diameter bin", lw=1.2)
        ax.set(
            title=f"Diameter {diameter[i]:.3f}; bin {ids[i]}",
            xlabel="Relative x",
            ylabel="Relative y",
        )
        ax.set_aspect("equal", adjustable="box")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(run / "inference.png", dpi=160)
    fig.savefig(run / "inference.pdf")
    plt.close(fig)
    (run / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
