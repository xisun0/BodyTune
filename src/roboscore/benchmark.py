"""Full-curve IK oracle and joint-increment copy control; no learned language."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import generate
from .kinematics import Arm


def execute(arm, q0, commands, dt):
    """Integrate bounded increments, then enforce all sampled rollout constraints."""
    commands = np.asarray(commands, dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 2 or not np.isfinite(commands).all():
        raise ValueError("invalid commands")
    if np.max(np.abs(commands)) > arm.max_speed * dt + 1e-10:
        raise ValueError("action exceeds velocity limit")
    q = np.vstack([q0, q0 + np.cumsum(commands, axis=0)])
    arm.validate(q, dt)
    return q


def reproduce(arm, pair, method):
    q0 = pair.q_b[0]
    if method == "ik_oracle":
        a = arm.fk(pair.q_a)
        relative = a - a[0]
        target = arm.ik(relative + arm.fk(q0), int(np.sign(q0[1])))
        commands = np.diff(target, axis=0)
    elif method == "joint_copy":
        commands = np.diff(pair.q_a, axis=0)
    else:
        raise ValueError("unknown method")
    return execute(arm, q0, commands, pair.dt)


def measure(arm, pair, q):
    target = arm.fk(pair.q_a)
    target -= target[0].copy()
    p = arm.fk(q)
    p -= p[0].copy()
    errors = np.linalg.norm(p - target, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    length = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    return {
        "rmse": rmse,
        "max_error": float(errors.max()),
        "nrmse": rmse / length,
        "max_joint_speed": float(np.abs(np.diff(q, axis=0)).max() / pair.dt),
        "success": rmse / length <= 0.01,
    }


def run(output, count=120, steps=96, seed=42, dt=0.05):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arm = Arm()
    pairs, sampling = generate(count, steps, seed, dt, arm)
    rows = []
    executed = {"ik_oracle": [], "joint_copy": []}
    for i, pair in enumerate(pairs):
        for method, rollouts in executed.items():
            row = {"pair": i, "family": pair.family, "method": method}
            try:
                q = reproduce(arm, pair, method)
                row.update(feasible=True, failure="", **measure(arm, pair, q))
            except ValueError as error:
                q = np.full_like(pair.q_b, np.nan)
                row.update(
                    feasible=False,
                    failure=str(error),
                    rmse=None,
                    max_error=None,
                    nrmse=None,
                    max_joint_speed=None,
                    success=False,
                )
            rollouts.append(q)
            rows.append(row)
    dataset = dict(
        q_a=np.stack([p.q_a for p in pairs]),
        q_b=np.stack([p.q_b for p in pairs]),
        relative=np.stack([p.relative for p in pairs]),
        family=np.array([p.family for p in pairs]),
        **{name: np.stack(q) for name, q in executed.items()},
    )
    np.savez_compressed(output / "trajectories.npz", **dataset)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for method in executed:
        subset = [r for r in rows if r["method"] == method]
        valid = [r for r in subset if r["feasible"]]
        summaries[method] = {
            "pairs": count,
            "feasible": len(valid),
            "success_rate": sum(r["success"] for r in subset) / count,
            "mean_nrmse_feasible": float(np.mean([r["nrmse"] for r in valid])) if valid else None,
            "max_error_feasible": max((r["max_error"] for r in valid), default=None),
        }

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    manifest = {
        "config": {"count": count, "steps": steps, "seed": seed, "dt": dt, "arm": asdict(arm)},
        "sampling": sampling,
        "summary": summaries,
        "provenance": {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        },
        "semantics": "Start-aligned ordered geometry, no rotation or scale alignment.",
        "success": "Feasible sampled rollout and RMSE/path_length <= 0.01.",
        "limitation": "Full-curve oracle, no discrete score; constraints at sampled times only.",
    }
    (output / "report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return json.loads(json.dumps(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
