"""Feasibility-aware paired demonstrations; labels are evaluation metadata only."""

from dataclasses import dataclass

import numpy as np

from .kinematics import Arm

FAMILIES = ("square", "rectangle", "triangle", "circle", "arc", "polyline")


def resample(vertices, steps):
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
    if distance[-1] <= 0:
        raise ValueError("zero-length curve")
    t = np.linspace(0, distance[-1], steps)
    return np.column_stack([np.interp(t, distance, vertices[:, i]) for i in range(2)])


def curve(family, steps, rng):
    if steps < 3:
        raise ValueError("at least three samples required")
    if family in ("circle", "arc"):
        t = np.linspace(0, 2 * np.pi if family == "circle" else rng.uniform(0.6, 1.6) * np.pi, 512)
        p = np.column_stack([np.cos(t), np.sin(t)])
    else:
        vertices = {
            "square": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
            "rectangle": [[0, 0], [1.5, 0], [1.5, 1], [0, 1], [0, 0]],
            "triangle": [[0, 0], [1, 0], [0.5, 0.9], [0, 0]],
            "polyline": [[0, 0], [0.6, 0], [0.6, 0.5], [1.1, 0.5], [0.9, 1]],
        }
        if family not in vertices:
            raise ValueError("unknown curve family")
        p = np.asarray(vertices[family], dtype=float)
    if rng.random() < 0.5:
        p = p[::-1]
    p = resample(p, steps)
    theta = rng.uniform(-np.pi, np.pi)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (p - p[0]) @ rotation.T * rng.uniform(0.12, 0.35)


@dataclass
class Pair:
    family: str
    relative: np.ndarray
    q_a: np.ndarray
    q_b: np.ndarray
    dt: float
    diameter: float | None = None


def generate(count=120, steps=96, seed=42, dt=0.05, arm=None, max_attempts=None, task="mixed"):
    if count < 1 or steps < 3 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid dataset parameters")
    if task not in ("mixed", "circles"):
        raise ValueError("unknown task")
    arm = arm or Arm()
    rng = np.random.default_rng(seed)
    # Hold each requested diameter fixed while resampling feasible initial states.
    diameters = rng.uniform(0.24, 0.70, count) if task == "circles" else None
    pairs, rejected = [], 0
    budget = max_attempts if max_attempts is not None else count * 100
    for _ in range(budget):
        diameter = None if diameters is None else float(diameters[len(pairs)])
        if task == "circles":
            family = "circle"
            theta = np.linspace(0, 2 * np.pi, steps)
            relative = diameter / 2 * np.column_stack((np.cos(theta) - 1, np.sin(theta)))
        else:
            family = FAMILIES[len(pairs) % len(FAMILIES)]
            relative = curve(family, steps, rng)
        paths = []
        try:
            for _role in range(2):
                q0 = np.array([rng.uniform(-1.5, 1.5), rng.choice([-1, 1]) * rng.uniform(0.7, 2.3)])
                q = arm.ik(relative + arm.fk(q0), branch=int(np.sign(q0[1])))
                if not np.allclose(q[0], q0, atol=1e-10):
                    raise ValueError("initial state mismatch")
                arm.validate(q, dt)
                paths.append(q)
            if np.linalg.norm(paths[0][0] - paths[1][0]) < 0.3:
                raise ValueError("initial configurations too similar")
        except ValueError:
            rejected += 1
            continue
        pairs.append(Pair(family, relative, *paths, dt, diameter))
        if len(pairs) == count:
            return pairs, {"attempted": len(pairs) + rejected, "rejected": rejected}
    raise RuntimeError(f"only {len(pairs)}/{count} feasible pairs within {budget} attempts")
