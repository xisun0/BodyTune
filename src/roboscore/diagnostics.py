"""Descriptive circle-fit diagnostics, separate from the task success criterion."""

import numpy as np

from .kinematics import Arm


def circle_quality(q, diameter):
    xy = Arm().fk(q)
    diameter = np.asarray(diameter, dtype=float)
    if (
        xy.ndim != 3
        or diameter.shape != (len(xy),)
        or not np.isfinite(diameter).all()
        or np.any(diameter <= 0)
    ):
        raise ValueError("expected batched paths and positive diameters")
    radial, scale, turns = [], [], []
    for path, d in zip(xy, diameter):
        x, y = path.T
        design = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
        coefficients, _, rank, _ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
        cx, cy, c = coefficients
        radius2 = c + cx * cx + cy * cy
        if rank < 3 or radius2 <= 1e-12:
            continue
        radius = np.sqrt(radius2)
        distances = np.linalg.norm(path - [cx, cy], axis=-1)
        radial.append(np.sqrt(np.mean((distances - radius) ** 2)) / radius)
        scale.append(abs(2 * radius - d) / d)
        phase = np.unwrap(np.arctan2(y - cy, x - cx))
        turns.append((phase[-1] - phase[0]) / (2 * np.pi))
    return {
        "circle_fit_valid": len(radial),
        "count": len(xy),
        "mean_radial_error_over_radius": float(np.mean(radial)) if radial else None,
        "mean_relative_diameter_error": float(np.mean(scale)) if scale else None,
        "mean_turns": float(np.mean(turns)) if turns else None,
        "mean_closure_over_target_diameter": float(
            np.mean(np.linalg.norm(xy[:, -1] - xy[:, 0], axis=-1) / diameter)
        ),
    }
