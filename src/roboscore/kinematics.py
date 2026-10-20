"""Analytic two-link kinematics, in radians and arbitrary length units."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arm:
    lengths: tuple[float, float] = (1.0, 1.0)
    lower: tuple[float, float] = (-np.pi, -2.9)
    upper: tuple[float, float] = (np.pi, 2.9)
    max_speed: float = 2.0  # radians per second, each joint
    singularity_margin: float = 0.05  # minimum |sin(q2)|

    def __post_init__(self):
        values = np.array(
            [*self.lengths, *self.lower, *self.upper, self.max_speed, self.singularity_margin]
        )
        if (
            not np.isfinite(values).all()
            or min(self.lengths) <= 0
            or self.max_speed <= 0
            or not 0 < self.singularity_margin < 1
            or np.any(np.array(self.lower) >= self.upper)
        ):
            raise ValueError("invalid arm configuration")

    @staticmethod
    def _points(values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 0 or values.shape[-1] != 2 or not np.isfinite(values).all():
            raise ValueError("expected finite (..., 2) coordinates")
        return values

    def fk(self, q):
        q = self._points(q)
        a, b = q[..., 0], q[..., 0] + q[..., 1]
        l1, l2 = self.lengths
        return np.stack([l1 * np.cos(a) + l2 * np.cos(b), l1 * np.sin(a) + l2 * np.sin(b)], axis=-1)

    def ik(self, points, branch=1):
        """Solve an entire ordered path on one elbow branch; never switch branches."""
        if branch not in (-1, 1):
            raise ValueError("branch must be -1 or 1")
        p = self._points(points)
        if p.ndim != 2 or len(p) < 2:
            raise ValueError("expected a path of at least two points")
        l1, l2 = self.lengths
        c = (np.sum(p * p, axis=-1) - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if np.any(np.abs(c) > 1 + 1e-12):
            raise ValueError("unreachable path")
        q2 = branch * np.arccos(np.clip(c, -1, 1))
        if np.any(np.abs(np.sin(q2)) < self.singularity_margin):
            raise ValueError("path too close to singularity")
        q1 = np.unwrap(
            np.arctan2(p[:, 1], p[:, 0]) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        )
        # Pick one global 2pi offset; pointwise wrapping would create discontinuities.
        offsets = [q1 + k * 2 * np.pi for k in range(-2, 3)]
        valid = [a for a in offsets if np.all((a >= self.lower[0]) & (a <= self.upper[0]))]
        if not valid:
            raise ValueError("path exceeds joint-1 limits")
        q = np.column_stack([min(valid, key=lambda a: np.mean(a * a)), q2])
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("path exceeds joint limits")
        return q

    def validate(self, q, dt):
        q = self._points(q)
        if q.ndim != 2 or len(q) < 2 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid trajectory or dt")
        if np.any(q < np.array(self.lower) - 1e-12) or np.any(q > np.array(self.upper) + 1e-12):
            raise ValueError("trajectory exceeds joint limits")
        if np.any(np.abs(np.sin(q[:, 1])) < self.singularity_margin):
            raise ValueError("trajectory too close to singularity")
        if np.max(np.abs(np.diff(q, axis=0))) / dt > self.max_speed + 1e-10:
            raise ValueError("trajectory exceeds velocity limits")
