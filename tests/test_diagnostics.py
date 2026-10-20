import numpy as np
import pytest

from roboscore.data import generate
from roboscore.diagnostics import circle_quality


def test_exact_circles_and_size_mismatch():
    pairs, _ = generate(6, task="circles")
    q = np.stack([pair.q_a for pair in pairs])
    d = np.array([pair.diameter for pair in pairs])
    exact = circle_quality(q, d)
    assert exact["circle_fit_valid"] == 6
    assert exact["mean_radial_error_over_radius"] < 1e-12
    assert exact["mean_relative_diameter_error"] < 1e-12
    assert exact["mean_turns"] == pytest.approx(1.0)
    assert exact["mean_closure_over_target_diameter"] < 1e-12
    assert circle_quality(q, d * 2)["mean_relative_diameter_error"] == pytest.approx(0.5)


def test_stationary_path_has_no_valid_circle_fit():
    q = np.tile([0.2, 1.1], (2, 96, 1))
    result = circle_quality(q, [0.3, 0.5])
    assert result["circle_fit_valid"] == 0
    assert result["mean_radial_error_over_radius"] is None
