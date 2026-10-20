import numpy as np
import pytest

from roboscore.data import generate
from roboscore.kinematics import Arm


def test_known_forward_positions():
    np.testing.assert_allclose(
        Arm().fk([[0, 0], [np.pi / 2, 0], [0, np.pi / 2]]), [[2, 0], [0, 2], [1, 1]], atol=1e-12
    )


@pytest.mark.parametrize("branch", [-1, 1])
def test_inverse_roundtrip_and_continuity(branch):
    arm = Arm()
    q = np.column_stack([np.linspace(-0.5, 0.5, 100), np.full(100, branch * 1.2)])
    recovered = arm.ik(arm.fk(q), branch)
    np.testing.assert_allclose(recovered, q, atol=1e-12)


@pytest.mark.parametrize(
    "points", [[[3, 0], [3, 0.1]], [[2, 0], [2, 0]], [[float("nan"), 0], [1, 0]]]
)
def test_reject_invalid_paths(points):
    with pytest.raises(ValueError):
        Arm().ik(points)


def test_reject_velocity_and_joint_limits():
    with pytest.raises(ValueError, match="velocity"):
        Arm().validate([[0, 1], [1, 1]], 0.01)
    with pytest.raises(ValueError, match="joint limits"):
        Arm().validate([[0, 1], [4, 1]], 10)


def test_feasible_pairs_are_deterministic_and_equivalent():
    pairs, stats = generate(30, seed=7)
    repeated, again = generate(30, seed=7)
    assert stats == again
    arm = Arm()
    for a, b in zip(pairs, repeated):
        np.testing.assert_array_equal(a.q_a, b.q_a)
        assert np.linalg.norm(a.q_a[0] - a.q_b[0]) >= 0.3
        for q in (a.q_a, a.q_b):
            arm.validate(q, a.dt)
            p = arm.fk(q)
            np.testing.assert_allclose(p - p[0], a.relative, atol=1e-12)
            assert np.all(np.sign(q[:, 1]) == np.sign(q[0, 1]))


def test_generation_budget_is_explicit():
    with pytest.raises(RuntimeError, match="within 0 attempts"):
        generate(2, max_attempts=0)


def test_full_posture_preserves_link_lengths():
    arm = Arm(lengths=(0.7, 1.3))
    q = np.array([[0.2, 1.1], [-0.4, -1.5]])
    joints = arm.joints(q)
    assert joints.shape == (2, 3, 2)
    np.testing.assert_allclose(joints[:, 0], 0)
    np.testing.assert_allclose(joints[:, -1], arm.fk(q))
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(joints, axis=1), axis=-1), [[0.7, 1.3], [0.7, 1.3]]
    )
