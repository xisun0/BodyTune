import json

import numpy as np
import pytest

from roboscore.benchmark import execute, measure, reproduce, run
from roboscore.data import generate
from roboscore.kinematics import Arm


def test_oracle_and_copy_are_different():
    arm = Arm()
    pairs, _ = generate(12)
    errors = []
    for pair in pairs:
        q = reproduce(arm, pair, "ik_oracle")
        assert measure(arm, pair, q)["nrmse"] < 1e-12
        np.testing.assert_allclose(q[0], pair.q_b[0])
        try:
            copy = reproduce(arm, pair, "joint_copy")
        except ValueError:
            continue
        errors.append(measure(arm, pair, copy)["nrmse"])
    assert errors and max(errors) > 0.01


def test_executor_rejects_fast_commands():
    with pytest.raises(ValueError, match="velocity"):
        execute(Arm(), [0, 1], [[1, 0]], 0.01)


def test_run_serializes_and_refuses_overwrite(tmp_path):
    output = tmp_path / "run"
    report = run(output, count=6)
    assert report["summary"]["ik_oracle"]["success_rate"] == 1
    assert json.loads((output / "report.json").read_text()) == report
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        assert data["q_a"].shape == (6, 96, 2)
    with pytest.raises(FileExistsError):
        run(output)


def test_executor_rejects_joint_limit_crossing():
    with pytest.raises(ValueError, match="joint limits"):
        execute(Arm(), [3.13, 1], [[0.05, 0]], 0.05)


def test_geometry_metric_is_sensitive_to_order():
    arm = Arm()
    pair = generate(1)[0][0]
    reversed_q = reproduce(arm, pair, "ik_oracle")[::-1]
    assert measure(arm, pair, reversed_q)["nrmse"] > 0.01
