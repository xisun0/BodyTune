import json
from dataclasses import dataclass

import numpy as np
import pytest

from bodytune.discovery.probe_schema import ProbeSessionMetadata, ProbeSessionWriter
from bodytune.discovery.probing import ProbeConfig, ProbeExecutionError, SafeProbeRunner


@dataclass
class FakeConfig:
    max_relative_target: float | None = 2.0
    use_degrees: bool = False


class FakeRobot:
    def __init__(self):
        self.config = FakeConfig()
        self.observation_calls = 0
        self.send_calls = 0
        self.fail_observation_call = None
        self.fail_send_call = None
        self.return_offset = 0.0
        self.positions = {
            f"{joint}.pos": 0.0
            for joint in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        }

    def get_observation(self):
        self.observation_calls += 1
        if self.observation_calls == self.fail_observation_call:
            raise RuntimeError("simulated observation failure")
        return {**self.positions, "front": np.zeros((2, 2, 3), dtype=np.uint8)}

    def send_action(self, action):
        self.send_calls += 1
        if self.send_calls == self.fail_send_call:
            raise RuntimeError("simulated send failure")
        self.positions = dict(action)
        if self.send_calls == 2 and self.return_offset:
            self.positions["wrist_roll.pos"] += self.return_offset
        return dict(action)


def make_writer(tmp_path):
    metadata = ProbeSessionMetadata(
        session_id="test",
        robot_id="fake",
        robot_type="so101_follower",
        bodytune_revision="test",
        lerobot_revision="test",
        calibration_identifier="test",
        calibration_sha256="0" * 64,
        probe_configuration={},
    )
    return ProbeSessionWriter(tmp_path / "session", metadata)


def test_probe_is_disarmed_by_default(tmp_path):
    runner = SafeProbeRunner(FakeRobot(), make_writer(tmp_path), ProbeConfig(joint="wrist_roll", settle_s=0))
    with pytest.raises(PermissionError, match="disarmed"):
        runner.run_one(1)


def test_probe_requires_confirmation(tmp_path):
    runner = SafeProbeRunner(FakeRobot(), make_writer(tmp_path), ProbeConfig(joint="wrist_roll", settle_s=0))
    with pytest.raises(PermissionError, match="confirmed"):
        runner.run_one(1, armed=True, confirm=lambda _: False)


def test_probe_moves_and_returns_with_structured_log(tmp_path):
    robot = FakeRobot()
    writer = make_writer(tmp_path)
    runner = SafeProbeRunner(robot, writer, ProbeConfig(joint="wrist_roll", settle_s=0))
    record = runner.run_one(1, armed=True, confirm=lambda _: True)

    assert record.delta_q["wrist_roll.pos"] == 2.0
    assert robot.positions["wrist_roll.pos"] == 0.0
    assert record.safety_status == "probe_completed"
    assert writer.records_path.read_text().count("\n") == 1
    assert len(list(writer.frames_dir.glob("*.npy"))) == 2


def test_probe_refuses_robot_without_relative_limit(tmp_path):
    robot = FakeRobot()
    robot.config.max_relative_target = None
    runner = SafeProbeRunner(robot, make_writer(tmp_path), ProbeConfig(joint="wrist_roll", settle_s=0))
    with pytest.raises(RuntimeError, match="max_relative_target"):
        runner.run_one(1, armed=True, confirm=lambda _: True)


def test_probe_refuses_target_near_normalized_boundary(tmp_path):
    robot = FakeRobot()
    robot.positions["wrist_roll.pos"] = 98.0
    runner = SafeProbeRunner(robot, make_writer(tmp_path), ProbeConfig(joint="wrist_roll", settle_s=0))

    with pytest.raises(ValueError, match="normalized safety bounds"):
        runner.run_one(1, armed=True, confirm=lambda _: True)

    assert robot.send_calls == 0


def test_observation_failure_still_returns_and_is_logged(tmp_path):
    robot = FakeRobot()
    robot.fail_observation_call = 2
    writer = make_writer(tmp_path)
    runner = SafeProbeRunner(robot, writer, ProbeConfig(joint="wrist_roll", settle_s=0))

    with pytest.raises(RuntimeError, match="simulated observation failure"):
        runner.run_one(1, armed=True, confirm=lambda _: True)

    assert robot.positions["wrist_roll.pos"] == 0.0
    saved = json.loads(writer.records_path.read_text())
    assert saved["safety_status"] == "probe_failed_returned"
    assert saved["execution_error"] == "RuntimeError: simulated observation failure"
    assert saved["return_error"] is None


def test_return_send_failure_is_logged_and_blocks_completion(tmp_path):
    robot = FakeRobot()
    robot.fail_send_call = 2
    writer = make_writer(tmp_path)
    runner = SafeProbeRunner(robot, writer, ProbeConfig(joint="wrist_roll", settle_s=0))

    with pytest.raises(ProbeExecutionError, match="return-to-start failed"):
        runner.run_one(1, armed=True, confirm=lambda _: True)

    saved = json.loads(writer.records_path.read_text())
    assert saved["safety_status"] == "return_failed"
    assert saved["return_error"] == "RuntimeError: simulated send failure"


def test_return_tolerance_failure_is_logged_and_blocks_completion(tmp_path):
    robot = FakeRobot()
    robot.return_offset = 2.0
    writer = make_writer(tmp_path)
    runner = SafeProbeRunner(robot, writer, ProbeConfig(joint="wrist_roll", settle_s=0))

    with pytest.raises(ProbeExecutionError, match="return tolerance"):
        runner.run_one(1, armed=True, confirm=lambda _: True)

    saved = json.loads(writer.records_path.read_text())
    assert saved["safety_status"] == "return_tolerance_exceeded"


def test_keyboard_interrupt_still_attempts_return(tmp_path):
    robot = FakeRobot()
    robot.fail_observation_call = 2

    def interrupting_observation():
        robot.observation_calls += 1
        if robot.observation_calls == 2:
            raise KeyboardInterrupt
        return {**robot.positions, "front": np.zeros((2, 2, 3), dtype=np.uint8)}

    robot.get_observation = interrupting_observation
    writer = make_writer(tmp_path)
    runner = SafeProbeRunner(robot, writer, ProbeConfig(joint="wrist_roll", settle_s=0))

    with pytest.raises(KeyboardInterrupt):
        runner.run_one(1, armed=True, confirm=lambda _: True)

    assert robot.positions["wrist_roll.pos"] == 0.0
    saved = json.loads(writer.records_path.read_text())
    assert saved["safety_status"] == "probe_failed_returned"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_probe_rejects_nonfinite_or_boolean_configuration(value):
    with pytest.raises(ValueError, match="finite number"):
        ProbeConfig(joint="wrist_roll", command_magnitude=value).validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_probe_rejects_nonfinite_robot_limit(tmp_path, value):
    robot = FakeRobot()
    robot.config.max_relative_target = value
    runner = SafeProbeRunner(robot, make_writer(tmp_path), ProbeConfig(joint="wrist_roll", settle_s=0))
    with pytest.raises(RuntimeError, match="max_relative_target"):
        runner.run_one(1, armed=True, confirm=lambda _: True)
    assert robot.send_calls == 0


def test_probe_rejects_nonfinite_initial_observation(tmp_path):
    robot = FakeRobot()
    robot.positions["elbow_flex.pos"] = float("nan")
    runner = SafeProbeRunner(robot, make_writer(tmp_path), ProbeConfig(joint="wrist_roll", settle_s=0))
    with pytest.raises(RuntimeError, match="non-finite"):
        runner.run_one(1, armed=True, confirm=lambda _: True)
    assert robot.send_calls == 0


@pytest.mark.parametrize("send_failure", [False, True])
def test_unsafe_return_blocks_subsequent_probes(tmp_path, send_failure):
    robot = FakeRobot()
    if send_failure:
        robot.fail_send_call = 2
    else:
        robot.return_offset = 2.0
    runner = SafeProbeRunner(robot, make_writer(tmp_path), ProbeConfig(joint="wrist_roll", settle_s=0))
    with pytest.raises(ProbeExecutionError):
        runner.run_one(1, armed=True, confirm=lambda _: True)
    calls = robot.send_calls
    with pytest.raises(ProbeExecutionError, match="runner blocked"):
        runner.run_one(1, armed=True, confirm=lambda _: True)
    assert robot.send_calls == calls


def test_probe_failure_with_return_offset_does_not_claim_returned(tmp_path):
    robot = FakeRobot()
    robot.fail_observation_call = 2
    robot.return_offset = 2.0
    writer = make_writer(tmp_path)
    runner = SafeProbeRunner(robot, writer, ProbeConfig(joint="wrist_roll", settle_s=0))
    with pytest.raises(RuntimeError, match="simulated observation failure"):
        runner.run_one(1, armed=True, confirm=lambda _: True)
    saved = json.loads(writer.records_path.read_text())
    assert saved["safety_status"] == "probe_failed_return_tolerance_exceeded"
    with pytest.raises(ProbeExecutionError, match="runner blocked"):
        runner.run_one(1, armed=True, confirm=lambda _: True)
