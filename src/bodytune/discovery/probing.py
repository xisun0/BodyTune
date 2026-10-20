"""Conservative, explicitly armed single-joint probing for a connected SO-101."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .probe_schema import ProbeRecord, ProbeSessionWriter

SO101_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class ProbeRobot(Protocol):
    """The verified subset of LeRobot's SO101Follower interface that we use."""

    config: Any

    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ProbeConfig:
    joint: str
    command_magnitude: float = 2.0
    settle_s: float = 0.25
    timeout_s: float = 1.0
    return_tolerance: float = 1.0
    max_session_deviation: float = 5.0
    boundary_margin: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name != "joint" and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.joint not in SO101_JOINTS:
            raise ValueError(f"Unknown SO-101 joint: {self.joint}")
        if not 0 < self.command_magnitude <= 5:
            raise ValueError("command_magnitude must be in (0, 5]")
        if not 0 <= self.settle_s <= self.timeout_s <= 3:
            raise ValueError("require 0 <= settle_s <= timeout_s <= 3")
        if not 0 < self.return_tolerance <= self.max_session_deviation <= 10:
            raise ValueError("invalid return/deviation bounds")
        if not 0 <= self.boundary_margin <= 10:
            raise ValueError("boundary_margin must be in [0, 10]")


class ProbeExecutionError(RuntimeError):
    """A probe or its mandatory return-to-start safety step failed."""


def _split_observation(observation: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    positions = {
        key: float(value) for key, value in observation.items() if key.endswith(".pos")
    }
    frames = {key: value for key, value in observation.items() if not key.endswith(".pos")}
    if set(positions) != {f"{joint}.pos" for joint in SO101_JOINTS}:
        raise RuntimeError("observation does not contain the six expected SO-101 joint positions")
    if not all(math.isfinite(value) for value in positions.values()):
        raise RuntimeError("observation contains non-finite joint positions")
    return positions, frames


class SafeProbeRunner:
    """Run one small probe; construction alone never connects or moves hardware."""

    def __init__(self, robot: ProbeRobot, writer: ProbeSessionWriter, config: ProbeConfig):
        config.validate()
        self.robot = robot
        self.writer = writer
        self.config = config
        self._blocked = False

    def run_one(
        self,
        direction: int,
        *,
        armed: bool = False,
        confirm: Callable[[str], bool] | None = None,
    ) -> ProbeRecord:
        if self._blocked:
            raise ProbeExecutionError(
                "runner blocked after unsafe return; inspect hardware before creating a new runner"
            )
        if not armed:
            raise PermissionError("probe is disarmed; pass armed=True only after physical safety checks")
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if confirm is None or not confirm(
            f"Probe {self.config.joint} by {direction * self.config.command_magnitude:+g} units?"
        ):
            raise PermissionError("probe was not explicitly confirmed")

        max_relative_target = getattr(self.robot.config, "max_relative_target", None)
        if max_relative_target is None:
            raise RuntimeError("robot.config.max_relative_target must be set for probing")
        if isinstance(max_relative_target, bool) or not isinstance(max_relative_target, (int, float)):
            raise RuntimeError("Phase 0 requires a scalar max_relative_target")
        if not math.isfinite(max_relative_target) or max_relative_target <= 0:
            raise RuntimeError("robot max_relative_target must be positive")
        if max_relative_target > self.config.command_magnitude:
            raise RuntimeError("robot max_relative_target exceeds the configured probe magnitude")
        if getattr(self.robot.config, "use_degrees", False):
            raise RuntimeError("Phase 0 probing currently requires normalized joint positions")

        before_ts = time.time_ns()
        before, frames_before = _split_observation(self.robot.get_observation())
        target = before.copy()
        target[f"{self.config.joint}.pos"] += direction * self.config.command_magnitude
        self._assert_close_to_start(target, before)
        self._assert_normalized_bounds(target, self.config.joint)

        sample_id = uuid.uuid4().hex
        command_ts = time.time_ns()
        started = time.monotonic()
        action_attempted = False
        action_sent: dict[str, Any] = {}
        return_action_sent: dict[str, Any] = {}
        after: dict[str, float] = {}
        returned: dict[str, float] = {}
        frames_after: dict[str, Any] = {}
        after_ts: int | None = None
        returned_ts: int | None = None
        execution_error: BaseException | None = None
        return_error: BaseException | None = None
        return_tolerance_exceeded = False
        try:
            action_attempted = True
            action_sent = self.robot.send_action(target)
            self._bounded_wait(started)
            after_ts = time.time_ns()
            after, frames_after = _split_observation(self.robot.get_observation())
        except BaseException as exc:
            execution_error = exc
        finally:
            if action_attempted:
                try:
                    return_started = time.monotonic()
                    return_action_sent = self.robot.send_action(before)
                    self._bounded_wait(return_started)
                    returned_ts = time.time_ns()
                    returned, _ = _split_observation(self.robot.get_observation())
                    return_tolerance_exceeded = (
                        max(abs(returned[key] - before[key]) for key in before)
                        > self.config.return_tolerance
                    )
                except BaseException as exc:
                    return_error = exc
            duration_ms = (time.monotonic() - started) * 1000

        if return_error is not None or return_tolerance_exceeded:
            self._blocked = True

        before_refs = self.writer.save_frames(sample_id, "before", frames_before)
        after_refs = self.writer.save_frames(sample_id, "after", frames_after)
        if execution_error is not None:
            if return_error is not None:
                safety_status = "probe_failed_return_failed"
            elif return_tolerance_exceeded:
                safety_status = "probe_failed_return_tolerance_exceeded"
            else:
                safety_status = "probe_failed_returned"
        elif return_error is not None:
            safety_status = "return_failed"
        elif return_tolerance_exceeded:
            safety_status = "return_tolerance_exceeded"
        else:
            safety_status = "probe_completed"
        record = ProbeRecord(
            sample_id=sample_id,
            joint=self.config.joint,
            direction=direction,
            command_magnitude=self.config.command_magnitude,
            commanded_action=target,
            action_sent={key: float(value) for key, value in action_sent.items()},
            return_action_sent={key: float(value) for key, value in return_action_sent.items()},
            q_before=before,
            q_after=after,
            q_returned=returned,
            delta_q={key: after[key] - before[key] for key in before} if after else {},
            command_timestamp_ns=command_ts,
            state_before_timestamp_ns=before_ts,
            state_after_timestamp_ns=after_ts,
            state_returned_timestamp_ns=returned_ts,
            camera_before_timestamp_ns=before_ts if frames_before else None,
            camera_after_timestamp_ns=after_ts if frames_after else None,
            camera_before_refs=before_refs,
            camera_after_refs=after_refs,
            probe_duration_ms=duration_ms,
            safety_status=safety_status,
            execution_error=self._format_error(execution_error),
            return_error=self._format_error(return_error),
        )
        self.writer.append(record)

        if execution_error is not None:
            raise execution_error
        if return_error is not None:
            raise ProbeExecutionError(
                "probe return-to-start failed; further probing is unsafe"
            ) from return_error
        if return_tolerance_exceeded:
            raise ProbeExecutionError("probe exceeded return tolerance; further probing is unsafe")
        return record

    def _bounded_wait(self, started: float) -> None:
        remaining = min(self.config.settle_s, self.config.timeout_s - (time.monotonic() - started))
        if remaining < 0:
            raise TimeoutError("probe exceeded timeout before observation")
        time.sleep(remaining)
        if time.monotonic() - started > self.config.timeout_s:
            raise TimeoutError("probe exceeded timeout")

    def _assert_close_to_start(
        self, target: Mapping[str, float], start: Mapping[str, float]
    ) -> None:
        if max(abs(target[key] - start[key]) for key in start) > self.config.max_session_deviation:
            raise ValueError("target exceeds max_session_deviation")

    def _assert_normalized_bounds(self, target: Mapping[str, float], joint: str) -> None:
        value = target[f"{joint}.pos"]
        lower, upper = (0.0, 100.0) if joint == "gripper" else (-100.0, 100.0)
        lower += self.config.boundary_margin
        upper -= self.config.boundary_margin
        if not lower <= value <= upper:
            raise ValueError(
                f"target for {joint}={value:g} exceeds normalized safety bounds [{lower:g}, {upper:g}]"
            )

    @staticmethod
    def _format_error(error: BaseException | None) -> str | None:
        if error is None:
            return None
        return f"{type(error).__name__}: {error}"

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self.config)
