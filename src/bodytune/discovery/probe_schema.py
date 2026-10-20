"""Serializable records for Phase-0 sensorimotor probes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProbeSessionMetadata:
    session_id: str
    robot_id: str
    robot_type: str
    bodytune_revision: str
    lerobot_revision: str
    calibration_identifier: str
    calibration_sha256: str
    probe_configuration: dict[str, Any]


@dataclass(frozen=True)
class ProbeRecord:
    sample_id: str
    joint: str
    direction: int
    command_magnitude: float
    commanded_action: dict[str, float]
    action_sent: dict[str, float]
    return_action_sent: dict[str, float]
    q_before: dict[str, float]
    q_after: dict[str, float]
    q_returned: dict[str, float]
    delta_q: dict[str, float]
    command_timestamp_ns: int
    state_before_timestamp_ns: int
    state_after_timestamp_ns: int | None
    state_returned_timestamp_ns: int | None
    camera_before_timestamp_ns: int | None
    camera_after_timestamp_ns: int | None
    camera_before_refs: dict[str, str]
    camera_after_refs: dict[str, str]
    probe_duration_ms: float
    safety_status: str
    execution_error: str | None
    return_error: str | None


class ProbeSessionWriter:
    """Append-only JSONL writer with camera arrays stored as sidecar files."""

    def __init__(self, root: Path, metadata: ProbeSessionMetadata):
        self.root = root
        self.frames_dir = root / "frames"
        self.root.mkdir(parents=True, exist_ok=False)
        self.frames_dir.mkdir()
        (self.root / "manifest.json").write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.records_path = self.root / "probes.jsonl"

    def save_frames(self, sample_id: str, phase: str, frames: dict[str, Any]) -> dict[str, str]:
        import numpy as np

        references: dict[str, str] = {}
        for camera, frame in frames.items():
            safe_camera = camera.replace("/", "_")
            relative = Path("frames") / f"{sample_id}.{phase}.{safe_camera}.npy"
            np.save(self.root / relative, np.asarray(frame), allow_pickle=False)
            references[camera] = relative.as_posix()
        return references

    def append(self, record: ProbeRecord) -> None:
        with self.records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
