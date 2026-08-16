# Initial environment audit

Audit date: 2026-08-12

No robot was connected or moved during this audit. No LeRobot source or calibration file was changed.

## Local LeRobot checkout

- Remote: `https://github.com/huggingface/lerobot.git`
- Branch: `main`
- Checked-out revision: `882c80d446a63a44868c67ae535467af32ce0e80`
- Commit subject: `Lower limits by 50% for current and torque for gripper motor (#1809)`
- Relationship to the locally known `origin/main`: the checked-out commit is an ancestor and is 700 commits behind; there is no divergent local commit.
- Worktree: exactly three tracked files are modified. Their origin and necessity are unknown.

### Unexplained source modifications

1. `src/lerobot/cameras/utils.py`
   - On Windows, returns OpenCV `CAP_DSHOW` before the existing `CAP_MSMF` return.
   - Effect: switches the camera backend to DirectShow and makes the existing MSMF return unreachable.

2. `src/lerobot/configs/policies.py`
   - Adds `gradient_accumulation_steps = 1` to all pretrained policy configs.
   - Changes state/action feature lookup from requiring canonical key names to accepting the first feature with the corresponding feature type.
   - Reworks `from_pretrained` typing/indentation and uses a named, explicitly closed temporary JSON file so Windows can reopen it, then deletes it.
   - The file lacks a final newline. These changes combine training configuration, relaxed feature selection, and a Windows compatibility fix.

3. `src/lerobot/motors/motors_bus.py`
   - Changes synchronous motor read/write defaults from zero retries to one retry.
   - Promotes failed retry messages from debug to info logging.
   - Effect: transient communication failures receive one additional attempt and become more visible.

These edits were not restored, committed, copied, or incorporated into BodyTune.

## Calibration

LeRobot's default calibration root exists at `~/.cache/huggingface/lerobot/calibration/`.

Active calibration files found:

| Role | LeRobot type | ID | SHA-256 |
|---|---|---|---|
| Robot | `so101_follower` | `my_follower` | `b05956506a8701fe9cdc93a6f84eeb17b7a5e353df5329a321b5bac2e9d70909` |
| Teleoperator | `so101_leader` | `my_leader` | `a53195cc9a4642fba51b9c37f75d143960c5d1ad9569792abd307824ee4a9144` |

Both active files contain entries for `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`. Two timestamp-free leader backup files also exist. BodyTune should record the role/type, ID, and content hash in run metadata while leaving raw calibration local. A local symlink or copy may be used under the gitignored `local/` directory, but portable configuration must not contain an absolute user path.

## Interfaces verified at the pinned revision

- Robot: `SO101FollowerConfig` configures port, safety target limit, cameras, and calibration identity inherited from `RobotConfig`. `SO101Follower` exposes `connect`, `get_observation`, `send_action`, and `disconnect`; its features are six named `.pos` values plus configured cameras.
- Teleoperation: `SO101Leader` is selected with the `so101_leader` type and its own ID/calibration file. The `lerobot-teleoperate` and `lerobot-record` entry points construct registered robot and teleoperator configs.
- Calibration: base `Robot` and `Teleoperator` classes resolve files as `<calibration root>/<role>/<type>/<id>.json`; `connect(calibrate=True)` may enter calibration behavior when cached and motor calibration do not match. Hardware verification must therefore be supervised and must not casually regenerate calibration.
- SmolVLA: the implementation is `SmolVLAPolicy` with `SmolVLAConfig`; the checked-out source documents `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`. LeRobot's recording path loads a `PreTrainedConfig` from `--policy.path`, then calls `make_policy` using dataset metadata. Inference runs through `predict_action` before mapping the returned vector to robot action features.
- Dataset/recording: `lerobot.record` uses `LeRobotDataset.create`, adds observation/action frames with `add_frame`, and finalizes episodes with `save_episode`. It records the action actually returned by `robot.send_action`. BodyTune should reuse this representation and add research provenance in an immutable sidecar/run manifest until a verified dataset-metadata extension is needed.

## Risks before hardware use

- The checkout is old relative to its remote-tracking branch, while the working hardware state may depend on its exact revision or unknown modifications.
- The three source edits are heterogeneous and unverified; changing revisions or cleaning the tree could alter camera selection, policy configuration/loading, and motor communication reliability.
- The pretrained SmolVLA reference is `lerobot/smolvla_base` at Hub revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` (model metadata inspected 2026-08-12). Compatibility with the pinned LeRobot revision still needs a non-hardware loading test before use.
- The shell's Python is Miniforge Python 3.10.14 and does not expose an installed `lerobot` distribution. The installed `lerobot-teleoperate` and `lerobot-record` commands instead use Homebrew Python 3.11 and an editable LeRobot 0.3.4 installation pointing at the inspected local checkout. BodyTune's uv environment must remain separate until this split setup is understood.
- No likely SO-101 USB serial device was present during the audit; only Bluetooth/debug-style `/dev/tty.*` entries were visible. Camera identifiers and actual teleoperation behavior have not been inventoried or tested.
