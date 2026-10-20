# RoboScore: Learning an Executable Motion Language for Robots

> Can robots autonomously discover a compact, executable, and compositional motion language from demonstrations, enabling communication and transfer of motion across different initial configurations?

Inspired by guqin jianzipu (减字谱), RoboScore investigates learning reusable motion symbols and composition rules from continuous demonstrations, without supplying action labels or a grammar. The full research design is in [the proposal](docs/proposal.md).

## Research target

Two simulated, identical fixed-base planar 2-DOF arms communicate a drawing. A Notator observes A's joint and end-effector trajectory and emits a compact discrete score. A Performer receives only that score and B's current state, then generates joint increments.

“Same drawing” means the same ordered curve, scale and orientation after aligning starting pen positions in a shared planar frame. The entire translated drawing must be feasible for B. Initial configurations are different; arm morphology is unchanged. No real hardware, cameras, ROS, LeRobot or physics engine is required.

A motion language should be **abstract** across joint realizations, **executable** from different feasible states, **compositional** on unseen combinations, and **editable** with predictable local effects.

## Language versus trajectory compression

Compression minimizes representation size and reconstruction error. A short discrete trajectory code can achieve both without discovering reusable motion semantics. Cross-configuration execution alone is also insufficient: transmitting an end-effector curve and using inverse kinematics can pass that test.

Phase 1 establishes discrete communication (initially four tokens from a 16-symbol vocabulary, at most 16 nominal bits). Phase 2 tests language through held-out composition, token intervention, local edits, and context-compatible subprogram substitution against matched compression baselines. Distinguish **reading** an unseen demonstrated combination from **writing** it by recombining symbols without its demonstration.

## Current implementation scope

The first milestone is an analytic NumPy benchmark: forward/inverse kinematics, continuous feasible trajectories, and A/B reproduction through conventional IK. This is an oracle/control baseline with access to the full relative target curve, **not a learned score or evidence of a language**. Learned Notator/Performer models and matched bit-budget compression baselines are later work.

## Independent environment

Use uv with a dedicated environment; the old `.venv` and local robot files are not used:

```bash
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv sync --frozen
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv run pytest
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv run roboscore-benchmark --help
```

The baseline needs only CPU, NumPy and Matplotlib. PyTorch is deliberately deferred until model training. `uv.lock` pins the resolved environment. Generated outputs, virtual environments and machine-local files are excluded from Git.
