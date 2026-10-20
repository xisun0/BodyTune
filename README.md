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

The first milestone is an analytic NumPy benchmark: forward/inverse kinematics, continuous feasible trajectories, and A/B reproduction through conventional IK. This is an oracle/control baseline with access to the full relative target curve, **not a learned score or evidence of a language**. A learned fixed-message Notator/Performer is now implemented below. Matched bit-budget compression baselines remain future work.

## Independent environment

Use uv with a dedicated environment; the old `.venv` and local robot files are not used:

```bash
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv sync --frozen
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv run pytest
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv run roboscore-benchmark --help
```

The analytic baseline needs only CPU, NumPy and Matplotlib; the independent environment now also pins PyTorch for model training. `uv.lock` pins the resolved environment. Generated outputs, virtual environments and machine-local files are excluded from Git.

## Run the analytic benchmark

From the repository root (each run needs a new output directory):

```bash
.venv-roboscore/bin/pytest -q
.venv-roboscore/bin/roboscore-benchmark --output outputs/baseline --count 1200 --seed 42
.venv-roboscore/bin/python figures/benchmark.py
```

The default is 96 samples at 0.05 seconds per step. Six drawing families are balanced: square, rectangle, triangle, circle, arc and polyline. Scale, orientation, direction, initial positions and elbow branches vary. Fixed seeds reproduce sampling. Rejection sampling accepts a pair only if both complete trajectories meet joint bounds, per-joint speed limits and a singularity margin. Accepted initial joint vectors differ by at least 0.3 radians in Euclidean norm. Rejected attempts are counted; feasibility conditioning means the accepted distribution is not uniform over all curves and initial states.

- **Full-curve IK oracle:** transmit A's full relative pen curve, solve on B's initial elbow branch, then integrate joint-increment commands from B's state.
- **Joint-copy control:** apply A's joint increments directly from B's initial configuration. Invalid executions count as failures and are not drawn as valid paths.

Evaluate the actual integrated rollout using start-aligned, ordered sample positions. Success requires a feasible rollout and positional RMSE divided by target path length ≤ 0.01. No rotation, rescaling, or dynamic time warping is applied. Feasibility is checked at sampled times; continuous-time dynamics, contact, collision and acceleration bounds are outside this benchmark.

`outputs/baseline/` contains the paired demonstrations and executed trajectories (`trajectories.npz`), per-pair metrics (`metrics.csv`), and configuration, rejection counts, aggregate metrics, Git revision/dirty state and lockfile hash (`report.json`). Raw trajectories and shape labels are baseline/evaluation data; they are not an approved input interface for a future learned Performer. These samples are not yet a structural train/validation/test split.

The figure script writes `figures/baseline.png`, vector `baseline.pdf`, and `arms.png`/`arms.pdf`. To visualize a different run, set `ROBOSCORE_RUN=outputs/your-run`. Figures show aligned geometry and an example of distinct A/B configurations. Generated artifacts remain untracked; summarized reproducible results belong in `docs/`.

## Verified baseline result

The first clean-revision run evaluated 1,200 pairs (seed 42): the full-curve IK oracle reproduced all drawings to numerical precision; joint-increment copying met the drawing criterion in none of the pairs, with 16 constraint-invalid rollouts. See [results and limitations](docs/baseline-results.md) and the [machine-readable report](docs/results/baseline-seed42.json). This establishes the analytic reproduction pipeline, not a learned motion language.

## Full arm posture animation

```bash
.venv-roboscore/bin/python figures/animate.py
```

This renders a feasible circle example as three synchronized panels: A, B with the full-curve IK oracle, and B copying A's joint increments. Orange links show the complete base–elbow–tip posture; blue traces show the executed pen path; gray curves show each robot's intended drawing. All panels use the same world-coordinate scale. Joint angles and simulation time are displayed. Outputs are `figures/arm_motion.gif`, a self-contained `arm_motion.html` with playback/scrubbing controls, and `posture_midpoint.png`. Set `ROBOSCORE_RUN` to choose another run.

Frames use recorded joint states at the run's sampling interval, with no dynamics or interpolated physics simulation. The oracle receives the full curve, not learned symbols.

## Learned fixed-length motion communication

```bash
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv sync --frozen
.venv-roboscore/bin/roboscore-train --output outputs/fixed32 --epochs 60
```

The Notator receives 96 steps of `[q1/pi, q2/pi, velocity1/2, velocity2/2, relative_x/2, relative_y/2]`. A 2-layer Transformer (hidden 64, 4 heads, feedforward 256, dropout 0) and 32 learned attention queries emit **32 tokens from 8 symbols** (96 nominal bits). Training uses deterministic straight-through argmax with a temperature-annealed soft backward approximation; inference sends exact integer symbols. This differs from the proposal's optional stochastic Gumbel estimator.

The Performer embeds the ordered message into a 64-dimensional context. A 64-dimensional GRUCell receives that context, sine/cosine of B's current angles, and normalized execution progress. It emits bounded increments (at most 0.1 rad per joint at dt=0.05 s). The recurrent rollout starts at B's actual initial state. Neither A's trajectory, target coordinates, shape labels nor B's target actions enter the Performer. The known fixed time horizon is shared protocol information.

Joint training uses differentiable unit-link forward kinematics and start-aligned, path-length-normalized squared drawing error, plus joint-limit, singularity-margin and weak acceleration penalties. No IK action imitation or primitive labels are used. Velocity is hard-bounded by the action head; other constraints are penalized in training and checked strictly in evaluation.

Train/validation/test use separate seeded trajectory draws (default 1,200/240/240). These are IID parameter variations of the same six shape templates, **not held-out structural compositions**. Select the best checkpoint by validation mean NRMSE plus infeasible fraction, then evaluate test data once. Compare against shuffled messages and a stationary receiver. A fixed 32-token representation does not assign token positions to motion segments and does not establish grammar.

Runs save `best.pt`, `history.csv`, `test_rollouts.npz` (hard scores and actual joint rollouts), and `report.json` with split seeds, configuration, checkpoint epoch, parameter count, metrics and source provenance. Default execution is CPU with four threads. Generated models and data remain outside Git.

Initial learned results and limitations: [learning experiments](docs/learning-results.md).

## EOS-terminated variable motion scores

```bash
.venv-roboscore/bin/roboscore-train --output outputs/eos32 --variable \
  --init-checkpoint outputs/fixed32/best.pt --epochs 40 --length-weight 0.01
.venv-roboscore/bin/python -m roboscore.evaluate outputs/eos32
ROBOSCORE_RUN=outputs/eos32 .venv-roboscore/bin/python figures/learning.py
```

The payload still uses **8 motion symbols (IDs 0–7)**. EOS is a separate control symbol (ID 8), and `-1` is storage padding, never a transmitted motion symbol. Scores contain 1–32 motion symbols followed by exactly one EOS; reaching the capacity forces a terminal EOS. A batch stores at most 33 entries. The Performer discards EOS and padding, and reconstructs the same zero-padded payload representation solely from the transmitted integers. All evaluation goes through this integer transport interface.

This first variable-length extension predicts candidate slots in parallel and truncates at the **first learned EOS**. It is not an autoregressive score decoder. A differentiable straight-through survival mask blocks every slot at/after EOS, so neither content nor gradients through motion embeddings can bypass termination. Soft termination probabilities provide gradient information to the EOS decision. A length penalty (`weight × payload_length/32`) trades drawing error against shorter messages. Report actual hard-message length histograms; supporting EOS does not guarantee that training discovers useful variable lengths.

The optional warm start copies the fixed model and adds an EOS output row. Variable training is additional optimization with a fresh optimizer, not an equal-compute comparison with fixed training. Training and validation selection include the length term. Termination length itself carries information; a shorter average payload is not a complete entropy/bit-budget comparison. `--variable` also supports training from scratch.

Reload evaluation for either mode with `.venv-roboscore/bin/python -m roboscore.evaluate outputs/fixed32`. This reloads `best.pt`, reads only the saved test demonstrations/receiver initial states, and recomputes integer-message rollouts; saved predictions are not reused as results.
