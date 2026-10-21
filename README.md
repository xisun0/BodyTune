# RoboScore: Learning an Executable Motion Language for Robots

> Can robots autonomously discover a compact, executable, and compositional motion language from demonstrations, enabling communication and transfer of motion across different initial configurations?

Inspired by guqin jianzipu (减字谱), RoboScore investigates learning reusable motion symbols and composition rules from continuous demonstrations, without supplying action labels or a grammar. The full research design is in [the proposal](docs/proposal.md).

## Research target

Two simulated, identical fixed-base planar 2-DOF arms communicate a drawing. A Notator observes A's joint and end-effector trajectory and emits a compact discrete score. A Performer receives only that score and B's current state, then generates joint increments.

“Same drawing” means the same ordered curve, scale and orientation after aligning starting pen positions in a shared planar frame. The entire translated drawing must be feasible for B. Initial configurations are different; arm morphology is unchanged. No real hardware, cameras, ROS, LeRobot or physics engine is required.

A motion language should be **abstract** across joint realizations, **executable** from different feasible states, **compositional** on unseen combinations, and **editable** with predictable local effects.

## Language versus trajectory compression

Compression minimizes representation size and reconstruction error. A short discrete trajectory code can achieve both without discovering reusable motion semantics. Cross-configuration execution alone is also insufficient: transmitting an end-effector curve and using inverse kinematics can pass that test.

Phase 1 establishes discrete communication. The original proposal suggested 4×16; the implemented fixed experiment uses 32 tokens from 8 symbols (96 nominal bits), followed by a separate EOS extension. Phase 2 tests language through held-out composition, token intervention, local edits, and context-compatible subprogram substitution against matched compression baselines. Distinguish **reading** an unseen demonstrated combination from **writing** it by recombining symbols without its demonstration.

## Current implementation scope

The first milestone is an analytic NumPy benchmark: forward/inverse kinematics, continuous feasible trajectories, and A/B reproduction through conventional IK. This is an oracle/control baseline with access to the full relative target curve, **not a learned score or evidence of a language**. Learned fixed-message and EOS-capable Notator/Performer models are now implemented below. Matched bit-budget compression baselines remain future work.

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


### Observed learning status

Both learning modes have completed training and checkpoint-reload evaluation. Fixed32 test mean NRMSE is 0.2075; EOS fine-tuning reaches 0.2049. **Both have 0% strict drawing success** at the 0.01 threshold. Although the EOS architecture supports 1–32 motion symbols, the selected trained checkpoint uses just one symbol for every test example (length collapse), with only four distinct test messages. This is not a useful variable-length motion notation yet. See [full results](docs/learning-results.md); the slight metric difference is not evidence that EOS helps, since the EOS run received 40 additional epochs of optimization.

## Single-example checkpoint inference

Run `.venv-roboscore/bin/python figures/inference.py` after both training runs. It reloads both checkpoints and uses the first circle in the saved test split (index 3), without selecting by error. The receiver gets only integer scores and its initial angles. Outputs in `outputs/inference/` include `example.json` (actual messages and errors), `inference.png` (aligned target/prediction) and `inference.gif` (full arm postures). No optimization or IK correction is performed during this inference.

## 96-token capacity experiment

```bash
.venv-roboscore/bin/roboscore-train --output outputs/fixed96 --tokens 96 --epochs 60
.venv-roboscore/bin/python -m roboscore.evaluate outputs/fixed96
ROBOSCORE_RUN=outputs/fixed96 .venv-roboscore/bin/python figures/learning.py
```

This experiment retains 8 symbols and the same 96-sample trajectories, seed, splits, optimizer and 60-epoch budget as fixed32. The channel increases from 96 to 288 nominal bits. **96 tokens does not impose one token per motion step**: learned attention queries still read the entire demonstration, and the Performer receives the complete score. The message projection and query parameter counts grow, so this is not a parameter-matched comparison. It tests a larger message capacity, not an explicit temporally aligned grammar.

New runs save `last.pt` as well as `best.pt`, including optimizer state, Torch and data-loader RNG states, epoch/history, validation-selection state and the temperature schedule horizon. `--init-checkpoint` remains a weights-only warm start with a new optimizer; it is not an exact-resume command. The stored state supports implementing a proper continuation without losing optimizer history later.

The first 96-token run completed 60 epochs: test mean NRMSE **0.2120**, strict success **0%**, versus **0.2075 / 0%** for fixed32. It used all eight symbols but produced only 71 distinct test messages (fixed32: 80). The best validation checkpoint was again the final epoch, so this is a budget-matched single-seed comparison, not a convergence result. [Full comparison](docs/learning-results.md).

## Simplified diameter-only circles: one 8-way token

```bash
.venv-roboscore/bin/roboscore-train --output outputs/circles1 --task circles --tokens 1 \
  --train-count 1200 --val-count 240 --test-count 240 \
  --epochs 300 --min-epochs 80 --patience 40
.venv-roboscore/bin/python -m roboscore.evaluate outputs/circles1
```

All curves are analytic circles sampled at 96 points, beginning at the rightmost point, traversed counterclockwise in a fixed shared orientation. Only diameter varies, uniformly drawn in [0.24, 0.70] link units before feasibility sampling. Each requested diameter is held fixed while A/B initial configurations are resampled until both whole paths are feasible. A/B configurations still differ; no real hardware is involved. The relative circle is `(d/2)*(cos(theta)-1, sin(theta))`. This isolates scale from orientation and drawing-direction variations.

There are **1,200 training demonstrations**, plus 240 independent validation and 240 test demonstrations (seeds 123/124/125). `train_demonstrations.npz`, `val_demonstrations.npz`, and `test_demonstrations.npz` save full A/B joint paths, relative pen paths, family, time step and exact diameter. Shape/diameter metadata are not passed to the model. A single token has eight choices (3 nominal bits), so continuous diameters must share approximate representations; eight diameter classes are not supplied as labels.

The Notator/Performer architecture and geometry objective are retained with `tokens=1`. At least 80 epochs run; thereafter 40 epochs without a strictly improved validation selection score trigger early stopping, with a maximum budget of 300. Both the completed and selected epoch are reported. The test set is used only after validation selection. This tests scale communication, not motion composition or a grammar.

First circle-only result: stopped at epoch **254**, selected epoch **214**. Test mean NRMSE **0.01691**, strict success **17.92%**, all rollouts feasible. Shuffling the message raises mean NRMSE to **0.07952**. The model uses five symbols associated with different diameter ranges; it has learned a coarse scale communication but has not solved accurate continuous-diameter reproduction. [Detailed circle results](docs/circle-experiment.md).

Run `.venv-roboscore/bin/python figures/circles.py` to generate the training preview, a CSV index of all 1,200 demonstrations, test inference plots, diameter/token associations, and same-state token interventions. Artifacts are written to `outputs/circles1/`; `ROBOSCORE_RUN` selects another circle run.

## Exact-diameter Performer capacity control (no Notator)

```bash
.venv-roboscore/bin/python -m roboscore.capacity --source outputs/circles1 \
  --output outputs/diameter_capacity/gru64 --hidden 64
.venv-roboscore/bin/python -m roboscore.capacity --source outputs/circles1 \
  --output outputs/diameter_capacity/gru128 --hidden 128
.venv-roboscore/bin/python -m roboscore.capacity --source outputs/circles1 \
  --output outputs/diameter_capacity/gru256 --hidden 256
```

These controls read the **same saved 1,200/240/240 circle demonstrations**, without regeneration. SHA-256 hashes identify every source split. The receiver gets the exact scalar diameter in link units and its own initial joint angles; there is no Notator, discrete token, source trajectory input, IK correction or hand-coded circular controller. A scalar Linear–Tanh conditioner replaces the token embedding, then reuses the original GRU rollout, time inputs, bounded action head and geometry/constraint/smoothness objective.

Each width (64/128/256) starts from scratch with seed 123, Adam lr=0.001, batch size 64, clipping 1, CPU four threads, maximum 300 epochs, minimum 80 and validation patience 40. The larger hidden state also enlarges its conditioning, initialization and action layers. Epoch and stopping budgets match; parameter counts, initialization realizations and FLOPs do not. This is a single-seed diagnostic of the controller, not a test of autonomous notation discovery.

Validation selects the checkpoint using the same mean NRMSE plus infeasible fraction criterion. Test evaluation includes correct diameters, shuffled diameters and a fixed training-mean diameter. `best.pt`/`last.pt` retain optimizer/RNG state; `history.csv`, `test_rollouts.npz` and `report.json` record results and source provenance. To reload and verify a finished run, add `--evaluate` to its command; dataset hash changes cause refusal rather than evaluating on different data.

After all three capacity runs finish, `.venv-roboscore/bin/python figures/capacity.py` verifies matching dataset hashes and test examples, then saves comparison JSON, validation curves and diameter-ordered inference overlays under `outputs/diameter_capacity/`. Additional least-squares circle fits report circularity, diameter error, closure and completed turns separately from the original ordered-trajectory success criterion; degenerate paths are excluded from fit-dependent metrics and counted explicitly.

Completed exact-diameter capacity results (same saved 1,200 training circles): GRU64 **0.008323 / 79.58%**, GRU128 **0.005544 / 92.92%**, GRU256 **0.004682 / 92.92%** (test mean NRMSE / strict success). All correct-diameter rollouts were feasible. Widths 64/128 ran 300 epochs; 256 stopped at 224 under validation patience. Capacity improves precision, but the 64-dimensional controller already learns accurate execution for most examples when the exact diameter is available. The earlier problem is not solely model size. [Detailed capacity results and limitations](docs/performer-capacity.md).

## Global 1 token versus 95 action-aligned local tokens (GRU128)

```bash
.venv-roboscore/bin/python -m roboscore.aligned --source outputs/circles1 \
  --output outputs/aligned128/global1 --mode global
.venv-roboscore/bin/python -m roboscore.aligned --source outputs/circles1 \
  --output outputs/aligned128/local95 --mode local
```

Both arms read the **same saved 1,200/240/240 circle splits**, with identical hashes. The source observation is the 95 successive **end-effector displacements**, computed by FK from A's recorded joint states. The encoder never receives joint increments, A/B joint angles, diameter labels, absolute endpoint positions or time indices. A fixed 0.05-link-unit scale normalizes each displacement.

For this new matched comparison, both Notators use the same pointwise 2→64→64 ReLU trunk and 64→8 head. The global arm averages the 95 hidden feature vectors before emitting one symbol. The local arm applies the head at each step to emit 95 symbols; each local symbol depends only on its corresponding displacement. This replaces the older Transformer baseline in **both** arms so their parameter shapes/counts and seeded initial weights match exactly. The global pooling baseline is appropriate to this fixed-phase circle task, not an order-preserving encoder for arbitrary drawings.

Both Performers are the same 128-unit GRU with an 8-symbol embedding. At action step t, the global model reads its one symbol again; the local model reads **only symbol t**, plus current receiver state, shared progress and recurrent state. Initialization uses the receiver state and the first available symbol. Future tokens cannot influence earlier actions. The exact-diameter baseline is not used for initialization or inference.

Both train from scratch with seed 123, Adam lr=0.001, batch size 64, clipping 1, maximum 300/minimum 80 epochs, validation patience 40, the same geometry/constraint/smoothness objective and deterministic straight-through discrete training. No teacher actions, token labels, circle primitives or extra losses are introduced. Evaluation transmits integer IDs only, and includes cross-demonstration message shuffling and a fixed temporal permutation (the latter is an identity control for the one-token arm). Add `--evaluate` to reload a completed run and verify source hashes.

**Interpretation boundary:** the global channel has 3 nominal bits; the local channel has 285. Even with identical model parameters, this compares alignment **and** message capacity, not locality alone. The Performer has a shared phase clock, and the dataset contains only fixed-phase circles. Local token input restrictions do not themselves establish reusable motion semantics, composition or editing.

After both runs, `.venv-roboscore/bin/python figures/aligned.py` checks exact agreement of source hashes, ordered test data, model parameter counts and initial weight hashes. It saves target/execution overlays, validation curves, local displacement/code scatter plots, score sequences and descriptive code statistics under `outputs/aligned128/`. The one-token step-permutation control must reproduce its normal result exactly. Symbol statistics are observational summaries, not supplied meanings or proof of compositionality.

Completed matched GRU128 result: global1 **0.03334 / 3.33%**, local95 **0.01395 / 30.42%** (test mean NRMSE / strict success). Permuting the local score's action positions changes performance to **0.01822 / 8.75%**, so order affects execution. The global codebook used only 2/8 symbols and local used 3/8. The global baseline is underutilized, message budgets differ by 95×, and these circle-only results do not establish compositional semantics. [Full aligned-score report](docs/aligned-comparison.md).

## Fixed eight-bin diameter control (GRU128)

```bash
.venv-roboscore/bin/python -m roboscore.aligned --source outputs/circles1 \
  --output outputs/aligned128/binned8 --mode binned
.venv-roboscore/bin/python -m roboscore.aligned --source outputs/circles1 \
  --output outputs/aligned128/binned8 --mode binned --evaluate
.venv-roboscore/bin/python figures/binned.py
```

This diagnostic bypasses the Notator. The true scalar diameter is mapped to one of eight **fixed equal-width bins** on the predeclared generation interval [0.24, 0.70], width 0.0575. Internal boundaries belong to the upper bin; the right endpoint belongs to bin 7. Only a single integer bin ID enters the Performer: neither the exact diameter nor its within-bin residual reaches the controller. Bin IDs have no imposed numerical relationship in the learned embedding.

The GRU128 Performer, including its seeded initial weights, matches the learned global1 arm exactly. The original 1,200/240/240 files, Adam lr=0.001, batch64, clip1, maximum300/minimum80 epochs, validation patience40 and rollout objective are retained. Only the learned encoder is removed from optimization. Tests check all eight bin centers, endpoints, identical Performer initialization, train/reload equivalence and dataset hash checks. This measures achievable performance for a fixed oracle quantizer under one training run, not a theoretical success ceiling or an autonomously learned notation.

`figures/binned.py` compares executions with the learned global baseline and computes an ideal bin-midpoint-circle reference. That reference excludes controller errors and kinematic feasibility checks; it is not an upper bound over bin choices, representative diameters or receiver-state-dependent policies.

Completed fixed-bin result: **NRMSE 0.01158, strict success 117/240 = 48.75%, feasible100%, all8 IDs used**; shuffled bins reduce success to7.5%. Training stopped at242 with best epoch202; checkpoint reload reproduces results. The ideal bin-midpoint-circle reference gives72.5% geometric success without controller/feasibility checks; neither number is a theoretical upper bound. This substantially improves on learned global1 (3.33%) while trailing the exact-continuous-diameter diagnostic (92.92%). [Protocol, collapse interpretation and results](docs/diameter-bins.md).

## TCN + VQ Translator, GRU128 Performer

```bash
.venv-roboscore/bin/python -m roboscore.vq --source outputs/circles1 \
  --output outputs/tcn_vq
.venv-roboscore/bin/python -m roboscore.vq --source outputs/circles1 \
  --output outputs/tcn_vq --evaluate
```

The Translator observes the same 95 A end-effector displacements, normalized by 0.05. A 2→64 pointwise input projection, three causal residual Conv1D layers (kernel3, dilations1/2/4; receptive field15 actions), and a 64→8 projection produce one 8D latent per action. Nearest-neighbor VQ assigns each latent to one of **eight learned 8D codewords**, yielding a fixed 95-ID motion score. There are no diameter labels, prescribed directions, joint-increment inputs, absolute positions or explicit time inputs to the Translator. The GRU128 Performer retains its architecture and receives current code, receiver joint state, progress and recurrent state. Its eight embedding vectors are also the VQ codebook. Integer-ID decoding is tested against the quantized forward path; continuous encoder residuals are never transmitted. There is no Transformer in this experiment.

Train-only deterministic k-means initializes the codebook from initial TCN features. Forty reconstruction pretraining epochs optimize the Translator, codebook and auxiliary 8→64→2 displacement decoder; GRU weights remain untouched. Joint training then uses the original rollout objective plus `0.01 * (normalized displacement reconstruction MSE + VQ loss)`. VQ loss is `MSE(code, stopgrad(latent)) + 0.25*MSE(latent, stopgrad(code))`, with a straight-through quantized latent for backward propagation. Reconstruction is a training auxiliary, not an inference input. No explicit balancing, dead-code restart or semantic supervision is used; code collapse remains possible and is measured.

The immutable 1,200/240/240 circle splits and hashes are reused. Both stages use Adam0.001, batch64, clip1 and seed123; the joint optimizer and data-shuffle seed restart after pretraining. Joint training runs maximum300/minimum80 epochs with validation patience40 and the existing selection criterion. Reports include code counts/perplexity, circle-fit diagnostics, cross-demonstration score shuffling and within-score temporal permutation. Checkpoints retain optimizer/RNG state, and evaluation reload checks source hashes. This adds a TCN, VQ objectives and pretraining, so comparisons with previous models are not parameter/compute matched. The 95-token/285-bit, fixed-phase, circle-only setup does not establish compositionality or editing.
