# RoboScore: Learning an Executable Motion Language for Robots

## 1. Research Objective

**Goal.** Investigate whether robots can learn a compact, executable, and compositional symbolic notation for motion directly from demonstrations, analogous to *jianzipu* (古琴减字谱), without being given predefined action tokens or a grammar.

The target is **end-effector drawing equivalence**, not replication of the demonstrator's joint-angle trajectory. A notation should ideally support:

- **Abstraction:** different joint-space executions can realize the same end-effector drawing.
- **Executability:** another robot can interpret the notation and execute it from a different feasible initial configuration.
- **Compositionality:** reusable symbols can be combined to express unseen drawings.
- **Editability:** modifying part of a score produces predictable changes to the drawing.

**Core question:** Can a model induce an executable, compositional motion notation from robot trajectories, rather than merely compressing them?

## 2. Minimal Experimental Setup

Two **identical fixed-base planar 2-DOF arms** play different roles:

1. **Robot A / Demonstrator:** executes a drawing and produces a joint-space trajectory with end-effector observations.
2. **Notator:** observes A's demonstration and encodes it into a short discrete **motion score**.
3. **Robot B / Performer:** receives only the score and its own current state, then produces joint commands to draw the same end-effector shape.

A and B can have different initial joint configurations, but **the same arm morphology**. Both initial configurations and the intended drawing must be jointly feasible. We do **not** require B to reproduce A's joint angles or motion style.

For the initial experiment, “same drawing” means the same sampled end-effector curve, scale, orientation, and drawing order **after aligning the starting pen positions** in the shared planar drawing frame. Different joint configurations do not themselves imply a rotation of that frame.

### Planar-arm kinematics

With joint angles \(q=(q_1,q_2)\) and link lengths \(l_1=l_2=1\):

\[
x=l_1\cos q_1+l_2\cos(q_1+q_2),\qquad
y=l_1\sin q_1+l_2\sin(q_1+q_2).
\]

| Item | Initial setting |
|---|---|
| Embodiment | Identical planar 2-DOF arms |
| Joint action | \((\Delta q_1,\Delta q_2)\) |
| Observations | Joint angles, end-effector coordinates |
| Trajectory length | 64–128 steps |
| Simulator | Analytic kinematics in NumPy/PyTorch |
| Training hardware | Mac CPU or Apple Silicon MPS |
| Physics engine | Not needed for the first PoC |

Data generation and rollout must enforce joint limits, velocity limits, continuous IK branches, and reachable workspaces. Drawing from different IK branches is useful when feasible, but branch transitions must not be forced through discontinuities.

## 3. Dataset

Programmatically generate planar curves, then construct feasible arm demonstrations using inverse kinematics and trajectory interpolation. Training observations contain **joint and end-effector trajectories only**; the model does **not** receive shape names, primitive labels, or source programs.

Initial drawing families:

- Squares, rectangles, triangles, circles, and circular arcs.
- Polylines and curves assembled from multiple line/arc segments.
- Variations in scale, location, orientation, and drawing order.
- The same end-effector drawing realized with different feasible initial joint configurations.

Fourier-generated curves may be used for additional out-of-distribution evaluation and as a signal-representation baseline. A finite Fourier series can approximate a broad class of curves, but this alone does not imply a compositional motion grammar.

**Starting dataset:** approximately 5,000–10,000 trajectories, with disjoint validation and test splits. Reserve **entire structural combinations**, not just random trajectories, for compositional evaluation. Ensure that similar initial states can correspond to different target drawings, so the Performer cannot infer the intended drawing from its initial state alone.

## 4. Model Architecture

### Notator: trajectory to discrete motion score

\[
m=N_\phi(\tau_A).
\]

- Small Transformer encoder, initially 2 layers, hidden dimension 64.
- Input: sampled \((q_t,\dot q_t,x_t,y_t)\) from Robot A.
- Output: initially 4 discrete tokens, each from a vocabulary of 16 symbols.
- Straight-through Gumbel-Softmax is one practical way to train the discrete communication channel.

The fixed-length message has a **maximum nominal capacity of 16 bits** (4 × log₂16). This is a communication constraint, not evidence of linguistic structure. The first PoC tests communication; variable-length scores and explicit grammatical structure can follow in Phase 2.

### Performer: motion score to joint actions

\[
\hat q_{t+1}=P_\theta(m,\hat q_t,h_t),
\qquad \hat p_t=FK(\hat q_t).
\]

- GRU or small Transformer policy.
- Input: score, B's current joint state, and recurrent state.
- Output: bounded joint increments \((\Delta q_1,\Delta q_2)\).
- Differentiable forward kinematics maps predicted joint states to pen positions for training.

The Performer does **not** receive A's original trajectory or source-program labels.

## 5. Training

Train Notator and Performer jointly, with Robot B's feasible initial joint state varied independently of A's demonstration.

The main objective is drawing equivalence in the end-effector workspace, not joint-space reconstruction:

\[
\mathcal L = \lambda_d\mathcal L_{\mathrm{drawing}} +
\lambda_s\mathcal L_{\mathrm{smooth}} +
\lambda_c\mathcal L_{\mathrm{constraint}}.
\]

- **Drawing loss:** compare B's executed pen trajectory with A's target curve after aligning the starting pen positions; preserve order and scale. Use arc-length resampling or progress alignment when timing differs.
- **Smoothness loss:** penalize excessive joint-velocity changes or accelerations, without rewarding trivial non-movement.
- **Constraint loss:** penalize joint-limit violations and excessive joint speeds; enforce hard bounds during rollout where practical.

For the initial differentiable experiment, unroll B's predicted joint increments and compute pen positions using analytic forward kinematics. Evaluate separately using a constrained rollout that checks feasibility and drawing accuracy.

**Important limitation:** this training objective still allows the score to encode a compressed end-effector curve. Successful communication alone does **not** establish that the learned symbols form a language.

## 6. Evaluation: Language Versus Compression

| Evaluation | Research purpose |
|---|---|
| Drawing accuracy | Can B reproduce the end-effector curve? |
| Cross-configuration execution | Can the same score be executed from different feasible initial joint configurations? |
| Token intervention | Does changing a token produce stable, predictable changes across contexts? |
| Held-out composition | Can learned symbols express unseen combinations of familiar drawing structures? |
| Local score editing | Can part of a drawing be changed without disrupting unrelated parts? |
| Subprogram substitution | Can equivalent motion expressions be exchanged inside a longer score, under compatible state conditions? |
| Matched compression baselines | Do learned scores generalize or edit better than ordinary encodings at comparable bit budgets? |

### Baselines

- Quantized Fourier coefficients or PCA coefficients, with a suitable decoder/controller.
- VQ-based trajectory autoencoder with matched communication capacity and comparable model complexity.
- Optionally, piecewise Fourier, splines, or hierarchical compression models as stronger baselines for local editing.
- Direct transmission of a compressed end-effector curve followed by a conventional inverse-kinematics controller.

**Interpretation:** Different initial configurations are only a basic transfer check. A simple task-space trajectory representation plus inverse kinematics could also pass. The stronger evidence for a *motion language* is systematic **compositional generalization, editable local semantics, and context-sensitive substitution**, especially when compared against equally constrained compression baselines.

### Important evaluation separation

- **Reading test:** can the Notator encode a demonstration with a previously unseen combination?
- **Writing test:** can existing learned symbols be recombined to produce that combination **without providing a demonstration of the held-out combination**?

The writing test is the stronger test of generative compositionality. Freeze the learned representation for the intervention evaluation and avoid selecting models based on the held-out editing results.

## 7. Project Phases

### Phase 1 — Motion Communication

**Objective:** Establish that the discrete communication channel can transfer end-effector drawings between identical 2-DOF arms with different feasible initial joint states.

Tasks:

1. Implement planar forward/inverse kinematics and feasibility-aware curve generation.
2. Generate joint-space demonstrations for multiple drawing families.
3. Train a small Notator–Performer model with a 4-token discrete bottleneck.
4. Evaluate drawing accuracy and cross-configuration execution.
5. Visualize triplets: **A's drawing → learned score → B's drawing**.
6. Compare against a matched trajectory-compression baseline.

**Deliverable:** Reproducible Mac-runnable training code, trained model, drawing visualizations, and initial benchmark metrics.

**Success criterion:** Reliable drawing transfer under feasible changes in B's initial joint configuration. This demonstrates communication, **not yet grammar discovery**.

### Phase 2 — Grammar Discovery

**Objective:** Test whether the learned representation supports reusable, compositional motion semantics beyond trajectory compression.

Tasks:

1. Extend from fixed-length messages to multi-segment, potentially variable-length scores.
2. Train on structured combinations of lines, arcs, turns, and repeated motifs without exposing their source-program labels.
3. Hold out entire combinations for testing.
4. Freeze models and perform token replacement, insertion, deletion, and subprogram substitution.
5. Evaluate predictable local effects, compositional generalization, and robustness across feasible initial states.
6. Compare against Fourier, VQ, and stronger structured-compression baselines under matched information budgets.

**Deliverable:** Symbol-intervention visualizations, held-out composition results, baseline comparisons, and an empirical characterization of any discovered notation.

**Success criterion:** Evidence that score elements have stable, reusable, context-sensitive compositional effects—not merely that the model reconstructs drawings well.

## 8. Scope and Non-Goals

**In scope:** Identical 2-DOF planar arms, different initial joint configurations, end-effector drawing equivalence, emergent discrete communication, compositionality and editability tests.

**Out of scope for now:** Cross-embodiment transfer, changing link lengths across robots, few-shot adaptation to new morphologies, real-hardware deployment, contact-rich manipulation, and general-purpose robot control.

## 9. Working Research Hypothesis

> Robots can learn an executable symbolic notation from motion demonstrations that transfers end-effector drawing behavior across feasible initial joint configurations. If the notation supports systematic recombination, editing, and contextual substitution of learned symbols—and outperforms matched compression baselines on these tests—it provides evidence for a compositional motion abstraction rather than mere trajectory coding.

**One-sentence summary:** RoboScore asks whether two identical 2-DOF robot arms can develop a reusable, executable “motion score” for communicating and composing end-effector drawings, analogous to how musical notation specifies a performance without encoding every physical movement.
