# Global one-symbol versus action-aligned local communication

## Protocol

Both runs use the original saved 1,200 training / 240 validation / 240 test
circles from `outputs/circles1`, without regeneration. Source split hashes,
ordered test displacements and receiver target trajectories were checked for
exact equality. The training source revision was clean `528b926` for both runs.

Each source trajectory has 96 states and 95 actions. Encoder inputs are only
the 95 end-effector displacement vectors computed from A's observed joint
states using FK. No joint increments, joint states, diameter labels, absolute
positions or explicit time indices enter the Notator.

For a parameter-controlled comparison, **both** arms use the same 2→64→64 ReLU
trunk and 64→8 head. Global encoding averages all 95 hidden vectors before
emitting one token; local encoding emits a token from each corresponding
vector. The local function is shared across steps, with no sequence context.
This new global pooling baseline replaces the earlier Transformer baseline in
both arms. It preserves scale information for fixed-phase circles but would
not preserve arbitrary motion order.

The Performer is the same 128-unit GRU in each arm: 119,874 parameters. Total
model size is 124,746 parameters. Initial weights match exactly (SHA-256
`263ecf05e00931e0312f6fb43de3c53d51f3bff2ab655274cd8e50f3360d47c0`).
During execution, the global symbol is repeated each step. The local policy
reads only the symbol for the current action, receiver state, recurrent state
and shared progress clock. Future symbols cannot affect earlier joint states;
this and repeated-displacement encoding consistency have regression tests.

Both start from scratch with seed 123, Adam lr 0.001, batch 64, clipping 1,
maximum 300 / minimum 80 epochs, and validation patience 40. The same drawing,
constraint and smoothness objective is used. There is no IK/action teacher,
auxiliary reconstruction objective, code-usage regularizer or predefined symbol
meaning. Training uses straight-through hard argmax with temperature
`max(0.4, 1 - 0.6*epoch/300)`; both early stops occur before the temperature
floor. All test execution uses serialized integer IDs.

## Results

| Encoding | Completed / selected epoch | Mean test NRMSE | Strict success | Used symbols | Distinct test messages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global, 1 token | 134 / 94 | 0.03334 | 8/240 = 3.33% | 2/8 | 2 |
| Local, 95 tokens | 235 / 195 | 0.01395 | 73/240 = 30.42% | 3/8 | 80 |

Both stopped under the predeclared validation patience rule. The realized
training epochs differ; the maximum budget and stopping rule match. All normal
and intervention rollouts satisfy sampled feasibility constraints. Strict
success retains ordered start-aligned RMSE / target path length ≤ 0.01.
Independent checkpoint reloads reproduce both test results.

| Local-model condition | Mean NRMSE | Strict success |
| --- | ---: | ---: |
| Original aligned score | 0.01395 | 30.42% |
| Same tokens, fixed permutation of action positions | 0.01822 | 8.75% |
| Score from another demonstration | 0.08175 | 1.67% |

Temporal permutation preserves each score's symbol counts and information
budget while changing alignment. The degradation demonstrates order-sensitive
execution in this model, not that the symbols have correct compositional
semantics. The global one-token permutation is an identity and reproduces its
normal result exactly. Cross-demonstration shuffling raises global mean NRMSE
to 0.07235 (2.5% strict success).

## What improved?

| Encoding | Mean fitted-diameter relative error | Radial residual / fitted radius | Closure gap / target diameter |
| --- | ---: | ---: | ---: |
| Global | 13.12% | 3.24% | 5.88% |
| Local | 3.42% | 2.82% | 4.31% |

The largest observed gain is in scale accuracy. Both models already draw
approximately circular paths. Circle-fit metrics are descriptive and do not
replace the ordered-trajectory success criterion.

The global encoder separates its test cases into two coarse diameter groups:
ID 5 has mean diameter 0.3421 (106 examples); ID 0 has mean 0.5759 (134).
This is substantial vocabulary underutilization, not an established limit of
one-token communication. The earlier Transformer/64-GRU experiment used five
symbols and achieved 17.92% success under a different architecture, input and
initialization. It must not be substituted into this matched-model table, nor
should this run's weak global baseline be generalized to all global encoders.

The local model uses IDs 1, 5 and 6. Its displacement scatter shows a mixture
of step-size and directional partitions, not eight clean direction primitives.
Token 1 has a strong negative-y association in this dataset; tokens 5 and 6
cover broader directions with different average magnitudes. These are
post-training associations, not labels supplied during learning. Independently
trained token IDs need not have the same meaning across the two arms.

## Interpretation limits

- Nominal capacity is **3 bits versus 285 bits**. Matching model weights does
  not match communication capacity. Locality and available information are not
  isolated by this two-arm experiment.
- Both codebooks are underused. Optimization and partial symbol collapse
  affect the comparison; one seed cannot establish the best global or local
  method.
- The encoder has no explicit clock, but displacement direction correlates
  with phase on these fixed-phase circles. The Performer also receives a
  shared phase clock. Order-sensitive execution is not proof of general local
  motion semantics.
- No held-out motion compositions, editable local substitutions, arbitrary
  shape transfer or cross-embodiment tests were run.

The result supports continued investigation of action-aligned communication.
It does not yet establish a compositional motion language or show that local
alignment alone explains the accuracy difference. Useful next controls include
a capacity-matched global sequence, better codebook utilization, and separating
the shared clock's contribution; these were not run here.

## Artifacts

Reproduce using the two `roboscore.aligned` commands in the README. Add
`--evaluate` to verify source hashes and reload a selected checkpoint. Run
`figures/aligned.py` after both runs to regenerate:

- `outputs/aligned128/aligned_inference.png/pdf`: six test examples selected
  by diameter quantiles, without ranking by error.
- `alignment_curves.png`: raw validation curves.
- `local_symbols.png`: source displacement/code associations and global
  diameter/code associations (diameter is evaluation metadata only).
- `local_scores.png`: complete 95-token scores for the same six examples.
- `comparison.json`: reports, fit diagnostics and descriptive code statistics.

Versioned run reports: [global](results/aligned128/global1.json) and
[local](results/aligned128/local95.json). Raw models, trajectories and generated
figures remain outside Git.
