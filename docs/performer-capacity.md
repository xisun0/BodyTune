# Performer capacity with an exact diameter

The question is whether the earlier circle errors primarily reflected an
undersized controller. This experiment removes the Notator and discrete
communication bottleneck, then compares GRU widths within one continuous-input
controller family.

## Controlled protocol

All models read the original `outputs/circles1` saved demonstration files:
1,200 training, 240 validation, 240 test circles. No trajectories were regenerated.
The three SHA-256 split hashes and ordered test arrays must match before the
comparison script will run. The input is the exact diameter in link units plus
B's initial/current joint state and shared execution progress. There is no
source-trajectory input, IK correction, teacher-forced joint trajectory or
hand-coded circle primitive in the Performer.

Each model starts from scratch with seed 123, Adam lr 0.001, batch size 64,
gradient clipping 1, four CPU threads, 300 maximum epochs, 80 minimum epochs and
validation patience 40. The original geometry, joint-limit, singularity and
smoothness objective and bounded increments are reused. Validation selection
uses mean NRMSE plus infeasible fraction; test trajectories are evaluated only
after selection. Models share the rollout implementation; the old token model's
checkpoint was reevaluated unchanged after refactoring.

Hidden width scales the conditioning, recurrent, initialization and action
layers together. Different widths have different parameter counts and random
weight realizations; epoch budgets match, FLOPs do not. All results use a single
seed and one learning rate. These controls diagnose the policy and cannot prove
that a particular width is optimal under all training conditions.

Strict success remains feasible sampled execution and ordered start-aligned
RMSE / target path length ≤ 0.01. Circle-fit diagnostics are separate: relative
radial residual, fitted diameter error, closure gap and turns completed.
They do not replace the original success criterion.

## Results

All three runs started from clean source revision `51a44c9`. All test rollouts
with correct diameters satisfied the sampled feasibility constraints. Checkpoint
reload evaluation reproduced the recorded metrics for every width.

| GRU hidden size | Parameters | Completed / selected epoch | Test mean NRMSE | Strict successes |
| --- | ---: | ---: | ---: | ---: |
| 64 | 30,786 | 300 / 294 | 0.008323 | 191/240 = 79.58% |
| 128 | 118,914 | 300 / 296 | 0.005544 | 223/240 = 92.92% |
| 256 | 467,202 | 224 / 184 | 0.004682 | 223/240 = 92.92% |

The 256 run stopped after 40 epochs without a validation-selection improvement.
The other runs exhausted the 300-epoch budget. This is a matched stopping-rule
comparison, not a statement that all models converged. Jobs ran concurrently;
recorded wall times reflect resource contention and are not controlled speed
benchmarks.

| Width | Mean radial residual / fitted radius | Mean fitted-diameter relative error | Mean closure gap / target diameter |
| --- | ---: | ---: | ---: |
| 64 | 1.955% | 1.683% | 2.810% |
| 128 | 1.325% | 0.842% | 2.050% |
| 256 | 1.144% | 0.772% | 1.838% |

All 240 trajectories per model admitted valid circle fits. These averages
separate shape, size and closure; a good fitted circle alone does not establish
correct ordered execution.

## Conditioning controls

| Width | Correct diameter NRMSE | Shuffled diameter NRMSE | Fixed training-mean diameter NRMSE |
| --- | ---: | ---: | ---: |
| 64 | 0.008323 | 0.081192 | 0.064747 |
| 128 | 0.005544 | 0.080913 | 0.064212 |
| 256 | 0.004682 | 0.080637 | 0.063919 |

The policies use the provided diameter rather than merely drawing an average
circle. One shuffled-diameter rollout for width 128 was constraint-invalid;
the controls deliberately alter the requested size while keeping each original
receiver state and evaluation target.

## Interpretation

1. **64 hidden units are sufficient to learn substantial circle control:** with
   accurate continuous conditioning, the policy meets the strict criterion on
   almost 80% of this test set. Earlier failures cannot be explained as an
   absolute inability of a 64-dimensional GRU to draw circles.
2. **Capacity improves precision in this run:** increasing to 128 reduces mean
   error and raises strict success by 13.33 percentage points. Increasing to
   256 further reduces mean error, circularity and closure errors, while strict
   success remains 223/240. The larger model has about four times as many
   parameters as the 128 model.
3. **The earlier communication setup has additional bottlenecks:** its
   64-dimensional controller jointly trained with a one-token Notator achieved
   NRMSE 0.01691 and 17.92% success. That difference is not a pure quantization
   estimate: conditioning, joint optimization, initialization and selected
   training epoch also differ. These controls do not apportion their effects.
4. **Optimization remains a concern:** validation curves contain substantial
   rebounds at constant learning rate. One seed and one optimizer configuration
   do not establish a universally optimal width or exclude more effective
   training of the smaller model. A 128-dimensional policy is a reasonable
   working baseline here, rather than an established optimum.

## Reproduction and artifacts

See the three commands in the README; they read the same saved source files and
refuse to overwrite run directories. Add `--evaluate` to each command to reload
its selected checkpoint, verify source hashes and recompute test metrics.

`figures/capacity.py` produces `outputs/diameter_capacity/comparison.json`,
`validation_curves.png/pdf`, and `capacity_inference.png/pdf`. The latter uses
six diameter-ordered test examples, without choosing by error. Reports are
versioned here: [64](results/diameter-capacity/gru64.json),
[128](results/diameter-capacity/gru128.json),
[256](results/diameter-capacity/gru256.json).

This experiment supplies the diameter explicitly. It does not learn a motion
notation, demonstrate compositionality or address cross-embodiment transfer.
