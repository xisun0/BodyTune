# Learned communication experiments

## Fixed 32 tokens, 8 symbols

Source: clean commit `f24814d`. Run: `outputs/fixed32`, seed 123, 60 epochs,
1,200 training / 240 validation / 240 test trajectories. Test data are IID draws
from the same six parameterized templates, not held-out compositions.

| Test condition | Feasible | Mean NRMSE | Success (NRMSE ≤ .01 and feasible) |
| --- | ---: | ---: | ---: |
| Learned hard message | 100% | 0.2075 | 0% |
| Shuffled message | 100% | 0.3152 | 0% |
| Stationary receiver | 100% | 0.2868 | 0% |

The hard channel produced 80 distinct test messages and used 7 of the 8 symbols.
Correct messages improve geometry relative to the two controls, but drawing
accuracy remains poor and the strict task is unsolved. This is a working
training/evaluation pipeline with preliminary message-dependent behavior, not
successful motion-language discovery. One seed does not establish robustness.

Configuration, best validation-selected epoch, parameter count, runtime and source
provenance are in [the fixed run report](results/fixed32-seed123.json).
Reproduce with the README training command; use a fresh output directory.

Visualizations: `ROBOSCORE_RUN=outputs/fixed32 .venv-roboscore/bin/python figures/learning.py`.
`drawings.png` selects the first test example of each family, without ranking by
error; `learning_curve.png` shows validation mean NRMSE. Models and raw rollouts
are ignored by Git.

## EOS extension: 8 motion symbols plus a terminator

Source: clean commit `e948426`. Run: `outputs/eos32`, same seeded data splits,
40 additional epochs initialized from the fixed run's best checkpoint. The
architecture supports 1–32 payload tokens plus EOS. A normalized length penalty
of 0.01 is added to drawing/constraint/smoothness training; validation selection
also includes normalized length cost. This is not a matched-compute comparison.

| Test condition | Feasible | Mean NRMSE | Strict success |
| --- | ---: | ---: | ---: |
| Learned hard EOS score | 100% | 0.2049 | 0% |
| Shuffled message | 100% | 0.3244 | 0% |
| Stationary receiver | 100% | 0.2868 | 0% |

**Length collapse:** all 240 test scores have one payload symbol followed by EOS.
Only four distinct messages appear, using motion IDs 3, 4, 6 and 7. Thus the
implementation supports variable-length communication, but this training run did
not discover a useful variable-length vocabulary or compositional score. The
small average-error improvement over fixed32 is not evidence of better notation;
the additional training and capacity change are confounded.

Earlier validation checkpoints used longer and varying lengths, but length
quickly collapsed. Length cost, the straight-through estimator, initialization,
and the decoder's tendency to produce coarse average motions need further study;
this run does not identify a single cause. This experiment used one seed and a fixed length penalty throughout training.

[EOS run report](results/eos32-seed123.json) records the validation-selected epoch,
length histogram, warm-start checkpoint hash, training time and provenance.
Both modes were reloaded and reevaluated through serialized integer messages;
reported test mean NRMSE reproduced exactly in the current pinned environment.
The complete trajectories are not supplied to the learned Performer.

## What remains unproven

These runs validate implementation and training mechanics, not adequate drawing
accuracy, meaningful token semantics, editable local structure or grammar.
Next research work should first establish high-fidelity drawing communication
and diagnose message/length collapse before making compositional claims. A
continuous-channel control and fixed-length ablations would help separate
controller difficulty from discrete-channel optimization.
