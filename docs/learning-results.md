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
