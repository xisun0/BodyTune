# One-token circle diameter experiment

## Dataset and protocol

Run `outputs/circles1` uses clean source commit `da99963` and seed 123.
There are exactly 1,200 training demonstrations with 1,200 distinct continuous
diameters, plus 240 validation and 240 test demonstrations (seeds 124/125).
Diameters are sampled uniformly in [0.24, 0.70] link units; the realized training
range is [0.2400857, 0.6998200]. All circles have the same initial phase,
counterclockwise direction and shared orientation, with 96 sampled states.
A/B initial joint configurations vary and differ; both full paths must satisfy
the analytic arm constraints. A diameter is held fixed across rejected state
samples, rather than redrawn when an initial state is infeasible.

`train_demonstrations.npz` contains full A and target-B joint paths, relative
pen coordinates, exact diameters and dt. The other splits use the same schema.
`training_index.csv` lists all 1,200 examples and their initial configurations.
`training_circles.png` previews eight diameter-ordered examples.

The model receives no diameter/shape class labels. The existing Transformer
Notator and closed-loop GRU Performer communicate a single symbol from IDs 0–7.
No hand-defined diameter thresholds, circle primitives, IK correction or
analytic circle generator are supplied to the learned Performer. Geometry
supervision still uses the full demonstrated pen curve during training.

## Training and evaluation

Maximum budget: 300 epochs; minimum: 80; validation patience: 40 epochs.
Training stopped at epoch **254**, with the best validation checkpoint at
**214**. That checkpoint was used for held-out test inference and successfully
reloaded for independent metric reproduction. This stopping rule indicates a
validation plateau under this run's settings, not global convergence.

| Test condition | Mean NRMSE | Feasible | Strict success |
| --- | ---: | ---: | ---: |
| Correct learned token | 0.01691 | 100% | 43/240 = 17.92% |
| Shuffled token | 0.07952 | 100% | 8/240 = 3.33% |
| Stationary receiver | 0.22394 | 100% | 0% |

Success retains the previous criterion: feasible execution and positional
RMSE/path length ≤ 0.01. The comparison supports message-dependent diameter
communication; high-fidelity reproduction across the full continuous diameter
range is not achieved. It is not a matched comparison to earlier six-family
experiments, which had different motion variability and training budgets.

## Emergent scale groups

The Notator used five of the eight available symbols on test demonstrations.
Sorted by the mean target diameter associated with each token:

| Token | Test count | Mean target diameter |
| --- | ---: | ---: |
| 5 | 36 | 0.2764 |
| 3 | 32 | 0.3409 |
| 1 | 48 | 0.4159 |
| 7 | 56 | 0.5280 |
| 6 | 68 | 0.6330 |

These are post-training descriptive associations, not supplied labels or exact
learned thresholds. Adjacent diameter ranges overlap, so the encoder is not yet
a clean scalar quantizer invariant to demonstration posture. Symbols 0, 2 and 4
were not used on the test set. A single 8-way message has only 3 nominal bits;
continuous diameters necessarily share codes. Both quantization and control
errors can affect drawing fidelity; this experiment does not separate them.

## Artifacts and reproduction

```bash
.venv-roboscore/bin/roboscore-train --output outputs/circles-reproduction \
  --task circles --tokens 1 --train-count 1200 --val-count 240 --test-count 240 \
  --epochs 300 --min-epochs 80 --patience 40
.venv-roboscore/bin/python -m roboscore.evaluate outputs/circles-reproduction
ROBOSCORE_RUN=outputs/circles-reproduction .venv-roboscore/bin/python figures/circles.py
```

Generated artifacts remain outside Git:

- `circle_inference.png/pdf`: six test examples chosen by diameter quantiles,
  not reconstruction error; black target and blue learned execution.
- `circle_tokens.png`: diameter/token association and validation history.
- `diameter_groups.json`: counts and target diameter ranges/means by symbol.
- `token_prototypes.png/npz`: execute all eight integer commands from the same
  B state (pen starts at world (1, 0.5), elbow-up). No demonstration is supplied
  for this intervention. Unused symbols are marked explicitly; their outputs
  should not be assigned learned meanings from this plot alone.

The [run report](results/circles1-seed123.json) records configuration, split seeds,
stopping summary, source provenance, and evaluation.
This is an elementary scale abstraction experiment, not evidence of composition,
editable motion programs or general cross-embodiment transfer.
