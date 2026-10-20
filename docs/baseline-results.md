# Analytic A/B reproduction baseline

Run on 2026-10-20 using clean source revision `7d03086e4f4d424177733abfcc8ab02ba5433bd2`.
This is an analytic kinematic control experiment, not a trained motion language.
The machine-readable configuration and provenance are in [the run report](results/baseline-seed42.json).

## Setup

- Seed 42; 1,200 paired trajectories; 200 pairs per drawing family.
- Six families: square, rectangle, triangle, circle, arc, polyline.
- 96 ordered samples, 0.05 s per step; unit-length links; 2 rad/s per-joint speed limit.
- Independently sampled A/B initial states, conditioned on full-path feasibility.
- 1,640 sampling attempts; 440 rejected (26.83%).
- Success: valid sampled rollout and positional RMSE / target path length ≤ 0.01.
- Shared planar frame: align starts only, preserving scale, orientation and order.

## Results

| Method | Feasible / total | Successful / total | Mean normalized RMSE, feasible only |
| --- | ---: | ---: | ---: |
| Full relative curve + IK | 1,200 / 1,200 | 1,200 / 1,200 | 2.95e-16 |
| Copy A's joint increments | 1,184 / 1,200 | 0 / 1,200 | 0.3611 |

The oracle's maximum point error was 1.26e-15 length units, consistent with numerical roundoff. The near-zero error is expected: generation and reproduction use the same analytic kinematics. It verifies the data flow, action integration and drawing-alignment convention, rather than learned generalization.

Joint-copy failures include 16 constraint-invalid executions. The error mean excludes those invalid cases; success rates include all pairs. Copying a joint trajectory from a different initial state generally does not preserve the target drawing in this sample distribution.

## Reproduction

From the project root:

```bash
UV_PROJECT_ENVIRONMENT=.venv-roboscore uv sync --frozen
.venv-roboscore/bin/pytest -q
.venv-roboscore/bin/roboscore-benchmark --output outputs/reproduction --count 1200 --seed 42
ROBOSCORE_RUN=outputs/reproduction .venv-roboscore/bin/python figures/benchmark.py
```

The destination must not already exist. The report records the source revision and dirty state. Numerical trajectories and metrics are deterministic for the pinned environment; serialized NPZ bytes are not the reproducibility criterion.

## Limits and next milestone

This run has no training split, learned tokens or learned Performer. The oracle transmits an unrestricted continuous relative curve and is not a 16-bit communication baseline. Feasibility is evaluated at sampled states; continuous interpolation, dynamics, contacts, collisions and acceleration constraints are not tested. Rejection sampling changes the curve/state distribution.

Next, freeze structural train/validation/test splits and add a capacity-matched compressed-curve baseline before evaluating the learned discrete channel. Preserve the distinction between communication success and evidence of compositional semantics.
