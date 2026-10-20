# Eight fixed diameter bins: separating quantization from learned encoding

The learned global1 experiment used only two of eight available symbols and achieved 3.33% strict test success. This control removes the learned Notator and assigns every diameter a fixed integer bin before training the same GRU128 Performer. It tests whether supplying a finer reliable discrete message improves execution; it does not establish the best possible one-token performance.

## Protocol

Eight equal-width bins cover the declared generation range [0.24, 0.70], with width 0.0575 and boundaries 0.24, 0.2975, 0.355, 0.4125, 0.47, 0.5275, 0.585, 0.6425, 0.70. Bins are left-closed/right-open except the final bin includes 0.70. Boundaries use float32 arithmetic, like the training inputs. The mapping is fixed in advance and does not fit validation/test diameters. The Performer sees only an integer ID through an unconstrained eight-entry embedding, never the scalar diameter or within-bin residual.

The original saved 1,200 training, 240 validation and 240 test demonstrations are unchanged, verified by file hashes. Each trajectory has 96 states / 95 actions. The entire Performer architecture and seeded initial weights match the learned global arm. A temporary learned encoder is initialized before being replaced by the parameter-free bin mapper, preserving the exact random-number sequence that initializes the Performer. The model trains from scratch, not from the continuous-diameter checkpoint.

Training remains Adam lr=0.001, seed123, batch64, clip1, CPU4, maximum300/minimum80 epochs and validation patience40. The same full-rollout endpoint loss, constraints and smoothness objective apply; no ground-truth actions feed back during rollout. Validation selects mean NRMSE plus infeasible fraction. Success requires feasibility and ordered, start-aligned positional NRMSE ≤0.01; no rotation, scaling or time warping is permitted. Test evaluation includes a shuffled-bin control and the one-token temporal-permutation identity control.

Reproduce with the commands in the README. Raw data, optimizer/RNG checkpoints, history, integer scores, test rollouts and inference figures are saved in `outputs/aligned128/binned8/`; versioned JSON records retain summary/provenance. Checkpoint reload verifies source hashes.

## Why did the learned encoder use two symbols?

The learned global run used one validation score at epoch1, two at epoch21, and remained at two through the end of training. This is observed underutilization. The current forward pass uses deterministic hard argmax with a soft straight-through gradient approximation. Only selected embeddings participate directly in the forward rollout; unused embeddings have no direct gradient from the message path. Although encoder logits can still receive surrogate gradients through the soft backward pass, there is no explicit exploration, balanced assignment or code-usage objective. Early winning codes can therefore become easier for the Performer to use while alternatives remain poorly trained. Encoder/decoder coadaptation and the nonlinear recurrent loss may sustain that local solution.

This is a plausible mechanism, not an experimentally isolated cause. Global pooled features must also separate radius, the model is trained end-to-end from random initialization, and only one seed was used. A successful fixed-label control would show that the previous two-code solution was not forced by the Performer architecture; it would not identify which optimization change best fixes the learned encoder. A failed control would also not prove eight bins insufficient: controller training could still be limiting.

## Quantization-only geometric reference

For each test example, construct an exact circle with the midpoint diameter of its assigned bin, aligned to the correct starting endpoint and sampled phase. This bypasses both controller error and kinematic feasibility. It gives mean NRMSE 0.00759045 and 174/240 = 72.5% geometric success under the same error threshold. This is a useful reference, **not an upper bound**: different representatives, different bins, state-dependent outputs or another selection objective can change success. The GRU also receives B's initial state, which may carry information correlated with diameter because full-path feasibility was required during dataset generation; a strict information-theoretic ceiling cannot be inferred from bin count alone.

## Completed result

| Input to GRU128 | Test mean NRMSE | Strict success | Symbols used |
|---|---:|---:|---:|
| Learned global1 | 0.0333414 | 8/240 = 3.33% | 2/8 |
| Fixed true-diameter bin | 0.0115797 | 117/240 = **48.75%** | 8/8 |
| Exact continuous diameter (earlier control) | 0.0055440 | 223/240 = 92.92% | continuous |

The fixed-bin run stopped at epoch242, selecting epoch202 using validation only. All normal and shuffled-bin test executions were feasible. Shuffling the bin across demonstrations gives NRMSE0.0808186 and success18/240 =7.5%. The single-token temporal identity control matches normal execution exactly; loading the best checkpoint reproduces every summary metric. The continuous-diameter row uses the earlier capacity run, with a different input projection and initialization; it is a diagnostic reference rather than an exactly matched model.

The 45.42 percentage-point gain over learned global1 shows that a single token and the same Performer architecture can do much better when all eight fixed labels convey diameter reliably. The remaining gap to the ideal midpoint-circle reference indicates additional execution/training error, alongside information discarded within bins. Mean endpoint NRMSE falls by about65.3%, but neither convergence nor a theoretical upper bound is established. Larger seed/optimization studies and alternative quantizers were not run, and no test examples were used to choose the checkpoint or bin edges.

The experiment supports investigating discrete assignment/optimization and code underutilization before blaming GRU128 capacity alone. It does not show that equal-width bins are optimal, that a learned encoder would necessarily choose diameter bins, or that these supervised IDs constitute a discovered compositional language.
