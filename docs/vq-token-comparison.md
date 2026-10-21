# Matched TCN/VQ: one global token versus 95 local tokens

This comparison asks how much the learned discrete sequence helps on fixed-phase circles. Both models infer symbols from A's endpoint motion without diameter labels. The original saved 1,200 training / 240 validation / 240 test trajectories are unchanged.

## Matching the experiment

Both models have the same causal TCN (64 channels, kernel3, dilations1/2/4, 15-action receptive field), eight 8D VQ codewords, GRU128 Performer and time-conditioned auxiliary decoder. Their parameter shapes, counts and seed123 initial values are identical **before** train-only k-means codebook initialization. Global1 averages all95TCN latent vectors before nearest-neighborVQ; local95 quantizes each latent separately. K-means uses each model's resulting training-latent distribution, so fitted initial codebooks differ. Neither mode uses hand-designed diameter bins.

Global1 emits one integerID and reuses its embedding throughout the95-action rollout. Local95 emits one integerID per action. The global message is encoded after observing the full demonstration; individual local TCN features are causal. No continuous encoder features reach the Performer at inference. Both controllers read B's current predicted joint state, shared time progress and recurrent memory.

### Why the auxiliary decoder changes for both arms

The legacy95-token experiment used an8→64→2 memoryless reconstruction decoder. Broadcasting one global code through that decoder would force the same reconstructed displacement at every time, which cannot reconstruct a circle. The new matched comparison therefore uses the same10→64→2 auxiliary decoder in **both** arms: quantized8D code, progress and progress². The global code broadcasts across time; the local code remains aligned. Time is not input to the Translator. This training-only decoder shares no weights with the Performer and does not supply actions or continuous side information during inference.

The new local95 arm is retrained, so the legacy50.42% result is a historical reference, not the matched control. Old checkpoint decoding remains unchanged and was reverified.

### Training and selection

Both runs use seed123, CPU4, batch64, Adam0.001 and clipping1. Train-only k-means++ plus20Lloyd iterations initializes the codebook. Forty pretraining epochs optimize normalized endpoint-displacement reconstruction plus VQ code/commitment loss. Then Adam and the shuffle generator restart; joint training minimizes the unchanged rollout objective plus0.01×(reconstruction+VQ loss), commitment coefficient0.25. Maximum300joint epochs, minimum80, validation patience40. There is no balancing loss, semantic label, code restart or warm start from a prior Performer.

The same validation criterion—mean ordered NRMSE plus infeasible fraction—selects each checkpoint. Test data are evaluated only after selection. Different completed epochs under the same patience rule are not equal-compute training. Checkpoints retain optimizer/RNG state and data hashes.

## Four primary measures

| Measure | Definition | Better |
|---|---|---|
| Strict success | Feasible joint rollout AND positional NRMSE≤0.01 | Higher |
| Mean NRMSE | Ordered, start-aligned endpoint RMSE divided by target path length | Lower |
| Diameter error | Mean absolute fitted-diameter error / true diameter | Lower |
| Circularity residual | Circle-fit radial RMSE / fitted radius | Lower |

Circularity fits each executed path independently, so it ignores target size, center, traversal speed and phase. A perfectly circular path with the wrong size can score well on circularity and fail the primary trajectory criterion. All examples, including failed ones, enter summaries; degenerate circle fits are counted separately and excluded only from fit-dependent measures. Closure and completed turns are additional diagnostics.

Evaluation also reports symbol counts/perplexity, cross-demonstration message shuffling and a seed777 temporal permutation. Temporal permutation must be an identity for global1. `figures/vq_tokens.py` checks model/training matching, data hashes, ordered test arrays and that identity, then exports inference overlays, metric comparison, validation curves and train/val/test integer score corpora.

## Interpretation boundaries

One token carries3nominal bits versus285for95tokens. This compares pooling and information budget together, not locality in isolation. One seed cannot establish a statistically reliable ranking. Both models share a circle phase/progress clock, and fixed-base arms have the same morphology. The experiment is not a theoretical capacity ceiling, a learned grammar, or a test of composition/editing/generalization to other shapes.

## Completed results

| Metric | Global1 | Local95 |
|---|---:|---:|
| Strict success |64/240 = **26.67%** |136/240 = **56.67%** |
| Mean NRMSE |0.0150688 |**0.0107438** |
| Mean absolute relative diameter error |4.5238% |**2.3636%** |
| Circle radial RMSE / radius |2.5532% |**2.3564%** |
| Feasible normal executions |240/240 |240/240 |
| Codes used |6/8 |8/8 |
| Code perplexity |5.9220 |6.6207 |
| Distinct scores |6 |106 |
| Completed joint epochs |221 |204 |
| Validation-selected epoch |181 |164 |

Both models have158,476parameters, including119,874in the Performer. Initial weights before k-means have the same SHA256`051b7240d278c99c6b52f94e9124e19bbb8670e1344260df7bbfefc92b4411cc`; data files and ordered test trajectories match exactly. Both runs stopped using validation patience40. Normal test circle fits were valid for all240examples. Best-checkpoint reload reproduces the hard-integer summaries.

Local95 improves success by30percentage points, reduces mean NRMSE by about28.7%, and approximately halves diameter error. Circularity improves only modestly, by about7.7%relative. Thus the extra tokens chiefly improve scale and ordered execution accuracy in this experiment, rather than transforming a noncircular path into a circle. These are descriptive error patterns, not a formal causal error decomposition.

Shuffling messages across demonstrations gives success5.00%andNRMSE0.0813117forglobal1, versus4.58%and0.0801418forlocal95. The global step-permutation identity reproduces its normal metrics exactly. Permuting local token order gives0%success andNRMSE0.0437192; all those permuted rollouts remain feasible. One shuffled local rollout is infeasible. The sequence content and order affect execution, although this alone does not establish independently editable or compositional symbols.

The legacy local95 experiment scored50.42%with a time-free auxiliary decoder, whereas the new matched local95 scores56.67%. Their difference cannot be attributed confidently to a single mechanism: the auxiliary architecture changes both the objective's usable conditioning and random initialization. The primary comparison is the new global1 versus new local95 pair. The per-run reports retain a generic historical parameter/compute disclaimer; this pair is parameter matched as verified above, while actual compute and message budgets differ.

The result establishes that a learned single-token TCN/VQ system can execute circles better than the previous collapsed two-code global model, but does not attain the manually assigned eight-bin oracle's48.75%. Global1 uses only six codes, and optimization and quantization quality still limit the comparison. Neither26.67%nor56.67%is a theoretical ceiling. Multiple seeds, unseen shapes and composition/editing tests remain untested.
