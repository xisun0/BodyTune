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
