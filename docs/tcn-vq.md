# TCN/VQ Translator and GRU128 execution

This experiment learns discrete motion scores before introducing a language model. A causal temporal convolutional Translator maps observed endpoint motion to VQ codes; the existing GRU128 Performer executes those codes from B's initial state. There is no Transformer and no diameter-label supervision.

## Architecture and information boundary

- Input: 95 successive 2D end-effector displacements from A's recorded joint trajectory via FK, divided by fixed scale0.05. No joint increments, absolute endpoint positions, diameter metadata or receiver state enter the Translator.
- TCN: pointwise Conv1D2→64 + ReLU; three residual blocks `x + ReLU(Conv1D(left_pad(x)))`, kernel3 and dilations1/2/4; pointwise Conv1D64→8. One causal block per dilation gives a15-action receptive field. Initial left-zero-padding supplies a boundary cue, even though no explicit position/time embedding is provided.
- VQ: eight learned8D codewords, nearest squared Euclidean distance, one integerID per action. Fixed95-token/285-nominal-bit scores. The shared codebook is exactly the Performer's existing8×8 embedding parameter, so no second hidden continuous channel is added.
- Performer: unchanged StepPerformer architecture: GRU128, embedding8D, conditioning projection128, receiver sin/cos state, progress and progress², hidden memory, bounded joint increments. It receives only current code; later codes cannot affect earlier executed actions. Translator decisions can depend on the previous14 observed displacement steps, unlike the old memoryless MLP local encoder.
- Auxiliary decoder:8→64ReLU→2 predicts normalized local displacement from the quantized latent. It is used only in training, never supplied to execution.

During training the quantized latent has the nearest code's forward value and the encoder's straight-through gradient. VQ loss combines `MSE(code, stopgrad(latent))` and0.25×commitment`MSE(latent, stopgrad(code))`. Codebook gradients come from the VQ code loss; the encoder receives reconstruction, commitment and joint-rollout gradients. Inference re-encodes hard integerIDs and decodes them with the codebook. Tests compare both paths within floating-point tolerance and ensure gradients reach the TCN, codebook and GRU. Integer-only checkpoint evaluation is authoritative.

## Training

Source files are the unchanged original1,200train/240validation/240test circles,96states,dt0.05. File hashes are recorded and verified on checkpoint reload. Data include different diameters and receiver initial configurations; all circles share phase and traversal direction.

1. Initialize TCN and controller from seed123. Initialize codewords using deterministic train-only k-means++ followed by20Lloyd iterations on initial TCN features. No labels or validation/test samples are used.
2. Forty epochs optimize reconstruction MSE plus VQ loss. The GRU, conditioning and action layers have no gradients and remain at their random initial weights; only the Translator, shared codebook and auxiliary decoder train.
3. Restart Adam and the training shuffle generator, then jointly train using `original rollout objective +0.01×(reconstruction+VQ loss)`. Maximum300joint epochs, minimum80, patience40. Both stages use Adam0.001, batch64, clipping1, seed123 and CPU4.

The trajectory objective is unchanged: squared start-aligned endpoint error normalized by target path length², joint-limit/singularity penalties and action smoothness. All95actions are rolled out with the model's own updated receiver joint state; no teacher actions are fed in. Validation mean NRMSE plus infeasible fraction selects the checkpoint. Success requires feasibility and ordered positional NRMSE≤0.01, with no rotation, scaling or time warping. The best checkpoint is reloaded before test evaluation.

No entropy balancing, dead-code restart, Gumbel sampling or hand-designed semantic symbols are introduced. Nearest-neighborVQ and initialization do not guarantee use of every code; report code counts and perplexity rather than assuming successful tokenization.

## Evaluation and reproducibility

Run the README commands; `figures/vq.py` compares with the saved MLP local95 run after checking dataset hashes and ordered test targets. Outputs under`outputs/tcn_vq/` include checkpoints with optimizer/RNG state, history, report, exported train/validation/test integer score corpora, executed joint trajectories, inference overlays, symbol scatter/heatmaps and validation/code-usage curves. JSON results are also versioned in`docs/results/tcn-vq/`.

Evaluation includes cross-demonstration score shuffling, fixed-seed temporal permutation, symbol usage/perplexity, circle-fit radial/diameter/closure/turns diagnostics and checkpoint reload. The symbol scatter is descriptive: TCN context means identical current displacements need not receive identical codes.

Comparisons are not parameter/compute matched: the TCN replaces a smaller MLP, objectives and initialization differ, and this experiment adds40pretraining epochs. One seed, fixed-phase circles and a shared Performer progress clock do not establish a general motion language, composition, local editing, cross-morphology transfer or autonomous grammar discovery. A future Transformer would be trained on stable discrete score sequences after this translation/execution stage is validated.
