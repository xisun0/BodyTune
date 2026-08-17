# BodyTune

BodyTune asks: **How can a pretrained generalist robot policy rapidly identify and adapt to a modified or degraded embodiment using minimal real-world interaction?**

Our hypothesis is that foundation policies should retain general skills and world knowledge, while a small machine-specific component learns the body available after deployment.

## First architecture

The first proof of concept keeps the pretrained VLA frozen:

```text
observation -> frozen pretrained VLA -> nominal action a_base
                                             |
body-specific signals -> residual adapter -> delta_a
                                             |
                                      a = a_base + delta_a
```

The residual body adapter will be small, independently trainable, saveable, loadable, and versioned. Full-VLA fine-tuning and learned body embeddings are later baselines, not initial infrastructure.

## Repository boundary

- **BodyTune** owns perturbations, adapters, experiment definitions, online adaptation, evaluation, and reproducibility metadata.
- **LeRobot** remains an external, pinned upstream dependency and provides the SO-101 interface, dataset/recording infrastructure, and SmolVLA implementation. Its source is not vendored here.
- **SmolVLA weights** remain immutable external Hugging Face Hub artifacts. They are identified by repository ID and revision, never committed to Git.
- **Calibration** is machine-specific runtime state. Raw files remain local and untracked; experiments record a robot ID, calibration identity/hash, and relevant code revisions.
- **Rollouts, checkpoints, videos, and run outputs** remain outside Git. Only lightweight configurations and provenance metadata belong in this repository.

The currently inspected LeRobot integration point is pinned in `pyproject.toml`. Before any real run, the SmolVLA Hub revision and the local robot/calibration identity must also be pinned in the run metadata. See `configs/provenance.example.yaml` and `docs/environment-audit.md`.

## First proof of concept

The first controlled embodiment shift is software-only:

1. Establish nominal SO-101 teleoperation, structured recording, and stable SmolVLA task performance.
2. Disable one joint in a BodyTune-owned action perturbation, producing an effective 5-DoF embodiment without changing LeRobot core.
3. Measure the immediate success-rate drop.
4. Train only an external residual adapter with increasing amounts of real-world interaction.
5. Plot task success and normalized recovery against the number of real-world interactions.

For baseline success `SR_original`, post-shift success `SR_shift`, and adapted success `SR_adapted`:

```text
Recovery Rate = (SR_adapted - SR_shift) / (SR_original - SR_shift)
```

Hardware pressure sensing is Phase 2. Pressure will initially be a machine-specific adapter input; it does not block the joint-disable proof of concept.

## State layout

```text
configs/       portable experiment and provenance templates
src/bodytune/  research-specific Python package
tests/         tests that do not require live hardware by default
local/         untracked machine configuration or calibration references
data/          untracked datasets and rollouts
checkpoints/   untracked trained adapters
runs/          untracked run outputs; immutable metadata may be promoted deliberately
```

Use [uv](https://docs.astral.sh/uv/) for the BodyTune environment. Do not alter a known-working robot environment until its dependencies and hardware behavior have been verified.

## Reproducible local environment

BodyTune uses its own `.venv`, Python 3.10, and the exact LeRobot Git revision
recorded in `pyproject.toml` and `uv.lock`. The pinned historical LeRobot commit
contains Git LFS test artifacts that are not needed at runtime and are no longer
available remotely. Create the environment without downloading those artifacts:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
uv run python -c 'import lerobot; print(lerobot.__version__)'
uv run pytest
```

This environment is separate from any system or Homebrew LeRobot installation.
Machine-specific robot settings belong in the ignored `local/` directory. A
`null` port or camera identifier means the hardware has not been identified and
must cause hardware construction to fail closed rather than guessing a device.
