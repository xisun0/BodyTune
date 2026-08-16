# Git Hygiene

This file is the project-local source of truth for BodyTune commits. It takes precedence over generic agent or tooling conventions.

## Commit authorization and identity

- Agents must not stage, commit, amend, rebase, reset, or push unless the user explicitly requests that action.
- For every new commit created by Codex or another automated agent, the Git author email must be `sunralm.bo@gmail.com`.
- Use the same email for the committer unless the user explicitly requests a different committer identity.
- Preserve the user's configured author name. If no author name is configured, ask the user rather than inventing one.
- Apply identity with repository-local configuration or per-command overrides. Do not change global Git configuration.
- Before reporting success, verify the resulting author and committer identities with Git metadata.

## Commit messages

Use an Angular-style subject:

```text
type(scope): short summary
```

Keep the summary imperative, specific, and under 72 characters when practical.

### Types

- `feat`: meaningful new research capability or workflow
- `fix`: correction of broken behavior
- `refactor`: behavior-preserving restructuring
- `docs`: documentation-only change
- `test`: test-only change
- `chore`: dependencies, tooling, configuration, or repository maintenance
- `style`: formatting-only change

### Scopes

Prefer these stable project scopes:

- `repo`: repository-wide initialization or housekeeping
- `deps`: Python and external dependency management
- `config`: experiment and provenance configuration
- `robot`: SO-101 integration and embodiment perturbations
- `policy`: frozen base-policy integration
- `adapter`: body-adapter models and checkpoints
- `adaptation`: training and online adaptation workflows
- `data`: rollout and dataset integration
- `eval`: metrics, evaluation, and recovery analysis
- `docs`: research and operational documentation
- `tests`: shared testing infrastructure

Use a new scope only when none of these describes a durable project area.

## Commit body

Add body sections only when they convey information not evident from the subject:

```text
[aim]
<why the change is needed>

[approach]
- <implementation choice>

[attention]
- <risk, limitation, migration note, or meaningful test gap>
```

Omit empty sections and routine successful test results.

For commits drafted or made in a Codex session, append the available session identifier as the final body line:

```text
codex resume <SESSION_ID>
```

Never invent a missing session identifier.

## Research-state boundaries

- Do not commit raw calibration files, pretrained model weights, datasets, rollout videos, run directories, secrets, or machine-specific absolute paths.
- Keep LeRobot source changes separate from BodyTune research commits.
- Keep base-model identity, calibration identity, robot identity, BodyTune revision, LeRobot revision, and adapter identity separately traceable.
- Prefer small, coherent commits; split unrelated source, documentation, and generated-artifact changes when doing so improves reviewability.
