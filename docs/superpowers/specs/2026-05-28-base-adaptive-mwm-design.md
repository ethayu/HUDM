# Base-Adaptive MWM Framework Design

## Purpose

MWM must be a framework that adapts to a Stable-WM base without changing the base's semantics. A Stable-WM checkpoint or object is used as an architecture and recipe oracle, not as a source of weights for fair MWM training. Fair MWM training fresh-initializes the base architecture, then applies the base training recipe across matryoshka levels.

The initial supported paper-parity scope is Push-T and Two-Room with Le-WM. The framework design must also support Stable-WM base families such as PreJEPA/DINO-WM and PLDM once their Stable-WM architecture configs and training recipes are available.

## Current Findings

The current repository is Le-WM-specific. `mwm.adapters.lewm` builds a fresh Le-WM-shaped model from local config, owns Le-WM loss and rollout behavior, and scales per-K heads directly in the adapter. The generic `MWMWorldModel` can instantiate default MLP dynamics and image decoders, which is not acceptable as production MWM behavior because it silently changes the base.

Stable-WM checkpoints saved through `stable_worldmodel.wm.utils.save_pretrained` contain a `config.json` next to weights. Stable-WM reloads them by instantiating the config and then loading weights. This is the right architecture source for fresh-init MWM: MWM should instantiate from `config.json` and deliberately skip loading source weights in fair-training mode.

The cached upstream Le-WM Push-T and Two-Room checkpoints have architecture configs that fully specify `encoder`, `predictor`, `action_encoder`, `projector`, and `pred_proj`. Those configs do not include the complete training recipe, so training settings such as optimizer, scheduler, batch size, SIGReg weight, dataset transforms, action preprocessing, horizon, and evaluation protocol must come from an explicit recipe artifact or YAML.

## Goals

- Build a base-adaptive MWM wrapper that starts from a Stable-WM base config or checkpoint directory.
- Fresh-initialize all fair-training models, including the encoder.
- Use the Stable-WM base config to copy architecture, not weights.
- Use the base training recipe unchanged across levels, except that adapters may create lower-K versions of duplicated non-encoder modules.
- Duplicate only configured non-encoder component groups per level.
- Keep at least one shared latent-producing path across all levels.
- Apply task/objective losses per level and aggregate them matryoshka-style.
- Keep shared-latent regularizers single by default.
- Train reconstructors as convenience modules without affecting encoder training unless explicitly configured.
- Validate the evaluator against upstream Le-WM paper-parity performance before using it to judge MWM.

## Non-Goals

- Do not implement arbitrary PyTorch module-path graph surgery.
- Do not infer missing training recipes from weights or arbitrary object attributes.
- Do not support silent best-effort adaptation for unknown Stable-WM bases.
- Do not copy pretrained weights into fair MWM runs.
- Do not let generic MWM dynamics or decoders become production fallbacks.

## Architecture

Add a `StableWMBaseAdapter` protocol and registry. The resolver reads a Stable-WM checkpoint directory or object reference, resolves its `config.json`, detects the base family from the root `_target_`, and asks the matching adapter to produce an explicit MWM base spec.

The resolver output is an `MWMBaseSpec` with:

- `family`: stable family key such as `lewm`, `prejepa`, or `pldm`.
- `source_config`: the exact Stable-WM constructor config.
- `training_recipe`: explicit recipe config for loss terms, optimizer, data preprocessing, action preprocessing, and evaluator settings.
- `component_groups`: adapter-declared top-level groups.
- `component_policy`: final shared/per-level/reconstructor policy after config overrides.
- `levels`: ordered `K` values.
- `D`: full latent dimension.
- `fresh_init`: always true for fair-training MWM.

The runtime MWM model is adapter-owned. It must not fall through to generic default MLP dynamics or default decoders. Unknown families fail with an actionable error that names the missing adapter or missing recipe artifact.

## Component Policy

The framework exposes a common top-level policy schema:

```yaml
mwm:
  K: [48, 96, 144, 192]
  component_policy:
    shared: [latent_producer]
    per_level: [transition]
    reconstructor: []
  loss_terms:
    regularizers: shared_latent
    reconstructor_detach_encoder: true
    reconstructor_contributes_to_encoder_loss: false
```

Each adapter maps those groups to concrete top-level Stable-WM components.

Le-WM default:

```yaml
groups:
  latent_producer: [encoder, projector]
  transition: [action_encoder, predictor, pred_proj]
  reconstructor: []
```

PreJEPA/DINO-WM default:

```yaml
groups:
  latent_producer: [backbone]
  transition: [predictor, extra_encoders]
  reconstructor: [decoder]
```

PLDM default:

```yaml
groups:
  latent_producer: [encoder, projector]
  transition: [action_encoder, predictor, pred_proj]
  reconstructor: []
```

Validation rules:

- At least one configured shared group must be an adapter-declared latent producer.
- A group cannot be both shared and per-level.
- A reconstructor group cannot affect encoder gradients unless explicitly opted into the encoder/matryoshka objective.
- Unknown groups fail.
- Unknown base families fail.

## Level Construction

For `K=[D]`, the generated MWM model must be architecturally and procedurally identical to the base trained from scratch.

For `K<D`, the adapter builds a level version of the base tail. The level version inherits the base architecture and training recipe, but adapters may scale internal width-like hyperparameters so lower-fidelity levels are smooth and computationally cheaper. For Le-WM, this allows scaling predictor heads, dim head, MLP width, action embedding width, and projection width while preserving the base objective shape.

The shared latent producer emits a full-D latent. Each level consumes its own prefix or adapter-defined level view of that latent. Planning can vary fidelity because all levels are tied to the same shared latent-producing path.

## Loss Semantics

The base training recipe is conceptually identical across levels. MWM applies duplicated task/objective terms per level and aggregates them with level weights.

For Le-WM:

- Encode pixels once through the shared `encoder + projector`.
- For each level, run that level's `action_encoder + predictor + pred_proj` on the level prefix.
- Compute the same prediction loss shape as base Le-WM for that level.
- Aggregate per-level prediction losses with a weighted mean.

Shared-latent regularizers are single by default. Le-WM SIGReg is applied once to the full-D shared latent, matching the base at `K=[D]`. Config may explicitly request per-level-prefix regularization, but that is not the default.

Reconstruction is separate:

- Reconstructors may be trained for convenience.
- Default reconstructor training detaches the encoder latent.
- Reconstructor loss does not contribute to encoder/matryoshka training unless explicitly configured.

## Evaluation Semantics

The evaluator must first prove that upstream Le-WM performance can be reproduced for Push-T and Two-Room before MWM claims are accepted.

Evaluator validation ladder:

1. Run upstream Le-WM through the MWM evaluator on Push-T and Two-Room.
2. If upstream Le-WM deviates from paper success by more than 1 percentage point for an environment, run the same upstream checkpoint through the reference Stable-WM evaluator path using Stable-WM's own CEM solver and policy path.
3. If the Stable-WM reference evaluator is within 1 percentage point and the MWM evaluator is not, MWM implementation is incomplete until evaluator or solver parameters are corrected.
4. If both evaluators miss the paper target, investigate data, checkpoint, and protocol mismatch before judging MWM.
5. Solver and evaluator parameters learned from the validated reference path become locked into paper-parity configs.

Paper-parity targets for the initial scope:

- Push-T upstream Le-WM: 96.0 percent success.
- Two-Room upstream Le-WM: 87.0 percent success.

Fresh single-level `K=[D]` MWM trained from scratch must match upstream Le-WM performance within 5 percentage points on the same Push-T and Two-Room paper-parity setup.

## Implementation Deliverables

- `StableWMBaseAdapter` protocol and adapter registry.
- Stable-WM base resolver that reads `config.json`, detects family, and builds fresh-init MWM specs without loading source weights.
- Le-WM implementation through the new protocol.
- Component policy config with shared, per-level, and reconstructor groups.
- Validation that forbids policies without a shared latent producer.
- Loss routing that applies per-level task losses, single shared regularizers by default, and detached reconstructor training by default.
- Checkpoint metadata that records source base config hash, adapter family, component policy, levels, fresh-init status, loss-scope policy, and evaluator protocol.
- Explicit unsupported errors for PreJEPA/DINO-WM or PLDM when required configs or recipes are missing.
- Documentation that explains how to add a new Stable-WM base adapter.

## Test Deliverables

Unit tests:

- `K=[D]` Le-WM MWM has the same fresh architecture as base Le-WM.
- `K=[D]` Le-WM MWM has the same loss, gradients, and optimizer step behavior as base Le-WM from scratch.
- Multi-K Le-WM duplicates only `action_encoder`, `predictor`, and `pred_proj`.
- Multi-K Le-WM shares exactly one `encoder + projector` latent producer.
- Invalid component policies fail.
- The base resolver does not load pretrained weights in fair-training mode.
- Shared regularizers are applied once by default.
- Reconstructor loss detaches the encoder by default.

Empirical gates:

- Upstream Le-WM paper-parity evaluator validation passes for Push-T.
- Upstream Le-WM paper-parity evaluator validation passes for Two-Room.
- Fresh single-level `K=[D]` MWM trained from scratch matches upstream Le-WM on Push-T within 5 percentage points.
- Fresh single-level `K=[D]` MWM trained from scratch matches upstream Le-WM on Two-Room within 5 percentage points.
- Scheduled multi-level MWM runs through the validated evaluator and emits fidelity/compute diagnostics.

## Completion Criteria

Implementation is complete only when the framework deliverables, unit tests, evaluator validation ladder, fresh single-level equivalence gates, and scheduled multi-level run all pass for the initial Push-T and Two-Room Le-WM scope.

If upstream evaluator validation fails and the Stable-WM reference evaluator resolves the discrepancy, the corrected evaluator parameters must be committed before any MWM training result is accepted.
