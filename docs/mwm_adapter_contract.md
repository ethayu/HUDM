# MWM Base Adapter Contract

MWM adapters translate a Stable-WM base world model into the shared MWM runtime.
The adapter owns base-specific architecture parsing. The shared world-model code
owns matryoshka aggregation, checkpoint loading, and fidelity-aware evaluation.

## Adapter Module

Create one adapter module per base family, for example `mwm/adapters/lewm.py` or
`mwm/adapters/prejepa.py`.

Each module should:

- define a `StableWMBaseAdapter` implementation;
- register it with `register_adapter(...)`;
- expose an importable checkpoint builder such as
  `build_mwm_<family>_from_stable_config`;
- avoid delegating runtime behavior to a `source_model` object.

Checkpoint `config.json` files store the builder import target. Prefer stable
targets under the family module, such as
`mwm.adapters.lewm.build_mwm_lewm_from_stable_config`.

## Required Adapter Methods

`family`

The Stable-WM family key used by configs and registry lookup.

`component_groups()`

Return top-level named component groups. At least one group must be a shared
latent producer:

```python
{
    "latent_producer": ComponentGroup(
        name="latent_producer",
        components=("encoder", "projector"),
        latent_producer=True,
    ),
    "transition": ComponentGroup(...),
    "reconstructor": ComponentGroup(...),
}
```

The component names should describe base architecture pieces, not MWM concepts.

`default_policy()`

Return the default sharing policy. For MWM, the latent-producing path must remain
shared and non-encoder/tail modules should be per-level unless the base design
requires otherwise:

```python
ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())
```

Invalid policies that leave no shared latent producer must be rejected through
`validate_component_policy`.

`resolve_spec(...)`

Validate the policy, deep-copy the base config and training recipe, derive the
base latent dimension `D` from the base config itself, and return
`StableWMBaseSpec`.

Do not infer `D` from `max(K)`. Configured levels may all be below `D`.

`build_model(spec, **runtime)`

Instantiate one shared latent producer and one per-level tail for each `K`.
For `K=D`, the instantiated architecture and objective path must be identical to
the base training path. For `K<D`, scale only the base tail internals that should
be smaller at that level.

The returned model must expose:

- `metadata` with adapter family, `D`, levels, component policy, source config
  hash, architecture version, action spec, preprocessing spec, and training
  recipe;
- `mwm_config`, an importable checkpoint builder config;
- normal `state_dict` loading with no hidden source-object delegation.

## Generic Runtime Pieces

`mwm.models.world_model.MatryoshkaWorldModel` currently provides the shared
runtime for Le-WM-shaped latent prediction:

- shared encoder/projector latent production;
- per-level `TransitionPackage(action_encoder, predictor, pred_proj)`;
- shared encoding followed by per-level prefix prediction losses;
- matryoshka aggregation via `matryoshka_base_loss`;
- shared-latent or per-level-prefix regularization policy;
- fidelity-aware rollout/cost for the current evaluator.

If another base has the same latent-transition shape, reuse
`TransitionPackage` and return `MatryoshkaWorldModel`.

If another base has a different objective or rollout contract, add a generic hook
to `world_model.py` for base-provided per-level loss/rollout behavior. Do not
put special inference behavior or hidden source-model calls in the adapter.

## Training And Evaluation Wiring

To make a new family trainable:

- add the family to the Stable-WM registry mapping in `mwm/adapters/registry.py`;
- add a train config with `base.family`, source checkpoint, data path, `D`, `K`,
  and the base training recipe;
- route `train_mwm.py` to the new builder for that family;
- export canonical checkpoints with `config.json`, `weights.pt`, and
  `world_metadata.json`;
- teach `verify_mwm_benchmark.py` the expected checkpoint contract for the new
  role/family;
- add benchmark config rows for the relevant environments.

Inference should remain base-aligned. Preserve the base action preprocessing,
frameskip/action block, image preprocessing, rollout horizon semantics, and
planner-facing action shape.

## PreJEPA/DINO-WM TODO

`mwm/adapters/prejepa.py` is currently only a policy stub. To complete it,
identify from the Stable-WM PreJEPA config:

- the latent producer components and the authoritative source of `D`;
- the non-encoder tail modules to duplicate per `K`;
- the tail width keys that become exactly `K`;
- the internal width keys that scale by `K / D`;
- the base objective terms and whether they fit the existing
  `TransitionPackage` prediction shape;
- the regularizer scope, if any;
- preprocessing and action/frame-skip conventions.

Then implement:

- `resolve_spec` with no `max(K)` fallback for `D`;
- `build_mwm_prejepa_from_stable_config`;
- a model path that applies the exact base recipe at each level and aggregates
  the per-level losses;
- checkpoint contract tests;
- `K=D` identity tests for architecture, loss keys, optimizer path, and
  inference recipe;
- a single-level benchmark check before using scheduled `K<D` results.

