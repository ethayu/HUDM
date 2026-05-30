# MWM Base Adapter Contract

MWM adapters translate a Stable-WM base world model into the shared MWM runtime.
The adapter owns base-specific architecture parsing. The shared world-model code
owns matryoshka aggregation, checkpoint loading, and fidelity-aware evaluation.

## Adapter Module

Create one adapter module per completed base family, for example
`mwm/adapters/lewm.py`.

Each module should:

- define a `StableWMBaseAdapter` implementation;
- register it with `register_adapter(...)`;
- avoid delegating runtime behavior to a source object.

Canonical checkpoint `config.json` files store generic builder import targets:
`mwm.adapters.builder.build_mwm_from_stable_config`. The checkpoint kwargs carry
`family`, so adding a new adapter does not churn checkpoint/config targets.

Adapters are construction code, not new model semantics. They should answer:

- which base components produce the shared latent;
- which base components are duplicated per fidelity level;
- how to instantiate a level-sized tail from the Stable-WM config;
- where the authoritative base latent dimension `D` comes from;
- which preprocessing, action shape, history length, and objective recipe the
  base already uses.

The adapter should not reimplement matryoshka aggregation, invent a new loss,
delegate inference to an upstream object, or create a one-off planner path.

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

## Generic Builder

Completed adapters are instantiated by the shared builders:

```python
def build_mwm_from_stable_config(
    *,
    family: str | None,
    source_config: dict[str, Any],
    source_config_sha256: str,
    training_recipe: dict[str, Any],
    K: Sequence[int],
    action_dim: int,
    expected_D: int | None = None,
    action_block: int = 1,
    image_shape: Sequence[int] = (224, 224),
    normalize_imagenet: bool = True,
    component_policy: ComponentPolicy | dict[str, Any] | None = None,
) -> nn.Module:
    ...
```

The shared builder:

- detects or validates the Stable-WM family from `source_config`;
- turn a mapping policy into `ComponentPolicy`;
- dispatch to the registered adapter's `resolve_spec`;
- reject `expected_D` mismatches against config-derived `D`;
- call `adapter.build_model(...)`;
- set `model.mwm_config["target"]` to
  `mwm.adapters.builder.build_mwm_from_stable_config`.

Adapters should not export family-named builder facades. The generic builder is
the public construction API.

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

Use this rule of thumb:

- if the base encodes pixels to a latent, encodes actions, predicts next latent
  prefixes, and scores planning rollouts through those predicted latents, the
  existing `MatryoshkaWorldModel` plus `TransitionPackage` may be enough;
- if the base has a different per-level objective, the generic model should gain
  a reusable loss hook such as "call this base loss on each level package, then
  aggregate";
- if the base has a different rollout contract, the generic model should gain a
  reusable rollout hook, while the adapter only supplies the modules and config
  needed by that hook.

The framework boundary is: adapters construct base-derived modules and metadata;
`world_model.py` owns shared latent reuse, per-level dispatch, loss aggregation,
checkpoint behavior, and planner-facing runtime behavior.

## Training And Evaluation Wiring

To make a new family trainable:

- add the family to the Stable-WM registry mapping in `mwm/adapters/registry.py`;
- add a train config with `base.family`, source checkpoint, data path, `D`, `K`,
  and the base training recipe;
- route `train_mwm.py` to the generic builder for that family and its dataset
  shape;
- export canonical checkpoints with `config.json`, `weights.pt`, and
  `world_metadata.json`;
- teach `verify_mwm_benchmark.py` the expected checkpoint contract for the new
  role/family;
- add benchmark config rows for the relevant environments.

Inference should remain base-aligned. Preserve the base action preprocessing,
frameskip/action block, image preprocessing, rollout horizon semantics, and
planner-facing action shape.

## Adding PreJEPA/DINO-WM

Do not add a placeholder adapter. First identify from the Stable-WM PreJEPA
config/model:

- latent producer components: the modules that create the one shared latent
  space consumed by all levels;
- authoritative `D`: the base config field or module dimension that defines the
  full latent width, never `max(K)`;
- per-level tail modules: every non-encoder component that should be freshly
  duplicated for each level;
- exact-`K` width keys: fields that should be set to the level size, such as
  latent input/output widths;
- scaled internal keys: fields that should scale by `K / D`, such as hidden
  widths, heads, or MLP dimensions when that preserves the base architecture;
- untouched base knobs: depth, normalization, activations, stochasticity,
  frame/history sizes, and other recipe fields that should remain identical;
- objective contract: the base loss terms, their inputs, and whether the
  existing `TransitionPackage` prediction shape can supply them;
- regularizer scope: whether regularization is once on the shared latent or per
  level prefix, matching the base recipe where applicable;
- inference contract: action preprocessing, action block/frame skip, image
  preprocessing, rollout horizon, and planner-facing action shape.

Then implement:

- `resolve_spec` with no `max(K)` fallback for `D`;
- a model path that applies the exact base recipe at each level and aggregates
  the per-level losses;
- canonical metadata and `mwm_config` fields matching the generic builder;
- checkpoint contract tests;
- `K=D` identity tests for architecture, loss keys, optimizer path, and
  inference recipe;
- an identity-parity `K=[D]` benchmark check before using scheduled `K<D` results.

Do not fill in a PreJEPA adapter by guessing component names from Le-WM. The
first step is to inspect the actual Stable-WM PreJEPA config/model and write
down the component map above. After that, the adapter implementation should be
mostly mechanical.
