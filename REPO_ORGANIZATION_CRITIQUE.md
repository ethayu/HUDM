# MWM Repository Organization Critique

Scope: production library modules under `mwm/`, with notes on configs, scripts,
tests, and reports where they affect the library boundary.

## Current Verdict

The repo is now organized substantially better than the initial audit. The P1/P2
issues from the organization review have been addressed:

1. Runtime ownership is split into explicit family classes under `mwm.models.*`.
2. `mwm.adapters.builder` no longer imports or returns the concrete model class.
3. ViT checkpoint key remapping lives outside generic checkpoint I/O.
4. Le-WM training transform construction lives in training code, not generic
   data transforms.
5. Private compatibility aliases were removed from public modules.
6. The dataset recording helper is now public where tests and callers need it.
7. Paper-parity dataset metadata facts are centralized in a small registry.

The remaining organization concerns are P3-level cleanup: test-file size,
research-script hygiene, restore-family growth, and optional benchmark helper
consolidation.

## Resolved Findings

### Resolved: Model Runtime Naming And Inheritance

`mwm.models.common.MatryoshkaRuntimeModel` is the shared runtime marker, while
family behavior lives in concrete model classes such as
`mwm.models.lewm.LeWMMatryoshkaWorldModel` and
`mwm.models.prejepa.PreJEPAMatryoshkaWorldModel`. `mwm.models.core` is a
lightweight namespace instead of a second model class with overlapping meaning.

Why this is better: there is no longer a split between an aspirational
`MWMWorldModel` base and the concrete runtime that bypassed that base's
initializer. The production paths now name the family runtime explicitly:

`build_mwm_from_stable_config -> LeWMStableWMAdapter -> LeWMMatryoshkaWorldModel -> MWMWorldModelPolicy -> MWMScheduledCEMSolver`

`build_mwm_from_stable_config -> PreJEPAStableWMAdapter -> PreJEPAMatryoshkaWorldModel -> MWMWorldModelPolicy -> MWMScheduledCEMSolver`

### Resolved: Generic Adapter Builder Coupling

`mwm.adapters.builder` now returns `nn.Module` and no longer imports concrete
runtime classes. Concrete runtime construction remains adapter-owned.

Why this is better: the generic builder can remain the public construction API
without leaking Le-WM's concrete runtime type into the adapter boundary.

### Resolved: Checkpoint Keymap Ownership

HF/custom ViT state-dict remapping moved to `mwm.checkpoint_keymaps`.
`mwm.checkpoint_io` still exposes and uses those helpers for load compatibility,
but no longer owns the architecture-specific maps.

Why this is better: generic checkpoint persistence is separated from
model-family state-dict compatibility.

### Resolved: Data/Training Transform Boundary

`build_stable_wm_adapter_dataset_transform` moved to
`mwm.training.stable_wm_transforms`. Generic helpers such as `MWMTrainSampleTransform`,
`ZScoreScaler`, and `column_normalizer` remain in `mwm.data.transforms`.

Why this is better: generic data transforms no longer import Le-WM-specific
stable-pretraining assembly logic.

### Resolved: Private Compatibility Aliases

The following private aliases were removed:

- `mwm.training.stable_wm_config._as_container`
- `mwm.benchmark.io._jsonable`
- private underscore aliases in `mwm.eval.action_preprocessing`

`mwm.data.collection._record_dataset_to_path` was promoted to
`record_dataset_to_path`.

Why this is better: tests and callers now use explicit public names, matching
the repo's migration-hygiene policy.

### Resolved: Paper-Parity Dataset Metadata Duplication

Paper-parity dataset facts now live in `mwm.upstream.paper_parity`. Upstream
data preparation and HDF5 converters render metadata from the shared registry.

Why this is better: env IDs, restore specs, action dimensions, source artifacts,
and HF dataset provenance are no longer duplicated across prep/converter files.

## Remaining Concerns

### P3: Tests Are Still Broad

`tests/test_mwm_core.py` and `tests/test_mwm_artifacts.py` remain broad
contract-heavy files. This is acceptable for now because they guard boundary
drift, but future cleanup should split them by subsystem once behavior settles.

### P3: Research Scripts Are Still Looser Than Production Code

`scripts/research` remains intentionally less polished than `mwm/`. That is
acceptable while research scripts stay quarantined and do not become package
APIs.

### P3: Restore Ownership May Need A Registry Later

`mwm.swm.restore` is still the core restore module, with OGBench specialization
in `mwm.ogbench.restore`. This is fine for the current env set. If more env
families arrive, restore specs should move toward an explicit registry pattern.

### P3: Benchmark Matrix Identity Is Slightly Over-Fragmented

`mwm.benchmark.matrix_identity` is still small enough that its helpers could
eventually live in `benchmark.config` or `benchmark.analysis`. This is optional
and not a blocker.

## Verification

Baseline before implementation:

`/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_migration_hygiene.py tests/test_mwm_core.py tests/test_mwm_base_adapter.py tests/test_mwm_artifacts.py -q`

Result: `118 passed, 109 subtests passed`.

Post-cleanup targeted suite:

`/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_migration_hygiene.py tests/test_mwm_data_boundaries.py tests/test_mwm_repo_hygiene.py tests/test_mwm_artifacts.py tests/test_mwm_core.py tests/test_mwm_base_adapter.py -q`

Result: `162 passed, 183 subtests passed`.
