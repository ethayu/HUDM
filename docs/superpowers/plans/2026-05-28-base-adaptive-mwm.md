# Base-Adaptive MWM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the base-adaptive MWM framework from `docs/superpowers/specs/2026-05-28-base-adaptive-mwm-design.md` and verify all unit and empirical completion gates for the initial Le-WM Push-T/Two-Room scope.

**Architecture:** Add a Stable-WM adapter protocol, registry, and config resolver that treat Stable-WM `config.json` as the source of architecture while never copying weights in fair-training mode. Move Le-WM trainable construction onto that protocol, keep `encoder + projector` shared, duplicate `action_encoder + predictor + pred_proj` per level, apply per-level prediction losses, and apply shared-latent regularizers once by default. Add evaluator validation with a Stable-WM reference fallback before accepting MWM results.

**Tech Stack:** Python 3.10 in `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`, PyTorch, OmegaConf/Hydra-style Stable-WM configs, Lightning/Stable-Pretraining, `unittest`/`pytest`, Stable-WM CEM and policy APIs.

---

## Subagent Strategy

Use subagents for disjoint write scopes:

- Worker A owns adapter protocol/config resolver files and their tests.
- Worker B owns Le-WM adapter refactor and Le-WM unit tests.
- Worker C owns loss/training/checkpoint metadata integration and tests.
- Worker D owns evaluator validation/reference fallback and tests.
- Main agent coordinates, reviews each worker diff, resolves integration issues, and runs final verification.

Workers are not alone in the codebase. They must not revert edits made by others and must adjust to concurrent changes.

## File Structure

- Create `mwm/adapters/base.py`: dataclasses/protocol for component policies, adapter specs, validation, and fresh-init flags.
- Create `mwm/adapters/stable_config.py`: Stable-WM config loading, target detection, config hashing, and fresh instantiation helpers.
- Create `mwm/adapters/registry.py`: adapter registration and family lookup.
- Modify `mwm/adapters/lewm.py`: implement the new Le-WM adapter on top of Stable-WM config specs while preserving current public builders.
- Modify `mwm/adapters/__init__.py`: export new adapter protocol and registry APIs.
- Modify `mwm/training.py`: pass loss config through adapter-owned `training_loss`, build shared regularizers once, and preserve generic fallback only as non-production compatibility.
- Modify `train_mwm.py`: add resolver-driven model construction, record source config and component policy metadata, and preserve current Le-WM config compatibility.
- Modify `mwm/checkpoints.py`: persist/validate source config hash, adapter family, component policy, fresh-init status, and loss-scope policy.
- Create `mwm/eval/reference.py`: reference Stable-WM evaluator helpers for upstream evaluator fallback.
- Modify `eval_mwm.py` and `verify_mwm_benchmark.py`: add evaluator validation ladder and 1 percentage point reference fallback semantics.
- Create `tests/test_mwm_base_adapter.py`: adapter policy/resolver tests.
- Extend `tests/test_mwm_core.py`: Le-WM K=D parity, sharing/duplication, regularizer, and reconstructor-gradient tests.
- Extend `tests/test_mwm_artifacts.py`: metadata and evaluator gate tests.
- Modify `README.md` and `REVIEW_GUIDE.md`: document base-adaptive adapter workflow and completion gates.

## Task 1: Adapter Protocol and Policy Validation

**Files:**
- Create: `mwm/adapters/base.py`
- Create: `tests/test_mwm_base_adapter.py`
- Modify: `mwm/adapters/__init__.py`

- [ ] **Step 1: Write failing policy validation tests**

Add these tests to `tests/test_mwm_base_adapter.py`:

```python
from __future__ import annotations

import unittest

from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy


class AdapterPolicyTests(unittest.TestCase):
    def test_policy_requires_shared_latent_producer(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder", "projector"), latent_producer=True),
            "transition": ComponentGroup(name="transition", components=("action_encoder", "predictor", "pred_proj")),
        }
        policy = ComponentPolicy(shared=(), per_level=("transition",), reconstructor=())

        with self.assertRaisesRegex(ValueError, "shared latent producer"):
            validate_component_policy(groups, policy)

    def test_policy_rejects_group_in_shared_and_per_level(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder",), latent_producer=True),
        }
        policy = ComponentPolicy(shared=("latent_producer",), per_level=("latent_producer",), reconstructor=())

        with self.assertRaisesRegex(ValueError, "both shared and per-level"):
            validate_component_policy(groups, policy)

    def test_policy_rejects_unknown_group(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder",), latent_producer=True),
        }
        policy = ComponentPolicy(shared=("latent_producer",), per_level=("missing",), reconstructor=())

        with self.assertRaisesRegex(ValueError, "Unknown component group"):
            validate_component_policy(groups, policy)

    def test_base_spec_stores_fresh_init_and_loss_scope(self) -> None:
        groups = {
            "latent_producer": ComponentGroup(name="latent_producer", components=("encoder",), latent_producer=True),
            "transition": ComponentGroup(name="transition", components=("predictor",)),
        }
        policy = ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())
        spec = StableWMBaseSpec(
            family="lewm",
            source_config={"_target_": "stable_worldmodel.wm.lewm.LeWM"},
            source_config_sha256="abc123",
            training_recipe={"loss": {"sigreg_weight": 0.09}},
            component_groups=groups,
            component_policy=policy,
            levels=(4,),
            D=4,
            fresh_init=True,
            loss_scope={"regularizers": "shared_latent"},
        )

        self.assertTrue(spec.fresh_init)
        self.assertEqual(spec.component_policy.shared, ("latent_producer",))
        self.assertEqual(spec.loss_scope["regularizers"], "shared_latent")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py -q
```

Expected: import failure for `mwm.adapters.base`.

- [ ] **Step 3: Implement `mwm/adapters/base.py`**

Create `mwm/adapters/base.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch.nn as nn


@dataclass(frozen=True)
class ComponentGroup:
    name: str
    components: tuple[str, ...]
    latent_producer: bool = False


@dataclass(frozen=True)
class ComponentPolicy:
    shared: tuple[str, ...] = ("latent_producer",)
    per_level: tuple[str, ...] = ("transition",)
    reconstructor: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ComponentPolicy":
        if value is None:
            return cls()
        return cls(
            shared=tuple(str(x) for x in value.get("shared", cls.shared)),
            per_level=tuple(str(x) for x in value.get("per_level", cls.per_level)),
            reconstructor=tuple(str(x) for x in value.get("reconstructor", cls.reconstructor)),
        )

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "shared": list(self.shared),
            "per_level": list(self.per_level),
            "reconstructor": list(self.reconstructor),
        }


@dataclass(frozen=True)
class StableWMBaseSpec:
    family: str
    source_config: dict[str, Any]
    source_config_sha256: str
    training_recipe: dict[str, Any]
    component_groups: Mapping[str, ComponentGroup]
    component_policy: ComponentPolicy
    levels: tuple[int, ...]
    D: int
    fresh_init: bool = True
    loss_scope: dict[str, Any] = field(default_factory=lambda: {"regularizers": "shared_latent"})

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter_family": self.family,
            "source_config_sha256": self.source_config_sha256,
            "component_policy": self.component_policy.as_dict(),
            "levels": [int(k) for k in self.levels],
            "D": int(self.D),
            "fresh_init": bool(self.fresh_init),
            "loss_scope": dict(self.loss_scope),
        }


class StableWMBaseAdapter(Protocol):
    family: str

    def component_groups(self) -> Mapping[str, ComponentGroup]:
        raise NotImplementedError

    def default_policy(self) -> ComponentPolicy:
        raise NotImplementedError

    def resolve_spec(
        self,
        *,
        source_config: dict[str, Any],
        source_config_sha256: str,
        training_recipe: dict[str, Any],
        levels: tuple[int, ...],
        component_policy: ComponentPolicy | None,
    ) -> StableWMBaseSpec:
        raise NotImplementedError

    def build_model(self, spec: StableWMBaseSpec) -> nn.Module:
        raise NotImplementedError


def validate_component_policy(groups: Mapping[str, ComponentGroup], policy: ComponentPolicy) -> None:
    known = set(groups)
    selected = set(policy.shared) | set(policy.per_level) | set(policy.reconstructor)
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"Unknown component group(s): {unknown}")
    overlap = sorted(set(policy.shared) & set(policy.per_level))
    if overlap:
        raise ValueError(f"Component group(s) cannot be both shared and per-level: {overlap}")
    shared_latent = [name for name in policy.shared if groups[name].latent_producer]
    if not shared_latent:
        raise ValueError("Component policy must include at least one shared latent producer group.")


__all__ = [
    "ComponentGroup",
    "ComponentPolicy",
    "StableWMBaseAdapter",
    "StableWMBaseSpec",
    "validate_component_policy",
]
```

- [ ] **Step 4: Export adapter base types**

Modify `mwm/adapters/__init__.py` to export these names:

```python
from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseAdapter, StableWMBaseSpec, validate_component_policy
from mwm.adapters.lewm import LeWMMatryoshkaWorldModel, MWMAdapter, MWMComponents, MWMImporter

__all__ = [
    "ComponentGroup",
    "ComponentPolicy",
    "LeWMMatryoshkaWorldModel",
    "MWMAdapter",
    "MWMComponents",
    "MWMImporter",
    "StableWMBaseAdapter",
    "StableWMBaseSpec",
    "validate_component_policy",
]
```

- [ ] **Step 5: Run task test**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py -q
```

Expected: all tests in `tests/test_mwm_base_adapter.py` pass.

- [ ] **Step 6: Commit**

```bash
git add mwm/adapters/base.py mwm/adapters/__init__.py tests/test_mwm_base_adapter.py
git commit -m "feat: add Stable-WM adapter policy protocol"
```

## Task 2: Stable-WM Config Resolver and Registry

**Files:**
- Create: `mwm/adapters/stable_config.py`
- Create: `mwm/adapters/registry.py`
- Modify: `mwm/adapters/__init__.py`
- Test: `tests/test_mwm_base_adapter.py`

- [ ] **Step 1: Add failing resolver and registry tests**

Append to `tests/test_mwm_base_adapter.py`:

```python
import json
from pathlib import Path

from mwm.adapters.registry import adapter_for_family, family_for_target, register_adapter
from mwm.adapters.stable_config import load_stable_wm_config, stable_config_sha256


class ConfigResolverTests(unittest.TestCase):
    def test_load_stable_wm_config_from_directory(self) -> None:
        with self.subTest("directory config"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload = {"_target_": "stable_worldmodel.wm.lewm.LeWM", "predictor": {"input_dim": 4}}
                (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")

                loaded, path = load_stable_wm_config(root)

                self.assertEqual(loaded, payload)
                self.assertEqual(path, root / "config.json")
                self.assertEqual(stable_config_sha256(path), stable_config_sha256(path))

    def test_family_detection_from_target(self) -> None:
        self.assertEqual(family_for_target("stable_worldmodel.wm.lewm.LeWM"), "lewm")
        self.assertEqual(family_for_target("stable_worldmodel.wm.prejepa.PreJEPA"), "prejepa")
        self.assertEqual(family_for_target("stable_worldmodel.wm.pldm.PLDM"), "pldm")
        with self.assertRaisesRegex(ValueError, "Unsupported Stable-WM target"):
            family_for_target("example.Unknown")

    def test_registry_returns_registered_adapter(self) -> None:
        class DummyAdapter:
            family = "dummy"

        register_adapter(DummyAdapter())
        self.assertEqual(adapter_for_family("dummy").family, "dummy")
```

- [ ] **Step 2: Run resolver tests to verify failure**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py::ConfigResolverTests -q
```

Expected: import failure for `mwm.adapters.registry` or `mwm.adapters.stable_config`.

- [ ] **Step 3: Implement `mwm/adapters/stable_config.py`**

Create:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_config_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_stable_wm_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    root = Path(path)
    cfg_path = root if root.name == "config.json" else root / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Stable-WM config.json not found: {cfg_path}")
    return dict(json.loads(cfg_path.read_text(encoding="utf-8"))), cfg_path


def root_target(config: dict[str, Any]) -> str:
    target = str(config.get("_target_", ""))
    if not target:
        raise ValueError("Stable-WM config must contain a root `_target_`.")
    return target


__all__ = ["load_stable_wm_config", "root_target", "stable_config_sha256"]
```

- [ ] **Step 4: Implement `mwm/adapters/registry.py`**

Create:

```python
from __future__ import annotations

from typing import Any


_ADAPTERS: dict[str, Any] = {}


def family_for_target(target: str) -> str:
    target = str(target)
    if target.endswith(".LeWM") or ".lewm." in target.lower():
        return "lewm"
    if target.endswith(".PreJEPA") or ".prejepa." in target.lower():
        return "prejepa"
    if target.endswith(".PLDM") or ".pldm." in target.lower():
        return "pldm"
    if target in _ADAPTERS:
        return target
    raise ValueError(f"Unsupported Stable-WM target {target!r}; add a StableWMBaseAdapter.")


def register_adapter(adapter: Any) -> None:
    family = str(getattr(adapter, "family"))
    _ADAPTERS[family] = adapter


def adapter_for_family(family: str) -> Any:
    try:
        return _ADAPTERS[str(family)]
    except KeyError as exc:
        raise ValueError(f"No Stable-WM adapter registered for family {family!r}.") from exc


def adapter_for_target(target: str) -> Any:
    return adapter_for_family(family_for_target(target))


__all__ = ["adapter_for_family", "adapter_for_target", "family_for_target", "register_adapter"]
```

- [ ] **Step 5: Export resolver and registry APIs**

Update `mwm/adapters/__init__.py` with `adapter_for_family`, `adapter_for_target`, `family_for_target`, `load_stable_wm_config`, `register_adapter`, `root_target`, and `stable_config_sha256`.

- [ ] **Step 6: Run task tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py -q
```

Expected: all adapter tests pass.

- [ ] **Step 7: Commit**

```bash
git add mwm/adapters/stable_config.py mwm/adapters/registry.py mwm/adapters/__init__.py tests/test_mwm_base_adapter.py
git commit -m "feat: resolve Stable-WM base configs"
```

## Task 3: Le-WM Adapter on Stable-WM Configs

**Files:**
- Modify: `mwm/adapters/lewm.py`
- Modify: `mwm/adapters/__init__.py`
- Test: `tests/test_mwm_core.py`
- Test: `tests/test_mwm_base_adapter.py`

- [ ] **Step 1: Add failing Le-WM config-builder tests**

Add imports:

```python
from mwm.adapters.lewm import LeWMStableWMAdapter, build_mwm_lewm_from_stable_config
```

Add this test class to `tests/test_mwm_base_adapter.py`:

```python
class LeWMStableConfigTests(unittest.TestCase):
    def _lewm_config(self) -> dict:
        return {
            "_target_": "stable_worldmodel.wm.lewm.LeWM",
            "encoder": {"_target_": "tests.test_mwm_core.FakeLeWMEncoder", "out_dim": 4},
            "predictor": {"_target_": "tests.test_mwm_core.FakeLeWMPredictor"},
            "action_encoder": {"_target_": "tests.test_mwm_core.FakeLeWMActionEncoder", "action_dim": 2, "out_dim": 4},
            "projector": {"_target_": "torch.nn.Identity"},
            "pred_proj": {"_target_": "torch.nn.Identity"},
        }

    def test_lewm_adapter_declares_groups(self) -> None:
        adapter = LeWMStableWMAdapter()
        groups = adapter.component_groups()

        self.assertEqual(groups["latent_producer"].components, ("encoder", "projector"))
        self.assertTrue(groups["latent_producer"].latent_producer)
        self.assertEqual(groups["transition"].components, ("action_encoder", "predictor", "pred_proj"))

    def test_build_from_stable_config_fresh_initializes_without_weights(self) -> None:
        model = build_mwm_lewm_from_stable_config(
            source_config=self._lewm_config(),
            source_config_sha256="abc",
            training_recipe={"loss": {"sigreg_weight": 0.0}},
            K=(4,),
            action_dim=2,
            action_block=1,
            image_shape=(8, 8),
            normalize_imagenet=False,
        )

        self.assertIsInstance(model, LeWMMatryoshkaWorldModel)
        self.assertEqual(model.metadata["adapter_family"], "lewm")
        self.assertTrue(model.metadata["fresh_init"])
        self.assertEqual(model.metadata["component_policy"]["shared"], ["latent_producer"])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py::LeWMStableConfigTests -q
```

Expected: missing `LeWMStableWMAdapter` and `build_mwm_lewm_from_stable_config`.

- [ ] **Step 3: Add Le-WM adapter class and Hydra-style instantiation helper**

In `mwm/adapters/lewm.py`, import:

```python
import copy
from hydra.utils import instantiate
from mwm.adapters.base import ComponentGroup, ComponentPolicy, StableWMBaseSpec, validate_component_policy
```

Add:

```python
class LeWMStableWMAdapter:
    family = "lewm"

    def component_groups(self) -> dict[str, ComponentGroup]:
        return {
            "latent_producer": ComponentGroup("latent_producer", ("encoder", "projector"), latent_producer=True),
            "transition": ComponentGroup("transition", ("action_encoder", "predictor", "pred_proj")),
            "reconstructor": ComponentGroup("reconstructor", ()),
        }

    def default_policy(self) -> ComponentPolicy:
        return ComponentPolicy(shared=("latent_producer",), per_level=("transition",), reconstructor=())

    def resolve_spec(
        self,
        *,
        source_config: dict[str, Any],
        source_config_sha256: str,
        training_recipe: dict[str, Any],
        levels: tuple[int, ...],
        component_policy: ComponentPolicy | None,
    ) -> StableWMBaseSpec:
        policy = component_policy or self.default_policy()
        groups = self.component_groups()
        validate_component_policy(groups, policy)
        predictor_cfg = dict(source_config.get("predictor", {}))
        d = int(predictor_cfg.get("output_dim", predictor_cfg.get("input_dim", max(levels))))
        return StableWMBaseSpec(
            family=self.family,
            source_config=copy.deepcopy(source_config),
            source_config_sha256=source_config_sha256,
            training_recipe=copy.deepcopy(training_recipe),
            component_groups=groups,
            component_policy=policy,
            levels=tuple(int(k) for k in levels),
            D=d,
            fresh_init=True,
            loss_scope=dict(training_recipe.get("loss_scope", {"regularizers": "shared_latent"})),
        )

    def build_model(self, spec: StableWMBaseSpec, **runtime: Any) -> LeWMMatryoshkaWorldModel:
        return _build_lewm_from_base_spec(spec, **runtime)
```

- [ ] **Step 4: Implement config-driven Le-WM construction**

Add:

```python
def _instantiate_module(config: dict[str, Any]) -> nn.Module:
    return instantiate(copy.deepcopy(config))


def _build_lewm_from_base_spec(
    spec: StableWMBaseSpec,
    *,
    action_dim: int,
    action_block: int,
    image_shape: Sequence[int],
    normalize_imagenet: bool,
) -> LeWMMatryoshkaWorldModel:
    cfg = copy.deepcopy(spec.source_config)
    encoder = _instantiate_module(cfg["encoder"])
    projector = _instantiate_module(cfg.get("projector", {"_target_": "torch.nn.Identity"}))
    transitions: list[LeWMTransitionPackage] = []
    arches: list[dict[str, Any]] = []
    for k in spec.levels:
        transition, arch = _build_transition_head_from_stable_config(int(k), int(spec.D), cfg)
        transitions.append(transition)
        arches.append(arch)
    metadata = {
        "adapter": "lewm",
        "adapter_family": "lewm",
        "architecture_version": LeWMMatryoshkaWorldModel.architecture_version,
        **spec.metadata(),
        "source_config": copy.deepcopy(spec.source_config),
        "training_recipe": copy.deepcopy(spec.training_recipe),
        "head_architectures": arches,
        "action_preprocessing": str(spec.training_recipe.get("action_preprocessing", "standard_scaler")),
    }
    model = LeWMMatryoshkaWorldModel(
        encoder=encoder,
        projector=projector,
        transitions=transitions,
        K=spec.levels,
        D=int(spec.D),
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
        history_size=int(spec.training_recipe.get("history_size", cfg.get("predictor", {}).get("num_frames", 3))),
        num_preds=int(spec.training_recipe.get("num_preds", 1)),
        head_architectures=arches,
        metadata=metadata,
    )
    model.mwm_config = {
        "target": "mwm.adapters.lewm.build_mwm_lewm_from_stable_config",
        "kwargs": {
            "source_config": copy.deepcopy(spec.source_config),
            "source_config_sha256": spec.source_config_sha256,
            "training_recipe": copy.deepcopy(spec.training_recipe),
            "K": list(spec.levels),
            "action_dim": int(action_dim),
            "action_block": int(action_block),
            "image_shape": list(image_shape),
            "normalize_imagenet": bool(normalize_imagenet),
            "component_policy": spec.component_policy.as_dict(),
        },
    }
    return model
```

Add:

```python
def _scale_positive_int(value: Any, *, k: int, D: int, minimum: int = 1) -> int:
    base = int(value)
    if k == D:
        return base
    return max(int(minimum), int(round(base * float(k) / float(D))))


def _set_if_present(config: dict[str, Any], key: str, value: int) -> None:
    if key in config:
        config[key] = int(value)


def _level_config(config: dict[str, Any], *, k: int, D: int, width_keys: tuple[str, ...], scaled_keys: tuple[str, ...]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for key in width_keys:
        _set_if_present(out, key, int(k))
    for key in scaled_keys:
        if key in out:
            out[key] = _scale_positive_int(out[key], k=int(k), D=int(D))
    return out


def _build_transition_head_from_stable_config(
    k: int,
    D: int,
    source_config: dict[str, Any],
) -> tuple[LeWMTransitionPackage, dict[str, Any]]:
    predictor_cfg = _level_config(
        source_config["predictor"],
        k=int(k),
        D=int(D),
        width_keys=("input_dim", "hidden_dim", "output_dim"),
        scaled_keys=("heads", "dim_head", "mlp_dim"),
    )
    action_cfg = _level_config(
        source_config["action_encoder"],
        k=int(k),
        D=int(D),
        width_keys=("emb_dim", "out_dim"),
        scaled_keys=("hidden_dim",),
    )
    pred_proj_cfg = _level_config(
        source_config.get("pred_proj", {"_target_": "torch.nn.Identity"}),
        k=int(k),
        D=int(D),
        width_keys=("input_dim", "output_dim", "hidden_dim"),
        scaled_keys=(),
    )
    transition = LeWMTransitionPackage(
        action_encoder=_instantiate_module(action_cfg),
        predictor=_instantiate_module(predictor_cfg),
        pred_proj=_instantiate_module(pred_proj_cfg),
    )
    arch = {
        "K": int(k),
        "predictor": copy.deepcopy(predictor_cfg),
        "action_encoder": copy.deepcopy(action_cfg),
        "pred_proj": copy.deepcopy(pred_proj_cfg),
    }
    return transition, arch
```

- [ ] **Step 5: Add public builder**

Add:

```python
def build_mwm_lewm_from_stable_config(
    *,
    source_config: dict[str, Any],
    source_config_sha256: str = "",
    training_recipe: dict[str, Any] | None = None,
    K: Sequence[int],
    action_dim: int,
    action_block: int = 1,
    image_shape: Sequence[int] = (224, 224),
    normalize_imagenet: bool = True,
    component_policy: dict[str, Any] | ComponentPolicy | None = None,
) -> LeWMMatryoshkaWorldModel:
    adapter = LeWMStableWMAdapter()
    policy = component_policy if isinstance(component_policy, ComponentPolicy) else ComponentPolicy.from_mapping(component_policy)
    spec = adapter.resolve_spec(
        source_config=source_config,
        source_config_sha256=str(source_config_sha256),
        training_recipe=dict(training_recipe or {}),
        levels=tuple(int(k) for k in K),
        component_policy=policy,
    )
    return adapter.build_model(
        spec,
        action_dim=int(action_dim),
        action_block=int(action_block),
        image_shape=tuple(int(x) for x in image_shape),
        normalize_imagenet=bool(normalize_imagenet),
    )
```

- [ ] **Step 6: Register Le-WM adapter on import**

At module bottom:

```python
from mwm.adapters.registry import register_adapter

register_adapter(LeWMStableWMAdapter())
```

Add `LeWMStableWMAdapter` and `build_mwm_lewm_from_stable_config` to `__all__`.

- [ ] **Step 7: Run task tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py tests/test_mwm_core.py::MWMCoreTests::test_k_equals_d_lewm_init_forward_grad_and_step_match_direct_backend -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add mwm/adapters/lewm.py mwm/adapters/__init__.py tests/test_mwm_base_adapter.py
git commit -m "feat: build Le-WM MWM from Stable-WM configs"
```

## Task 4: Shared Regularizer and Reconstructor Loss Scope

**Files:**
- Modify: `mwm/adapters/lewm.py`
- Modify: `mwm/training.py`
- Test: `tests/test_mwm_core.py`

- [ ] **Step 1: Add failing loss-scope tests**

Add to `tests/test_mwm_core.py`:

```python
class CountingRegularizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.shapes.append(tuple(value.shape))
        return value.square().mean() * 0.0
```

Add tests:

```python
    def test_lewm_sigreg_is_shared_once_by_default(self) -> None:
        model = build_mwm_lewm(
            {
                "encoder": "cnn",
                "D": 8,
                "K": [4, 8],
                "action_dim": 2,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
                "history_size": 2,
                "num_preds": 1,
                "predictor_depth": 1,
                "predictor_heads": 2,
                "predictor_dim_head": 4,
                "predictor_mlp_dim": 16,
                "projector_hidden_dim": 16,
            }
        )
        reg = CountingRegularizer()
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        out = model.training_loss(batch, sigreg=reg, sigreg_weight=0.5, sigreg_scope="shared_latent")

        self.assertIn("sigreg_loss", out)
        self.assertEqual(reg.calls, 1)
        self.assertEqual(reg.shapes[0][-1], 8)

    def test_lewm_sigreg_can_be_per_level_when_explicit(self) -> None:
        model = build_mwm_lewm(
            {
                "encoder": "cnn",
                "D": 8,
                "K": [4, 8],
                "action_dim": 2,
                "image_shape": (8, 8),
                "normalize_imagenet": False,
                "history_size": 2,
                "num_preds": 1,
                "predictor_depth": 1,
                "predictor_heads": 2,
                "predictor_dim_head": 4,
                "predictor_mlp_dim": 16,
                "projector_hidden_dim": 16,
            }
        )
        reg = CountingRegularizer()
        batch = {"pixels": torch.rand(2, 3, 3, 8, 8), "action": torch.randn(2, 3, 2)}

        model.training_loss(batch, sigreg=reg, sigreg_weight=0.5, sigreg_scope="per_level_prefix")

        self.assertEqual(reg.calls, 2)
        self.assertEqual([shape[-1] for shape in reg.shapes], [4, 8])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_core.py::MWMCoreTests::test_lewm_sigreg_is_shared_once_by_default tests/test_mwm_core.py::MWMCoreTests::test_lewm_sigreg_can_be_per_level_when_explicit -q
```

Expected: `training_loss` does not accept `sigreg_scope`.

- [ ] **Step 3: Update Le-WM `training_loss`**

Change signature:

```python
def training_loss(
    self,
    batch: dict[str, torch.Tensor],
    *,
    level_weights: Sequence[float] | None = None,
    rollout_weight: float = 1.0,
    sigreg: nn.Module | None = None,
    sigreg_weight: float = 0.0,
    sigreg_scope: str = "shared_latent",
) -> dict[str, torch.Tensor]:
```

Use this logic after per-level prediction loss aggregation:

```python
if sigreg is not None and float(sigreg_weight):
    if sigreg_scope == "shared_latent":
        sigreg_total = sigreg(emb.transpose(0, 1))
        logs["sigreg_loss"] = sigreg_total.detach()
    elif sigreg_scope == "per_level_prefix":
        sigreg_total = emb.new_tensor(0.0)
        for level_idx, weight in zip(levels, weights):
            k = self.K[level_idx]
            sigreg_loss = sigreg(emb[..., :k].transpose(0, 1))
            logs[f"sigreg_loss_l{level_idx}"] = sigreg_loss.detach()
            sigreg_total = sigreg_total + float(weight) * sigreg_loss / denom
        logs["sigreg_loss"] = sigreg_total.detach()
    else:
        raise ValueError(f"Unknown sigreg_scope {sigreg_scope!r}")
    loss = loss + float(sigreg_weight) * sigreg_total
```

Remove the old per-level SIGReg block from inside the prediction loop.

- [ ] **Step 4: Update `mwm/training.py` dispatch**

When `model.training_loss` exists, pass:

```python
sigreg=cfg.get("sigreg_module"),
sigreg_weight=float(cfg.get("sigreg_weight", 0.0)),
sigreg_scope=str(cfg.get("sigreg_scope", cfg.get("regularizers", "shared_latent"))),
```

Keep current callers that pass `sigreg` directly working by updating `_exact_lewm_forward` to pass `sigreg=module.sigreg`, `sigreg_weight=float(cfg.loss.get("sigreg_weight", cfg.loss.get("sigreg", {}).get("weight", 0.0)))`, and `sigreg_scope=cfg.loss.get("sigreg_scope", "shared_latent")` into `model.training_loss`.

- [ ] **Step 5: Run task tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_core.py::MWMCoreTests::test_lewm_sigreg_is_shared_once_by_default tests/test_mwm_core.py::MWMCoreTests::test_lewm_sigreg_can_be_per_level_when_explicit -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add mwm/adapters/lewm.py mwm/training.py train_mwm.py tests/test_mwm_core.py
git commit -m "feat: scope MWM regularizers explicitly"
```

## Task 5: Checkpoint Metadata Contract

**Files:**
- Modify: `mwm/checkpoints.py`
- Modify: `mwm/adapters/lewm.py`
- Test: `tests/test_mwm_artifacts.py`
- Test: `tests/test_mwm_core.py`

- [ ] **Step 1: Add failing metadata tests**

Add to `tests/test_mwm_artifacts.py`:

```python
def test_base_adaptive_metadata_is_persisted(tmp_path):
    from mwm.adapters.lewm import build_mwm_lewm_from_stable_config
    from mwm.checkpoints import load_world_metadata, save_world_checkpoint

    source_config = {
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {"_target_": "tests.test_mwm_core.FakeLeWMEncoder", "out_dim": 4},
        "predictor": {"_target_": "tests.test_mwm_core.FakeLeWMPredictor"},
        "action_encoder": {"_target_": "tests.test_mwm_core.FakeLeWMActionEncoder", "action_dim": 2, "out_dim": 4},
        "projector": {"_target_": "torch.nn.Identity"},
        "pred_proj": {"_target_": "torch.nn.Identity"},
    }
    model = build_mwm_lewm_from_stable_config(
        source_config=source_config,
        source_config_sha256="abc",
        training_recipe={"loss_scope": {"regularizers": "shared_latent"}},
        K=(4,),
        action_dim=2,
        image_shape=(8, 8),
        normalize_imagenet=False,
    )

    save_world_checkpoint(model, tmp_path, metadata={"env_id": "swm/PushT-v1", "levels": [4]})
    metadata = load_world_metadata(tmp_path)

    assert metadata["adapter_family"] == "lewm"
    assert metadata["source_config_sha256"] == "abc"
    assert metadata["fresh_init"] is True
    assert metadata["component_policy"]["shared"] == ["latent_producer"]
    assert metadata["loss_scope"]["regularizers"] == "shared_latent"
```

- [ ] **Step 2: Run metadata test to verify failure**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_artifacts.py::test_base_adaptive_metadata_is_persisted -q
```

Expected: metadata keys missing.

- [ ] **Step 3: Persist metadata keys**

In `save_world_checkpoint`, extend propagated keys:

```python
for key in (
    "action_spec",
    "preprocessing_spec",
    "architecture_version",
    "head_architectures",
    "action_preprocessing",
    "adapter_family",
    "source_config_sha256",
    "component_policy",
    "fresh_init",
    "loss_scope",
    "training_recipe",
):
```

- [ ] **Step 4: Validate base-adaptive checkpoints**

In `validate_checkpoint_contract`, when target ends with `build_mwm_lewm_from_stable_config`, require:

```python
if metadata.get("adapter_family") != "lewm":
    raise ValueError("Base-adaptive Le-WM checkpoint must declare adapter_family='lewm'.")
if metadata.get("fresh_init") is not True:
    raise ValueError("Fair MWM checkpoints must declare fresh_init=true.")
if not metadata.get("source_config_sha256"):
    raise ValueError("Base-adaptive MWM checkpoints require source_config_sha256.")
if "component_policy" not in metadata:
    raise ValueError("Base-adaptive MWM checkpoints require component_policy.")
```

- [ ] **Step 5: Run checkpoint tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_artifacts.py::test_base_adaptive_metadata_is_persisted tests/test_mwm_core.py::MWMCoreTests::test_lewm_object_import_roundtrips_canonical_checkpoint -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add mwm/checkpoints.py mwm/adapters/lewm.py tests/test_mwm_artifacts.py tests/test_mwm_core.py
git commit -m "feat: record base-adaptive checkpoint metadata"
```

## Task 6: Training Entry Point Resolver Integration

**Files:**
- Modify: `train_mwm.py`
- Modify: `configs/train_mwm_lewm_pusht.yaml`
- Modify: `configs/train_mwm_lewm_tworoom.yaml`
- Modify: `configs/train_mwm_scheduled_pusht.yaml`
- Modify: `configs/train_mwm_scheduled_tworoom.yaml`
- Test: `tests/test_mwm_repo_hygiene.py`
- Test: `tests/test_mwm_core.py`

- [ ] **Step 1: Add failing config hygiene tests**

Extend `tests/test_mwm_repo_hygiene.py` scheduled/single config assertions:

```python
self.assertIn("base", cfg, name)
self.assertEqual(cfg["base"]["family"], "lewm", name)
self.assertIn("checkpoint", cfg["base"], name)
self.assertEqual(cfg["mwm"]["component_policy"]["shared"], ["latent_producer"], name)
self.assertEqual(cfg["mwm"]["component_policy"]["per_level"], ["transition"], name)
self.assertEqual(cfg["mwm"]["loss_terms"]["regularizers"], "shared_latent", name)
```

- [ ] **Step 2: Run hygiene tests to verify failure**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_public_single_fidelity_configs_use_exact_lewm_backend tests/test_mwm_repo_hygiene.py::MWMRepoHygieneTests::test_scheduled_configs_use_lewm_base_adapter_training_recipe -q
```

Expected: missing `base`/`mwm` keys.

- [ ] **Step 3: Add config keys while preserving current fields**

For Push-T configs add:

```yaml
base:
  family: lewm
  checkpoint: models--quentinll--lewm-pusht
mwm:
  component_policy:
    shared: [latent_producer]
    per_level: [transition]
    reconstructor: []
  loss_terms:
    regularizers: shared_latent
    reconstructor_detach_encoder: true
    reconstructor_contributes_to_encoder_loss: false
```

For Two-Room configs use `models--quentinll--lewm-tworooms`.

- [ ] **Step 4: Implement resolver-driven model construction**

Add helper in `train_mwm.py`:

```python
def _stable_checkpoint_config_path(checkpoint: str) -> Path:
    from stable_worldmodel.data import get_cache_dir

    root = Path(checkpoint)
    if root.exists():
        return root / "config.json" if root.is_dir() else root.parent / "config.json"
    return get_cache_dir(None, sub_folder="checkpoints") / checkpoint / "config.json"


def _build_trainable_model_from_base(cfg: Any, model_cfg: dict[str, Any]) -> torch.nn.Module:
    from mwm.adapters.base import ComponentPolicy
    from mwm.adapters.lewm import build_mwm_lewm_from_stable_config
    from mwm.adapters.stable_config import load_stable_wm_config, stable_config_sha256

    base = cfg.get("base", {})
    if not base:
        return build_mwm_lewm(model_cfg)
    config_path = _stable_checkpoint_config_path(str(base["checkpoint"]))
    source_config, loaded_path = load_stable_wm_config(config_path)
    policy = ComponentPolicy.from_mapping(cfg.get("mwm", {}).get("component_policy", None))
    recipe = {
        **OmegaConf.to_container(cfg.loss, resolve=True),
        "history_size": int(cfg.model.get("history_size", cfg.loss.get("history_size", 3))),
        "num_preds": int(cfg.model.get("num_preds", cfg.loss.get("num_preds", 1))),
        "action_preprocessing": "standard_scaler",
        "loss_scope": dict(cfg.get("mwm", {}).get("loss_terms", {"regularizers": "shared_latent"})),
    }
    return build_mwm_lewm_from_stable_config(
        source_config=source_config,
        source_config_sha256=stable_config_sha256(loaded_path),
        training_recipe=recipe,
        K=tuple(int(k) for k in model_cfg["K"]),
        action_dim=int(model_cfg["action_dim"]),
        action_block=int(model_cfg.get("action_block", 1)),
        image_shape=tuple(int(x) for x in model_cfg["image_shape"]),
        normalize_imagenet=bool(model_cfg.get("normalize_imagenet", True)),
        component_policy=policy,
    )
```

Use this helper in exact Le-WM training and export paths in place of direct `build_mwm_lewm(model_cfg)` when `cfg.base` exists.

- [ ] **Step 5: Metadata includes source config**

When building `metadata` in `_prepare_exact_lewm_context`, merge `model.metadata` after model construction or add these fields before `save_world_checkpoint`:

```python
metadata = {**metadata, **getattr(model, "metadata", {})}
```

- [ ] **Step 6: Run task tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_repo_hygiene.py tests/test_mwm_core.py::MWMCoreTests::test_k_equals_d_lewm_init_forward_grad_and_step_match_direct_backend -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add train_mwm.py configs/train_mwm_lewm_pusht.yaml configs/train_mwm_lewm_tworoom.yaml configs/train_mwm_scheduled_pusht.yaml configs/train_mwm_scheduled_tworoom.yaml tests/test_mwm_repo_hygiene.py
git commit -m "feat: train MWM from Stable-WM base configs"
```

## Task 7: Evaluator Validation Ladder

**Files:**
- Create: `mwm/eval/reference.py`
- Modify: `eval_mwm.py`
- Modify: `verify_mwm_benchmark.py`
- Modify: `configs/benchmark_mwm_paper_parity.yaml`
- Test: `tests/test_mwm_artifacts.py`

- [ ] **Step 1: Add failing verification tests**

Add to `tests/test_mwm_artifacts.py`:

```python
def test_paper_target_gate_requires_reference_when_mwm_misses_by_more_than_one_point():
    from verify_mwm_benchmark import validate_paper_targets

    rows = [
        {"env_id": "swm/PushT-v1", "role": "upstream_lewm_converted", "success_rate": 94.0},
        {"env_id": "swm/PushT-v1", "role": "stable_wm_reference", "success_rate": 96.0},
        {"env_id": "swm/PushT-v1", "role": "retrained_lewm_single", "success_rate": 95.0},
    ]
    cfg = {
        "paper_targets": {
            "tolerance_pp": 1.0,
            "single_level_tolerance_pp": 5.0,
            "success_rate": {"swm/PushT-v1": 96.0},
        }
    }

    errors = validate_paper_targets(rows, cfg)

    assert any("MWM evaluator" in err for err in errors)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_artifacts.py::test_paper_target_gate_requires_reference_when_mwm_misses_by_more_than_one_point -q
```

Expected: current validator does not distinguish reference fallback.

- [ ] **Step 3: Implement `mwm/eval/reference.py`**

Create a helper with a pure function for tests and a runtime function for empirical runs:

```python
from __future__ import annotations

from typing import Any


REFERENCE_ROLE = "stable_wm_reference"


def needs_reference_evaluator(upstream_success: float, target_success: float, tolerance_pp: float = 1.0) -> bool:
    return abs(float(upstream_success) - float(target_success)) > float(tolerance_pp)


def reference_role_name() -> str:
    return REFERENCE_ROLE


def build_stable_wm_reference_policy(model: Any, plan_config: Any, *, cem_kwargs: dict[str, Any]) -> Any:
    from stable_worldmodel.policy import WorldModelPolicy
    from stable_worldmodel.solver import CEMSolver

    solver = CEMSolver(model=model, **cem_kwargs)
    return WorldModelPolicy(solver=solver, config=plan_config)
```

- [ ] **Step 4: Update verifier logic**

In `verify_mwm_benchmark.py`, treat `paper_targets.tolerance_pp` as 1.0 by default. If upstream misses target and a `stable_wm_reference` row exists within tolerance, emit an error that names the MWM evaluator discrepancy. If both upstream and reference miss target, emit an investigation error that names data/checkpoint/protocol mismatch.

- [ ] **Step 5: Add config knobs**

In `configs/benchmark_mwm_paper_parity.yaml`, add:

```yaml
paper_targets:
  tolerance_pp: 1.0
  single_level_tolerance_pp: 5.0
  success_rate:
    swm/PushT-v1: 96.0
    swm/TwoRoom-v1: 87.0
```

- [ ] **Step 6: Run task tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_artifacts.py::test_paper_target_gate_requires_reference_when_mwm_misses_by_more_than_one_point -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add mwm/eval/reference.py eval_mwm.py verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml tests/test_mwm_artifacts.py
git commit -m "feat: validate evaluator against Stable-WM reference"
```

## Task 8: Unsupported Adapter Declarations and Docs

**Files:**
- Create: `mwm/adapters/prejepa.py`
- Create: `mwm/adapters/pldm.py`
- Modify: `mwm/adapters/__init__.py`
- Modify: `README.md`
- Modify: `REVIEW_GUIDE.md`
- Test: `tests/test_mwm_base_adapter.py`

- [ ] **Step 1: Add failing unsupported adapter tests**

Add to `tests/test_mwm_base_adapter.py`:

```python
class UnsupportedAdapterTests(unittest.TestCase):
    def test_prejepa_adapter_declares_groups_but_requires_recipe(self) -> None:
        from mwm.adapters.prejepa import PreJEPAStableWMAdapter

        adapter = PreJEPAStableWMAdapter()
        groups = adapter.component_groups()
        self.assertEqual(groups["latent_producer"].components, ("backbone",))
        self.assertEqual(groups["transition"].components, ("predictor", "extra_encoders"))
        with self.assertRaisesRegex(NotImplementedError, "training recipe"):
            adapter.resolve_spec(
                source_config={"_target_": "stable_worldmodel.wm.prejepa.PreJEPA"},
                source_config_sha256="abc",
                training_recipe={},
                levels=(4,),
                component_policy=None,
            )

    def test_pldm_adapter_declares_groups_but_requires_recipe(self) -> None:
        from mwm.adapters.pldm import PLDMStableWMAdapter

        adapter = PLDMStableWMAdapter()
        groups = adapter.component_groups()
        self.assertEqual(groups["latent_producer"].components, ("encoder", "projector"))
        self.assertEqual(groups["transition"].components, ("action_encoder", "predictor", "pred_proj"))
        with self.assertRaisesRegex(NotImplementedError, "training recipe"):
            adapter.resolve_spec(
                source_config={"_target_": "stable_worldmodel.wm.pldm.PLDM"},
                source_config_sha256="abc",
                training_recipe={},
                levels=(4,),
                component_policy=None,
            )
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py::UnsupportedAdapterTests -q
```

Expected: missing modules.

- [ ] **Step 3: Implement explicit unsupported adapters**

Create `mwm/adapters/prejepa.py` and `mwm/adapters/pldm.py` with adapter classes that declare groups and default policies, register themselves, and raise `NotImplementedError` from `resolve_spec` with the exact message `"PreJEPA/DINO-WM MWM support requires an explicit Stable-WM training recipe artifact."` or `"PLDM MWM support requires an explicit Stable-WM training recipe artifact."`.

- [ ] **Step 4: Document adapter workflow**

Add to `README.md` an "Base-adaptive MWM" section explaining:

```text
MWM reads Stable-WM config.json for architecture, never copies weights for fair training, and uses adapter-declared component policies. Le-WM is implemented first. PreJEPA/DINO-WM and PLDM have group declarations but fail until a training recipe artifact is supplied.
```

Add matching review notes to `REVIEW_GUIDE.md`.

- [ ] **Step 5: Run task tests**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py::UnsupportedAdapterTests -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add mwm/adapters/prejepa.py mwm/adapters/pldm.py mwm/adapters/__init__.py README.md REVIEW_GUIDE.md tests/test_mwm_base_adapter.py
git commit -m "feat: declare unsupported Stable-WM base adapters"
```

## Task 9: Full Unit Suite and Static Verification

**Files:**
- All files changed by prior tasks.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python -m pytest tests/test_mwm_base_adapter.py tests/test_mwm_core.py tests/test_mwm_artifacts.py tests/test_mwm_repo_hygiene.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run benchmark verifier unit/static path**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml --static-only
```

Expected: verifier accepts config/schema or exits with an error only if `--static-only` is not supported. If unsupported, add a lightweight static mode to the verifier and rerun until it passes.

- [ ] **Step 3: Run import smoke**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python - <<'PY'
from mwm.adapters import ComponentPolicy, adapter_for_target
import mwm.adapters.lewm
print(ComponentPolicy().as_dict())
print(adapter_for_target("stable_worldmodel.wm.lewm.LeWM").family)
PY
```

Expected output includes `lewm`.

- [ ] **Step 4: Commit fixes if needed**

```bash
git add mwm tests configs README.md REVIEW_GUIDE.md train_mwm.py eval_mwm.py verify_mwm_benchmark.py
git commit -m "test: verify base-adaptive MWM unit gates"
```

Commit only if Step 1-3 required changes after the previous task commits.

## Task 10: Empirical Completion Gates

**Files:**
- Runtime artifacts under `checkpoints_mwm/`, `rollouts/`, and `logs/` are ignored by git.
- Configs and verifier code changed only if gates reveal incorrect parameters.

- [ ] **Step 0: PARCC/Betty Slurm preflight before GPU or long-running jobs**

Before running any GPU-backed or long-running empirical command, do not execute it directly on a login node. First inspect the current PARCC documentation:

- Login/auth reference: `https://parcc.upenn.edu/training/getting-started/logging-in/`
- Slurm, GPU partitions, `sbatch`/`srun`, and monitoring reference: `https://parcc.upenn.edu/training/slurm/`

Then update this plan or add a checked-in launch script with the exact `sbatch` or `srun` command/script that will run the gate, including partition, GPU, CPU, memory, wall-time, conda env activation, working directory, output log path, and the Python command. Only after that plan/script update should the job be submitted. Use `squeue`/`sacct` or the documented monitoring command to track the job instead of leaving long processes running in the current terminal.

PARCC docs inspected on 2026-05-28:

- Login/auth: Betty requires PARCC login through `login.betty.parcc.upenn.edu`; current control shell is `login02`, so GPU and long work must be submitted through Slurm, not run directly here.
- Slurm: `sbatch` submits background jobs, `squeue`/`scontrol` monitor active jobs, and `sacct` reports completed resource usage.
- Local `sinfo` confirmed `dgx-b200` exposes `gpu:B200` and `b200-mig90` exposes `gpu:90gb`, both with a 7 day partition limit.

Exact launch path to use:

```bash
cd "$(git rev-parse --show-toplevel)"
scripts/submit_mwm_gates.sh
```

The launcher submits:

```bash
sbatch --parsable scripts/slurm_mwm_paper_parity.sbatch
sbatch --parsable --dependency=afterok:${paper_id} scripts/slurm_mwm_v1_gate.sbatch
```

Paper-parity job details: `scripts/slurm_mwm_paper_parity.sbatch` uses partition `dgx-b200`, GRES `gpu:B200:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `4-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, logs `logs/mwm_paper_parity_%j.out` and `logs/mwm_paper_parity_%j.err`, and runs `scripts/run_mwm_paper_parity.sh` with `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.

Full MWM gate details: `scripts/slurm_mwm_v1_gate.sbatch` uses partition `b200-mig90`, GRES `gpu:90gb:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `7-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, logs `logs/mwm_v1_gate_%j.out` and `logs/mwm_v1_gate_%j.err`, and runs `scripts/run_mwm_v1_gate.sh` with `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.

Single-level fresh-training split launch path, added after the user requested separate jobs for Push-T and Two-Room:

```bash
cd "$(git rev-parse --show-toplevel)"
scripts/submit_mwm_single_level_split.sh
```

The split launcher submits:

```bash
sbatch --parsable scripts/slurm_mwm_train_pusht_single.sbatch
sbatch --parsable scripts/slurm_mwm_train_tworoom_single.sbatch
sbatch --parsable --dependency=afterok:${pusht_id}:${tworoom_id} scripts/slurm_mwm_single_level_benchmark.sbatch
```

Push-T single-level training details: `scripts/slurm_mwm_train_pusht_single.sbatch` uses partition `dgx-b200`, GRES `gpu:B200:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `2-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, logs `logs/mwm_train_pusht_single_%j.out` and `logs/mwm_train_pusht_single_%j.err`, and runs `scripts/run_mwm_train_single_level_env.sh pusht` with `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.

Two-Room single-level training details: `scripts/slurm_mwm_train_tworoom_single.sbatch` uses partition `dgx-b200`, GRES `gpu:B200:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `2-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, logs `logs/mwm_train_tworoom_single_%j.out` and `logs/mwm_train_tworoom_single_%j.err`, and runs `scripts/run_mwm_train_single_level_env.sh tworoom` with `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.

Single-level benchmark details: `scripts/slurm_mwm_single_level_benchmark.sbatch` uses partition `dgx-b200`, GRES `gpu:B200:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `1-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, logs `logs/mwm_single_level_benchmark_%j.out` and `logs/mwm_single_level_benchmark_%j.err`, and runs `scripts/run_mwm_single_level_benchmark.sh` with `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.

Full V1 split launch path, used because old `rollouts/mwm_benchmark` artifacts point at stale generic checkpoints and fresh base-adapter single/scheduled checkpoints are required:

```bash
cd "$(git rev-parse --show-toplevel)"
scripts/submit_mwm_v1_split.sh
```

The full V1 split launcher submits:

```bash
sbatch --parsable scripts/slurm_mwm_train_pusht_v1_single.sbatch
sbatch --parsable scripts/slurm_mwm_train_tworoom_v1_single.sbatch
sbatch --parsable scripts/slurm_mwm_train_pusht_v1_scheduled.sbatch
sbatch --parsable scripts/slurm_mwm_train_tworoom_v1_scheduled.sbatch
sbatch --parsable --dependency=afterok:${pusht_single_id}:${tworoom_single_id}:${pusht_scheduled_id}:${tworoom_scheduled_id} scripts/slurm_mwm_v1_benchmark.sbatch
```

Each full V1 training job uses partition `dgx-b200`, GRES `gpu:B200:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `2-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, and `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`. The four train scripts run `scripts/run_mwm_train_v1_env.sh` with one of `pusht-single`, `tworoom-single`, `pusht-scheduled`, or `tworoom-scheduled`, writing logs under `logs/mwm_train_*_v1_*_%j.{out,err}`.

Full V1 benchmark details: `scripts/slurm_mwm_v1_benchmark.sbatch` uses partition `dgx-b200`, GRES `gpu:B200:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `1-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, logs `logs/mwm_v1_benchmark_%j.out` and `logs/mwm_v1_benchmark_%j.err`, and runs `scripts/run_mwm_v1_benchmark.sh` with `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.

Full V1 benchmark rerun after the blocked-action rollout fix:

```bash
rm -rf rollouts/mwm_benchmark
sbatch --parsable scripts/slurm_mwm_v1_benchmark.sbatch
```

Paper reference Lance-only rerun path, used to refresh stale reference artifacts:

```bash
sbatch --parsable scripts/slurm_mwm_paper_reference.sbatch
```

Paper reference details: `scripts/slurm_mwm_paper_reference.sbatch` uses partition `dgx-b200`, GRES `gpu:B200:1`, `ntasks=1`, `cpus-per-task=16`, memory `128G`, wall time `1-00:00:00`, working directory `SLURM_SUBMIT_DIR` / repository root, logs `logs/mwm_paper_reference_%j.out` and `logs/mwm_paper_reference_%j.err`, and runs `scripts/run_mwm_paper_reference.sh` with `/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python`.

Monitor with:

```bash
squeue -j "${paper_id},${v1_id}" -o '%.18i %.30j %.8T %.10M %.9l %.20R'
sacct -j "${paper_id},${v1_id}" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,ReqMem
squeue -j "${pusht_id},${tworoom_id},${benchmark_id}" -o '%.18i %.30j %.8T %.10M %.9l %.20R'
sacct -j "${pusht_id},${tworoom_id},${benchmark_id}" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,ReqMem
squeue -j "${pusht_single_id},${tworoom_single_id},${pusht_scheduled_id},${tworoom_scheduled_id},${v1_benchmark_id}" -o '%.18i %.30j %.8T %.10M %.9l %.20R'
sacct -j "${pusht_single_id},${tworoom_single_id},${pusht_scheduled_id},${tworoom_scheduled_id},${v1_benchmark_id}" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,ReqMem
squeue -j "${paper_reference_id}" -o '%.18i %.30j %.8T %.10M %.9l %.20R'
sacct -j "${paper_reference_id}" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,ReqMem
```

- [ ] **Step 1: Prepare upstream Le-WM checkpoints and data**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python prepare_upstream_lewm.py
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python prepare_upstream_lewm_data.py
```

Expected: upstream Push-T and Two-Room Le-WM canonical checkpoints and upstream Lance data exist under `checkpoints_mwm/` and `data/upstream/`.

- [ ] **Step 2: Run upstream paper-parity evaluator**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python benchmark_mwm.py configs/benchmark_mwm_paper_parity.yaml --roles upstream_lewm_converted
```

Expected: Push-T upstream success is within 1 percentage point of 96.0 and Two-Room upstream success is within 1 percentage point of 87.0, or the benchmark records that reference fallback is required.

- [ ] **Step 3: Run Stable-WM reference fallback when required**

If Step 2 misses either paper target by more than 1 percentage point, run the reference Stable-WM evaluator path for the same environment/checkpoint. Record rows using role `stable_wm_reference`.

Expected: if reference is within 1 percentage point and MWM evaluator is not, update MWM evaluator/solver parameters and repeat Step 2. If reference also misses, document data/checkpoint/protocol mismatch and do not mark completion.

- [ ] **Step 4: Train fresh single-level MWM from Stable-WM base configs**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python train_mwm.py configs/train_mwm_lewm_pusht.yaml
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python train_mwm.py configs/train_mwm_lewm_tworoom.yaml
```

Expected: exported checkpoints declare `fresh_init: true`, `adapter_family: lewm`, `component_policy.shared: [latent_producer]`, and `K=[192]`.

- [ ] **Step 5: Evaluate fresh single-level MWM**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python benchmark_mwm.py configs/benchmark_mwm_paper_parity.yaml --roles retrained_lewm_single
```

Expected: fresh single-level MWM is within 5 percentage points of validated upstream Le-WM on Push-T and Two-Room.

- [ ] **Step 6: Train and evaluate scheduled multi-level MWM**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python train_mwm.py configs/train_mwm_scheduled_pusht.yaml
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python train_mwm.py configs/train_mwm_scheduled_tworoom.yaml
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python benchmark_mwm.py configs/benchmark_mwm.yaml --roles mwm_scheduled
```

Expected: scheduled checkpoints use `K=[48,96,144,192]`, evaluator runs successfully, and diagnostics include fidelity/latent-work fields.

- [ ] **Step 7: Run final benchmark verifier**

Run:

```bash
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark_mwm_paper_parity.yaml
/vast/projects/dineshj/lab/ethanyu/conda/envs/mwm/bin/python verify_mwm_benchmark.py configs/benchmark_mwm.yaml
```

Expected: both verifiers pass. If not, fix the implementation or evaluator parameters and repeat the relevant empirical gate.

- [ ] **Step 8: Final commit of parameter fixes**

If empirical gates required code/config changes, commit them:

```bash
git add configs eval_mwm.py verify_mwm_benchmark.py benchmark_mwm.py mwm README.md REVIEW_GUIDE.md
git commit -m "fix: align MWM empirical gates with Stable-WM reference"
```

## Self-Review Checklist

- Spec coverage: Tasks 1-8 implement framework deliverables, Task 9 covers unit/static verification, and Task 10 covers empirical completion gates.
- Fresh-init invariant: Tasks 2, 3, 5, and 6 all require source configs without loading weights and metadata records `fresh_init`.
- Shared latent producer invariant: Tasks 1 and 3 enforce `latent_producer` validation and Le-WM `encoder + projector` sharing.
- Loss semantics: Task 4 implements shared regularizers and explicit per-level override.
- Evaluator ladder: Task 7 implements 1 percentage point fallback behavior and Task 10 runs it.
- No arbitrary graph surgery: Component policies are top-level groups only.
