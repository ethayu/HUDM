# Identity Delta Static Audit

Artifact root: `${MWM_ARTIFACT_ROOT}`

## PushT

### Rollout Rows

| role | success | episodes | seed | manifest | config | wall sec |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| upstream_lewm_converted | 98.0 | 50 | 42 | `12efc949078d` | `256b3578ef49` | 79.43 |
| retrained_lewm_identity | 92.0 | 50 | 42 | `12efc949078d` | `72d040a7222f` | 51.16 |

### Metadata Differences

| field | upstream | identity | same |
| --- | --- | --- | --- |
| `role` | `"upstream_lewm_converted"` | `null` | False |
| `fresh_init` | `false` | `true` | False |
| `adapter_family` | `"lewm"` | `"lewm"` | True |
| `architecture_version` | `"lewm_base_adapter_v1"` | `"lewm_base_adapter_v1"` | True |
| `training_backend` | `null` | `"stable_worldmodel_lewm"` | False |
| `levels` | `[192]` | `[192]` | True |
| `D` | `null` | `192` | False |
| `action_dim` | `2` | `2` | True |
| `action_block` | `5` | `5` | True |
| `image_shape` | `[224, 224]` | `[224, 224]` | True |
| `restore_spec` | `"pusht_state_goal_state"` | `"pusht_state_goal_state"` | True |
| `source_config_sha256` | `"2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"` | `"2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"` | True |
| `component_policy` | `{"per_level": ["transition"], "reconstructor": [], "shared": ["latent_producer"]}` | `{"per_level": ["transition"], "reconstructor": [], "shared": ["latent_producer"]}` | True |
| `loss_scope` | `{"regularizers": "shared_latent"}` | `{"reconstructor_contributes_to_encoder_loss": false, "reconstructor_detach_encoder": true, "regularizers": "shared_latent"}` | False |
| `training_recipe` | `{"history_size": 3, "loss_scope": {"regularizers": "shared_latent"}, "num_preds": 1}` | `{"action_preprocessing": "standard_scaler", "history_size": 3, "loss_scope": {"reconstructor_contributes_to_encoder_loss": false, "reconstructor_detach_encod...` | False |
| `dataset` | `{"action_key": "action", "pixels_key": "pixels"}` | `{"action_key": "action", "normalized_columns": ["pixels", "action", "proprio", "state"], "path": "data/upstream/pusht_expert_train.lance", "pixels_key": "pix...` | False |
| `action_preprocessing` | `"standard_scaler"` | `"standard_scaler"` | True |
| `epoch` | `null` | `10` | False |
| `last_checkpoint` | `null` | `"logs/mwm_training/retrained_lewm_identity_pusht_upstream/checkpoints/last.ckpt"` | False |

### Training Log Last Metrics

- Training log: `${MWM_ARTIFACT_ROOT}/logs/mwm_train_pusht_identity_6192391.out`
- Max epochs reached: `True`
- Lightning checkpoint epoch/global_step: `9` / `139000`
- Last `fit/loss`: `0.08405739068984985`
- Last `fit/pred_loss`: `0.007397233508527279`
- Last `fit/pred_loss_l0`: `0.007397233508527279`
- Last `fit/rollout_loss`: `0.007397233508527279`
- Last `fit/sigreg_loss`: `0.8515625`
- Last `validate/loss`: `0.12297240644693375`
- Last `validate/loss_epoch`: `0.12297240644693375`
- Last `validate/pred_loss`: `0.003396380692720413`
- Last `validate/pred_loss_epoch`: `0.003396380692720413`
- Last `validate/pred_loss_l0`: `0.003396380692720413`
- Last `validate/pred_loss_l0_epoch`: `0.003396380692720413`
- Last `validate/rollout_loss`: `0.003396380692720413`
- Last `validate/rollout_loss_epoch`: `0.003396380692720413`
- Last `validate/sigreg_loss`: `1.3286446332931519`
- Last `validate/sigreg_loss_epoch`: `1.3286446332931519`

## TwoRoom

### Rollout Rows

| role | success | episodes | seed | manifest | config | wall sec |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| upstream_lewm_converted | 86.0 | 50 | 42 | `6d96d9ae3d59` | `cd6980cf467d` | 58.33 |
| retrained_lewm_identity | 90.0 | 50 | 42 | `6d96d9ae3d59` | `e619d1e0dc4e` | 28.42 |

### Metadata Differences

| field | upstream | identity | same |
| --- | --- | --- | --- |
| `role` | `"upstream_lewm_converted"` | `null` | False |
| `fresh_init` | `false` | `true` | False |
| `adapter_family` | `"lewm"` | `"lewm"` | True |
| `architecture_version` | `"lewm_base_adapter_v1"` | `"lewm_base_adapter_v1"` | True |
| `training_backend` | `null` | `"stable_worldmodel_lewm"` | False |
| `levels` | `[192]` | `[192]` | True |
| `D` | `null` | `192` | False |
| `action_dim` | `2` | `2` | True |
| `action_block` | `5` | `5` | True |
| `image_shape` | `[224, 224]` | `[224, 224]` | True |
| `restore_spec` | `"point_state_goal_state"` | `"point_state_goal_state"` | True |
| `source_config_sha256` | `"2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"` | `"2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"` | True |
| `component_policy` | `{"per_level": ["transition"], "reconstructor": [], "shared": ["latent_producer"]}` | `{"per_level": ["transition"], "reconstructor": [], "shared": ["latent_producer"]}` | True |
| `loss_scope` | `{"regularizers": "shared_latent"}` | `{"reconstructor_contributes_to_encoder_loss": false, "reconstructor_detach_encoder": true, "regularizers": "shared_latent"}` | False |
| `training_recipe` | `{"history_size": 3, "loss_scope": {"regularizers": "shared_latent"}, "num_preds": 1}` | `{"action_preprocessing": "standard_scaler", "history_size": 3, "loss_scope": {"reconstructor_contributes_to_encoder_loss": false, "reconstructor_detach_encod...` | False |
| `dataset` | `{"action_key": "action", "pixels_key": "pixels"}` | `{"action_key": "action", "normalized_columns": ["pixels", "action", "proprio"], "path": "data/upstream/tworoom.lance", "pixels_key": "pixels", "split": "stab...` | False |
| `action_preprocessing` | `"standard_scaler"` | `"standard_scaler"` | True |
| `epoch` | `null` | `10` | False |
| `last_checkpoint` | `null` | `"logs/mwm_training/retrained_lewm_identity_tworoom_upstream/checkpoints/last.ckpt"` | False |

### Training Log Last Metrics

- Training log: `${MWM_ARTIFACT_ROOT}/logs/mwm_train_tworoom_identity_6192392.out`
- Max epochs reached: `True`
- Lightning checkpoint epoch/global_step: `9` / `51000`
- Last `fit/loss`: `0.1453637331724167`
- Last `fit/pred_loss`: `0.01645747758448124`
- Last `fit/pred_loss_l0`: `0.01645747758448124`
- Last `fit/rollout_loss`: `0.01645747758448124`
- Last `fit/sigreg_loss`: `1.4375`
- Last `validate/loss`: `0.15898269414901733`
- Last `validate/loss_epoch`: `0.15898269414901733`
- Last `validate/pred_loss`: `0.007543545216321945`
- Last `validate/pred_loss_epoch`: `0.007543545216321945`
- Last `validate/pred_loss_l0`: `0.007543545216321945`
- Last `validate/pred_loss_l0_epoch`: `0.007543545216321945`
- Last `validate/rollout_loss`: `0.007543545216321945`
- Last `validate/rollout_loss_epoch`: `0.007543545216321945`
- Last `validate/sigreg_loss`: `1.6825405359268188`
- Last `validate/sigreg_loss_epoch`: `1.6825405359268188`
