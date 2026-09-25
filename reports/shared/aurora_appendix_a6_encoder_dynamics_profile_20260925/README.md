# Aurora Appendix A.6: encoder and dynamics profile

This folder supports the corrected Table 6 in the ICLR Aurora manuscript. Section 3's representation module `E` is the ViT-Tiny image encoder plus latent projector. Its action-conditioned dynamics module `G_k` is the **action encoder + predictor + prediction projection**. The auxiliary `ConvImageDecoder` is not `G_k` and is not included. The earlier [`aurora_appendix_a6_component_profile_20260925`](../aurora_appendix_a6_component_profile_20260925/README.md) folder profiled the reconstruction decoder and is superseded for A.6.

## Files and provenance

- [`profile_encoder_dynamics.py`](profile_encoder_dynamics.py): instantiate and measure the encoder and dynamics modules. For every `d` in `[48,72,96,120,144,168,192]`, it builds both a seven-level MWM and a separate Single-`d` architecture, checks equality of matched component parameters and counted FLOPs, and times the components separately. No trained weights are loaded.
- [`profile_encoder_dynamics_a6.json`](profile_encoder_dynamics_a6.json): TwoRoom-source profile used in Table 6. The source is the saved Single-K=192 checkpoint [`config.json`](../published_leworldmodel_k192_parity_20260924_rtx5090/checkpoint_metadata/mwm_paper10_tworoom_k192_sepopt_retrain_20260922/config.json), SHA-256 `adcd01731671dabfacfb3913ef51a19fbfad6fc2deba13ea563ebe0af29e9a2a`, with `K` replaced by the released seven-level list for MWM or one `d` for a Single-`d` instantiation. The seven-level list and history size one are in the [released training configuration](../published_leworldmodel_k192_parity_20260924_rtx5090/reproduction/HUDM/configs/research/train_mwm_lewm_dense_tworoom_k48_72_96_120_144_168_192_paper10_sepopt_20260917.yaml). The profiling host used the archived [release-source reproduction tree](../published_leworldmodel_k192_parity_20260924_rtx5090/reproduction/HUDM/README.md): adapter SHA-256 `26165b03c363948e2a01a6037707dc4135ab22c1ba1379b236e27650a93fac16`, LeWM runtime SHA-256 `ba01613c1c848c229a227a11ed8bdf43582ccc3a88c1b2c3193aeeddadb089df`.
- [`profile_encoder_dynamics_cube_a6.json`](profile_encoder_dynamics_cube_a6.json): action-dimension sensitivity check using the saved OGBench-Cube K192 config (`action_dim=25`; TwoRoom, PushT, and Reacher have `action_dim=10`). Cube adds 150 parameters and 300 counted FLOPs per dynamics pass at each `d`, below Table 6's display precision.
- [`A6_encoder_dynamics_section.tex`](A6_encoder_dynamics_section.tex): text and table inserted into the Overleaf manuscript.

## Measurement definition

The input for one encoder pass is `(1,3,224,224)` FP32. One dynamics pass takes a latent history `(1,1,d)` and an action block `(1,1,10)` and returns `(1,1,d)`; this represents one prediction at TwoRoom's history length one. Timings are uninstrumented synchronized CUDA-event medians over 100 passes after 20 warm-ups, in evaluation and inference mode on an NVIDIA RTX 5090 with PyTorch 2.11.0+cu130 and TF32 disabled. PyTorch `FlopCounterMode` counts supported operators in a separate pass. Inputs are synthetic. The script excludes image preprocessing, transfers, the reconstruction decoder, CEM, and an episode's full rollout loop.

Run with CUDA and the archived experiment dependencies available:

```bash
python reports/shared/aurora_appendix_a6_encoder_dynamics_profile_20260925/profile_encoder_dynamics.py \
  reports/shared/published_leworldmodel_k192_parity_20260924_rtx5090/checkpoint_metadata/mwm_paper10_tworoom_k192_sepopt_retrain_20260922/config.json \
  --output /tmp/aurora_a6_encoder_dynamics.json --warmup 20 --repeats 100 --context 1
```

## Main findings and cross-check

| Module | d | Parameters (M) | Counted MFLOPs/pass | MWM ms | Single-d ms |
|---|---:|---:|---:|---:|---:|
| Encoder + projector | all | 6.294 | 3396.609 | 1.534 | 1.534 |
| Dynamics | 48 | 0.521 | 1.025 | 0.815 | 0.816 |
| Dynamics | 72 | 1.246 | 2.470 | 0.817 | 0.817 |
| Dynamics | 96 | 2.356 | 4.683 | 0.816 | 0.817 |
| Dynamics | 120 | 3.905 | 7.776 | 0.815 | 0.823 |
| Dynamics | 144 | 5.948 | 11.859 | 0.828 | 0.835 |
| Dynamics | 168 | 8.542 | 17.042 | 0.831 | 0.831 |
| Dynamics | 192 | 11.740 | 23.436 | 0.859 | 0.864 |

The encoder is invariant to `d`; the dynamics module grows by 22.9x in counted MFLOPs/pass from `d=48` to `d=192`. One MWM encoder plus all seven dynamics modules contains 40.552M parameters, compared with 78.317M for the seven independent Single-`d` encoder-plus-dynamics models. A single Single-`d` checkpoint remains smaller than the full MWM hierarchy. These totals exclude reconstruction networks.

The [released TwoRoom fixed-level sweep](../tworoom_goal25_sepopt_k48to192_all_configs/data/dense_checkpoint_individual_levels_summary.json) independently audits accumulated dynamics FLOPs from one trained seven-level checkpoint. At matched horizon two, population 100, 30 CEM iterations, and 100 episodes, the reported dynamics costs are 31.101, 74.814, 141.728, 235.161, 358.431, 514.855, and 707.752 GFLOPs per episode for increasing `d`. This confirms the large level-dependent arithmetic difference at planning scale; it is not derived by multiplying the isolated table values. The sweep uses `flop_accounting: dynamics_audit`, so its wall times include profiler overhead and should not be treated as clean end-to-end latency measurements.

The table's near-flat batch-one GPU times do not imply equal dynamics compute or equal planner cost. The component profile is architecture-only and has not been validated by loading each released Single-`d` checkpoint's weights; counts follow the saved base configuration and release adapter scaling. End-to-end latency would require an uninstrumented, matched-hardware evaluation with FLOP auditing disabled.
