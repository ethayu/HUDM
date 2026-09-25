# Aurora Appendix A.6: encoder and dynamics profile

This folder supports the corrected Tables 6--7 in the ICLR Aurora manuscript. Section 3's representation module `E` is the ViT-Tiny image encoder plus latent projector. Its action-conditioned dynamics module `G_k` is the **action encoder + predictor + prediction projection**. The auxiliary `ConvImageDecoder` is not `G_k` and is not included. The earlier [`aurora_appendix_a6_component_profile_20260925`](../aurora_appendix_a6_component_profile_20260925/README.md) folder profiled the reconstruction decoder and is superseded for A.6.

## Files and provenance

- [`profile_encoder_dynamics.py`](profile_encoder_dynamics.py): instantiate and measure the encoder and dynamics modules. For every `d` in `[48,72,96,120,144,168,192]`, it builds both a seven-level MWM and a separate Single-`d` architecture, checks equality of matched component parameters and counted FLOPs, and times the components separately. No trained weights are loaded.
- [`profile_encoder_dynamics_a6.json`](profile_encoder_dynamics_a6.json): TwoRoom-source architecture profile used in Table 6. The source is the saved Single-K=192 checkpoint [`config.json`](../published_leworldmodel_k192_parity_20260924_rtx5090/checkpoint_metadata/mwm_paper10_tworoom_k192_sepopt_retrain_20260922/config.json), SHA-256 `adcd01731671dabfacfb3913ef51a19fbfad6fc2deba13ea563ebe0af29e9a2a`, with `K` replaced by the released seven-level list for MWM or one `d` for a Single-`d` instantiation. The seven-level list and history size one are in the [released training configuration](../published_leworldmodel_k192_parity_20260924_rtx5090/reproduction/HUDM/configs/research/train_mwm_lewm_dense_tworoom_k48_72_96_120_144_168_192_paper10_sepopt_20260917.yaml). The profiling host used the archived [release-source reproduction tree](../published_leworldmodel_k192_parity_20260924_rtx5090/reproduction/HUDM/README.md): adapter SHA-256 `26165b03c363948e2a01a6037707dc4135ab22c1ba1379b236e27650a93fac16`, LeWM runtime SHA-256 `ba01613c1c848c229a227a11ed8bdf43582ccc3a88c1b2c3193aeeddadb089df`.
- [Retrained Single-`d` TwoRoom `d=48,72` summary](../individual_vs_dense_matched_k_horizon5_iter30_all_envs/data/tworoom_individual_k48_72_summary.json) and [`d=96`--`192` summary](../individual_vs_dense_matched_k_horizon5_iter30_all_envs/data/tworoom_individual_new_k96_192_summary.json): trained-checkpoint CEM runs used for Table 7. All seven rows are from the new retrained release. The separate performance plot substitutes an older TwoRoom `d=168` checkpoint because the new one's success rate regressed; Table 7 deliberately retains the retrained `d=168` run for a consistent checkpoint family.
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

## Retrained individual planning time and effective audited rate

Table 7 uses the [released `d=48,72` runs](../individual_vs_dense_matched_k_horizon5_iter30_all_envs/data/tworoom_individual_k48_72_summary.json) and [released `d=96`--`192` runs](../individual_vs_dense_matched_k_horizon5_iter30_all_envs/data/tworoom_individual_new_k96_192_summary.json), not the RTX 5090 component microbenchmark or a theoretical peak. The matched protocol is TwoRoom goal offset 25, horizon and receding horizon 5, population 300, 30 CEM iterations, 100 episodes, budget 50 environment steps, and seed 42. For each run, `GFLOPs/episode = dynamics_flops_total / 100 / 1e9`, `plan seconds/episode = plan_time_total_sec / 100`, and `effective GFLOP/s = dynamics_flops_total / plan_time_total_sec / 1e9`. Values are computed from unrounded source totals.

| Single-d level | Dynamics GFLOPs/episode | Plan seconds/episode | Effective audited GFLOP/s |
|---:|---:|---:|---:|
| 48 | 96.3 | 4.14 | 23.3 |
| 72 | 230.8 | 4.13 | 55.9 |
| 96 | 436.2 | 4.13 | 105.6 |
| 120 | 722.4 | 4.17 | 173.1 |
| 144 | 1099.3 | 4.17 | 263.7 |
| 168 | 1576.9 | 4.17 | 377.9 |
| 192 | 2165.2 | 4.23 | 511.9 |

The recorded `plan_time_total_sec` covers whole CEM solves; `dynamics_flops_total` counts only supported dynamics operators. The nonzero FLOP counts arise from `flop_accounting: dynamics_audit`: [`profile_dynamics_call`](../../../mwm/diagnostics/flops.py) wraps each dynamics forward in PyTorch `FlopCounterMode` inside the timed solve, so auditing itself contributes overhead. The source summaries do not record a GPU model. Consequently these are **observed audited-planner ratios**, not dynamics-kernel throughput, hardware utilization, or a clean cross-level latency benchmark. The nearly flat times do not erase the 22.5x increase in counted work; obtaining clean end-to-end time requires matched identified hardware with auditing disabled. The seven source runs each have one seed; Table 7 has no timing uncertainty estimate.

The encoder is invariant to `d`; the dynamics module grows by 22.9x in counted MFLOPs/pass from `d=48` to `d=192`. One MWM encoder plus all seven dynamics modules contains 40.552M parameters, compared with 78.317M for the seven independent Single-`d` encoder-plus-dynamics models. A single Single-`d` checkpoint remains smaller than the full MWM hierarchy. These totals exclude reconstruction networks.

The component profile's near-flat batch-one GPU times do not imply equal dynamics compute or equal planner cost. That profile is architecture-only and has not been validated by loading each released Single-`d` checkpoint's weights; counts follow the saved base configuration and release adapter scaling. Table 7, in contrast, uses trained-checkpoint planning runs, but its elapsed times include FLOP-audit overhead.
