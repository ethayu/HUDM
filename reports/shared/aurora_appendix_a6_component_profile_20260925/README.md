# Aurora Appendix A.6 component profile

This folder supports Table 6, “Encoder and decoder size and isolated inference cost,” in Section A.6 of the ICLR Aurora manuscript. The raw output is [`profile_components_rtx5090.json`](profile_components_rtx5090.json); [`profile_components.py`](profile_components.py) is the executable measurement script.

## Provenance and method

- Source architecture: the saved TwoRoom K192 retraining config at [`../published_leworldmodel_k192_parity_20260924_rtx5090/checkpoint_metadata/mwm_paper10_tworoom_k192_sepopt_retrain_20260922/config.json`](../published_leworldmodel_k192_parity_20260924_rtx5090/checkpoint_metadata/mwm_paper10_tworoom_k192_sepopt_retrain_20260922/config.json), SHA-256 `adcd01731671dabfacfb3913ef51a19fbfad6fc2deba13ea563ebe0af29e9a2a`. The JSON records its original path on the somatic-2 measurement host; the archived copy is byte-identical.
- Source code: the `mwm` package from the saved HUDM experiment environment at `/tmp/hudm-5090-20260924-0z4ayR/HUDM` on somatic-2. The archived release source under [`../published_leworldmodel_k192_parity_20260924_rtx5090/reproduction/HUDM`](../published_leworldmodel_k192_parity_20260924_rtx5090/reproduction/HUDM) contains the corresponding module definitions. `ConvImageDecoder` is instantiated for each of the seven latent widths.
- Hardware/software: NVIDIA GeForce RTX 5090, PyTorch 2.11.0+cu130, Python 3.11.15. Batch size 1, FP32, evaluation mode, `torch.inference_mode()`. Inputs are a `(1,3,224,224)` image tensor or `(1,d)` latent tensor. There are 20 warm-up and 100 timed passes per component. Timings are medians of synchronized CUDA-event durations; FLOPs are PyTorch `FlopCounterMode` supported-operator totals for one forward pass.
- The script instantiates the architecture but does **not** load trained weights. Its parameter counts and operator shapes reflect the saved configuration. The device timings are component microbenchmarks, not trained-checkpoint or end-to-end planner timings.

Run from a HUDM checkout with the archived experiment dependencies installed and CUDA available:

```bash
python reports/shared/aurora_appendix_a6_component_profile_20260925/profile_components.py \
  reports/shared/published_leworldmodel_k192_parity_20260924_rtx5090/checkpoint_metadata/mwm_paper10_tworoom_k192_sepopt_retrain_20260922/config.json \
  --output /tmp/aurora_a6_profile.json --warmup 20 --repeats 100
```

## Results used in the manuscript

| Component | Level | Parameters (M) | Counted FLOPs/pass (G) | Median (ms) |
|---|---:|---:|---:|---:|
| Encoder + projector | all | 6.294 | 3.397 | 1.534 |
| Decoder | 48 | 1.235 | 2.920 | 0.513 |
| Decoder | 72 | 1.536 | 2.921 | 0.507 |
| Decoder | 96 | 1.837 | 2.921 | 0.508 |
| Decoder | 120 | 2.138 | 2.922 | 0.507 |
| Decoder | 144 | 2.439 | 2.923 | 0.508 |
| Decoder | 168 | 2.740 | 2.923 | 0.509 |
| Decoder | 192 | 3.041 | 2.924 | 0.510 |

The encoder/projector path is shared across levels in MWM and independently replicated in the seven Single-\(d\) models. Decoder architecture at a given width is the same component profile for either family. Decoder FLOPs grow by 0.124% from \(d=48\) to \(d=192\), despite the larger parameter increase, because the fixed-resolution convolutional stages dominate arithmetic.

## Scope limits

These figures exclude image preprocessing, host-device transfer, transition/dynamics modules, action encoding, CEM, and episode-level control cost. Decoders are used for reconstruction analysis, not the CEM planning path. Therefore the table does not support an inference about planning-speed parity or overall model size. It also is not a per-checkpoint timing audit: the non-K192 level architectures come from the released dense configuration and decoder definition rather than loading each separately trained checkpoint. Some planning frontiers in the manuscript use earlier checkpoint releases, as documented in [`../individual_vs_dense_matched_k_horizon5_iter30_all_envs/README.md`](../individual_vs_dense_matched_k_horizon5_iter30_all_envs/README.md).
