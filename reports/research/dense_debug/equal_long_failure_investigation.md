# Equal-Long Dense Failure Investigation

Date: 2026-06-07

## Conclusion

The 80-epoch equal-weight run was worse for two different reasons:

1. TwoRoom is a clear final-checkpoint failure. The run reached its best validation prediction loss around validation snapshot 34, then collapsed badly before export. Because training uses `ModelCheckpoint(save_last=True, save_top_k=0)`, only the terminal `last.ckpt` was exported to the canonical MWM checkpoint. This explains the very poor TwoRoom equal-long eval.
2. PushT is not ordinary validation-loss overtraining. Its final validation prediction losses are better than the 10-epoch checkpoint, but K144/K192 planning performance is worse while K96 remains roughly comparable. This means the longer equal-weight objective improved one-step prediction loss without preserving the high-K planner geometry that made the canonical dense checkpoint strong.

So the earlier shorthand "train until convergence made it worse" is too broad. The verified story is: the equal-long experiment was not a clean convergence continuation, exported last rather than best, and for PushT the supervised prediction metric is misaligned with high-K planning quality.

## Evidence

Training/export mechanics:

- The training code computes scheduler length as `total_steps = max_epochs * len(train_loader)`. Raising `max_epochs` from 10 to 80 stretched the cosine LR schedule 8x; it did not simply continue the 10-epoch recipe.
- Approximate LR by epoch under the existing schedule:
  - 10-epoch config: epoch 10 is at `0`.
  - 80-epoch config: epoch 10 is still near `4.84e-5`; epoch 43 is still near `2.24e-5`.
- Checkpointing uses `save_last=True, save_top_k=0`, so no best-validation checkpoint was retained.
- Equal-long configs also changed checkpoint cadence from every 1000 train steps to every 5000 train steps.

Validation loss comparison:

| run | snapshots | best pred | final pred | final / best | best l5 | final l5 | final / best l5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical PushT | 11 | 0.002341 | 0.002341 | 1.00 | 0.008210 | 0.008210 | 1.00 |
| equal-long PushT | 81 | 0.001088 | 0.001103 | 1.01 | 0.003396 | 0.003473 | 1.02 |
| canonical TwoRoom | 11 | 0.002884 | 0.002884 | 1.00 | 0.012756 | 0.012756 | 1.00 |
| equal-long TwoRoom | 45 | 0.003875 | 0.197181 | 50.88 | 0.014291 | 0.779307 | 54.53 |

Planning eval comparison:

| checkpoint | env | K96 easy | K144 easy | K192 easy | K192 cost96 easy | K96 all | K144 all | K192 all | K192 cost96 all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical dense | PushT | 63.3 | 84.7 | 89.3 | 87.3 | 22.6 | 32.7 | 32.1 | 32.8 |
| equal-long dense | PushT | 62.7 | 66.0 | 65.3 | 70.7 | 23.9 | 26.6 | 24.1 | 30.0 |
| canonical dense | TwoRoom | 100.0 | 93.3 | 86.0 | 99.3 | 74.4 | 60.8 | 46.1 | 79.0 |
| equal-long dense | TwoRoom | 29.3 | 24.0 | 23.3 | 20.7 | 13.9 | 12.8 | 11.9 | 14.4 |

Interpretation:

- TwoRoom: validation loss blow-up is sufficient to explain the poor eval. The exported checkpoint is after the collapse.
- PushT: K96 is essentially unchanged, so the low/mid prefix remains usable. The degradation appears at K144/K192, where the canonical checkpoint had a large planning gain and equal-long lost most of it. Because validation loss improved, the cause is objective/metric mismatch for high-K planning rather than stale training or simple overtraining.

## What To Do Next

- Do not rerun equal-weight 80-epoch training as-is.
- For future long training, save best checkpoints (`save_top_k > 0`) on a high-K-aware validation metric and export best, not last.
- If the goal is a true "continue training" test, keep the 10-epoch LR schedule semantics or resume from the 10-epoch checkpoint with an explicit continuation schedule; do not silently stretch the cosine schedule by changing `max_epochs`.
- The next useful training probe remains the high-K-weighted config, because the confirmed PushT failure is high-K planner utility, not low-K model quality.
