# Follow-up: paper figures and TwoRoom configuration discrepancy

Added 2026-09-24 after the original report was packaged. This is an interpretive
addendum, **not** a rerun or a change to the recorded results. The original
`handoff_manifest.json` covers the original evidence set; it intentionally does
not list this later note. No original report file was changed for this follow-up.

## Success rates: three different reference points

| Environment | LeWM paper, Figure 6 | Released LeWM checkpoint in this report | September 22 K=192 retrain in this report | Retrain minus released checkpoint |
| --- | ---: | ---: | ---: | ---: |
| TwoRoom | 87% | 83.4% | 93.6% | +10.2 percentage points |
| Reacher | 86% | 96.6% | 97.6% | +1.0 percentage point |
| PushT | 96% | 86.4% | 92.4% | +6.0 percentage points |
| OGBench Cube | 74% | 71.2% | 71.2% | 0.0 percentage points |

Paper values are read from [LeWM Figure 6](https://arxiv.org/pdf/2603.19312).
The two report columns are reconstructed from `seed_results.csv` / `analysis.json`:
five final evaluation seeds, 100 paired episodes per model per seed (500 per
model and environment). Only those two columns were run on paired episode
manifests. The paper figures are not paired with these episodes, and differences
from the paper column must not be treated as a controlled checkpoint effect.

The original `README.md` says that the published TwoRoom model "uses history 3."
Read this as **this report's converted checkpoint/evaluation is configured for
effective history 3**, not as proof of the original authors' training context.
The latter cannot be established from the released configuration alone.

## TwoRoom: what is observed and what is not known

1. The [LeWM paper, Appendix D](https://arxiv.org/html/2603.19312v3), says the
   predictor history length is **1 for TwoRoom** and **3 for PushT and
   OGBench-Cube**. The original v1 appendix also states 1 for TwoRoom. This is
   the documented reason the September 22 TwoRoom retrain used effective
   `training_recipe.history_size: 1`.
2. The authors' public [generic training configuration](https://github.com/lucas-maes/le-wm/blob/main/config/train/lewm.yaml)
   defaults to `history_size: 3`, and its [model configuration](https://github.com/lucas-maes/le-wm/blob/main/config/train/model/lewm.yaml)
   sets `predictor.num_frames: ${history_size}`. A task-specific command-line
   override could have changed the actual TwoRoom training run; that run's
   resolved config/log is not established by these files.
3. The [released TwoRoom Hugging Face config](https://huggingface.co/quentinll/lewm-tworooms/blob/main/config.json)
   declares `predictor.num_frames: 3`. The report copy
   `checkpoint_metadata/upstream_lewm_tworoom/config.json` carries the same
   source setting. Its converted `world_metadata.json` records
   `source_history_size: 3` and `training_recipe.history_size: 3`; these are
   conversion/evaluation metadata inferred from the released config, **not**
   independently recovered original training logs.
4. The retrain's
   `checkpoint_metadata/mwm_paper10_tworoom_k192_sepopt_retrain_20260922/world_metadata.json`
   records `training_recipe.history_size: 1` while retaining source
   `predictor.num_frames: 3`. Both TwoRoom metadata files record the same
   `source_config_sha256` (`2564086e...`), so architecture capacity alone does
   not explain the difference. The bundled adapter
   `reproduction/HUDM/mwm/adapters/lewm.py` selects effective history from
   `training_recipe` before falling back to predictor `num_frames`; the model
   and objective code then use that effective history. Thus the retrain was a
   three-position-capacity predictor trained/evaluated with one-frame context.

`num_frames: 3` is a positional-embedding capacity, not by itself proof that
three observations were supplied on every training step. The authors'
[predictor code](https://github.com/lucas-maes/le-wm/blob/main/module.py)
slices its positional embeddings to the input's actual length. Likewise, the
released config does not prove that the model behind Figure 6 used history 3.
The unresolved discrepancy is between the paper's stated TwoRoom history 1,
the public default/released checkpoint capacity 3, and our adapter's distinct
effective histories for the two checkpoints.

## Interpretation and follow-up

The report's paired manifests and planner parameters were matched, but the
TwoRoom checkpoints were **not matched on effective history** (released
conversion 3; retrain 1). The +10.2-point TwoRoom difference is a measured
planning result under this report's protocol, not evidence of training-recipe
parity and not clean evidence that one training recipe is superior. The other
reported gaps also do not, by themselves, establish paper reproduction:
the paper figures come from a separate evaluation, and the retrain's recorded
recipe includes separate optimizers and a detached auxiliary decoder absent
from the released checkpoint's recorded recipe. Check the checkpoint metadata
for the exact recorded settings; absence from released metadata is not proof
that an operation never occurred in the authors' original run.

The authors' [official repository](https://github.com/lucas-maes/le-wm) says
it builds on `stable-worldmodel` for environments, planning, and evaluation,
and `stable-pretraining` for training. Therefore this should not be framed as
two wholly separate "stable-wm" versus "published LeWM" implementations.

To resolve the history question, obtain the paper run's resolved TwoRoom
training/evaluation configuration or training logs. For a controlled new
comparison, train/evaluate a TwoRoom checkpoint with the same effective
history on both sides, while holding data, objective, architecture, optimizer,
and planning protocol fixed. Merely changing the context length at evaluation
would be a diagnostic, not an equivalent retraining comparison.
