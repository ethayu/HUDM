# July 28 Release Schedule Benchmarks

This suite contains eight schedule matrices: four environments at two task
geometries. Every matrix uses the July 28 dense checkpoint with
`K=[96,120,144,168,192]`, has the upstream run disabled, and contains 3,120
cells after excluding the invalid one-elite CEM settings.

## Configurations

Goal 25, plan 25 primitive actions, execute 10, budget 50:

- `configs/research/release20260728_dense_pusht_all_fidelity_schedules.yaml`
- `configs/research/release20260728_dense_reacher_all_fidelity_schedules.yaml`
- `configs/research/release20260728_dense_ogb_cube_all_fidelity_schedules.yaml`
- `configs/research/release20260728_dense_tworoom_all_fidelity_schedules.yaml`

Goal 50, plan 50 primitive actions, execute 20, budget 100:

- `configs/research/release20260728_dense_pusht_goal50_plan50_execute20_all_fidelity_schedules.yaml`
- `configs/research/release20260728_dense_reacher_goal50_plan50_execute20_all_fidelity_schedules.yaml`
- `configs/research/release20260728_dense_ogb_cube_goal50_plan50_execute20_all_fidelity_schedules.yaml`
- `configs/research/release20260728_dense_tworoom_goal50_plan50_execute20_all_fidelity_schedules.yaml`

All eight use exact goal indexing, 250 episodes, seed 42, dynamics FLOP
accounting, and CEM iteration values `[5,10,15,20,30]`. Evaluation is processed
in five batches of 50 vectorized environments. Their immutable 250-pair
manifests are tracked under `configs/manifest/data/release20260728/`.

## External prerequisites

Git does not contain the large datasets or checkpoints. The execution host
must provide:

```text
data/upstream/pusht_expert_train.lance
data/upstream/reacher.lance
data/upstream/ogb_cube_single_expert.lance
data/upstream/tworoom.lance

checkpoints_mwm/mwm_paper10_pusht_k96_120_144_168_192_release20260728
checkpoints_mwm/mwm_paper10_reacher_k96_120_144_168_192_release20260728
checkpoints_mwm/mwm_paper10_ogb_cube_k96_120_144_168_192_release20260728
checkpoints_mwm/mwm_paper10_tworoom_k96_120_144_168_192_release20260728
```

## Verify the handoff

From the repository root, using the project Python environment:

```bash
python -m pytest -q \
  tests/test_mwm_benchmark_sweep_exclusions.py \
  tests/test_mwm_benchmark_sweep.py
```

## Run one smoke cell

Set `CONFIG` to one of the eight paths above. A 3,120-way shard selection runs
exactly one matrix cell:

```bash
CONFIG=configs/research/release20260728_dense_pusht_all_fidelity_schedules.yaml
python -m mwm.benchmark.matrix "$CONFIG" \
  --num-shards 3120 \
  --shard-index 0 \
  --resume
```

## Run a sharded matrix

Choose a shard count appropriate for the cluster. Every shard must use the
same config and shard count, with zero-based shard indices:

```bash
CONFIG=configs/research/release20260728_dense_pusht_all_fidelity_schedules.yaml
NUM_SHARDS=120
python -m mwm.benchmark.matrix "$CONFIG" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --resume
```

After every shard completes, build the aggregate summaries and Pareto plots:

```bash
python -m mwm.benchmark.matrix "$CONFIG" --finalize-only
```

Each configuration has a separate output directory, so the eight matrices can
run independently. Avoid an unsharded full run unless running all 3,120 cells
serially is intentional.
