# RTX 5090 published vs September 22 K=192 comparison

All results use the relayed HUDM source snapshot and the verified published
converted checkpoints and September 22 separate-optimizer checkpoints.
`analysis.json` reports the screen and conditional final runs separately.
`paired_episode_outcomes.csv` contains the paired binary outcomes;
`performance.csv` contains measured planning time, wall time and audited dynamics FLOPs,
alongside plan, step and CEM cost-call counts.

Rebuild all statistics and the PNG/PDF/SVG figure using:

    python analyze_results.py .

The analysis requires matplotlib and the Python standard library. Full execution
uses the pinned environment in `provenance/local/`, the copied experiment code
in `reproduction/`, and the checkpoint/data identities in the provenance records.
The completed source comparison is recorded in `source_input_verification_receipt.json`
under `provenance/local/`, with the actual execution-tree comparison in
`execution_source_hash_check.json`. Preliminary inventory receipts retain their
original capture-time status; `readiness.json` and the final verification receipt
record the completed input checks.
Wall time is the duration of the original per-model evaluator call. Command
history separately records resume and validation executions; verified completed
cells retain their original evaluation totals.
Original evaluator artifacts remain at the paths and SHA-256 hashes recorded in
`external_artifacts.json`. Compact config, manifest, dependency, trace and audit
evidence is included under `evidence/`. Source collector outputs are independently
checked against statistics reconstructed from the episode outcomes.

Protocol: CEM population 300, top-k 30, horizon/receding horizon/action block 5,
10 iterations for Cube/Reacher/TwoRoom and 30 for PushT, rollout budget 50,
50 simultaneous environments, fixed singleton K=192, dynamics-audit FLOPs.
The configured goal offset is 25 with `upstream_lewm_end_exclusive` indexing,
so the actual start-to-goal step difference is 24. The published TwoRoom model
uses history 3 and the retrained model uses history 1. This is a planning
performance comparison of fixed checkpoints; it does not establish that their
training recipes are identical. Local Python is 3.11 while source Python is 3.10;
the core package versions and package integrity checks are recorded.

Screen seeds: 0, 1, 2, 42, 100, at 50 episodes per model per seed.
Only environments with mean retrained-minus-published screen difference >= -5 pp
advance. Final seeds: 2002, 71623, 82715, 86604, 91943, at 100 episodes per model
per seed. Every reported interval is a two-sided Student t interval across five
evaluation seeds (df=4). Intervals describe evaluation-manifest variability for
fixed checkpoints, not training-seed uncertainty. The screen gate uses the point
estimate; passing it is not a statistical noninferiority or equivalence claim.
Conditional final results must be interpreted alongside all four screen results.

Execution recovery: the first Cube screen attempt was killed by host-memory
exhaustion while substantial unrelated shared memory occupied this machine.
No result from that failed attempt was accepted. The already-qualified TwoRoom,
Reacher and PushT final runs proceeded while Cube awaited more available RAM;
all runs retain their original episode counts, 50-environment concurrency,
checkpoints, seeds and planner settings. The failed attempt, retry and actual
execution order remain recorded in the local receipts, logs and command history.
Separate guarded CPU and CUDA setup diagnostics tested unused-memory release;
the CPU test also compared Python allocators with fixed diagnostic seeds.
Model/scaler values and RNG states were preserved. Diagnostic allocator settings
were not applied to the scientific runs. Diagnostic logs and script snapshots
are included alongside their receipts.
