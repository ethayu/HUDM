#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs rollouts checkpoints_mwm

paper_job="$(sbatch --parsable scripts/slurm_mwm_paper_parity.sbatch)"
paper_id="${paper_job%%;*}"
echo "paper_parity_job=${paper_job}"

v1_job="$(sbatch --parsable --dependency=afterok:${paper_id} scripts/slurm_mwm_v1_gate.sbatch)"
v1_id="${v1_job%%;*}"
echo "v1_gate_job=${v1_job}"

echo "monitor_active=squeue -j ${paper_id},${v1_id} -o '%.18i %.30j %.8T %.10M %.9l %.20R'"
echo "monitor_complete=sacct -j ${paper_id},${v1_id} --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,ReqMem"
squeue -j "${paper_id},${v1_id}" -o '%.18i %.30j %.8T %.10M %.9l %.20R'
