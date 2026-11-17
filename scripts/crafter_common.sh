#!/bin/bash
# Shared Crafter launcher: one method, one seed. Called by scripts/crafter_<method>.sh.
# Env vars: METHOD (required), SEED (0), STEPS (1000000), CYCLE (50000), NUM_ENVS (cores - 2),
# BUDGET_AXIS (total: wake + dream + gate steps count toward STEPS, as in the Crafter paper's 1M
# protocol; set to `wake` for the wake-only budget).
# TAG (optional) is appended to the run name to distinguish variants of one method.
# Extra CLI flags are passed through to dreaming.train.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${METHOD:?set METHOD}"
SEED=${SEED:-0}
STEPS=${STEPS:-1000000}
CYCLE=${CYCLE:-50000}
BUDGET_AXIS=${BUDGET_AXIS:-total}
TAG=${TAG:+_${TAG}}
CORES=${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
# Leave two cores for the learner process and data movement.
NUM_ENVS=${NUM_ENVS:-$(( CORES > 4 ? CORES - 2 : CORES ))}
TRACK=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then TRACK="--track"; fi

if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY="uv run python"; fi

${PY} -m dreaming.train \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --env.name crafter \
    --env.num-envs "${NUM_ENVS}" \
    --env.pool-mode process \
    --total-wake-steps "${STEPS}" \
    --budget-axis "${BUDGET_AXIS}" \
    --wake-steps-per-cycle "${CYCLE}" \
    --ppo.num-steps 256 \
    --ppo.num-minibatches 8 \
    --ppo.update-epochs 3 \
    --ppo.lr 3e-4 \
    --gate.eps-abs 0.25 \
    --gate.n-seeds 16 \
    --gate.max-steps 1000 \
    --dream.minibatch-size 1024 \
    --consolidate.minibatch-size 1024 \
    --log-every 5 \
    --run-name "crafter_${METHOD}${TAG}_s${SEED}_${SLURM_JOB_ID:-local}" \
    ${TRACK} \
    "$@"
