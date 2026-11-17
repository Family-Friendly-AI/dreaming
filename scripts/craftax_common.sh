#!/bin/bash
# Shared Craftax-Classic-Symbolic launcher: one method, one seed. Called by scripts/craftax_<method>.sh.
# Env vars: METHOD (required), SEED (0), STEPS (1000000), CYCLE (50000), NUM_ENVS (64), NUM_STEPS (64),
# BUDGET_AXIS (total: wake + dream + gate steps count toward STEPS; set to `wake` for the wake-only budget).
# TAG (optional) is appended to the run name to distinguish variants of one method.
# Extra CLI flags are passed through to dreaming.train.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${METHOD:?set METHOD}"
SEED=${SEED:-0}
STEPS=${STEPS:-1000000}
CYCLE=${CYCLE:-50000}
BUDGET_AXIS=${BUDGET_AXIS:-total}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-64}
TAG=${TAG:+_${TAG}}
TRACK=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then TRACK="--track"; fi

if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY="uv run python"; fi

${PY} -m dreaming.train \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --env.num-envs "${NUM_ENVS}" \
    --total-wake-steps "${STEPS}" \
    --budget-axis "${BUDGET_AXIS}" \
    --wake-steps-per-cycle "${CYCLE}" \
    --ppo.num-steps "${NUM_STEPS}" \
    --ppo.num-minibatches 8 \
    --ppo.update-epochs 4 \
    --ppo.lr 3e-4 \
    --gate.eps-abs 0.25 \
    --gate.n-seeds 16 \
    --gate.max-steps 1000 \
    --dream.minibatch-size 1024 \
    --consolidate.minibatch-size 1024 \
    --log-every 5 \
    --run-name "craftax_${METHOD}${TAG}_s${SEED}_${SLURM_JOB_ID:-local}" \
    ${TRACK} \
    "$@"
