#!/bin/bash
# Single-seed MiniGrid smoke run (CPU is fine). Override via env vars, e.g.
#   METHOD=ppo ENV=MiniGrid-DoorKey-8x8-v0 STEPS=1000000 bash scripts/minigrid_smoke.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

METHOD=${METHOD:-full}
ENV=${ENV:-MiniGrid-DoorKey-5x5-v0}
STEPS=${STEPS:-200000}
CYCLE=${CYCLE:-50000}
SEED=${SEED:-0}
NUM_ENVS=${NUM_ENVS:-16}

uv run python -m dreaming.train \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --env.name "minigrid:${ENV}" \
    --env.num-envs "${NUM_ENVS}" \
    --env.pool-mode serial \
    --total-wake-steps "${STEPS}" \
    --wake-steps-per-cycle "${CYCLE}" \
    --ppo.lr 7e-4 \
    --gate.eps-abs 0.1 \
    --log-every 10 \
    "$@"
