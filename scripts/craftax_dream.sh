#!/bin/bash
# Craftax-Classic-Symbolic, method=dream (targeted branches update pi_f each cycle), single seed (SEED, default 0).
# Wake-step budget by default: dreams are free, the upper bound on what they can add over ppo at equal wake steps.
METHOD=dream BUDGET_AXIS=${BUDGET_AXIS:-wake} exec bash "$(dirname "${BASH_SOURCE[0]}")/craftax_common.sh" "$@"
