#!/bin/bash
# Craftax-Classic-Symbolic, method=ppo, single seed (SEED env var, default 0).
METHOD=ppo exec bash "$(dirname "${BASH_SOURCE[0]}")/craftax_common.sh" "$@"
