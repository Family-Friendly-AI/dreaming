#!/bin/bash
# Crafter, method=no_bonus, single seed (SEED env var, default 0).
METHOD=no_bonus exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
