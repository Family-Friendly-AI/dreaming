#!/bin/bash
# Crafter, method=no_dream, single seed (SEED env var, default 0).
METHOD=no_dream exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
