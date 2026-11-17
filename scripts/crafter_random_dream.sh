#!/bin/bash
# Crafter, method=random_dream, single seed (SEED env var, default 0).
METHOD=random_dream exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
