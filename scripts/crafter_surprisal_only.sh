#!/bin/bash
# Crafter, method=surprisal_only, single seed (SEED env var, default 0).
METHOD=surprisal_only exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
