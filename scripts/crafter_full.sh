#!/bin/bash
# Crafter, method=full, single seed (SEED env var, default 0).
METHOD=full exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
