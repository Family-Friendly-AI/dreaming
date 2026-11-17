#!/bin/bash
# Crafter, method=go_explore, single seed (SEED env var, default 0).
METHOD=go_explore exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
