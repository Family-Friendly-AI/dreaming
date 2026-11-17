#!/bin/bash
# Crafter, method=no_gate, single seed (SEED env var, default 0).
METHOD=no_gate exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
