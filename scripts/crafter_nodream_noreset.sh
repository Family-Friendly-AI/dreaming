#!/bin/bash
# Consolidation ablation: no_dream cycle, but pi_f keeps its own weights (no pi_f := pi_s reset).
METHOD=no_dream TAG=noreset exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" \
    --no-reset-fast "$@"
