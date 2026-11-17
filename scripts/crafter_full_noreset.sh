#!/bin/bash
# Dreams under the no-reset cycle: full method, but pi_f keeps its own weights (no pi_f := pi_s reset).
# Compare against scripts/crafter_nodream_noreset.sh.
METHOD=full TAG=noreset exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" \
    --no-reset-fast "$@"
