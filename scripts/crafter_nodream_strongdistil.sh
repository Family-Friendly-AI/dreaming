#!/bin/bash
# Consolidation ablation: no_dream cycle with no trust region to the old slow policy and 4x distil epochs.
METHOD=no_dream TAG=strongdistil exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" \
    --consolidate.beta-tr 0 --consolidate.distil-epochs 16 "$@"
