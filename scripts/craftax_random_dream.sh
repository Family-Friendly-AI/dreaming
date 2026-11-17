#!/bin/bash
# Craftax-Classic-Symbolic, method=random_dream (uniform-action branches, same budget as dream), single seed.
METHOD=random_dream BUDGET_AXIS=${BUDGET_AXIS:-wake} exec bash "$(dirname "${BASH_SOURCE[0]}")/craftax_common.sh" "$@"
