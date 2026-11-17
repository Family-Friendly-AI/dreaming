#!/bin/bash
# Crafter, method=kl_only, single seed (SEED env var, default 0).
METHOD=kl_only exec bash "$(dirname "${BASH_SOURCE[0]}")/crafter_common.sh" "$@"
