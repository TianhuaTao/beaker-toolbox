#!/usr/bin/env bash

# Usage (normally called through run_script_remotely.sh):
#   node_cmd-tech-report-lbl.sh \
#     NODE_RANK HOSTFILE TIMESTAMP SCRIPT_NAME RUN_NAME

set -euo pipefail

WORKSPACE_DIR=/workspace
NODE_RANK=${1:-0}
HOSTFILE=${2:-${WORKSPACE_DIR}/hostfile2}
TIMESTAMP=${3:-latest}
SCRIPT_NAME=${4:-moe_10l_ddp_lbl_0p50}
RUN_NAME=${5:-token-gerrymandering-10l-lbl-0p50}

export USE_PROFILE=0
# export JOB_ID=${RUN_NAME}
# export WANDB_RUN_ID=${RUN_NAME}
# export WANDB_RESUME=allow
export PYTHONPATH=${WORKSPACE_DIR}/OLMo-core/src${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3

if [[ ${TECH_REPORT_LAUNCHER_DRY_RUN:-0} == 1 ]]; then
    printf 'node_rank=%s hostfile=%s nnodes=%s script=%s run_name=%s profile=%s\n' \
        "${NODE_RANK}" \
        "${HOSTFILE}" \
        "$(wc -l < "${HOSTFILE}")" \
        "${SCRIPT_NAME}" \
        "${RUN_NAME}" \
        "${USE_PROFILE}"
    exit 0
fi

exec ${WORKSPACE_DIR}/beaker-toolbox/node_cmd-ablation-train-general.sh \
    "${NODE_RANK}" \
    "${HOSTFILE}" \
    "${TIMESTAMP}" \
    "${SCRIPT_NAME}" \
    "${WORKSPACE_DIR}/OLMo-core" \
    "" \
    "${RUN_NAME}"
