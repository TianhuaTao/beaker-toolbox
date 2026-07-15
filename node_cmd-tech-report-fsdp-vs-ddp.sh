#!/usr/bin/env bash

# Usage (normally called through run_script_remotely.sh):
#   node_cmd-tech-report-fsdp-vs-ddp.sh \
#     LOCAL_RANK HOSTFILE TIMESTAMP moe_8l_ddp NUM_EXPERTS GLOBAL_BATCH_K EP_DEGREE
#   node_cmd-tech-report-fsdp-vs-ddp.sh \
#     LOCAL_RANK HOSTFILE TIMESTAMP moe_8l_fsdp NUM_EXPERTS GLOBAL_BATCH_K
#   node_cmd-tech-report-fsdp-vs-ddp.sh \
#     LOCAL_RANK HOSTFILE TIMESTAMP DENSE_SCRIPT GLOBAL_BATCH_K
#
# SCRIPT_NAME is bare, for example:
#   moe_8l_ddp
#   moe_8l_fsdp
#   dense_8l_ddp
#   dense_8l_fsdp

set -euo pipefail

WORKSPACE_DIR=/workspace
LOCAL_RANK=${1:-0}
HOSTFILE=${2:-${WORKSPACE_DIR}/hostfile2}
TIMESTAMP=${3:-latest}
SCRIPT_NAME=${4:-moe_8l_ddp}
if [[ ${SCRIPT_NAME} == dense_8l_fsdp || ${SCRIPT_NAME} == dense_8l_ddp ]]; then
    NUM_EXPERTS=
    GLOBAL_BATCH_K=${5:-4096}
    EP_DEGREE=
else
    NUM_EXPERTS=${5:-64}
    GLOBAL_BATCH_K=${6:-4096}
    case "${NUM_EXPERTS}" in
        8|32|48|64|128) ;;
        *)
            echo "NUM_EXPERTS must be one of: 8 32 48 64 128; got '${NUM_EXPERTS}'" >&2
            exit 2
            ;;
    esac
    if [[ ${SCRIPT_NAME} == moe_8l_ddp ]]; then
        EP_DEGREE=${7:-8}
        case "${EP_DEGREE}" in
            1|2|4|8) ;;
            *)
                echo "EP_DEGREE must be one of: 1 2 4 8; got '${EP_DEGREE}'" >&2
                exit 2
                ;;
        esac
    else
        EP_DEGREE=
    fi
fi

case "${GLOBAL_BATCH_K}" in
    128|256|512|1024|2048|4096|8192|16384|32768) ;;
    *)
        echo "GLOBAL_BATCH_K must be one of: 128 256 512 1024 2048 4096 8192 16384 32768; got '${GLOBAL_BATCH_K}'" >&2
        exit 2
        ;;
esac

case "${SCRIPT_NAME}" in
    moe_8l_ddp|dense_8l_ddp)
        OLMO_CORE_DIR=${WORKSPACE_DIR}/OLMo-core
        ;;
    moe_8l_fsdp|dense_8l_fsdp)
        OLMO_CORE_DIR=${WORKSPACE_DIR}/OLMo-core-main
        ;;
    *)
        echo "Unsupported SCRIPT_NAME '${SCRIPT_NAME}'" >&2
        echo "Expected: moe_8l_ddp, moe_8l_fsdp, dense_8l_ddp, or dense_8l_fsdp" >&2
        exit 2
        ;;
esac

export TECH_REPORT_GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_K * 1024))
if [[ -n ${NUM_EXPERTS} ]]; then
    export TECH_REPORT_NUM_EXPERTS=${NUM_EXPERTS}
    if [[ -n ${EP_DEGREE} ]]; then
        export TECH_REPORT_PARALLEL_DEGREE=${EP_DEGREE}
        RUN_TAG=${SCRIPT_NAME}-ep${EP_DEGREE}-e${NUM_EXPERTS}-gb${GLOBAL_BATCH_K}k
    else
        unset TECH_REPORT_PARALLEL_DEGREE
        RUN_TAG=${SCRIPT_NAME}-e${NUM_EXPERTS}-gb${GLOBAL_BATCH_K}k
    fi
else
    unset TECH_REPORT_NUM_EXPERTS
    unset TECH_REPORT_PARALLEL_DEGREE
    RUN_TAG=${SCRIPT_NAME}-gb${GLOBAL_BATCH_K}k
fi

if [[ ${TECH_REPORT_LAUNCHER_DRY_RUN:-0} == 1 ]]; then
    printf 'script=%s repo=%s experts=%s ep_degree=%s global_batch_tokens=%s run_tag=%s\n' \
        "${SCRIPT_NAME}" \
        "${OLMO_CORE_DIR}" \
        "${NUM_EXPERTS:-dense}" \
        "${EP_DEGREE:-none}" \
        "${TECH_REPORT_GLOBAL_BATCH_SIZE}" \
        "${RUN_TAG}"
    exit 0
fi

exec ${WORKSPACE_DIR}/beaker-toolbox/node_cmd-ablation-train-general.sh \
    "${LOCAL_RANK}" \
    "${HOSTFILE}" \
    "${TIMESTAMP}" \
    "${SCRIPT_NAME}" \
    "${OLMO_CORE_DIR}" \
    "" \
    "${RUN_TAG}"
