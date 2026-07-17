#!/usr/bin/env bash

# Usage (normally called through run_script_remotely.sh):
#   node_cmd-tech-report.sh \
#     NODE_RANK HOSTFILE TIMESTAMP SCRIPT_NAME RUN_NAME
#
# SCRIPT_NAME may be either a bare name (for example, OLMoE3-dev-t001) or a
# path relative to src/scripts/train (for example, tech_report/OLMoE3-dev-t001).
# The .py suffix is optional.

set -euo pipefail

WORKSPACE_DIR='/workspace'
SLURM_NODEID=${1:-0}
HOST_FILE_PATH=${2:-"${WORKSPACE_DIR}/hostfile2"}
TIMESTAMP=${3:-"latest"}
PYTHON_SCRIPT=${4:-"OLMoE3-dev-t001"}
RUN_TAG=${5:-""}
EXTRA_ARGS=${6:-""}
OLMO_CORE_DIR="${WORKSPACE_DIR}/OLMo-core"

PYTHON_SCRIPT=${PYTHON_SCRIPT%.py}
case "${PYTHON_SCRIPT}" in
    tech_report/*)
        SCRIPT_REL=${PYTHON_SCRIPT}
        ;;
    */*)
        echo "PYTHON_SCRIPT must be a bare name or start with 'tech_report/'; got '${PYTHON_SCRIPT}'" >&2
        exit 2
        ;;
    *)
        SCRIPT_REL=tech_report/${PYTHON_SCRIPT}
        ;;
esac

SCRIPT_PATH=${OLMO_CORE_DIR}/src/scripts/train/${SCRIPT_REL}.py
if [[ ! -f ${SCRIPT_PATH} ]]; then
    echo "Could not find tech-report training script '${SCRIPT_PATH}'" >&2
    exit 2
fi

TAG=${RUN_TAG:-${SCRIPT_REL##*/}}
NUM_NODES=$(wc -l < "${HOST_FILE_PATH}")

echo "${SLURM_NODEID} ${HOST_FILE_PATH} ${NUM_NODES}"

ulimit -n 1048576

############## High-level configs ############## BEGIN
NUM_GPUS_PER_WORKER=8
USE_PROFILE=${USE_PROFILE:-0}
# The source tree is already made importable through PYTHONPATH below. Avoid
# rewriting editable-package metadata from every node at job startup; concurrent
# editable installs against the shared source tree can expose partial metadata
# to another pip process. Set this to 1 only for an explicit maintenance run.
INSTALL_OLMO_CORE_EDITABLE=${INSTALL_OLMO_CORE_EDITABLE:-0}
############## High-level configs ############## END

export JOB_ID=${TAG}
# export WANDB_RUN_ID=${TAG}
export PYTHONPATH=${OLMO_CORE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
export WEKA_PROFILE=weka

export OMP_NUM_THREADS=1
export OLMO_MIDTRAIN_WORK_DIR=${OLMO_MIDTRAIN_WORK_DIR:-${WORKSPACE_DIR}/dataset-cache}

# port=24759
port=10086

NODE0=$(head -n 1 "${HOST_FILE_PATH}" | awk '{print $1}')

if [[ $(hostname) == *"augusta"* ]]; then
    CLUSTER="ai2/augusta"
    # export NCCL_NET=FasTrak
    export NCCL_DEBUG=WARN
    export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH:-}
else
    CLUSTER="ai2/jupiter"
    export OLMO_SHARED_FS=1
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
    export TRITON_CACHE_DIR=/tmp/triton_cache
    # export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME='ib'
fi

unset BEAKER_NODE_HOSTNAME
export BEAKER_NODE_HOSTNAME
BEAKER_NODE_HOSTNAME=$(hostname)

script_path="./src/scripts/train/${SCRIPT_REL}.py"
script_args="train ${TAG} ${CLUSTER} ${EXTRA_ARGS}"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_CPP_LOG_COMPONENTS=c10d,TCPStore,TCPStoreLibUvBackend,socket
# export UV_DEBUG=1
# export USE_LIBUV=0
export CUDA_SCALE_LAUNCH_QUEUES=4x
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export TORCH_SYMMEM_NBLOCKS=256
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
# export NVSHMEM_SYMMETRIC_SIZE=8G
export OLMO_OWN_SYMM_PREWARM=1

export NVSHMEM_IB_ENABLE_IBGDA=1
export OLMO_TBO_DEBUG_PRINT=${OLMO_TBO_DEBUG_PRINT:-0}
export OLMO_TBO_VERBOSE_DEBUG_PRINT=${OLMO_TBO_VERBOSE_DEBUG_PRINT:-0}
export OLMO_TBO_DEBUG_RANKS=${OLMO_TBO_DEBUG_RANKS:-0,8}
export OLMO_TBO_DEBUG_SYNC=${OLMO_TBO_DEBUG_SYNC:-0}
export OLMO_ROWWISE_DEBUG_PRINT=${OLMO_ROWWISE_DEBUG_PRINT:-0}
export OLMO_ROWWISE_DEBUG_RANKS=${OLMO_ROWWISE_DEBUG_RANKS:-0,8}
export OLMO_ROWWISE_DEBUG_SYNC=${OLMO_ROWWISE_DEBUG_SYNC:-0}
export OLMO_EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG=${OLMO_EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG:-0}
export OLMO_EP_NO_SYNC_SYMM_BUFFER_SUMMARY=${OLMO_EP_NO_SYNC_SYMM_BUFFER_SUMMARY:-0}
export OLMO_DDP_DEBUG_NONFINITE_GRAD=${OLMO_DDP_DEBUG_NONFINITE_GRAD:-0}
export OLMO_DDP_DEBUG_NONFINITE_GRAD_RANKS=${OLMO_DDP_DEBUG_NONFINITE_GRAD_RANKS:-all}
export OLMO_DDP_DEBUG_NONFINITE_GRAD_TOPK=${OLMO_DDP_DEBUG_NONFINITE_GRAD_TOPK:-20}
# export OLMO_MOE_SYMM_LEASE_DEBUG=1
# export OLMO_MOE_SYMM_LEASE_DEBUG_RANKS=8,16
# Set NVSHMEM_IBGDA_NIC_HANDLER to avoid GPU-handler initialization warnings.
export NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory

echo "PATH: ${PATH}"

if [[ ${USE_PROFILE} -eq 1 ]]; then
    run_cmd="nsys profile \
        -t nvtx,cuda \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        -o ${WORKSPACE_DIR}/p_${SLURM_NODEID}_${TAG} \
        torchrun --rdzv_endpoint ${NODE0}:${port} --rdzv_id 20086 --rdzv_backend static --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank ${SLURM_NODEID} ${script_path} ${script_args}"
else
    run_cmd="torchrun --rdzv_endpoint ${NODE0}:${port} --rdzv_id 20086 --rdzv_backend static --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank ${SLURM_NODEID} ${script_path} ${script_args}"
fi

if [[ ${TECH_REPORT_LAUNCHER_DRY_RUN:-0} == 1 ]]; then
    printf 'node_rank=%s hostfile=%s nnodes=%s script=%s script_path=%s run_name=%s profile=%s\ncommand=%s\n' \
        "${SLURM_NODEID}" \
        "${HOST_FILE_PATH}" \
        "${NUM_NODES}" \
        "${SCRIPT_REL}" \
        "${SCRIPT_PATH}" \
        "${TAG}" \
        "${USE_PROFILE}" \
        "${run_cmd}"
    exit 0
fi

cd "${OLMO_CORE_DIR}"

# PYTHONPATH points at the shared source tree, so a per-launch editable install
# is unnecessary. Keep an opt-in path for repairing/provisioning an environment,
# serialized to prevent multiple nodes from mutating shared package metadata.
if [[ ${INSTALL_OLMO_CORE_EDITABLE} -eq 1 ]]; then
    echo "Installing ai2-olmo-core in editable mode (serialized maintenance path)"
    (
        flock -x 9
        /opt/conda/bin/python -m pip install -e '.[all]' --no-deps
    ) 9>"${WORKSPACE_DIR}/.olmo-core-editable-install.lock"
else
    echo "Skipping editable install; using ${OLMO_CORE_DIR}/src via PYTHONPATH"
fi
# sudo apt-get update
# sudo apt-get install -y infiniband-diags

echo "${run_cmd}"
mkdir -p "${WORKSPACE_DIR}/logs"
eval "${run_cmd}" 2>&1 | tee "${WORKSPACE_DIR}/logs/logs_${SLURM_NODEID}_${TAG}_${TIMESTAMP}.txt"
