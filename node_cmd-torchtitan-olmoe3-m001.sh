#! /bin/bash
# usage:
#   node_cmd-torchtitan-olmoe3-m001.sh [NODE_RANK] [HOSTFILE] [TIMESTAMP] [BACKEND] [QUANT] [EXTRA_TORCHTITAN_ARGS...]
#
# Intended caller:
#   bash /workspace/beaker-toolbox/run_script_remotely.sh \
#     /workspace/hostfile \
#     /workspace/beaker-toolbox/node_cmd-torchtitan-olmoe3-m001.sh
#
# Variant examples through run_script_remotely.sh:
#   ... node_cmd-torchtitan-olmoe3-m001.sh "standard bf16"
#   ... node_cmd-torchtitan-olmoe3-m001.sh "standard float8"
#   ... node_cmd-torchtitan-olmoe3-m001.sh "deepep bf16"
#   ... node_cmd-torchtitan-olmoe3-m001.sh "hybridep mxfp8"
#   ... node_cmd-torchtitan-olmoe3-m001.sh "standard bf16 --training.steps 10"

set -euo pipefail

WORKSPACE_DIR="/workspace"
NODE_RANK_ARG=${1:-0}
HOST_FILE_PATH=${2:-"${WORKSPACE_DIR}/hostfile"}
TIMESTAMP=${3:-"latest"}
BACKEND_ARG=${4:-${BACKEND:-standard}}
QUANT_ARG=${5:-${QUANT:-bf16}}

shift $(( $# < 5 ? $# : 5 ))
EXTRA_ARGS=("$@")

if [[ ! -f "${HOST_FILE_PATH}" ]]; then
  echo "Hostfile not found: ${HOST_FILE_PATH}" >&2
  exit 1
fi

NUM_NODES=$(grep -vcE '^[[:space:]]*(#|$)' "${HOST_FILE_PATH}")
NODE0=$(grep -vE '^[[:space:]]*(#|$)' "${HOST_FILE_PATH}" | head -n 1 | awk '{print $1}')

NUM_GPUS_PER_WORKER=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-10087}
WORLD_SIZE=$((NUM_NODES * NUM_GPUS_PER_WORKER))

echo "TorchTitan OLMoE3 m001 compare"
echo "NODE_RANK=${NODE_RANK_ARG} HOST_FILE_PATH=${HOST_FILE_PATH} NUM_NODES=${NUM_NODES} NODE0=${NODE0}"
echo "BACKEND=${BACKEND_ARG} QUANT=${QUANT_ARG} GPUS_PER_NODE=${NUM_GPUS_PER_WORKER} WORLD_SIZE=${WORLD_SIZE}"

ulimit -n 1048576

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

if [[ $(hostname) == *"augusta"* ]]; then
  export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH:-}
else
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib}
fi

cd "${WORKSPACE_DIR}/ref/torchtitan"

if [[ "${INSTALL_TORCHTITAN:-1}" == "1" ]]; then
  python3 -m pip install -e .
fi

export WORLD_SIZE
export NNODES=${NUM_NODES}
export GPUS_PER_NODE=${NUM_GPUS_PER_WORKER}
export NODE_RANK=${NODE_RANK_ARG}
export MASTER_ADDR=${NODE0}
export MASTER_PORT
export RDZV_ID=${RDZV_ID:-torchtitan-olmoe3-m001-${TIMESTAMP}}
export BACKEND=${BACKEND_ARG}
export QUANT=${QUANT_ARG}

mkdir -p "${WORKSPACE_DIR}/logs"

run_cmd=(
  "${WORKSPACE_DIR}/ref/torchtitan/scripts/run_deepseek_v3_olmoe3_m001.sh"
  "${EXTRA_ARGS[@]}"
)

printf 'Run command:'
printf ' %q' "${run_cmd[@]}"
printf '\n'

"${run_cmd[@]}" 2>&1 | tee "${WORKSPACE_DIR}/logs/torchtitan_olmoe3_m001_${NODE_RANK_ARG}_${BACKEND_ARG}_${QUANT_ARG}_${TIMESTAMP}.txt"
