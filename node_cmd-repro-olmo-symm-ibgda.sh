#! /bin/bash
# usage:
#   bash /workspace/beaker-toolbox/run_script_remotely.sh \
#     /workspace/hostfile2 \
#     /workspace/beaker-toolbox/node_cmd-repro-olmo-symm-ibgda.sh
#
# Optional extra args are passed to olmo_symm_ibgda_repro.py, for example:
#   "--symm-source"

WORKSPACE_DIR="/workspace"
SLURM_NODEID=${1:-0}
HOST_FILE_PATH=${2:-"${WORKSPACE_DIR}/hostfile2"}
TIMESTAMP=${3:-"latest"}
EXTRA_ARGS="${*:4}"

NUM_NODES=$(wc -l < "${HOST_FILE_PATH}")
NUM_GPUS_PER_WORKER=${NUM_GPUS_PER_WORKER:-8}
NODE0=$(head -n 1 "${HOST_FILE_PATH}" | awk '{print $1}')
PORT=${PORT:-10087}

echo "node_rank=${SLURM_NODEID} hostfile=${HOST_FILE_PATH} nnodes=${NUM_NODES}"

ulimit -n 1048576

cd "${WORKSPACE_DIR}/OLMo-core" || exit 1
pip install -e .[all] --no-deps

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_SCALE_LAUNCH_QUEUES=${CUDA_SCALE_LAUNCH_QUEUES:-4x}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export OLMO_SYMM_VDEV2D_AUTO_BUILD=${OLMO_SYMM_VDEV2D_AUTO_BUILD:-0}
export OLMO_SYMM_VDEV2D_BUILD_BACKEND=${OLMO_SYMM_VDEV2D_BUILD_BACKEND:-cmake}

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib}

export OLMO_USE_OWN_SYMM_MEM=1
export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-1}
export NVSHMEM_IBGDA_NIC_HANDLER=${NVSHMEM_IBGDA_NIC_HANDLER:-cpu_host_memory}
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME}}

mkdir -p "${WORKSPACE_DIR}/logs"

run_cmd="torchrun \
  --rdzv_endpoint ${NODE0}:${PORT} \
  --rdzv_id olmo-symm-ibgda-repro \
  --rdzv_backend static \
  --nnodes ${NUM_NODES} \
  --nproc-per-node ${NUM_GPUS_PER_WORKER} \
  --node_rank ${SLURM_NODEID} \
  ./src/scripts/repro/olmo_symm_ibgda_repro.py ${EXTRA_ARGS}"

echo "${run_cmd}"
eval "${run_cmd}" 2>&1 | tee "${WORKSPACE_DIR}/logs/olmo_symm_ibgda_repro_${SLURM_NODEID}_${TIMESTAMP}.txt"
