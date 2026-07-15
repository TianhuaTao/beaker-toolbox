#! /bin/bash
# usage: ./script.sh [LOCAL_RANK] [HOST_FILE_PATH] [TIMESTAMP] [PYTHON_SCRIPT] [OLMO_CORE_DIR] [EXTRA_ARGS] [RUN_TAG]
WORKSPACE_DIR='/workspace'
SLURM_NODEID=${1:-0} # default to 0
HOST_FILE_PATH=${2:-"${WORKSPACE_DIR}/hostfile1"} 
TIMESTAMP=${3:-"latest"}
PYTHON_SCRIPT=${4:-"OLMoE3-ablation-dense"}
OLMO_CORE_DIR=${5:-"${WORKSPACE_DIR}/OLMo-core"}
EXTRA_ARGS=${6:-""}
RUN_TAG=${7:-""}

NUM_NODES=$(wc -l < ${HOST_FILE_PATH})

echo $SLURM_NODEID $HOST_FILE_PATH $NUM_NODES

ulimit -n 1048576

############## High-level configs ############## BEGIN
# NODE_NETWORK_TYPE="eth"
NUM_GPUS_PER_WORKER=8
USE_PROFILE=${USE_PROFILE:-1}
############## High-level configs ############## END


export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
export WEKA_PROFILE=weka 
# export WEKA_ENDPOINT_URL=https://weka-aus.beaker.org:9000
# export OLMO_NUM_NODES_ENV_VAR=$NUM_NODES
export OMP_NUM_THREADS=1
export OLMO_MIDTRAIN_WORK_DIR=${OLMO_MIDTRAIN_WORK_DIR:-${WORKSPACE_DIR}/dataset-cache}
# cd ${WORKSPACE_DIR}/beaker-toolbox

# git pull

cd "${OLMO_CORE_DIR}"

# git pull
# python -m pip install --upgrade --no-deps nvidia-nccl-cu12==2.29.7 # unsafe; dev/DEBUG usagel; one-off install at first time in a session.
pip install -e .[all] --no-deps
# pip install matplotlib
# pip install -U liger-kernel==0.6.2
# pip install -U ai2-olmo-eval==0.8.5
# pip install transformers==4.57.3 -U
# pip install triton==3.3.0
# # pip install -e .[all]
# pip install -U liger-kernel==0.6.2
# pip install -U ai2-olmo-eval==0.8.5
# pip install transformers==4.57.3 -U
# pip install triton==3.3.0

# python -c "import torch; print('torch=='+torch.__version__)" > /tmp/torch-constraint.txt
# python -m pip install -c /tmp/torch-constraint.txt "flash-attn-4==4.0.0b12"
sudo apt-get update
sudo apt-get install -y infiniband-diags

# port=24759
port=10086

NODE0=$(head -n 1 "$HOST_FILE_PATH" | awk '{print $1}')

# Accept either a path relative to src/scripts/train or a bare name from its
# tech_report subdirectory. Keep the bare input as the default run tag.
if [[ -f "./src/scripts/train/${PYTHON_SCRIPT}.py" ]]; then
    SCRIPT_REL=${PYTHON_SCRIPT}
elif [[ -f "./src/scripts/train/tech_report/${PYTHON_SCRIPT}.py" ]]; then
    SCRIPT_REL=tech_report/${PYTHON_SCRIPT}
else
    echo "Could not find training script '${PYTHON_SCRIPT}' in ${OLMO_CORE_DIR}" >&2
    exit 2
fi
TAG=${RUN_TAG:-${PYTHON_SCRIPT//\//_}}

# if "google" in hostname
if [[ $(hostname) == *"augusta"* ]]; then
    CLUSTER="ai2/augusta"
    # export NCCL_NET=FasTrak
    export NCCL_DEBUG=WARN 
    export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH
else
    CLUSTER="ai2/jupiter"
    export OLMO_SHARED_FS=1 # shared fs
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache # avoid NFS issue
    export TRITON_CACHE_DIR=/tmp/triton_cache
    # export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME='ib'
fi




unset BEAKER_NODE_HOSTNAME # this node is set to the node that builds the image, not the node that runs the job
export BEAKER_NODE_HOSTNAME=$HOSTNAME
script_path="./src/scripts/train/${SCRIPT_REL}.py"
script_args="train $TAG $CLUSTER $EXTRA_ARGS"
# script_path="${WORKSPACE_DIR}/Megatron-LM/scripts/min_torchrun.py"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL 
# export TORCH_CPP_LOG_LEVEL=INFO 
# export TORCH_CPP_LOG_COMPONENTS=c10d,TCPStore,TCPStoreLibUvBackend,socket 
# export UV_DEBUG=1
# export USE_LIBUV=0
export CUDA_SCALE_LAUNCH_QUEUES=4x # allow more pending kernels
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export TORCH_SYMMMEM_NBLOCKS=256 # [recommend] intra-node: 128 H100, 256 B200; inter-node: max(EP_WORLD_SIZE, 16)
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
# export NVSHMEM_SYMMETRIC_SIZE=8G # OLMo-owned symm backend keeps MoE buffers alive across blocks.
export OLMO_OWN_SYMM_PREWARM=1 # Allocate symmetric MoE buffers before the PP dry-run schedule.

export NVSHMEM_IB_ENABLE_IBGDA=1 # for inter node communication, default to 0
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
# optional: set NVSHMEM_IBGDA_NIC_HANDLER to disable following init warnings:
# WARN: cudaHostRegister with IoMemory failed with error=800. We may need to use a fallback path.
# WARN: ibgda_nic_mem_gpu_map failed. We may need to use the CPU fallback path.
# WARN: ibgda_alloc_and_map_qp_uar with GPU as handler failed. We may need to enter the CPU fallback path.
export NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory #  NVSHMEM v3.4.5

echo "PATH:" $PATH

# run_cmd=${report_mem_cmd}
if [ $USE_PROFILE -eq 1 ]; then
        run_cmd="nsys profile \
        -t nvtx,cuda \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        -o ${WORKSPACE_DIR}/p_${SLURM_NODEID}_${TAG} \
        torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend static --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${script_args}"
        
else
        run_cmd="torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend static --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${script_args}"
fi

echo ${run_cmd}
mkdir -p ${WORKSPACE_DIR}/logs
eval ${run_cmd} 2>&1 | tee ${WORKSPACE_DIR}/logs/logs_${SLURM_NODEID}_${TAG}_${TIMESTAMP}.txt
