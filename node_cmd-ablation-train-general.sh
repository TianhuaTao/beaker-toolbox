#! /bin/bash
# usage: ./script.sh [LOCAL_RANK] [HOST_FILE_PATH] [TIMESTAMP] [PYTHON_SCRIPT] [EXTRA_ARGS]
WORKSPACE_DIR='/workspace'
SLURM_NODEID=${1:-0} # default to 0
HOST_FILE_PATH=${2:-"${WORKSPACE_DIR}/hostfile1"} 
TIMESTAMP=${3:-"latest"}
PYTHON_SCRIPT=${4:-"OLMoE3-ablation-dense"}
EXTRA_ARGS=${5:-""}

NUM_NODES=$(wc -l < ${HOST_FILE_PATH})

echo $SLURM_NODEID $HOST_FILE_PATH $NUM_NODES

ulimit -n 1048576

############## High-level configs ############## BEGIN
# NODE_NETWORK_TYPE="eth"
NUM_GPUS_PER_WORKER=8
USE_PROFILE=0
############## High-level configs ############## END


export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
export WEKA_PROFILE=weka 
# export WEKA_ENDPOINT_URL=https://weka-aus.beaker.org:9000
# export OLMO_NUM_NODES_ENV_VAR=$NUM_NODES
export OMP_NUM_THREADS=1
# cd ${WORKSPACE_DIR}/beaker-toolbox

# git pull

cd ${WORKSPACE_DIR}/OLMo-core

# git pull

pip install -e .[all] --no-deps
pip install matplotlib
# pip install -U liger-kernel==0.6.2
# pip install -U ai2-olmo-eval==0.8.5
# pip install transformers==4.57.3 -U
# pip install triton==3.3.0
# # pip install -e .[all]
# pip install -U liger-kernel==0.6.2
# pip install -U ai2-olmo-eval==0.8.5
# pip install transformers==4.57.3 -U
# pip install triton==3.3.0

# port=24759
port=10086

NODE0=$(head -n 1 "$HOST_FILE_PATH" | awk '{print $1}')

TAG=$PYTHON_SCRIPT # use the same

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
script_path="./src/scripts/train/$PYTHON_SCRIPT.py"
script_args="train $TAG $CLUSTER "
# script_path="${WORKSPACE_DIR}/Megatron-LM/scripts/min_torchrun.py"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL 
# export TORCH_CPP_LOG_LEVEL=INFO 
# export TORCH_CPP_LOG_COMPONENTS=c10d,TCPStore,TCPStoreLibUvBackend,socket 
# export UV_DEBUG=1
# export USE_LIBUV=0
export CUDA_SCALE_LAUNCH_QUEUES=4x # allow more pending kernels
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SYMMMEM_NBLOCKS=256 # [recommend] intra-node: 128 H100, 256 B200; inter-node: max(EP_WORLD_SIZE, 16)

export NVSHMEM_IB_ENABLE_IBGDA=1 # for inter node communication, default to 0

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
