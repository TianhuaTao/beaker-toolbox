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
USE_PROFILE=1
############## High-level configs ############## END


export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
export WEKA_PROFILE=weka 
export WEKA_ENDPOINT_URL=https://weka-aus.beaker.org:9000
export OLMO_NUM_NODES_ENV_VAR=$NUM_NODES
export OMP_NUM_THREADS=1
# cd ${WORKSPACE_DIR}/beaker-toolbox

# git pull

cd ${WORKSPACE_DIR}/OLMo-core

git pull

# pip install -e .[all]
# pip install -U liger-kernel==0.6.2
# pip install -U ai2-olmo-eval
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

# run_cmd=${report_mem_cmd}
if [ $USE_PROFILE -eq 1 ]; then
        run_cmd="nsys profile \
        -t nvtx,cuda \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        -o /workspace/prof_${SLURM_NODEID}_${TAG} \
        torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${script_args}"
        
else
        run_cmd="torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${script_args}"
fi

echo ${run_cmd}

eval ${run_cmd} 2>&1 | tee /workspace/logs_${SLURM_NODEID}_${TAG}_${TIMESTAMP}.txt
