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

ulimit -n 65536

############## High-level configs ############## BEGIN
# NODE_NETWORK_TYPE="eth"
NUM_GPUS_PER_WORKER=8

############## High-level configs ############## END



cd ${WORKSPACE_DIR}

port=24759

NODE0=$(head -n 1 "$HOST_FILE_PATH" | awk '{print $1}')


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

script_path="/workspace/beaker-toolbox/torchrun_test.py"
script_args=""
# script_path="${WORKSPACE_DIR}/Megatron-LM/scripts/min_torchrun.py"


run_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} ${OLMO_OPTION} torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20186 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${script_args}"


echo ${run_cmd}

eval ${run_cmd} 2>&1 | tee /workspace/logs_${SLURM_NODEID}_${TAG}_${TIMESTAMP}.txt
