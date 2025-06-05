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


############## High-level configs ############## BEGIN
NODE_NETWORK_TYPE="eth"
NUM_GPUS_PER_WORKER=8
USE_PROFILE=0
############## High-level configs ############## END

# if [ $NODE_NETWORK_TYPE == "ib" ]; then
#         echo "Using Infiniband"
#         # setup infiniband (ai2/jupiter-cirrascale-2)

#         # Use all interfaces starting with `ib`. This selects the IB cards and avoids 
#         # interfaces with names like bond0 and enp0, which are usually ethernet devices.
#         # Ethernet networks are not robust/fast enough for most distributed training workloads.
#         # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname
#         export NCCL_SOCKET_IFNAME=ib

#         # Don't use the IB bond (which uses the attached ethernet cards) for the same reason.
#         # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca
#         export NCCL_IB_HCA=^=mlx5_bond_0
# elif [ $NODE_NETWORK_TYPE == "eth" ]; then
#         echo "Using eth"
#         export NCCL_SOCKET_IFNAME=bond0
# elif [ $NODE_NETWORK_TYPE == "tcpxo" ]; then
#         echo "Using TCPXO"
#         # setup tcpxo 
#         NCCL_LIB_DIR="/var/lib/tcpxo/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
#         export NCCL_NET=FasTrak # optional, it should find this automatically if everything is set correctly
#         # I don't know why this is needed
#         export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
#         export NCCL_FASTRAK_DATA_TRANSFER_TIMEOUT_MS=600000 # 10 min 
# else
#         echo "Unknown network type"
#         exit 1
# fi


export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3 
export WEKA_PROFILE=weka 
export WEKA_ENDPOINT_URL=https://weka-aus.beaker.org:9000
export OLMO_SHARED_FS=1 # shared fs

cd ${WORKSPACE_DIR}/OLMo-core

pip install -e .[all]

port=24759

NODE0=$(head -n 1 "$HOST_FILE_PATH" | awk '{print $1}')

TAG=$PYTHON_SCRIPT # use the same

# if "google" in hostname
if [[ $(hostname) == *"augusta"* ]]; then
    CLUSTER="ai2/augusta-google-1"
else
    CLUSTER="ai2/jupiter-cirrascale-2"
fi


script_path="./src/scripts/train/$PYTHON_SCRIPT.py"
script_args="train $TAG $CLUSTER "
# script_path="${WORKSPACE_DIR}/Megatron-LM/scripts/min_torchrun.py"


# run_cmd=${report_mem_cmd}
if [ $USE_PROFILE -eq 1 ]; then
        run_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} NSYS_ENABLE_PYTHON_SOURCE_CORRELATION=1 nsys profile \
        -t nvtx,cuda,osrt,cublas,cudnn \
        --sample=process-tree \
        --cuda-event-trace=false \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        --trace-fork-before-exec='true' \
        -o /workspace/prof_${SLURM_NODEID}_${TAG} \
        torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${script_args}"
else
        run_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${script_args}"
fi

echo ${run_cmd}

eval ${run_cmd} 2>&1
