
EXP_NAME=$1

WORKSPACE_DIR='/workspace'

NUM_NODES=$BEAKER_REPLICA_COUNT

ulimit -n 1048576

# apt-get update && apt-get install -y openssh-server
echo "Port 30255" >> /etc/ssh/sshd_config

# start ssh service
service ssh start

NUM_GPUS_PER_WORKER=8

export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
export WEKA_PROFILE=weka 
export OMP_NUM_THREADS=1


cd ${WORKSPACE_DIR}/OLMo-core

pip install --no-build-isolation --no-deps --no-cache-dir -e .[all]
# pip install nvtx
# pip install matplotlib

port=10086

if [ -z "${BEAKER_LEADER_REPLICA_HOSTNAME:-}" ]; then
    NODE0="localhost"
else
    NODE0="$BEAKER_LEADER_REPLICA_HOSTNAME"
fi

export OLMO_SHARED_FS=1 # shared fs
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache # avoid NFS issue
export TRITON_CACHE_DIR=/tmp/triton_cache


unset BEAKER_NODE_HOSTNAME # this node is set to the node that builds the image, not the node that runs the job
export BEAKER_NODE_HOSTNAME=$HOSTNAME


export CUDA_SCALE_LAUNCH_QUEUES=4x # allow more pending kernels

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SYMMMEM_NBLOCKS=128 # [recommend] intra-node: 128 H100, 256 B200; inter-node: max(EP_WORLD_SIZE, 16)
export NVSHMEM_IB_ENABLE_IBGDA=1 # for inter node communication, default to 0

# optional: set NVSHMEM_IBGDA_NIC_HANDLER to disable following init warnings:
# WARN: cudaHostRegister with IoMemory failed with error=800. We may need to use a fallback path.
# WARN: ibgda_nic_mem_gpu_map failed. We may need to use the CPU fallback path.
# WARN: ibgda_alloc_and_map_qp_uar with GPU as handler failed. We may need to enter the CPU fallback path.
export NVSHMEM_IBGDA_NIC_HANDLER=cpu_host_memory #  NVSHMEM v3.4.5


torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend static --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${BEAKER_REPLICA_RANK}" ./src/scripts/train/$EXP_NAME.py train $EXP_NAME ai2/jupiter 

# sleep 3d
