mkdir /workspace
cd /workspace

git clone https://github.com/allenai/OLMo-core

cd /workspace/OLMo-core
git checkout tianhua/olmoe-dev
pip install nvtx nvitop
apt update && apt install -y htop
pip install -e .

export OLMO_NUM_NODES_ENV_VAR=${BEAKER_REPLICA_COUNT}

# export NCCL_NET=FasTrak
export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH
# export NCCL_DEBUG=INFO 

torchrun --rdzv_endpoint $BEAKER_LEADER_REPLICA_HOSTNAME:10086 --rdzv_id 20086 --rdzv_backend c10d --nnodes ${BEAKER_REPLICA_COUNT} --nproc-per-node ${BEAKER_ASSIGNED_GPU_COUNT} --node_rank "${BEAKER_REPLICA_RANK}" ./src/scripts/train/OLMoE3-dev-48l-jul10-gcp-warm1.py train OLMo3-moe-integrationtest-5-48L ai2/augusta-google-1


