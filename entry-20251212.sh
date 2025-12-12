set -x

service ssh start


echo $HOSTNAME
echo "BEAKER_REPLICA_COUNT" $BEAKER_REPLICA_COUNT
echo "BEAKER_WORKLOAD_ID" $BEAKER_WORKLOAD_ID
echo "BEAKER_REPLICA_RANK" $BEAKER_REPLICA_RANK
echo "BEAKER_LEADER_REPLICA_HOSTNAME" $BEAKER_LEADER_REPLICA_HOSTNAME

rm /workspace # remove weka link

mkdir -p /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID

touch /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/$BEAKER_REPLICA_RANK.$HOSTNAME

# if BEAKER_LEADER_REPLICA_HOSTNAME is not set, use current hostname
if [ -z "$BEAKER_LEADER_REPLICA_HOSTNAME" ]; then
    BEAKER_LEADER_REPLICA_HOSTNAME=$HOSTNAME
fi

# ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

cd /workspace
# rm -rf beaker-toolbox
# git clone git@github.com:TianhuaTao/beaker-toolbox.git
bash /workspace/beaker-toolbox/init.sh

pip install triton==3.3.0

echo "Ready ..."

# sleep 1d
##### payload

cd /workspace/OLMo-core
unset BEAKER_NODE_HOSTNAME # this node is set to the node that builds the image, not the node that runs the job
export BEAKER_NODE_HOSTNAME=$HOSTNAME

export OLMO_NUM_NODES_ENV_VAR=${BEAKER_REPLICA_COUNT}
export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3

# export NCCL_NET=FasTrak
export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH
# export NCCL_DEBUG=INFO 

torchrun --rdzv_endpoint $BEAKER_LEADER_REPLICA_HOSTNAME:10086 --rdzv_id 20086 --rdzv_backend c10d --nnodes ${BEAKER_REPLICA_COUNT} --nproc-per-node ${BEAKER_ASSIGNED_GPU_COUNT} --node_rank "${BEAKER_REPLICA_RANK}" ./src/scripts/train/OLMoE3-dec12.py train OLMoE3-dec12 ai2/augusta 2>&1 | tee /workspace/workload.log


