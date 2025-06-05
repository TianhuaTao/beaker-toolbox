#!/bin/bash

echo "calling init.sh ============"

echo "torchrun make_hostfile_torchrun.py start"

# write hostfile to a file
torchrun \
  --nnodes $BEAKER_REPLICA_COUNT \
  --nproc_per_node 1 \
  --node_rank $BEAKER_REPLICA_RANK \
  --rdzv_id=make_hostfile_job \
  --rdzv_backend=c10d \
  --rdzv_endpoint $BEAKER_LEADER_REPLICA_HOSTNAME:29500 \
  /workspace/beaker-toolbox/make_hostfile_torchrun.py $BEAKER_WORKLOAD_ID

echo "make_hostfile_torchrun.py finished"

rm /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/$BEAKER_REPLICA_RANK.$HOSTNAME

# now hostfile is ready at /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/hostfile
HOSTFILE=/workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/hostfile
TIMESTAMP_FILE=/workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/latest-timestamp.txt

cp $HOSTFILE /workspace/hostfile
cp $TIMESTAMP_FILE /workspace/latest-timestamp.txt

# read the latest timestamp from the file
if [ -f $TIMESTAMP_FILE ]; then
    LATEST_TIMESTAMP=$(cat $TIMESTAMP_FILE)
else
    LATEST_TIMESTAMP=latest
fi

# update lateset codebase
cd /workspace
rm -rf /workspace/OLMo-core
git clone git@github.com:allenai/OLMo-core.git
cd /workspace/OLMo-core
git checkout tianhua/ablation

cd /workspace


echo "init.sh done ============"

