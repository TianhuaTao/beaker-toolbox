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
beaker experiment get $BEAKER_EXPERIMENT_ID --format=json | /workspace/beaker-toolbox/make_hostname_from_json.py > /workspace/hostfile
echo "hostfile created"
cat /workspace/hostfile

pip install triton==3.3.0

echo "Ready ..."
sleep 7d
