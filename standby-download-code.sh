service ssh start


echo $HOSTNAME
echo "BEAKER_REPLICA_COUNT" $BEAKER_REPLICA_COUNT
echo "BEAKER_WORKLOAD_ID" $BEAKER_WORKLOAD_ID
echo "BEAKER_REPLICA_RANK" $BEAKER_REPLICA_RANK
echo "BEAKER_LEADER_REPLICA_HOSTNAME" $BEAKER_LEADER_REPLICA_HOSTNAME

mkdir -p /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID

touch /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/$BEAKER_REPLICA_RANK.$HOSTNAME

# if BEAKER_LEADER_REPLICA_HOSTNAME is not set, use current hostname
if [ -z "$BEAKER_LEADER_REPLICA_HOSTNAME" ]; then
    BEAKER_LEADER_REPLICA_HOSTNAME=$HOSTNAME
fi

cd /workspace
git clone git@github.com:TianhuaTao/beaker-toolbox.git
bash /workspace/beaker-toolbox/init.sh


echo "Ready ..."
sleep 7d
