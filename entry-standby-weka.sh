set -x

# start ssh service
service ssh start

# print environment variables
echo $HOSTNAME
echo "BEAKER_REPLICA_COUNT" $BEAKER_REPLICA_COUNT
echo "BEAKER_WORKLOAD_ID" $BEAKER_WORKLOAD_ID
echo "BEAKER_REPLICA_RANK" $BEAKER_REPLICA_RANK
echo "BEAKER_LEADER_REPLICA_HOSTNAME" $BEAKER_LEADER_REPLICA_HOSTNAME

# rm /workspace # remove weka link

# touch a file to indicate this replica is running
# mkdir -p /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID
# touch /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/$BEAKER_REPLICA_RANK.$HOSTNAME

# if BEAKER_LEADER_REPLICA_HOSTNAME is not set, use current hostname
if [ -z "$BEAKER_LEADER_REPLICA_HOSTNAME" ]; then
    BEAKER_LEADER_REPLICA_HOSTNAME=$HOSTNAME
fi

# ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

cd /workspace

# make hostfile
beaker experiment get $BEAKER_EXPERIMENT_ID --format=json | /workspace/beaker-toolbox/make_hostname_from_json.py > /workspace/hostfile
echo "hostfile created"
cat /workspace/hostfile

# ----------- install tmporary dependencies

# gcloud cli
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get -y install google-cloud-cli

sudo apt-get install -y bwm-ng

# olmo-core
cd /workspace
pip install -e ./OLMo-core

pip install triton==3.3.0

# ----------- install tmporary dependencies - done


echo "Ready ..."
sleep 7d
