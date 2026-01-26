set -x

echo $PATH
which python
which pip

# apt-get update && apt-get install -y openssh-server
echo "Port 30255" >> /etc/ssh/sshd_config

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

# ----------- install temporary dependencies
apt-get update

# gcloud cli

# sudo apt-get install apt-transport-https ca-certificates gnupg curl
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
# echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# sudo apt-get update && sudo apt-get -y install google-cloud-cli

apt-get install -y bwm-ng

# olmo-core
cd /workspace/OLMo-core
pip install -e .[all] --no-deps # assume dependencies are already installed in image
# pip install -U ai2-olmo-eval==0.8.5
# pip install transformers==4.57.3 -U
# pip install triton==3.3.0

# ----------- install temporary dependencies - done
git config --global user.name "Tianhua Tao"
git config --global user.email "taotianhua@outlook.com"


echo "Ready ..."
sleep 7d
