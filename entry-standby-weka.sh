# set -euo pipefail
set -x

echo "${PATH}"
which python
which pip

# apt-get update && apt-get install -y openssh-server
echo "Port 30255" >> /etc/ssh/sshd_config

# start ssh service
service ssh start

# print environment variables
echo "${HOSTNAME}"
echo "BEAKER_REPLICA_COUNT" "${BEAKER_REPLICA_COUNT:-}"
echo "BEAKER_WORKLOAD_ID" "${BEAKER_WORKLOAD_ID:-}"
echo "BEAKER_REPLICA_RANK" "${BEAKER_REPLICA_RANK:-}"
echo "BEAKER_LEADER_REPLICA_HOSTNAME" "${BEAKER_LEADER_REPLICA_HOSTNAME:-}"

# rm /workspace # remove weka link

# touch a file to indicate this replica is running
# mkdir -p /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID
# touch /workspace/beaker_jobs/$BEAKER_WORKLOAD_ID/$BEAKER_REPLICA_RANK.$HOSTNAME

# if BEAKER_LEADER_REPLICA_HOSTNAME is not set, use current hostname
if [ -z "${BEAKER_LEADER_REPLICA_HOSTNAME:-}" ]; then
    BEAKER_LEADER_REPLICA_HOSTNAME="${HOSTNAME}"
fi

# ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

cd /workspace

# /workspace is a shared Weka mount, so all replicas writing /workspace/hostfile
# at once can leave a partial file. Build one workload-scoped file under a lock,
# then atomically point /workspace/hostfile at it for existing scripts.
mkdir -p /workspace/.locks /workspace/hostfiles
WORKLOAD_ID="${BEAKER_WORKLOAD_ID:-${BEAKER_EXPERIMENT_ID:-manual}}"
HOSTFILE="/workspace/hostfiles/${WORKLOAD_ID}.hostfile"
HOSTFILE_LINK="/workspace/hostfile"
HOSTFILE_LOCK="/workspace/.locks/hostfile.${WORKLOAD_ID}.lock"
(
    flock 200
    if [ ! -s "${HOSTFILE}" ]; then
        tmp_hostfile="${HOSTFILE}.${HOSTNAME}.$$.tmp"
        beaker experiment get "${BEAKER_EXPERIMENT_ID:?BEAKER_EXPERIMENT_ID is required to create hostfile}" --format=json \
            | /workspace/beaker-toolbox/make_hostname_from_json.py > "${tmp_hostfile}"
        test -s "${tmp_hostfile}"
        mv -f "${tmp_hostfile}" "${HOSTFILE}"
    fi

    tmp_link="${HOSTFILE_LINK}.${HOSTNAME}.$$.tmp"
    rm -f "${tmp_link}"
    ln -s "${HOSTFILE}" "${tmp_link}"
    mv -Tf "${tmp_link}" "${HOSTFILE_LINK}"
) 200>"${HOSTFILE_LOCK}"
echo "hostfile created at ${HOSTFILE}"
cat /workspace/hostfile

# ----------- install temporary dependencies
apt-get update

# gcloud cli

# sudo apt-get install apt-transport-https ca-certificates gnupg curl
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
# echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# sudo apt-get update && sudo apt-get -y install google-cloud-cli

# apt-get install -y bwm-ng tmux
# pip install nvtx

# olmo-core
# The Python environment is local to each node, but setuptools writes editable
# install metadata into the shared source tree (src/*.egg-info and build/).
# Serialize the install so 64 replicas do not race on those shared files.

cd /workspace/OLMo-core
python -m pip install -e '.[all]' --no-deps # assume dependencies are already installed in image

python - <<'PY'
import olmo_core

print("OLMo-core import OK:", olmo_core.__file__)
PY
# pip install -U "beaker-py<2.0"
# pip install -U ai2-olmo-eval==0.8.5
# pip install transformers==4.57.3 -U
# pip install triton==3.3.0

# ----------- install temporary dependencies - done
git config --global user.name "Tianhua Tao"
git config --global user.email "taotianhua@outlook.com"


# apt remove -y nsight-systems-cli
# apt install -y nsight-systems-2025.5.1

echo "Ready ..."
sleep 7d
