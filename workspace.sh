# GCP multi node
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-nov25

while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "pkill -9 -f \"train/OLMoE3-\"" &
done < /workspace/hostfile
