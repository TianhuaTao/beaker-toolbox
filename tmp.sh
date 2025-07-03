bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-dev-jul3-gcp




while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep -e "train/OLMoE3-" -e "nsys" | grep -v grep | awk '{print \$2}')" &
done < /workspace/hostfile