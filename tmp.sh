bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile2 /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-ablation-16L-A
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile4 /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-ablation-16L-A-small-lr-experts
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile4B /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-ablation-16L-B
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile4C /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-ablation-16L-C
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile8A /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-ablation-16L-A-orth


bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-dev-32l-jul10-gcp
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-dev-32l-jul10-gcp-s1
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-dev-48l-jul10-gcp-s1


kill -9 $(ps aux | grep -e "train/OLMoE3-" -e "nsys" | grep -v grep | awk '{print $2}')

while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep -e "train/OLMoE3-" -e "nsys" | grep -v grep | awk '{print \$2}')" &
done < /workspace/hostfile4


while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "ulimit -n" &
done < /workspace/hostfile

# STEP=103000
# gcloud storage mv gs://ai2-llm/checkpoints/OLMo3-moe-integrationtest-5-32L-lbl-fix/OLMo3-moe-integrationtest-5-32L-lbl-fix_2048d_32L2560M2560S_64E4K_dev/step${STEP}  gs://ai2-llm/checkpoints/OLMo3-moe-integrationtest-5-32L-lbl-fix-decay2/OLMo3-moe-integrationtest-5-32L-lbl-fix_2048d_32L2560M2560S_64E4K_dev/step${STEP}

