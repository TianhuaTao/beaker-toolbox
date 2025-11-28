# GCP multi node
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-nov25

while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "pkill -9 -f \"train/OLMoE3-\"" &
done < /workspace/hostfile


pkill -9 -f "train/OLMoE3-"



export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
torchrun --nproc-per-node=8 OLMo-core/src/scripts/train/OLMoE3-nov25.py train OLMoE3-nov25 ai2/augusta

nsys profile -t nvtx,cuda --capture-range=cudaProfilerApi --force-overwrite=true --output OLMoE3-nov25.nsys-rep torchrun --nproc-per-node=8 OLMo-core/src/scripts/train/OLMoE3-nov25.py train OLMoE3-nov25 ai2/augusta