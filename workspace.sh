# GCP multi node
bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-dec12

bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/node_cmd-ablation-train-general.sh OLMoE3-dec12-reprod

bash /workspace/beaker-toolbox/run_script_remotely.sh /workspace/hostfile /workspace/beaker-toolbox/torchrun_test.sh


while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "pkill -9 -f \"train/OLMoE3-\"" &
done < /workspace/hostfile


while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "nvidia-smi -L" &
done < /workspace/hostfile

pkill -9 -f "train/OLMoE3-"


while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "ls /workspace/OLMo-core/input_ids*" &
done < /workspace/hostfile

export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3
torchrun --nproc-per-node=8 OLMo-core/src/scripts/train/OLMoE3-nov25.py train OLMoE3-nov25 ai2/augusta

nsys profile -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output OLMoE3-nov25.nsys-rep torchrun --nproc-per-node=8 OLMo-core/src/scripts/train/OLMoE3-nov25.py train OLMoE3-nov25 ai2/augusta


scp -P30255 root@augusta-gcp-299:/workspace/prof_5_OLMoE3-nov25.nsys-rep ./


torchrun --rdzv_endpoint augusta-gcp-280:29500 --rdzv_id 33333 --rdzv_backend c10d --nnodes 64 --nproc-per-node 8 --node_rank 55 ./src/scripts/train/OLMoE3-dec10.py train OLMoE3-dec10 ai2/augusta


sudo apt install bwm-ng

# gcloud
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
