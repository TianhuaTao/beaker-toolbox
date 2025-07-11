
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "cd /workspace/OLMo-core && git stash && git pull" &
done < /workspace/hostfile


while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "cd /workspace/beaker-toolbox && git pull" &
done < /workspace/hostfile

while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.4" &
done < /workspace/hostfile


pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.4




while read -r host; do
    scp -P30255 /workspace/OLMo-core/src/scripts/train/OLMoE3-dev-32l-jul10-gcp.py "$host:/workspace/OLMo-core/src/scripts/train/OLMoE3-dev-32l-jul10-gcp.py"
done < /workspace/hostfile

while read -r host; do
    scp -P30255 /workspace/hostfile "$host:/workspace/hostfile"
done < /workspace/hostfile
