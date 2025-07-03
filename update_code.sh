
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "cd /workspace/OLMo-core && git pull" &
done < /workspace/hostfile


while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.4" &
done < /workspace/hostfile


pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.4
