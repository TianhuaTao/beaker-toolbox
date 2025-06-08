
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "cd /workspace/OLMo-core && git pull" &
done < /workspace/hostfile