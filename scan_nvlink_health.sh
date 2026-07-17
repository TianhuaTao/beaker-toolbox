#!/usr/bin/env bash
#
# Scan Beaker nodes for missing or inactive NVLinks.
#
# Usage:
#   /workspace/beaker-toolbox/scan_nvlink_health.sh [hostfile]
#   /workspace/beaker-toolbox/scan_nvlink_health.sh /workspace/hostfile32 --bad-only
#   /workspace/beaker-toolbox/scan_nvlink_health.sh /workspace/hostfile64 --good-only
#
# Environment overrides:
#   SSH_PORT=30255
#   SSH_JOBS=16
#   SSH_TIMEOUT=6
#   NVLINK_TIMEOUT=12
#   EXPECTED_GPUS=8
#   EXPECTED_LINKS_PER_GPU=18

set -u

WORKSPACE_DIR="/workspace"
HOSTFILE="${WORKSPACE_DIR}/hostfile"
BAD_ONLY=0
GOOD_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bad-only)
      BAD_ONLY=1
      shift
      ;;
    --good-only)
      GOOD_ONLY=1
      shift
      ;;
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *)
      if [[ "$HOSTFILE" == "${WORKSPACE_DIR}/hostfile" ]]; then
        HOSTFILE="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ "$BAD_ONLY" -eq 1 && "$GOOD_ONLY" -eq 1 ]]; then
  echo "Error: use only one of --bad-only or --good-only." >&2
  exit 2
fi

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Error: hostfile '$HOSTFILE' not found." >&2
  exit 1
fi

SSH_PORT="${SSH_PORT:-30255}"
SSH_JOBS="${SSH_JOBS:-16}"
SSH_TIMEOUT="${SSH_TIMEOUT:-6}"
NVLINK_TIMEOUT="${NVLINK_TIMEOUT:-12}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
EXPECTED_LINKS_PER_GPU="${EXPECTED_LINKS_PER_GPU:-18}"

for value_name in SSH_JOBS SSH_TIMEOUT NVLINK_TIMEOUT EXPECTED_GPUS EXPECTED_LINKS_PER_GPU; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: $value_name must be a positive integer, got '$value'." >&2
    exit 2
  fi
done

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

scan_host() {
  local index="$1"
  local host="$2"
  local outfile="${TMPDIR}/${index}.out"

  ssh -n \
    -p "$SSH_PORT" \
    -o BatchMode=yes \
    -o ConnectTimeout="$SSH_TIMEOUT" \
    -o StrictHostKeyChecking=no \
    "$host" "EXPECTED_GPUS=$(printf '%q' "$EXPECTED_GPUS") EXPECTED_LINKS_PER_GPU=$(printf '%q' "$EXPECTED_LINKS_PER_GPU") NVLINK_TIMEOUT=$(printf '%q' "$NVLINK_TIMEOUT") bash -lc '
      gpu=\$(nvidia-smi -L 2>/dev/null | wc -l | tr -d \" \")
      [[ \"\$gpu\" =~ ^[0-9]+$ ]] || gpu=0

      bad=\"\"
      total_active=0
      expected_total=\$((EXPECTED_GPUS * EXPECTED_LINKS_PER_GPU))

      if [[ \"\$gpu\" -ne \"\$EXPECTED_GPUS\" ]]; then
        bad=\"\$bad gpu_count=\$gpu/\$EXPECTED_GPUS\"
      fi

      for ((gpu_index = 0; gpu_index < gpu; gpu_index++)); do
        status=\$(timeout \"\$NVLINK_TIMEOUT\" nvidia-smi nvlink -s -i \"\$gpu_index\" 2>&1)
        status_rc=\$?
        active=\$(printf \"%s\n\" \"\$status\" | grep -cE \"^[[:space:]]*Link [0-9]+:.*GB/s\" || true)
        total_active=\$((total_active + active))

        if [[ \"\$status_rc\" -ne 0 || \"\$active\" -ne \"\$EXPECTED_LINKS_PER_GPU\" ]]; then
          if [[ \"\$status_rc\" -eq 124 ]]; then
            detail=\"TIMEOUT\"
          else
            detail=\$(printf \"%s\n\" \"\$status\" | grep -Ei \"NVML:|inactive|error|failed|unable\" | tail -n 1 | tr \" \" \"_\" || true)
            [[ -n \"\$detail\" ]] || detail=\"status_rc=\$status_rc\"
          fi
          bad=\"\$bad gpu\$gpu_index:active=\$active/\$EXPECTED_LINKS_PER_GPU:\$detail\"
        fi
      done

      ecc=0
      while IFS= read -r value; do
        [[ \"\$value\" =~ ^[0-9]+$ ]] || value=0
        ecc=\$((ecc + value))
      done < <(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null)

      fabric_bad=0
      while IFS=, read -r state fabric_status; do
        state=\$(printf \"%s\" \"\$state\" | xargs)
        fabric_status=\$(printf \"%s\" \"\$fabric_status\" | xargs)
        if [[ \"\$state\" != \"Completed\" || \"\$fabric_status\" != \"Success\" ]]; then
          fabric_bad=\$((fabric_bad + 1))
        fi
      done < <(nvidia-smi --query-gpu=fabric.state,fabric.status --format=csv,noheader 2>/dev/null)

      if [[ -z \"\$bad\" && \"\$fabric_bad\" -eq 0 ]]; then
        echo \"\$(hostname -s) OK gpu=\$gpu ecc_uncorr=\$ecc nvlink_active=\$total_active/\$expected_total fabric_bad=\$fabric_bad\"
      else
        [[ \"\$fabric_bad\" -eq 0 ]] || bad=\"\$bad fabric_bad=\$fabric_bad\"
        echo \"\$(hostname -s) BAD gpu=\$gpu ecc_uncorr=\$ecc nvlink_active=\$total_active/\$expected_total bad_nvlink:\$bad\"
      fi
    '" >"$outfile" 2>&1 || echo "${host%%.*} SSH_FAIL" >"$outfile"
}

echo "HOSTFILE: $HOSTFILE"
echo "EXPECTED_GPUS: $EXPECTED_GPUS"
echo "EXPECTED_LINKS_PER_GPU: $EXPECTED_LINKS_PER_GPU"
echo

index=0
valid_indexes=()
while IFS= read -r host || [[ -n "$host" ]]; do
  [[ -z "$host" || "$host" =~ ^# ]] && continue

  valid_indexes+=("$index")
  scan_host "$index" "$host" &

  while [[ "$(jobs -rp | wc -l)" -ge "$SSH_JOBS" ]]; do
    sleep 0.2
  done

  ((index++))
done < "$HOSTFILE"

wait

bad_count=0
for i in "${valid_indexes[@]}"; do
  line=$(cat "${TMPDIR}/${i}.out")
  if [[ "$line" == *" BAD "* || "$line" == *"SSH_FAIL"* ]]; then
    ((bad_count++))
    if [[ "$GOOD_ONLY" -eq 0 ]]; then
      echo "$line"
    fi
  elif [[ "$BAD_ONLY" -eq 0 ]]; then
    echo "$line"
  fi
done

echo
echo "bad_or_unreachable=${bad_count} total=${#valid_indexes[@]}"
