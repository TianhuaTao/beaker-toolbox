#!/usr/bin/env bash
#
# Scan Beaker nodes for down 800G InfiniBand HCAs.
#
# Usage:
#   /workspace/beaker-toolbox/scan_ib_health.sh [hostfile]
#   /workspace/beaker-toolbox/scan_ib_health.sh /workspace/hostfile32 --bad-only
#   /workspace/beaker-toolbox/scan_ib_health.sh /workspace/hostfile44 --good-only
#
# Environment overrides:
#   SSH_PORT=30255
#   SSH_JOBS=16
#   SSH_TIMEOUT=6
#   HCAS="mlx5_0 mlx5_1 mlx5_4 mlx5_5 mlx5_6 mlx5_11 mlx5_14 mlx5_15"

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
      sed -n '2,14p' "$0"
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
HCAS="${HCAS:-mlx5_0 mlx5_1 mlx5_4 mlx5_5 mlx5_6 mlx5_11 mlx5_14 mlx5_15}"

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
    "$host" "HCAS=$(printf '%q' "$HCAS") bash -lc '
      bad=\"\"
      for d in \$HCAS; do
        if [[ ! -e /sys/class/infiniband/\$d ]]; then
          bad=\"\$bad \$d:MISSING\"
          continue
        fi

        state=\$(cat /sys/class/infiniband/\$d/ports/1/state 2>/dev/null | tr \" \" \"_\" || true)
        phys=\$(cat /sys/class/infiniband/\$d/ports/1/phys_state 2>/dev/null | tr \" \" \"_\" || true)
        rate=\$(cat /sys/class/infiniband/\$d/ports/1/rate 2>/dev/null | tr \" \" \"_\" || true)
        pci=\$(basename \"\$(readlink -f /sys/class/infiniband/\$d/device 2>/dev/null)\" 2>/dev/null || true)
        xmit=\$(cat /sys/class/infiniband/\$d/ports/1/counters/port_xmit_discards 2>/dev/null || true)

        if [[ \"\$state\" != \"4:_ACTIVE\" || \"\$phys\" != \"5:_LinkUp\" || \"\$rate\" != \"800_Gb/sec_(4X_XDR)\" ]]; then
          bad=\"\$bad \$d@\$pci:\$state:\$phys:\$rate:xmit_discards=\$xmit\"
        fi
      done

      gpu=\$(nvidia-smi -L 2>/dev/null | wc -l | tr -d \" \")
      ecc=0
      while IFS= read -r v; do
        [[ \"\$v\" =~ ^[0-9]+$ ]] || v=0
        ecc=\$((ecc + v))
      done < <(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null)

      if [[ -z \"\$bad\" ]]; then
        echo \"\$(hostname -s) OK gpu=\$gpu ecc_uncorr=\$ecc\"
      else
        echo \"\$(hostname -s) BAD gpu=\$gpu ecc_uncorr=\$ecc bad_hca:\$bad\"
      fi
    '" >"$outfile" 2>&1 || echo "${host%%.*} SSH_FAIL" >"$outfile"
}

echo "HOSTFILE: $HOSTFILE"
echo "HCAS: $HCAS"
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
