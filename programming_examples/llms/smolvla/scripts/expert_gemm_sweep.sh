#!/usr/bin/env bash
# SmolVLA action-expert GEMM sweep on real NPU2 (bf16-in/bf16-out, drain).
#
# Every config is a row: label,M,K,N,TILE_M,TILE_K_L2,TILE_K_L1,TILE_N,HERD_M,HERD_N
# Results are appended to $CSV incrementally (resume-safe: a config already in the
# CSV is skipped). Each NPU run is wrapped in flock (shared single device).
#
# Usage:
#   bash expert_gemm_sweep.sh <configs_file> <csv_out>

set -u
CFG="${1:?configs file}"
CSV="${2:?csv out}"
GEMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../matrix_multiplication/bf16_in_bf16_out" && pwd)"
LOGDIR="$(dirname "$CSV")/logs"
mkdir -p "$LOGDIR"

if [ ! -f "$CSV" ]; then
  echo "label,M,K,N,tile_m,tile_k_l2,tile_k_l1,tile_n,herd_m,herd_n,status,latency_us,gflops,mean_rel_L1" > "$CSV"
fi

while IFS=, read -r label M K N TM TK2 TK1 TN HM HN; do
  case "$label" in ''|\#*) continue;; esac
  key="$label,$M,$K,$N,$TM,$TK2,$TK1,$TN,$HM,$HN,"
  if grep -q "^$key" "$CSV"; then
    echo "[skip] $key (already in CSV)"; continue
  fi
  log="$LOGDIR/${label}_${M}x${K}x${N}_tm${TM}_tk2${TK2}_tk1${TK1}_tn${TN}_hm${HM}_hn${HN}.log"
  echo "[run ] $label ${M}x${K}x${N} tm=$TM tk2=$TK2 tk1=$TK1 tn=$TN hm=$HM hn=$HN"
  start=$(date +%s)
  flock -x -w 1800 /tmp/mlir-air-npu.lock \
    make -C "$GEMM_DIR" run \
      M=$M K=$K N=$N TILE_M=$TM TILE_K_L2=$TK2 TILE_K_L1=$TK1 TILE_N=$TN \
      HERD_M=$HM HERD_N=$HN METHOD=drain HIGH_PRECISION=true \
      AIE_TARGET=aie2p PERF_ITERS=20 > "$log" 2>&1
  rc=$?
  el=$(( $(date +%s) - start ))

  lat=$(grep -oP 'Latency \(us\):\s*\K[\d.]+' "$log" | tail -1)
  gfl=$(grep -oP 'GFLOP/s' -B0 "$log" >/dev/null 2>&1; grep -oP 'Throughput:\s*\K[\d.eE+-]+' "$log" | tail -1)
  rel=$(grep -oP 'mean_rel_L1[= ]*\K[\d.eE+-]+' "$log" | tail -1)
  [ -z "${lat:-}" ] && lat=""
  [ -z "${gfl:-}" ] && gfl=""
  [ -z "${rel:-}" ] && rel=""

  if grep -q "PASS!" "$log"; then status=PASS
  elif [ $rc -ne 0 ]; then
    if grep -qi "Mismatches:" "$log"; then status=NUMFAIL; else status=BUILDFAIL; fi
  else status=UNKNOWN; fi

  echo "$key$status,$lat,$gfl,$rel" >> "$CSV"
  echo "       -> $status lat=${lat:-?}us gflops=${gfl:-?} rel=${rel:-?}  (${el}s)"
done < "$CFG"

echo "=== done: $CSV ==="
column -s, -t < "$CSV"
