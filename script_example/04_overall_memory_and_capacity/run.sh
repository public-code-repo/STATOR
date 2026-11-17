set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/04_overall_memory_and_capacity
C=$REPO_ROOT/script_example/common
D=${RUN_DATE:-$(date +%Y%m%d)}
R=$STATOR_ROOT/results
QWEN7B=$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
QWEN14B=$STATOR_ROOT/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
M=$STATOR_ROOT/mounts_model
unmount() { rm -f "$M/model.gguf"; sudo -n umount "$M" 2>/dev/null || true; rmdir "$M" 2>/dev/null || true; }
memfs() {
  mkdir -p "$M"
  sudo -n mount -t tmpfs -o "huge=always,size=$1,uid=$(id -u),gid=$(id -g)" tmpfs "$M"
  dd if="$QWEN7B" of="$M/model.gguf" bs=64M iflag=direct status=none
  cmp -s "$QWEN7B" "$M/model.gguf"
}
trap unmount EXIT
RESULT_TAG=$D-stack bash "$S/run_stack.sh"
RESERVE_GIB=16 NONE_COUNTS="16" KV_COUNTS="19" WEIGHTS_COUNTS="61" BOTH_COUNTS="136 140" \
  ESTIMATE_NONE_MIB=6200 ESTIMATE_KV_MIB=5300 ESTIMATE_WEIGHTS_MIB=1650 ESTIMATE_BOTH_MIB=730 \
  RESULT_TAG=$D-stack-capacity bash "$S/run_stack.sh"
MODEL=$QWEN14B RESERVE_GIB=16 NONE_COUNTS="1 4" KV_COUNTS=" " WEIGHTS_COUNTS=" " BOTH_COUNTS="1 4 8" \
  ESTIMATE_NONE_MIB=14000 ESTIMATE_BOTH_MIB=1800 RESULT_TAG=$D-stack-qwen14b bash "$S/run_stack.sh"
MODEL=$QWEN14B RESERVE_GIB=16 NONE_COUNTS="8" KV_COUNTS=" " WEIGHTS_COUNTS=" " BOTH_COUNTS=" " \
  ESTIMATE_NONE_MIB=12600 RESULT_TAG=$D-stack-qwen14b-none8 bash "$S/run_stack.sh"
MODEL=$QWEN14B RESERVE_GIB=16 NONE_COUNTS=" " KV_COUNTS=" " WEIGHTS_COUNTS=" " BOTH_COUNTS="118" \
  ESTIMATE_BOTH_MIB=800 RESULT_TAG=$D-stack-capacity-qwen14b bash "$S/run_stack.sh"
memfs 6144m
MODEL=$M/model.gguf NONE_COUNTS="1 8" KV_COUNTS="" WEIGHTS_COUNTS="1 8" BOTH_COUNTS="1 8 32" \
  RESULT_TAG=$D-stack-huge bash "$S/run_stack.sh"
MODEL=$M/model.gguf RESERVE_GIB=16 NONE_COUNTS="16" KV_COUNTS="19" WEIGHTS_COUNTS="61" BOTH_COUNTS="136 140" \
  ESTIMATE_NONE_MIB=6200 ESTIMATE_KV_MIB=5300 ESTIMATE_WEIGHTS_MIB=1650 ESTIMATE_BOTH_MIB=730 \
  RESULT_TAG=$D-stack-capacity-huge bash "$S/run_stack.sh"
unmount
memfs 5120m
python3 "$S/vmstat_sampler.py" "$R/$D-resident.vmstat.jsonl" &
sampler=$!
python3 -u "$S/run_resident.py" --tag $D-resident --model "$M/model.gguf" --scenario resident --repetitions 6 \
  --modes 'copy extent' --counts '8 16 32 48 56 64 96 128' --active 8 --context 16384 --tokens 64 --reserve-gib 4
kill "$sampler"
wait "$sampler" || true
python3 "$C/check_scenarios.py" "$R/$D-resident"
