set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${GPU_UUID:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
size_mib=${SIZE_MIB:-896}
probe=${PROBE:-"$STATOR_ROOT/bin/populate_probe"}
anchor_probe=${ANCHOR:-"$STATOR_ROOT/bin/linux_share_probe"}
mount_root=${MOUNT_ROOT:-"$STATOR_ROOT/mounts"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-gpu-faultaround-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
fab_file=/sys/kernel/debug/fault_around_bytes
[[ -x "$probe" && -x "$anchor_probe" ]] || { echo "missing programs; run make populate_probe linux_share_probe" >&2; exit 2; }
sudo -n true 2>/dev/null || { echo "passwordless sudo is required" >&2; exit 2; }
[[ -e "$result_dir" ]] && { echo "result directory already exists: $result_dir" >&2; exit 2; }
default_fab=$(sudo -n cat "$fab_file")

fault_counters() {
  local pages=0 ticks=0 stat line rest
  for stat in /proc/[0-9]*/stat; do
    { read -r line <"$stat"; } 2>/dev/null || continue
    [[ "$line" == *"(UVM GPU"*" BH)"* ]] || continue
    rest=${line##*) }
    set -- $rest
    pages=$(( pages + $8 + ${10} ))
    ticks=$(( ticks + ${12} + ${13} ))
  done
  printf '%s %s' "$pages" "$ticks"
}
tick_ms=$(( 1000 / $(getconf CLK_TCK) ))
ext4_dir="$mount_root/faultaround_ext4"
anchor_pid=
cleanup() {
  echo "$default_fab" | sudo -n tee "$fab_file" >/dev/null || true
  if [[ -n "$anchor_pid" ]]; then kill "$anchor_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
  rm -rf "${ext4_dir:?}"
  rmdir "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$ext4_dir" "$result_dir"
head -c $(( size_mib * 1024 * 1024 )) /dev/urandom >"$ext4_dir/data"
{
  printf 'timestamp=%s\ngpu=%s\nrepetitions=%s\nsize_mib=%s\ndefault_fault_around_bytes=%s\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$repetitions" "$size_mib" "$default_fab"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

export CUDA_VISIBLE_DEVICES="$gpu"
"$anchor_probe" hold 1000000 >/dev/null 2>&1 &
anchor_pid=$!
sleep 3
run_case() {
  local label=$1 before after out
  shift
  before=$(fault_counters)
  out=$(timeout -s KILL 300 "$probe" "$@" 2>&1 || echo "PREPARE failed=1")
  after=$(fault_counters)
  read -r -a b <<<"$before"
  read -r -a a <<<"$after"
  printf 'CASE %s\n%s\nKERNEL gpu_fault_pages=%s fault_thread_ms=%s\n' "$label" \
    "$(grep -E '^(PREPARE|ERROR)' <<<"$out" || echo 'PREPARE failed=1')" \
    "$(( a[0] - b[0] ))" "$(( (a[1] - b[1]) * tick_ms ))" >>"$raw_log"
}
for run in $(seq 1 "$repetitions"); do
  sizes=("$default_fab" 4096)
  steps=(none touch populate)
  if (( run % 2 == 0 )); then sizes=(4096 "$default_fab"); steps=(populate touch none); fi
  printf 'BEGIN_FAULTAROUND run=%s\n' "$run" >>"$raw_log"
  for fab in "${sizes[@]}"; do
    echo "$fab" | sudo -n tee "$fab_file" >/dev/null
    for step in "${steps[@]}"; do
      cat "$ext4_dir/data" >/dev/null
      run_case "mode=ro fs=ext4 fab=$fab step=$step" ro "$ext4_dir/data" "$size_mib" "$step"
    done
  done
  printf 'END_FAULTAROUND\n' >>"$raw_log"
done
echo "$default_fab" | sudo -n tee "$fab_file" >/dev/null
python3 "$script_dir/summarize_fault_around_probe.py" "$raw_log" >"$result_dir/fault_around_probe_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/probes/populate_probe.cu script_example/01_linux_sharing_mechanisms/run_fault_around_probe.sh script_example/01_linux_sharing_mechanisms/summarize_fault_around_probe.py
) >"$result_dir/source_hashes.txt"
printf 'fault_around_probe_result_dir=%s\n' "$result_dir"
