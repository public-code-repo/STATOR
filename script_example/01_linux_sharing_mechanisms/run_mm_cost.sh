set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
producer_gpu=${STATOR_GPU_A:-$STATOR_GPU}
consumer_gpu=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
mib=${MIB:-1024}
page_mib=${PAGE_MIB:-128}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-mm-tax"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

for value in "$repetitions" "$mib" "$page_mib"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "parameters must be positive integers" >&2
    exit 2
  }
done
(( repetitions % 2 == 0 )) || {
  echo "REPETITIONS must be even so that both orders are used" >&2
  exit 2
}
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'producer_gpu=%s\nconsumer_gpu=%s\n' "$producer_gpu" "$consumer_gpu"
  printf 'repetitions=%s\nmib=%s\npage_mib=%s\n' "$repetitions" "$mib" "$page_mib"
  printf 'thp_enabled=%s\n' "$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$STATOR_ROOT/bin/mm_cost_probe" "$STATOR_ROOT/bin/fork_probe"
} >"$result_dir/metadata.txt"

cases=(
  cow_break:"$mib":1 cow_break:"$mib":2 cow_break:"$mib":4 cow_break:"$mib":8
  cow_break:"$mib":12
  reuse_fault:"$mib":1 reuse_fault:"$mib":12
  fork:"$mib":1 fork_thp:"$mib":1
  mprotect_page:"$page_mib":1 mprotect_range:"$mib":1
  dontneed_page:"$page_mib":1 dontneed_range:"$mib":1
  munmap_range:"$mib":1 mremap_range:"$mib":1
  thp_split:"$mib":1 thp_collapse:"$mib":1
)
gpus=("$producer_gpu" "$consumer_gpu")
for run in $(seq 1 "$repetitions"); do
  gpu=${gpus[$((run % 2))]}
  if (( run % 2 == 1 )); then phases=(no_cuda cuda); else phases=(cuda no_cuda); fi
  for spec in "${cases[@]}"; do
    IFS=: read -r operation size threads <<<"$spec"
    for phase in "${phases[@]}"; do
      printf 'BEGIN_MM_COST run=%s gpu=%s\n' "$run" "${gpu:4:8}" >>"$raw_log"
      status=0
      CUDA_VISIBLE_DEVICES=$gpu "$STATOR_ROOT/bin/mm_cost_probe" "$phase" \
        "$operation" "$size" "$threads" >>"$raw_log" 2>&1 || status=$?
      if (( status != 0 )); then
        printf 'PROBE_FAILED phase=%s operation=%s exit_status=%s\n' \
          "$phase" "$operation" "$status" >>"$raw_log"
      fi
    done
  done
done

for run in $(seq 1 "$repetitions"); do
  gpu=${gpus[$((run % 2))]}
  guards=(fork dontfork posix_spawn repair)
  for offset in 0 1 2 3; do
    mode=${guards[$(((run + offset) % 4))]}
    printf 'BEGIN_FORK_PROBE run=%s gpu=%s\n' "$run" "${gpu:4:8}" >>"$raw_log"
    status=0
    CUDA_VISIBLE_DEVICES=$gpu "$STATOR_ROOT/bin/fork_probe" "$mode" "$mib" \
      >>"$raw_log" 2>&1 || status=$?
    if (( status != 0 )); then
      printf 'PROBE_FAILED phase=cuda operation=fork_probe_%s exit_status=%s\n' \
        "$mode" "$status" >>"$raw_log"
    fi
  done
done

awk -f "$script_dir/summarize_mm_cost.awk" "$raw_log" \
  >"$result_dir/mm_cost_summary.csv"
awk '
  function field(name, i, pair) {
    for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
    return ""
  }
  /^probe=fork_probe / {
    mode = field("mode")
    count[mode]++
    helper[mode] += field("helper_ms")
    first[mode] += field("first_gpu_write_after_helper_ms")
    steady[mode] += field("steady_gpu_write_ms")
    if (field("ok") == 1) ok[mode]++
  }
  END {
    print "mode,runs,ok_runs,steady_gpu_write_ms,helper_ms,first_gpu_write_after_helper_ms"
    split("fork dontfork posix_spawn repair", order, " ")
    for (i = 1; i <= 4; ++i) {
      mode = order[i]
      if (!(mode in count)) continue
      printf "%s,%d,%d,%.3f,%.3f,%.3f\n", mode, count[mode], ok[mode] + 0, \
             steady[mode] / count[mode], helper[mode] / count[mode], \
             first[mode] / count[mode]
    }
  }' "$raw_log" >"$result_dir/fork_probe_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/probes/mm_cost_probe.cu stator_code/probes/fork_probe.cu stator_code/probes/extent_cow.cpp stator_code/probes/extent_cow.hpp \
    script_example/01_linux_sharing_mechanisms/summarize_mm_cost.awk script_example/01_linux_sharing_mechanisms/run_mm_cost.sh
) >"$result_dir/source_hashes.txt"
printf 'mm_cost_result_dir=%s\n' "$result_dir"
