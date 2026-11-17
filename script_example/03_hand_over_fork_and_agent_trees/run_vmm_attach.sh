set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu=${STATOR_GPU_A:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
total=${TOTAL_MIB:-896}
granules=${GRANULES_MIB:-"2 4 8 16 32 64 128 448 896"}
probe="$STATOR_ROOT/bin/vmm_attach_probe"
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-vmm-attach"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

[[ -x "$probe" ]] || { echo "missing $probe; run make vmm_attach_probe" >&2; exit 2; }
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu=%s\nrepetitions=%s\ntotal_mib=%s\ngranules_mib=%s\n' "$gpu" \
    "$repetitions" "$total" "$granules"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

for run in $(seq 1 "$repetitions"); do
  order=$granules
  if (( run % 2 == 0 )); then order=$(tr ' ' '\n' <<<"$granules" | tac | tr '\n' ' '); fi
  for granule in $order; do
    line=$(CUDA_VISIBLE_DEVICES="$gpu" "$probe" "$total" "$granule" \
      2>"$result_dir/probe.err" || true)
    if [[ "$line" == RESULT* ]]; then
      printf 'PROBE run=%s %s\n' "$run" "${line#RESULT }"
    else
      printf 'FAILED run=%s granule_mib=%s %s\n' "$run" "$granule" \
        "$(tail -n 1 "$result_dir/probe.err" | tr ' ' '_')"
    fi >>"$raw_log"
  done
done
rm -f "$result_dir/probe.err"

awk -f "$script_dir/summarize_vmm_attach.awk" "$raw_log" \
  >"$result_dir/vmm_attach_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/probes/vmm_attach_probe.cu script_example/03_hand_over_fork_and_agent_trees/summarize_vmm_attach.awk script_example/03_hand_over_fork_and_agent_trees/run_vmm_attach.sh
) >"$result_dir/source_hashes.txt"
printf 'vmm_attach_result_dir=%s\n' "$result_dir"
