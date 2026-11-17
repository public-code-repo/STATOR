set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
sizes=${SIZES_MIB:-"1024"}
strides=${STRIDES_KIB:-"4 64"}
probe="$STATOR_ROOT/bin/tlb_probe"
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-tlb-probe"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

[[ -x "$probe" ]] || { echo "missing $probe; run make tlb_probe" >&2; exit 2; }
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
mkdir -p "$result_dir"
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\nsizes_mib=%s\nstrides_kib=%s\n' \
    "$gpu_a" "$gpu_b" "$repetitions" "$sizes" "$strides"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    gpus=("$gpu_a" "$gpu_b") kinds=(device host_huge host_small)
  else
    gpus=("$gpu_b" "$gpu_a") kinds=(host_small host_huge device)
  fi
  for gpu in "${gpus[@]}"; do
    for size in $sizes; do
      for stride in $strides; do
        for kind in "${kinds[@]}"; do
          line=$(CUDA_VISIBLE_DEVICES="$gpu" "$probe" "$kind" "$size" "$stride" 2>&1 || true)
          printf 'PROBE run=%s gpu=%s %s\n' "$run" "${gpu:4:8}" "${line#RESULT }" >>"$raw_log"
        done
      done
    done
  done
done

awk -f "$script_dir/summarize_tlb_probe.awk" "$raw_log" >"$result_dir/tlb_probe_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/probes/tlb_probe.cu script_example/02_in_place_extents/summarize_tlb_probe.awk script_example/02_in_place_extents/run_tlb_probe.sh
) >"$result_dir/source_hashes.txt"
printf 'tlb_probe_result_dir=%s\n' "$result_dir"
