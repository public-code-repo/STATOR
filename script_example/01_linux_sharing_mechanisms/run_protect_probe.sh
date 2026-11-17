set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
size=${SIZE_MIB:-64}
probe="$STATOR_ROOT/bin/protect_probe"
mount_root=${MOUNT_ROOT:-"$STATOR_ROOT/mounts"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-protect"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

[[ -x "$probe" ]] || { echo "missing $probe; run make protect_probe" >&2; exit 2; }
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

huge_dir="$mount_root/kv_huge"
mounted_huge=0
cleanup() {
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=1024m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\nsize_mib=%s\n' "$gpu_a" "$gpu_b" \
    "$repetitions" "$size"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$probe"
} >"$result_dir/metadata.txt"

file="$huge_dir/pattern"
python3 -c "
import sys
word = (0x5eedf00d).to_bytes(4, 'little')
with open(sys.argv[1], 'wb') as handle:
    for _ in range(int(sys.argv[2])):
        handle.write(word * (1 << 18))
" "$file" "$size"
chmod 0400 "$file"

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then first=$gpu_a second=$gpu_b; else first=$gpu_b second=$gpu_a; fi
  {
    for gpu in "$first" "$second"; do
      line=$(CUDA_VISIBLE_DEVICES="$gpu" "$probe" host "$file" "$size" 2>/dev/null || true)
      printf 'PROBE run=%s gpu=%s %s\n' "$run" "${gpu:4:8}" "${line#RESULT }"
    done
    line=$(CUDA_VISIBLE_DEVICES="$first" "$probe" vmm "$size" 2>/dev/null || true)
    printf 'PROBE run=%s gpu=%s %s\n' "$run" "${first:4:8}" "${line#RESULT }"
  } >>"$raw_log"
done

awk -f "$script_dir/summarize_protect_probe.awk" "$raw_log" \
  >"$result_dir/protect_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/probes/protect_probe.cu script_example/01_linux_sharing_mechanisms/summarize_protect_probe.awk script_example/01_linux_sharing_mechanisms/run_protect_probe.sh
) >"$result_dir/source_hashes.txt"
printf 'protect_result_dir=%s\n' "$result_dir"
