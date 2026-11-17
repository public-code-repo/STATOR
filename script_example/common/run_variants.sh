set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
mode_set=${MODE_SET:-baselines}
case "$mode_set" in
  baselines)
    forward=(copy demand device device_tuned device_merged cow extent)
    default_cells="same:8:320 cross:8:320 same:4:320 same:1:320 same:8:80" ;;
  ablation)
    forward=(copy extent no_read no_populate writable writable_no_read
             grow_1024 grow_4096 grow_all small_pages)
    default_cells="cross:8:320" ;;
  locality)
    forward=(copy host_copy_huge host_copy extent grow_all small_pages)
    default_cells="same:4:320" ;;
  *) echo "MODE_SET must be baselines, ablation or locality" >&2; exit 2 ;;
esac
if [[ -n "${MODES:-}" ]]; then read -r -a forward <<<"$MODES"; fi
if [[ -n "${CELLS:-}" ]]; then
  cells=$CELLS
elif [[ -n "${PLACEMENTS:-}${CHILD_COUNTS:-}${PARAGRAPHS:-}" ]]; then
  cells=
  for placement in ${PLACEMENTS:-same cross}; do
    for count in ${CHILD_COUNTS:-8}; do
      cells+="$placement:$count:${PARAGRAPHS:-320} "
    done
  done
else
  cells=$default_cells
fi
for cell in $cells; do
  if ! [[ "$cell" =~ ^(same|cross):[1-9][0-9]*:[1-9][0-9]*$ ]]; then
    echo "a cell is PLACEMENT:CHILDREN:PARAGRAPHS, not $cell" >&2
    exit 2
  fi
done
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-900}
model=${MODEL:-"$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${VARIANT_ENGINE:-"$STATOR_ROOT/engines/variants"}
parent_driver=${PARENT_DRIVER:-"$STATOR_ROOT/bin/publisher_variants"}
child_driver=${CHILD_DRIVER:-"$STATOR_ROOT/bin/prefix_agent_variants"}
mount_root=${MOUNT_ROOT:-"$STATOR_ROOT/mounts"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvmech-$mode_set"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$parent_driver" || ! -x "$child_driver" || ! -r "$model" ]]; then
  echo "missing drivers or model; run make publisher_variants prefix_agent_variants" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

huge_dir="$mount_root/kv_huge"
small_dir="$mount_root/kv_small"
mounted_huge=0
mounted_small=0
work=
parent_pid=
cleanup() {
  if [[ -n "$parent_pid" ]]; then kill -9 "$parent_pid" 2>/dev/null || true; fi
  pkill -9 -P $$ 2>/dev/null || true
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  if (( mounted_small )); then sudo -n umount "$small_dir" || true; fi
  rmdir "$huge_dir" "$small_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$small_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
sudo -n mount -t tmpfs -o "huge=never,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$small_dir"
mounted_small=1
work=$(mktemp -d "$result_dir/work.XXXXXX")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\ncells=%s\n' \
    "$gpu_a" "$gpu_b" "$repetitions" "$cells"
  printf 'mode_set=%s\nmodes=%s\n' "$mode_set" "${forward[*]}"
  printf 'n_gen=%s\ngrow_rows=%s\ncase_limit_s=%s\n' \
    "$n_gen" "$grow" "$limit_s"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$parent_driver" "$child_driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
context=
prefix=
paragraphs=
use_prefix() {
  paragraphs=$1
  context=4096
  while (( context < paragraphs * 52 + 512 )); do context=$(( context * 2 )); done
  prefix="$work/prefix.$paragraphs"
  if [[ ! -e "$prefix" ]]; then
    for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph"; done >"$prefix"
  fi
}
text_of() {
  { grep '^TEXT ' "$1" 2>/dev/null || true; } | sha256sum | cut -c1-16
}

run_case() {
  local placement=$1 mode=$2 count=$3 index envs=() gpus=() state="$work/state"
  local file="$huge_dir/kv"
  case $mode in
    copy) ;;
    demand) envs=(STATOR_KV_VMM=2 STATOR_KV_GROW="$grow") ;;
    device) envs=(STATOR_KV_VMM=1 STATOR_KV_GROW="$grow") ;;
    device_tuned) envs=(STATOR_KV_VMM=1 STATOR_KV_VMM_BATCH=1 STATOR_KV_GROW="$grow") ;;
    device_merged) envs=(STATOR_KV_VMM=1 STATOR_KV_VMM_BATCH=2 STATOR_KV_GROW="$grow") ;;
    cow | writable) envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW="$grow" STATOR_KV_COW=1) ;;
    writable_no_read) envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW="$grow" STATOR_KV_COW=2) ;;
    extent) envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW="$grow") ;;
    no_read) envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW="$grow" STATOR_KV_TOUCH=0) ;;
    no_populate) envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW="$grow" STATOR_KV_POPULATE=0) ;;
    grow_1024) envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW=1024) ;;
    grow_4096) envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW=4096) ;;
    grow_all) envs=(STATOR_KV_HOST="$file") ;;
    small_pages) file="$small_dir/kv"; envs=(STATOR_KV_HOST="$file" STATOR_KV_GROW="$grow") ;;
    host_copy) envs=(STATOR_KV_HOST=anon STATOR_KV_GROW="$grow") ;;
    host_copy_huge) envs=(STATOR_KV_HOST=anon) ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  for index in $(seq 0 $((count - 1))); do
    if [[ "$placement" == cross ]] && (( index % 2 == 1 )); then
      gpus+=("$gpu_b")
    else
      gpus+=("$gpu_a")
    fi
  done
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$small_dir/kv.0" "$work/out"
  mkdir -p "$work/out"
  env CUDA_VISIBLE_DEVICES="$gpu_a" STATOR_HOST_PTR=1 "${envs[@]}" \
    "$parent_driver" "$model" "$context" "$prefix" "$state" "$n_gen" "$work/go" \
    "$work/out" "$child_driver" "${gpus[@]}" >"$work/parent" 2>"$work/parent.err" &
  parent_pid=$!
  until grep -q '^PUBLISHED ' "$work/parent" 2>/dev/null; do
    kill -0 "$parent_pid" 2>/dev/null || break
    sleep 0.02
  done
  local before lowest sample failed=0 timed_out=0 started
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s)
  touch "$work/go"
  while kill -0 "$parent_pid" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    if (( $(date +%s) - started > limit_s )); then
      timed_out=1
      pkill -9 -P "$parent_pid" 2>/dev/null || true
      kill -9 "$parent_pid" 2>/dev/null || true
    fi
    sleep 0.05
  done
  {
    printf 'CASE placement=%s mode=%s children=%s paragraphs=%s context=%s\n' \
      "$placement" "$mode" "$count" "$paragraphs" "$context"
    local code=0 result published
    wait "$parent_pid" || code=$?
    parent_pid=
    result=$(grep '^RESULT ' "$work/parent" || true)
    published=$(grep '^PUBLISHED ' "$work/parent" || true)
    if (( code != 0 )) || [[ -z "$result" ]]; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/parent.err" | tail -n 5
    fi
    printf 'PARENT exit=%s text=%s %s %s\n' "$code" "$(text_of "$work/parent")" \
      "${published#PUBLISHED }" \
      "$(sed 's/^RESULT role=parent prefix_tokens=[0-9]* //' <<<"$result")"
    for index in $(seq 0 $((count - 1))); do
      code=$(sed -n "s/^CHILD index=$index exit=\\([0-9]*\\)\$/\\1/p" "$work/parent")
      result=$(grep '^RESULT ' "$work/out/child.$index" 2>/dev/null || true)
      local own=1
      [[ "${gpus[$index]}" == "$gpu_a" ]] || own=0
      if [[ "${code:-1}" != 0 || -z "$result" ]]; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/out/child.$index.err" 2>/dev/null | tail -n 2
      fi
      printf 'CHILD index=%s own_instance=%s exit=%s text=%s %s\n' "$index" "$own" \
        "${code:-none}" "$(text_of "$work/out/child.$index")" "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s timed_out=%s\n' "$failed" "$timed_out"
  } >>"$raw_log"
  rm -rf "$state" "$work/go" "$huge_dir/kv.0" "$small_dir/kv.0" "$work/out"
}

backward=()
for mode in "${forward[@]}"; do backward=("$mode" "${backward[@]}"); done
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=("${forward[@]}"); else modes=("${backward[@]}"); fi
  printf 'BEGIN_KVMECH run=%s\n' "$run" >>"$raw_log"
  for cell in $cells; do
    IFS=: read -r placement count size <<<"$cell"
    use_prefix "$size"
    for mode in "${modes[@]}"; do
      run_case "$placement" "$mode" "$count"
    done
  done
  printf 'END_KVMECH\n' >>"$raw_log"
done

awk -v order="${forward[*]}" -f "$script_dir/summarize_variants.awk" "$raw_log" \
  >"$result_dir/engine_kvmech_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/patches/stator_variants.patch stator_code/drivers/publisher.cpp stator_code/drivers/prefix_agent.cpp script_example/common/summarize_variants.awk \
    script_example/common/run_variants.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvmech_result_dir=%s\n' "$result_dir"
