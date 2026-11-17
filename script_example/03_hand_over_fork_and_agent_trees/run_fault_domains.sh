set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
agents=${AGENTS:-8}
paragraphs=${PARAGRAPHS:-320}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
model=${MODEL:-"$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$STATOR_ROOT/engines/extents"}
driver=${DRIVER:-"$STATOR_ROOT/bin/batch_server_extents"}
mount_root=${MOUNT_ROOT:-"$STATOR_ROOT/mounts"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvbatchfault"}
fault_lib=${FAULT_LIB:-"$STATOR_ROOT/bin/fault_inject.so"}
fault_at_ms=${FAULT_AT_MS:-15000}
fault_after_b_ms=${FAULT_AFTER_B_MS:-1500}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if (( agents % 2 != 0 )); then
  echo "AGENTS must be even" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -r "$model" || ! -r "$fault_lib" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
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
mounted_huge=0
work=
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=4096m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
work=$(mktemp -d "$result_dir/work.XXXXXX")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\nagents=%s\n' \
    "$gpu_a" "$gpu_b" "$repetitions" "$agents"
  printf 'paragraphs=%s\nn_gen=%s\ngrow_rows=%s\n' "$paragraphs" "$n_gen" "$grow"
  printf 'fault_at_ms=%s\nfault_after_b_ms=%s\n' "$fault_at_ms" "$fault_after_b_ms"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$fault_lib" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
half=$(( agents / 2 ))
context=4096
while (( context < paragraphs * 52 + 512 + agents * (n_gen + 32) )); do
  context=$(( context * 2 ))
done
prefix="$work/prefix"
: >"$prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph" >>"$prefix"; done

server() {
  local name=$1 gpu=$2
  shift 2
  local envs=()
  while [[ "$1" != -- ]]; do envs+=("$1"); shift; done
  shift
  env CUDA_VISIBLE_DEVICES="$gpu" STATOR_HOST_PTR=1 "${envs[@]}" \
    "$driver" "$@" >"$work/server.$name" 2>"$work/server.$name.err" &
  pids+=($!)
}

run_case() {
  local config=$1 fault=$2 before lowest sample start second_start=0 tokens failed=0
  local trigger="$work/fault.trigger" injected=0 elapsed
  local fault_env=()
  rm -f "$trigger"
  if [[ "$fault" == write ]]; then fault_env=(LD_PRELOAD="$fault_lib" FAULT_TRIGGER="$trigger"); fi
  pids=()
  rm -f "$work"/server.* "$work"/state.* "$huge_dir/kv.0"
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  start=$(date +%s%N)
  case $config in
    one_server)
      server a "$gpu_a" "${fault_env[@]}" -- alone "$model" "$context" "$prefix" "$agents" 0 "$n_gen" ;;
    two_servers_compute)
      server a "$gpu_a" -- alone "$model" "$context" "$prefix" "$half" 0 "$n_gen"
      server b "$gpu_b" -- alone "$model" "$context" "$prefix" "$half" "$half" "$n_gen" ;;
    two_servers_copy)
      server a "$gpu_a" -- parent "$model" "$context" "$prefix" "$work/state.device" \
        "$half" 0 "$n_gen" ;;
    two_servers_extent)
      server a "$gpu_a" "${fault_env[@]}" STATOR_KV_HOST="$huge_dir/kv" STATOR_KV_GROW="$grow" -- \
        parent "$model" "$context" "$prefix" "$work/state.huge" "$half" 0 "$n_gen" ;;
  esac
  local waiting=0
  if [[ "$config" == two_servers_copy || "$config" == two_servers_extent ]]; then
    waiting=1
  fi
  while kill -0 "${pids[@]}" 2>/dev/null; do
    if (( waiting )) && grep -q '^PUBLISHED ' "$work/server.a" 2>/dev/null; then
      waiting=0
      tokens=$(sed -n 's/^PUBLISHED prefix_tokens=\([0-9]*\).*/\1/p' "$work/server.a")
      second_start=$(( ($(date +%s%N) - start) / 1000000 ))
      if [[ "$config" == two_servers_copy ]]; then
        server b "$gpu_b" -- child "$model" "$context" "$work/state.device" \
          "$half" "$half" "$n_gen"
      else
        server b "$gpu_b" STATOR_KV_HOST="$huge_dir/kv" STATOR_KV_PREFIX="$tokens" \
          STATOR_KV_GROW="$grow" -- child "$model" "$context" "$work/state.huge" \
          "$half" "$half" "$n_gen"
      fi
    fi
    if [[ "$fault" == write ]] && (( ! injected )); then
      elapsed=$(( ($(date +%s%N) - start) / 1000000 ))
      if [[ "$config" == one_server ]] && (( elapsed >= fault_at_ms )); then
        : >"$trigger"; injected=$elapsed
      elif [[ "$config" == two_servers_extent ]] && (( second_start > 0 )) &&
           (( elapsed >= second_start + fault_after_b_ms )); then
        : >"$trigger"; injected=$elapsed
      fi
    fi
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sleep 0.05
  done
  {
    printf 'CASE config=%s fault=%s trigger_ms=%s\n' "$config" "$fault" "$injected"
    grep -h '^FAULT_INJECTED ' "$work"/server.*.err 2>/dev/null | sed 's/^/FAULT /' || true
    local name index=0 code result published
    for name in a b; do
      [[ -e "$work/server.$name" ]] || continue
      code=0
      wait "${pids[$index]}" || code=$?
      index=$((index + 1))
      result=$(grep '^RESULT ' "$work/server.$name" || true)
      published=$(grep '^PUBLISHED ' "$work/server.$name" || true)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        grep -v '^FAULT_INJECTED ' "$work/server.$name.err" | sed 's/^/STDERR /' | tail -n 3
      fi
      local started=0
      if [[ "$name" == b && "$config" != two_servers_compute ]]; then started=$second_start; fi
      printf 'SERVER name=%s exit=%s started_ms=%s %s\n' "$name" "$code" "$started" \
        "${result#RESULT }"
      if [[ -n "$published" ]]; then printf 'PUBLISH %s\n' "${published#PUBLISHED }"; fi
      { grep '^TEXT ' "$work/server.$name" || true; } |
        while read -r _ agent text; do
          printf 'AGENT index=%s text=%s\n' "$agent" \
            "$(printf '%s' "$text" | sha256sum | cut -c1-16)"
        done
    done
    printf 'MEMORY mem_available_drop_mib=%s\n' "$(( (before - lowest) / 1024 ))"
    printf 'END_CASE failed=%s\n' "$failed"
  } >>"$raw_log"
  pids=()
  rm -f "$work"/server.* "$work"/state.* "$huge_dir/kv.0" "$trigger"
}

forward=("one_server none" "one_server write" "two_servers_extent none" "two_servers_extent write")
backward=("two_servers_extent write" "two_servers_extent none" "one_server write" "one_server none")
for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then cases=("${forward[@]}"); else cases=("${backward[@]}"); fi
  printf 'BEGIN_KVBATCHFAULT run=%s paragraphs=%s context=%s agents=%s\n' "$run" \
    "$paragraphs" "$context" "$agents" >>"$raw_log"
  for item in "${cases[@]}"; do
    run_case $item
    sleep 2
  done
  printf 'END_KVBATCHFAULT\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_fault_domains.awk" "$raw_log" \
  >"$result_dir/engine_kvbatchfault_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/patches/stator_extents.patch stator_code/drivers/batch_server.cpp stator_code/drivers/fault_inject.cu script_example/03_hand_over_fork_and_agent_trees/summarize_fault_domains.awk \
    script_example/03_hand_over_fork_and_agent_trees/run_fault_domains.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvbatchfault_result_dir=%s\n' "$result_dir"
