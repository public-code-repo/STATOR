set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
none_counts=${NONE_COUNTS:-"1 2 4 8 12"}
kv_counts=${KV_COUNTS:-"1 2 4 8 12"}
weights_counts=${WEIGHTS_COUNTS:-"1 2 4 8 12 16 32"}
both_counts=${BOTH_COUNTS:-"1 2 4 8 12 16 32 64"}
prefix_file=${PREFIX_FILE:-"$STATOR_ROOT/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$STATOR_ROOT/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-1200}
sample_ms=${SAMPLE_MS:-4000}
reserve_gib=${RESERVE_GIB:-20}
estimate_none=${ESTIMATE_NONE_MIB:-6600}
estimate_kv=${ESTIMATE_KV_MIB:-5600}
estimate_weights=${ESTIMATE_WEIGHTS_MIB:-2000}
estimate_both=${ESTIMATE_BOTH_MIB:-900}
model=${MODEL:-"$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${KV_ENGINE:-"$STATOR_ROOT/engines/extents"}
stock_engine=${STOCK_ENGINE:-"$STATOR_ROOT/engines/stock"}
driver=${DRIVER:-"$STATOR_ROOT/bin/prefix_agent_extents"}
stock_driver=${STOCK_DRIVER:-"$STATOR_ROOT/bin/prefix_agent_stock"}
copy_driver=${COPY_DRIVER:-"$STATOR_ROOT/bin/prefix_agent_private_weights"}
mount_root=${MOUNT_ROOT:-"$STATOR_ROOT/mounts"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvstack"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$stock_driver" || ! -x "$copy_driver" ||
      ! -r "$model" ]]; then
  echo "missing driver or model; run make and see README.md" >&2
  exit 2
fi
for file in "$prefix_file" "$tasks_file"; do
  if [[ ! -r "$file" ]]; then echo "cannot read $file" >&2; exit 2; fi
done
if [[ -n "$(git -C "$stock_engine" status --porcelain --untracked-files=no)" ]]; then
  echo "the engine of the stack that shares nothing is not unmodified" >&2
  exit 2
fi
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount and the cache drop" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  if (( mounted_huge )); then sudo -n umount "$huge_dir" || true; fi
  rmdir "$huge_dir" "$mount_root" 2>/dev/null || true
  if [[ -n "$work" ]]; then rm -rf "$work"; fi
}
trap cleanup EXIT
mkdir -p "$huge_dir" "$result_dir"
sudo -n mount -t tmpfs -o "huge=always,size=8192m,uid=$(id -u),gid=$(id -g)" \
  tmpfs "$huge_dir"
mounted_huge=1
work=$(mktemp -d "$result_dir/work.XXXXXX")
prefix="$work/prefix"
cp "$prefix_file" "$prefix"
task_lines=$(wc -l <"$tasks_file")

suffix_of() {
  printf '\n\nTask for agent %s: %s\n' "$1" \
    "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
}
field_of() {
  sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p" "$2" | head -n 1
}
meminfo_kib() {
  awk -v name="$1:" '$1 == name { print $2 }' /proc/meminfo
}

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\n' "$gpu_a" "$gpu_b" "$repetitions"
  printf 'none_counts=%s\nkv_counts=%s\nweights_counts=%s\nboth_counts=%s\n' \
    "$none_counts" "$kv_counts" "$weights_counts" "$both_counts"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\nsample_ms=%s\n' "$context" \
    "$n_gen" "$grow" "$sample_ms"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'reserve_gib=%s\nestimate_mib=none:%s,kv:%s,weights:%s,both:%s\n' \
    "$reserve_gib" "$estimate_none" "$estimate_kv" "$estimate_weights" \
    "$estimate_both"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'mem_total_kib=%s\n' "$(meminfo_kib MemTotal)"
  printf 'engine_commit=%s\nstock_engine_commit=%s\n' \
    "$(git -C "$engine" rev-parse HEAD)" "$(git -C "$stock_engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$copy_driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so" "$stock_driver" \
    "$stock_engine/build/bin/libllama.so" \
    "$stock_engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

publish() {
  local label=$1 gpu=$2 program=$3 state=$4
  shift 4
  rm -f "$state"
  env CUDA_VISIBLE_DEVICES="$gpu" "$@" \
    "$program" parent "$model" "$context" "$prefix" "$state" "" 0 \
    >"$work/publish.out" 2>"$work/publish.err" || true
  local line
  line=$(grep '^PUBLISHED ' "$work/publish.out" || true)
  printf 'PUBLISH store=%s %s' "$label" "${line#PUBLISHED }"
}

sample_agents() {
  local rollups=() pid
  for pid in "$@"; do rollups+=("/proc/$pid/smaps_rollup"); done
  cat "${rollups[@]}" 2>/dev/null | awk '
    /^Pss_Anon:/ { anon += $2 }
    /^Pss_File:/ { file += $2 }
    /^Pss_Shmem:/ { shmem += $2 }
    END { printf "%d %d %d", anon / 1024, file / 1024, shmem / 1024 }' || true
  printf ' '
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    2>/dev/null | awk -F', *' -v list="$*" '
    BEGIN { n = split(list, pids, " "); for (i = 1; i <= n; ++i) mine[pids[i]] = 1 }
    ($1 in mine) && $2 ~ /^[0-9]+$/ { total += $2 }
    END { printf "%d", total }' || true
}

run_case() {
  local mode=$1 agents=$2 tokens=$3 first=$4 other=$5
  local state program envs=() estimate index gpu
  local pids=() gpus=() starts=() ends=() failed=0
  case "$mode" in
    none)
      program=$stock_driver state="$work/state.stock" estimate=$estimate_none ;;
    weights)
      program=$driver state="$work/state.device" estimate=$estimate_weights
      envs=(STATOR_HOST_PTR=1) ;;
    kv)
      program=$copy_driver state="$work/state.huge" estimate=$estimate_kv
      envs=(STATOR_HOST_PTR=1 STATOR_KV_HOST="$huge_dir/kv"
            STATOR_KV_PREFIX="$tokens" STATOR_KV_GROW="$grow") ;;
    both)
      program=$driver state="$work/state.huge" estimate=$estimate_both
      envs=(STATOR_HOST_PTR=1 STATOR_KV_HOST="$huge_dir/kv"
            STATOR_KV_PREFIX="$tokens" STATOR_KV_GROW="$grow") ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  local before lowest sample free_before
  before=$(meminfo_kib MemAvailable)
  free_before=$(meminfo_kib MemFree)
  if (( before / 1024 - agents * estimate < reserve_gib * 1024 )); then
    printf 'SKIPPED mode=%s agents=%s mem_available_mib=%s\n' "$mode" "$agents" \
      "$(( before / 1024 ))" >>"$raw_log"
    return 0
  fi
  lowest=$before
  local case_start
  case_start=$(date +%s%N)
  for index in $(seq 0 $((agents - 1))); do
    if (( index % 2 == 0 )); then gpu=$first; else gpu=$other; fi
    gpus+=("$gpu")
    starts+=("$(date +%s%N)")
    env CUDA_VISIBLE_DEVICES="$gpu" "${envs[@]}" \
      "$program" child "$model" "$context" "$state" "$(suffix_of "$index")" \
      "$n_gen" >"$work/agent.$index" 2>"$work/agent.$index.err" &
    pids+=($!)
    ends+=(0)
  done
  live_pids=("${pids[@]}")
  local alive=$agents tick=0 elapsed_ms next_sample_ms=2500 samples=0
  local anon=0 file=0 shmem=0 device=0 parts=()
  while (( alive > 0 )); do
    alive=0
    for index in "${!pids[@]}"; do
      if (( ends[index] == 0 )); then
        if kill -0 "${pids[$index]}" 2>/dev/null; then
          alive=$((alive + 1))
        else
          ends[index]=$(date +%s%N)
        fi
      fi
    done
    if (( tick % 5 == 0 )); then
      sample=$(meminfo_kib MemAvailable)
      (( sample < lowest )) && lowest=$sample
    fi
    elapsed_ms=$(( ($(date +%s%N) - case_start) / 1000000 ))
    if (( alive == agents && elapsed_ms >= next_sample_ms )); then
      read -r -a parts <<<"$(sample_agents "${pids[@]}")"
      if (( ${#parts[@]} == 4 )); then
        samples=$((samples + 1))
        (( parts[0] > anon )) && anon=${parts[0]}
        (( parts[1] > file )) && file=${parts[1]}
        (( parts[2] > shmem )) && shmem=${parts[2]}
        (( parts[3] > device )) && device=${parts[3]}
      fi
      next_sample_ms=$(( elapsed_ms + sample_ms ))
    fi
    if (( elapsed_ms / 1000 > limit_s )); then
      kill -9 "${pids[@]}" 2>/dev/null || true
    fi
    tick=$((tick + 1))
    sleep 0.02
  done
  live_pids=()
  {
    printf 'CASE mode=%s agents=%s\n' "$mode" "$agents"
    for index in "${!pids[@]}"; do
      local code=0 text result
      wait "${pids[$index]}" || code=$?
      result=$(grep '^RESULT ' "$work/agent.$index" || true)
      text=$({ grep '^TEXT ' "$work/agent.$index" || true; } | sha256sum | cut -c1-16)
      if (( code != 0 )) || [[ -z "$result" ]]; then
        failed=$((failed + 1))
        text=none
        sed 's/^/STDERR /' "$work/agent.$index.err" | tail -n 5
      fi
      printf 'AGENT index=%s gpu=%s exit=%s wall_ms=%s text=%s %s\n' "$index" \
        "${gpus[$index]:4:8}" "$code" \
        "$(( (ends[index] - starts[index]) / 1000000 ))" "$text" \
        "${result#RESULT }"
    done
    printf 'MEMORY mem_available_drop_mib=%s mem_available_before_mib=%s ' \
      "$(( (before - lowest) / 1024 ))" "$(( before / 1024 ))"
    printf 'mem_free_before_mib=%s ' "$(( free_before / 1024 ))"
    printf 'pss_anon_mib=%s pss_file_mib=%s pss_shmem_mib=%s device_mib=%s samples=%s\n' \
      "$anon" "$file" "$shmem" "$device" "$samples"
    printf 'END_CASE failed_agents=%s wall_ms=%s\n' "$failed" \
      "$(( ($(date +%s%N) - case_start) / 1000000 ))"
  } >>"$raw_log"
}

counts_of() {
  case "$1" in
    none) printf '%s' "$none_counts" ;;
    kv) printf '%s' "$kv_counts" ;;
    weights) printf '%s' "$weights_counts" ;;
    both) printf '%s' "$both_counts" ;;
  esac
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    first=$gpu_a other=$gpu_b modes=(none kv weights both)
  else
    first=$gpu_b other=$gpu_a modes=(both weights kv none)
  fi
  rm -f "$huge_dir/kv.0"
  sync
  sudo -n sh -c 'echo 1 >/proc/sys/vm/drop_caches'
  cat "$model" >/dev/null
  {
    printf 'BEGIN_KVSTACK run=%s context=%s publisher_gpu=%s mem_free_mib=%s ' \
      "$run" "$context" "${first:4:8}" "$(( $(meminfo_kib MemFree) / 1024 ))"
    printf 'kernel_reclaimable_mib=%s\n' "$(( $(meminfo_kib KReclaimable) / 1024 ))"
    publish stock "$first" "$stock_driver" "$work/state.stock"
    printf '\n'
    publish device "$first" "$driver" "$work/state.device" STATOR_HOST_PTR=1
    printf '\n'
    publish huge "$first" "$driver" "$work/state.huge" STATOR_HOST_PTR=1 \
      STATOR_KV_HOST="$huge_dir/kv" STATOR_KV_GROW="$grow"
    printf ' file_used_kib=%s\n' "$(du -k "$huge_dir/kv.0" | cut -f1)"
  } >>"$raw_log"
  chmod 0400 "$huge_dir/kv.0"
  tokens=$(field_of prefix_tokens "$work/publish.out")
  for mode in "${modes[@]}"; do
    for agents in $(counts_of "$mode"); do
      run_case "$mode" "$agents" "$tokens" "$first" "$other"
    done
  done
  printf 'END_KVSTACK\n' >>"$raw_log"
  rm -f "$huge_dir/kv.0" "$work"/state.*
done

awk -f "$script_dir/summarize_stack.awk" "$raw_log" \
  >"$result_dir/engine_kvstack_summary.csv"
awk -v table=fit -v model_bytes="$(stat -c %s "$model")" \
  -v mem_total_kib="$(meminfo_kib MemTotal)" \
  -f "$script_dir/summarize_stack.awk" "$raw_log" \
  >"$result_dir/engine_kvstack_fit.csv"
awk -v table=texts -f "$script_dir/summarize_stack.awk" "$raw_log" \
  >"$result_dir/engine_kvstack_texts.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/patches/stator_extents.patch stator_code/drivers/prefix_agent.cpp stator_code/drivers/prefix_agent_private_weights.cpp \
    script_example/04_overall_memory_and_capacity/summarize_stack.awk \
    script_example/04_overall_memory_and_capacity/run_stack.sh "$prefix_file" "$tasks_file"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvstack_result_dir=%s\n' "$result_dir"
