set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
config_list=${CONFIGS:-"timeslice mps"}
fault_list=${FAULTS:-"none write kill_worker kill_leader"}
groups=${GROUPS_COUNT:-2}
workers=${WORKERS:-4}
turns=${TURNS:-3}
prefix_file=${PREFIX_FILE:-"$STATOR_ROOT/workloads/bfcl_prefix.txt"}
tasks_file=${TASKS_FILE:-"$STATOR_ROOT/workloads/bfcl_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
grow=${GROW_ROWS:-256}
fault_after_s=${FAULT_AFTER_S:-10}
probe_mib=${PROBE_MIB:-64}
limit_s=${CASE_LIMIT_S:-600}
model=${MODEL:-"$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${CHAIN_ENGINE:-"$STATOR_ROOT/engines/chains"}
driver=${DRIVER:-"$STATOR_ROOT/bin/tree_agent_chains"}
probe=${PROBE:-"$STATOR_ROOT/bin/protect_probe"}
mount_root=${MOUNT_ROOT:-"$STATOR_ROOT/mounts"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvpipefault"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if [[ ! -x "$driver" || ! -x "$probe" || ! -r "$model" ]]; then
  echo "missing driver, probe or model; run make tree_agent_chains protect_probe" >&2
  exit 2
fi
for file in "$prefix_file" "$tasks_file"; do
  if [[ ! -r "$file" ]]; then echo "cannot read $file" >&2; exit 2; fi
done
if ! sudo -n true 2>/dev/null; then
  echo "passwordless sudo is required for the tmpfs mount" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi
if pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' >/dev/null ||
   pgrep -f '(^|/)nvidia-cuda-mps-server( |$)' >/dev/null; then
  echo "an MPS daemon is already running; refusing to reuse it" >&2
  exit 2
fi

if [[ -n "${MPS_ROOT:-}" ]]; then
  mps_root=$MPS_ROOT
  mkdir -p "$mps_root"
  [[ -z "$(ls -A "$mps_root")" ]] || { echo "MPS_ROOT is not empty" >&2; exit 2; }
else
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/kq.XXXXXX")
fi
(( ${#mps_root} <= 84 )) || { echo "MPS directory path too long" >&2; exit 2; }
pipe_a="$mps_root/a"
pipe_b="$mps_root/b"
mkdir -p "$pipe_a" "$pipe_b" "$mps_root/la" "$mps_root/lb"
running_a=0
running_b=0
daemon_pids=()
await_control() {
  local waited=0
  until [[ -e "$1/control" ]] || (( waited >= 100 )); do
    sleep 0.1
    waited=$((waited + 1))
  done
}
start_mps() {
  if [[ "$1" == a && "$running_a" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_a" CUDA_MPS_PIPE_DIRECTORY="$pipe_a" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/la" nvidia-cuda-mps-control -d
    running_a=1
    await_control "$pipe_a"
  elif [[ "$1" == b && "$running_b" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_b" CUDA_MPS_PIPE_DIRECTORY="$pipe_b" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" nvidia-cuda-mps-control -d
    running_b=1
    await_control "$pipe_b"
  fi
  mapfile -t daemon_pids < <(pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' || true)
}
await_no_mps() {
  local waited=0 pid
  while { pgrep -f '(^|/)nvidia-cuda-mps-server( |$)' >/dev/null ||
          pgrep -f '(^|/)nvidia-cuda-mps-control( |$)' >/dev/null; } &&
        (( waited < 300 )); do
    sleep 0.1
    waited=$((waited + 1))
  done
  for pid in "${daemon_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; sleep 1; fi
  done
  daemon_pids=()
}
stop_mps() {
  if [[ "$1" == a && "$running_a" -eq 1 ]]; then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe_a" CUDA_MPS_LOG_DIRECTORY="$mps_root/la" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    running_a=0
  elif [[ "$1" == b && "$running_b" -eq 1 ]]; then
    env CUDA_MPS_PIPE_DIRECTORY="$pipe_b" CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" \
      bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    running_b=0
  fi
}
prepare() {
  stop_mps a
  stop_mps b
  await_no_mps
  if [[ "$1" == gpu_mps ]]; then start_mps a; start_mps b; fi
  if [[ "$1" == mps ]]; then start_mps a; fi
}

huge_dir="$mount_root/kv_huge"
mounted_huge=0
work=
live_pids=()
cleanup() {
  if (( ${#live_pids[@]} )); then kill -9 "${live_pids[@]}" 2>/dev/null || true; fi
  stop_mps a
  stop_mps b
  await_no_mps
  rm -rf "${mps_root:?}"
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
task_lines=$(wc -l <"$tasks_file")

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\nconfigs=%s\nfaults=%s\n' "$gpu_a" \
    "$gpu_b" "$repetitions" "$config_list" "$fault_list"
  printf 'groups=%s\nworkers=%s\nturns=%s\n' "$groups" "$workers" "$turns"
  printf 'context=%s\nn_gen=%s\ngrow_rows=%s\nfault_after_s=%s\nprobe_mib=%s\n' \
    "$context" "$n_gen" "$grow" "$fault_after_s" "$probe_mib"
  printf 'prefix_file=%s\nprefix_sha256=%s\ntasks_file=%s\ntasks_sha256=%s\n' \
    "$(basename "$prefix_file")" "$(sha256sum "$prefix_file" | cut -d' ' -f1)" \
    "$(basename "$tasks_file")" "$(sha256sum "$tasks_file" | cut -d' ' -f1)"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$probe" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

planner_task=$'\n\nRequest to the planner: name the three tools above that a travel assistant needs most.\n'
group_brief() {
  printf '\n\nBrief for the leader of group %s: your workers answer user requests with the tools above. Write the rules they follow when they choose a tool and fill its parameters.\n' "$1"
}
leader_task() {
  printf '\n\nRequest to the leader of group %s: state in one sentence what your workers must check before a call.\n' "$1"
}
worker_turn() {
  if (( $2 == 0 )); then
    printf '\n\nUser request for worker %s: %s\nCall:\n' "$1" \
      "$(sed -n "$(( $1 % task_lines + 1 ))p" "$tasks_file")"
  else
    printf '\n\nTool result %s for worker %s: {"status": "ok", "items": %s}\nNext call or final answer:\n' \
      "$2" "$1" "$(( $1 + $2 ))"
  fi
}
leader_gpu() { if (( $1 % 2 == 0 )); then echo "$gpu_a"; else echo "$gpu_b"; fi; }
worker_gpu() {
  if (( ($1 + $2) % 2 == 0 )); then echo "$gpu_a"; else echo "$gpu_b"; fi
}
text_of() { { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16; }
field_of() {
  { grep "^$2 " "$3" || true; } | head -n 1 |
    sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p"
}
wait_line() {
  until grep -q "^$1 " "$2" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || return 0
    sleep 0.02
  done
}
launch() {
  local config=$1 gpu=$2 output=$3 mps_env=()
  shift 3
  if [[ "$config" == mps ]]; then
    mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe_a")
  elif [[ "$config" == gpu_mps ]]; then
    if [[ "$gpu" == "$gpu_a" ]]; then
      mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe_a")
    else
      mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe_b")
    fi
  fi
  env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$gpu" \
    STATOR_HOST_PTR=1 "${mps_env[@]}" "$@" >"$output" 2>"$output.err" &
}

run_case() {
  local config=$1 fault=$2 group worker number gpu turn
  prepare "$config"
  rm -f "$work"/state.* "$work/go" "$work"/out.* "$work"/probe* "$huge_dir"/*
  local started t_go=0
  started=$(date +%s%N)

  launch "$config" "$gpu_a" "$work/out.root" \
    STATOR_KV_HOST="$huge_dir/r" STATOR_KV_GROW="$grow" \
    "$driver" "$model" "$context" "$n_gen" "text:$prefix_file" \
    "save:$work/state.root" "wait:$work/go" "say:$planner_task" gen
  local root_pid=$!
  live_pids=("$root_pid")
  wait_line PUBLISHED "$work/out.root" "$root_pid"
  local root_tokens root_file
  root_tokens=$(field_of tokens PUBLISHED "$work/out.root")
  root_file=$(find "$huge_dir" -maxdepth 1 -type f -name 'r*' | sort | head -n 1)

  local leader_pids=() leader_tokens=()
  for group in $(seq 0 $((groups - 1))); do
    launch "$config" "$(leader_gpu "$group")" "$work/out.leader.$group" \
      STATOR_KV_CHAIN="$huge_dir/r:${root_tokens:-0}" \
      STATOR_KV_HOST="$huge_dir/m$group" STATOR_KV_GROW="$grow" \
      "$driver" "$model" "$context" "$n_gen" \
      "load:$work/state.root" "say:$(group_brief "$group")" gen \
      "save:$work/state.leader.$group" "wait:$work/go" \
      "say:$(leader_task "$group")" gen
    leader_pids+=($!)
  done
  live_pids+=("${leader_pids[@]}")
  for group in $(seq 0 $((groups - 1))); do
    wait_line PUBLISHED "$work/out.leader.$group" "${leader_pids[$group]}"
    leader_tokens+=("$(field_of tokens PUBLISHED "$work/out.leader.$group")")
  done

  local worker_pids=() worker_groups=() worker_gpus=() worker_files=()
  number=0
  for group in $(seq 0 $((groups - 1))); do
    for worker in $(seq 0 $((workers - 1))); do
      gpu=$(worker_gpu "$group" "$worker")
      local steps=("load:$work/state.leader.$group")
      for turn in $(seq 0 $((turns - 1))); do
        steps+=("say:$(worker_turn "$number" "$turn")" gen)
      done
      launch "$config" "$gpu" "$work/out.worker.$number" \
        STATOR_KV_CHAIN="$huge_dir/r:${root_tokens:-0},$huge_dir/m$group:${leader_tokens[$group]:-0}" \
        STATOR_KV_HOST=anon STATOR_KV_GROW="$grow" \
        "$driver" "$model" "$context" "$n_gen" "${steps[@]}"
      worker_pids+=($!)
      worker_groups+=("$group")
      worker_gpus+=("$gpu")
      worker_files+=("$work/out.worker.$number")
      number=$((number + 1))
    done
  done
  live_pids+=("${worker_pids[@]}")
  touch "$work/go"
  t_go=$(date +%s%N)

  local victim=none victim_alive=0 alive_at_fault=0 probe_line=none probe_exit=none pid
  if [[ "$fault" != none ]]; then
    sleep "$fault_after_s"
    for pid in "${live_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive_at_fault=$((alive_at_fault + 1)); fi
    done
    case $fault in
      write)
        victim=probe
        victim_alive=1
        launch "$config" "${worker_gpus[0]}" "$work/probe" \
          "$probe" host "$root_file" "$probe_mib"
        local probe_pid=$!
        probe_exit=0
        wait "$probe_pid" || probe_exit=$?
        probe_line=$(grep '^RESULT ' "$work/probe" | head -n 1 || true)
        probe_line=${probe_line#RESULT }
        probe_line=${probe_line// /,}
        [[ -n "$probe_line" ]] || probe_line=no_result ;;
      kill_worker)
        victim=worker0
        if kill -0 "${worker_pids[0]}" 2>/dev/null; then victim_alive=1; fi
        kill -9 "${worker_pids[0]}" 2>/dev/null || true ;;
      kill_leader)
        victim=leader0
        if kill -0 "${leader_pids[0]}" 2>/dev/null; then victim_alive=1; fi
        kill -9 "${leader_pids[0]}" 2>/dev/null || true ;;
      *) echo "unknown fault $fault" >&2; exit 2 ;;
    esac
  fi

  local alive=1 timed_out=0
  while (( alive > 0 )); do
    alive=0
    for pid in "${live_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    if (( ($(date +%s%N) - started) / 1000000000 > limit_s )); then
      timed_out=1
      kill -9 "${live_pids[@]}" 2>/dev/null || true
    fi
    sleep 0.1
  done
  local ended
  ended=$(date +%s%N)
  {
    printf 'CASE config=%s fault=%s groups=%s workers=%s turns=%s\n' "$config" \
      "$fault" "$groups" "$workers" "$turns"
    local code=0 completed
    wait "$root_pid" 2>/dev/null || code=$?
    completed=0
    if (( code == 0 )) && grep -q '^RESULT ' "$work/out.root"; then completed=1; fi
    if (( completed == 0 )); then sed 's/^/STDERR /' "$work/out.root.err" | tail -n 3; fi
    printf 'PLANNER gpu=%s exit=%s completed=%s text=%s tokens=%s publish_ms=%s\n' \
      "${gpu_a:4:8}" "$code" "$completed" "$(text_of "$work/out.root")" \
      "${root_tokens:-0}" "$(field_of publish_ms PUBLISHED "$work/out.root")"
    for group in $(seq 0 $((groups - 1))); do
      code=0
      wait "${leader_pids[$group]}" 2>/dev/null || code=$?
      completed=0
      if (( code == 0 )) && grep -q '^RESULT ' "$work/out.leader.$group"; then completed=1; fi
      if (( completed == 0 )); then
        sed 's/^/STDERR /' "$work/out.leader.$group.err" | tail -n 3
      fi
      printf 'LEADER group=%s gpu=%s exit=%s completed=%s text=%s tokens=%s publish_ms=%s\n' \
        "$group" "$(leader_gpu "$group" | cut -c5-12)" "$code" "$completed" \
        "$(text_of "$work/out.leader.$group")" "${leader_tokens[$group]:-0}" \
        "$(field_of publish_ms PUBLISHED "$work/out.leader.$group")"
    done
    for number in "${!worker_pids[@]}"; do
      code=0
      wait "${worker_pids[$number]}" 2>/dev/null || code=$?
      completed=0
      if (( code == 0 )) && grep -q '^RESULT ' "${worker_files[$number]}"; then completed=1; fi
      if (( completed == 0 )); then
        sed 's/^/STDERR /' "${worker_files[$number]}.err" | tail -n 3
      fi
      printf 'WORKER index=%s group=%s gpu=%s exit=%s completed=%s text=%s generated=%s generation_tps=%s first_token_ms=%s total_ms=%s\n' \
        "$number" "${worker_groups[$number]}" "${worker_gpus[$number]:4:8}" "$code" \
        "$completed" "$(text_of "${worker_files[$number]}")" \
        "$(field_of generated RESULT "${worker_files[$number]}")" \
        "$(field_of generation_tps RESULT "${worker_files[$number]}")" \
        "$(field_of first_token_ms RESULT "${worker_files[$number]}")" \
        "$(field_of total_ms RESULT "${worker_files[$number]}")"
    done
    printf 'FAULT victim=%s victim_alive=%s alive_at_fault=%s probe_exit=%s probe=%s\n' \
      "$victim" "$victim_alive" "$alive_at_fault" "$probe_exit" "$probe_line"
    if [[ "$fault" == write ]]; then sed 's/^/PROBE_STDERR /' "$work/probe.err" | tail -n 3; fi
    printf 'END_CASE timed_out=%s mps_servers=%s wall_ms=%s workers_ms=%s\n' "$timed_out" \
      "$(pgrep -fc '(^|/)nvidia-cuda-mps-server( |$)' || true)" \
      "$(( (ended - started) / 1000000 ))" "$(( (ended - t_go) / 1000000 ))"
  } >>"$raw_log"
  live_pids=()
  rm -f "$work"/state.* "$work/go" "$work"/out.* "$work"/probe* "$huge_dir"/*
}

read -r -a faults <<<"$fault_list"
for run in $(seq 1 "$repetitions"); do
  configs=()
  for config in $config_list; do
    if (( run % 2 == 1 )); then configs+=("$config"); else configs=("$config" "${configs[@]}"); fi
  done
  printf 'BEGIN_KVPIPEFAULT run=%s context=%s\n' "$run" "$context" >>"$raw_log"
  for config in "${configs[@]}"; do
    for fault in "${faults[@]}"; do
      run_case "$config" "$fault"
    done
  done
  printf 'END_KVPIPEFAULT\n' >>"$raw_log"
done
stop_mps a
stop_mps b
await_no_mps

awk -f "$script_dir/summarize_tree_faults.awk" "$raw_log" \
  >"$result_dir/engine_kvpipefault_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/patches/extent_chains.patch stator_code/drivers/tree_agent.cpp stator_code/probes/protect_probe.cu \
    script_example/03_hand_over_fork_and_agent_trees/summarize_tree_faults.awk script_example/03_hand_over_fork_and_agent_trees/run_tree_faults.sh \
    "$(realpath --relative-to=. "$prefix_file")" \
    "$(realpath --relative-to=. "$tasks_file")"
) >"$result_dir/source_hashes.txt"
printf 'engine_kvpipefault_result_dir=%s\n' "$result_dir"
