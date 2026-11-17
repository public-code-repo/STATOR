set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
read -r -a fanouts <<<"${FANOUTS:-2 2 4}"
paragraphs=${PARAGRAPHS:-320}
level_paragraphs=${LEVEL_PARAGRAPHS:-20}
n_gen=${N_GEN:-64}
late_factor=${LATE_FACTOR:-3}
grow=${GROW_ROWS:-256}
limit_s=${CASE_LIMIT_S:-1200}
model=${MODEL:-"$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
engine=${CHAIN_ENGINE:-"$STATOR_ROOT/engines/chains"}
driver=${DRIVER:-"$STATOR_ROOT/bin/tree_agent_chains"}
mount_root=${MOUNT_ROOT:-"$STATOR_ROOT/mounts"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-kvdeep"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
depth=${#fanouts[@]}

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if (( depth < 1 )); then echo "FANOUTS names no level" >&2; exit 2; fi
if [[ ! -x "$driver" || ! -r "$model" ]]; then
  echo "missing driver or model; run make tree_agent and see README.md" >&2
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

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\nfanouts=%s\n' "$gpu_a" "$gpu_b" \
    "$repetitions" "${fanouts[*]}"
  printf 'paragraphs=%s\nlevel_paragraphs=%s\nn_gen=%s\nlate_factor=%s\ngrow_rows=%s\n' \
    "$paragraphs" "$level_paragraphs" "$n_gen" "$late_factor" "$grow"
  printf 'model=%s\n' "$(basename "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$engine" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$driver" "$engine/build/bin/libllama.so" \
    "$engine/build/bin/libggml-cuda.so"
} >"$result_dir/metadata.txt"

paragraph="The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. "
context=4096
while (( context < (paragraphs + (depth - 1) * level_paragraphs) * 52 + 1024 )); do
  context=$(( context * 2 ))
done
prefix="$work/prefix"
for _ in $(seq 1 "$paragraphs"); do printf '%s' "$paragraph"; done >"$prefix"
node_context() {
  local text="" count
  for count in $(seq 1 "$level_paragraphs"); do
    text+="Note $count of subtree $1: the subtree reviews how the kernel handles memory pressure, which pages it reclaims first, and how a task that waits for a page is resumed. "
  done
  printf '\n\n%s' "$text"
}
inner_task() {
  printf '\n\nTask for the agent of subtree %s: state the goal of the subtree in one sentence.' "$1"
}
leaf_task() {
  printf '\n\nTask for agent %s: summarize the text above in %s sentences.' \
    "$1" "$(( $1 % 12 + 2 ))"
}
text_of() { { grep '^TEXT ' "$1" || true; } | sha256sum | cut -c1-16; }
field_of() {
  { grep "^$2 " "$3" || true; } | head -n 1 |
    sed -n "s/^.* $1=\\([^ ]*\\).*\$/\\1/p"
}
used_mib() {
  df -k --output=used "$huge_dir" | awk 'NR > 1 { printf "%d", $1 / 1024 }'
}
wait_line() {
  until grep -q "^$1 " "$2" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || return 0
    sleep 0.02
  done
}
gen_of() {
  if [[ "$1" == r || "$1" == r.0 || "$1" == r.0.* ]]; then
    printf '%s' "$n_gen"
  else
    printf '%s' "$(( n_gen * late_factor ))"
  fi
}

run_case() {
  local mode=$1 run=$2 level node parent child
  rm -f "$work"/state.* "$work/go" "$work"/out.* "$huge_dir"/*
  declare -A pid_of=() tokens_of=() chain_of=() gpu_of=()
  local before lowest sample started
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  started=$(date +%s%N)
  local timeline="$result_dir/timeline.$mode.$run"
  : >"$timeline"

  local root_env=()
  [[ "$mode" == chain ]] && root_env=(STATOR_KV_HOST="$huge_dir/r" STATOR_KV_GROW="$grow")
  env CUDA_VISIBLE_DEVICES="$gpu_a" STATOR_HOST_PTR=1 "${root_env[@]}" \
    "$driver" "$model" "$context" "$(gen_of r)" "text:$prefix" \
    "save:$work/state.r" "wait:$work/go" "say:$(inner_task r)" gen \
    >"$work/out.r" 2>"$work/out.r.err" &
  pid_of[r]=$!
  gpu_of[r]=$gpu_a
  live_pids=("${pid_of[r]}")
  wait_line PUBLISHED "$work/out.r" "${pid_of[r]}"
  tokens_of[r]=$(field_of tokens PUBLISHED "$work/out.r")
  chain_of[r]="$huge_dir/r:${tokens_of[r]:-0}"

  local parents=(r) nodes=() leaves=() inner=() number=0
  for level in $(seq 0 $((depth - 1))); do
    nodes=()
    local position=0
    for parent in "${parents[@]}"; do
      for child in $(seq 0 $((fanouts[level] - 1))); do
        node="$parent.$child"
        nodes+=("$node")
        if (( position % 2 == 0 )); then gpu_of[$node]=$gpu_a; else gpu_of[$node]=$gpu_b; fi
        position=$((position + 1))
        local node_env=() steps=("load:$work/state.$parent")
        if (( level < depth - 1 )); then
          if [[ "$mode" == chain ]]; then
            node_env=(STATOR_KV_CHAIN="${chain_of[$parent]}"
              STATOR_KV_HOST="$huge_dir/${node//./_}" STATOR_KV_GROW="$grow")
          fi
          steps+=("say:$(node_context "$node")" "save:$work/state.$node"
            "wait:$work/go" "say:$(inner_task "$node")" gen)
          inner+=("$node")
        else
          if [[ "$mode" == chain ]]; then
            node_env=(STATOR_KV_CHAIN="${chain_of[$parent]}" STATOR_KV_HOST=anon
              STATOR_KV_GROW="$grow")
          fi
          steps+=("say:$(leaf_task "$number")" gen)
          leaves+=("$node")
          number=$((number + 1))
        fi
        env CUDA_VISIBLE_DEVICES="${gpu_of[$node]}" STATOR_HOST_PTR=1 \
          "${node_env[@]}" "$driver" "$model" "$context" "$(gen_of "$node")" \
          "${steps[@]}" >"$work/out.$node" 2>"$work/out.$node.err" &
        pid_of[$node]=$!
        live_pids+=("${pid_of[$node]}")
      done
    done
    if (( level < depth - 1 )); then
      for node in "${nodes[@]}"; do
        wait_line PUBLISHED "$work/out.$node" "${pid_of[$node]}"
        tokens_of[$node]=$(field_of tokens PUBLISHED "$work/out.$node")
        chain_of[$node]="${chain_of[${node%.*}]},$huge_dir/${node//./_}:${tokens_of[$node]:-0}"
      done
    fi
    parents=("${nodes[@]}")
  done
  touch "$work/go"
  for node in "${leaves[@]}"; do
    wait_line ATTACHED "$work/out.$node" "${pid_of[$node]}"
  done
  local files_attached files_removed=0
  files_attached=$(used_mib)
  if [[ "$mode" == chain ]]; then
    rm -f "$huge_dir"/*
    files_removed=1
  fi

  local alive=1 timed_out=0 pid first_alive files_all=-1 files_first_left=-1
  while (( alive > 0 )); do
    alive=0
    first_alive=0
    for node in "${!pid_of[@]}"; do
      if kill -0 "${pid_of[$node]}" 2>/dev/null; then
        alive=$((alive + 1))
        if [[ "$node" == r.0 || "$node" == r.0.* ]]; then first_alive=$((first_alive + 1)); fi
      fi
    done
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    local used
    used=$(used_mib)
    printf '%s %s %s %s %s\n' "$(( ($(date +%s%N) - started) / 1000000 ))" \
      "$(( (before - sample) / 1024 ))" "$used" "$alive" "$first_alive" >>"$timeline"
    if (( alive == ${#pid_of[@]} )); then files_all=$used; fi
    if (( first_alive == 0 && alive > 1 && files_first_left < 0 )); then
      sleep 0.5
      files_first_left=$(used_mib)
    fi
    if (( ($(date +%s%N) - started) / 1000000000 > limit_s )); then
      timed_out=1
      kill -9 "${live_pids[@]}" 2>/dev/null || true
    fi
    sleep 0.2
  done
  local files_end
  files_end=$(used_mib)
  {
    printf 'CASE mode=%s fanouts=%s processes=%s leaves=%s\n' "$mode" \
      "$(IFS=x; echo "${fanouts[*]}")" "${#pid_of[@]}" "${#leaves[@]}"
    local code failed=0
    code=0
    wait "${pid_of[r]}" || code=$?
    if (( code != 0 )) || ! grep -q '^RESULT ' "$work/out.r"; then
      failed=$((failed + 1))
      sed 's/^/STDERR /' "$work/out.r.err" | tail -n 5
    fi
    printf 'ROOT exit=%s tokens=%s publish_ms=%s state_bytes=%s\n' "$code" \
      "${tokens_of[r]:-0}" "$(field_of publish_ms PUBLISHED "$work/out.r")" \
      "$(field_of state_bytes PUBLISHED "$work/out.r")"
    for node in "${inner[@]}"; do
      code=0
      wait "${pid_of[$node]}" || code=$?
      if (( code != 0 )) || ! grep -q '^RESULT ' "$work/out.$node"; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/out.$node.err" | tail -n 5
      fi
      printf 'INNER node=%s gpu=%s exit=%s text=%s tokens=%s context_ms=%s load_ms=%s publish_ms=%s state_bytes=%s\n' \
        "$node" "${gpu_of[$node]:4:8}" "$code" "$(text_of "$work/out.$node")" \
        "${tokens_of[$node]:-0}" "$(field_of context_ms ATTACHED "$work/out.$node")" \
        "$(field_of load_ms ATTACHED "$work/out.$node")" \
        "$(field_of publish_ms PUBLISHED "$work/out.$node")" \
        "$(field_of state_bytes PUBLISHED "$work/out.$node")"
    done
    for node in "${leaves[@]}"; do
      code=0
      wait "${pid_of[$node]}" || code=$?
      if (( code != 0 )) || ! grep -q '^RESULT ' "$work/out.$node"; then
        failed=$((failed + 1))
        sed 's/^/STDERR /' "$work/out.$node.err" | tail -n 5
      fi
      printf 'LEAF node=%s gpu=%s exit=%s text=%s context_ms=%s load_ms=%s first_token_ms=%s generation_tps=%s\n' \
        "$node" "${gpu_of[$node]:4:8}" "$code" "$(text_of "$work/out.$node")" \
        "$(field_of context_ms ATTACHED "$work/out.$node")" \
        "$(field_of load_ms ATTACHED "$work/out.$node")" \
        "$(field_of first_token_ms RESULT "$work/out.$node")" \
        "$(field_of generation_tps RESULT "$work/out.$node")"
    done
    printf 'MEMORY tree_drop_mib=%s files_attached_mib=%s files_removed=%s ' \
      "$(( (before - lowest) / 1024 ))" "$files_attached" "$files_removed"
    printf 'files_all_alive_mib=%s files_first_subtree_left_mib=%s files_end_mib=%s\n' \
      "$files_all" "$files_first_left" "$files_end"
    printf 'END_CASE failed=%s timed_out=%s wall_ms=%s\n' "$failed" "$timed_out" \
      "$(( ($(date +%s%N) - started) / 1000000 ))"
  } >>"$raw_log"
  live_pids=()
  rm -f "$work"/state.* "$work/go" "$huge_dir"/*
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then modes=(copy chain); else modes=(chain copy); fi
  printf 'BEGIN_KVDEEP run=%s paragraphs=%s context=%s\n' "$run" "$paragraphs" \
    "$context" >>"$raw_log"
  for mode in "${modes[@]}"; do
    run_case "$mode" "$run"
  done
  printf 'END_KVDEEP\n' >>"$raw_log"
done

awk -f "$script_dir/summarize_deep_tree.awk" "$raw_log" \
  >"$result_dir/engine_kvdeep_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/patches/extent_chains.patch stator_code/drivers/tree_agent.cpp script_example/03_hand_over_fork_and_agent_trees/summarize_deep_tree.awk \
    script_example/03_hand_over_fork_and_agent_trees/run_deep_tree.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_kvdeep_result_dir=%s\n' "$result_dir"
