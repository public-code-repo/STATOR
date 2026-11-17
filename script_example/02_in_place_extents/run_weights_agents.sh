set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_a=${STATOR_GPU_A:-$STATOR_GPU}
gpu_b=${STATOR_GPU_B:-$STATOR_GPU}
repetitions=${REPETITIONS:-6}
agent_counts=${AGENT_COUNTS:-"1 2 4 8"}
n_prompt=${N_PROMPT:-512}
n_gen=${N_GEN:-128}
model=${MODEL:-"$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"}
bin_dir=${BIN_DIR:-"$STATOR_ROOT/engines/weights/build/bin"}
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-engine-agents"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer" >&2
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
for file in "$model" "$bin_dir/llama-bench"; do
  [[ -e "$file" ]] || { echo "missing $file" >&2; exit 2; }
done
model_kb=$(( $(stat -c %s "$model") / 1024 ))
needed_kb=$(( 10 * model_kb + 24 * 1024 * 1024 ))
available_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
(( available_kb > needed_kb )) || {
  echo "not enough available memory for eight device copies" >&2
  exit 2
}

if [[ -n "${MPS_ROOT:-}" ]]; then
  mps_root=$MPS_ROOT
  mkdir -p "$mps_root"
  [[ -z "$(ls -A "$mps_root")" ]] || { echo "MPS_ROOT is not empty" >&2; exit 2; }
else
  mps_root=$(mktemp -d "${TMPDIR:-/tmp}/hm.XXXXXX")
fi
(( ${#mps_root} <= 84 )) || { echo "MPS directory path too long" >&2; exit 2; }
pipe_a="$mps_root/a"
pipe_b="$mps_root/b"
mkdir -p "$pipe_a" "$pipe_b" "$mps_root/la" "$mps_root/lb"
running_a=0
running_b=0
start_mps() {
  if [[ "$1" == a && "$running_a" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_a" CUDA_MPS_PIPE_DIRECTORY="$pipe_a" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/la" nvidia-cuda-mps-control -d
    running_a=1
  elif [[ "$1" == b && "$running_b" -eq 0 ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_b" CUDA_MPS_PIPE_DIRECTORY="$pipe_b" \
      CUDA_MPS_LOG_DIRECTORY="$mps_root/lb" nvidia-cuda-mps-control -d
    running_b=1
  fi
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
cleanup() {
  stop_mps a
  stop_mps b
  rm -rf "$mps_root"
}
trap cleanup EXIT INT TERM
prepare() {
  case $1 in
    gpu | timeslice) stop_mps a; stop_mps b ;;
    mps) start_mps a; stop_mps b ;;
    gpu_mps) start_mps a; start_mps b ;;
  esac
}
placement() {
  local config=$1 index=$2
  case $config in
    gpu) if (( index % 2 == 0 )); then echo "$gpu_a -"; else echo "$gpu_b -"; fi ;;
    timeslice) echo "$gpu_a -" ;;
    mps) echo "$gpu_a $pipe_a" ;;
    gpu_mps)
      if (( index % 2 == 0 )); then echo "$gpu_a $pipe_a"; else echo "$gpu_b $pipe_b"; fi ;;
  esac
}

mkdir -p "$result_dir"
work=$(mktemp -d "$result_dir/work.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'gpu_a=%s\nmig_b=%s\nrepetitions=%s\nagent_counts=%s\n' \
    "$gpu_a" "$gpu_b" "$repetitions" "$agent_counts"
  printf 'n_prompt=%s\nn_gen=%s\n' "$n_prompt" "$n_gen"
  printf 'model=%s\nmodel_bytes=%s\n' "$(basename "$model")" "$(stat -c %s "$model")"
  printf 'engine_commit=%s\n' "$(git -C "$STATOR_ROOT/engines/weights" rev-parse HEAD)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
  sha256sum "$bin_dir/llama-bench"
} >"$result_dir/metadata.txt"

run_case() {
  local run=$1 config=$2 agents=$3 mode=$4 index gpu pipe pids=() failed=0
  local extra=()
  if [[ "$mode" == inplace ]]; then extra=(STATOR_HOST_PTR=1); fi
  local before lowest sample mapped=0
  before=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
  lowest=$before
  for index in $(seq 0 $((agents - 1))); do
    read -r gpu pipe <<<"$(placement "$config" "$index")"
    local mps_env=()
    if [[ "$pipe" != - ]]; then mps_env=(CUDA_MPS_PIPE_DIRECTORY="$pipe"); fi
    env -u CUDA_MPS_PIPE_DIRECTORY CUDA_VISIBLE_DEVICES="$gpu" "${mps_env[@]}" \
      "${extra[@]}" "$bin_dir/llama-bench" -m "$model" -ngl 99 -p "$n_prompt" \
      -n "$n_gen" -r 1 -o jsonl >"$work/agent.$index" 2>/dev/null &
    pids+=($!)
  done
  local smaps=()
  for index in "${pids[@]}"; do smaps+=("/proc/$index/smaps"); done
  while kill -0 "${pids[@]}" 2>/dev/null; do
    sample=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    (( sample < lowest )) && lowest=$sample
    sample=$(cat "${smaps[@]}" 2>/dev/null |
      awk -v file="$model" '
        /^[0-9a-f]+-[0-9a-f]+ / { inside = ($6 == file) }
        inside && /^Pss:/ { pss += $2 }
        END { printf "%d", pss / 1024 }' || true)
    (( ${sample:-0} > mapped )) && mapped=$sample
    sleep 0.5
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || failed=$((failed + 1))
  done
  {
    printf 'BEGIN_ENGINE_AGENTS run=%s config=%s agents=%s mode=%s\n' \
      "$run" "$config" "$agents" "$mode"
    for index in $(seq 0 $((agents - 1))); do
      sed "s/^/AGENT $index /" "$work/agent.$index"
    done
    printf 'MEMORY mem_available_drop_mib=%s model_mapped_mib=%s\n' \
      "$(( (before - lowest) / 1024 ))" "$mapped"
    printf 'END_ENGINE_AGENTS failed_agents=%s\n' "$failed"
  } >>"$raw_log"
}

for run in $(seq 1 "$repetitions"); do
  if (( run % 2 == 1 )); then
    configs=(timeslice mps)
    modes=(copy inplace)
  else
    configs=(mps timeslice)
    modes=(inplace copy)
  fi
  for config in "${configs[@]}"; do
    prepare "$config"
    for agents in $agent_counts; do
      for mode in "${modes[@]}"; do
        run_case "$run" "$config" "$agents" "$mode"
      done
    done
  done
done
stop_mps a
stop_mps b
rm -rf "$work"

awk -f "$script_dir/summarize_weights_agents.awk" "$raw_log" \
  >"$result_dir/engine_agents_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum stator_code/patches/weights_in_place.patch script_example/02_in_place_extents/summarize_weights_agents.awk \
    script_example/02_in_place_extents/run_weights_agents.sh
) >"$result_dir/source_hashes.txt"
printf 'engine_agents_result_dir=%s\n' "$result_dir"
