set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repetitions=${REPETITIONS:-6}
modes=${MODES:-"vllm cpu lmcache stator_weights stator_kv stator"}
agents=${AGENTS:-8}
servers=${SERVERS:-2}
prefix_file=${PREFIX_FILE:-"$STATOR_ROOT/workloads/agent_prefix.txt"}
tasks_file=${TASKS_FILE:-"$STATOR_ROOT/workloads/agent_tasks.txt"}
context=${CONTEXT:-16384}
n_gen=${N_GEN:-64}
kv_cache_gib=${KV_CACHE_GIB:-2}
offload_gib=${OFFLOAD_GIB:-$kv_cache_gib}
port=${PORT:-8210}
start_limit_s=${START_LIMIT_S:-900}
first_gpu=${STATOR_GPU_A:-$STATOR_GPU}
second_gpu=${STATOR_GPU_B:-$STATOR_GPU}
venv=${VLLM_VENV:-"$STATOR_ROOT/ext/venv-vllm"}
model_dir=${VLLM_MODEL:-"$(ls -d "$STATOR_ROOT"/ext/hf/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/* | head -n 1)"}
extra_args=${VLLM_ARGS:-"-O0 --enable-prompt-tokens-details --gpu-memory-utilization 0.4"}
cuda_include=${CUDA_INCLUDE:-/usr/local/cuda-13.0/include}
lmcache_site=${LMCACHE_SITE:-"$STATOR_ROOT/ext/lmcache-site"}
lmcache_port=${LMCACHE_PORT:-8290}
lmcache_staging_gib=${LMCACHE_STAGING_GIB:-1}
model_link=${VLLM_MODEL_LINK:-"$STATOR_ROOT/ext/hf/qwen-awq"}
mount_dir=${SHARE_MOUNT:-"$STATOR_ROOT/vllm_share_mount"}
mount_mib=${SHARE_MOUNT_MIB:-$(( kv_cache_gib * 1024 + 8192 ))}
client="$script_dir/vllm_diagnostic_client.py"
result_root=${RESULT_ROOT:-"$STATOR_ROOT/results"}
result_tag=${RESULT_TAG:-"$(date +%Y%m%d-%H%M%S)-vllm-share"}
result_dir="$result_root/$result_tag"
raw_log="$result_dir/raw.log"
connector='{"kv_connector":"StatorConnector","kv_connector_module_path":"stator_vllm.connector","kv_role":"kv_both"}'

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || (( repetitions % 2 != 0 )); then
  echo "REPETITIONS must be a positive even integer" >&2
  exit 2
fi
if (( agents < 2 || agents % 2 != 0 || servers < 2 )); then
  echo "AGENTS must be even and SERVERS at least 2" >&2
  exit 2
fi
if [[ ! -x "$venv/bin/vllm" || ! -d "$model_dir" ]]; then
  echo "missing the vLLM environment or the model; see ext/FEASIBILITY.md" >&2
  exit 2
fi
if ! "$venv/bin/python" -c 'import stator_vllm' 2>/dev/null; then
  echo "the plugin is not installed: $venv/bin/pip install --no-deps -e vllm_stator" >&2
  exit 2
fi
if [[ " $modes " == *" lmcache "* && ! -d "$lmcache_site" ]]; then
  echo "mode lmcache needs LMCache in $lmcache_site; see ext/FEASIBILITY.md" >&2
  exit 2
fi
if [[ -e "$result_dir" ]]; then
  echo "result directory already exists: $result_dir" >&2
  exit 2
fi

ln -sfn "$model_dir" "$model_link"

pids=()
stop_servers() {
  local pid
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    local alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    (( alive )) || break
    sleep 0.5
  done
  for pid in "${pids[@]}"; do
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  pids=()
}
cleanup() {
  rm -f "$result_dir/sampling" "$result_dir/lowest"
  stop_servers
  if mountpoint -q "$mount_dir"; then sudo -n umount "$mount_dir" || true; fi
  rmdir "$mount_dir" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$result_dir"
source "$script_dir/env_torch.sh" "$venv"
export PYTHONPATH="$REPO_ROOT/stator_code/vllm/diagnostic${PYTHONPATH:+:$PYTHONPATH}"
export STATOR_BYTE_AUDIT_DIR="$result_dir/byte_audit"

if [[ " $modes " == *" stator"* ]]; then
  mkdir -p "$mount_dir"
  mountpoint -q "$mount_dir" || sudo -n mount -t tmpfs \
    -o "huge=always,size=${mount_mib}m,uid=$(id -u),gid=$(id -g)" tmpfs "$mount_dir"
fi

{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'repetitions=%s\nmodes=%s\nservers=%s\nagents=%s\ncontext=%s\nn_gen=%s\n' \
    "$repetitions" "$modes" "$servers" "$agents" "$context" "$n_gen"
  printf 'kv_cache_gib=%s\noffload_gib=%s\nlmcache_staging_gib=%s\n' "$kv_cache_gib" "$offload_gib" \
    "$lmcache_staging_gib"
  if [[ -f "$lmcache_site/lmcache/_version.py" ]]; then
    printf 'lmcache=%s\n' "$(awk -F"'" '/^__version__ = version/ { print $2 }' "$lmcache_site/lmcache/_version.py")"
  fi
  printf 'first_gpu=%s\nsecond_gpu=%s\n' "$first_gpu" "$second_gpu"
  printf 'model_dir=%s\nmodel_bytes=%s\n' "$(basename "$(dirname "$(dirname "$model_dir")")")" \
    "$(du -sbL "$model_dir" | cut -f1)"
  for package in vllm torch stator-vllm; do
    printf '%s=%s\n' "$package" \
      "$("$venv/bin/python" -c "import importlib.metadata as m; print(m.version('$package'))")"
  done
  printf 'extra_args=%s\n' "$extra_args"
  printf 'shmem_huge=%s\n' "$(cat /sys/kernel/mm/transparent_hugepage/shmem_enabled)"
  sed 's/^/boot_id=/' /proc/sys/kernel/random/boot_id
  nvidia-smi -L
} >"$result_dir/metadata.txt"

meminfo_kib() { awk -v name="$1:" '$1 == name { print $2 }' /proc/meminfo; }
ports_free() {
  local busy
  for _ in $(seq 1 60); do
    busy=$(ss -ltn 2>/dev/null | grep -cE ":($(seq -s '|' "$port" "$(( port + servers - 1 ))")|$lmcache_port) " || true)
    (( busy == 0 )) && return 0
    sleep 0.5
  done
  return 1
}

serve() {
  local slot=$1 device=$2 server_port=$3 log=$4
  shift 4
  local -a assignments=()
  while [[ $1 != -- ]]; do assignments+=("$1"); shift; done
  shift
  env CUDA_VISIBLE_DEVICES="$device" HF_HUB_OFFLINE=1 HF_HOME="$STATOR_ROOT/ext/hf" PYTHONHASHSEED=0 \
    TRITON_CUDART_PATH="$cuda_include" TRITON_CUDACRT_PATH="$cuda_include" "${assignments[@]}" \
    "$venv/bin/python" -m stator_vllm.serve "$model_link" --port "$server_port" \
    --served-model-name stator-share \
    --max-model-len "$context" --max-num-seqs "$agents" \
    --kv-cache-memory-bytes "$(( kv_cache_gib * 1073741824 ))" $extra_args "$@" >"$log" 2>&1 &
  pids+=($!)
  local started ready=0
  started=$(date +%s%N)
  while kill -0 "${pids[-1]}" 2>/dev/null && (( ($(date +%s%N) - started) / 1000000000 < start_limit_s )); do
    if python3 "$client" wait "$server_port" 1; then ready=1; break; fi
  done
  printf 'SERVER slot=%s ready=%s alive=%s startup_ms=%s drop_mib=%s\n' "$slot" "$ready" \
    "$(kill -0 "${pids[-1]}" 2>/dev/null && echo 1 || echo 0)" \
    "$(( ($(date +%s%N) - started) / 1000000 ))" \
    "$(( (before - $(meminfo_kib MemAvailable)) / 1024 ))" >>"$raw_log"
  (( ready ))
}

sample_lowest() {
  local low=$before sample
  while [[ -e "$result_dir/sampling" ]]; do
    sample=$(meminfo_kib MemAvailable)
    if (( sample < low )); then low=$sample; printf '%s\n' "$low" >"$result_dir/lowest"; fi
    sleep 0.05
  done
}

round() {
  { printf 'ROUND name=%s\n' "$1"
    STATOR_DIAG_DIR="$result_dir/requests/$run-$mode/$1" STATOR_DIAG_SALT="${STATOR_DIAG_SALT:-}" python3 "$client" agents stator-share "$prefix_file" "$tasks_file" "$n_gen" "$2" "$3" "$4" || true
  } >>"$raw_log"
}

read -r -a mode_list <<<"$modes"
for run in $(seq 1 "$repetitions"); do
  printf 'BEGIN_VSHARE run=%s\n' "$run" >>"$raw_log"
  for position in $(seq 0 $(( ${#mode_list[@]} - 1 ))); do
    mode=${mode_list[$(( (position + run - 1) % ${#mode_list[@]} ))]}
    env1=() env2=() arguments=()
    case $mode in
      vllm) ;;
      cpu) arguments=(--kv-offloading-size "$offload_gib" --kv-offloading-backend native
                      --disable-hybrid-kv-cache-manager) ;;
      lmcache)
        env1=(PYTHONPATH="$lmcache_site" LMCACHE_CONFIG_FILE="$result_dir/lmcache.yaml" LMCACHE_TRACK_USAGE=false)
        env2=("${env1[@]}")
        arguments=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}') ;;
      stator_weights)
        env1=(STATOR_WEIGHTS_PUBLISH="$mount_dir/weights")
        env2=(STATOR_WEIGHTS_ATTACH="$mount_dir/weights") ;;
      stator_kv)
        env1=(STATOR_KV_PUBLISH="$mount_dir/kv")
        env2=(STATOR_KV_ATTACH="$mount_dir/kv")
        arguments=(--kv-transfer-config "$connector") ;;
      stator)
        env1=(STATOR_WEIGHTS_PUBLISH="$mount_dir/weights" STATOR_KV_PUBLISH="$mount_dir/kv")
        env2=(STATOR_WEIGHTS_ATTACH="$mount_dir/weights" STATOR_KV_ATTACH="$mount_dir/kv")
        arguments=(--kv-transfer-config "$connector") ;;
      *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    ports_free || { echo "the ports of the servers are in use" >&2; exit 1; }
    sync
    before=$(meminfo_kib MemAvailable)
    printf '%s\n' "$before" >"$result_dir/lowest"
    : >"$result_dir/sampling"
    sample_lowest &
    sampler=$!
    failed=0
    printf 'CASE mode=%s servers=%s kv_cache_gib=%s agents=%s\n' "$mode" "$servers" "$kv_cache_gib" \
      "$agents" >>"$raw_log"
    if [[ $mode == lmcache ]]; then
      printf '%s\n' 'chunk_size: 256' 'local_cpu: false' "max_local_cpu_size: $lmcache_staging_gib" \
        "remote_url: \"lm://127.0.0.1:$lmcache_port\"" 'remote_serde: "naive"' \
        'save_decode_cache: false' 'save_unfull_chunk: false' >"$result_dir/lmcache.yaml"
      env CUDA_VISIBLE_DEVICES="" LMCACHE_TRACK_USAGE=false PYTHONPATH="$lmcache_site" \
        "$venv/bin/python" -m lmcache.v1.server 127.0.0.1 "$lmcache_port" cpu \
        >"$result_dir/lmcache.$run.log" 2>&1 &
      pids+=($!)
      sleep 5
    fi
    all_ports=$port
    if serve 1 "$first_gpu" "$port" "$result_dir/server1.$mode.$run.log" "${env1[@]}" -- "${arguments[@]}"; then
      STATOR_DIAG_DIR="$result_dir/requests/$run-$mode/prefix" python3 "$client" prefix stator-share "$prefix_file" "$port" >>"$raw_log"
      sleep 2
      for slot in $(seq 2 "$servers"); do
        device=$second_gpu
        (( slot % 2 == 1 )) && device=$first_gpu
        if serve "$slot" "$device" "$(( port + slot - 1 ))" "$result_dir/server$slot.$mode.$run.log" \
            "${env2[@]}" -- "${arguments[@]}"; then
          all_ports+=",$(( port + slot - 1 ))"
        else
          failed=1
          tail -n 20 "$result_dir/server$slot.$mode.$run.log" | sed "s/^/SERVER_LOG slot=$slot /" >>"$raw_log"
          break
        fi
      done
      if (( ! failed )); then
        printf 'MEMORY at=ready drop_mib=%s\n' "$(( (before - $(meminfo_kib MemAvailable)) / 1024 ))" >>"$raw_log"
        round first "$(( agents / 2 ))" "$(( port + 1 ))" 0
        for slot in $(seq 3 "$servers"); do
          round "warm$slot" 1 "$(( port + slot - 1 ))" 29
        done
        round both "$(( agents / 2 * servers ))" "$all_ports" "$agents"
        round both_self "$(( agents / 2 * servers ))" "$all_ports" "$agents"
        round exact9 1 "$(( port + 1 ))" 17
        round exact11 1 "$(( port + servers - 1 ))" 19
        STATOR_DIAG_SALT="stator-recompute-$run" round recompute "$(( agents / 2 * servers ))" "$all_ports" "$agents"
        round alone1 1 "$port" 30
        round alone2 1 "$(( port + 1 ))" 31
        sleep 1
        files_mib=0
        mountpoint -q "$mount_dir" && files_mib=$(( $(du -sk "$mount_dir" | cut -f1) / 1024 ))
        printf 'MEMORY at=end drop_mib=%s peak_drop_mib=%s files_mib=%s\n' \
          "$(( (before - $(meminfo_kib MemAvailable)) / 1024 ))" \
          "$(( (before - $(cat "$result_dir/lowest")) / 1024 ))" "$files_mib" >>"$raw_log"
      fi
    else
      failed=1
      tail -n 20 "$result_dir/server1.$mode.$run.log" | sed 's/^/SERVER_LOG slot=1 /' >>"$raw_log"
    fi
    for slot in $(seq 1 "$servers"); do
      log="$result_dir/server$slot.$mode.$run.log"
      [[ -f $log ]] || continue
      grep -o 'stator: .*' "$log" | sed "s/^stator: /PLUGIN slot=$slot /" >>"$raw_log" || true
      grep -oE '(Stored|Retrieved) [0-9]+ out of (total )?[0-9]+ (required )?tokens.*' "$log" |
        sed -E "s/\x1b\[[0-9;]*m//g; s/^/LMCACHE slot=$slot /" >>"$raw_log" || true
    done
    rm -f "$result_dir/sampling"
    wait "$sampler" 2>/dev/null || true
    rm -f "$result_dir/lowest"
    stop_servers
    rm -f "$mount_dir"/weights "$mount_dir"/weights.json "$mount_dir"/kv "$mount_dir"/kv.json 2>/dev/null || true
    printf 'END_CASE failed=%s\n' "$failed" >>"$raw_log"
    sleep 3
  done
done

awk -f "$script_dir/summarize_vllm_share.awk" "$raw_log" >"$result_dir/vllm_share_summary.csv"
(
  cd "$REPO_ROOT"
  sha256sum script_example/06_batching_server/openai_client.py script_example/06_batching_server/vllm_share_client.py script_example/06_batching_server/summarize_vllm_share.awk \
    stator_code/vllm/stator_vllm/pyproject.toml stator_code/vllm/stator_vllm/stator_vllm/__init__.py stator_code/vllm/stator_vllm/stator_vllm/connector.py \
    stator_code/vllm/stator_vllm/stator_vllm/serve.py
) >"$result_dir/source_hashes.txt"
printf 'vllm_share_result_dir=%s\n' "$result_dir"
