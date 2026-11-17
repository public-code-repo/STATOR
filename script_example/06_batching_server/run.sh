set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/06_batching_server
C=$REPO_ROOT/script_example/common
D=${RUN_DATE:-$(date +%Y%m%d)}
R=$STATOR_ROOT/results
RESULT_TAG=$D-vllm-share bash "$S/run_vllm_share.sh"
SERVERS=4 MODES="vllm lmcache stator" REPETITIONS=4 RESULT_TAG=$D-vllm-share-4 bash "$S/run_vllm_share.sh"
REPETITIONS=2 SERVERS=4 AGENTS=8 MODES='vllm stator_weights stator' KV_CACHE_GIB=6 \
  RESULT_TAG=$D-vllm-diagnostic SHARE_MOUNT=$STATOR_ROOT/vllm_byte_mount PORT=8310 LMCACHE_PORT=8390 \
  python3 -u "$C/guarded_command.py" --report "$R/$D-vllm-diagnostic-guard.json" --limit-s 7200 -- \
  bash "$S/run_vllm_diagnostic.sh"
python3 "$S/summarize_vllm_diagnostic.py" "$R/$D-vllm-diagnostic"
python3 "$S/analyze_vllm_diagnostic.py" "$R/$D-vllm-diagnostic"
