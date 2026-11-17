set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/05_agents_over_time
D=${RUN_DATE:-$(date +%Y%m%d)}
QWEN7B=$STATOR_ROOT/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
python3 -u "$S/run_traces.py" waits_60_120_240 --execute --resume
python3 -u "$S/run_traces.py" waits_480_960 --execute --resume
python3 -u "$S/run_traces.py" existing_60_240 --execute --resume
python3 -u "$S/run_traces.py" existing_120 --execute --resume
M=$STATOR_ROOT/mounts_model
unmount() { rm -f "$M/model.gguf"; sudo -n umount "$M" 2>/dev/null || true; rmdir "$M" 2>/dev/null || true; }
trap unmount EXIT
mkdir -p "$M"
sudo -n mount -t tmpfs -o "huge=always,size=6144m,uid=$(id -u),gid=$(id -g)" tmpfs "$M"
dd if="$QWEN7B" of="$M/model.gguf" bs=64M iflag=direct status=none
MODEL=$M/model.gguf python3 -u "$S/run_traces.py" arrivals_1440 --execute --resume
unmount
trap - EXIT
REPS=6 SYSTEMS="both weights none" RESULT_TAG=$D-restart python3 -u "$S/run_restart.py"
