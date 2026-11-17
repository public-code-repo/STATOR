set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/07_shared_prefix_ratio
D=${RUN_DATE:-$(date +%Y%m%d)}
R=$STATOR_ROOT/results
knob=/sys/kernel/mm/transparent_hugepage/shmem_enabled
old=$(sed 's/.*\[\(.*\)\].*/\1/' "$knob")
restore() { echo "$old" | sudo -n tee "$knob" >/dev/null; }
trap restore EXIT
echo advise | sudo -n tee "$knob" >/dev/null
STATOR_FATTN_HOST_GQA=2 python3 -u "$S/run_prefix_ratio.py" --tag $D-prefix-ratio --engine publish \
  --repetitions 6 --cells '8:2048 8:4096 8:8192 8:12288 8:16321' --modes 'copy extent alone' --tokens 64
python3 "$S/check_prefix_ratio.py" "$R/$D-prefix-ratio"
python3 "$S/summarize_prefix_ratio.py" "$R/$D-prefix-ratio"
restore
trap - EXIT
RESULT_TAG=$D-copy-on-write-bfcl bash "$S/run_copy_on_write_bfcl.sh"
MODE_SET=mechanisms CELLS="same:8:320" RESULT_TAG=$D-linux-sharing-bfcl bash "$S/run_linux_sharing_bfcl.sh"
