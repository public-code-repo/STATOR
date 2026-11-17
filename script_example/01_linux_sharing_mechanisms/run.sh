set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/01_linux_sharing_mechanisms
C=$REPO_ROOT/script_example/common
D=${RUN_DATE:-$(date +%Y%m%d)}
MODE_SET=mechanisms CELLS="same:8:320" RESULT_TAG=$D-linux-sharing-mechanisms bash "$S/run_linux_sharing.sh"
MODE_SET=scaling CELLS="same:1:320 same:4:320 same:8:320 same:16:320 same:32:320" \
  RESULT_TAG=$D-linux-sharing-scaling bash "$S/run_linux_sharing.sh"
MODE_SET=scaling MODES="ro_plain extent" CELLS="same:48:320 same:64:320 same:96:320 same:128:320" \
  REPETITIONS=2 CASE_LIMIT_S=1800 RESULT_TAG=$D-linux-sharing-scaling-large bash "$S/run_linux_sharing.sh"
MODE_SET=scaling MODES="cow_plain" CELLS="same:48:320" \
  REPETITIONS=2 CASE_LIMIT_S=1800 RESULT_TAG=$D-linux-sharing-scaling-large-cow bash "$S/run_linux_sharing.sh"
RESULT_TAG=$D-copy-on-write bash "$S/run_copy_on_write.sh"
MODE_SET=ablation CELLS="same:8:320" RESULT_TAG=$D-ablation bash "$C/run_variants.sh"
RESULT_TAG=$D-page-cache bash "$S/run_page_cache.sh"
RESULT_TAG=$D-mechanism-probe bash "$S/run_mechanism_probe.sh"
RESULT_TAG=$D-seal-probe bash "$S/run_seal_probe.sh"
RESULT_TAG=$D-populate-probe bash "$S/run_populate_probe.sh"
RESULT_TAG=$D-fault-around-probe bash "$S/run_fault_around_probe.sh"
RESULT_TAG=$D-protect-probe bash "$S/run_protect_probe.sh"
RESULT_TAG=$D-mm-cost bash "$S/run_mm_cost.sh"
bash "$S/run_prefetch.sh"
