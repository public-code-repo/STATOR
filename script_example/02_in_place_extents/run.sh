set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/02_in_place_extents
C=$REPO_ROOT/script_example/common
D=${RUN_DATE:-$(date +%Y%m%d)}
LLAMA8B=$STATOR_ROOT/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
MPS_ROOT=${TMPDIR:-/tmp}/stator-mps RESULT_TAG=$D-weights-agents bash "$S/run_weights_agents.sh"
RESULT_TAG=$D-weights-pages bash "$S/run_weights_pages.sh"
RESULT_TAG=$D-kv-extents bash "$S/run_kv_extents.sh"
MODE_SET=locality N_GEN=128 CELLS="same:4:320" RESULT_TAG=$D-locality bash "$C/run_variants.sh"
MODE_SET=locality N_GEN=128 CELLS="same:4:320" MODEL=$LLAMA8B RESULT_TAG=$D-locality-llama8b bash "$C/run_variants.sh"
VARIANT_ENGINE=$STATOR_ROOT/engines/variants_gqa PARENT_DRIVER=$STATOR_ROOT/bin/publisher_variants_gqa \
  CHILD_DRIVER=$STATOR_ROOT/bin/prefix_agent_variants_gqa STATOR_FATTN_HOST_GQA=2 \
  MODE_SET=locality N_GEN=128 CELLS="same:4:320" MODEL=$LLAMA8B RESULT_TAG=$D-locality-llama8b-gqa bash "$C/run_variants.sh"
RESULT_TAG=$D-tlb-probe bash "$S/run_tlb_probe.sh"
