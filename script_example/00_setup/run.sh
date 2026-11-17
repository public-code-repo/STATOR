set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/00_setup
bash "$S/build_engines.sh"
python3 "$S/build_drivers.py"
bash "$S/prepare_models.sh"
bash "$S/prepare_workloads.sh"
