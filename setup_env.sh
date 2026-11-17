REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT
export STATOR_ROOT=${STATOR_ROOT:-$REPO_ROOT}
if [[ -z "${STATOR_GPU:-}" ]]; then
  STATOR_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -n 1)
fi
export STATOR_GPU
export STATOR_GPU_A=${STATOR_GPU_A:-$STATOR_GPU}
export STATOR_GPU_B=${STATOR_GPU_B:-$STATOR_GPU}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
_stator_path=$REPO_ROOT/stator_code
for _stator_dir in "$REPO_ROOT"/script_example/*/; do
  _stator_path=$_stator_path:${_stator_dir%/}
done
export PYTHONPATH=$_stator_path${PYTHONPATH:+:$PYTHONPATH}
unset _stator_path _stator_dir
echo "python: $(python3 --version 2>&1)"
echo "code: $REPO_ROOT/stator_code"
echo "data root: $STATOR_ROOT"
echo "gpu: ${STATOR_GPU:-none}"
