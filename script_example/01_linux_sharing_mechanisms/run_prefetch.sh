set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scratch="$STATOR_ROOT/results"
parameter=/sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable
result_dir="$scratch/prefetch-$(date +%Y%m%d-%H%M%S)"
mkdir "$result_dir"
exec > >(tee "$result_dir/control.log") 2>&1
[[ "$(cat "$parameter")" == 1 ]]
restore_needed=0
restore() {
  local saved=$?
  trap - EXIT INT TERM
  if (( restore_needed )); then
    echo 'Restoring nvidia_uvm with uvm_perf_prefetch_enable=1'
    if [[ -d /sys/module/nvidia_uvm ]]; then
      sudo -n modprobe -r nvidia_uvm || { echo 'RESTORE FAILED: unload'; exit 43; }
    fi
    sudo -n modprobe nvidia_uvm uvm_perf_prefetch_enable=1 || {
      echo 'RESTORE FAILED: load'; exit 43;
    }
    [[ "$(cat "$parameter")" == 1 ]] || { echo 'RESTORE FAILED: parameter'; exit 43; }
    "$STATOR_ROOT/bin/access_flag_probe" 2 none readonly >"$result_dir/restore-cuda-check.out" || {
      echo 'RESTORE FAILED: CUDA check'; exit 43;
    }
    grep -q 'sums_ok=1' "$result_dir/restore-cuda-check.out" || {
      echo 'RESTORE FAILED: CUDA checksum'; exit 43;
    }
  fi
  exit "$saved"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo 'Loading nvidia_uvm with uvm_perf_prefetch_enable=0'
sudo -n modprobe -r nvidia_uvm || {
  echo 'MODULE RELOAD BLOCKED: stopping; no probes started'
  exit 42
}
restore_needed=1
sudo -n modprobe nvidia_uvm uvm_perf_prefetch_enable=0
[[ "$(cat "$parameter")" == 0 ]]
python3 "$script_dir/prefetch_probes.py" "$result_dir/off" --expected-prefetch 0
echo 'Restoring nvidia_uvm with uvm_perf_prefetch_enable=1'
sudo -n modprobe -r nvidia_uvm
sudo -n modprobe nvidia_uvm uvm_perf_prefetch_enable=1
[[ "$(cat "$parameter")" == 1 ]]
restore_needed=0
python3 "$script_dir/prefetch_probes.py" "$result_dir/on" --expected-prefetch 1
echo "prefetch_result_dir=$result_dir"
