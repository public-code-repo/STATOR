_venv=${1:-$(dirname "${BASH_SOURCE[0]}")/venv-torch}
_venv=$(cd "$_venv" && pwd)
_sp=$_venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$_sp/nvpl/lib:$_sp/nvidia/cu13/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}
export PY=$_venv/bin/python
unset _venv _sp
