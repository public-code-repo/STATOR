set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
index=https://pypi.jetson-ai-lab.io/sbsa/cu130
venv=$STATOR_ROOT/ext/venv-vllm
mkdir -p ext/wheels ext/lmcache-dl ext/lmcache-site
python3 -m venv "$venv"
pip=("$venv/bin/pip")
"${pip[@]}" install --upgrade pip
"${pip[@]}" download --no-deps --only-binary :all: --index-url "$index" -d ext/wheels vllm==0.20.0+cu130 torch==2.10.0
"${pip[@]}" download --no-deps --only-binary :all: -d ext/lmcache-dl lmcache==0.5.5
sha256sum --check <<'WHEELS'
cc2c3edee7c0897cd8b82d78086b107843aca92cb1f4fb534788bf0b4671a1f6  ext/wheels/vllm-0.20.0+cu130-cp312-cp312-linux_aarch64.whl
c597ddc0c1f8de6c54fc7c7c09bfe5f70e04a5024383740e99a02eb13498ced5  ext/wheels/torch-2.10.0-cp312-cp312-linux_aarch64.whl
8f255aacb43e2126145bd4ded51b8e26320ddd313634b1b5bbee49273efd08c7  ext/lmcache-dl/lmcache-0.5.5-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl
WHEELS
"${pip[@]}" install numpy filelock 'typing-extensions>=4.10.0' 'setuptools<81.0.0,>=77.0.3' 'sympy>=1.13.3' \
  'networkx>=2.5.1' jinja2 'fsspec>=0.8.5' nvidia-cudss-cu13 nvpl-blas nvpl-lapack
"${pip[@]}" install --no-deps --only-binary :all: --index-url "$index" \
  nvidia-cudnn-frontend==1.14.1 flashinfer-python==0.6.8 flashinfer-cubin==0.6.8 cuda-bindings==13.0.1 cuda-python==13.0.1
"${pip[@]}" install --no-deps --only-binary :all: fastsafetensors==0.4.0
"${pip[@]}" install --only-binary :all: --uploaded-prior-to 2026-05-15T00:00:00Z \
  ext/wheels/vllm-0.20.0+cu130-cp312-cp312-linux_aarch64.whl
"${pip[@]}" install --no-deps --force-reinstall ext/wheels/torch-2.10.0-cp312-cp312-linux_aarch64.whl
"${pip[@]}" install --no-deps --no-build-isolation -e "$REPO_ROOT/stator_code/vllm/stator_vllm"
"${pip[@]}" install --no-deps --no-build-isolation -e "$REPO_ROOT/stator_code/vllm/stator_vllm_direct"
python3 -m zipfile -e ext/lmcache-dl/lmcache-0.5.5-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl ext/lmcache-site/
"${pip[@]}" install --no-deps --target ext/lmcache-site sortedcontainers aiofile caio
