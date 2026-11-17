set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
upstream=${LLAMA_CPP_URL:-https://github.com/ggml-org/llama.cpp}
commit=6f767fe960c3b97cf37fac4626c86400561ca1e4
patches=$REPO_ROOT/stator_code/patches
declare -A patch=(
  [stock]=
  [weights]=weights_in_place
  [kv_only]=kv_extents_only
  [extents]=stator_extents
  [chains]=extent_chains
  [variants]=stator_variants
  [variants_gqa]=stator_variants_gqa
  [publish]=stator_publish
  [publish_repeat]=stator_repeat_publish
  [linux]=linux_baselines
)
engines=${ENGINES:-"stock weights kv_only extents chains variants variants_gqa publish publish_repeat linux"}
mkdir -p engines
if [[ ! -d engines/upstream ]]; then
  git clone --quiet "$upstream" engines/upstream
fi
git -C engines/upstream cat-file -e "$commit^{commit}"
for name in $engines; do
  [[ -v "patch[$name]" ]] || { echo "unknown engine $name" >&2; exit 2; }
  dir=engines/$name
  if [[ ! -d "$dir" ]]; then
    git clone --quiet --no-checkout engines/upstream "$dir"
    git -C "$dir" checkout --quiet "$commit"
    if [[ -n "${patch[$name]}" ]]; then
      git -C "$dir" apply "$patches/${patch[$name]}.patch"
    fi
  fi
  targets=(llama)
  case $name in
    stock) targets+=(llama-tokenize) ;;
    weights) targets+=(llama-bench llama-completion) ;;
  esac
  cmake -S "$dir" -B "$dir/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=110 -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$dir/build" --config Release -j "${JOBS:-$(nproc)}" --target "${targets[@]}"
  echo "engine $name: $dir/build/bin"
done
