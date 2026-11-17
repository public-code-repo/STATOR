set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
mkdir -p models
while read -r repository file checksum; do
  path=models/$file
  if [[ ! -s "$path" ]]; then
    curl -fL --retry 5 --retry-delay 10 -sS -o "$path.part" "https://huggingface.co/$repository/resolve/main/$file"
    mv "$path.part" "$path"
  fi
  echo "$checksum  $path" | sha256sum --check
done <<'MODELS'
bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf 65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423
bartowski/Qwen2.5-14B-Instruct-GGUF Qwen2.5-14B-Instruct-Q4_K_M.gguf e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008
bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf 7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c
MODELS
HF_HOME="$STATOR_ROOT/ext/hf" python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ', revision='b25037543e9394b818fdfca67ab2a00ecc7dd641'))"
