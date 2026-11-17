set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/00_setup
commit=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
source_dir=workloads/src_bfcl
tokenize=engines/stock/build/bin/llama-tokenize
model=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
tokenizer=${TOKENIZER_JSON:-$(find "$STATOR_ROOT/ext/hf/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots" -name tokenizer.json 2>/dev/null | head -n 1)}
mkdir -p "$source_dir/possible_answer"
base=https://raw.githubusercontent.com/ShishirPatil/gorilla/$commit
for path in LICENSE \
    berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json \
    berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json; do
  case $path in
    LICENSE) out=$source_dir/LICENSE ;;
    */possible_answer/*) out=$source_dir/possible_answer/${path##*/} ;;
    *) out=$source_dir/${path##*/} ;;
  esac
  [[ -s "$out" ]] || curl -fL --retry 5 -sS -o "$out" "$base/$path"
done
python3 "$S/make_workload.py" engines/stock workloads
python3 "$S/make_workload_bfcl.py" "$source_dir" workloads
[[ -n "$tokenizer" ]] || { echo "tokenizer.json of Qwen2.5-7B-Instruct not found; set TOKENIZER_JSON" >&2; exit 2; }
python3 "$S/make_workload_bfcl_tokens.py" "$source_dir" workloads/bfcl_len "$tokenizer" "$tokenize" "$model" 4081 16321
python3 "$S/make_workload_ratio.py" "$source_dir" engines/publish workloads/ratio "$tokenizer" "$tokenize" "$model" \
  16384 8 2048 4096 8192 12288 16321
