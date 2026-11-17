import itertools
import os
import subprocess
import sys

import make_workload_bfcl as bfcl


def engine_count(llama_tokenize, model, text, scratch, special=True):
    with open(scratch, "w", encoding="ascii", newline="\n") as handle:
        handle.write(text)
    out = subprocess.run([llama_tokenize, "-m", model, "-f", scratch, "--ids", "--log-disable", "--no-escape"]
                         + ([] if special else ["--no-parse-special"]),
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True,
                         check=True).stdout.strip()
    return out.count(",") + 1 if out.startswith("[") and out != "[]" else 0


def fit(ordered, head, target, count):
    chosen = []
    for item in ordered:
        if count(head + "".join(line + "\n" for _, _, line, _ in chosen + [item])) > target:
            break
        chosen.append(item)
    unused = [item for item in ordered if item not in chosen]
    for drop in (1, 2):
        base = chosen[:-drop]
        for extra in itertools.combinations(unused[:120], drop):
            lines = base + list(extra)
            if count(head + "".join(line + "\n" for _, _, line, _ in lines)) == target:
                return lines
    return None


def main():
    if len(sys.argv) < 7:
        print(__doc__, file=sys.stderr)
        return 2
    src, out_dir, tokenizer_json, llama_tokenize, model = sys.argv[1:6]
    targets = [int(value) for value in sys.argv[6:]]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_json)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    data = bfcl.read_sources(src)
    usable = bfcl.usable_entries(data)
    ordered = [usable[(number * bfcl.STRIDE) % len(usable)] for number in range(len(usable))]
    head = bfcl.SYSTEM + "\n# Tools\n\n"
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(out_dir, ".count.txt")
    for target in targets:
        lines = fit(ordered, head, target, count)
        if lines is None:
            raise SystemExit(f"no set of whole tool lines has {target} tokens")
        prefix = head + "".join(line + "\n" for _, _, line, _ in lines)
        engine = engine_count(llama_tokenize, model, prefix, scratch)
        if engine != target:
            raise SystemExit(f"{target}: the engine counts {engine} tokens")
        source = (f"Source of bfcl_prefix_{target}.txt and bfcl_tasks_{target}.txt "
                  "(written by make_workload_bfcl_tokens.py)\n\n"
                  "dataset: Berkeley Function Calling Leaderboard (BFCL v4), category simple_python\n"
                  f"repository: {bfcl.REPOSITORY}\nrevision: {bfcl.REVISION}\n"
                  f"license: {bfcl.LICENSE} ({bfcl.RAW}LICENSE)\n"
                  f"prefix: {len(prefix)} bytes, {target} tokens of the engine, {len(lines)} functions, "
                  f"{len(lines)} tasks\n\n# Raw files: sha256, bytes, name in SRC_DIR, URL\n\n")
        for name, path, pinned in bfcl.SOURCES:
            source += f"{pinned} {len(data[name])} {name} {bfcl.RAW}{path}\n"
        source += "\n# Tasks: line of the tasks file, dataset entry, function in the prefix\n\n"
        for number, (entry, name, _, _) in enumerate(lines, 1):
            source += f"{number} {entry} {name}\n"
        for name, text in ((f"bfcl_prefix_{target}.txt", prefix),
                           (f"bfcl_tasks_{target}.txt", "".join(request + "\n" for _, _, _, request in lines)),
                           (f"bfcl_source_{target}.txt", source)):
            with open(os.path.join(out_dir, name), "w", encoding="ascii", newline="\n") as handle:
                handle.write(text)
        print(f"{target} tokens: {len(prefix)} bytes, {len(lines)} functions")
    os.remove(scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
