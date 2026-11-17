import os
import subprocess
import sys

import make_workload_bfcl_tokens as bfcl_tokens
bfcl = bfcl_tokens.bfcl


def corpus(engine):
    paths = sorted(subprocess.run(["git", "-C", engine, "ls-files", "*.md"], capture_output=True, text=True,
                                  check=True).stdout.split())
    commit = subprocess.run(["git", "-C", engine, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    parts = []
    for path in paths:
        text = subprocess.run(["git", "-C", engine, "show", f"HEAD:{path}"], capture_output=True,
                              check=True).stdout.decode("utf-8", "replace")
        text = "".join(c for c in text if c.isascii() and (c.isprintable() or c in "\n\t"))
        text = text.replace("<|", "<").replace("|>", ">")
        parts.append(f"## {path}\n\n{text.strip()}\n\n")
    return commit, paths, "".join(parts)


def private(tokenizer, ids, start, agent, request, need):
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    head = f"\n\n# Reference documents of agent {agent}\n\n"
    tail = f"\n\nUser request for agent {agent}: {request}\nCall:\n"
    for shift in range(0, 16):
        size = need - count(head) - count(tail)
        for _ in range(16):
            documents = tokenizer.decode(ids[start + shift:start + shift + size])
            found = count(head + documents + tail)
            if found == need:
                return head + documents + tail
            if abs(found - need) <= 2:
                break
            size += need - found
        for near in sorted(range(size - 12, size + 13), key=lambda value: abs(value - size)):
            documents = tokenizer.decode(ids[start + shift:start + shift + near])
            for cut in range(0, 41):
                text = head + documents[:len(documents) - cut] + tail
                if count(text) == need:
                    return text
    raise SystemExit(f"agent {agent}: no slice of the documents gives {need} tokens")


def main():
    if len(sys.argv) < 10:
        print(__doc__, file=sys.stderr)
        return 2
    src, engine, out_dir, tokenizer_json, llama_tokenize, model = sys.argv[1:7]
    total, agents = int(sys.argv[7]), int(sys.argv[8])
    shares = [int(value) for value in sys.argv[9:]]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_json)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    data = bfcl.read_sources(src)
    usable = bfcl.usable_entries(data)
    ordered = [usable[(number * bfcl.STRIDE) % len(usable)] for number in range(len(usable))]
    head = bfcl.SYSTEM + "\n# Tools\n\n"
    commit, paths, text = corpus(engine)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    stride = total - min(shares)
    if agents * stride + 64 > len(ids):
        raise SystemExit(f"the documents have {len(ids)} tokens, fewer than {agents} slices of {stride}")
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(out_dir, ".count.txt")
    for shared in shares:
        lines = bfcl_tokens.fit(ordered, head, shared, count)
        if lines is None:
            raise SystemExit(f"no set of whole tool lines has {shared} tokens")
        prefix = head + "".join(line + "\n" for _, _, line, _ in lines)
        if bfcl_tokens.engine_count(llama_tokenize, model, prefix, scratch) != shared:
            raise SystemExit(f"{shared}: the engine counts another number of tokens")
        task_dir = os.path.join(out_dir, f"tasks_{shared}")
        os.makedirs(task_dir, exist_ok=True)
        source = (f"Shared-prefix ratio workload (make_workload_ratio.py): {total} tokens per agent, "
                  f"{shared} shared, {total - shared} private, {agents} agents\n\n"
                  f"shared: BFCL v4 simple_python ({bfcl.REPOSITORY} at {bfcl.REVISION}, {bfcl.LICENSE}), "
                  f"system prompt and {len(lines)} functions\n"
                  f"private: Markdown files of {engine} at {commit} ({len(paths)} files, {len(ids)} tokens), "
                  f"agent i from token i * {stride}, and a BFCL request\n\n")
        for agent in range(agents):
            entry, name, _, request = lines[agent % len(lines)]
            if total - shared < 128:
                text_agent = f"\n\nUser request for agent {agent}: {request}\nCall:\n"
            else:
                text_agent = private(tokenizer, ids, agent * stride, agent, request, total - shared)
                if bfcl_tokens.engine_count(llama_tokenize, model, text_agent, scratch, special=False) != total - shared:
                    raise SystemExit(f"{shared}, agent {agent}: the engine counts another number of tokens")
            with open(os.path.join(task_dir, f"task.{agent}.txt"), "w", encoding="ascii", newline="\n") as handle:
                handle.write(text_agent)
            source += f"agent {agent}: request of entry {entry} ({name})\n"
        with open(os.path.join(out_dir, f"prefix_{shared}.txt"), "w", encoding="ascii", newline="\n") as handle:
            handle.write(prefix)
        with open(os.path.join(out_dir, f"source_{shared}.txt"), "w", encoding="ascii", newline="\n") as handle:
            handle.write(source)
        print(f"shared {shared}: {len(lines)} functions, private {total - shared} tokens per agent")
    os.remove(scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
