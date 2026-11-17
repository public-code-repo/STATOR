import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
from trace_read import load_trace, generation
from trace_estimate import estimate

def duration(events, begin, end, required=True):
    starts = [e["mono_ns"] for e in events if e["event"] == begin]
    ends = [e["mono_ns"] for e in events if e["event"] == end]
    if len(starts) != len(ends) or any(e < b for b, e in zip(starts, ends)) or required and not starts:
        raise ValueError(f"missing/unpaired phase {begin}/{end}")
    return sum(e - b for b, e in zip(starts, ends)) / 1e6

def summarize(source, output):
    meta = json.loads((source / "metadata.json").read_text())["arguments"]
    validation = json.loads((source / "validation.json").read_text())
    cases = json.loads((source / "cases.json").read_text())
    if meta["scenario"] != "tree" or not validation["valid"] or not validation["all_requested_cases_executed"]:
        raise ValueError("tree run incomplete/invalid; inspect failures and skips first")
    expected = {(rep, mode, tuple(int(n) for n in cell.split(",")))
                for rep in range(1, meta["repetitions"] + 1)
                for mode in meta["modes"].split() for cell in meta["fanouts"].split()}
    keys = [(r["run"], r["mode"], tuple(r["cell"])) for r in cases]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("metadata/case inventory mismatch")
    reference = {(r["run"], tuple(r["cell"])): r for r in cases if r["mode"] == "copy"}
    rows, comparisons = [], []
    for case in cases:
        name = case["case"]
        ref = reference[case["run"], tuple(case["cell"])]
        expected_count = 1 + sum(math.prod(case["cell"][:i]) for i in range(1, len(case["cell"]) + 1))
        if case["status"] != "ok" or case["count"] != expected_count or len(case["processes"]) != expected_count:
            raise ValueError(f"{name}: process/status mismatch")
        if case["tmpfs_after_unlink_bytes"] or case["gpu_faults_total"] is None or case["memory_min_available"] < meta["guard_gib"] * 2**30:
            raise ValueError(f"{name}: reclamation/fault-count/guard check failed")
        traces = load_trace(source / name / "timing")
        if set(traces) != {p["pid"] for p in case["processes"].values()}:
            raise ValueError(f"{name}: timing PID inventory differs")
        leaves, internal, generations = [], [], []
        for process, p in case["processes"].items():
            if p["exit"] != 0:
                raise ValueError(f"{name}/{process}: nonzero exit")
            events = traces[p["pid"]]
            observed = generation(events, meta["tokens"])
            if observed != case["generations"][process]:
                raise ValueError(f"{name}/{process}: saved generation differs from trace")
            generations.append(observed)
            text_lines = [line for line in (source / name / (process + ".out")).read_text().splitlines() if line.startswith("TEXT ")]
            digest = hashlib.sha256("\n".join(text_lines).encode()).hexdigest()
            if len(text_lines) != 1 or digest != p["text_sha256"]:
                raise ValueError(f"{name}/{process}: stdout text differs from record")
            comparisons.append({"case": name, "process": process, "self": case is ref,
                                "text_match": digest == ref["processes"][process]["text_sha256"],
                                "token_match": observed["tokens"] == ref["generations"][process]["tokens"]})
            commands = p["command"][4:]
            phases = {key + "_ms": duration(events, key + "_start", key + "_end", required=(key != "attach" or any(s.startswith("load:") for s in commands)))
                      for key in ("backend", "model", "context", "attach", "input_decode")}
            phases["publish_ms"] = duration(events, "publish_start", "publish_end", required=any(s.startswith("save:") for s in commands))
            phases["harness_wait_ms"] = duration(events, "wait_start", "released", required=any(s.startswith("wait:") for s in commands))
            phases["raw_ttft_ms"] = observed["ttft_ms"]
            phases["ttft_without_harness_wait_ms"] = observed["ttft_ms"] - phases["harness_wait_ms"]
            phases["other_before_first_token_ms"] = observed["ttft_ms"] - sum(phases[key] for key in ("backend_ms", "model_ms", "context_ms", "attach_ms", "input_decode_ms", "publish_ms", "harness_wait_ms"))
            if phases["other_before_first_token_ms"] < 0:
                raise ValueError(f"{name}/{process}: overlapping first-token phases")
            if process.startswith("leaf"):
                leaves.append(phases)
            elif process != "root":
                internal.append(phases)
        rate = sum(len(g["tokens"]) for g in generations) * 1e9 / (max(g["end_ns"] for g in generations) - min(g["start_ns"] for g in generations))
        if not math.isclose(rate, case["generation_window_tps"], rel_tol=1e-12):
            raise ValueError(f"{name}: window throughput differs")
        rows.append({"case": name, "run": case["run"], "mode": case["mode"], "agents": case["count"],
                     "fanouts": ",".join(map(str, case["cell"])), "total_peak_gib": case["total_peak_accounted_bytes"] / 2**30,
                     "minimum_available_gib": case["memory_min_available"] / 2**30,
                     "generation_window_tps": rate, "throughput_over_copy": rate / ref["generation_window_tps"],
                     "gpu_faults": case["gpu_faults_total"], "internal_publish_ms": statistics.mean(p["publish_ms"] for p in internal),
                     "leaf_phases": {key:statistics.mean(p[key] for p in leaves) for key in leaves[0]},
                     "tmpfs_all_attached_bytes": case["all_attached"]["tmpfs_allocated"],
                     "tmpfs_names_removed_bytes": case["after_unlink"]["tmpfs_allocated"],
                     "tmpfs_after_all_exit_bytes": case["after_all_exit"]["tmpfs_allocated"]})
    if any(not r["text_match"] or not r["token_match"] for r in comparisons):
        raise ValueError("cross-mode text/token mismatch")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["fanouts"], row["mode"]].append(row)
    summaries, flat = [], []
    for (fanouts, mode), group in sorted(groups.items()):
        if len(group) != meta["repetitions"]:
            raise ValueError("incomplete repetition group")
        metrics = {key:estimate((r[key] for r in group), geometric=key.endswith("_over_copy"))
                   for key in group[0] if key not in ("case", "run", "mode", "agents", "fanouts", "leaf_phases")}
        phases = {key:estimate(r["leaf_phases"][key] for r in group) for key in group[0]["leaf_phases"]}
        summaries.append({"fanouts": fanouts, "mode": mode, "agents": group[0]["agents"], "repetitions": len(group), "metrics": metrics, "leaf_phases": phases})
        flat.append({"fanouts":fanouts, "agents":group[0]["agents"], "mode":mode, "repetitions":len(group),
                     **{k:v["mean"] for k,v in metrics.items()}, **{"leaf_"+k:v["mean"] for k,v in phases.items()}})
    result = {"measurement_valid":True, "case_count":len(cases), "groups":summaries,
              "nonself_comparisons":[r for r in comparisons if not r["self"]], "cases":rows,
              "definitions":{"TTFT":"process start to first token; harness waits are reported separately",
                             "generation_window":"all actual generated tokens / (latest generation end - earliest generation start)",
                             "memory":"whole-case MemAvailable drop + warmed model file once",
                             "flat":"internal publish is absent; leaves recompute unshared intermediate prefixes"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "analysis_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(f"tree summary: cases={len(cases)} groups={len(summaries)} nonself_matches={len(result['nonself_comparisons'])}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.output)
