import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from measured_analysis import analyze, percentile


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def latency_distribution(values, offered_count=None):
    values = sorted(values)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("invalid latency observation")
    if offered_count is not None and len(values) > offered_count:
        raise ValueError("more first-token observations than offered sessions")
    cumulative, points = 0, []
    for value, count in sorted(collections.Counter(values).items()):
        cumulative += count
        points.append({"latency_s": value, "observed_at_or_below": cumulative,
                       "fraction_of_observed": cumulative / len(values),
                       "fraction_of_all_offered": cumulative / offered_count if offered_count else None})
    return {"observed_count": len(values), "offered_count": offered_count,
            "without_first_token_count": offered_count - len(values) if offered_count is not None else None,
            "p50_s": percentile(values, .5), "p95_s": percentile(values, .95),
            "p99_s": percentile(values, .99), "points": points,
            "quantile_population": "observations only; unserved sessions are listed separately"}

def inspect(path):
    result = analyze(path)
    meta = json.loads((path / "metadata.json").read_text())
    trace = json.loads((path / "trace.json").read_text())
    events = read_jsonl(path / "raw.jsonl")
    completion = json.loads((path / "completion.json").read_text()) if (path / "completion.json").exists() else {}
    errors = result["errors"]
    if not completion.get("runner_completed"):
        errors.append("missing runner completion")
    if completion.get("guard_events") != result["guard_events"]:
        errors.append("guard count differs from completion record")
    for key in ("n_gen", "window_s", "seed", "fork_k", "memory_budget_gib", "reserve_gib", "guard_gib"):
        if meta.get(key) != trace.get(key):
            errors.append(f"metadata/trace mismatch: {key}")
    if meta.get("sessions") != len(trace["sessions"]):
        errors.append("metadata session count mismatch")
    sessions = {s["sid"]: s for s in trace["sessions"]}
    if len(sessions) != len(trace["sessions"]):
        errors.append("duplicate session ids in trace")
    procs = [e for e in events if e["event"] == "proc"]
    launches = [e for e in events if e["event"] == "launch"]
    unique = [(p["sid"], p["role"], p["index"]) for p in procs]
    if len(unique) != len(set(unique)):
        errors.append("duplicate process identity")
    launched_pids = [e["pid"] for e in launches]
    if len(launched_pids) != len(set(launched_pids)):
        errors.append("duplicate launch PID")
    if set(launched_pids) != {p["pid"] for p in procs if p.get("pid") is not None}:
        errors.append("launch/process inventory mismatch")
    actual_tokens, starts, ends, text_hashes, generations = 0, [], [], {}, []
    for proc in procs:
        key = f"{proc['sid']}.{proc['role']}{proc['index']}"
        session = sessions.get(proc["sid"])
        if session is None or session["kind"] != proc["kind"] or session["arrival"] != proc["arrival"]:
            errors.append(f"{key}: process differs from frozen session")
        allowed = proc["role"] == "a" and proc["index"] == 0 or proc["role"] == "c" and proc["kind"] == "fork" and 0 <= proc["index"] < trace["fork_k"]
        if not allowed:
            errors.append(f"{key}: unexpected role/index")
        if proc["exit"] != 0:
            continue
        timing = []
        if len(proc.get("timing_paths", [])) != 1:
            errors.append(f"{key}: expected exactly one timing stream")
        for name in proc.get("timing_paths", []):
            timing.extend(read_jsonl(Path(name)))
        if any(e.get("schema") != 1 or e.get("pid") != proc["pid"] for e in timing):
            errors.append(f"{key}: timing schema/PID mismatch")
        process_starts = [e["mono_ns"] for e in timing if e["event"] == "process_start"]
        if len(process_starts) != 1 or process_starts[0] < proc["launch_ns"]:
            errors.append(f"{key}: invalid process/launch clock")
        else:
            row = next((r for r in result["requests"] if (r["sid"], r["role"], r["index"]) == (proc["sid"], proc["role"], proc["index"])), None)
            if row is not None:
                row["phases"]["spawn_s"] = (process_starts[0] - proc["launch_ns"]) / 1e9
                measured = [row["phases"].get(name + "_s") for name in ("backend", "model", "context", "attach", "input_decode")]
                row["phases"]["other_process_s"] = row["process_to_first_s"] - sum(measured) if all(v is not None for v in measured) else None
        groups = collections.defaultdict(list)
        for event in timing:
            if event["event"] in ("generation_start", "generation_end", "token"):
                groups[event["generation"]].append(event)
        expected_generations = 2 if proc["role"] == "c" or proc["kind"] == "fork" else 3 if proc["kind"] == "tool" else 1
        if set(groups) != set(range(1, expected_generations + 1)):
            errors.append(f"{key}: generation identity mismatch")
        tokens = []
        for number, group in sorted(groups.items()):
            bs = [e["mono_ns"] for e in group if e["event"] == "generation_start"]
            es = [e["mono_ns"] for e in group if e["event"] == "generation_end"]
            ts = [e for e in group if e["event"] == "token"]
            tokens += [e["value"] for e in ts]
            if len(bs) == len(es) == 1 and es[0] > bs[0]:
                starts.append(bs[0]); ends.append(es[0])
                generations.append({"process": key, "generation": number, "tokens": len(ts), "start_ns": bs[0], "end_ns": es[0]})
            else:
                errors.append(f"{key}/gen{number}: nonpositive/missing generation span")
        actual_tokens += len(tokens)
        output = Path(proc["output_path"]).read_text().splitlines()
        text = [line for line in output if line.startswith("TEXT ")]
        token_lines = [line for line in output if line.startswith("TOKEN_IDS:")]
        if len(text) != 1 or len(token_lines) != 1:
            errors.append(f"{key}: missing stdout text/token vector")
        else:
            vector = [int(v) for v in token_lines[0].split(":")[1:]]
            if vector != tokens:
                errors.append(f"{key}: stdout token IDs differ from timing stream")
            text_hashes[key] = hashlib.sha256(text[0].encode()).hexdigest()
    samples = [e for e in events if e["event"] == "sample"]
    end = next(e for e in events if e["event"] == "end")
    start = next(e["mono_ns"] for e in events if e["event"] == "arrival_clock")
    last_observed_s = max((e["mono_ns"] - start) / 1e9 for e in samples) if samples else 0
    served = {r["sid"] for r in result["requests"] if r["role"] == "a"}
    censored = [{"sid": s["sid"], "observed_wait_lower_bound_s": max(0, last_observed_s - s["arrival"])}
                for s in trace["sessions"] if s["sid"] not in served]
    primary = [r for r in result["requests"] if r["role"] == "a"]
    phase_means = {}
    for key in ("spawn_s", "backend_s", "model_s", "context_s", "attach_s", "input_decode_s", "other_process_s"):
        values = [r["phases"].get(key) for r in primary]
        phase_means[key] = statistics.mean(values) if values and all(v is not None for v in values) else None
    admission = [r["admission_s"] for r in primary if r["admission_s"] is not None]
    distributions = {
        "ttft": latency_distribution([r["arrival_to_first_s"] for r in primary], len(trace["sessions"])),
        "admission_of_first_token_observed_requests": latency_distribution(admission),
        "tool_resume": latency_distribution([r["release_to_first_s"] for r in result["turns"] if r["kind"] == "tool"]),
    }
    window_s = (max(ends) - min(starts)) / 1e9 if starts and ends else None
    complete = not result["unfinished"] and not result["failed_processes"]
    if complete and actual_tokens != result["actual_recorded_tokens"]:
        errors.append("complete-process token total differs from full timing stream")
    result.update(seed=meta["seed"], configured_tool_mean_s=meta.get("tool_mean_s"),
                  offered_rate=meta.get("rate"), offered_window_s=meta["window_s"],
                  generation_window_s=window_s,
                  generation_window_tps=actual_tokens / window_s if complete and window_s and not errors else None,
                  completed_process_tokens=actual_tokens, generation_intervals=generations,
                  text_hashes=text_hashes, first_token_phase_means=phase_means,
                  latency_distributions=distributions,
                  admission_p50_s=percentile(admission, .5), admission_p95_s=percentile(admission, .95), admission_p99_s=percentile(admission, .99),
                  tool_resume_p99_s=percentile([r["release_to_first_s"] for r in result["turns"] if r["kind"] == "tool"], .99),
                  live_peak=max((e["live"] for e in samples), default=None),
                  live_mean=statistics.mean(e["live"] for e in samples) if samples else None,
                  residual_memavailable_mib=end["residual_mib"], tmpfs_peak_mib=end["tmpfs_peak_mib"],
                  unserved_requests=censored, expected_processes=sum(1 + trace["fork_k"] if s["kind"] == "fork" else 1 for s in trace["sessions"]),
                  observed_processes=len(procs), measurement_valid=not errors,
                  trace_signature=hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                  source_metadata_sha256=hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest())
    result["gates"] = {"D1": not result["failed_processes"] and result["guard_events"] == 0,
                       "D2": "pending_pair", "D3": result["extent_reclaimed"] if meta["system"] == "both" else "not_applicable",
                       "all_sessions_completed": complete, "memory_budget_met": result["memory_budget_met"]}
    return result

def compare(copy, stator):
    for key in ("seed", "configured_tool_mean_s", "requested_tokens", "sessions", "offered_rate", "offered_window_s", "trace_signature"):
        if copy[key] != stator[key]:
            raise ValueError(f"unmatched pair: {key}")
    common = set(copy["text_hashes"]) & set(stator["text_hashes"])
    token_equal = sum(copy["token_ids"][k] == stator["token_ids"][k] for k in common)
    text_equal = sum(copy["text_hashes"][k] == stator["text_hashes"][k] for k in common)
    complete = not copy["unfinished"] and not stator["unfinished"] and len(common) == copy["expected_processes"]
    return {"compared_processes": len(common), "token_ids_equal": token_equal,
            "texts_equal": text_equal, "complete_coverage": complete,
            "D2": complete and len(common) == token_equal == text_equal,
            "mismatched": sorted(k for k in common if copy["token_ids"][k] != stator["token_ids"][k] or copy["text_hashes"][k] != stator["text_hashes"][k]),
            "missing_or_incomplete": sorted((set(copy["token_ids"]) | set(stator["token_ids"])) - common)}


