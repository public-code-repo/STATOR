import argparse
import csv
import json
from pathlib import Path
import sys
import paths

sys.dont_write_bytecode = True
from analyze_traces import inspect, compare

HERE = Path(paths.RESULTS) / "agents_over_time"

def collect(root, output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    analyses, pending, errors = {}, [], []
    for job in manifest["jobs"]:
        tag = job["tag"]
        source = root / tag
        if not (source / "completion.json").exists():
            pending.append(tag)
            continue
        try:
            result = inspect(source)
        except (OSError, ValueError, KeyError) as error:
            errors.append({"tag": tag, "error": str(error)})
            continue
        analyses[tag] = result
        (output / (tag + ".json")).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    pairs, rows, cdf_rows = [], [], []
    for seed in (1, 2, 3):
        for mean in (60, 120, 240):
            prefix = f"trace-seed{seed}-mean{mean}-"
            copy, stator = (analyses.get(prefix + system) for system in ("weights", "both"))
            pair = compare(copy, stator) if copy and stator else None
            if pair:
                pairs.append({"seed": seed, "configured_tool_mean_s": mean, **pair})
            for system in ("weights", "both"):
                result = analyses.get(prefix + system)
                if not result:
                    continue
                control = next(c for c in manifest["trace_controls"] if c["seed"] == seed and c["configured_tool_mean_s"] == mean)
                rows.append({"seed": seed, "tool_mean_setting_s": mean,
                    "actual_trace_tool_mean_s": control["observed_tool_mean_s"], "system": system,
                    **{key:result[key] for key in ("sessions", "completed", "requested_tokens", "actual_recorded_tokens",
                        "memory_peak_accounted_gib", "live_peak", "queue_peak", "ttft_count", "ttft_p50_s", "ttft_p95_s", "ttft_p99_s",
                        "admission_p50_s", "admission_p95_s", "admission_p99_s", "tool_resume_count", "tool_resume_p50_s", "tool_resume_p95_s", "tool_resume_p99_s",
                        "generation_window_tps", "window_token_rate", "last_generation_end_s",
                        "drain_wall_s", "gpu_faults", "guard_events", "tmpfs_before_bytes", "tmpfs_after_bytes", "measurement_valid")},
                    "unfinished_sessions": len(result["unfinished"]), "failed_processes": len(result["failed_processes"]),
                    "extent_reclaimed": result["extent_reclaimed"] if system == "both" else "NA",
                    "pair_token_match": f"{pair['token_ids_equal']}/{pair['compared_processes']}" if pair else "pending",
                    "pair_text_match": f"{pair['texts_equal']}/{pair['compared_processes']}" if pair else "pending",
                    "pair_complete_coverage": pair["complete_coverage"] if pair else "pending",
                    "without_first_token_count": result["latency_distributions"]["ttft"]["without_first_token_count"],
                    **{f"ttft_within_{limit}s_count": result["deadlines"][str(limit)]["served_within"] for limit in (1, 2, 5)},
                    **result["first_token_phase_means"]})
                for metric, distribution in result["latency_distributions"].items():
                    for point in distribution["points"]:
                        cdf_rows.append({"seed": seed, "tool_mean_setting_s": mean, "system": system,
                                         "metric": metric, "observed_count": distribution["observed_count"],
                                         "offered_count": distribution["offered_count"], **point})
    if rows:
        with (output / "review.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    if cdf_rows:
        with (output / "latency_cdf.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(cdf_rows[0]))
            writer.writeheader(); writer.writerows(cdf_rows)
    state = {"expected_jobs": len(manifest["jobs"]), "analyzed_jobs": len(analyses),
             "pending": pending, "errors": errors, "pairs": pairs,
             "measurement_valid": not errors and all(a["measurement_valid"] for a in analyses.values()),
             "all_jobs_analyzed": len(analyses) == len(manifest["jobs"])}
    (output / "validation.json").write_text(json.dumps(state, indent=2) + "\n")
    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, default=HERE/"analysis")
    args = parser.parse_args()
    state = collect(args.root, args.output, json.loads((HERE/"manifest.json").read_text()))
    print(f"trace analysis: {state['analyzed_jobs']}/{state['expected_jobs']}, parse_errors={len(state['errors'])}")
    raise SystemExit(0 if state["measurement_valid"] else 1)
