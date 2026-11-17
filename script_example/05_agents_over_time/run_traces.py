import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import paths
from summarize_traces import collect

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jobs_file = args.jobs if args.jobs.exists() else HERE / "config" / f"{args.jobs}.json"
    manifest = json.loads(jobs_file.read_text())
    config = jobs_file.parent
    for job in manifest["jobs"]:
        if sha(config / job["trace"]) != job["sha256"]:
            raise RuntimeError(f"trace changed: {job['trace']}")
    if not args.execute:
        print(f"{jobs_file.stem}: {len(manifest['jobs'])} jobs; traces verified; nothing run")
        return
    gpu = os.environ.get("STATOR_GPU_A") or os.environ.get("STATOR_GPU")
    if not gpu:
        raise RuntimeError("STATOR_GPU is not set; source setup_env.sh")
    root = Path(paths.RESULTS) / "agents_over_time" / jobs_file.stem
    result_root = root / "results"
    logs = root / "logs"
    result_root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    journal = root / "execution.jsonl"
    for job in manifest["jobs"]:
        name = job["tag"]
        done = logs / (name + "-guard.json")
        if (result_root / name).exists():
            if args.resume and done.exists():
                record = json.loads(done.read_text())
                if record["exit"] == 0 and record["reason"] is None and (result_root / name / "completion.json").exists():
                    print(f"preserving completed job: {name}", flush=True)
                    continue
            raise RuntimeError(f"existing result needs review, not overwrite: {name}")
        env = {**os.environ, "SYSTEMS": job["system"], "MEASURED_TRACE": str(config / job["trace"]),
               "RESULT_ROOT": str(result_root), "RESULT_TAG": name, "STATOR_GPU_A": gpu,
               "RESERVE_GIB": "16", "GUARD_GIB": "12", "ESTIMATE_NONE": "6400",
               "ESTIMATE_WEIGHTS": "1800", "ESTIMATE_BOTH": "850"}
        command = [sys.executable, "-u", str(Path(paths.REPO) / "script_example/common/guarded_command.py"),
                   "--report", str(done), "--limit-s", str(job["limit_s"] + 300), "--",
                   sys.executable, "-u", str(HERE / "agents_over_time.py")]
        start = time.time()
        with journal.open("a") as output:
            output.write(json.dumps({"event": "start", "tag": name, "time": start}) + "\n")
        print(f"starting {name}", flush=True)
        with (logs / (name + ".log")).open("w") as output:
            process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        with journal.open("a") as output:
            output.write(json.dumps({"event": "end", "tag": name, "time": time.time(),
                                     "elapsed_s": time.time() - start, "exit": code}) + "\n")
        print(f"finished {name}: exit={code}", flush=True)
        if code:
            raise RuntimeError(f"job failed; inspect the raw data: {name}")
        state = collect(result_root, root / "analysis", manifest)
        print(f"analysis: {state['analyzed_jobs']}/{len(manifest['jobs'])}; "
              f"measurement_valid={state['measurement_valid']}", flush=True)
        if not state["measurement_valid"]:
            raise RuntimeError("measurement validation failed; raw results and analysis preserved")


if __name__ == "__main__":
    main()
