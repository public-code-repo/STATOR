import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import paths

CODE = Path(paths.CODE)
ROOT = Path(paths.ROOT)
DRIVERS = CODE / "drivers"
TIMED = DRIVERS / "timed"
PROBES = CODE / "probes"
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
CXX = os.environ.get("CXX", "g++")
CXXFLAGS = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
NVCCFLAGS = ["-std=c++17", "-O3", "-lineinfo", "-arch=sm_110", "-Xcompiler=-Wall,-Wextra,-Werror"]

DRIVER_BUILDS = [
    ("prefix_agent_extents", "prefix_agent.cpp", "extents"),
    ("prefix_agent_stock", "prefix_agent.cpp", "stock"),
    ("prefix_agent_private_weights", "prefix_agent_private_weights.cpp", "kv_only"),
    ("prefix_agent_variants", "prefix_agent.cpp", "variants"),
    ("publisher_variants", "publisher.cpp", "variants"),
    ("prefix_agent_variants_gqa", "prefix_agent.cpp", "variants_gqa"),
    ("publisher_variants_gqa", "publisher.cpp", "variants_gqa"),
    ("prefix_agent_linux", "prefix_agent.cpp", "linux"),
    ("publisher_linux", "publisher.cpp", "linux"),
    ("publisher_tasks_linux", "publisher_tasks.cpp", "linux"),
    ("batch_server_extents", "batch_server.cpp", "extents"),
    ("tree_agent_chains", "tree_agent.cpp", "chains"),
    ("session_agent_extents", "session_agent.cpp", "extents"),
    ("session_agent_stock", "session_agent.cpp", "stock"),
]
TIMED_BUILDS = [
    ("session_agent_extents", "session_agent.cpp", "extents"),
    ("session_agent_chains", "session_agent.cpp", "chains"),
    ("session_agent_stock", "session_agent.cpp", "stock"),
    ("prefix_agent_publish", "prefix_agent.cpp", "publish"),
    ("publisher_publish", "publisher.cpp", "publish"),
    ("fault_publisher_publish", "fault_publisher.cpp", "publish"),
    ("ratio_publisher_publish", "ratio_publisher.cpp", "publish"),
    ("prefix_agent_publish_repeat", "prefix_agent.cpp", "publish_repeat"),
    ("fork_publisher_publish_repeat", "fork_publisher.cpp", "publish_repeat"),
]
PROBE_BUILDS = [
    ("tlb_probe", ["tlb_probe.cu"], []),
    ("vmm_attach_probe", ["vmm_attach_probe.cu"], ["-lcuda"]),
    ("protect_probe", ["protect_probe.cu"], ["-lcuda"]),
    ("linux_share_probe", ["linux_share_probe.cu"], []),
    ("mechanism_probe", ["mechanism_probe.cu"], []),
    ("mechanism_probe_rw", ["mechanism_probe_rw.cu"], []),
    ("seal_probe", ["seal_probe.cu"], []),
    ("populate_probe", ["populate_probe.cu"], []),
    ("access_flag_probe", ["access_flag_probe.cu"], []),
    ("fork_probe", ["fork_probe.cu", "extent_cow.cpp"], ["-lpthread"]),
    ("mm_cost_probe", ["mm_cost_probe.cu", "extent_cow.cpp"], ["-lpthread"]),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command):
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def driver(output, source, engine_name):
    engine = ROOT / "engines" / engine_name
    library = engine / "build" / "bin"
    command = [CXX, *CXXFLAGS, "-isystem", engine / "include", "-isystem", engine / "ggml" / "include",
               source, "-o", output, f"-L{library}", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{library}"]
    run(command)
    return {"name": output.name, "source": source.name, "engine": engine_name, "binary_sha256": sha(output),
            "source_sha256": sha(source), "libllama_sha256": sha(library / "libllama.so")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    selected = lambda name: args.only is None or name in args.only
    bin_dir = ROOT / "bin"
    timed_dir = bin_dir / "timed"
    timed_dir.mkdir(parents=True, exist_ok=True)
    builds, timed = [], []
    for name, source, engine in DRIVER_BUILDS:
        if selected(name):
            builds.append(driver(bin_dir / name, DRIVERS / source, engine))
    for name, source, engine in TIMED_BUILDS:
        if selected(name):
            record = driver(timed_dir / name, TIMED / source, engine)
            record["timing_header_sha256"] = sha(TIMED / "stator_timing.h")
            timed.append(record)
    for name, sources, libraries in PROBE_BUILDS:
        if selected(name):
            run([NVCC, *NVCCFLAGS, *(PROBES / source for source in sources), "-o", bin_dir / name, *libraries])
            builds.append({"name": name, "binary_sha256": sha(bin_dir / name)})
    if selected("fault_writer"):
        run([NVCC, *NVCCFLAGS, TIMED / "fault_writer.cu", "-o", timed_dir / "fault_writer", "-lcuda"])
        timed.append({"name": "fault_writer", "binary_sha256": sha(timed_dir / "fault_writer")})
    if selected("fault_inject.so"):
        run([NVCC, "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared", "-arch=sm_110", "-O2",
             DRIVERS / "fault_inject.cu", "-o", bin_dir / "fault_inject.so"])
        builds.append({"name": "fault_inject.so", "binary_sha256": sha(bin_dir / "fault_inject.so")})
    for path, records in ((bin_dir / "builds.json", builds), (timed_dir / "builds.json", timed)):
        old = json.loads(path.read_text()) if path.exists() else []
        names = {record["name"] for record in records}
        path.write_text(json.dumps([r for r in old if r["name"] not in names] + records, indent=2) + "\n")


if __name__ == "__main__":
    main()
