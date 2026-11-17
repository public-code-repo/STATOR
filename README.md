# STATOR
**STATOR: Immutable State Extents for Copy-Free Sharing and Forking in Multi-Process LLM Serving (Submitted OSDI'27)**

## Introduction
STATOR is an ownership model designed for inference processes that run in isolation on unified-memory systems.  
It shares the weights and the key-value (KV) cache of a prompt prefix among the processes without copies or GPU faults.  
To share GPU-visible model state among isolated processes, STATOR introduces:
- Immutable state extents, which every process maps read-only through the host page tables that the GPU uses for address translation,
- Private extents, each owned by one process, whose page table entries are populated before any GPU access,
- An in-place publish and a live state fork, which convert the whole pages of the written rows of a running process into an immutable extent while the process continues to write after the published rows,
- Extent chains, which share the extents of a root and its leaders with every process of an agent tree.


## Key Features
- In-place weights map the model file read-only in every process, and the GPU reads the weights from the mapped pages without a copy in device memory.
- KV cache extents hold the rows of a published prefix in one cache file that every process maps read-only, and each process writes its own rows into a private extent.
- Attach with a CPU populate prepares the page table entries of both extents before the first GPU access, and the GPU reads the shared pages without faults.
- An in-place publish changes the protection of the whole pages of the written rows to read-only without a copy of any row.
- A live state fork allows the publisher to generate after its published rows while its children attach to the published extent.
- Extent chains publish only the rows that each level of an agent tree appends to the extents of its ancestors.
- A vLLM plugin shares the weights and the KV cache blocks of a finished request among vLLM server processes.

## Core Components
- `patches/*.patch` — Changes of STATOR to llama.cpp at commit `6f767fe96`, with one complete patch for each engine configuration of the evaluation.
- `drivers/publisher.cpp` / `drivers/publisher_tasks.cpp` — Publisher, which computes a prefix, publishes the prefix, and starts the agents that attach to the published extent.
- `drivers/prefix_agent.cpp` / `drivers/prefix_agent_private_weights.cpp` — Agent process, which attaches to a published prefix and generates a continuation, and the variant of the agent with the weights in device memory for STATOR (KV).
- `drivers/tree_agent.cpp` — Root, leader, or leaf of an agent tree, which runs a list of steps on one context and publishes on an extent chain.
- `drivers/session_agent.cpp` — Agent session with attach, decode, generation, tool wait, and resume steps, which a runner releases one at a time.
- `drivers/batch_server.cpp` — Batching process, which continues one prefix with several sequences in one batch.
- `drivers/fault_inject.cu` — Preloaded library, which raises one fatal GPU error in the process that loads the library.
- `drivers/timed/` — Drivers with timing points, and publishers for repeated forks, a defective sharer, and prompts with a private part.
- `probes/*.cu` — Measurement programs for the sharing mechanisms of Linux under GPU access: shared, private, and read-only mappings, fork, page cache, page merging, seals, populate requests, address translation reach, and device memory sharing.
- `vllm/stator_vllm/` / `vllm/stator_vllm_direct/` — vLLM plugin with a KV connector, which registers the blocks of a finished request as an immutable extent, and with the direct attachment of the weights.
- `paths.py` — Data root (`STATOR_ROOT`) and the directories of the engines, binaries, models, workloads, and results.

Each engine configuration applies one patch to llama.cpp at commit `6f767fe96`.

| Engine | Patch | Configuration |
|---|---|---|
| `stock` | none | llama.cpp |
| `weights` | `weights_in_place.patch` | in-place weights |
| `kv_only` | `kv_extents_only.patch` | STATOR (KV), whose agents read the weights into device memory |
| `extents` | `stator_extents.patch` | immutable and private extents with in-place weights |
| `chains` | `extent_chains.patch` | extent chains |
| `variants` | `stator_variants.patch` | variants of STATOR in Table 3 and the speed of a host-mapped cache |
| `variants_gqa` | `stator_variants_gqa.patch` | attention over grouped query heads on a host-mapped cache |
| `publish` | `stator_publish.patch` | in-place publish, live state fork, and STATOR (Copied File) |
| `publish_repeat` | `stator_repeat_publish.patch` | repeated live state forks |
| `linux` | `linux_baselines.patch` | mappings of the cache file with the sharing mechanisms of Linux |

## Installation

### Requirements
- NVIDIA Jetson AGX Thor in the MAXN power mode, with 122.8 GiB of memory shared by the CPU and the GPU and no swap space
- Linux 6.8.12-tegra with the open GPU kernel modules 595.78
- CUDA 13.0, CMake 3.28, GCC with C++20 support, and Python 3.12
- Transparent 2 MiB pages for process memory, and tmpfs with 2 MiB pages for the memory file system
- Passwordless `sudo` for tmpfs mounts, page cache drops, the huge page setting of shared memory, and the reload of `nvidia_uvm`
- vLLM 0.20.0 (cu130), PyTorch 2.10.0, and LMCache 0.5.5 for Section 4.7 (`script_example/00_setup/setup_vllm.sh`)

### Set Up Environment
```bash
git clone https://github.com/<organization>/STATOR.git
cd STATOR

pip install -r requirements.txt

source setup_env.sh
```

`setup_env.sh` adds `stator_code/` and the experiment directories to `PYTHONPATH`, sets the data root `STATOR_ROOT`, which defaults to the repository root, and sets `STATOR_GPU` to the first GPU that `nvidia-smi` lists.

### Verify Installation
```bash
python3 -c "import paths; print(paths.ROOT)"
python3 script_example/05_agents_over_time/run_traces.py waits_60_120_240
```

The first command prints the data root. The second command checks the session traces of Section 4.6 against their sha256 values without a GPU run and prints `waits_60_120_240: 18 jobs; traces verified; nothing run`.

## Usage

### Data Layout
STATOR reads the engines, the binaries, the models, and the workloads from the data root and writes one directory per run under `results/`.

| Directory | Content |
|---|---|
| `engines/<engine>/` | llama.cpp at commit `6f767fe96` with the patch of the engine, built in `build/` |
| `bin/`, `bin/timed/` | drivers and probes, with the hashes of each build in `builds.json` |
| `models/` | GGUF model files |
| `workloads/` | agent, BFCL, and shared-prefix ratio workloads |
| `ext/` | vLLM environment, LMCache, and the Hugging Face cache |
| `results/` | raw log, metadata, and summary tables of each run |

### Building Engines and Preparing Inputs
```bash
bash script_example/00_setup/build_engines.sh
python3 script_example/00_setup/build_drivers.py
bash script_example/00_setup/prepare_models.sh
bash script_example/00_setup/prepare_workloads.sh
bash script_example/00_setup/setup_vllm.sh
```

`build_engines.sh` clones llama.cpp at commit `6f767fe96`, applies the patch of each engine, and builds the engine in `engines/<engine>/build`. `ENGINES` selects a subset of the engines.
`build_drivers.py` builds the drivers into `bin/` and `bin/timed/` and the probes into `bin/`, and records the hash of each binary and of its engine library in `builds.json`.
`prepare_models.sh` downloads Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and Llama-3.1-8B-Instruct as GGUF files with `Q4_K_M` quantization, checks their sha256 values, and downloads Qwen2.5-7B-Instruct-AWQ for vLLM.
`prepare_workloads.sh` writes the agent workload from the Markdown files of the engine and the tool-calling workloads from BFCL v4 (`simple_python`) at commit `6ea5797` of the gorilla repository.
`setup_vllm.sh` installs vLLM, PyTorch, LMCache, and the plugin of STATOR in `ext/` and checks the sha256 values of the vLLM, PyTorch, and LMCache wheels.

### Reproducing paper results
[script_example/](script_example/) holds one directory per subsection of the evaluation, each with the runners, the summary scripts, and a `run.sh`.
Each runner records the raw log, the metadata of the machine and the binaries, and the summary tables of a run under `results/`.
See [script_example/README.md](script_example/README.md).

## Repository Structure
```
STATOR/
├── stator_code/
│   ├── paths.py
│   ├── patches/
│   │   ├── weights_in_place.patch
│   │   ├── kv_extents_only.patch
│   │   ├── stator_extents.patch
│   │   ├── extent_chains.patch
│   │   ├── stator_variants.patch
│   │   ├── stator_variants_gqa.patch
│   │   ├── stator_publish.patch
│   │   ├── stator_repeat_publish.patch
│   │   └── linux_baselines.patch
│   ├── drivers/
│   │   ├── publisher.cpp
│   │   ├── publisher_tasks.cpp
│   │   ├── prefix_agent.cpp
│   │   ├── prefix_agent_private_weights.cpp
│   │   ├── tree_agent.cpp
│   │   ├── session_agent.cpp
│   │   ├── batch_server.cpp
│   │   ├── fault_inject.cu
│   │   └── timed/
│   ├── probes/
│   └── vllm/
│       ├── stator_vllm/
│       ├── stator_vllm_direct/
│       └── diagnostic/
├── script_example/
│   ├── 00_setup/
│   ├── 01_linux_sharing_mechanisms/
│   ├── 02_in_place_extents/
│   ├── 03_hand_over_fork_and_agent_trees/
│   ├── 04_overall_memory_and_capacity/
│   ├── 05_agents_over_time/
│   ├── 06_batching_server/
│   ├── 07_shared_prefix_ratio/
│   ├── common/
│   └── README.md
├── requirements.txt
└── setup_env.sh
```
