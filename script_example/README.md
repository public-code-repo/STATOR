# STATOR — Example Scripts

Scripts for reproducing the STATOR results. One directory per subsection of the
evaluation, each with the runners, the summary scripts, and a `run.sh` that runs
STATOR and the baselines with the parameters of the paper.

## Layout

| Directory | What it measures |
|---|---|
| `00_setup/` | engines, drivers, probes, models, workloads, and the vLLM environment |
| `01_linux_sharing_mechanisms/` | memory, TTFT, and GPU faults of shared, private, and read-only mappings, fork, page cache, page merging, seals, populate requests, and device memory sharing (Section 4.2) |
| `02_in_place_extents/` | memory of in-place weights and KV cache extents, and the speed of a host-mapped cache (Section 4.3) |
| `03_hand_over_fork_and_agent_trees/` | hand-over to 8 and 32 children, repeated live state forks, a defective sharer, extent chains, fault domains, and device memory attach (Section 4.4) |
| `04_overall_memory_and_capacity/` | memory and agent capacity of each configuration, and resident agents (Section 4.5) |
| `05_agents_over_time/` | agent sessions that arrive over time and wait for tools, and resident agents compared with agents restarted after a tool wait (Section 4.6) |
| `06_batching_server/` | vLLM with shared weights and a shared prefix, compared with vLLM and LMCache (Section 4.7) |
| `07_shared_prefix_ratio/` | tool-calling prompts with a shared part of 2K-16K tokens (Section 4.8) |
| `common/` | runner of the variants of STATOR, shared summary scripts, live-publish helpers, and the run guard |

## Environment

Set before running.

```
export STATOR_ROOT=/path/to/data
source setup_env.sh
```

Every `run.sh` runs from `${STATOR_ROOT}`, which holds `engines/`, `bin/`,
`models/`, `workloads/`, and `results/`. `setup_env.sh` adds `stator_code/` and
each directory here to `PYTHONPATH`, and a runner resolves the modules in
`common/` from any directory.

The runners mount tmpfs file systems, drop the page cache, and change the huge
page setting of shared memory through `sudo -n`. The prefetch measurement of
Section 4.2 reloads `nvidia_uvm` with prefetching disabled and restores the
module after the measurement. A run requires that no other process uses the GPU
or a large part of the memory.

## Data

`00_setup/` writes every input under `${STATOR_ROOT}`: the engines, the
binaries, the GGUF models, the agent workload (`agent_prefix.txt`,
`agent_tasks.txt`), the BFCL workload (`bfcl_prefix.txt`, `bfcl_tasks.txt`,
`bfcl_len/`), and the shared-prefix ratio workload (`ratio/`). The runners of
Sections 4.2-4.4 generate the synthetic workload, which repeats one paragraph
to a prefix of 4K or 16K tokens. `05_agents_over_time/config/` holds the
session traces of Section 4.6, with the arrival time, the task, and the tool
waits of each session, and the job lists that assign each trace to a
configuration.

## Running

```
bash 00_setup/run.sh
bash 00_setup/setup_vllm.sh

bash 01_linux_sharing_mechanisms/run.sh
bash 02_in_place_extents/run.sh
bash 03_hand_over_fork_and_agent_trees/run.sh
bash 04_overall_memory_and_capacity/run.sh
bash 05_agents_over_time/run.sh
bash 06_batching_server/run.sh
bash 07_shared_prefix_ratio/run.sh
```

The measurements of the paper on NVIDIA Thor took about 5 h for Section 4.2,
3.5 h for Section 4.3, 5.5 h for Section 4.4, 6.5 h for Section 4.5, 34 h for
Section 4.6, 1.5 h for Section 4.7, and 3.5 h for Section 4.8. `RUN_DATE` sets
the date prefix of the result directories. `run_traces.py <job list> --execute
--resume` continues a job list of Section 4.6 after an interruption without a
second run of a completed job.

## Baselines

| name | mode in the scripts | what it is |
|---|---|---|
| llama.cpp | `none` | unmodified llama.cpp, in which every process reads the weights into device memory and computes the prefix |
| Copy | `copy`, `restore`, `weights` | a restore of the state file of the prefix into the private cache of each process, with the weights in place |
| Demand | `demand` | Copy with the cache allocated on demand |
| Device | `device_merged`, `vmm` | the prefix rows in device memory, shared through CUDA virtual memory management |
| CoW Map | `writable_no_read`, `cow_plain` | a private writable mapping of the cache file of the publisher |
| Shared Map | `shared_rw` | a shared writable mapping of the cache file of the publisher |
| Read-Only Map | `ro_plain` | a read-only mapping of the cache file without the CPU read and the populate-ahead |
| vLLM | `vllm` | unmodified vLLM, with a prefix cache in each server process |
| vLLM-CPU | `cpu` | vLLM with the offloading of its KV cache to host memory |
| vLLM-LMCache | `lmcache` | vLLM with LMCache on a store process in host memory |
| vLLM-STATOR | `stator`, `stator_weights`, `stator_kv` | vLLM with the plugin of STATOR, with both extents, the weights alone, or the KV cache alone |
| STATOR (Weights) | `weights` | the in-place weights alone, which is the configuration of Copy |
| STATOR (KV) | `kv` | the KV cache extents alone |
| STATOR (Copied File) | `copy_map` | a copy of the published rows into a file on 2 MiB pages in place of the conversion |
| STATOR (One Level) | `flat` | an agent tree that shares only the prefix of the root and recomputes the rows that the leaders append |
| STATOR | `extent`, `both`, `chain` | immutable and private extents with the weights in place |
