import csv
import glob
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from live_publish import bh_snapshot, bh_delta, kill_group
import paths

HERE = paths.ROOT
ROOT = Path(paths.ROOT)
STATE_BUDGET_GIB = 96
MEASURED_TRACE = os.environ.get("MEASURED_TRACE", "")


def env(name, default):
    return os.environ.get(name, default)


SYSTEMS = env("SYSTEMS", "both weights none").split()
RATES = [float(x) for x in env("RATES", "0.07 0.14 0.21").split()]
WINDOW_S = float(env("WINDOW_S", "480"))
LIMIT_S = float(env("LIMIT_S", "1080"))
SEED = int(env("SEED", "1"))
N_GEN = int(env("N_GEN", "32"))
CONTEXT = int(env("CONTEXT", "16384"))
MIX = [float(x) for x in env("MIX", "0.2 0.6 0.2").split()]
TOOL_MEAN_S = float(env("TOOL_MEAN_S", "120"))
TOOL_MIN_S = float(env("TOOL_MIN_S", "10"))
TOOL_MAX_S = float(env("TOOL_MAX_S", "300"))
FORK_K = int(env("FORK_K", "3"))
RESERVE_GIB = float(env("RESERVE_GIB", "16"))
GUARD_GIB = float(env("GUARD_GIB", "12"))
SAMPLE_S = float(env("SAMPLE_S", "0.5"))
GROW = env("GROW_ROWS", "256")
ESTIMATE = {"none": float(env("ESTIMATE_NONE", "6400")),
            "weights": float(env("ESTIMATE_WEIGHTS", "1800")),
            "both": float(env("ESTIMATE_BOTH", "850"))}
GPU = env("STATOR_GPU_A", "")
MODEL = env("MODEL", os.path.join(HERE, "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
PREFIX_FILE = env("PREFIX_FILE", os.path.join(HERE, "workloads/agent_prefix.txt"))
TASKS_FILE = env("TASKS_FILE", os.path.join(HERE, "workloads/agent_tasks.txt"))
CHAIN_LIB = env("CHAIN_LIB", os.path.join(HERE, "engines/chains/build/bin"))
DRIVER = env("DRIVER", os.path.join(HERE, "bin/session_agent_extents"))
STOCK_DRIVER = env("STOCK_DRIVER", os.path.join(HERE, "bin/session_agent_stock"))
MOUNT_ROOT = env("MOUNT_ROOT", os.path.join(HERE, "mounts"))
RESULT_ROOT = env("RESULT_ROOT", os.path.join(HERE, "results"))
RESULT_TAG = env("RESULT_TAG", time.strftime("%Y%m%d-%H%M%S") + "-engine-kvdynamic")
TMPFS_MIB = int(env("TMPFS_MIB", "16384"))
NAMES = {"none": "Existing", "weights": "Copy", "both": "STATOR"}


def meminfo(name):
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith(name + ":"):
                return int(line.split()[1]) // 1024
    raise KeyError(name)


def tmpfs_mib(path):
    st = os.statvfs(path)
    return (st.f_blocks - st.f_bfree) * st.f_frsize // (1 << 20)


def sudo(*argv):
    subprocess.run(["sudo", "-n", *argv], check=True)


def fields(line):
    out = {}
    for part in line.split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def make_trace(rate, seed):
    top = max(RATES)
    rng = random.Random(seed)
    tasks = [line.rstrip("\n") for line in open(TASKS_FILE) if line.strip()]

    def tool():
        return min(TOOL_MAX_S, max(TOOL_MIN_S, rng.expovariate(1.0 / TOOL_MEAN_S)))

    sessions, t, sid = [], 0.0, 0
    while True:
        t += rng.expovariate(top)
        if t >= WINDOW_S:
            break
        u, keep = rng.random(), rng.random()
        kind = "single" if u < MIX[0] else ("tool" if u < MIX[0] + MIX[1] else "fork")
        session = {"sid": sid, "arrival": t, "kind": kind, "task": tasks[sid % len(tasks)]}
        session["tools"] = [tool(), tool()]
        session["child_tools"] = [tool() for _ in range(FORK_K)]
        if keep < rate / top:
            sessions.append(session)
        sid += 1
    return sessions


def task_text(sid, task):
    return f"\n\nTask for agent {sid}: {task}\n"


def tool_text(sid, turn):
    return (f'\n\nTool result {turn} for agent {sid}: {{"status": "ok", "items": {sid % 7 + turn}}}\n'
            "Next call or final answer:\n")


def subtask_text(sid, index):
    return f"\n\nSubtask {index + 1} of {FORK_K} for agent {sid}: carry out step {index + 1} of the plan above.\n"


def merge_text(sid):
    return f"\n\nThe {FORK_K} subtasks of agent {sid} returned their results.\nFinal answer:\n"


class Proc:
    def __init__(self, session, role, index, argv, envs, out):
        self.session, self.role, self.index = session, role, index
        self.argv, self.envs, self.out = argv, envs, out
        self.popen = None
        self.launch = None
        self.offset = 0
        self.events = []
        self.exit = None
        self.decoded_first = False


class Run:
    def __init__(self, system, rate, trace, work, huge, log):
        self.system, self.rate, self.trace = system, rate, trace
        self.work, self.huge, self.log = work, huge, log
        self.queue = []
        self.live = []
        self.done = []
        self.timers = []
        self.guard_events = 0
        self.model_mib = os.path.getsize(MODEL) // (1 << 20)
        self.root_tokens = 0
        self.t0 = None
        self.t0_ns = None
        self.accounting_baseline_mib = None
        self.releases = []
        self.session_state = {}

    def base_env(self):
        e = dict(os.environ)
        e["CUDA_VISIBLE_DEVICES"] = GPU
        for key in ("STATOR_KV_HOST", "STATOR_KV_CHAIN", "STATOR_KV_PREFIX", "STATOR_KV_GROW",
                    "STATOR_HOST_PTR", "LD_LIBRARY_PATH"):
            e.pop(key, None)
        if self.system != "none":
            e["STATOR_HOST_PTR"] = "1"
            e["LD_LIBRARY_PATH"] = CHAIN_LIB
        return e

    def driver(self):
        return STOCK_DRIVER if self.system == "none" else DRIVER

    def proc(self, session, role, index, steps, extra):
        name = f"{session['sid']}.{role}{index}"
        argv = [self.driver(), MODEL, str(CONTEXT), str(N_GEN), *steps]
        return Proc(session, role, index, argv, extra, os.path.join(self.work, "out." + name))

    def start(self, p, now):
        e = self.base_env()
        e.update(p.envs)
        e.update(STATOR_TIMING_DIR=self.work + "/timing", STATOR_TRACE="1")
        p.launch_ns = time.monotonic_ns()
        with open(p.out, "w") as out, open(p.out + ".err", "w") as err:
            p.popen = subprocess.Popen(p.argv, env=e, stdout=out, stderr=err)
        p.launch = p.launch_ns / 1e9
        self.live.append(p)
        self.write({"event":"launch", "mono_ns":p.launch_ns,"pid":p.popen.pid,"sid":p.session["sid"],"role":p.role,"index":p.index,"argv":p.argv,"environment":p.envs})

    def admit(self, now):
        while self.queue:
            avail = meminfo("MemAvailable") - self.model_mib
            pending = sum(ESTIMATE[self.system] for p in self.live if not p.decoded_first)
            observed = max(0, self.accounting_baseline_mib - meminfo("MemAvailable")) + self.model_mib
            if (avail - pending - ESTIMATE[self.system] < RESERVE_GIB * 1024 or
                    observed + pending + ESTIMATE[self.system] > STATE_BUDGET_GIB * 1024):
                return
            self.start(self.queue.pop(0), now)

    def timing_events(self, p):
        if p.popen is None:
            return []
        paths = glob.glob(f"{self.work}/timing/{p.popen.pid}-*.jsonl")
        if not paths:
            return []
        if len(paths) != 1:
            raise RuntimeError("ambiguous timing PID")
        with open(paths[0]) as f:
            return [json.loads(line) for line in f if line.endswith("\n")]

    def release(self, path):
        before_ns = time.monotonic_ns()
        with open(path, "w"):
            pass
        after_ns = time.monotonic_ns()
        self.releases.append({"path":path,"before_ns":before_ns,"after_ns":after_ns})
        self.write({"event":"release", **self.releases[-1]})

    def read(self, p):
        try:
            with open(p.out) as handle:
                handle.seek(p.offset)
                data = handle.read()
        except FileNotFoundError:
            return []
        cut = data.rfind("\n")
        if cut < 0:
            return []
        p.offset += cut + 1
        new = []
        for line in data[:cut].split("\n"):
            kind = line.split(" ", 1)[0]
            if kind in ("ATTACHED", "DECODED", "PUBLISHED", "MARK", "RELEASED", "GENERATED", "RESULT"):
                f = fields(line)
                p.events.append((kind, f))
                new.append((kind, f))
            elif kind == "TEXT":
                p.events.append(("TEXT", {"hash": hashlib.sha256(line.encode()).hexdigest()[:16]}))
        return new

    def first_steps(self):
        return [f"load:{self.work}/state.root"]

    def agent_env(self):
        if self.system == "both":
            return {"STATOR_KV_CHAIN": f"{self.huge}/r:{self.root_tokens}", "STATOR_KV_HOST": "anon",
                    "STATOR_KV_GROW": GROW}
        return {}

    def go(self, sid, role, index, turn):
        return f"{self.work}/go.{sid}.{role}{index}.{turn}"

    def arrive(self, s):
        sid, kind = s["sid"], s["kind"]
        steps = self.first_steps() + [f"say:{task_text(sid, s['task'])}", "gen"]
        extra = self.agent_env()
        if kind == "tool":
            for turn in (1, 2):
                steps += [f"mark:t{turn}", f"wait:{self.go(sid, 'a', 0, turn)}",
                          f"say:{tool_text(sid, turn)}", "gen"]
        if kind == "fork":
            steps += [f"save:{self.work}/state.s{sid}", "mark:forked", f"wait:{self.go(sid, 'a', 0, 9)}",
                      f"say:{merge_text(sid)}", "gen"]
            if self.system == "both":
                extra = {"STATOR_KV_CHAIN": f"{self.huge}/r:{self.root_tokens}",
                         "STATOR_KV_HOST": f"{self.huge}/m{sid}", "STATOR_KV_GROW": GROW}
        self.session_state[sid] = {"children": [], "attached": 0, "removed": False}
        self.queue.append(self.proc(s, "a", 0, steps, extra))

    def fork(self, parent, published, now):
        s = parent.session
        sid = s["sid"]
        tokens = published.get("tokens", "0")
        state = self.session_state[sid]
        deadline = time.monotonic()+2
        ends = []
        while not ends and time.monotonic()<deadline:
            ends = [e["mono_ns"] for e in self.timing_events(parent) if e["event"]=="publish_end"]
            if not ends:
                time.sleep(.001)
        if not ends:
            raise RuntimeError("fork lacks publish clock")
        state["publish_seen"] = ends[-1] / 1e9
        state["publish_ms"] = float(published.get("publish_ms", "0"))
        children = []
        for index in range(FORK_K):
            steps = [f"load:{self.work}/state.s{sid}", f"say:{subtask_text(sid, index)}", "gen",
                     "mark:t1", f"wait:{self.go(sid, 'c', index, 1)}", f"say:{tool_text(sid, 1)}", "gen"]
            extra = {}
            if self.system == "both":
                extra = {"STATOR_KV_CHAIN": f"{self.huge}/r:{self.root_tokens},{self.huge}/m{sid}:{tokens}",
                         "STATOR_KV_HOST": "anon", "STATOR_KV_GROW": GROW}
            children.append(self.proc(s, "c", index, steps, extra))
        state["children"] = children
        self.queue[0:0] = children

    def remove_fork_files(self, sid):
        state = self.session_state[sid]
        if state["removed"]:
            return
        state["removed"] = True
        for path in glob.glob(f"{self.huge}/m{sid}") + glob.glob(f"{self.huge}/m{sid}.*") + \
                glob.glob(f"{self.work}/state.s{sid}"):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def handle(self, p, kind, f, now):
        s = p.session
        sid = s["sid"]
        if kind == "DECODED":
            p.decoded_first = True
        if kind == "MARK":
            name = f.get("name", "")
            ends = [e["mono_ns"] for e in self.timing_events(p) if e["event"]=="generation_end"]
            if not ends:
                raise RuntimeError("tool timer lacks generation end")
            at = ends[-1] / 1e9
            if p.role == "a" and name in ("t1", "t2"):
                turn = int(name[1])
                self.timers.append((at + s["tools"][turn - 1], self.go(sid, "a", 0, turn)))
            elif p.role == "c" and name == "t1":
                self.timers.append((at + s["child_tools"][p.index], self.go(sid, "c", p.index, 1)))
        if kind == "PUBLISHED" and p.role == "a" and s["kind"] == "fork":
            self.fork(p, f, now)
        if kind == "ATTACHED" and p.role == "c":
            state = self.session_state[sid]
            state["attached"] += 1
            if state["attached"] == FORK_K:
                self.remove_fork_files(sid)

    def finish(self, p, now):
        p.exit = p.popen.returncode
        self.live.remove(p)
        self.done.append(p)
        s = p.session
        if p.role == "c":
            state = self.session_state[s["sid"]]
            if all(c.exit is not None for c in state["children"]):
                self.remove_fork_files(s["sid"])
                self.release(self.go(s["sid"], "a", 0, 9))

    def guard(self):
        avail = meminfo("MemAvailable") - self.model_mib
        if avail < GUARD_GIB * 1024 and self.live:
            newest = max(self.live, key=lambda q: q.launch)
            newest.popen.kill()
            self.guard_events += 1
            self.write({"event": "guard", "t": time.monotonic() - self.t0, "avail_mib": avail,
                        "sid": newest.session["sid"], "role": newest.role, "index": newest.index})

    def write(self, record):
        self.log.write(json.dumps(record) + "\n")

    def run(self):
        trace = sorted(self.trace, key=lambda s: s["arrival"])
        before = meminfo("MemAvailable")
        tmpfs_before = tmpfs_mib(self.huge)
        fs = os.statvfs(self.huge)
        tmpfs_before_bytes = (fs.f_blocks-fs.f_bfree)*fs.f_frsize
        tmpfs_peak = tmpfs_before
        self.t0_ns = time.monotonic_ns()
        self.t0 = self.t0_ns / 1e9
        self.write({"event":"arrival_clock","mono_ns":self.t0_ns,"budget_gib":STATE_BUDGET_GIB,
                    "accounting_baseline_mib":self.accounting_baseline_mib,"model_mib":self.model_mib})
        next_sample, arrived, ended = 0.0, 0, "drained"
        while True:
            now = time.monotonic()
            t = now - self.t0
            while arrived < len(trace) and trace[arrived]["arrival"] <= t:
                self.arrive(trace[arrived])
                arrived += 1
            for p in list(self.live):
                for kind, f in self.read(p):
                    self.handle(p, kind, f, now)
                if p.popen.poll() is not None:
                    for kind, f in self.read(p):
                        self.handle(p, kind, f, now)
                    self.finish(p, now)
            due = [x for x in self.timers if x[0] <= now]
            if due:
                self.timers = [x for x in self.timers if x[0] > now]
                for _, path in due:
                    self.release(path)
            self.guard()
            self.admit(now)
            if t >= next_sample:
                used = tmpfs_mib(self.huge)
                tmpfs_peak = max(tmpfs_peak, used)
                self.write({"event": "sample", "t": round(t, 2), "avail_mib": meminfo("MemAvailable"),
                            "drop_mib": before - meminfo("MemAvailable"), "live": len(self.live),
                            "queue": len(self.queue), "tmpfs_mib": used,
                            "mono_ns":time.monotonic_ns(), "kreclaimable_mib":meminfo("KReclaimable"),
                            "total_state_accounted_mib":max(0,self.accounting_baseline_mib-meminfo("MemAvailable"))+self.model_mib})
                next_sample = t + SAMPLE_S
            if arrived == len(trace) and not self.live and not self.queue:
                break
            if t > LIMIT_S:
                ended = "limit"
                break
            time.sleep(0.05)
        unfinished_live = list(self.live)
        for p in unfinished_live:
            p.popen.kill()
        for p in unfinished_live:
            p.popen.wait()
            self.read(p)
            p.exit = "unfinished"
            self.live.remove(p)
            self.done.append(p)
        for p in self.queue:
            p.exit = "queued"
            self.done.append(p)
        self.queue = []
        for sid in list(self.session_state):
            self.remove_fork_files(sid)
        time.sleep(5)
        after = meminfo("MemAvailable")
        tmpfs_after = tmpfs_mib(self.huge)
        fs = os.statvfs(self.huge)
        tmpfs_after_bytes = (fs.f_blocks-fs.f_bfree)*fs.f_frsize
        for p in self.done:
            record = {"event": "proc", "sid": p.session["sid"], "kind": p.session["kind"],
                      "arrival": p.session["arrival"], "role": p.role, "index": p.index,
                      "launch": None if p.launch is None else round(p.launch - self.t0, 3),
                      "exit": p.exit, "events": p.events,
                      "pid":p.popen.pid if p.popen else None,
                      "launch_ns":getattr(p,"launch_ns",None),
                      "timing_paths":glob.glob(f"{self.work}/timing/{p.popen.pid}-*.jsonl") if p.popen else [],
                      "output_path":p.out}
            if p.role == "c":
                state = self.session_state[p.session["sid"]]
                record["publish_seen"] = round(state.get("publish_seen", self.t0) - self.t0, 3)
                record["publish_ms"] = state.get("publish_ms", 0.0)
            self.write(record)
        self.write({"event": "end", "ended": ended, "t": round(time.monotonic() - self.t0, 2),
                    "before_mib": before, "after_mib": after, "residual_mib": before - after,
                    "tmpfs_before_mib": tmpfs_before, "tmpfs_peak_mib": tmpfs_peak,
                    "tmpfs_after_mib": tmpfs_after,
                    "tmpfs_before_bytes":tmpfs_before_bytes,"tmpfs_after_bytes":tmpfs_after_bytes,
                    "guard_events": self.guard_events, "sessions": len(trace)})


def publish(system, work, huge):
    env_ = dict(os.environ)
    env_["CUDA_VISIBLE_DEVICES"] = GPU
    for key in ("STATOR_KV_HOST", "STATOR_KV_CHAIN", "STATOR_KV_PREFIX", "STATOR_KV_GROW", "STATOR_HOST_PTR",
                "LD_LIBRARY_PATH"):
        env_.pop(key, None)
    driver = STOCK_DRIVER
    if system != "none":
        env_.update({"STATOR_HOST_PTR": "1", "LD_LIBRARY_PATH": CHAIN_LIB})
        driver = DRIVER
    if system == "both":
        env_.update({"STATOR_KV_HOST": f"{huge}/r", "STATOR_KV_GROW": GROW})
    env_.update(STATOR_TIMING_DIR=work + "/timing", STATOR_TRACE="1")
    out = os.path.join(work, "publish.out")
    with open(out, "w") as o, open(out + ".err", "w") as e:
        subprocess.run([driver, MODEL, str(CONTEXT), "1", f"text:{PREFIX_FILE}", f"save:{work}/state.root"],
                       env=env_, stdout=o, stderr=e, check=True)
    tokens = 0
    for line in open(out):
        if line.startswith("PUBLISHED"):
            tokens = int(fields(line).get("tokens", "0"))
    if system == "both":
        for path in glob.glob(f"{huge}/r*"):
            os.chmod(path, 0o400)
    return tokens


def main():
    global DRIVER, STOCK_DRIVER, CHAIN_LIB, LIMIT_S, N_GEN, WINDOW_S, SEED, FORK_K
    if not MEASURED_TRACE:
        raise SystemExit("MEASURED_TRACE is required")
    spec = json.loads(Path(MEASURED_TRACE).read_text())
    if len(SYSTEMS)!=1:
        raise SystemExit("one system per frozen trace job")
    system = SYSTEMS[0]
    if system not in NAMES:
        raise SystemExit("unknown system")
    DRIVER = str(ROOT/"bin/timed/session_agent_chains")
    STOCK_DRIVER = str(ROOT/"bin/timed/session_agent_stock")
    CHAIN_LIB = HERE + "/engines/chains/build/bin"
    LIMIT_S, N_GEN, WINDOW_S, SEED, FORK_K = (spec[k] for k in ("trace_derived_limit_s","n_gen","window_s","seed","fork_k"))
    if spec["memory_budget_gib"]!=STATE_BUDGET_GIB or spec["reserve_gib"]!=RESERVE_GIB or spec["guard_gib"]!=GUARD_GIB:
        raise SystemExit("safety/budget manifest mismatch")
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
        raise SystemExit("another CUDA process is active")
    result_dir = Path(RESULT_ROOT) / RESULT_TAG
    result_dir.mkdir()
    work = result_dir / "work"
    work.mkdir()
    (work/"timing").mkdir()
    huge = result_dir / "tmpfs"
    huge.mkdir()
    (result_dir/"trace.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n")
    meta = dict(spec)
    meta.pop("sessions")
    meta.update(system=system,sessions=len(spec["sessions"]),estimate_mib=ESTIMATE,
        driver=DRIVER,stock_driver=STOCK_DRIVER,clock="CLOCK_MONOTONIC",gpu=GPU,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(Path(MEASURED_TRACE).read_bytes()).hexdigest(),
        builds=json.loads((ROOT/"bin/timed/builds.json").read_text()))
    (result_dir/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    anchor = None
    mounted = False
    run = None
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        sudo("mount","-t","tmpfs","-o",f"huge=always,size={TMPFS_MIB}m,uid={os.getuid()},gid={os.getgid()}","tmpfs",str(huge))
        mounted=True
        with (result_dir/"anchor.log").open("w") as f:
            anchor=subprocess.Popen([HERE + "/bin/linux_share_probe","hold","86400"],env=os.environ|{"CUDA_VISIBLE_DEVICES":GPU},stdout=f,stderr=f,start_new_session=True)
        deadline=time.monotonic()+30
        while not bh_snapshot():
            if anchor.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError("CUDA anchor failed")
            time.sleep(.1)
        sudo("sh","-c","echo 1 >/proc/sys/vm/drop_caches")
        with open(MODEL,"rb") as f:
            while f.read(8*2**20):pass
        time.sleep(.5)
        baseline=meminfo("MemAvailable")
        faults=bh_snapshot()
        tokens=publish(system,str(work),str(huge))
        with (result_dir/"raw.jsonl").open("w",buffering=1) as log:
            log.write(json.dumps({"event":"begin","system":system,"rate":spec["rate"],"root_tokens":tokens,
                "sessions":len(spec["sessions"]),"huge_used_mib":tmpfs_mib(huge),"kreclaimable_mib":meminfo("KReclaimable")})+"\n")
            run=Run(system,spec["rate"],spec["sessions"],str(work),str(huge),log)
            run.root_tokens=tokens
            run.accounting_baseline_mib=baseline
            run.run()
            log.write(json.dumps({"event":"fault_counter","total":bh_delta(faults,bh_snapshot())})+"\n")
        (result_dir/"completion.json").write_text(json.dumps({"runner_completed":True,"guard_events":run.guard_events})+"\n")
    finally:
        kill_group(anchor)
        anchor = None
        if run is not None:
            for process in run.live:
                if process.popen.poll() is None:
                    process.popen.kill()
                process.popen.wait(timeout=30)
        kill_group(anchor)
        for path in list(huge.glob("*"))+list(work.glob("state.*")):
            if path.is_file():path.unlink()
        if mounted:
            subprocess.run(["sudo","-n","umount",str(huge)],check=True)
        huge.rmdir()
    print(f"measured_dynamic_result_dir={result_dir}",flush=True)


def pct(values, q):
    if not values:
        return ""
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(result_dir):
    runs, current = [], None
    for line in open(os.path.join(result_dir, "raw.jsonl")):
        r = json.loads(line)
        if r["event"] == "begin":
            current = {"begin": r, "samples": [], "procs": [], "guards": 0}
            runs.append(current)
        elif r["event"] == "sample":
            current["samples"].append(r)
        elif r["event"] == "proc":
            current["procs"].append(r)
        elif r["event"] == "guard":
            current["guards"] += 1
        elif r["event"] == "end":
            current["end"] = r
    rows, texts = [], {}
    for run in runs:
        system, rate = run["begin"]["system"], run["begin"]["rate"]
        ttft, turn, child, pause, failed, unfinished, completed = [], [], [], [], 0, 0, set()
        censored = []
        sessions = {}
        for p in run["procs"]:
            sid = p["sid"]
            ev = p["events"]
            ok = p["exit"] == 0
            if p["exit"] in ("unfinished", "queued"):
                unfinished += 1
            elif not ok:
                failed += 1
            sessions.setdefault(sid, []).append(ok)
            decoded = [f for k, f in ev if k == "DECODED"]
            if p["role"] == "a" and decoded and p["launch"] is not None:
                ttft.append(p["launch"] - p["arrival"] + float(decoded[0]["since_start_ms"]) / 1000)
            elif p["role"] == "a":
                censored.append(float(run.get("end", {}).get("t", LIMIT_S)) - p["arrival"])
            if p["role"] == "c" and decoded and p["launch"] is not None:
                child.append(p["launch"] - p["publish_seen"] + float(decoded[0]["since_start_ms"]) / 1000)
            released = [float(f["since_start_ms"]) for k, f in ev if k == "RELEASED"]
            for r_ms in released:
                after = [float(f["since_start_ms"]) for f in decoded if float(f["since_start_ms"]) > r_ms]
                if after:
                    turn.append((after[0] - r_ms) / 1000)
            pause += [float(f["publish_ms"]) for k, f in ev if k == "PUBLISHED"]
            text = [f["hash"] for k, f in ev if k == "TEXT"]
            if ok and text:
                texts.setdefault((rate, sid, p["role"], p["index"]), {})[system] = text[0]
        for sid, oks in sessions.items():
            if all(oks):
                completed.add(sid)
        samples = run["samples"]
        drops = [s["drop_mib"] for s in samples]
        end = run.get("end", {})
        rows.append({
            "system": system, "name": NAMES[system], "rate": rate, "sessions": run["begin"]["sessions"],
            "completed": len(completed), "failed_procs": failed, "unfinished_procs": unfinished,
            "guard_events": run["guards"], "ended": end.get("ended", ""), "wall_s": end.get("t", ""),
            "ttft_p50_s": pct(ttft, 0.5), "ttft_p99_s": pct(ttft, 0.99),
            "ttft_p99_with_unserved_s": pct(ttft + censored, 0.99), "unserved_sessions": len(censored),
            "ttft_mean_s":
                statistics.mean(ttft) if ttft else "",
            "turn_p50_s": pct(turn, 0.5), "turn_p99_s": pct(turn, 0.99),
            "child_p50_s": pct(child, 0.5), "child_p99_s": pct(child, 0.99),
            "fork_pause_ms": statistics.mean(pause) if pause else "",
            "mem_peak_gib": max(drops) / 1024 if drops else "", "mem_mean_gib": statistics.mean(drops) / 1024
            if drops else "", "live_peak": max(s["live"] for s in samples) if samples else "",
            "live_mean": statistics.mean(s["live"] for s in samples) if samples else "",
            "queue_peak": max(s["queue"] for s in samples) if samples else "",
            "residual_gib": end.get("residual_mib", 0) / 1024,
            "tmpfs_before_mib": end.get("tmpfs_before_mib", ""), "tmpfs_peak_mib": end.get("tmpfs_peak_mib", ""),
            "tmpfs_after_mib": end.get("tmpfs_after_mib", "")})
    for row in rows:
        same = compared = 0
        for key, by_system in texts.items():
            if key[0] != row["rate"] or row["system"] not in by_system or len(by_system) < 2:
                continue
            others = [v for s, v in by_system.items() if s != row["system"]]
            compared += 1
            same += all(v == by_system[row["system"]] for v in others)
        row["texts_equal"] = f"{same}/{compared}"
    path = os.path.join(result_dir, "engine_kvdynamic_summary.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["system"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})
    gates = {"D1": all(r["failed_procs"] == 0 and r["guard_events"] == 0 for r in rows),
             "D2": all(r["texts_equal"].split("/")[0] == r["texts_equal"].split("/")[1] for r in rows),
             "D3": all(r["tmpfs_after_mib"] != "" and r["tmpfs_after_mib"] - r["tmpfs_before_mib"] < 2
                       for r in rows if r["system"] == "both"),
             "D4": all(r["completed"] == r["sessions"] for r in rows
                       if r["system"] in ("both", "weights") and rows and r["rate"] == min(x["rate"] for x in rows))}
    with open(os.path.join(result_dir, "gates.txt"), "w") as handle:
        for key, value in gates.items():
            handle.write(f"{key} {'PASS' if value else 'FAIL'}\n")


if __name__ == "__main__":
    if os.environ.get("SUMMARIZE"):
        summarize(os.environ["SUMMARIZE"])
    else:
        main()
