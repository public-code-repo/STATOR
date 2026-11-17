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
import paths

HERE = paths.ROOT


def env(name, default):
    return os.environ.get(name, default)


LIMIT_S = float(env("LIMIT_S", "1080"))
N_GEN = int(env("N_GEN", "32"))
CONTEXT = int(env("CONTEXT", "16384"))
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
        with open(p.out, "w") as out, open(p.out + ".err", "w") as err:
            p.popen = subprocess.Popen(p.argv, env=e, stdout=out, stderr=err)
        p.launch = now
        self.live.append(p)

    def admit(self, now):
        while self.queue:
            avail = meminfo("MemAvailable") - self.model_mib
            pending = sum(ESTIMATE[self.system] for p in self.live if not p.decoded_first)
            if avail - pending - ESTIMATE[self.system] < RESERVE_GIB * 1024:
                return
            self.start(self.queue.pop(0), now)

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
        state["publish_seen"] = now
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
            at = p.launch + float(f.get("since_start_ms", "0")) / 1000.0
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
                open(self.go(s["sid"], "a", 0, 9), "w").close()

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
        tmpfs_peak = tmpfs_before
        self.t0 = time.monotonic()
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
                    open(path, "w").close()
            self.guard()
            self.admit(now)
            if t >= next_sample:
                used = tmpfs_mib(self.huge)
                tmpfs_peak = max(tmpfs_peak, used)
                self.write({"event": "sample", "t": round(t, 2), "avail_mib": meminfo("MemAvailable"),
                            "drop_mib": before - meminfo("MemAvailable"), "live": len(self.live),
                            "queue": len(self.queue), "tmpfs_mib": used})
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
        for p in self.done:
            record = {"event": "proc", "sid": p.session["sid"], "kind": p.session["kind"],
                      "arrival": p.session["arrival"], "role": p.role, "index": p.index,
                      "launch": None if p.launch is None else round(p.launch - self.t0, 3),
                      "exit": p.exit, "events": p.events}
            if p.role == "c":
                state = self.session_state[p.session["sid"]]
                record["publish_seen"] = round(state.get("publish_seen", self.t0) - self.t0, 3)
                record["publish_ms"] = state.get("publish_ms", 0.0)
            self.write(record)
        self.write({"event": "end", "ended": ended, "t": round(time.monotonic() - self.t0, 2),
                    "before_mib": before, "after_mib": after, "residual_mib": before - after,
                    "tmpfs_before_mib": tmpfs_before, "tmpfs_peak_mib": tmpfs_peak,
                    "tmpfs_after_mib": tmpfs_after,
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


