import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from live_publish import ROOT, GPU, PARAGRAPH, meminfo, allocated, write_json, tagged, bh_snapshot, bh_delta, kill_group, load_trace, phase, generation, generation_window_tps, text_hash


class Case:
    def __init__(self, args, result, mount, name, mode, count):
        self.args, self.mount = args, mount
        self.path = result / name
        self.path.mkdir()
        (self.path / 'timing').mkdir()
        (self.path / 'payload').mkdir()
        self.mode, self.count = mode, count
        self.procs, self.details = {}, {}
        self.stop = threading.Event()
        self.problem = None
        self.thread = None
        self.record = dict(case=name, mode=mode, count=count, status='preparing')

    def sample(self):
        m = meminfo()
        return dict(mono_ns=time.monotonic_ns(), mem_available=m['MemAvailable'],
                    k_reclaimable=m['KReclaimable'], tmpfs_allocated=allocated(self.mount),
                    live=sum(p.poll() is None for p in list(self.procs.values())))

    def watch(self):
        try:
            with (self.path / 'samples.jsonl').open('w') as f:
                while not self.stop.is_set():
                    s = self.sample()
                    self.lowest = min(self.lowest, s['mem_available'])
                    self.tmpfs_peak = max(self.tmpfs_peak, s['tmpfs_allocated'])
                    f.write(json.dumps(s) + '\n')
                    if s['mem_available'] < self.args.guard_gib * 2**30:
                        self.problem = 'guard'
                    if (s['mono_ns'] - self.start_ns) / 1e9 > self.args.timeout:
                        self.problem = 'timeout'
                    if self.problem:
                        for p in list(self.procs.values()):
                            kill_group(p)
                        return
                    self.stop.wait(.05)
        except BaseException as e:
            self.problem = f'monitor_error: {e}'
            for p in list(self.procs.values()):
                kill_group(p)

    def start(self):
        estimate = {'copy': 1750, 'flat': 900, 'chain': 850, 'extent': 720}[self.mode]
        required = (self.count * estimate + 8192) * 2**20
        self.record['admission_estimated_bytes'] = required
        if meminfo()['MemAvailable'] - required < self.args.reserve_gib * 2**30:
            self.record.update(status='skip', reason='admission_estimate')
            write_json(self.path / 'result.json', self.record)
            return False
        subprocess.run(['sudo', '-n', 'sh', '-c', 'echo 1 > /proc/sys/vm/drop_caches'], check=True)
        with self.args.model.open('rb') as f:
            while f.read(8 * 1024 * 1024):
                pass
        time.sleep(.5)
        self.before = meminfo()
        self.lowest = self.before['MemAvailable']
        self.tmpfs_peak = allocated(self.mount)
        self.start_ns = time.monotonic_ns()
        self.fault_before = bh_snapshot()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return True

    def spawn(self, name, steps, extra, generating=False):
        self.check()
        env = {k:v for k,v in os.environ.items() if not k.startswith('STATOR_KV_')
               and k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
        env.update(CUDA_VISIBLE_DEVICES=GPU, STATOR_HOST_PTR='1',
                   STATOR_TIMING_DIR=str(self.path / 'timing'), STATOR_TRACE='1')
        env.update(extra)
        driver = 'session_agent_chains' if self.args.scenario == 'tree' else 'session_agent_extents'
        cmd = [str(ROOT/'bin/timed' / driver), str(self.args.model), str(self.args.context),
               str(self.args.tokens), *steps]
        with (self.path / f'{name}.out').open('w') as out, (self.path / f'{name}.err').open('w') as err:
            p = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
        self.procs[name] = p
        self.details[name] = dict(pid=p.pid, command=cmd, generating=generating,
            environment={k:v for k,v in env.items() if k.startswith(('LLAMA_', 'GGML_', 'STATOR_', 'CUDA_'))})
        write_json(self.path / 'processes.json', self.details)
        return p

    def check(self):
        if self.problem:
            raise RuntimeError(self.problem)

    def wait_line(self, name, tag):
        while True:
            self.check()
            rows = tagged(self.path / f'{name}.out', tag)
            if rows:
                return rows[-1]
            if self.procs[name].poll() is not None:
                raise RuntimeError(f'{name} exited before {tag}')
            time.sleep(.03)

    def wait_done(self, names=None):
        for name in names or list(self.procs):
            while self.procs[name].poll() is None:
                self.check()
                time.sleep(.03)
            if self.procs[name].returncode != 0:
                raise RuntimeError(f'{name} exit={self.procs[name].returncode}')

    def release(self, name):
        now = time.monotonic_ns()
        (self.path / name).touch()
        self.record[name + '_release_ns'] = now
        return now

    def finish(self, error=None):
        for p in self.procs.values():
            kill_group(p)
        if self.thread is not None:
            self.stop.set()
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                error = error or RuntimeError('monitor did not stop')
            self.record.update(memory_before=self.before, memory_min_available=self.lowest,
                total_peak_accounted_bytes=self.before['MemAvailable']-self.lowest+self.args.model.stat().st_size,
                warmed_model_file_bytes=self.args.model.stat().st_size,
                gpu_faults_total=bh_delta(self.fault_before, bh_snapshot()),
                tmpfs_peak_bytes=self.tmpfs_peak,
                tmpfs_before_unlink_bytes=allocated(self.mount),
                wall_s=(time.monotonic_ns()-self.start_ns)/1e9)
        try:
            if error:
                raise error
            traces = load_trace(self.path / 'timing')
            if set(traces) != {p.pid for p in self.procs.values()}:
                raise ValueError('trace pid coverage differs')
            generations = {}
            for name, p in self.procs.items():
                r = self.details[name]
                r.update(exit=p.returncode, text_sha256=text_hash(self.path / f'{name}.out'),
                         result=tagged(self.path / f'{name}.out', 'RESULT'))
                if p.returncode != 0 or len(r['result']) != 1:
                    raise ValueError(f'incomplete process {name}')
                if r['generating']:
                    generations[name] = generation(traces[p.pid], self.args.tokens)
                    r['generation'] = generations[name]
                    e = traces[p.pid]
                    released = [v['mono_ns'] for v in e if v['event']=='released']
                    if released:
                        r['release_to_first_ms'] = (generations[name]['first_token_ns']-released[-1])/1e6
            self.record.update(status='ok', processes=self.details, generations=generations,
                generation_window_tps=generation_window_tps(list(generations.values())))
            leaves = [g for n,g in generations.items() if n.startswith('leaf')]
            if leaves:
                self.record['leaf_window_tps'] = generation_window_tps(leaves)
        except BaseException as e:
            self.record.update(status=self.problem or 'invalid', error=f'{type(e).__name__}: {e}')
        finally:
            for directory in (self.mount, self.path / 'payload'):
                for path in directory.iterdir():
                    if path.is_file():
                        path.unlink()
            self.record['tmpfs_after_unlink_bytes'] = allocated(self.mount)
            write_json(self.path / 'processes.json', self.details)
            write_json(self.path / 'result.json', self.record)
        if self.record['status'] != 'ok':
            raise RuntimeError(self.record.get('error', self.record['status']))
        return self.record


def tree(c, fanouts):
    a = c.args
    prefix = c.path / 'prefix.txt'
    prefix.write_text(PARAGRAPH * a.paragraphs)
    state = lambda name: str(c.path / 'payload' / name)
    host = lambda name: str(c.mount / name)
    shared = c.mode != 'copy'
    root_env = {'STATOR_KV_HOST':host('root'), 'STATOR_KV_GROW':'256'} if shared else {}
    c.spawn('root', [f'text:{prefix}', f'save:{state("root")}', f'wait:{c.path}/go',
                    'say:\n\nTask for the root: list the kernel mechanisms named above.', 'gen'], root_env, True)
    tokens = {'root':int(c.wait_line('root', 'PUBLISHED')['tokens'])}
    chains = {'root':f'{host("root")}:{tokens["root"]}'}
    contexts = {'root':[]}
    parents = ['root']
    leaf_number = 0
    for depth, fanout in enumerate(fanouts):
        next_parents = []
        pending = []
        leaf_level = depth == len(fanouts)-1
        for parent in parents:
            for index in range(fanout):
                name = f'leaf{leaf_number}' if leaf_level else f'{parent}_{index}'
                extra = {}
                if c.mode == 'copy':
                    steps = [f'load:{state(parent)}']
                elif c.mode == 'flat':
                    extra = {'STATOR_KV_HOST':host('root'), 'STATOR_KV_PREFIX':str(tokens['root']), 'STATOR_KV_GROW':'256'}
                    steps = [f'load:{state("root")}', *['say:'+s for s in contexts[parent]]]
                else:
                    extra = {'STATOR_KV_CHAIN':chains[parent], 'STATOR_KV_HOST':'anon' if leaf_level else host(name), 'STATOR_KV_GROW':'256'}
                    steps = [f'load:{state(parent)}']
                if leaf_level:
                    steps += [f'mark:ready',
                              f'say:\n\nTask for agent {leaf_number}: summarize the text above in {leaf_number+2} sentences.', 'gen']
                    leaf_number += 1
                else:
                    own = '\n\n' + ''.join((f'Group {index} note {j}: ' if len(fanouts)==2 else f'Node {name} note {j}: ')+'the group reviews how the kernel handles memory pressure, which pages it reclaims first, and how a task that waits for a page is resumed. ' for j in range(1,a.level_paragraphs+1))
                    contexts[name] = contexts[parent]+[own]
                    steps += ['say:'+own]
                    if c.mode != 'flat':
                        steps += [f'save:{state(name)}']
                    steps += ['mark:ready', f'wait:{c.path}/go',
                              (f'say:\n\nTask for the leader of group {index}: state the goal of the group in one sentence.' if len(fanouts)==2 else f'say:\n\nTask for the leader of node {name}: state the goal of the group in one sentence.'), 'gen']
                    next_parents.append(name)
                c.spawn(name, steps, extra, True)
                pending.append((name,parent))
        if leaf_level:
            c.release('go')
        for name,parent in pending:
            c.wait_line(name,'MARK')
            if not leaf_level and c.mode != 'flat':
                tokens[name] = int(c.wait_line(name,'PUBLISHED')['tokens'])
                chains[name] = chains[parent]+f',{host(name)}:{tokens[name]}'
        parents = next_parents
    c.record['all_attached'] = c.sample()
    for path in c.mount.iterdir():
        if path.is_file():
            path.unlink()
    c.record['after_unlink'] = c.sample()
    c.wait_done()
    c.record['after_all_exit'] = c.sample()


def resident(c, count):
    a = c.args
    prefix = ROOT/'workloads/agent_prefix.txt'
    tasks = (ROOT/'workloads/agent_tasks.txt').read_text().splitlines()
    state = c.path / 'payload/root'
    host = c.mount / 'root'
    extra = {'STATOR_KV_HOST':str(host), 'STATOR_KV_GROW':'256'} if c.mode=='extent' else {}
    c.spawn('publisher', [f'text:{prefix}', f'save:{state}'], extra)
    tokens = int(c.wait_line('publisher','PUBLISHED')['tokens'])
    c.wait_done(['publisher'])
    c.record['after_publish'] = c.sample()
    if c.mode == 'extent':
        extra['STATOR_KV_PREFIX'] = str(tokens)
    for index in range(count):
        suffix = f'\n\nTask for agent {index}: {tasks[index%len(tasks)]}\n'
        steps = [f'load:{state}', 'say:'+suffix, 'mark:ready']
        if index < a.active:
            steps += [f'wait:{c.path}/active',
                      f'say:\n\nTool result for agent {index}: {{"status": "ok", "items": {index+1}}}\nNext call or final answer:\n', 'gen']
        else:
            steps += [f'wait:{c.path}/idle']
        c.spawn(f'agent{index}', steps, extra, index<a.active)
    for index in range(count):
        c.wait_line(f'agent{index}', 'MARK')
    end = time.monotonic()+a.settle_s
    while time.monotonic()<end:
        c.check()
        time.sleep(.05)
    c.record['resident_ready'] = c.sample()
    c.record['resident_accounted_bytes'] = (c.before['MemAvailable']-c.record['resident_ready']['mem_available']+a.model.stat().st_size)
    c.release('active')
    c.wait_done([f'agent{i}' for i in range(a.active)])
    c.record['active_done'] = c.sample()
    c.release('idle')
    c.wait_done()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--scenario', choices=['tree','resident'], required=True)
    p.add_argument('--repetitions', type=int, default=6)
    p.add_argument('--modes', default='copy chain')
    p.add_argument('--fanouts', default='2,4')
    p.add_argument('--counts', default='8 16')
    p.add_argument('--active', type=int, default=8)
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--paragraphs', type=int, default=320)
    p.add_argument('--level-paragraphs', type=int, default=20)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--settle-s', type=float, default=2)
    p.add_argument('--reserve-gib', type=float, default=16)
    p.add_argument('--guard-gib', type=float, default=12)
    p.add_argument('--timeout', type=float, default=1200)
    p.add_argument('--model', type=Path, default=ROOT/'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf')
    a = p.parse_args()
    if '/' in a.tag or a.tag in ('.','..') or a.repetitions<2 or a.repetitions%2 or a.tokens<1:
        p.error('invalid tag, repetitions, or token count')
    modes = a.modes.split()
    allowed = ['copy','flat','chain'] if a.scenario=='tree' else ['copy','extent']
    if not modes or any(m not in allowed for m in modes):
        p.error('invalid modes')
    cells = [list(map(int,s.split(','))) for s in a.fanouts.split()] if a.scenario=='tree' else [int(s) for s in a.counts.split()]
    if a.scenario=='tree' and any(not f or min(f)<1 for f in cells):
        p.error('invalid fanout')
    if a.scenario=='resident' and any(n<a.active for n in cells):
        p.error('resident count below active count')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise SystemExit('another CUDA process is active')
    result = ROOT/'results'/a.tag
    result.mkdir()
    mount = result/'tmpfs'
    mount.mkdir()
    write_json(result/'metadata.json', dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        gpu=GPU, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        builds=json.loads((ROOT/'bin/timed/builds.json').read_text()),
        memory_definition='warmed-file baseline drop plus model file bytes once', clock='CLOCK_MONOTONIC'))
    anchor = None
    mounted = False
    records = []
    try:
        subprocess.run(['sudo','-n','mount','-t','tmpfs','-o',f'huge=always,size=8192m,uid={os.getuid()},gid={os.getgid()}','tmpfs',str(mount)],check=True)
        mounted = True
        with (result/'anchor.log').open('w') as log:
            anchor = subprocess.Popen([str(ROOT/'bin/linux_share_probe'),'hold','86400'],
                env=os.environ|{'CUDA_VISIBLE_DEVICES':GPU},stdout=log,stderr=log,start_new_session=True)
        deadline = time.monotonic()+30
        while not bh_snapshot():
            if anchor.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('CUDA anchor failed')
            time.sleep(.1)
        for run in range(1,a.repetitions+1):
            for cell in cells:
                count = cell if a.scenario=='resident' else 1
                if a.scenario=='tree':
                    width = 1
                    for f in cell:
                        width *= f
                        count += width
                label = 'n'+str(cell) if a.scenario=='resident' else 'f'+'x'.join(map(str,cell))
                for mode in ((modes if run%2 else list(reversed(modes))) if len(modes)==2 else modes[(run-1)%len(modes):]+modes[:(run-1)%len(modes)]):
                    name = f'r{run:02d}-{mode}-{label}'
                    print('START',name,flush=True)
                    c = Case(a,result,mount,name,mode,count)
                    c.record.update(run=run, scenario=a.scenario, cell=cell)
                    if c.start():
                        error = None
                        try:
                            tree(c,cell) if a.scenario=='tree' else resident(c,cell)
                        except BaseException as e:
                            error = e
                        c.finish(error)
                    records.append(c.record)
                    write_json(result/'cases.json',records)
                    print('END',name,c.record['status'],flush=True)
        write_json(result/'completion.json',dict(completed=True,cases=len(records),skipped=sum(r['status']=='skip' for r in records)))
    finally:
        kill_group(anchor)
        if mounted:
            subprocess.run(['sudo','-n','umount',str(mount)],check=True)
        mount.rmdir()


if __name__ == '__main__':
    main()
