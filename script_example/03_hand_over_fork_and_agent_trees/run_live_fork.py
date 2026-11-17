import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import paths

ROOT = Path(paths.ROOT)
GPU = os.environ['STATOR_GPU']
PARAGRAPH = ('The scheduler keeps a run queue for every processor and moves a task between queues when the load is uneven. '
             'A page fault enters the kernel, which finds the mapping, allocates a frame, fills it, and updates the page table before the task resumes. ')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def meminfo():
    return {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}


def fields(line):
    return dict(x.split('=', 1) for x in line.split()[1:] if '=' in x)


def tagged(path, tag):
    if not path.exists():
        return []
    return [fields(line) for line in path.read_text(errors='replace').splitlines() if line.startswith(tag + ' ')]


def text_hash(path):
    if not path.exists():
        return None
    lines = [x for x in path.read_text(errors='replace').splitlines() if x.startswith('TEXT ')]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest() if lines else None


def bh_snapshot():
    counters = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            line = (path / 'stat').read_text(); end = line.rfind(')')
            name = line[line.find('(') + 1:end]
            if name.startswith('UVM GPU') and name.endswith(' BH'):
                s = line[end + 2:].split()
                counters[path.name] = int(s[7]) + int(s[9])
        except (OSError, ValueError):
            pass
    return counters


def bh_delta(before, after):
    if not before or before.keys() != after.keys():
        return None
    return sum(after[p] - before[p] for p in before)


def allocated(path):
    s = os.statvfs(path)
    return (s.f_blocks - s.f_bfree) * s.f_frsize


def mode_environment(mode, mount):
    env = {k: v for k, v in os.environ.items() if not k.startswith('STATOR_KV_') and k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES=GPU, STATOR_HOST_PTR='1', STATOR_TRACE='0')
    if mode in ('extent', 'copy_map', 'copy_delta'):
        env.update(STATOR_KV_HOST=str(mount / 'kv'), STATOR_KV_GROW='256')
        if mode == 'copy_map':
            env['STATOR_KV_PUBLISH_COPY'] = '1'
        if mode == 'copy_delta':
            env['STATOR_KV_PUBLISH_COPY'] = '2'
    elif mode != 'copy':
        raise ValueError(mode)
    return env


def run_case(args, result, mount, run, mode):
    name = f'r{run:02d}-{mode}'
    case = result / name; case.mkdir(); (case / 'out').mkdir(); (case / 'timing').mkdir()
    context = 4096
    while context < args.paragraphs * 52 + 512 + args.forks * args.step + 128:
        context *= 2
    prefix = case / 'prefix.txt'; prefix.write_text(PARAGRAPH * args.paragraphs)
    state = case / 'state'
    env = mode_environment(mode, mount)
    env['STATOR_TIMING_DIR'] = str(case / 'timing')
    if args.unlink:
        env['STATOR_REPEAT_UNLINK'] = '1'
    cmd = [str(ROOT/'bin/timed' / 'fork_publisher_publish_repeat'), str(args.model), str(context), str(prefix), str(state),
           str(args.tokens), str(args.forks), str(args.children), str(args.step), str(case / 'out'),
           str(ROOT/'bin/timed' / 'prefix_agent_publish_repeat'), GPU]
    record = {'case': name, 'run': run, 'mode': mode, 'forks': args.forks, 'children_per_fork': args.children,
              'step': args.step, 'paragraphs': args.paragraphs, 'context': context, 'command': cmd,
              'environment': {k: v for k, v in env.items() if k.startswith(('LLAMA_', 'GGML_', 'CUDA_', 'STATOR_'))}}
    subprocess.run(['sudo', '-n', 'sh', '-c', 'echo 1 > /proc/sys/vm/drop_caches'], check=True)
    with args.model.open('rb') as model:
        while model.read(8 * 1024 * 1024):
            pass
    time.sleep(.5)
    before = meminfo(); lowest = before['MemAvailable']; tmpfs_peak = allocated(mount)
    before_fault = bh_snapshot(); start = time.monotonic_ns(); aborted = None; proc = None
    raw = case / 'parent.out'
    try:
        with raw.open('w') as out, (case / 'parent.err').open('w') as err:
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
            while proc.poll() is None:
                m = meminfo(); lowest = min(lowest, m['MemAvailable']); tmpfs_peak = max(tmpfs_peak, allocated(mount))
                if m['MemAvailable'] < args.guard_gib * 1024 ** 3:
                    aborted = 'guard'; os.killpg(proc.pid, signal.SIGKILL); break
                if (time.monotonic_ns() - start) / 1e9 > args.timeout:
                    aborted = 'timeout'; os.killpg(proc.pid, signal.SIGKILL); break
                time.sleep(.05)
            rc = proc.wait(timeout=60)
        after_fault = bh_snapshot()
        published = tagged(raw, 'PUBLISHED'); parent = tagged(raw, 'RESULT'); exits = tagged(raw, 'CHILD')
        children = []
        for k in range(1, args.forks + 1):
            for i in range(args.children):
                path = case / 'out' / f'child.{k}.{i}'
                r = tagged(path, 'RESULT')
                children.append({'fork': k, 'index': i, 'result': r[0] if r else None, 'text_sha256': text_hash(path)})
        record.update(exit=rc, status=aborted or ('ok' if rc == 0 else 'failed'),
                      publishes=[{'fork': int(p['fork']), 'rows': int(p['rows']), 'publish_ms': float(p['publish_ms'])} for p in published],
                      parent=parent[0] if parent else None, parent_text_sha256=text_hash(raw), children=children,
                      children_exit=exits, peak_drop_bytes=before['MemAvailable'] - lowest,
                      total_peak_accounted_bytes=before['MemAvailable'] - lowest + args.model.stat().st_size,
                      tmpfs_peak_bytes=tmpfs_peak, gpu_faults_total=bh_delta(before_fault, after_fault),
                      wall_s=(time.monotonic_ns() - start) / 1e9)
        if record['status'] == 'ok' and (len(published) != args.forks or not parent or len(exits) != args.forks * args.children
                                         or any(x.get('exit') != '0' for x in exits)
                                         or any(c['result'] is None or int(c['result'].get('generated', -1)) != args.tokens for c in children)):
            record['status'] = 'invalid'
    finally:
        for owned in [*case.glob('state.*'), *mount.glob('kv*')]:
            if owned.is_file():
                owned.unlink()
        write_json(case / 'result.json', record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--repetitions', type=int, default=6)
    p.add_argument('--modes', default='copy copy_map copy_delta extent')
    p.add_argument('--paragraphs', type=int, default=320)
    p.add_argument('--forks', type=int, default=4)
    p.add_argument('--children', type=int, default=2)
    p.add_argument('--step', type=int, default=128)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--model', type=Path, default=ROOT/'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf')
    p.add_argument('--guard-gib', type=float, default=12)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--unlink', action='store_true', help='remove the files of the previous publish of STATOR (Copied File)')
    args = p.parse_args()
    if args.repetitions < 2 or args.repetitions % 2:
        raise SystemExit('positive even repetitions required')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise SystemExit('another CUDA compute process is active')
    modes = args.modes.split()
    result = ROOT/'results' / args.tag; result.mkdir()
    mount = result / 'tmpfs'; mount.mkdir()
    write_json(result / 'metadata.json', {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                          'gpu': GPU, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                          'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                          'driver_sha256': hashlib.sha256((ROOT/'bin/timed/fork_publisher_publish_repeat').read_bytes()).hexdigest(),
                                          'gqa': os.environ.get('STATOR_FATTN_HOST_GQA')})
    records = []; mounted = False; anchor = None
    try:
        subprocess.run(['sudo', '-n', 'mount', '-t', 'tmpfs', '-o', f'huge=always,size=16384m,uid={os.getuid()},gid={os.getgid()}',
                        'tmpfs', str(mount)], check=True)
        mounted = True
        with (result / 'anchor.log').open('w') as log:
            anchor = subprocess.Popen([str(ROOT/'bin/linux_share_probe'), 'hold', '86400'],
                                      env=os.environ | {'CUDA_VISIBLE_DEVICES': GPU}, stdout=log, stderr=log,
                                      start_new_session=True)
        deadline = time.monotonic() + 30
        while not bh_snapshot():
            if anchor.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('CUDA anchor did not create the fault threads')
            time.sleep(.1)
        for run in range(1, args.repetitions + 1):
            order = modes[(run - 1) % len(modes):] + modes[:(run - 1) % len(modes)]
            for mode in order:
                print('START', run, mode, flush=True)
                r = run_case(args, result, mount, run, mode)
                records.append(r); write_json(result / 'cases.json', records)
                print('END', r['case'], r['status'], 'publishes', [x['publish_ms'] for x in r.get('publishes', [])],
                      'peak_GiB', round(r.get('total_peak_accounted_bytes', 0) / 1024 ** 3, 2), flush=True)
                if r['status'] != 'ok':
                    raise RuntimeError('case failed')
        with (result / 'summary.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['case', 'run', 'mode', 'fork', 'rows', 'publish_ms', 'parent_ms', 'total_peak_gib', 'gpu_faults',
                        'children_first_token_ms_mean'])
            for r in records:
                ttft = [float(c['result']['first_token_ms']) for c in r['children']]
                for pub in r['publishes']:
                    w.writerow([r['case'], r['run'], r['mode'], pub['fork'], pub['rows'], pub['publish_ms'],
                                r['parent']['parent_ms'], round(r['total_peak_accounted_bytes'] / 2 ** 30, 3),
                                r['gpu_faults_total'], round(sum(ttft) / len(ttft), 1)])
        write_json(result / 'completion.json', {'completed': True, 'cases': len(records)})
        print('RESULT_DIR', result, flush=True)
    finally:
        if anchor is not None and anchor.poll() is None:
            os.killpg(anchor.pid, signal.SIGKILL); anchor.wait()
        if mounted:
            subprocess.run(['sudo', '-n', 'umount', str(mount)])


if __name__ == '__main__':
    main()
