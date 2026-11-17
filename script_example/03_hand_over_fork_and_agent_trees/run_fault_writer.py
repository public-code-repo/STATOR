import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_hand_over import ROOT, GPU, PARAGRAPH, allocated, kill_group, load_trace, meminfo, mode_environment, phase, tagged, text_hash, write_json

MODES = ('device_merged', 'extent')


def tokens_of(events):
    return [{'id': e['value'], 'mono_ns': e['mono_ns']} for e in events if e['event'] == 'token']


def run_case(args, result, mount, run, mode, fault):
    name = f'r{run:02d}-{mode}-{"fault" if fault else "control"}'
    case = result/name; case.mkdir(); (case/'timing').mkdir(); (case/'out').mkdir()
    record = {'case': name, 'run': run, 'mode': mode, 'fault': fault, 'children': args.children,
              'paragraphs': args.paragraphs, 'n_gen': args.tokens, 'status': 'preparing'}
    context = 4096
    while context < args.paragraphs*52+512: context *= 2
    prefix = case/'prefix.txt'; prefix.write_text(PARAGRAPH*args.paragraphs)
    state = case/'state'; go = case/'go'
    env = mode_environment(mode, mount)
    env.update(STATOR_TIMING_DIR=str(case/'timing'), STATOR_TRACE='1')
    if fault:
        env.update(STATOR_FAULT_WRITER=str(ROOT/'bin/timed/fault_writer'), STATOR_FAULT_DELAY_MS=str(args.delay_ms))
    cmd = [str(ROOT/'bin/timed/fault_publisher_publish'), str(args.model), str(context), str(prefix), str(state),
           str(args.tokens), str(go), str(case/'out'), str(ROOT/'bin/timed/prefix_agent_publish')]+[GPU]*args.children
    record.update(context=context, command=cmd, environment={k: v for k, v in env.items()
                  if k.startswith(('LLAMA_', 'GGML_', 'CUDA_', 'STATOR_'))})
    write_json(case/'result.json', record)
    subprocess.run(['sudo', '-n', 'sh', '-c', 'echo 1 > /proc/sys/vm/drop_caches'], check=True)
    with args.model.open('rb') as model:
        while model.read(8*1024*1024): pass
    time.sleep(.5)
    proc = None; aborted = None; published = False; release_ns = None
    start_ns = time.monotonic_ns()
    raw = case/'parent.out'
    try:
        with raw.open('w') as out, (case/'parent.err').open('w') as err:
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
            record['parent_pid'] = proc.pid
            while proc.poll() is None:
                if meminfo()['MemAvailable'] < args.guard_gib*1024**3:
                    aborted = 'guard'; kill_group(proc); break
                if (time.monotonic_ns()-start_ns)/1e9 > args.timeout:
                    aborted = 'timeout'; kill_group(proc); break
                if not published and tagged(raw, 'PUBLISHED'):
                    release_ns = time.monotonic_ns(); go.touch(); published = True
                time.sleep(.05)
            rc = proc.wait(timeout=30)
        record.update(exit=rc, status=aborted or ('ok' if rc == 0 else 'failed'), release_ns=release_ns,
                      wall_s=(time.monotonic_ns()-start_ns)/1e9)
        record['children_exit'] = tagged(raw, 'CHILD')
        record['writer_exit'] = tagged(raw, 'WRITER')
        record['write'] = tagged(case/'out'/'writer', 'WRITE')
        record['parent_text_sha256'] = text_hash(raw)
        record['child_text_sha256'] = [text_hash(case/'out'/f'child.{i}') for i in range(args.children)]
        traces = load_trace(case/'timing')
        parent = traces.pop(proc.pid)
        indices = {e['value']: e['sequence'] for e in parent if e['event'] == 'child_spawn'}
        record['parent_tokens'] = tokens_of(parent)
        record['publish_ms'] = (phase(parent, 'publish_end')-phase(parent, 'publish_start'))/1e6
        record['child_tokens'] = {str(indices[pid]): tokens_of(events) for pid, events in traces.items() if pid in indices}
        if record['status'] == 'ok' and len(record['children_exit']) != args.children:
            record['status'] = 'invalid'
        if fault and (len(record['write']) != 1 or len(record['writer_exit']) != 1):
            record['status'] = 'invalid'
    except BaseException as error:
        kill_group(proc)
        record.update(status=aborted or 'invalid', error=f'{type(error).__name__}: {error}')
        write_json(case/'result.json', record)
        raise
    finally:
        for owned in [state, *mount.glob('kv*')]:
            if owned.is_file(): owned.unlink()
        write_json(case/'result.json', record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--children', type=int, default=8)
    p.add_argument('--paragraphs', type=int, default=320)
    p.add_argument('--tokens', type=int, default=96)
    p.add_argument('--delay-ms', type=int, default=5000)
    p.add_argument('--modes', default=' '.join(MODES))
    p.add_argument('--model', type=Path, default=ROOT/'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf')
    p.add_argument('--guard-gib', type=float, default=12)
    p.add_argument('--timeout', type=float, default=600)
    args = p.parse_args()
    modes = args.modes.split()
    if any(m not in MODES for m in modes) or args.repetitions < 1: raise SystemExit('invalid matrix')
    if '/' in args.tag or args.tag in ('.', '..'): raise SystemExit('tag must be a directory basename')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise SystemExit('another CUDA compute process is active')
    result = ROOT/'results'/args.tag; result.mkdir()
    mount = result/'tmpfs'; mount.mkdir()
    write_json(result/'metadata.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'gpu': GPU, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'builds': json.loads((ROOT/'bin/timed/builds.json').read_text()),
        'builds_publish': json.loads((ROOT/'bin/timed/builds.json').read_text()),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helpers_sha256': hashlib.sha256((Path(__file__).parent/'run_hand_over.py').read_bytes()).hexdigest(),
        'clock': 'CLOCK_MONOTONIC'})
    mounted = False; records = []
    try:
        subprocess.run(['sudo', '-n', 'mount', '-t', 'tmpfs', '-o', f'huge=always,size=4096m,uid={os.getuid()},gid={os.getgid()}',
                        'tmpfs', str(mount)], check=True)
        mounted = True
        for run in range(1, args.repetitions+1):
            order = modes if run % 2 else list(reversed(modes))
            for mode in order:
                for fault in ((False, True) if run % 2 else (True, False)):
                    print('START', run, mode, 'fault' if fault else 'control', flush=True)
                    r = run_case(args, result, mount, run, mode, fault)
                    records.append({k: v for k, v in r.items() if k not in ('child_tokens', 'parent_tokens')})
                    write_json(result/'cases.json', records)
                    print('END', r['case'], r['status'], 'write', r.get('write'), 'children_exit',
                          [x.get('exit') for x in r.get('children_exit', [])], flush=True)
                    if r['status'] not in ('ok', 'failed'): raise RuntimeError('case validity failed')
        write_json(result/'completion.json', {'completed': True, 'cases': len(records)})
        print('RESULT_DIR', result, flush=True)
    finally:
        if mounted: subprocess.run(['sudo', '-n', 'umount', str(mount)], check=True)
        mount.rmdir()


if __name__ == '__main__':
    main()
