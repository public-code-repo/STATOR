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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def meminfo():
    return {line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}


def fields(line):
    return dict(x.split('=',1) for x in line.split()[1:] if '=' in x)


def tagged(path, tag):
    if not path.exists(): return []
    return [fields(line) for line in path.read_text(errors='replace').splitlines()
            if line.startswith(tag+' ')]


def text_hash(path):
    if not path.exists(): return None
    lines=[x for x in path.read_text(errors='replace').splitlines() if x.startswith('TEXT ')]
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest() if lines else None


def bh_snapshot():
    counters={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            line=(path/'stat').read_text(); end=line.rfind(')')
            name=line[line.find('(')+1:end]
            if name.startswith('UVM GPU') and name.endswith(' BH'):
                s=line[end+2:].split()
                counters[path.name]={'faults':int(s[7])+int(s[9]), 'cpu_ticks':int(s[11])+int(s[12]), 'name':name}
        except (OSError, ValueError): pass
    return counters


def bh_delta(before,after):
    if not before or before.keys()!=after.keys(): return None
    delta=sum(after[p]['faults']-before[p]['faults'] for p in before)
    return delta if delta>=0 else None


def allocated(path):
    s=os.statvfs(path)
    return (s.f_blocks-s.f_bfree)*s.f_frsize


def kill_group(proc):
    if proc is not None and proc.poll() is None:
        try: os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=30)


def load_trace(directory):
    traces={}
    for path in directory.glob('*.jsonl'):
        events=[json.loads(x) for x in path.read_text().splitlines()]
        if not events: raise ValueError(f'empty timing trace: {path}')
        pid=events[0]['pid']
        if pid in traces: raise ValueError(f'duplicate pid: {pid}')
        if any(b['mono_ns']<a['mono_ns'] for a,b in zip(events,events[1:])):
            raise ValueError(f'nonmonotonic trace: {path}')
        traces[pid]=events
    return traces


def phase(events,name):
    selected=[e['mono_ns'] for e in events if e['event']==name]
    if len(selected)!=1: raise ValueError(f'expected one {name}, got {len(selected)}')
    return selected[0]


def generation(events,expected):
    start,end=phase(events,'generation_start'),phase(events,'generation_end')
    tokens=[e for e in events if e['event']=='token']
    counts=[e['value'] for e in events if e['event']=='generation_end']
    if len(tokens)!=expected or counts!=[expected] or not tokens:
        raise ValueError(f'token counts differ: {len(tokens)}, {counts}, expected={expected}')
    if not start<=tokens[0]['mono_ns']<=tokens[-1]['mono_ns']<=end or end<=start:
        raise ValueError('invalid generation interval')
    return {'start_ns':start,'end_ns':end,'first_token_ns':tokens[0]['mono_ns'],
            'last_token_ns':tokens[-1]['mono_ns'],'tokens':[e['value'] for e in tokens],
            'ttft_ms':(tokens[0]['mono_ns']-phase(events,'process_start'))/1e6}


def generation_window_tps(generations):
    if not generations:raise ValueError('empty generation set')
    for g in generations:
        if g['end_ns']<=g['start_ns'] or not g['tokens']:
            raise ValueError('invalid generation or no tokens')
    return sum(len(g['tokens']) for g in generations)*1e9/(max(g['end_ns'] for g in generations)-min(g['start_ns'] for g in generations))


