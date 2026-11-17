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
ESTIMATE = {'copy':2700, 'demand':1900, 'device_tuned':1100,
            'device_merged':1100, 'writable_no_read':1800, 'extent':850,
            'copy_map':1100, 'extent_sealed':850, 'alone':2700}


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


def mode_environment(mode, mount):
    env={k:v for k,v in os.environ.items()
         if not k.startswith('STATOR_KV_') and k not in ('LD_LIBRARY_PATH','LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES=GPU,STATOR_HOST_PTR='1')
    if mode=='demand': env.update(STATOR_KV_VMM='2',STATOR_KV_GROW='256')
    elif mode in ('device_tuned','device_merged'):
        env.update(STATOR_KV_VMM='1',STATOR_KV_VMM_BATCH='1' if mode=='device_tuned' else '2',STATOR_KV_GROW='256')
    elif mode in ('extent','writable_no_read','copy_map','extent_sealed'):
        env.update(STATOR_KV_HOST=str(mount/'kv'),STATOR_KV_GROW='256')
        if mode=='writable_no_read': env['STATOR_KV_COW']='2'
        if mode=='copy_map': env['STATOR_KV_PUBLISH_COPY']='1'
        if mode=='extent_sealed': env['STATOR_KV_SEAL']='1'
    elif mode=='alone': env['STATOR_CHILD_ALONE']='1'
    elif mode!='copy': raise ValueError(mode)
    return env


def write_attempt(pid,mount,mode):
    import mmap as mm
    found={}
    if mode=='extent_sealed':
        for link in Path(f'/proc/{pid}/fd').iterdir():
            try: target=os.readlink(link)
            except OSError: continue
            if target.startswith('/memfd:llama-kv-'):
                found[target]=link
    else:
        found={str(p):p for p in mount.glob('kv.*')}
    out=[]
    for target,path in sorted(found.items()):
        r={'file':target}
        try: fd=os.open(path,os.O_RDWR)
        except OSError as e: r['open_rdwr']=f'refused {e.errno}'; out.append(r); continue
        r['open_rdwr']='allowed'
        if mode!='extent_sealed':
            os.close(fd); out.append(r); continue
        byte=os.pread(fd,1,0)
        try: os.pwrite(fd,byte,0); r['pwrite']='allowed'
        except OSError as e: r['pwrite']=f'refused {e.errno}'
        try:
            m=mm.mmap(fd,4096,mm.MAP_SHARED,mm.PROT_READ|mm.PROT_WRITE); m.close(); r['mmap_write']='allowed'
        except OSError as e: r['mmap_write']=f'refused {e.errno}'
        os.close(fd); out.append(r)
    return out


def run_case(args,result,mount,run,mode,count,shared,trace_on):
    name=f'r{run:02d}-{mode}-n{count}-s{shared}-trace{int(trace_on)}'
    case=result/name; case.mkdir(); (case/'timing').mkdir(); (case/'out').mkdir()
    record={'case':name,'run':run,'mode':mode,'children':count,'shared':shared,
            'n_gen':args.tokens,'trace_on':trace_on,'status':'preparing'}
    need=(count*ESTIMATE[mode]+8192)*1024**2
    if meminfo()['MemAvailable']-need < args.reserve_gib*1024**3:
        record.update(status='skip',reason='admission_estimate',estimated_bytes=need)
        write_json(case/'result.json',record); return record
    context=4096
    while context<args.total+512:context*=2
    prefix=case/'prefix.txt';prefix.write_text((args.ratio_dir/f'prefix_{shared}.txt').read_text())
    state=case/'state';go=case/'go'
    env=mode_environment(mode,mount)
    env.update(STATOR_TIMING_DIR=str(case/'timing'),STATOR_TRACE=str(int(trace_on)),
               STATOR_TASK_DIR=str(args.ratio_dir/f'tasks_{shared}'))
    parent=ROOT/'bin/timed'/f'ratio_publisher_{args.engine}'
    child=ROOT/'bin/timed'/f'prefix_agent_{args.engine}'
    cmd=[str(parent),str(args.model),str(context),str(prefix),str(state),str(args.tokens),str(go),
         str(case/'out'),str(child)]+[GPU]*count
    record.update(context=context,command=cmd,environment={k:v for k,v in env.items()
                  if k.startswith(('LLAMA_','GGML_','CUDA_','STATOR_'))})
    write_json(case/'result.json',record)
    subprocess.run(['sudo','-n','sh','-c','echo 1 > /proc/sys/vm/drop_caches'],check=True)
    with args.model.open('rb') as model:
        while model.read(8*1024*1024): pass
    time.sleep(.5)
    before=meminfo();lowest=before['MemAvailable'];lowest_children=None
    before_fault=bh_snapshot();publish_fault=None
    proc=None;published=False;aborted=None;release_ns=None
    before_ns=time.monotonic_ns()
    raw=case/'parent.out';err=case/'parent.err'
    tmpfs_peak=allocated(mount)
    try:
        with raw.open('w') as out,err.open('w') as stderr,(case/'samples.jsonl').open('w') as samples:
            proc=subprocess.Popen(cmd,env=env,stdout=out,stderr=stderr,start_new_session=True)
            record['parent_pid']=proc.pid
            while proc.poll() is None:
                now=time.monotonic_ns();m=meminfo();av=m['MemAvailable']
                lowest=min(lowest,av);tmpfs=allocated(mount);tmpfs_peak=max(tmpfs_peak,tmpfs)
                if published:lowest_children=min(lowest_children,av)
                samples.write(json.dumps({'mono_ns':now,'mem_available':av,'k_reclaimable':m['KReclaimable'],
                                          'tmpfs_allocated':tmpfs,'released':published})+'\n')
                if av<args.guard_gib*1024**3:
                    aborted='guard';kill_group(proc);break
                if (now-before_ns)/1e9>args.timeout:
                    aborted='timeout';kill_group(proc);break
                if not published and tagged(raw,'PUBLISHED'):
                    publish_fault=bh_snapshot()
                    if mode in ('extent','extent_sealed'):
                        record['write_attempt']=write_attempt(proc.pid,mount,mode)
                    record['before_children_available']=av
                    lowest_children=av
                    release_ns=time.monotonic_ns();go.touch();published=True
                time.sleep(.05)
            rc=proc.wait(timeout=30)
        after_fault=bh_snapshot()
        record.update(exit=rc,status=aborted or ('ok' if rc==0 else 'failed'),
                      guard=int(aborted=='guard'),timed_out=int(aborted=='timeout'),
                      memory_before=before,memory_min_available=lowest,memory_after=meminfo(),
                      peak_drop_bytes=before['MemAvailable']-lowest,
                      warmed_model_file_bytes=args.model.stat().st_size,
                      total_peak_accounted_bytes=before['MemAvailable']-lowest+args.model.stat().st_size,
                      child_peak_drop_bytes=(record['before_children_available']-lowest_children) if published else None,
                      tmpfs_peak_bytes=tmpfs_peak,tmpfs_end_before_unlink_bytes=allocated(mount),
                      fault_before=before_fault,fault_after_publish=publish_fault,fault_after=after_fault,
                      gpu_faults_total=bh_delta(before_fault,after_fault),
                      gpu_faults_children_and_parent_resume=bh_delta(publish_fault,after_fault) if publish_fault else None,
                      release_ns=release_ns,wall_s=(time.monotonic_ns()-before_ns)/1e9)
        published_rows=tagged(raw,'PUBLISHED');parent_rows=tagged(raw,'RESULT')
        exits=tagged(raw,'CHILD')
        record['published']=published_rows;record['parent_result']=parent_rows
        record['children_exit']=exits;record['parent_text_sha256']=text_hash(raw)
        record['child_results']=[tagged(case/'out'/f'child.{i}','RESULT') for i in range(count)]
        record['child_text_sha256']=[text_hash(case/'out'/f'child.{i}') for i in range(count)]
        if not parent_rows or len(exits)!=count or any(x.get('exit')!='0' for x in exits):
            raise ValueError('missing parent/child result or unsuccessful child')
        if any(len(x)!=1 or int(x[0].get('generated',-1))!=args.tokens for x in record['child_results']):
            raise ValueError('missing/incomplete child generation')
        if trace_on:
            traces=load_trace(case/'timing')
            if len(traces)!=count+1:raise ValueError(f'trace process count {len(traces)} != {count+1}')
            parent_trace=traces.pop(proc.pid)
            pg=generation(parent_trace,args.tokens)
            indices={e['value']:e['sequence'] for e in parent_trace if e['event']=='child_spawn'}
            if set(indices)!=set(traces):raise ValueError('spawned child pids do not match traces')
            children=[generation(events,args.tokens) | {'pid':pid,'index':indices[pid]} for pid,events in traces.items()]
            children.sort(key=lambda g:g['index'])
            record['parent_generation']=pg;record['child_generations']=children
            record['children_window_tps']=generation_window_tps(children)
            record['all_window_tps']=generation_window_tps(children+[pg])
            record['publish_ms_measured']=(phase(parent_trace,'publish_end')-phase(parent_trace,'publish_start'))/1e6
            record['harness_wait_ms']=(phase(parent_trace,'released')-phase(parent_trace,'publish_end'))/1e6
            record['parent_resume_to_token_ms']=(pg['first_token_ns']-phase(parent_trace,'released'))/1e6
            record['parent_publish_plus_resume_ms']=record['publish_ms_measured']+record['parent_resume_to_token_ms']
            record['all_children_first_token_after_release_ms']=(max(g['first_token_ns'] for g in children)-release_ns)/1e6
    except BaseException as error:
        kill_group(proc)
        record.update(status=aborted or 'invalid',error=f'{type(error).__name__}: {error}')
        write_json(case/'result.json',record)
        raise
    finally:
        for owned in [state,*mount.glob('kv*')]:
            if owned.is_file():owned.unlink()
        record['tmpfs_after_unlink_bytes']=allocated(mount)
        write_json(case/'result.json',record)
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    p.add_argument('--repetitions',type=int,default=6)
    p.add_argument('--cells',default='8:320')
    p.add_argument('--modes',default='copy demand device_tuned device_merged writable_no_read extent')
    p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--engine',choices=['variants','variants_gqa','publish'],default='variants')
    p.add_argument('--model',type=Path,default=ROOT/'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf')
    p.add_argument('--trace-overhead',action='store_true')
    p.add_argument('--reserve-gib',type=float,default=16)
    p.add_argument('--guard-gib',type=float,default=12)
    p.add_argument('--timeout',type=float,default=900)
    p.add_argument('--ratio-dir',type=Path,default=ROOT/'workloads/ratio')
    p.add_argument('--total',type=int,default=16384)
    args=p.parse_args()
    if args.repetitions<2 or args.repetitions%2 or args.tokens<1:raise SystemExit('positive even repetitions and positive tokens required')
    if '/' in args.tag or args.tag in ('.','..'):raise SystemExit('tag must be a directory basename')
    modes=args.modes.split();cells=[tuple(map(int,x.split(':'))) for x in args.cells.split()]
    if any(m not in ESTIMATE for m in modes) or any(n<1 or q<1 for n,q in cells):raise SystemExit('invalid matrix')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if gpu:raise SystemExit('another CUDA compute process is active')
    result=ROOT/'results'/args.tag;result.mkdir()
    mount=result/'tmpfs';mount.mkdir()
    write_json(result/'metadata.json',{'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'gpu':GPU,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'builds':json.loads((ROOT/'bin/timed/builds.json').read_text()),
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'spawn_ratio_sha256':hashlib.sha256((ROOT/'bin/timed'/f'ratio_publisher_{args.engine}').read_bytes()).hexdigest(),
                'ratio_files_sha256':{str(f.relative_to(args.ratio_dir)):hashlib.sha256(f.read_bytes()).hexdigest()
                                      for f in sorted(args.ratio_dir.rglob('*.txt'))},
                'clock':'CLOCK_MONOTONIC','memory_definition':'warmed-file baseline drop plus model file bytes once'})
    anchor=None;mounted=False;records=[]
    try:
        subprocess.run(['sudo','-n','mount','-t','tmpfs','-o',f'huge=always,size=4096m,uid={os.getuid()},gid={os.getgid()}','tmpfs',str(mount)],check=True)
        mounted=True
        with (result/'anchor.log').open('w') as log:
            anchor=subprocess.Popen([str(ROOT/'bin/linux_share_probe'),'hold','86400'],
                     env=os.environ|{'CUDA_VISIBLE_DEVICES':GPU},stdout=log,stderr=log,start_new_session=True)
        deadline=time.monotonic()+30
        while not bh_snapshot():
            if anchor.poll() is not None or time.monotonic()>deadline:raise RuntimeError('CUDA anchor did not create BH thread')
            time.sleep(.1)
        for repetition in range(1,args.repetitions+1):
            order=(modes if repetition%2 else list(reversed(modes))) if len(modes)==2 else modes[(repetition-1)%len(modes):]+modes[:(repetition-1)%len(modes)]
            for count,shared in cells:
                for mode in order:
                    traces=([False,True] if repetition%2 else [True,False]) if args.trace_overhead else [True]
                    for trace_on in traces:
                        print('START',repetition,mode,count,shared,'trace',trace_on,flush=True)
                        r=run_case(args,result,mount,repetition,mode,count,shared,trace_on)
                        records.append(r);write_json(result/'cases.json',records)
                        print('END',r['case'],r['status'],'window_tps',r.get('children_window_tps'),
                              'peak_GiB',round(r.get('total_peak_accounted_bytes',0)/1024**3,3),flush=True)
                        if r['status'] not in ('ok','skip'):raise RuntimeError('case validity failed')
        with (result/'summary.csv').open('w',newline='') as f:
            keys=['case','run','mode','children','shared','trace_on','status','publish_ms_measured',
                  'parent_resume_to_token_ms','children_window_tps','all_window_tps','total_peak_accounted_bytes',
                  'child_peak_drop_bytes','memory_min_available','gpu_faults_total','gpu_faults_children_and_parent_resume']
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
        write_json(result/'completion.json',{'completed':True,'cases':len(records),'skipped':sum(r['status']=='skip' for r in records)})
        print('RESULT_DIR',result,flush=True)
    finally:
        kill_group(anchor)
        if mounted:subprocess.run(['sudo','-n','umount',str(mount)],check=True)
        mount.rmdir()


if __name__=='__main__':main()
