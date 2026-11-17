import argparse
import collections
import json
from pathlib import Path


def percentile(values,q):
    if not values:return None
    x=sorted(values);p=(len(x)-1)*q;i=int(p);j=min(i+1,len(x)-1)
    return x[i]+(x[j]-x[i])*(p-i)


def analyze(path):
    meta=json.loads((path/'metadata.json').read_text())
    trace=json.loads((path/'trace.json').read_text())
    raw=[json.loads(x) for x in (path/'raw.jsonl').read_text().splitlines()]
    errors=[]
    starts=[r for r in raw if r['event']=='arrival_clock']
    ends=[r for r in raw if r['event']=='end']
    if len(starts)!=1 or len(ends)!=1:raise ValueError('missing common clock/end')
    start=starts[0]['mono_ns'];end=ends[0]
    releases={r['path']:r for r in raw if r['event']=='release'}
    launches={r['pid']:r for r in raw if r['event']=='launch'}
    procs=[r for r in raw if r['event']=='proc']
    samples=[r for r in raw if r['event']=='sample']
    session_groups=collections.defaultdict(list)
    all_tokens=[];request_rows=[];turn_rows=[];ids={};finished=[]
    for proc in procs:
        sid=proc['sid'];role=proc['role'];index=proc['index']
        session_groups[sid].append(proc)
        key=f'{sid}.{role}{index}'
        events=[]
        for name in proc.get('timing_paths',[]):
            events.extend(json.loads(x) for x in Path(name).read_text().splitlines())
        if any(b['mono_ns']<a['mono_ns'] for a,b in zip(events,events[1:])):
            errors.append(f'{key}: nonmonotonic events')
        if proc['exit']==0 and not events:errors.append(f'{key}: missing trace')
        tokens=[e for e in events if e['event']=='token']
        all_tokens.extend(tokens)
        groups=collections.defaultdict(list)
        for event in events:
            if event['event'] in ('generation_start','generation_end','token'):
                groups[event['generation']].append(event)
        expected=2 if role=='c' or proc['kind']=='fork' else 3 if proc['kind']=='tool' else 1
        if proc['exit']==0 and len(groups)!=expected:
            errors.append(f'{key}: {len(groups)} generations != {expected}')
        ids[key]=[]
        waits=[s[5:] for s in launches.get(proc.get('pid'),{}).get('argv',[]) if s.startswith('wait:')]
        for number,ev in sorted(groups.items()):
            ts=[e for e in ev if e['event']=='token']
            bs=[e['mono_ns'] for e in ev if e['event']=='generation_start']
            es=[e['mono_ns'] for e in ev if e['event']=='generation_end']
            if proc['exit']==0 and (len(ts)!=meta['n_gen'] or len(bs)!=1 or len(es)!=1):
                errors.append(f'{key}/gen{number}: incomplete timing/token count')
            if bs and es:
                if not ts or not bs[0]<=ts[0]['mono_ns']<=ts[-1]['mono_ns']<=es[0]:
                    errors.append(f'{key}/gen{number}: invalid generation interval')
                finished.append(es[0])
            ids[key].append([e['value'] for e in ts])
            if not ts:continue
            first=ts[0]['mono_ns']
            if number==1:
                process_start=next((e['mono_ns'] for e in events if e['event']=='process_start'),None)
                row=dict(sid=sid,role=role,index=index,first_token_ns=first,
                    process_to_first_s=(first-process_start)/1e9 if process_start else None,
                    arrival_to_first_s=(first-start)/1e9-proc['arrival'] if role=='a' else None,
                    admission_s=(proc['launch_ns']-start)/1e9-proc['arrival'] if role=='a' and proc['launch_ns'] else None)
                phases={}
                for name in ['backend','model','context','attach','input_decode']:
                    b=[e['mono_ns'] for e in events if e['event']==name+'_start' and e['generation']==0]
                    e=[e['mono_ns'] for e in events if e['event']==name+'_end' and e['generation']==0]
                    if b and len(b)==len(e) and all(z>=y for y,z in zip(b,e)):
                        phases[name+'_s']=sum(z-y for y,z in zip(b,e))/1e9
                    else:
                        phases[name+'_s']=None
                        errors.append(f'{key}: missing/invalid first-token phase {name}')
                row['phases']=phases
                request_rows.append(row)
            elif number-2<len(waits):
                release=releases.get(waits[number-2])
                if release is None:errors.append(f'{key}/gen{number}: missing release')
                else:turn_rows.append(dict(sid=sid,role=role,index=index,generation=number,
                    kind='merge' if role=='a' and proc['kind']=='fork' else 'tool',
                    release_to_first_s=(first-release['before_ns'])/1e9,
                    release_write_ns=release['after_ns']-release['before_ns']))
    completed=[];unfinished=[]
    for s in trace['sessions']:
        ps=session_groups[s['sid']]
        expected=1+trace['fork_k'] if s['kind']=='fork' else 1
        ok=len(ps)==expected and all(p['exit']==0 for p in ps)
        (completed if ok else unfinished).append(s['sid'])
    expected_tokens=trace['total_requested_tokens']
    if not unfinished and len(all_tokens)!=expected_tokens:
        errors.append(f'actual tokens {len(all_tokens)} != requested {expected_tokens}')
    faults=[r['total'] for r in raw if r['event']=='fault_counter']
    if len(faults)!=1 or faults[0] is None:errors.append('missing fault counter')
    ttft=[r['arrival_to_first_s'] for r in request_rows if r['role']=='a']
    turn=[r['release_to_first_s'] for r in turn_rows if r['kind']=='tool']
    window_end=start+int(meta['window_s']*1e9)
    window_tokens=sum(start<=t['mono_ns']<window_end for t in all_tokens)
    peak=max((r['total_state_accounted_mib'] for r in samples),default=None)
    budget_met=peak is not None and peak<=meta['memory_budget_gib']*1024
    result=dict(system=meta['system'],sessions=len(trace['sessions']),completed=len(completed),
        unfinished=unfinished,failed_processes=[f"{p['sid']}.{p['role']}{p['index']}" for p in procs if p['exit'] not in (0,'unfinished','queued')],
        guard_events=end['guard_events'],ended=end['ended'],actual_recorded_tokens=len(all_tokens),
        requested_tokens=expected_tokens,window_completed_tokens=window_tokens,
        window_token_rate=window_tokens/meta['window_s'],
        last_generation_end_s=(max(finished)-start)/1e9 if finished else None,
        drain_wall_s=end['t'],ttft_count=len(ttft),
        ttft_p50_s=percentile(ttft,.5),ttft_p95_s=percentile(ttft,.95),ttft_p99_s=percentile(ttft,.99),
        tool_resume_count=len(turn),tool_resume_p50_s=percentile(turn,.5),tool_resume_p95_s=percentile(turn,.95),
        deadlines={str(limit):dict(served_within=sum(t<=limit for t in ttft),all_sessions=len(trace['sessions'])) for limit in [1,2,5]},
        memory_peak_accounted_gib=peak/1024 if peak is not None else None,memory_budget_met=budget_met,
        queue_peak=max((s['queue'] for s in samples),default=None),gpu_faults=faults[0] if faults else None,
        tmpfs_before_bytes=end['tmpfs_before_bytes'],tmpfs_after_bytes=end['tmpfs_after_bytes'],
        extent_reclaimed=end['tmpfs_after_bytes']<=end['tmpfs_before_bytes'],
        requests=request_rows,turns=turn_rows,token_ids=ids,errors=errors,
        measurement_valid=not errors,
        caveat='P99 descriptive; unfinished requests remain in denominators and are listed separately')
    return result


