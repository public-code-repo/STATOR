import json

def load_trace(directory):
    traces={}
    for path in directory.glob('*.jsonl'):
        events=[json.loads(x) for x in path.read_text().splitlines()]
        if not events: raise ValueError(f'empty timing trace: {path}')
        pid=events[0]['pid']
        if pid in traces: raise ValueError(f'duplicate pid: {pid}')
        if any(e['pid'] != pid for e in events): raise ValueError(f'mixed PIDs in trace: {path}')
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
