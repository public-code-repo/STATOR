import argparse
import collections
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args()
    meta=json.loads((a.result/'metadata.json').read_text())['arguments']
    cases=json.loads((a.result/'cases.json').read_text())
    expected=meta['repetitions']*len(meta['modes'].split())*len(meta['cells'].split())*(2 if meta['trace_overhead'] else 1)
    errors=[]
    if len(cases)!=expected:errors.append(f'case count {len(cases)} != {expected}')
    groups=collections.defaultdict(list)
    for c in cases:
        if c['status']!='ok':errors.append(f"{c['case']}: {c['status']}")
        if c.get('guard') or c.get('timed_out'):errors.append(f"{c['case']}: guard/timeout")
        if c.get('memory_min_available',0)<12*1024**3:errors.append(f"{c['case']}: memory floor")
        if c.get('gpu_faults_total') is None:errors.append(f"{c['case']}: missing fault counter")
        if c.get('tmpfs_after_unlink_bytes')!=0:errors.append(f"{c['case']}: tmpfs allocation not released")
        groups[(c['run'],c['children'],c['paragraphs'])].append(c)
    compared=0
    for key,rows in groups.items():
        refs=[x for x in rows if x['mode']=='copy']
        if not refs:errors.append(f'{key}: no Copy reference');continue
        ref=refs[0]
        for row in rows:
            if row['child_text_sha256']!=ref['child_text_sha256'] or row['parent_text_sha256']!=ref['parent_text_sha256']:
                errors.append(f"{row['case']}: text mismatch against Copy")
            else:compared+=row['children']+1
        traced=[x for x in rows if x['trace_on']]
        tref=next((x for x in traced if x['mode']=='copy'),None)
        if tref:
            reference=[x['tokens'] for x in tref['child_generations']]+[tref['parent_generation']['tokens']]
            for row in traced:
                observed=[x['tokens'] for x in row['child_generations']]+[row['parent_generation']['tokens']]
                if observed!=reference:errors.append(f"{row['case']}: token ids mismatch")
    overhead=[]
    if meta['trace_overhead']:
        pairs=collections.defaultdict(dict)
        for c in cases:pairs[(c['run'],c['mode'],c['children'],c['paragraphs'])][c['trace_on']]=c
        for key,pair in pairs.items():
            if set(pair)!={False,True}:errors.append(f'{key}: incomplete instrumentation control');continue
            on=statistics.mean(float(x[0]['generation_ms']) for x in pair[True]['child_results'])
            off=statistics.mean(float(x[0]['generation_ms']) for x in pair[False]['child_results'])
            overhead.append({'run':key[0],'mode':key[1],'children':key[2],'paragraphs':key[3],'on_off_generation_time':on/off})
    result={'case_count':len(cases),'expected_cases':expected,'text_comparisons_including_reference':compared,
            'errors':errors,'valid':not errors,'instrumentation_controls':overhead}
    (a.result/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
