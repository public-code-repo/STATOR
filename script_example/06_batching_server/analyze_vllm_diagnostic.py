import argparse
import json
from pathlib import Path


def log_probs(request):
    output=[]
    for chunk in request.get('chunks',[]):
        for choice in chunk['response'].get('choices',[]):
            lp=choice.get('logprobs') or {}
            output.extend(dict(token=t,log_probability=p,top=top)
                          for t,p,top in zip(lp.get('tokens',[]),lp.get('token_logprobs',[]),lp.get('top_logprobs',[])))
    return output


def compare(left,right,label):
    cases=[]
    for task in sorted(left.keys() & right.keys()):
        a,b=left[task],right[task]
        same_prompt=a.get('prompt_sha256')==b.get('prompt_sha256')
        at,bt=a.get('token_ids',[]),b.get('token_ids',[])
        first=next((i for i,(u,v) in enumerate(zip(at,bt)) if u!=v),None)
        if first is None and len(at)!=len(bt):
            first=min(len(at),len(bt))
        row=dict(task_id=task,same_prompt=same_prompt,
                 same_prompt_ids=a.get('prompt_token_ids')==b.get('prompt_token_ids'),
                 token_match=bool(at) and at==bt,text_match=a.get('text')==b.get('text'),
                 first_different_zero_based=first)
        if first is not None:
            la,lb=log_probs(a),log_probs(b)
            row['left_logits']=la[first] if len(la)>first else None
            row['right_logits']=lb[first] if len(lb)>first else None
        cases.append(row)
    return dict(label=label,compared=len(cases),token_matches=sum(x['token_match'] for x in cases),
                missing_left=sorted(right.keys()-left.keys()),missing_right=sorted(left.keys()-right.keys()),
                differences=[x for x in cases if not x['token_match'] or not x['same_prompt'] or not x['same_prompt_ids']])


def main():
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args()
    meta=dict(line.split('=',1) for line in (a.result/'metadata.txt').read_text().splitlines() if '=' in line)
    repetitions=int(meta['repetitions']);modes=meta['modes'].split();servers=int(meta['servers']);agents=int(meta['agents'])
    expected_rounds={'prefix':1,'first':agents//2,'both':agents*servers//2,'both_self':agents*servers//2,
                     'exact9':1,'exact11':1,'recompute':agents*servers//2,'alone1':1,'alone2':1}
    for slot in range(3,servers+1):expected_rounds[f'warm{slot}']=1
    errors=[];data={};coverage=[]
    for run in range(1,repetitions+1):
        for mode in modes:
            for round,count in expected_rounds.items():
                paths=sorted((a.result/'requests'/f'{run}-{mode}'/round).glob('*.json'))
                rows=[json.loads(f.read_text()) for f in paths]
                if len(rows)!=count:errors.append(f'{run}-{mode}/{round}: {len(rows)} != {count}')
                mapping={r['task_id']:r for r in rows};data[run,mode,round]=mapping
                if len(mapping)!=len(rows):errors.append(f'{run}-{mode}/{round}: duplicate task IDs')
                for r in rows:
                    if r.get('exit')!=0 or len(r.get('token_ids',[]))!=r.get('usage',{}).get('completion_tokens'):
                        errors.append(f'{run}-{mode}/{round}/{r["task_id"]}: failed or token count missing')
                coverage.append(dict(run=run,mode=mode,round=round,requests=len(rows),
                    cached_tokens=[(r.get('usage',{}).get('prompt_tokens_details') or {}).get('cached_tokens') for r in rows]))
    comparisons=[]
    for run in range(1,repetitions+1):
        for mode in modes:
            base=data[run,mode,'both']
            for other in ['both_self','recompute']:
                comparisons.append(compare(base,data[run,mode,other],f'r{run}/{mode}: both vs {other}'))
            for task,round in [(17,'exact9'),(19,'exact11')]:
                comparisons.append(compare({task:base[task]} if task in base else {},data[run,mode,round],f'r{run}/{mode}: both vs {round}'))
            if mode!='vllm':
                for round in expected_rounds:
                    comparisons.append(compare(data[run,'vllm',round],data[run,mode,round],f'r{run}/{round}: vllm vs {mode}'))
    for mode in modes:
        for round in ['first','both','both_self','recompute']:
            for run in range(2,repetitions+1):
                comparisons.append(compare(data[1,mode,round],data[run,mode,round],f'{mode}/{round}: r1 vs r{run}'))
    result=dict(coverage=coverage,comparisons=comparisons,errors=errors,
                execution_records_valid=not errors,
                exact_output_valid=not errors and all(x['compared']==x['token_matches'] for x in comparisons),
                caveat='logprobs-enabled diagnostic, not a performance run; missing cached_tokens is NA, not proven zero')
    (a.result/'diagnostic_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('coverage','comparisons')},indent=2))
    for x in comparisons:
        if x['token_matches']!=x['compared']:
            print(x['label'],x['token_matches'],'/',x['compared'],[(d['task_id'],d['first_different_zero_based']) for d in x['differences']])
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
