import argparse
import collections
import json
from pathlib import Path


def validate(path):
    meta = json.loads((path/'metadata.json').read_text())['arguments']
    rows = json.loads((path/'cases.json').read_text())
    cells = meta['fanouts'].split() if meta['scenario']=='tree' else meta['counts'].split()
    expected = len(cells)*len(meta['modes'].split())*meta['repetitions']
    errors, skips, comparisons, missing_reference = [], [], [], []
    if len(rows)!=expected:
        errors.append(f'case count {len(rows)} != {expected}')
    keys = [(r['run'],r['mode'],str(r['cell'])) for r in rows]
    if len(set(keys))!=len(keys):
        errors.append('duplicate cases')
    groups = collections.defaultdict(list)
    for r in rows:
        if r['status']=='skip':
            skips.append(r['case'])
            continue
        if r['status']!='ok':
            errors.append(f"{r['case']}: {r['status']}")
            continue
        if r['memory_min_available']<meta['guard_gib']*2**30:
            errors.append(f"{r['case']}: memory floor")
        if r['tmpfs_after_unlink_bytes']!=0:
            errors.append(f"{r['case']}: unreleased tmpfs")
        if r['gpu_faults_total'] is None:
            errors.append(f"{r['case']}: missing GPU fault counter")
        if len(r['processes'])!=r['count']+(meta['scenario']=='resident'):
            errors.append(f"{r['case']}: process count")
        expected_gen = r['count'] if meta['scenario']=='tree' else meta['active']
        if len(r['generations'])!=expected_gen:
            errors.append(f"{r['case']}: generation count")
        for name,g in r['generations'].items():
            if len(g['tokens'])!=meta['tokens']:
                errors.append(f"{r['case']}/{name}: token count")
        groups[(r['run'],str(r['cell']))].append(r)
    for key, group in groups.items():
        ref = next((r for r in group if r['mode']=='copy'),None)
        if ref is None:
            missing_reference.append(str(key))
            continue
        for r in group:
            if r is ref:
                continue
            for name,g in ref['generations'].items():
                same_tokens = r['generations'].get(name,{}).get('tokens')==g['tokens']
                same_text = r['processes'].get(name,{}).get('text_sha256')==ref['processes'][name]['text_sha256']
                comparisons.append(dict(case=r['case'],process=name,token_match=same_tokens,text_match=same_text))
                if not same_tokens or not same_text:
                    errors.append(f"{r['case']}/{name}: Copy mismatch")
    result = dict(expected_cases=expected, cases=len(rows), skipped=skips,
                  missing_copy_reference=missing_reference, comparisons=comparisons,
                  errors=errors, valid=not errors,
                  all_requested_cases_executed=not skips and len(rows)==expected)
    (path/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('result',type=Path)
    a=p.parse_args()
    result=validate(a.result)
    print(json.dumps(result,indent=2))
    raise SystemExit(not result['valid'])
