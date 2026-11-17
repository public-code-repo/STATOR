import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

from live_publish import load_trace, phase


def estimate(xs, geometric=False):
    xs = list(xs)
    if geometric and any(x <= 0 for x in xs):
        raise ValueError('nonpositive paired ratio')
    values = [math.log(x) for x in xs] if geometric else xs
    avg = statistics.mean(values)
    critical = {2: 12.706205, 4: 3.182446, 6: 2.570582}.get(len(xs))
    half = critical * statistics.stdev(values) / math.sqrt(len(xs)) if critical else None
    transform = math.exp if geometric else lambda x:x
    return dict(n=len(xs), mean=transform(avg), ci95_low=transform(avg-half) if half is not None else None,
                ci95_high=transform(avg+half) if half is not None else None, values=xs,
                averaging='paired geometric' if geometric else 'arithmetic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    root = args.result
    validation = json.loads((root/'validation.json').read_text())
    cases = json.loads((root/'cases.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())['arguments']
    expected = metadata['repetitions']*len(metadata['modes'].split())*len(metadata['cells'].split())
    if metadata['trace_overhead'] or len(cases) != expected:
        raise ValueError('summary requires a complete run with tracing enabled in all cases')
    if any(r['status'] != 'ok' or not r['trace_on'] for r in cases):
        raise ValueError('case record incomplete; inspect the failure before aggregation')
    keys = [(r['run'], r['mode'], r['children'], r['paragraphs']) for r in cases]
    if len(set(keys)) != expected:
        raise ValueError('duplicate or missing case keys')
    reference = {(r['run'], r['children'], r['paragraphs']): r for r in cases if r['mode'] == 'copy'}
    enriched = []
    comparisons = []
    for row in cases:
        ref = reference[row['run'], row['children'], row['paragraphs']]
        count = row['children']+1
        text_matches = sum(x == y for x,y in zip(row['child_text_sha256'], ref['child_text_sha256']))
        text_matches += row['parent_text_sha256'] == ref['parent_text_sha256']
        token_matches = sum(x['tokens'] == y['tokens'] for x,y in zip(row['child_generations'], ref['child_generations']))
        token_matches += row['parent_generation']['tokens'] == ref['parent_generation']['tokens']
        comparisons.append(dict(case=row['case'], reference_case=ref['case'], includes_copy_self=row is ref,
                                processes=count, text_matches=text_matches, token_matches=token_matches))
        traces = load_trace(root/row['case']/'timing')
        child_phases = []
        for generation in row['child_generations']:
            events = traces[generation['pid']]
            child_phases.append({name+'_ms': (phase(events,name+'_end')-phase(events,name+'_start'))/1e6
                                 for name in ('backend','model','context','attach','input_decode')})
        data = {key: row[key] for key in ('case','run','mode','children','paragraphs','n_gen')}
        data.update(prefix_tokens=int(row['published'][0]['prefix_tokens']),
                    total_peak_gib=row['total_peak_accounted_bytes']/2**30,
                    child_increment_peak_gib=row['child_peak_drop_bytes']/2**30,
                    publish_ms=row['publish_ms_measured'],
                    parent_resume_to_token_ms=row['parent_resume_to_token_ms'],
                    parent_publish_plus_resume_ms=row['parent_publish_plus_resume_ms'],
                    child_ttft_ms=statistics.mean(g['ttft_ms'] for g in row['child_generations']),
                    child_ttft_max_ms=max(g['ttft_ms'] for g in row['child_generations']),
                    children_window_tps=row['children_window_tps'], all_window_tps=row['all_window_tps'],
                    gpu_faults_total=row['gpu_faults_total'],
                    gpu_faults_children_and_parent_resume=row['gpu_faults_children_and_parent_resume'],
                    minimum_available_gib=row['memory_min_available']/2**30,
                    child_tps_over_copy=row['children_window_tps']/ref['children_window_tps'],
                    publish_over_copy=row['publish_ms_measured']/ref['publish_ms_measured'],
                    child_phases_ms={key:statistics.mean(p[key] for p in child_phases) for key in child_phases[0]})
        enriched.append(data)
    groups = collections.defaultdict(list)
    for row in enriched:
        groups[row['paragraphs'],row['children'],row['mode']].append(row)
    summaries = []
    for (paragraphs, children, mode), group in sorted(groups.items()):
        if len(group) != metadata['repetitions']:
            raise ValueError('incomplete repetition group')
        metrics = {key:estimate((r[key] for r in group), geometric=key.endswith('_over_copy')) for key in group[0]
                   if key not in ('case','run','mode','children','paragraphs','n_gen','prefix_tokens','child_phases_ms')}
        phases = {key:estimate(r['child_phases_ms'][key] for r in group) for key in group[0]['child_phases_ms']}
        cmp = [c for c in comparisons if c['case'] in {r['case'] for r in group}]
        summaries.append(dict(paragraphs=paragraphs, prefix_tokens=group[0]['prefix_tokens'], children=children,
                              mode=mode, repetitions=len(group), metrics=metrics, phases_ms=phases,
                              processes_compared=sum(c['processes'] for c in cmp),
                              text_matches=sum(c['text_matches'] for c in cmp),
                              token_matches=sum(c['token_matches'] for c in cmp),
                              comparison_includes_copy_self=mode=='copy'))
    result = dict(case_count=len(cases), measurement_validation=validation, groups=summaries,
                  nonself_comparisons=[c for c in comparisons if not c['includes_copy_self']], cases=enriched,
                  definitions={'generation_window':'sum(actual tokens)/(max generation_end - min generation_start)',
                               'CI':'arithmetic means and Student t intervals for absolute values; paired geometric means and t intervals on log ratios',
                               'fault_scope':'total BH minor+major fault delta; post-publish counter includes both children and parent resume',
                               'memory':'MemAvailable drop plus warmed model file once; child increment uses post-publish baseline'},
                  gpu_faults_missing=any(r['gpu_faults_total'] is None for r in cases))
    (root/'analysis_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    rows = []
    for group in summaries:
        flat = {key:group[key] for key in ('prefix_tokens','children','mode','repetitions','processes_compared','text_matches','token_matches')}
        flat.update({key:stat['mean'] for key,stat in group['metrics'].items()})
        flat.update({'child_'+key:stat['mean'] for key,stat in group['phases_ms'].items()})
        rows.append(flat)
    with (root/'analysis_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(cases=len(cases),groups=len(summaries),validation_valid=validation['valid'],
                          nonself_processes=sum(c['processes'] for c in comparisons if not c['includes_copy_self']),
                          token_mismatches=sum(c['processes']-c['token_matches'] for c in comparisons)),indent=2))


if __name__ == '__main__':
    main()
