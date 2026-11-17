import argparse
import json
from pathlib import Path


def ids(tokens):
    return [t['id'] for t in tokens]


def diverge(reference, tokens):
    return next((k for k, (a, b) in enumerate(zip(reference, tokens)) if a != b),
                None if len(reference) == len(tokens) else min(len(reference), len(tokens)))


def rate(tokens, start_ns):
    after = [t['mono_ns'] for t in tokens if t['mono_ns'] >= start_ns]
    return (len(after)-1)*1e9/(after[-1]-after[0]) if len(after) > 1 else None


def main():
    p = argparse.ArgumentParser(); p.add_argument('result', type=Path); a = p.parse_args()
    cases = {}
    for path in sorted(a.result.glob('r*/result.json')):
        r = json.loads(path.read_text()); cases[(r['run'], r['mode'], r['fault'])] = r
    out = {'cases': [], 'modes': {}}
    for mode in sorted({m for _, m, _ in cases}):
        children, controls_equal, parents = [], [], []
        for run in sorted({r for r, m, _ in cases if m == mode}):
            control, fault = cases.get((run, mode, False)), cases.get((run, mode, True))
            if control is None or fault is None: continue
            write = fault['write'][0]
            start = int(write['start_ns'])
            first = cases[(min(r for r, m, _ in cases if m == mode), mode, False)]
            controls_equal.append(control['child_text_sha256'] == first['child_text_sha256'])
            row = {'run': run, 'mode': mode, 'write': write, 'writer_exit': fault['writer_exit'],
                   'children_exit': [x.get('exit') for x in fault['children_exit']],
                   'children_text_equal': sum(x == y for x, y in zip(control['child_text_sha256'], fault['child_text_sha256'])),
                   'parent_text_equal': control['parent_text_sha256'] == fault['parent_text_sha256'],
                   'parent_diverge_token': diverge(ids(control['parent_tokens']), ids(fault['parent_tokens'])),
                   'children': []}
            parents.append(row['parent_text_equal'])
            for index in sorted(fault['child_tokens'], key=int):
                tokens = fault['child_tokens'][index]; reference = ids(control['child_tokens'][index])
                k0 = next((k for k, t in enumerate(tokens) if t['mono_ns'] >= start), len(tokens))
                d = diverge(reference, ids(tokens))
                child = {'index': int(index), 'tokens': len(tokens), 'first_after_write': k0,
                         'diverge_token': d, 'diverge_after_write': None if d is None else d-k0,
                         'rate_after_write': rate(tokens, start),
                         'control_rate_same_window': rate(control['child_tokens'][index],
                             control['child_tokens'][index][min(k0, len(control['child_tokens'][index])-1)]['mono_ns'])}
                child['on_reference'] = [d is None or d > k0+k for k in range(len(tokens)-k0)]
                row['children'].append(child); children.append(child)
            out['cases'].append(row)
        span = min(len(c['on_reference']) for c in children)
        curve = [100.0*sum(c['on_reference'][k] for c in children)/len(children) for k in range(span)]
        after = [c['diverge_after_write'] for c in children if c['diverge_after_write'] is not None]
        out['modes'][mode] = {'children': len(children), 'tokens_after_write': span, 'curve_percent': curve,
                              'children_diverged': len(after), 'diverge_after_write_min': min(after, default=None),
                              'diverge_after_write_max': max(after, default=None),
                              'tokens_before_write': [c['first_after_write'] for c in children],
                              'controls_equal_across_runs': all(controls_equal), 'parent_text_equal': parents,
                              'rate_after_write': [c['rate_after_write'] for c in children],
                              'control_rate': [c['control_rate_same_window'] for c in children]}
    (a.result/'analysis.json').write_text(json.dumps(out, indent=2)+'\n')
    for mode, m in out['modes'].items():
        print(mode, {k: v for k, v in m.items() if k not in ('curve_percent', 'rate_after_write', 'control_rate')},
              'curve', [round(x) for x in m['curve_percent'][:12]], '...', round(m['curve_percent'][-1]))
    for row in out['cases']:
        print(row['run'], row['mode'], row['write']['result'], row['write']['written'], '/', row['write']['targets'],
              'exits', row['children_exit'], 'texts equal', row['children_text_equal'], 'parent equal', row['parent_text_equal'])


if __name__ == '__main__':
    main()
