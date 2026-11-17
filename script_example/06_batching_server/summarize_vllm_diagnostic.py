import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    root = args.result
    metadata = dict(line.split('=', 1) for line in (root / 'metadata.txt').read_text().splitlines() if '=' in line)
    assert int(metadata['repetitions']) == 2 and int(metadata['servers']) == 4
    assert metadata['modes'].split() == ['vllm', 'stator_weights', 'stator']
    records = [json.loads(line) for path in sorted((root / 'byte_audit').glob('*.jsonl')) for line in path.read_text().splitlines()]
    groups = collections.defaultdict(list)
    for record in records:
        groups[record['kind']].append(record)
    errors = []
    expected = dict(weight_before_replace=4528, kv_publish=2, kv_publish_protection=2, kv_attach=6)
    for kind, count in expected.items():
        if len(groups[kind]) != count:
            errors.append(f'{kind}: {len(groups[kind])} != {count}')
    weights = groups['weight_before_replace']
    by_pid = collections.defaultdict(list)
    for row in weights:
        by_pid[row['pid']].append(row)
        if not row['exact_match'] or row['original_sha256'] != row['mapped_sha256']:
            errors.append(f'weight mismatch {row["pid"]}/{row["name"]}')
    if len(by_pid) != 16 or any(len(xs) != 283 or len({x['name'] for x in xs}) != 283 for xs in by_pid.values()):
        errors.append('weight process or unique tensor coverage is incomplete')
    pub = groups['kv_publish']
    for row in groups['kv_publish_protection']:
        if row['result'] != 0 or row['blocks'] != 892 or row['block_ids'] != list(range(1, 893)):
            errors.append(f'protection or block layout {row["pid"]}')
    for row in pub + groups['kv_attach']:
        if row['blocks'] != 892 or row['block_bytes'] != 917504 or len(row['content_sha256']) != 892:
            errors.append(f'KV hash coverage {row["pid"]}')
        covered = 0
        for mapping in row['maps']:
            parts = mapping.split()
            start, end = [int(s, 16) for s in parts[0].split('-')]
            covered += end - start
            if parts[1] != 'r--s' or not parts[-1].endswith('/vllm_byte_mount/kv'):
                errors.append(f'KV mapping permissions/path {row["pid"]}')
        if covered != row['blocks'] * row['block_bytes']:
            errors.append(f'KV read-only mapping coverage {row["pid"]}')
        if row['kind'] == 'kv_attach':
            if not row['expected_present'] or not row['exact_match']:
                errors.append(f'KV attachment mismatch {row["pid"]}')
            if not any(row['content_sha256'] == p['content_sha256'] for p in pub):
                errors.append(f'KV attachment has no matching publisher {row["pid"]}')

    cases = []
    current = None
    run = None
    for line in (root / 'raw.log').read_text().splitlines():
        if line.startswith('RUN '):
            match = re.search(r'(?:index|run|rep)=(\d+)', line)
            run = int(match.group(1)) if match else None
        if line.startswith('CASE '):
            current = dict(re.findall(r'(\w+)=([^ ]+)', line)); current['run'] = run
            current['startup_ms'] = []
        elif line.startswith('SERVER ') and current is not None:
            fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
            current['startup_ms'].append(int(fields['startup_ms']))
        elif line.startswith('MEMORY at=end ') and current is not None:
            current.update(dict(re.findall(r'(\w+)=([^ ]+)', line)))
        elif line.startswith('END_CASE ') and current is not None:
            current['failed'] = int(dict(re.findall(r'(\w+)=([^ ]+)', line))['failed'])
            cases.append(current); current = None
    if len(cases) != 6 or any(c['failed'] or len(c['startup_ms']) != 4 for c in cases):
        errors.append('incomplete or failed server cases')

    diagnostic = json.loads((root / 'diagnostic_analysis.json').read_text())
    requests = [json.loads(path.read_text()) for path in sorted((root / 'requests').glob('*/*/*.json'))]
    if len(requests) != 354 or any(r['exit'] != 0 for r in requests):
        errors.append('request count or completion')
    modes = []
    for mode in metadata['modes'].split():
        mode_cases = [c for c in cases if c['mode'] == mode]
        stats = []
        salt_counts = []
        missing_cached = 0
        explicit_cached = []
        for run in (1, 2):
            rows = [json.loads(path.read_text()) for path in (root / 'requests' / f'{run}-{mode}' / 'both').glob('*.json')]
            stats.append(dict(request_ttft_s=statistics.mean((r['first_token_ns']-r['send_ns'])/1e9 for r in rows),
                              request_window_tps=sum(len(r['token_ids']) for r in rows)*1e9/(max(r['done_ns'] for r in rows)-min(r['send_ns'] for r in rows))))
            recompute = [json.loads(path.read_text()) for path in (root / 'requests' / f'{run}-{mode}' / 'recompute').glob('*.json')]
            salts = [r['request'].get('cache_salt') for r in recompute]
            salt_counts.append(len(set(salts)))
            if len(recompute) != 16 or len(set(salts)) != 16 or None in salts:
                errors.append(f'recompute request salt coverage {run}/{mode}')
            for row in recompute:
                cached = (row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                if cached is None:
                    missing_cached += 1
                else:
                    explicit_cached.append(cached)
        comparisons = {}
        if mode != 'vllm':
            for round_name in ('prefix', 'first', 'both', 'both_self', 'exact9', 'exact11', 'recompute', 'alone1', 'alone2'):
                rows = [c for c in diagnostic['comparisons'] if c['label'] in (f'r1/{round_name}: vllm vs {mode}', f'r2/{round_name}: vllm vs {mode}')]
                comparisons[round_name] = dict(matches=sum(c['token_matches'] for c in rows), compared=sum(c['compared'] for c in rows))
        repeats = [c for c in diagnostic['comparisons'] if c['label'] in (f'r1/{mode}: both vs both_self', f'r2/{mode}: both vs both_self')]
        modes.append(dict(mode=mode, whole_case_peak_gib=statistics.mean(int(c['peak_drop_mib'])/1024 for c in mode_cases),
                          server_startup_mean_s=statistics.mean(t/1000 for c in mode_cases for t in c['startup_ms']),
                          both_request_ttft_s=statistics.mean(s['request_ttft_s'] for s in stats),
                          both_request_window_tps=statistics.mean(s['request_window_tps'] for s in stats),
                          gpu_faults=None, cross_mode_token_matches=comparisons,
                          both_self_matches=dict(matches=sum(c['token_matches'] for c in repeats),compared=sum(c['compared'] for c in repeats)),
                          recompute_unique_salts_per_case=salt_counts,
                          recompute_missing_cached_token_field=missing_cached,
                          recompute_explicit_cached_tokens=explicit_cached))
    report = dict(byte_checks_valid=not errors, errors=errors, request_count=len(requests),
                  weight_checks=len(weights), weight_matches=sum(r['exact_match'] for r in weights),
                  kv_import_block_checks=sum(r['blocks'] for r in groups['kv_attach']),
                  kv_importers=len(groups['kv_attach']), kv_publishers=len(pub),
                  readonly_kv_mappings=len(pub)+len(groups['kv_attach']),
                  execution_records_valid=diagnostic['execution_records_valid'],
                  exact_output_valid=diagnostic['exact_output_valid'], modes=modes,
                  limitations=['Hashes were recorded at replacement/attach, not every decoding step.',
                               'Request-window throughput includes TTFT and is not generation-window throughput.',
                               'Diagnostic hashing/logprobs overhead prevents using these timings as formal performance evidence.',
                               'Missing cached_tokens means NA, not measured zero; request-specific salt prevents intended cache-key reuse.'])
    (root / 'byte_diagnostic_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    sources = [root/'metadata.txt', root/'raw.log', *sorted((root/'byte_audit').glob('*.jsonl'))]
    (root / 'byte_diagnostic_source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
