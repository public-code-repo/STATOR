from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import paths

ROOT = Path(paths.ROOT)
AF = ROOT/'bin/access_flag_probe'
MECH = ROOT/'bin/mechanism_probe_rw'
PARAM = Path('/sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--expected-prefetch', choices=['0', '1'], required=True)
    args = parser.parse_args()
    assert PARAM.read_text().strip() == args.expected_prefetch
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert '\n' not in gpu and gpu.startswith('GPU-')
    binaries = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [AF, MECH]}
    metadata = {'prefetch': args.expected_prefetch, 'mib': 1024, 'repetitions': 2,
                'gpu': gpu, 'binary_sha256': binaries, 'boot_id':
                Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [Path(__file__), AF.with_suffix('.cu'), MECH.with_suffix('.cu')]}}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
    buffer = args.output / 'private-test-buffer.bin'
    records = []

    def run(name, command, rep, expected_lines):
        mem = Path('/proc/meminfo').read_text()
        assert int(re.search(r'MemAvailable:\s+(\d+)', mem)[1]) >= 16 * 1024 * 1024
        assert PARAM.read_text().strip() == args.expected_prefetch
        started = time.monotonic()
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        (args.output / f'{rep}-{name}.out').write_text(result.stdout)
        (args.output / f'{rep}-{name}.err').write_text(result.stderr)
        fields = [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in result.stdout.splitlines()
                  if line.startswith(('probe=access_flag ', 'STEP '))]
        record = {'name': name, 'rep': rep, 'command': list(map(str, command)),
                  'returncode': result.returncode, 'elapsed_s': time.monotonic() - started,
                  'fields': fields}
        records.append(record)
        (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        assert result.returncode == 0 and len(fields) == expected_lines, name
        assert all(f.get('sums_ok', f.get('right')) == '1' for f in fields), name

    patterns = [('writable', x) for x in ['none', 'one_page', 'one_per_8_blocks', 'one_per_block', 'all']]
    patterns += [('readonly', 'one_per_block'), ('readonly', 'all')]
    try:
        for rep in (1, 2):
            for offset in range(len(patterns)):
                protection, pattern = patterns[(rep + offset) % len(patterns)]
                run(f'af-{protection}-{pattern}', [AF, '1024', pattern, protection], rep, 1)
        with buffer.open('xb') as handle:
            subprocess.run(['head', '-c', str(1024 * 1024 * 1024), '/dev/urandom'], stdout=handle, check=True)
        metadata['private_file_fs'] = subprocess.check_output(['stat', '-f', '-c', '%T', str(buffer)], text=True).strip()
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for rep in (1, 2):
            for access in (['cpu', 'gpu'] if rep == 1 else ['gpu', 'cpu']):
                run(f'private-{access}', [MECH, 'private', buffer, access], rep, 2)
        assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [AF, MECH]} == binaries
        (args.output / 'complete.txt').write_text('18 invocations passed; read checksums correct\n')
    finally:
        buffer.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
