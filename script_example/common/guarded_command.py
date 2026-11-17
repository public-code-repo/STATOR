import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable missing')


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--guard-gib',type=float,default=12);p.add_argument('--limit-s',type=float,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise SystemExit('command required')
    if available()<60*1024**3:raise SystemExit('insufficient memory before starting')
    a.report.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic_ns();low=available();reason=None
    child=subprocess.Popen(command,start_new_session=True)
    try:
        while child.poll() is None:
            low=min(low,available())
            if low<a.guard_gib*1024**3:reason='memory_guard'
            if (time.monotonic_ns()-started)/1e9>a.limit_s:reason='timeout'
            if reason:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30)
                break
            time.sleep(.1)
        rc=child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30)
        a.report.write_text(json.dumps({'command':command,'pid':child.pid,'exit':child.returncode,
                     'reason':reason,'minimum_available_bytes':low,
                     'elapsed_s':(time.monotonic_ns()-started)/1e9},indent=2)+'\n')
    raise SystemExit(1 if reason else rc)


if __name__=='__main__':main()
