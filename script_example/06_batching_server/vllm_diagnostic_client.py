import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request


def wait(port, seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200:return 0
        except OSError:pass
        time.sleep(.25)
    return 1


def request(index, task_id, port, model, prompt, n_gen, lines, started, barrier):
    root=Path(os.environ['STATOR_DIAG_DIR']);root.mkdir(parents=True,exist_ok=True)
    body={'model':model,'prompt':prompt,'max_tokens':n_gen,'temperature':0,'seed':1,
          'stream':True,'stream_options':{'include_usage':True},'return_token_ids':True,
          'return_tokens_as_token_ids':True,'logprobs':2}
    salt=os.environ.get('STATOR_DIAG_SALT','')
    if salt:body['cache_salt']=f'{salt}-task-{task_id}-index-{index}'
    record={'index':index,'task_id':task_id,'port':port,'request':body,'common_start_ns':started,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'chunks':[]}
    path=root/f'agent-{index:03d}-task-{task_id}.json'
    try:
        barrier.wait(timeout=30)
        record['send_ns']=time.monotonic_ns()
        req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(body).encode(),
                                   headers={'Content-Type':'application/json'},method='POST')
        text='';tokens=[];prompt_tokens=None;usage={};first=None
        with urllib.request.urlopen(req,timeout=900) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data:'):continue
                payload=line[5:].strip()
                if payload=='[DONE]':break
                now=time.monotonic_ns();piece=json.loads(payload)
                record['chunks'].append({'received_ns':now,'response':piece})
                if 'error' in piece:raise RuntimeError(str(piece['error']))
                if piece.get('prompt_token_ids') is not None:prompt_tokens=piece['prompt_token_ids']
                for choice in piece.get('choices',[]):
                    ids=choice.get('token_ids') or []
                    if first is None and (ids or choice.get('text')):first=now
                    tokens.extend(ids);text+=choice.get('text','')
                    if choice.get('prompt_token_ids') is not None:prompt_tokens=choice['prompt_token_ids']
                if piece.get('usage'):usage=piece['usage']
        done=time.monotonic_ns();generated=int(usage.get('completion_tokens',0))
        if generated<1 or generated!=len(tokens):raise ValueError(f'returned {len(tokens)} token IDs for {generated} tokens')
        record.update(exit=0,done_ns=done,first_token_ns=first,text=text,token_ids=tokens,
                      prompt_token_ids=prompt_tokens,usage=usage)
        first_ms=(first-started)/1e6
        lines[index]=(f'AGENT index={index} task_id={task_id} port={port} exit=0 '
                      f'text={hashlib.sha256(text.encode()).hexdigest()[:16]} '
                      f'prompt_tokens={usage.get("prompt_tokens",0)} '
                      f'prompt_cached={(usage.get("prompt_tokens_details") or {}).get("cached_tokens",-1)} '
                      f'first_token_ms={first_ms:.3f} generated={generated} '
                      f'generation_tps={(generated-1)*1e9/(done-first) if done>first else 0:.3f} '
                      f'request_ms={(done-started)/1e6:.3f}')
    except Exception as error:
        record.update(exit=1,error=f'{type(error).__name__}: {error}')
        lines[index]=f'AGENT index={index} task_id={task_id} port={port} exit=1 error={type(error).__name__}'
    finally:
        path.write_text(json.dumps(record,ensure_ascii=False)+'\n')


def main():
    a=sys.argv[1:]
    if len(a)>=2 and a[0]=='wait':return wait(int(a[1]),float(a[2]) if len(a)>2 else 60)
    if len(a)==4 and a[0]=='prefix':
        model,prefix,port=a[1:];lines=[None];started=time.monotonic_ns()
        request(0,-1,int(port),model,Path(prefix).read_text(),1,lines,started,threading.Barrier(1))
        print(lines[0].replace('AGENT index=0','PREFIX',1));return int('exit=1' in lines[0])
    if len(a) in (7,8) and a[0]=='agents':
        model,prefix,tasks=a[1:4];n_gen,count=int(a[4]),int(a[5]);ports=list(map(int,a[6].split(',')))
        first_task=int(a[7]) if len(a)==8 else 0
        shared=Path(prefix).read_text();task_lines=Path(tasks).read_text().splitlines()
        lines=[None]*count;threads=[];barrier=threading.Barrier(count);started=time.monotonic_ns()
        for i in range(count):
            tid=first_task+i
            prompt=shared+f'\n\nTask for agent {tid}: {task_lines[tid%len(task_lines)]}\n'
            t=threading.Thread(target=request,args=(i,tid,ports[i%len(ports)],model,prompt,n_gen,lines,started,barrier))
            t.start();threads.append(t)
        for t in threads:t.join()
        print('\n'.join(lines));return int(any('exit=1' in x for x in lines))
    raise SystemExit('usage: wait PORT [SECONDS] | prefix MODEL PREFIX PORT | agents MODEL PREFIX TASKS N_GEN COUNT PORTS [FIRST_TASK]')


if __name__=='__main__':sys.exit(main())
