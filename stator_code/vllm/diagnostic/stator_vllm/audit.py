import ctypes
import hashlib
import json
import os
from pathlib import Path
import time


def emit(kind, **values):
    root = os.environ.get('STATOR_BYTE_AUDIT_DIR')
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f'{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind, mono_ns=time.monotonic_ns(), pid=os.getpid(), **values))+'\n')


def memory_hash(base, size):
    h = hashlib.sha256()
    for offset in range(0,size,8*1024*1024):
        h.update((ctypes.c_ubyte*min(8*1024*1024,size-offset)).from_address(base+offset))
    return h.hexdigest()


def kv(kind, state, count, expected=None):
    if not os.environ.get('STATOR_BYTE_AUDIT_DIR'):
        return None
    size = state['block_bytes']
    base = state['base']+size
    hashes = [memory_hash(base+i*size,size) for i in range(count)]
    maps = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        region = line.split()[0]
        start,end = [int(x,16) for x in region.split('-')]
        if start<base+count*size and end>base:
            maps.append(line)
    emit(kind, blocks=count, block_bytes=size, content_sha256=hashes,
         maps=maps, expected_present=expected is not None,
         exact_match=hashes==expected if expected is not None else None)
    return hashes


def tensor_hash(tensor):
    import torch
    raw = tensor.detach().contiguous().view(-1).view(torch.uint8)
    h = hashlib.sha256()
    for start in range(0,raw.numel(),8*1024*1024):
        h.update(raw[start:start+8*1024*1024].cpu().numpy())
    return h.hexdigest()


def weight(entry, tensor, base):
    if not os.environ.get('STATOR_BYTE_AUDIT_DIR'):
        return
    original = tensor_hash(tensor)
    mapped = memory_hash(base+entry['offset'],entry['bytes'])
    emit('weight_before_replace', name=entry['name'], bytes=entry['bytes'],
         dtype=entry['dtype'], shape=entry['shape'], original_sha256=original,
         mapped_sha256=mapped, exact_match=original==mapped)
