import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from trace_read import load_trace, phase, generation
import paths


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


