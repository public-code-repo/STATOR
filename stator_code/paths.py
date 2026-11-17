import os

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
ROOT = os.environ.get("STATOR_ROOT") or REPO
BIN = os.path.join(ROOT, "bin")
TIMED_BIN = os.path.join(BIN, "timed")
ENGINES = os.path.join(ROOT, "engines")
MODELS = os.path.join(ROOT, "models")
WORKLOADS = os.path.join(ROOT, "workloads")
RESULTS = os.path.join(ROOT, "results")
