import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request


def wait(port, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as answer:
                if answer.status == 200:
                    return 0
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return 1


def agent(index, port, model, prompt, n_gen, lines, started):
    body = {"model": model, "prompt": prompt, "max_tokens": n_gen, "temperature": 0,
            "seed": 1, "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        text, first, usage = "", None, {}
        with urllib.request.urlopen(request, timeout=3600) as answer:
            for raw in answer:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                piece = json.loads(payload)
                if "error" in piece:
                    raise RuntimeError(str(piece["error"]))
                for choice in piece.get("choices", []):
                    if first is None and choice.get("text"):
                        first = time.time()
                    text += choice.get("text", "")
                if piece.get("usage"):
                    usage = piece["usage"]
        done = time.time()
        generated = int(usage.get("completion_tokens", 0))
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", -1)
        generation = done - (first or done)
        lines[index] = (
            f"AGENT index={index} port={port} exit=0 "
            f"text={hashlib.sha256(text.encode()).hexdigest()[:16]} "
            f"prompt_tokens={usage.get('prompt_tokens', 0)} prompt_cached={cached} "
            f"first_token_ms={((first or done) - started) * 1000:.1f} "
            f"generated={generated} "
            f"generation_tps={((generated - 1) / generation) if generation > 0 and generated > 1 else 0:.3f} "
            f"request_ms={(done - started) * 1000:.1f}")
    except Exception as error:
        lines[index] = (f"AGENT index={index} port={port} exit=1 "
                        f"error={type(error).__name__}:{str(error)[:80].replace(' ', '_')}")


