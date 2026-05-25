"""Inference latency / throughput on the RTX 3090.

Reports tokens/second, time-to-first-token (TTFT), and end-to-end completion time. The
hypothesis: a denser vocabulary => fewer tokens per file => fewer decode steps => faster
generation, even though the embedding matrix is slightly larger. Supports a native
``transformers`` path and an optional Ollama (GGUF) path.

Requires a GPU; not runnable on CPU here.

Usage:
    python -m src.eval_latency configs/experiments/semid_qlora.yaml \
        model_path=data/models/semid_qlora variant=full
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.utils import cfg_get, get_logger, load_config, write_json

log = get_logger("eval_latency")

_PROMPTS = [
    "def parse_config(path):\n",
    "class Cache:\n    def __init__(self):\n",
    "async def fetch(url):\n",
    "# Implement a binary search\n",
]


def _bench_transformers(cfg: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = cfg_get(cfg, "model_path") or cfg_get(cfg, "model.id")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    max_new = cfg_get(cfg, "eval.completion_max_new_tokens", 256)
    warmup = cfg_get(cfg, "eval.latency_warmup", 5)
    iters = cfg_get(cfg, "eval.latency_iters", 50)

    def run_once(prompt: str):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            first = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        ttft = time.perf_counter() - t0
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        e2e = time.perf_counter() - t1
        n_new = out.shape[1] - inputs.input_ids.shape[1]
        return ttft, e2e, n_new

    for i in range(warmup):
        run_once(_PROMPTS[i % len(_PROMPTS)])

    ttfts, e2es, tps = [], [], []
    for i in range(iters):
        ttft, e2e, n_new = run_once(_PROMPTS[i % len(_PROMPTS)])
        ttfts.append(ttft)
        e2es.append(e2e)
        tps.append(n_new / e2e if e2e > 0 else 0.0)

    return {
        "backend": "transformers",
        "tokens_per_sec": round(sum(tps) / len(tps), 2),
        "ttft_ms": round(1000 * sum(ttfts) / len(ttfts), 2),
        "e2e_ms": round(1000 * sum(e2es) / len(e2es), 2),
    }


def _bench_ollama(cfg: dict) -> dict | None:
    """Optional: benchmark a model already imported into Ollama (see scripts/export_ollama.sh)."""
    model_name = cfg_get(cfg, "ollama_model")
    if not model_name:
        return None
    try:
        import requests

        iters = cfg_get(cfg, "eval.latency_iters", 50)
        tps = []
        for i in range(iters):
            t0 = time.perf_counter()
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": _PROMPTS[i % len(_PROMPTS)], "stream": False},
                timeout=120,
            ).json()
            dt = time.perf_counter() - t0
            n = r.get("eval_count", 0)
            tps.append(n / dt if dt else 0.0)
        return {"backend": "ollama", "model": model_name, "tokens_per_sec": round(sum(tps) / len(tps), 2)}
    except Exception as exc:
        log.warning("Ollama benchmark skipped (%s)", exc)
        return None


def evaluate(cfg: dict) -> dict:
    result = {
        "config_name": cfg.get("name"),
        "variant": cfg_get(cfg, "variant", cfg.get("name", "unknown")),
        "transformers": _bench_transformers(cfg),
    }
    ollama = _bench_ollama(cfg)
    if ollama:
        result["ollama"] = ollama
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    summary = evaluate(cfg)
    variant = summary["variant"]
    out = Path(cfg_get(cfg, "paths.results_dir", "results")) / f"latency_{variant}.json"
    write_json(summary, out)
    log.info("Latency eval (%s) -> %s", variant, out)


if __name__ == "__main__":
    main()
