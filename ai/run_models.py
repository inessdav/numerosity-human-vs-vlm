#!/usr/bin/env python3
"""Run a vision-language model on the dot-numerosity estimation task.

One run = one virtual participant: the fixed set of 16 images in random order, one
independent API call per image (no conversation history, like a fresh trial).
The same --seed-base gives every model exactly the same presentation order.

Examples
--------
  # Anthropic API (needs ANTHROPIC_API_KEY)
  python run_models.py --provider anthropic \
      --model MODEL_ID --runs 20 --out ../data/results_a.csv

  # Any OpenAI-compatible endpoint (OpenAI, Gemini, vLLM/Qwen-VL, OpenRouter ...)
  python run_models.py --provider openai --model MODEL_ID \
      --api-key-env OPENAI_API_KEY --runs 20 --out ../data/results_b.csv
  python run_models.py --provider openai --model MODEL_ID \
      --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
      --api-key-env GEMINI_API_KEY --runs 20 --out ../data/results_c.csv

  # Offline test of the pipeline, no API calls
  python run_models.py --provider mock --model mock --runs 3 --out test.csv

A <out>.meta.json file with model ID, settings, prompt, library versions and
dates is written next to the CSV. Re-running the same command resumes and skips
trials that are already in the CSV.
"""
import argparse
import base64
import csv
import json
import math
import os
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stimuli import COLUMNS, DEFAULT_REPO, fixed_session, load_stimuli

PROMPT = "How many dots do you see? Answer with a single whole number only."


# ----------------------------------------------------------------------------
# Providers: each returns the raw text answer of the model for one image.
# ----------------------------------------------------------------------------
def make_caller(args):
    if args.provider == "mock":
        rng = random.Random(0)

        def call(stim, image_b64):
            est = max(1, round(stim["true_count"] * math.exp(rng.gauss(-0.1, 0.25))))
            return str(est)
        return call

    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ[args.api_key_env or "ANTHROPIC_API_KEY"],
                                     timeout=args.timeout, max_retries=0)

        def call(stim, image_b64):
            resp = client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": PROMPT},
                ]}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return call

    if args.provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ[args.api_key_env or "OPENAI_API_KEY"],
                        base_url=args.base_url, timeout=args.timeout, max_retries=0)

        def call(stim, image_b64):
            kwargs = {args.token_param: args.max_tokens}
            if args.reasoning_effort:                     # Gemini thinking level (low / medium / high)
                kwargs["reasoning_effort"] = args.reasoning_effort
            resp = client.chat.completions.create(
                model=args.model,
                temperature=args.temperature,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]}],
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        return call

    sys.exit(f"Unknown provider: {args.provider}")


def call_with_retry(call, stim, image_b64, tries=8):
    """Retry transient failures (overload 503, rate limit 429, timeouts) with capped backoff."""
    for attempt in range(tries):
        try:
            return call(stim, image_b64)
        except Exception as err:                      # network / overload / rate limit
            wait = min(2 ** attempt * 2, 60)
            print(f"  API error ({err.__class__.__name__}: {err}); retry {attempt + 1}/{tries} in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("API call failed repeatedly")


def parse_count(text):
    """First integer in the answer (commas as thousands separators removed)."""
    m = re.search(r"\d[\d,]*", text)
    return int(m.group().replace(",", "")) if m else None


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="repository root (default: parent of ai/)")
    ap.add_argument("--provider", choices=["anthropic", "openai", "mock"], required=True)
    ap.add_argument("--model", required=True, help="exact model ID, logged as is")
    ap.add_argument("--runs", type=int, default=20, help="virtual participants (default 20)")
    ap.add_argument("--seed-base", type=int, default=1000, help="run i uses seed seed_base+i")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--token-param", default="max_tokens",
                    help="OpenAI-compatible only; use max_completion_tokens for newer OpenAI models")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="OpenAI-compatible only; turns on model reasoning at this level "
                         "(raise --max-tokens, thinking tokens count towards it)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds to wait for one API response before retrying (default 60)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stimuli = load_stimuli(args.repo)
    call = make_caller(args)
    out = Path(args.out)

    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["subject_id"], r["stimulus_id"]) for r in csv.DictReader(f)}
    new_file = not out.exists()

    started = datetime.now(timezone.utc).isoformat()
    f = open(out, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    if new_file:
        writer.writeheader()

    safe = re.sub(r"[^A-Za-z0-9]+", "-", args.model)
    if args.reasoning_effort:
        safe += f"-{args.reasoning_effort}"
    for run in range(args.runs):
        seed = args.seed_base + run
        subject = f"{safe}-run{run + 1:02d}"
        session = fixed_session(stimuli, seed)
        for trial, stim in enumerate(session, start=1):
            if (subject, stim["stimulus_id"]) in done:
                continue
            image_b64 = base64.b64encode(stim["path"].read_bytes()).decode()
            raw = call_with_retry(call, stim, image_b64)
            est = parse_count(raw)
            row = {
                "subject_id": subject, "age": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trial_number": trial, "stimulus_id": stim["stimulus_id"],
                "true_count": stim["true_count"],
                "user_estimate": est if est is not None else "",
                "absolute_error": abs(est - stim["true_count"]) if est is not None else "",
                "signed_error": est - stim["true_count"] if est is not None else "",
                "reaction_time_ms": "", "stimulus_duration_ms": "", "render_mode": "image",
                "agent_type": "model", "model_id": args.model, "provider": args.provider,
                "temperature": args.temperature, "run_index": run + 1, "seed": seed,
                "size_mode": stim["size_mode"], "color_mode": stim["color_mode"],
                "coverage_pct": stim["coverage_pct"], "prompt": PROMPT,
                "raw_response": raw.strip().replace("\n", " "), "parse_ok": est is not None,
            }
            writer.writerow(row)
            f.flush()
            print(f"  run {run + 1}/{args.runs}  trial {trial}/{len(session)}  {stim['stimulus_id']}  "
                  f"true={stim['true_count']}  answer={raw.strip()[:40]!r}", flush=True)
        print(f"run {run + 1}/{args.runs} done", flush=True)
    f.close()

    meta = {
        "model_id": args.model, "provider": args.provider, "base_url": args.base_url,
        "temperature": args.temperature, "max_tokens": args.max_tokens, "timeout_s": args.timeout,
        "reasoning_effort": args.reasoning_effort,
        "prompt": PROMPT, "runs": args.runs, "seed_base": args.seed_base,
        "trials_per_run": 16, "design": "fixed balanced set of 16 images, random order per run",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
    }
    for lib in ("anthropic", "openai"):
        try:
            meta[f"{lib}_version"] = __import__(lib).__version__
        except Exception:
            pass
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out} and {out}.meta.json")


if __name__ == "__main__":
    main()
