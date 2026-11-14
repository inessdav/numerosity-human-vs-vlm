#!/usr/bin/env python3
"""Classic cognitive baselines for the dot-estimation task (second model class).

Both baselines run on exactly the same fixed set of 16 images as the humans and
the VLMs (same seeds). They embody two competing mechanisms:

  weber : approximate number system (ANS). Estimate depends only on the true
          count n, with scalar variability (noise proportional to n):
              estimate = c * n^b * exp(N(0, w))
          It predicts NO effect of dot size or colour.

  area  : area heuristic. Numerosity is read off the total ink area, assuming a
          reference dot of radius REF_RADIUS_PX:
              estimate = total_dot_area / (pi * REF_RADIUS^2) * exp(N(0, w))
          It is exact for the same-size images and overestimates when dots vary
          in size, so it predicts a strong size effect (H2) and no colour effect.

w, b and c are free parameters; defaults are placeholders and should be fitted to
the human data (see --w/--b/--c) before the baselines are reported.

Usage:
  python baselines.py --runs 20 --out baselines.csv
"""
import argparse
import csv
import math
import random
from datetime import datetime, timezone

from stimuli import COLUMNS, DEFAULT_REPO, fixed_session, load_stimuli

REF_RADIUS_PX = 10.0       # radius of the dots in the same-size images


def estimate(kind, stim, rng, w, b, c):
    noise = math.exp(rng.gauss(0.0, w))
    if kind == "weber":
        value = c * stim["true_count"] ** b * noise
    else:
        value = stim["total_dot_area_px"] / (math.pi * REF_RADIUS_PX ** 2) * noise
    return max(1, round(value))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="repository root (default: parent of ai/)")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=1000, help="same value as in run_models.py")
    ap.add_argument("--w", type=float, default=0.20, help="Weber fraction (SD of log estimate)")
    ap.add_argument("--b", type=float, default=1.0, help="power-law exponent of the weber baseline")
    ap.add_argument("--c", type=float, default=1.0, help="scale of the weber baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stimuli = load_stimuli(args.repo)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for kind in ("weber", "area"):
            for run in range(args.runs):
                seed = args.seed_base + run
                rng = random.Random(f"{kind}-{seed}")        # independent noise stream per baseline
                for trial, stim in enumerate(fixed_session(stimuli, seed), start=1):
                    est = estimate(kind, stim, rng, args.w, args.b, args.c)
                    writer.writerow({
                        "subject_id": f"baseline-{kind}-run{run + 1:02d}", "age": "",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trial_number": trial, "stimulus_id": stim["stimulus_id"],
                        "true_count": stim["true_count"], "user_estimate": est,
                        "absolute_error": abs(est - stim["true_count"]),
                        "signed_error": est - stim["true_count"],
                        "reaction_time_ms": "", "stimulus_duration_ms": "", "render_mode": "simulated",
                        "agent_type": "baseline", "model_id": f"baseline_{kind}",
                        "provider": "local", "temperature": "", "run_index": run + 1, "seed": seed,
                        "size_mode": stim["size_mode"], "color_mode": stim["color_mode"],
                        "coverage_pct": stim["coverage_pct"],
                        "prompt": f"w={args.w} b={args.b} c={args.c}", "raw_response": str(est),
                        "parse_ok": True,
                    })
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
