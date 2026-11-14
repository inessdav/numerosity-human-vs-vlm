"""Stimulus loading and the fixed presentation design of the dot-estimation experiment.

Humans, vision-language models and the cognitive baselines all see the same 16 images
(4 conditions x 4 count levels); only the presentation order is randomised per run.
"""
import csv
import random
from pathlib import Path

# Repository root (this file lives in <root>/ai/)
DEFAULT_REPO = Path(__file__).resolve().parent.parent

# (first image, last image) covered by each summary CSV in stimuli/
SUMMARY_RANGES = [(1, 7), (8, 14), (15, 21), (22, 28)]
TRIALS_PER_SESSION = 15


def load_stimuli(repo):
    """Return a list of 28 stimulus dicts built from the repo's summary CSVs."""
    repo = Path(repo)
    stimuli = []
    for first, last in SUMMARY_RANGES:
        path = repo / "stimuli" / f"dot-images_summary{first}-{last}.csv"
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                local = int(row["file"].split(".")[0])      # 01..07 inside the file
                g = first + local - 1                        # global image number 1..28
                stimuli.append({
                    "stimulus_id": f"img_{g:02d}",
                    "path": repo / "docs" / "images" / f"{g:02d}.png",
                    "true_count": int(row["true_count"]),
                    "size_mode": row["size_mode"],           # same | varied
                    "color_mode": row["color_mode"],         # single | multi
                    "total_dot_area_px": float(row["total_dot_area_px"]),
                    "coverage_pct": float(row["coverage_pct"]),
                })
    stimuli.sort(key=lambda s: s["stimulus_id"])
    assert len(stimuli) == 28, f"expected 28 stimuli, found {len(stimuli)}"
    return stimuli


# Balanced fixed design: 4 conditions x 4 count levels (about 9, 32, 56 and 78 dots).
# Every participant, model run and baseline run sees exactly these 16 images.
FIXED_IDS = [
    "img_01", "img_03", "img_05", "img_07",     # same colour, same size
    "img_08", "img_10", "img_12", "img_14",     # different colours, same size
    "img_22", "img_24", "img_26", "img_28",     # same colour, different sizes
    "img_15", "img_17", "img_19", "img_21",     # different colours, different sizes
]


def fixed_session(stimuli, seed):
    """One virtual participant on the fixed design: all 16 images in random order."""
    chosen = [s for s in stimuli if s["stimulus_id"] in FIXED_IDS]
    assert len(chosen) == len(FIXED_IDS), "fixed stimulus set is incomplete"
    random.Random(seed).shuffle(chosen)
    return chosen


# Columns 1-12 are identical to the human Google Sheet so the files can be concatenated.
COLUMNS = [
    "subject_id", "age", "timestamp", "trial_number", "stimulus_id", "true_count",
    "user_estimate", "absolute_error", "signed_error", "reaction_time_ms",
    "stimulus_duration_ms", "render_mode",
    # extra columns for the model / baseline runs
    "agent_type", "model_id", "provider", "temperature", "run_index", "seed",
    "size_mode", "color_mode", "coverage_pct", "prompt", "raw_response", "parse_ok",
]
