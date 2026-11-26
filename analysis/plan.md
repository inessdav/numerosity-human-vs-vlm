# Plan

## Design
Dot images are shown for 500 ms. There are 16 images: colour (same / different) × dot size (same / different) × 4 numbers of dots (about 9, 30, 55 and 78). Every participant and every model sees all 16 images in random order and says how many dots there were.

Tested: humans (50), Gemini (20 runs), Gemini with reasoning (5 runs) and two simple models (Weber and area).

## Hypotheses
- H1: Different colours make people slower, but do not change how accurate they are.
- H2: Larger dots lead to higher estimates.

## Measures
Absolute error, signed error (over- or underestimation) and reaction time (humans only).

## Exclusions
- Reaction times above 200 s (participant likely left the task) are excluded from reaction time analysis only.
- Estimates above three times the true number of dots are typing errors and are excluded.

## Analysis
Each condition is compared with the other within participants, with effect sizes and 95% confidence intervals. Estimates are plotted against the true number of dots for humans and models.

## Scripts
- `stats.py`: bias per level, condition effects (colour, dot size) with 95% confidence intervals, mixed models.
- `figures.py`: summary figures.
- `mechanisms.py`: per-image answers, log-log slope, spread of answers, and how well a number-only and an ink-area observer fit each agent.
- `results/`: output tables (`levels.csv`, `effects.csv`) and numbers (`numbers.json`, `mechanisms.json`).