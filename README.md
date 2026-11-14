# Numerosity estimation: humans vs. vision-language models

Does a model that matches human numerosity judgements do so by the same mechanism?
This project compares how humans and AI models estimate the number of dots in a
briefly presented display, and tests whether dot colour and dot size influence the
estimate in the same way.

The test can be completed [here](https://inessdav.github.io/numerosity-human-vs-vlm/).

## Research question and hypotheses

Human estimates of large numerosities are generally attributed to the approximate
number system (ANS), whose noise grows in proportion to the quantity (Weber's law).
Whether vision-language models rely on a comparable mechanism is an open question.
Two stimulus manipulations are used to probe it:

- **H1 – colour.** Breaking colour homogeneity increases response time without
  substantially changing estimation accuracy.
- **H2 – dot size.** Larger elements cover more visual area, so the scene appears
  "fuller" and the estimate is higher than for the same number of small elements.

Measured outcomes: absolute error, signed error (over-/underestimation) and response
time (humans only). Models are compared on accuracy and signed error.

## Design

Within-subject, 2 (colour: uniform / varied) x 2 (dot size: uniform / varied)
x 4 (numerosity: about 9, 32, 56 and 78 dots). All 16 images are shown to every human
participant and to every model run, in a fresh random order each time.

| Condition | Images |
|---|---|
| uniform colour, uniform size | 01, 03, 05, 07 |
| varied colours, uniform size | 08, 10, 12, 14 |
| uniform colour, varied sizes | 22, 24, 26, 28 |
| varied colours, varied sizes | 15, 17, 19, 21 |

Numerosity levels are matched across conditions to within +-3 dots. Dot radius is
10 px in the uniform-size images and 5-25 px in the varied-size images; varied
colours are drawn from a palette of seven. The images were generated
programmatically; the exact geometry of every dot and per-image summary statistics
(total dot area, coverage, colour counts) are stored in `stimuli/`.

**Human procedure.** A fixation cross (500 ms), the image (500 ms, too short to
count), then a numeric response. Two practice trials with feedback precede the
16 recorded trials. The experiment interface is in Czech.

**Model procedure.** Each model receives each image in an independent request with
the prompt *"How many dots do you see? Answer with a single whole number only."*
One run corresponds to one virtual participant (16 images, random order). Model ID,
sampling settings, prompt, library versions and dates are stored in a `.meta.json`
file next to every result file.

**Cognitive baselines.** Two simulated observers run on the same stimuli: an ANS
observer with scalar variability (estimate independent of size and colour) and an
area-heuristic observer (estimate proportional to total dot area). They make
opposite predictions for H2.

## Repository structure

```
docs/        browser experiment (served with GitHub Pages) and the 16 stimulus images
stimuli/     stimulus specification: per-dot geometry and per-image summary statistics
ai/          model benchmark and baseline scripts
analysis/    analysis scripts and figures
data/        collected data (human, model, baseline)
```

## Running

Human experiment: open `docs/index.html` locally or via GitHub Pages. Results are posted
to a Google Apps Script web app; setup instructions are in the comment block inside
`docs/index.html`.

Model benchmark (`pip install -r requirements.txt`):

```bash
cd ai
export ANTHROPIC_API_KEY=...
python run_models.py --provider anthropic --model MODEL_ID --runs 20 --out ../data/results_a.csv
python baselines.py --runs 20 --out ../data/baselines.csv
```

OpenAI-compatible endpoints (OpenAI, Gemini, vLLM, OpenRouter) are supported through
`--provider openai --base-url ... --api-key-env ...`; see the header of `run_models.py`.

## Status

Data collection in progress.
