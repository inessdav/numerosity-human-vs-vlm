"""Statistics for the dot-estimation experiment. Writes analysis/results/{levels,effects}.csv and numbers.json.

Outcome: signed log ratio  ln(estimate / true count)  (negative = underestimation),
reported as percentage bias  exp(x) - 1.  Unit of replication: participant (humans) or run (Gemini).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
D, OUT = ROOT / "data", ROOT / "analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)
BASELINES = D / "results_baselines_2000.csv"

LEVELS = [9, 30, 55, 78]
UNIFORM_COLOR = (1, 3, 5, 7, 22, 24, 26, 28)
UNIFORM_SIZE = (1, 3, 5, 7, 8, 10, 12, 14)


def load_baselines():
    """Simulated-observer runs (2000 per observer). Regenerate with ai/baselines.py if missing."""
    if not BASELINES.exists():
        raise SystemExit(f"{BASELINES.name} not found. Create it with:\n"
                         f"  python ai/baselines.py --runs 2000 --seed-base 5000 --out data/{BASELINES.name}")
    return pd.read_csv(BASELINES)


def load():
    h = pd.read_csv(D / "human_results.csv").assign(agent="human")
    g = pd.read_csv(D / "results_gemini.csv").assign(agent="gemini")
    r = pd.read_csv(D / "results_gemini_reasoning.csv").assign(agent="reason")
    b = load_baselines()
    b["agent"] = np.where(b.model_id == "baseline_weber", "weber", "area")
    cols = ["agent", "subject_id", "stimulus_id", "true_count", "user_estimate", "reaction_time_ms"]
    d = pd.concat([x[cols] for x in (h, g, r, b)], ignore_index=True)
    num = d.stimulus_id.str.extract(r"(\d+)")[0].astype(int)
    d["color"] = np.where(num.isin(UNIFORM_COLOR), "uniform", "varied")
    d["size"] = np.where(num.isin(UNIFORM_SIZE), "uniform", "varied")
    d["level"] = d.true_count.apply(lambda n: min(LEVELS, key=lambda L: abs(L - n)))
    d["logr"] = np.log(d.user_estimate / d.true_count)
    d["ok_acc"] = (d.user_estimate <= 3 * d.true_count) | (d.agent != "human")
    d["ok_rt"] = d.reaction_time_ms <= 200_000
    return d


def paired(df, outcome, factor):
    p = df.groupby(["subject_id", factor])[outcome].mean().unstack()
    diff = (p["varied"] - p["uniform"]).dropna()
    n, m, sd = len(diff), diff.mean(), diff.std(ddof=1)
    lo, hi = stats.t.interval(0.95, n - 1, m, sd / np.sqrt(n))
    return dict(n=n, effect=m, lo=lo, hi=hi, dz=m / sd, p=stats.ttest_1samp(diff, 0).pvalue)


def pct(x):
    return 100 * (np.exp(x) - 1)


def main():
    d = load()
    a = d[d.ok_acc]
    nums = {}

    def put(name, value):
        nums[name] = value

    # ---- sample sizes
    for ag in ("human", "gemini", "reason", "weber", "area"):
        put(f"n{ag}", int(d[d.agent == ag].subject_id.nunique()))
    put("ntrialsgemini", int((d.agent == "gemini").sum()))
    put("ntrialsreason", int((d.agent == "reason").sum()))
    put("nexcludedhuman", int((~d.ok_acc).sum()))
    put("nexcludedrt", int((~d.ok_rt & (d.agent == "human")).sum()))

    # ---- level table (bias in %, CV)
    rows = []
    for (ag, lv), g in a.groupby(["agent", "level"]):
        unit = g.groupby("subject_id").logr.mean()
        rows.append(dict(agent=ag, level=lv, true=g.true_count.mean(), est=g.user_estimate.mean(),
                         bias=pct(g.logr.mean()), bias_lo=pct(unit.mean() - 1.96 * unit.sem()), bias_hi=pct(unit.mean() + 1.96 * unit.sem()),
                         cv=g.user_estimate.std(ddof=1) / g.user_estimate.mean()))
    lv = pd.DataFrame(rows)
    lv.to_csv(OUT / "levels.csv", index=False)
    for _, r in lv.iterrows():
        for k in ("est", "bias", "cv"):
            put(f"{r.agent}{k}L{int(r.level)}", r[k])

    # ---- exponent (slope of ln estimate on ln true count), per participant/run
    ex = {}
    for ag in ("human", "gemini", "reason", "weber", "area"):
        s = np.array([np.polyfit(np.log(g.true_count), np.log(g.user_estimate), 1)[0]
                      for _, g in a[a.agent == ag].groupby("subject_id")])
        lo, hi = stats.t.interval(0.95, len(s) - 1, s.mean(), s.std(ddof=1) / np.sqrt(len(s)))
        ex[ag] = (s.mean(), lo, hi, s.std(ddof=1))
        for k, v in zip(("mean", "lo", "hi", "sd"), ex[ag]):
            put(f"{ag}exp{k}", v)

    # ---- condition effects
    eff = []
    for ag in ("human", "gemini", "reason", "weber", "area"):
        for factor in ("color", "size"):
            e = paired(a[a.agent == ag], "logr", factor)
            eff.append(dict(agent=ag, factor=factor, pct=pct(e["effect"]), pct_lo=pct(e["lo"]), pct_hi=pct(e["hi"]),
                            dz=e["dz"], p=e["p"], n=e["n"], logeffect=e["effect"]))
            for k in ("pct", "pct_lo", "pct_hi", "dz", "p"):
                put(f"{ag}{factor}{k}", eff[-1][k])
    pd.DataFrame(eff).to_csv(OUT / "effects.csv", index=False)

    # ---- human reaction time (colour effect), ln scale -> percent
    h = d[(d.agent == "human") & d.ok_rt].copy()
    h["lrt"] = np.log(h.reaction_time_ms)
    for factor in ("color", "size"):
        e = paired(h, "lrt", factor)
        put(f"humanrt{factor}pct", pct(e["effect"]))
        put(f"humanrt{factor}lo", pct(e["lo"]))
        put(f"humanrt{factor}hi", pct(e["hi"]))
        put(f"humanrt{factor}dz", e["dz"])
        put(f"humanrt{factor}p", e["p"])
    put("humanrtmedian", h.reaction_time_ms.median() / 1000)

    # ---- human colour effect on absolute relative error + equivalence test (margin +-5 percentage points)
    ah = a[a.agent == "human"].copy()
    ah["absrel"] = (ah.user_estimate - ah.true_count).abs() / ah.true_count
    p = ah.groupby(["subject_id", "color"]).absrel.mean().unstack()
    diff = (p["varied"] - p["uniform"]).dropna()
    m, se, n = diff.mean(), diff.sem(), len(diff)
    margin = 0.05
    p_tost = max(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))
    lo90, hi90 = stats.t.interval(0.90, n - 1, m, se)
    put("humanabsrelcolor", 100 * m)
    put("humanabsrelcolorcilo", 100 * lo90)
    put("humanabsrelcolorcihi", 100 * hi90)
    put("humantostp", p_tost)

    # ---- robustness: mixed model with crossed random intercepts for participant/run and for image
    for ag in ("human", "gemini", "reason"):
        m_ = a[a.agent == ag].copy()
        m_["one"] = 1
        m_["lognc"] = np.log(m_.true_count) - np.log(m_.true_count).mean()
        fit = smf.mixedlm("logr ~ C(color) + C(size) + lognc", m_, groups="one", re_formula="0",
                          vc_formula={"subj": "0 + C(subject_id)", "stim": "0 + C(stimulus_id)"}).fit(reml=True, method="lbfgs")
        ci = fit.conf_int()
        for factor in ("color", "size"):
            term = f"C({factor})[T.varied]"
            put(f"{ag}mm{factor}pct", pct(fit.params[term]))
            put(f"{ag}mm{factor}pctlo", pct(ci.loc[term, 0]))
            put(f"{ag}mm{factor}pcthi", pct(ci.loc[term, 1]))
            put(f"{ag}mm{factor}p", fit.pvalues[term])
            print("mixed", ag, factor, round(pct(fit.params[term]), 1), round(pct(ci.loc[term, 0]), 1), round(pct(ci.loc[term, 1]), 1), round(fit.pvalues[term], 3))

    # ---- Gemini consistency: how often does a run give the most common answer for an image?
    for ag in ("human", "gemini", "reason"):
        g = a[a.agent == ag]
        put(f"{ag}cvmedian", float((g.groupby("stimulus_id").user_estimate.std() / g.groupby("stimulus_id").user_estimate.mean()).median()))
        modal = g.groupby("stimulus_id").user_estimate.agg(lambda s: s.value_counts().iloc[0] / len(s))
        put(f"{ag}modalshare", 100 * modal.mean())
        put(f"{ag}imagesd", g.groupby("stimulus_id").user_estimate.std().mean())
        put(f"{ag}runsd", g.groupby("subject_id").logr.mean().std())
        # overall accuracy: mean ratio estimate/true and mean absolute relative error
        put(f"{ag}ratiopct", 100 * (g.user_estimate / g.true_count).mean())
        put(f"{ag}absrelpct", 100 * ((g.user_estimate - g.true_count).abs() / g.true_count).mean())

    # ---- fitted baseline parameters (from run files)
    b = load_baselines()
    nums["weberparams"] = b[b.model_id == "baseline_weber"].prompt.iloc[0]
    nums["areaparams"] = b[b.model_id != "baseline_weber"].prompt.iloc[0]

    json.dump({k: v for k, v in nums.items()}, open(OUT / "numbers.json", "w"), indent=1, default=float)
    print(lv.round(2).to_string(index=False))
    print(pd.DataFrame(eff).round(3).to_string(index=False))
    print({ag: tuple(round(x, 3) for x in ex[ag]) for ag in ex})
    for k in ("humanrtcolorpct", "humanrtcolorlo", "humanrtcolorhi", "humanrtcolorp", "humanrtsizepct", "humanrtmedian",
              "humanabsrelcolor", "humanabsrelcolorcilo", "humanabsrelcolorcihi", "humantostp",
              "geminimodalshare", "geminiimagesd", "geminirunsd", "nexcludedhuman", "nexcludedrt"):
        print(k, round(nums[k], 3))
    print(nums["weberparams"], "|", nums["areaparams"])


if __name__ == "__main__":
    main()
