"""Why do the answers look the way they do? Per-image answers, simulated-observer fits and mechanism figures.

Run after stats.py. Writes analysis/results/mechanisms.json, figures/answers.png and figures/mechanisms.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures as F  # noqa: E402

ROOT = F.ROOT
rng = np.random.default_rng(7)
STIM = pd.concat([pd.read_csv(ROOT / "stimuli" / f"dot-images_summary{a}-{b}.csv", encoding="utf-8-sig")
                  .assign(first=a) for a, b in [(1, 7), (8, 14), (15, 21), (22, 28)]], ignore_index=True)
STIM["num"] = STIM.apply(lambda r: r["first"] + int(r["file"].split(".")[0]) - 1, axis=1)
STIM["stimulus_id"] = STIM.num.apply(lambda g: f"img_{g:02d}")
AREA = STIM.set_index("stimulus_id").total_dot_area_px

h = pd.read_csv(ROOT / "data" / "human_results.csv")
h = h[h.user_estimate <= 3 * h.true_count].assign(agent="human")
g = pd.read_csv(ROOT / "data" / "results_gemini.csv").assign(agent="gemini")
rz = pd.read_csv(ROOT / "data" / "results_gemini_reasoning.csv").assign(agent="reason")
cols = ["agent", "subject_id", "stimulus_id", "true_count", "user_estimate"]
D = pd.concat([h[cols], g[cols], rz[cols]], ignore_index=True)
D["logn"] = np.log(D.true_count)
D["logest"] = np.log(D.user_estimate)
D["logA"] = np.log(D.stimulus_id.map(AREA))
IDS = sorted(D.stimulus_id.unique())
meta = D.drop_duplicates("stimulus_id").set_index("stimulus_id")[["true_count", "logn", "logA"]].loc[IDS]


def per_image(d):
    """Mean log estimate, SD of the log estimate and CV of the raw estimate per image."""
    grp = d.groupby("stimulus_id")
    out = pd.DataFrame({"mlog": grp.logest.mean(), "sdlog": grp.logest.std(ddof=1),
                        "mean": grp.user_estimate.mean(), "sd": grp.user_estimate.std(ddof=1)})
    out["cv"] = out["sd"] / out["mean"]
    return out.loc[IDS]


def fits(pi):
    """Slope and Weber-type fits of per-image means."""
    x, y = meta.logn.values, pi.mlog.values
    b, a = np.polyfit(x, y, 1)
    X3 = np.c_[np.ones(16), x, meta.logA.values]
    coef3 = np.linalg.lstsq(X3, y, rcond=None)[0]
    res_w = y - (a + b * x)
    # area heuristic: estimate proportional to ink area (one free offset)
    off = np.mean(y - meta.logA.values)
    res_a = y - (off + meta.logA.values)
    # slope of log SD on log mean (Weber's law: about 1)
    wl = np.polyfit(np.log(pi["mean"].values), np.log(pi["sd"].values), 1)[0]
    return dict(b=b, a=a, g=coef3[2], b3=coef3[1], rmse_weber=np.sqrt(np.mean(res_w ** 2)),
                rmse_area=np.sqrt(np.mean(res_a ** 2)), weber_slope=wl)


units = {ag: [u for _, u in D[D.agent == ag].groupby("subject_id")] for ag in ("human", "gemini", "reason")}
R, BOOT, PI = {}, {}, {}
for ag in units:
    d = D[D.agent == ag]
    PI[ag] = per_image(d)
    R[ag] = fits(PI[ag])
    reps = []
    ids = list(d.subject_id.unique())
    for _ in range(2000):
        pick = rng.choice(ids, len(ids))
        dd = pd.concat([units[ag][ids.index(s)] for s in pick])
        reps.append(fits(per_image(dd)))
    BOOT[ag] = pd.DataFrame(reps)
    for k in ("b", "g", "rmse_weber", "rmse_area", "weber_slope"):
        R[ag][k + "_lo"], R[ag][k + "_hi"] = np.percentile(BOOT[ag][k], [2.5, 97.5])
    R[ag]["cv_median"] = float(PI[ag].cv.median())
    R[ag]["cv_min"], R[ag]["cv_max"] = float(PI[ag].cv.min()), float(PI[ag].cv.max())
    ests = d.user_estimate
    R[ag]["heap5"] = float(((ests >= 10) & (ests % 5 == 0)).sum() / (ests >= 10).sum())
    R[ag]["heap10"] = float(((ests >= 10) & (ests % 10 == 0)).sum() / (ests >= 10).sum())
for ag_ in ("gemini", "reason"):
    gd = D[D.agent == ag_]
    R[ag_]["modal_share"] = float(gd.groupby("stimulus_id").user_estimate.agg(lambda s: s.value_counts().iloc[0] / len(s)).mean())
    R[ag_]["ratio_mean"] = float(np.exp(PI[ag_].mlog.values - meta.logn.values).mean())
    R[ag_]["ratio_sd"] = float(np.exp(PI[ag_].mlog.values - meta.logn.values).std())
# per-image view of the reasoning run: where does it still miss?
_pi = pd.DataFrame({"true": meta.true_count, "area_px": AREA.loc[IDS],
                    "gemini": PI["gemini"]["mean"], "reason": PI["reason"]["mean"]})
_pi["reason_ratio"] = _pi.reason / _pi.true
_pi["gemini_ratio"] = _pi.gemini / _pi.true
print(_pi.sort_values("true").round(2).to_string())
R["reason"]["corr_ratio_area_within_n"] = float(np.corrcoef(np.log(_pi.reason_ratio) - np.log(_pi.reason_ratio).groupby(_pi.true.round(-1)).transform("mean"),
                                                            np.log(_pi.area_px) - np.log(_pi.area_px).groupby(_pi.true.round(-1)).transform("mean"))[0, 1])
R["human"]["ratio_9"] = float(np.exp(PI["human"].mlog.values - meta.logn.values)[meta.true_count.values < 15].mean())
R["human"]["ratio_78"] = float(np.exp(PI["human"].mlog.values - meta.logn.values)[meta.true_count.values > 70].mean())
(ROOT / "analysis" / "results" / "mechanisms.json").write_text(json.dumps(R, indent=1, default=float))
print(json.dumps(R, indent=1, default=float))


# ------------------------------------------------------------------ figures
import matplotlib.pyplot as plt  # noqa: E402

HC, GC = F.C["human"], F.C["gemini"]
order = meta.sort_values(["true_count"]).index.tolist()
xpos = np.arange(16) + np.repeat(np.arange(4) * 0.6, 4)  # small gap between the four levels


def fig_answers():
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    jr = np.random.default_rng(3)
    for i, sid in enumerate(order):
        x = xpos[i]
        hv = D[(D.agent == "human") & (D.stimulus_id == sid)].user_estimate.values
        gv = D[(D.agent == "gemini") & (D.stimulus_id == sid)].user_estimate.values
        ax.scatter(x - 0.17 + jr.uniform(-.13, .13, len(hv)), hv, s=7, color=F.lighten(HC, .35), lw=0, zorder=2)
        ax.scatter(x + 0.17 + jr.uniform(-.13, .13, len(gv)), gv, s=7, color=F.lighten(GC, .2), lw=0, zorder=2)
        ax.plot([x - .36, x - .0], [np.median(hv)] * 2, color=HC, lw=1.6, zorder=3, solid_capstyle="butt")
        ax.plot([x + .0, x + .36], [np.median(gv)] * 2, color=GC, lw=1.6, zorder=3, solid_capstyle="butt")
        ax.plot([x - .4, x + .4], [meta.loc[sid, "true_count"]] * 2, color=F.INK, lw=0.9, ls=(0, (2, 1.5)), zorder=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(int(meta.loc[s, "true_count"])) for s in order], fontsize=6.5)
    ax.set_xlabel("True number of dots in the image (one column per image)")
    ax.set_ylabel("Estimated number of dots")
    ax.set_ylim(0, 92)
    F.clean(ax)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=F.lighten(HC, .35), ms=4, label="Humans (50 answers per image)"),
        Line2D([], [], marker="o", ls="", color=F.lighten(GC, .2), ms=4, label="Gemini (20 answers per image)"),
        Line2D([], [], color=F.INK, lw=0.9, ls=(0, (2, 1.5)), label="True number"),
        Line2D([], [], color="#777", lw=1.6, label="Median")],
        loc="upper left", frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    F.save(fig, "answers")


def fig_mech():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.05, 1, 1.05]})
    # a) exponent
    ax = axs[0]
    rows = [("Krueger 1972", [0.72, 0.78], None),
            ("Humans (here)", [R["human"]["b"]], (R["human"]["b_lo"], R["human"]["b_hi"])),
            ("Gemini (here)", [R["gemini"]["b"]], (R["gemini"]["b_lo"], R["gemini"]["b_hi"])),
            ("Gemini + reasoning", [R["reason"]["b"]], (R["reason"]["b_lo"], R["reason"]["b_hi"]))]
    cols_ = ["#9a9a9a", HC, GC, F.C["reason"]]
    for i, ((lab, vals, ci), c) in enumerate(zip(rows, cols_)):
        y = 3 - i
        if ci:
            ax.plot(ci, [y, y], color=c, lw=2)
        ax.scatter(vals, [y] * len(vals), color=c, s=22, zorder=3)
    ax.axvline(1, color=F.INK, lw=0.8, ls=(0, (2, 1.5)))
    ax.text(1.0, 3.55, "perfect\nscaling", ha="right", va="bottom", fontsize=6.5, color=F.MUTED)
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(0.65, 1.02); ax.set_ylim(-0.6, 4.4)
    ax.set_xlabel("Exponent of the log–log slope")
    F.clean(ax, grid="x"); F.panel(ax, "a")
    # b) variability
    ax = axs[1]
    for ag, c, mk in (("human", HC, "o"), ("gemini", GC, "s"), ("reason", F.C["reason"], "P")):
        pi = PI[ag].loc[order]
        n = meta.loc[order, "true_count"].values
        ax.scatter(n, pi.cv.clip(lower=0.0065), s=14, color=c, marker=mk, zorder=3, lw=0)
        ax.axhline(pi.cv[pi.cv > 0].median(), color=c, lw=1, alpha=.6)
    ax.set_yscale("log"); ax.set_ylim(0.0048, 1.6)
    ax.set_yticks([0.0065, 0.01, 0.03, 0.1, 0.3]); ax.set_yticklabels(["0", "0.01", "0.03", "0.1", "0.3"]); ax.minorticks_off()
    ax.set_xlabel("True number of dots"); ax.set_ylabel("Spread of answers (CV)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=HC, ms=4, label="Humans"),
                       Line2D([], [], marker="s", ls="", color=GC, ms=4, label="Gemini"),
                       Line2D([], [], marker="P", ls="", color=F.C["reason"], ms=4.5, label="Gemini + reasoning"),
                       Line2D([], [], color="#777", lw=1, label="Median over images")],
              loc="upper center", ncol=2, frameon=False, fontsize=6.3, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    ax.annotate("same count,\n(7 every time)", xy=(8, 0.0035), fontsize=0, alpha=0)
    F.clean(ax); F.panel(ax, "b")
    # c) model error
    ax = axs[2]
    def pct(x): return 100 * (np.exp(x) - 1)
    for i, ag in enumerate(("human", "gemini", "reason")):
        y = 2 - i
        wv, av = pct(R[ag]["rmse_weber"]), pct(R[ag]["rmse_area"])
        ax.plot([wv, av], [y, y], color="#bbbbbb", lw=1.2, zorder=1)
        ax.scatter([wv], [y], color=F.C["weber"], marker="^", s=30, zorder=3)
        ax.scatter([av], [y], color=F.C["area"], marker="D", s=24, zorder=3)
        ax.text(wv, y + 0.2, f"{wv:.0f}%", ha="center", fontsize=7, color=F.C["weber"])
        ax.text(av, y + 0.2, f"{av:.0f}%", ha="center", fontsize=7, color=F.C["area"])
    ax.set_xscale("log"); ax.set_xlim(1.5, 130); ax.set_ylim(-0.6, 3.5)
    ax.set_xticks([2, 5, 10, 30, 100]); ax.set_xticklabels(["2", "5", "10", "30", "100"])
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Humans", "Gemini", "Gemini +\nreasoning"], fontsize=7)
    ax.set_xlabel("Typical model error (%)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="^", ls="", color=F.C["weber"], ms=5, label="Number only"),
                       Line2D([], [], marker="D", ls="", color=F.C["area"], ms=4.5, label="Ink area")],
              loc="upper center", bbox_to_anchor=(0.6, 1.0), ncol=2, frameon=False, fontsize=6.3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
    F.clean(ax, grid="x"); F.panel(ax, "c")
    fig.tight_layout(w_pad=1.0)
    F.save(fig, "mechanisms")


fig_answers()
fig_mech()
