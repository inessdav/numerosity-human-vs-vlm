"""Figures for the human vs. Gemini dot-estimation experiment. Run stats.py first; writes PNGs to figures/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG, OUT, DATA = ROOT / "figures", ROOT / "analysis" / "results", ROOT / "data"
FIG.mkdir(exist_ok=True)
BASELINES = DATA / "results_baselines_2000.csv"

# ---------------------------------------------------------------- style
for f in font_manager.findSystemFonts():
    if "Inter" in Path(f).name and "Display" not in Path(f).name:
        font_manager.fontManager.addfont(f)
try:
    font_manager.findfont("Inter", fallback_to_default=False)
    FONT = "Inter"
except ValueError:
    FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222", "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#222222", "figure.dpi": 150,
})
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
C = {"human": "#2a6fdb", "gemini": "#e8602c", "reason": "#8a2d0a", "weber": "#1b9e77", "area": "#8c6bb1"}
NAME = {"human": "Humans", "gemini": "Gemini", "reason": "Gemini + reasoning", "weber": "Weber observer", "area": "Area heuristic"}
MARK = {"human": "o", "gemini": "s", "reason": "P", "weber": "^", "area": "D"}
LEVELS = [9, 30, 55, 78]


def load_baselines():
    """Simulated-observer runs (2000 per observer). Regenerate with ai/baselines.py if missing."""
    if not BASELINES.exists():
        raise SystemExit(f"{BASELINES.name} not found. Create it with:\n"
                         f"  python ai/baselines.py --runs 2000 --seed-base 5000 --out data/{BASELINES.name}")
    return pd.read_csv(BASELINES)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200)


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def clean(ax, grid="y"):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)


def lighten(color, amount=0.55):
    c = np.array(matplotlib.colors.to_rgb(color))
    return tuple(c + (1 - c) * amount)


# ---------------------------------------------------------------- data
def load():
    h = pd.read_csv(DATA / "human_results.csv").assign(agent="human")
    g = pd.read_csv(DATA / "results_gemini.csv").assign(agent="gemini")
    r = pd.read_csv(DATA / "results_gemini_reasoning.csv").assign(agent="reason")
    b = load_baselines()
    b["agent"] = np.where(b.model_id == "baseline_weber", "weber", "area")
    cols = ["agent", "subject_id", "stimulus_id", "true_count", "user_estimate"]
    d = pd.concat([x[cols] for x in (h, g, r, b)], ignore_index=True)
    d = d[(d.user_estimate <= 3 * d.true_count) | (d.agent != "human")].copy()
    d["level"] = d.true_count.apply(lambda n: min(LEVELS, key=lambda L: abs(L - n)))
    d["logr"] = np.log(d.user_estimate / d.true_count)
    d["bias"] = 100 * (np.exp(d.logr) - 1)
    return d


def unit_bias(d, agent):
    """Mean bias (%) per participant/run and level."""
    a = d[d.agent == agent]
    u = a.groupby(["subject_id", "level"]).logr.mean().apply(lambda x: 100 * (np.exp(x) - 1)).reset_index(name="bias")
    return u


def mean_ci(u, level):
    x = u[u.level == level].bias.values
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se


# ---- Gemini
def fig_gemini(d):
    g = d[d.agent == "gemini"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.28))
    rng = np.random.default_rng(3)

    # (a) estimates of all runs vs true count
    ax = axes[0]
    ax.plot([0, 90], [0, 90], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    jit = rng.uniform(-0.9, 0.9, len(g))
    ax.scatter(g.true_count + jit, g.user_estimate, s=7, color=lighten(C["gemini"], 0.35), alpha=0.65, lw=0, zorder=2)
    m = g.groupby("true_count").user_estimate.mean()
    ax.scatter(m.index, m.values, s=22, color=C["gemini"], edgecolor="white", lw=0.6, zorder=3)
    ax.set(xlim=(0, 90), ylim=(0, 90), xlabel="True number of dots", ylabel="Gemini estimate")
    ax.set_aspect("equal"); clean(ax, "both"); panel(ax, "a")
    ax.text(43, 45, "perfect accuracy", fontsize=7, color=MUTED, ha="center", va="bottom", rotation=45, rotation_mode="anchor")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=lighten(C["gemini"], 0.35), ms=3.5, label="Single run"),
                       Line2D([], [], marker="o", ls="", color=C["gemini"], mec="white", mew=0.6, ms=5.5, label="Mean of 20 runs (one image)")],
              loc="lower right", frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0.2)

    # (b) bias per level
    ax = axes[1]
    u = unit_bias(d, "gemini")
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    for k, lv in enumerate(LEVELS):
        vals = u[u.level == lv].bias.values
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(vals)), vals, s=9, color=lighten(C["gemini"], 0.35), alpha=0.8, lw=0, zorder=2)
        mu, lo, hi = mean_ci(u, lv)
        ax.errorbar(k, mu, yerr=[[mu - lo], [hi - mu]], fmt="s", color=C["gemini"], ms=5, mec="white", mew=0.6, capsize=0, lw=1.6, zorder=3)
        ax.text(k + 0.22, mu, f"{mu:.0f}%", fontsize=7.5, va="center", color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"~{lv}" for lv in LEVELS])
    ax.set(xlim=(-0.5, 3.6), ylim=(-45, 5), xlabel="Number of dots", ylabel="Bias (% of true number)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=lighten(C["gemini"], 0.35), ms=3.5, label="Single run"),
                       Line2D([], [], marker="s", color=C["gemini"], mec="white", mew=0.6, ms=5.5, lw=1.6, label="Mean with 95% CI"),
                       Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="No bias")],
              loc="lower center", ncol=3, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    clean(ax); panel(ax, "b")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.9)
    save(fig, "gemini")
    plt.close(fig)


# ---- condition effects of models
def dot_row(ax, y, agent, eff, xlim, label=True):
    r = eff
    ax.errorbar(r.pct, y, xerr=[[r.pct - r.pct_lo], [r.pct_hi - r.pct]], fmt=MARK[agent], color=C[agent], ms=6, mec="white", mew=0.6,
                capsize=0, lw=1.8, zorder=3)


def fig_effects(d, with_humans=False, name="effects"):
    eff = pd.read_csv(OUT / "effects.csv")
    agents = ["human", "gemini", "weber", "area"] if with_humans else ["gemini", "weber", "area"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45 if not with_humans else 2.85), sharey=True, gridspec_kw=dict(wspace=0.08))
    lims = {"color": (-16, 16), "size": (-16, 16)}
    titles = {"color": "Different colours", "size": "Different dot sizes"}
    for ax, factor in zip(axes, ("color", "size")):
        ax.axvline(0, color=MUTED, lw=0.8)
        for k, ag in enumerate(agents):
            y = len(agents) - 1 - k
            r = eff[(eff.agent == ag) & (eff.factor == factor)].iloc[0]
            lo, hi = lims[factor]
            if r.pct > hi:
                ax.annotate(f"+{r.pct:.0f}%", xy=(hi - 0.3, y), xytext=(hi - 7.5, y), fontsize=7.5, color=C[ag], va="center", ha="right",
                            arrowprops=dict(arrowstyle="-|>", color=C[ag], lw=1.1, mutation_scale=8))
                continue
            if ag in ("weber", "area"):
                # model predictions (no CI: 2000 simulated observers)
                ax.scatter(r.pct, y, s=34, marker=MARK[ag], facecolor="white", edgecolor=C[ag], lw=1.5, zorder=3)
            else:
                dot_row(ax, y, ag, r, lims[factor])
            ax.text(r.pct, y + 0.34, f"{r.pct:+.1f}%", fontsize=7, color=INK, ha="center", va="bottom")
        ax.set_xlim(*lims[factor]); ax.set_ylim(-0.6, len(agents) - 0.35)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels([NAME[a] for a in agents][::-1] if False else [NAME[a] for a in agents[::-1]])
        ax.set_xlabel("Change in estimate (%) with varied vs. uniform")
        ax.set_title(titles[factor], fontsize=8.5, fontweight="bold", loc="left", pad=6)
        clean(ax, "x")
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", labelleft=False)
    panel(axes[0], "a"); panel(axes[1], "b")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.37 if not with_humans else 0.3, top=0.86)
    handles = [Line2D([], [], marker=MARK[a], color=C[a], mec="white", mew=0.6, ms=6, lw=1.8, label=f"{NAME[a]}: mean with 95% CI")
               for a in agents if a in ("human", "gemini")]
    handles += [Line2D([], [], marker=MARK[a], ls="", mfc="white", mec=C[a], mew=1.5, ms=6, label=f"{NAME[a]}: model prediction")
                for a in agents if a in ("weber", "area")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.3,
               bbox_to_anchor=(0.58, 0.0))
    save(fig, name)
    plt.close(fig)


# ---- comparison (shared section)
def fig_compare(d):
    lv = pd.read_csv(OUT / "levels.csv")
    eff = pd.read_csv(OUT / "effects.csv")
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.42, left=0.075, right=0.985, bottom=0.085, top=0.88)
    rng = np.random.default_rng(5)
    agents = ["human", "gemini", "weber", "area"]

    # (a) estimates vs true count
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 90], [0, 90], color=MUTED, lw=0.8, ls=(0, (4, 3)))
    for ag in agents:
        l = lv[lv.agent == ag].sort_values("level")
        ax.plot(l.true, l.est, color=C[ag], lw=1.8, marker=MARK[ag], ms=5, mec="white", mew=0.6, label=NAME[ag], zorder=3)
    ax.set(xlim=(0, 90), ylim=(0, 90), xlabel="True number of dots", ylabel="Mean estimate"); ax.set_aspect("equal")
    clean(ax, "both"); panel(ax, "a")

    # (b) bias per level
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    off = {"human": -0.27, "gemini": -0.09, "weber": 0.09, "area": 0.27}
    uh = unit_bias(d, "human")
    for k, lvl in enumerate(LEVELS):
        vals = uh[uh.level == lvl].bias.values
        ax.scatter(k + off["human"] + rng.uniform(-0.05, 0.05, len(vals)), vals, s=5, color=lighten(C["human"], 0.55), alpha=0.8, lw=0, zorder=1)
    for ag in agents:
        l = lv[lv.agent == ag].sort_values("level")
        x = np.arange(4) + off[ag]
        ax.errorbar(x, l.bias, yerr=[l.bias - l.bias_lo, l.bias_hi - l.bias], fmt=MARK[ag], color=C[ag], ms=5, mec="white", mew=0.6, lw=1.4, capsize=0, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"~{v}" for v in LEVELS])
    ax.set(ylim=(-75, 62), xlabel="Number of dots", ylabel="Bias (% of true number)", xlim=(-0.55, 3.55))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=lighten(C["human"], 0.55), ms=3, label="Single participant"),
                       Line2D([], [], marker="o", color="#555555", ms=4.5, lw=1.4, mec="white", mew=0.6, label="Mean with 95% CI")],
              loc="upper center", ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.15)
    clean(ax); panel(ax, "b")

    # (c) variability
    ax = fig.add_subplot(gs[1, 0])
    for ag in agents:
        l = lv[lv.agent == ag].sort_values("level")
        ax.plot(l.level, l.cv, color=C[ag], lw=1.8, marker=MARK[ag], ms=5, mec="white", mew=0.6, zorder=3)
    ax.set(xlim=(0, 90), ylim=(0, 1.05), xlabel="Number of dots", ylabel="Coefficient of variation")
    ax.set_xticks(LEVELS)
    clean(ax); panel(ax, "c")

    # (d) condition effects
    ax = fig.add_subplot(gs[1, 1])
    ax.axvline(0, color=MUTED, lw=0.8)
    rows = []
    for factor, lab in (("color", "Different\ncolours"), ("size", "Different\ndot sizes")):
        for ag in agents:
            rows.append((factor, ag))
    ypos, ytxt = {}, []
    y = 0
    for factor in ("size", "color"):
        for ag in agents[::-1]:
            ypos[(factor, ag)] = y; y += 1
        y += 1.1
    lim = 16
    for (factor, ag), y in ypos.items():
        r = eff[(eff.agent == ag) & (eff.factor == factor)].iloc[0]
        if r.pct > lim:
            ax.annotate("", xy=(lim - 0.3, y), xytext=(lim - 5.5, y), arrowprops=dict(arrowstyle="-|>", color=C[ag], lw=1.3, mutation_scale=8))
            ax.text(lim - 6.2, y, f"+{r.pct:.0f}% (off scale)", fontsize=7, color=C[ag], va="center", ha="right")
            continue
        if ag in ("weber", "area"):
            ax.scatter(r.pct, y, s=30, marker=MARK[ag], facecolor="white", edgecolor=C[ag], lw=1.4, zorder=3)
        else:
            ax.errorbar(r.pct, y, xerr=[[r.pct - r.pct_lo], [r.pct_hi - r.pct]], fmt=MARK[ag], color=C[ag], ms=5.5, mec="white", mew=0.6, lw=1.6, capsize=0, zorder=3)
    ax.set_yticks([ypos[("size", "area")], ypos[("size", "human")] - 0.0])
    ax.set_yticks([np.mean([ypos[("size", a)] for a in agents]), np.mean([ypos[("color", a)] for a in agents])])
    ax.set_yticklabels(["Different\ndot sizes", "Different\ncolours"], fontsize=8)
    handles = [Line2D([0], [0], color=C[a], lw=1.8, marker=MARK[a], ms=5.5, mec="white", mew=0.6, label=NAME[a]) for a in agents]
    handles.append(Line2D([0], [0], color=MUTED, lw=0.9, ls=(0, (4, 3)), label="Perfect accuracy / no change"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=5, frameon=False, handlelength=1.8, columnspacing=1.5, fontsize=7.2)
    ax.set(xlim=(-lim, lim), ylim=(-0.8, max(ypos.values()) + 0.7), xlabel="Change in estimate (%) with varied vs. uniform")
    clean(ax, "x"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); panel(ax, "d")
    save(fig, "comparison")
    plt.close(fig)


# ---------------------------------------------------------------- Human figures
CONDITIONS = [("uniform", "uniform", "One colour\nsame size"), ("varied", "uniform", "Many colours\nsame size"),
              ("uniform", "varied", "One colour\nvaried size"), ("varied", "varied", "Many colours\nvaried size")]
UNIFORM_COLOR = (1, 3, 5, 7, 22, 24, 26, 28)
UNIFORM_SIZE = (1, 3, 5, 7, 8, 10, 12, 14)


def human_data():
    h = pd.read_csv(DATA / "human_results.csv")
    h = h[h.user_estimate <= 3 * h.true_count].copy()
    num = h.stimulus_id.str.extract(r"(\d+)")[0].astype(int)
    h["color"] = np.where(num.isin(UNIFORM_COLOR), "uniform", "varied")
    h["size"] = np.where(num.isin(UNIFORM_SIZE), "uniform", "varied")
    h["level"] = h.true_count.apply(lambda n: min(LEVELS, key=lambda L: abs(L - n)))
    h["logr"] = np.log(h.user_estimate / h.true_count)
    h["rt"] = h.reaction_time_ms / 1000
    return h


def fig_human_curve(h):
    from matplotlib.colors import LinearSegmentedColormap
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.5))
    rng = np.random.default_rng(11)
    ax = axes[0]
    ax.plot([0, 90], [0, 90], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.scatter(h.true_count + rng.uniform(-1.0, 1.0, len(h)), h.user_estimate, s=6, color=lighten(C["human"], 0.6), alpha=0.7, lw=0, zorder=2)
    m = h.groupby("true_count").user_estimate.mean()
    ax.scatter(m.index, m.values, s=24, color=C["human"], edgecolor="white", lw=0.6, zorder=3)
    ax.set(xlim=(0, 90), ylim=(0, 100), xlabel="True number of dots", ylabel="Estimate of the participant")
    ax.text(43, 45, "perfect accuracy", fontsize=7, color=MUTED, ha="center", va="bottom", rotation=45, rotation_mode="anchor")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=lighten(C["human"], 0.6), ms=3.5, label="Single answer"),
                       Line2D([], [], marker="o", ls="", color=C["human"], mec="white", mew=0.6, ms=5.5, label="Mean of 50 answers (one image)")],
              loc="upper left", frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0.2)
    clean(ax, "both"); panel(ax, "a")

    ax = axes[1]
    cm = LinearSegmentedColormap.from_list("hb", ["#eef3fc", C["human"], "#123a82"])
    M = np.zeros((4, 4))
    for i, (col, siz, _) in enumerate(CONDITIONS):
        for j, lv in enumerate(LEVELS):
            g = h[(h.color == col) & (h["size"] == siz) & (h.level == lv)]
            M[i, j] = 100 * (np.exp(g.logr.mean()) - 1)
    im = ax.imshow(-M, cmap=cm, vmin=0, vmax=50, aspect="auto")
    for i in range(4):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white" if -v > 18 else INK, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"~{v}" for v in LEVELS])
    ax.set_yticks(range(4)); ax.set_yticklabels([c[2] for c in CONDITIONS], fontsize=7.5, linespacing=1.15)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xlabel("Number of dots", labelpad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True); ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    panel(ax, "b")
    ax.text(0.0, 1.17, "", transform=ax.transAxes)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.8)
    fig.text(0.62, 0.06, "Bias of the mean estimate (darker = larger underestimation)", fontsize=7, color=MUTED, ha="center")
    save(fig, "human_curve")
    plt.close(fig)


def fig_human_effects(h):
    import json
    nums = json.load(open(OUT / "numbers.json"))
    eff = pd.read_csv(OUT / "effects.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw=dict(wspace=0.32, hspace=0.5))
    rng = np.random.default_rng(2)
    ok = h[h.reaction_time_ms <= 200_000]
    specs = [("color", "Dot colour", "uniform\ncolours", "different\ncolours"), ("size", "Dot size", "uniform\nsizes", "different\nsizes")]
    for c_, (factor, title, l0, l1) in enumerate(specs):
        for r_, kind in enumerate(("bias", "rt")):
            ax = axes[r_, c_]
            if kind == "bias":
                u = h.groupby(["subject_id", factor]).logr.mean().unstack()
                y0, y1 = 100 * (np.exp(u["uniform"]) - 1), 100 * (np.exp(u["varied"]) - 1)
                e = eff[(eff.agent == "human") & (eff.factor == factor)].iloc[0]
                txt = f"{e.pct:+.1f}%  [{e.pct_lo:.1f}, {e.pct_hi:.1f}]"
                ylabel = "Bias (% of true number)"
            else:
                u = ok.groupby(["subject_id", factor]).rt.apply(lambda x: np.exp(np.log(x).mean())).unstack()
                y0, y1 = u["uniform"], u["varied"]
                txt = f"{nums[f'humanrt{factor}pct']:+.1f}%  [{nums[f'humanrt{factor}lo']:.1f}, {nums[f'humanrt{factor}hi']:.1f}]"
                ylabel = "Reaction time (s)"
            for a_, b_ in zip(y0, y1):
                ax.plot([0, 1], [a_, b_], color=lighten(C["human"], 0.72), lw=0.7, zorder=1)
            ax.scatter(np.zeros(len(y0)) + rng.uniform(-0.04, 0.04, len(y0)), y0, s=7, color=lighten(C["human"], 0.4), lw=0, zorder=2)
            ax.scatter(np.ones(len(y1)) + rng.uniform(-0.04, 0.04, len(y1)), y1, s=7, color=lighten(C["human"], 0.4), lw=0, zorder=2)
            for x_, y_ in ((0, y0), (1, y1)):
                if kind == "rt":
                    lg = np.log(y_); mu = lg.mean(); se = lg.std(ddof=1) / np.sqrt(len(lg))
                    mean, lo, hi = np.exp(mu), np.exp(mu - 1.96 * se), np.exp(mu + 1.96 * se)
                else:
                    mean = y_.mean(); se = y_.std(ddof=1) / np.sqrt(len(y_)); lo, hi = mean - 1.96 * se, mean + 1.96 * se
                ax.errorbar(x_, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color=C["human"], ms=5, mec="white", mew=0.7, lw=2.2, capsize=3, capthick=1.6, zorder=4)
            ax.plot([0, 1], [y0.mean() if kind == "bias" else np.exp(np.log(y0).mean()), y1.mean() if kind == "bias" else np.exp(np.log(y1).mean())], color=C["human"], lw=2, zorder=3)
            ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1]); ax.set_xticklabels([l0, l1])
            if kind == "rt":
                ax.set_yscale("log"); ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(["1", "2", "4", "8", "16"]); ax.set_ylim(0.8, 22)
                ax.axhline(1e9)
            else:
                ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
                ax.set_ylim(-75, 45)
            ax.set_ylabel(ylabel)
            ax.set_title(txt, fontsize=8, color=INK, loc="center", pad=5)
            clean(ax)
    axes[0, 0].text(-0.28, 1.2, "Bias of the estimate", transform=axes[0, 0].transAxes, fontsize=8.5, fontweight="bold")
    axes[1, 0].text(-0.28, 1.2, "Reaction time", transform=axes[1, 0].transAxes, fontsize=8.5, fontweight="bold")
    for ax, ltr in zip(axes.flat, "abcd"):
        panel(ax, ltr)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.14, top=0.9)
    fig.legend(handles=[Line2D([], [], marker="o", color=lighten(C["human"], 0.72), mfc=lighten(C["human"], 0.4), mec=lighten(C["human"], 0.4), ms=3.5, lw=0.8, label="One participant"),
                        Line2D([], [], marker="o", color=C["human"], mec="white", mew=0.7, ms=5.5, lw=2, label="Mean of all participants with 95% CI")],
               loc="lower center", ncol=2, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=2.0, bbox_to_anchor=(0.55, 0.0))
    save(fig, "human_effects")
    plt.close(fig)


# ---------------------------------------------------------------- Figure: Gemini with reasoning
def fig_reasoning(d):
    lv = pd.read_csv(OUT / "levels.csv")
    eff = pd.read_csv(OUT / "effects.csv")
    agents = ["human", "gemini", "reason"]
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.45, left=0.075, right=0.985, bottom=0.085, top=0.89)
    rng = np.random.default_rng(8)

    # (a) mean estimate per image
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 90], [0, 90], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for ag, ms in (("human", 4.5), ("gemini", 4.5), ("reason", 5.5)):
        m = d[d.agent == ag].groupby("true_count").user_estimate.mean()
        ax.scatter(m.index, m.values, s=ms ** 2 * 1.2, marker=MARK[ag], color=C[ag], edgecolor="white", lw=0.5, zorder=3 if ag == "reason" else 2,
                   alpha=0.95)
    ax.set(xlim=(0, 90), ylim=(0, 90), xlabel="True number of dots", ylabel="Mean estimate of one image")
    ax.set_aspect("equal"); clean(ax, "both"); panel(ax, "a")
    ax.text(41, 43, "perfect accuracy", fontsize=7, color=MUTED, ha="center", va="bottom", rotation=45, rotation_mode="anchor")

    # (b) bias per level
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    off = {"human": -0.27, "gemini": 0.0, "reason": 0.27}
    ur = unit_bias(d, "reason")
    for k, lvl in enumerate(LEVELS):
        vals = ur[ur.level == lvl].bias.values
        ax.scatter(k + off["reason"] + 0.13 + rng.uniform(-0.03, 0.03, len(vals)), vals, s=6, color=lighten(C["reason"], 0.55), lw=0, zorder=1)
    for ag in agents:
        l = lv[lv.agent == ag].sort_values("level")
        x = np.arange(4) + off[ag]
        ax.errorbar(x, l.bias, yerr=[l.bias - l.bias_lo, l.bias_hi - l.bias], fmt=MARK[ag], color=C[ag], ms=5.5, mec="white", mew=0.6, lw=1.5,
                    capsize=0, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"~{v}" for v in LEVELS])
    ax.set(ylim=(-52, 22), xlabel="Number of dots", ylabel="Bias (% of true number)", xlim=(-0.55, 3.6))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=lighten(C["reason"], 0.55), ms=3.5, label="Single run with reasoning"),
                       Line2D([], [], marker="o", color="#555555", ms=4.5, lw=1.4, mec="white", mew=0.6, label="Mean with 95% CI")],
              loc="upper center", ncol=2, frameon=False, fontsize=6.8, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
    clean(ax); panel(ax, "b")

    # (c) variability
    ax = fig.add_subplot(gs[1, 0])
    for ag in agents:
        l = lv[lv.agent == ag].sort_values("level")
        ax.plot(l.level, l.cv, color=C[ag], lw=1.8, marker=MARK[ag], ms=5.5, mec="white", mew=0.6, zorder=3)
    ax.set(xlim=(0, 90), ylim=(0, 0.42), xlabel="Number of dots", ylabel="Coefficient of variation")
    ax.set_xticks(LEVELS)
    clean(ax); panel(ax, "c")

    # (d) colour and size effects
    ax = fig.add_subplot(gs[1, 1])
    ax.axvline(0, color=MUTED, lw=0.8)
    ypos, y = {}, 0
    for factor in ("size", "color"):
        for ag in agents[::-1]:
            ypos[(factor, ag)] = y; y += 1
        y += 1.0
    for (factor, ag), y in ypos.items():
        r = eff[(eff.agent == ag) & (eff.factor == factor)].iloc[0]
        ax.errorbar(r.pct, y, xerr=[[r.pct - r.pct_lo], [r.pct_hi - r.pct]], fmt=MARK[ag], color=C[ag], ms=5.5, mec="white", mew=0.6, lw=1.6,
                    capsize=0, zorder=3)
    ax.set_yticks([np.mean([ypos[("size", a)] for a in agents]), np.mean([ypos[("color", a)] for a in agents])])
    ax.set_yticklabels(["Different\ndot sizes", "Different\ncolours"], fontsize=8)
    ax.set(xlim=(-14, 14), ylim=(-0.8, max(ypos.values()) + 0.7), xlabel="Change in estimate (%) with varied vs. uniform")
    clean(ax, "x"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); panel(ax, "d")

    handles = [Line2D([0], [0], color=C[a], lw=1.8, marker=MARK[a], ms=5.5, mec="white", mew=0.6, label=NAME[a]) for a in agents]
    handles.append(Line2D([0], [0], color=MUTED, lw=0.9, ls=(0, (4, 3)), label="Perfect accuracy / no change"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=4, frameon=False, handlelength=1.8, columnspacing=1.6, fontsize=7.4)
    save(fig, "reasoning")
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    fig_gemini(d)
    fig_effects(d)
    fig_compare(d)
    hh = human_data()
    fig_human_curve(hh)
    fig_human_effects(hh)
    fig_reasoning(d)
    print("figures written to", FIG)
