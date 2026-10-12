"""Figures of the paper from the run directories (runs/<group>/<task>/<model>/<seed>/metrics.jsonl).

Uses the run helpers of scripts/main_tables.py, as paper_tables.py does: the same mapping of our model, the same runs
(BSuite L8-L64: rl3) and the rerun rule of the R2I lengths (the reported seed).
Rerun after new results: python paper/scripts/make_figures.py  ->  paper/fig/*.pdf
A curve of a run that is still going is drawn up to its latest evaluation.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS, FIG = ROOT / "runs", ROOT / "paper" / "fig"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
import main_tables as mt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
})

# Okabe-Ito colours; ours in vermillion and drawn on top. The models of the main tables (paper_tables.MODELS);
# A+B+G is an ablation of Spiking-WM (EXPERIMENTS.md §19.44) and appears in no figure.
STYLE = {
    "base_tau2": dict(label="Spiking-WM", color="#E69F00"),
    "LIF_tau2": dict(label="LIF", color="#56B4E9"),
    "GRSN_tau2": dict(label="GRSN", color="#009E73"),
    "DHSNN": dict(label="DH-SNN", color="#CC79A7"),
    "TCLIF": dict(label="TC-LIF", color="#0072B2"),
    "ann_tau2": dict(label="DreamerV3 (ANN)", color="#000000", ls="--"),
    "R2I": dict(label="R2I (ANN)", color="#7F7F7F", ls="-."),
    "OURS": dict(label="SD-Dreamer (ours)", color="#D55E00", lw=1.8, zorder=5),
}
BASELINES = ["base_tau2", "LIF_tau2", "GRSN_tau2", "DHSNN", "TCLIF", "ann_tau2"]


def curve(path, key):
    r = [x for x in mt.rows(path) if key in x]
    return np.array([x["step"] for x in r], float), np.array([x[key] for x in r], float)


def ema(y, a=0.6):
    out, m = [], None
    for v in y:
        m = v if m is None else a * m + (1 - a) * v
        out.append(m)
    return np.array(out)


def run(model, family):
    return mt.key(model, family)


def final(path, key="eval_return", n=3):
    _, y = curve(path, key)
    return float(np.mean(y[-n:])) if len(y) else None


def finished(path):
    return mt.exit_code(path) == "0"


def memory_length():
    """Fig. 1: final evaluation return against the BSuite memory length; left: L8-L64 (0.15-0.5M env
    steps), right: L51-L101 under the protocol of R2I (1M env steps)."""
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    groups = [[(8, "rl3"), (16, "rl3"), (32, "rl3"), (64, "rl3")], [(51, "r2i"), (71, "r2i"), (101, "r2i")]]
    pos = {8: 0, 16: 1, 32: 2, 64: 3, 51: 4.6, 71: 5.6, 101: 6.6}
    for model in BASELINES + ["OURS"]:
        st = dict(STYLE[model])
        label = st.pop("label")
        for g, group in enumerate(groups):
            xs, ys, live = [], [], []
            for L, sub in group:
                path = f"{sub}/bsuite_memlen-{L}/{run(model, 'bsuite')}"
                v = final(path)
                if v is not None:
                    xs.append(pos[L]), ys.append(v), live.append(not finished(path))
            if xs:
                ax.plot(xs, ys, label=label if g == 0 else None, marker="o", ms=2.8, **st)
                for x, y, l in zip(xs, ys, live):  # runs still in progress: open marker
                    if l:
                        ax.plot([x], [y], marker="o", ms=4.2, mfc="white", mec=st["color"], mew=1.0, ls="none", zorder=6)
    ax.axvline(3.8, color="0.6", lw=0.6, ls=":")
    ax.text(1.5, 1.13, "0.15-0.5M env steps", ha="center", fontsize=6.5, color="0.35")
    ax.text(5.6, 1.13, "R2I protocol, 1M steps", ha="center", fontsize=6.5, color="0.35")
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([str(k) for k in pos])
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.set_ylim(-0.42, 1.2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("memory length $L$ (steps between cue and recall)")
    ax.set_ylabel("final return")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.3), handlelength=1.2, handletextpad=0.4,
              columnspacing=0.7)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "memory_length.pdf")
    plt.close(fig)


def rl_curves():
    """Fig. 4: learning curves under the R2I protocol (BSuite L51/L71/L101) and POPGym RepeatPrevious."""
    panels = [("r2i/bsuite_memlen-51", "bsuite", "BSuite $L=51$"), ("r2i/bsuite_memlen-71", "bsuite", "BSuite $L=71$"),
              ("r2i/bsuite_memlen-101", "bsuite", "BSuite $L=101$"),
              ("popgym/popgym_RepeatPreviousEasy", "popgym", "RepeatPrev. Easy"),
              ("popgym/popgym_RepeatPreviousMedium", "popgym", "RepeatPrev. Medium")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.9, 1.5), sharey=False)
    for ax, (task, fam, title) in zip(axes, panels):
        for model in BASELINES + ["OURS"]:
            x, y = curve(f"{task}/{run(model, fam)}", "eval_return")
            if not len(x):
                continue
            st = dict(STYLE[model])
            ax.plot(x / 1e6, ema(y), **st)
        if fam == "popgym":  # R2I, official code: training-episode score, mean per 50k env steps
            sc = RUNS / f"r2i_agent/{task.split('/')[1]}/R2I/s0/scores.jsonl"
            if sc.exists():
                r = [json.loads(l) for l in sc.read_text().splitlines() if l.strip()]
                s_, v_ = np.array([q["step"] for q in r]), np.array([q["episode/score"] for q in r])
                bins = np.arange(0, s_.max() + 50_000, 50_000)
                idx = np.digitize(s_, bins)
                xb = [s_[idx == i].mean() for i in np.unique(idx)]
                yb = [v_[idx == i].mean() for i in np.unique(idx)]
                ax.plot(np.array(xb) / 1e6, ema(np.array(yb)), **STYLE["R2I"])
        ax.set_title(title, pad=2)
        ax.set_xlabel("env steps (M)", labelpad=1)
        lo = -1.0 if fam == "popgym" else -0.5
        ax.set_ylim(lo, 1.05)
        if fam == "popgym":
            ax.axhline(-0.5, color="0.75", lw=0.5, zorder=0)
    axes[0].set_ylabel("eval. return")
    handles = [plt.Line2D([], [], **{k: v for k, v in STYLE[m].items() if k != "zorder"}) for m in STYLE]
    fig.legend(handles=handles, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=2.0)
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.88), w_pad=0.6)
    fig.savefig(FIG / "rl_curves.pdf")
    plt.close(fig)


def probe_panels():
    """Supp. figure: (a) ablations of the final model on cue recall 64 steps back; (b) recall of the exact lag,
    RepeatPrevious Medium probe (input of exactly 32 steps ago). Runs as in main_tables.ABLATIONS."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.2))
    ax = axes[0]
    sweep = [("SDDreset_tau1000", "no PDM (cleared every step)", "#999999", ":"),
             ("SDD_tau16", r"single $\tau=16$", "#56B4E9", "-."),
             ("SDD_tau128", r"single $\tau=128$", "#0072B2", "--"),
             ("SDDeuler_tau1000", r"no NWG (Euler gain $1-\lambda$)", "#009E73", (0, (3, 1, 1, 1))),
             (mt.OURS["probe"], r"SD-Dreamer ($\tau\in[1,1000]$, PDM and NWG)", "#D55E00", "-")]
    for name, label, color, ls in sweep:
        x, y = curve(f"recall/bsuite_memlen-64-recall/{name}", "eval_auc")
        if len(x):
            ax.plot(x / 1e3, y, label=label, color=color, ls=ls, lw=1.8 if name == mt.OURS["probe"] else 1.2)
    ax.set_title("(a) cue recall, 64 steps back", pad=2)
    ax.set_xlabel("world model updates (k)")
    ax.set_ylabel("reward AUC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=2.2, columnspacing=1.0)
    ax = axes[1]
    lag = [(m, None) for m in BASELINES] + [
           ("SDDnores_tau1000", dict(label="SD-Dreamer, no resonant channels", color="#D55E00", ls=":", lw=1.4)),
           ("SDDres_tau1000", dict(label="SD-Dreamer, all channels resonant", color="#D55E00", ls="--", lw=1.2)),
           (mt.OURS["probe"], dict(STYLE["OURS"], label="SD-Dreamer (ours)"))]
    for name, st in lag:
        st = dict(STYLE[name]) if st is None else st
        x, y = curve(f"diag_rp/popgym_RepeatPreviousMedium/{name}", "eval_auc")
        if len(x):
            ax.plot(x / 1e3, y, **st)
    ax.set_title("(b) card exactly 32 steps back (RepeatPrevious Medium)", pad=2)
    ax.set_xlabel("world model updates (k)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, handlelength=2.0, columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "probe_panels.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


MM_MODELS = {"base_tau2": "base", "LIF_tau2": "lif", "GRSN_tau2": "grsn", "ann_tau2": "ann", "OURS": "ours"}
MM_DATA = ROOT / "data" / "memory_maze_9x9_official" / "eval.zip"


def mm_visible(layout, agent, direction, obj, half_fov=np.pi / 4):
    """The object is in view: within the 90 degree field of view and no wall tile on the line of sight
    (layout[y, x] = 1 marks a free tile)."""
    v = obj - agent
    dist = np.linalg.norm(v)
    if dist < 1e-6:
        return True
    if np.arccos(np.clip(v @ direction / dist, -1, 1)) > half_fov:
        return False
    pts = agent + np.outer(np.linspace(0, 1, int(dist * 20) + 2), v)
    ix = np.clip(np.floor(pts).astype(int), 0, layout.shape[0] - 1)
    return bool(np.all(layout[ix[:, 1], ix[:, 0]] == 1))


def mm_since_seen(d):
    """Steps since each object was last in view, (episodes, steps, 3); a large number before the first sighting."""
    E, T = d["agent_pos"].shape[:2]
    out = np.full((E, T, 3), 10**6)
    for e in range(E):
        for o in range(3):
            last = -10**6
            for t in range(T):
                if mm_visible(d["maze_layout"][e], d["agent_pos"][e, t], d["agent_dir"][e, t], d["targets_pos"][e, o]):
                    last = t
                out[e, t, o] = t - last
    return out


def mm_frames(episode, steps):
    """First-person frames of an evaluation trajectory, read from the official dataset (as probe_mm.Episodes)."""
    import io
    import zipfile
    with zipfile.ZipFile(MM_DATA) as zf:
        name = sorted(n for n in zf.namelist() if n.endswith(".npz"))[episode]
        return np.load(io.BytesIO(zf.read(name)))["image"][list(steps)]


def mmaze_viz(seen=5, gone=40, shown=(-3, 0, 5, 15), maps=(3, 10, 15)):
    """Fig.: Memory Maze 9x9, the first 100 evaluation trajectories (fig/data/mmaze_viz.npz, scripts/viz_infer.py mmaze).
    (a) first-person views around the first time that an object, in view for `seen` steps, leaves the view for at
    least `gone` steps; (b) the positions of that object that the probe of each model predicts `maps` steps later;
    (c) object error against the steps since the object was last in view, all trajectories."""
    d = dict(np.load(FIG / "data" / "mmaze_viz.npz"))
    since = mm_since_seen(d)
    # the example: first trajectory and step (in order) that satisfy the rule; no choice by the predictions
    ev = next((e, t, o) for e in range(len(since)) for t in range(seen, since.shape[1] - gone) for o in range(3)
              if since[e, t, o] == 0 and np.all(since[e, t - seen: t, o] == 0) and np.all(since[e, t + 1: t + 1 + gone, o] > 0))
    e, t0, o = ev
    fig = plt.figure(figsize=(6.9, 2.1))
    gs = fig.add_gridspec(2, 8, width_ratios=[0.62, 0.62, 0.06, 1, 1, 1, 0.3, 2.05], wspace=0.06, hspace=0.16,
                          left=0.005, right=0.99, top=0.87, bottom=0.19)
    frames = mm_frames(int(d["episodes"][e]), [t0 + k for k in shown])
    for k, (dt, img) in enumerate(zip(shown, frames)):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        label = "in view" if dt < 0 else "last seen" if dt == 0 else f"+{dt}"
        ax.text(3, 10, label, color="white", fontsize=6)
        if k == 0:
            ax.set_title("(a) first-person view", loc="left", pad=2)
    for k, dt in enumerate(maps):
        ax = fig.add_subplot(gs[:, 3 + k])
        t1 = t0 + dt
        ax.imshow(1 - d["maze_layout"][e], cmap="Greys", vmin=0, vmax=2.2, origin="lower", extent=(0, 9, 0, 9))
        ax.plot(*d["agent_pos"][e, t0: t1 + 1].T, color="#9AA0A6", lw=0.8)
        p, q = d["agent_pos"][e, t1], d["agent_dir"][e, t1]
        ax.annotate("", xy=p + 0.8 * q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#1F2328", lw=0.8))
        ax.scatter(*d["targets_pos"][e, o], s=60, facecolor="none", edgecolor="#1F2328", lw=1.1, zorder=4)
        for m, n in MM_MODELS.items():
            ax.scatter(*d["pred_" + n][e, t1, o], s=30 if m == "OURS" else 12, color=STYLE[m]["color"],
                       marker="*" if m == "OURS" else "o", zorder=5 if m == "OURS" else 3, lw=0)
        ax.set_xlim(0, 9), ax.set_ylim(0, 9), ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_linewidth(0.4), sp.set_color("#9AA0A6")
        ax.text(4.5, -0.5, f"+{dt} steps", ha="center", va="top", fontsize=7)
        if k == 0:
            ax.set_title("(b) predicted position of the object (ring)", loc="left", pad=2)
    ax = fig.add_subplot(gs[:, 7])
    edges = np.array([0, 1, 3, 6, 11, 21, 51, 101, 201, 1001])
    for m, n in MM_MODELS.items():
        err = ((d["pred_" + n] - d["targets_pos"][:, None]) ** 2).sum(-1)[:, 1:]
        s = since[:, 1:]
        y = [err[(s >= a) & (s < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        x = [np.sqrt(a * (b - 1)) if a else 0.5 for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(x, y, marker="o", ms=2.2, **{k: v for k, v in STYLE[m].items() if k != "label"}, label=STYLE[m]["label"])
    ax.set_xscale("log")
    ax.set_xlim(0.28, 700)
    ax.set_xticks([0.5, 2, 10, 100], ["in view", "2", "10", "100"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("steps since the object was last in view")
    ax.set_ylabel("squared error")
    ax.set_title("(c) 100 evaluation trajectories", loc="left", pad=2)
    ax.legend(loc="lower right", handlelength=1.6, ncol=1)
    fig.savefig(FIG / "mmaze_viz.pdf")
    plt.close(fig)
    return ev


def dendrites_viz(show=(0, 101)):
    """Fig.: BSuite memory length 101, two evaluation episodes that differ only in the cue
    (fig/data/dendrites_viz.npz, scripts/viz_infer.py dendrites). (a) difference of the spikes that carry the
    recurrent state; (b) difference of the dendritic channels of the second PDN layer, integrators sorted by
    retention time and resonators by phase; (c) the difference of three resonant channels in their (a, b) plane."""
    d = np.load(FIG / "data" / "dendrites_viz.npz")
    M = int(d["channels"])
    n = 512 * M
    mem = d["mem_d1"]
    delta = mem[0] - mem[1]
    da, db = delta[:, :n], delta[:, n:]
    tau, theta = d["tau1"], d["theta1"]
    res = theta != 0
    t = np.arange(delta.shape[0]) - 1  # step 0: the first cue row
    fig = plt.figure(figsize=(6.9, 2.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.3, 1.3, 1.0], wspace=0.42, left=0.065, right=0.965, top=0.86, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    for name, key in (("Spiking-WM", "base_tau2"), ("SD-Dreamer", "OURS")):
        r = d["rate_" + ("ours" if key == "OURS" else "base")]
        y = np.abs(r[0] - r[1]).mean(-1) / r.mean()
        ax.plot(t[1:], y[1:], color=STYLE[key]["color"], lw=STYLE[key].get("lw", 1.2), label=name + " spikes")
    y = np.abs(da).mean(-1) / np.abs(mem[0][:, :n]).mean()
    ax.plot(t[1:], y[1:], color=STYLE["OURS"]["color"], ls="--", lw=1.2, label="SD-Dreamer dendrites")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1e-2, 1e-1, 1])
    ax.set_xlabel("steps after the cue")
    ax.set_ylabel("relative difference")
    ax.set_title("(a) cue A vs. cue B", loc="left", pad=2)
    ax.legend(loc="upper right", handlelength=1.6)
    freq = np.minimum(theta, 2 * np.pi - theta)  # rotation per step; theta and 2 pi - theta oscillate alike
    for k, (sel, order, title) in enumerate((
            (~res, np.argsort(tau[~res]), "(b) integrator channels"),
            (res, np.argsort(freq[res]), "(c) resonant channels"))):
        ax = fig.add_subplot(gs[1 + k])
        z = da[:, sel][:, order]
        z = z / (np.abs(z).max(0, keepdims=True) + 1e-9)
        ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                  extent=(t[0] - 0.5, t[-1] + 0.5, sel.sum(), 0))
        ax.set_xlabel("steps after the cue")
        ax.set_title(title, loc="left", pad=2)
        if k == 0:
            q = np.sort(tau[~res])
            ticks = [np.searchsorted(q, v) for v in (1, 10, 100, 1000) if v <= q[-1] + 1]
            ax.set_yticks(ticks, [str(v) for v in (1, 10, 100, 1000)][: len(ticks)])
            ax.set_ylabel(r"retention time $\tau$")
        else:
            q = np.sort(freq[res])
            per = [100, 20, 5, 2]
            ticks = [np.searchsorted(q, 2 * np.pi / v) for v in per]
            ax.set_yticks(ticks, [str(v) for v in per])
            ax.set_ylabel("period (steps)")
    ax = fig.add_subplot(gs[3])
    idx = np.where(res & (tau > 200))[0]
    for c, (period, col) in enumerate(zip((50, 25, 12), ["#0072B2", "#009E73", "#CC79A7"])):
        j = idx[np.argmin(np.abs(freq[idx] - 2 * np.pi / period))]
        y = da[1:, j] / np.abs(da[1:, j]).max()
        ax.plot(t[1:], y - 2.3 * c, color=col, lw=0.9)
        ax.text(102, -2.3 * c, f"{2 * np.pi / freq[j]:.0f}", fontsize=6, va="center", color=col)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("steps after the cue")
    ax.set_title("(d) three resonators", loc="left", pad=2)
    ax.text(102, 1.1, "period", fontsize=6, ha="left", color="#5F6670")
    fig.savefig(FIG / "dendrites_viz.pdf")
    plt.close(fig)



def walker_viz(steps=(1, 5, 10, 15, 20, 30, 45)):
    """Fig.: DMC walker walk from pixels, open-loop prediction (fig/data/walker_viz.npz, scripts/viz_infer.py walker).
    Left: one segment of an evaluation episode, the last observed frame and the frames predicted from the actions
    alone; right: pixel MSE of the prediction over the segments of the episode."""
    d = np.load(FIG / "data" / "walker_viz.npz")
    C = int(d["context"])
    rows = [("truth", None, d["truth"])] + [(STYLE[k]["label"].replace(" (ours)", "").replace(" (ANN)", ""), k, d["pred_" + n])
                                           for k, n in (("base_tau2", "base"), ("ann_tau2", "ann"), ("OURS", "ours"))]
    cols = [C - 1] + [C - 1 + h for h in steps]
    fig = plt.figure(figsize=(6.9, 2.35))
    gs = fig.add_gridspec(len(rows), len(cols) + 2, width_ratios=[1] * len(cols) + [0.75, 2.6], wspace=0.04, hspace=0.06,
                          left=0.095, right=0.99, top=0.9, bottom=0.17)
    for r, (name, key, frames) in enumerate(rows):
        for c, t in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(frames[t])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(c == 0), sp.set_color("#5F6670"), sp.set_linewidth(0.8)
            if r == 0:
                ax.set_title("observed" if c == 0 else f"+{t - C + 1}", pad=2, fontsize=7)
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7,
                              color=STYLE["OURS"]["color"] if key == "OURS" else "black",
                              fontweight="bold" if key == "OURS" else "normal")
    ax = fig.add_subplot(gs[:, -1])
    for k, n in (("base_tau2", "base"), ("ann_tau2", "ann"), ("OURS", "ours")):
        y = d["mse_" + n][C:]
        ax.plot(np.arange(1, len(y) + 1), y, **{a: v for a, v in STYLE[k].items() if a != "label"}, label=STYLE[k]["label"])
    ax.set_xlabel("prediction step")
    ax.set_ylabel("pixel MSE")
    ax.legend(loc="upper left", handlelength=1.6)
    fig.savefig(FIG / "walker_viz.pdf")
    plt.close(fig)


if __name__ == "__main__":
    memory_length()
    rl_curves()
    probe_panels()
    for viz in (mmaze_viz, dendrites_viz, walker_viz):  # inference results of scripts/viz_infer.py, kept out of git
        if (FIG / "data" / f"{viz.__name__}.npz").exists():
            viz()
    print("figures written to", FIG)
