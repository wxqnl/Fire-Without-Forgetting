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


if __name__ == "__main__":
    memory_length()
    rl_curves()
    probe_panels()
    print("figures written to", FIG)
