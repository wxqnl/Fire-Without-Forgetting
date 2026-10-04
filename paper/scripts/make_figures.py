"""Figures of the paper from the run directories (runs/<group>/<task>/<model>/s0/metrics.jsonl).

Rerun after new results: python paper/scripts/make_figures.py  ->  paper/fig/*.pdf
A curve of a run that is still going is drawn up to its latest evaluation.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS, FIG = ROOT / "runs", ROOT / "paper" / "fig"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False,
})

# Okabe-Ito colours; FwF in vermillion and drawn on top
STYLE = {
    "base_tau2": dict(label="Spiking-WM", color="#E69F00"),
    "LIF_tau2": dict(label="LIF", color="#56B4E9"),
    "GRSN_tau2": dict(label="GRSN", color="#009E73"),
    "ABG_tau128": dict(label="Spiking-WM-NR", color="#CC79A7"),
    "ann_tau2": dict(label="DreamerV3 (ANN)", color="#000000", ls="--"),
    "OURS": dict(label="FwF (ours)", color="#D55E00", lw=1.8, zorder=5),
    "R2I": dict(label="R2I (ANN)", color="#0072B2", ls="-."),
}
# FwF's runs: one configuration everywhere (EXPERIMENTS.md §19.6); the probe runs have no actor
OURS = {"bsuite": "PD2Lv2_tau1000", "probe": "PD2Lreshalf_tau1000", "popgym": "PD2Lv2_tau1000"}


def rows(path):
    path = RUNS / path / "s0" / "metrics.jsonl"
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def curve(path, key):
    r = [x for x in rows(path) if key in x]
    return np.array([x["step"] for x in r], float), np.array([x[key] for x in r], float)


def ema(y, a=0.6):
    out, m = [], None
    for v in y:
        m = v if m is None else a * m + (1 - a) * v
        out.append(m)
    return np.array(out)


def run(model, family):
    return OURS[family] if model == "OURS" else model


def final(path, key="eval_return", n=3):
    _, y = curve(path, key)
    return float(np.mean(y[-n:])) if len(y) else None


def memory_length():
    """Fig. 1: final evaluation return against the BSuite memory length; left: L8-L64 (0.15-0.5M env
    steps), right: L51-L101 under the protocol of R2I (1M env steps)."""
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    groups = [[(8, "rl2"), (16, "rl2"), (32, "rl2"), (64, "rl2")], [(51, "r2i"), (71, "r2i"), (101, "r2i")]]
    pos = {8: 0, 16: 1, 32: 2, 64: 3, 51: 4.6, 71: 5.6, 101: 6.6}
    for model in ["base_tau2", "LIF_tau2", "GRSN_tau2", "ABG_tau128", "ann_tau2", "OURS"]:
        st = dict(STYLE[model])
        label = st.pop("label")
        for g, group in enumerate(groups):
            xs, ys = [], []
            for L, sub in group:
                v = final(f"{sub}/bsuite_memlen-{L}/{run(model, 'bsuite')}")
                if v is not None:
                    xs.append(pos[L]), ys.append(v)
            if xs:
                ax.plot(xs, ys, label=label if g == 0 else None, marker="o", ms=2.8, **st)
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
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3), handlelength=1.6, columnspacing=0.9)
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
        models = ["base_tau2", "ABG_tau128", "ann_tau2", "OURS"]
        if fam == "popgym" and "Easy" in task:
            models = ["base_tau2", "LIF_tau2", "GRSN_tau2", "ABG_tau128", "ann_tau2", "OURS"]
        for model in models:
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
    fig.legend(handles=handles, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=2.0)
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.88), w_pad=0.6)
    fig.savefig(FIG / "rl_curves.pdf")
    plt.close(fig)


def probe_panels():
    """Fig. 5: (a) retention-time sweep inside FwF, memlen-64-recall probe; (b) precise-lag recall,
    RepeatPrevious Medium probe (input of exactly 32 steps ago)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.6))
    ax = axes[0]
    sweep = [("PD2Lreshalfreset_tau1000", "cleared every step", "#999999", ":"),
             ("PD2Lreshalf_tau16", r"single $\tau=16$", "#56B4E9", "-."),
             ("PD2Lreshalf_tau128", r"single $\tau=128$", "#0072B2", "--"),
             ("PD2Lreshalfeuler_tau1000", r"Euler write gain $1-\lambda$", "#009E73", (0, (3, 1, 1, 1))),
             ("PD2Lreshalf_tau1000", r"FwF ($\tau\in[1,1000]$, gain $\sqrt{1-\lambda^2}$)", "#D55E00", "-")]
    for name, label, color, ls in sweep:
        x, y = curve(f"recall/bsuite_memlen-64-recall/{name}", "eval_auc")
        if len(x):
            ax.plot(x / 1e3, y, label=label, color=color, ls=ls, lw=1.8 if name.endswith("half_tau1000") else 1.2)
    ax.set_title("(a) cue 64 steps back (memlen-64-recall)", pad=2)
    ax.set_xlabel("world-model updates (k)")
    ax.set_ylabel("reward AUC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6), handlelength=2.2)
    ax = axes[1]
    lag = [("base_tau2", None), ("LIF_tau2", None), ("GRSN_tau2", None), ("ABG_tau128", None), ("ann_tau2", None),
           ("PD2L_tau1000", dict(label="FwF, integrator dendrites only", color="#D55E00", ls=":", lw=1.4)),
           ("PD2Lres_tau1000", dict(label="FwF, all channels resonant", color="#D55E00", ls="--", lw=1.2)),
           ("PD2Lreshalf_tau1000", dict(STYLE["OURS"], label="FwF (half resonant)"))]
    for name, st in lag:
        st = dict(STYLE[name]) if st is None else st
        x, y = curve(f"diag_rp/popgym_RepeatPreviousMedium/{name}", "eval_auc")
        if len(x):
            ax.plot(x / 1e3, y, **st)
    ax.set_title("(b) card exactly 32 steps back (RepeatPrevious Medium)", pad=2)
    ax.set_xlabel("world-model updates (k)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.57), ncol=2, handlelength=2.0, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "probe_panels.pdf")
    plt.close(fig)


if __name__ == "__main__":
    memory_length()
    rl_curves()
    probe_panels()
    print("figures written to", FIG)
