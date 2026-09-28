"""Aggregates Phase 0 runs into runs/summary.md and runs/plots/*.png.

probe:  reward-prediction AUC on held-out full episodes (chance = 0.5); reported at the
        latest common update and as the best over training.
recall: the same for the memlen-L-recall grid, plus cue retention: held-out accuracy of a
        linear probe decoding the cue from the recurrent state (h: the spikes the heads
        read; mem: membrane potentials, only for A/AB) at the last row, and its profile
        over rows (plots/retention_*.png).
rl:     evaluation return, mean of the last 3 evaluations of each seed (rl2: round 3/4 RL with
        the action-indexed reward head).
xor:    the original memlen-L reward (cue XOR action) with a longer budget (round 2).

runs/learning_speed.md: updates until the reward AUC first exceeds 0.9, per seed.
"""
import collections
import json
import pathlib

import numpy as np

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"
# round 2 (2026-09-28): thr025 / thr100 = soma threshold 0.25 / 1.0 (Pr4); ABGcap = G with the
# firing rate capped at 0.06
COND_ORDER = ["base", "G", "A", "Athr025", "Athr100", "B", "AB", "ABthr025", "ABthr100", "AG", "AGthr025", "AGthr100",
              "ABG", "ABGthr025", "ABGthr100", "ABGcap", "PD", "PDeuler", "PDfix", "PD2L", "PDT1", "PDcap",
              "PD2Leuler", "PD2Lfix", "PD2Lcap", "LIF", "LIFp", "LIFpg", "GRSN", "gru", "ann"]
COND_LABEL = {"base": "base (original)", "A": "A: no inter-step reset", "B": "B: no dendritic reset",
              "AB": "A+B", "G": "G only (resets kept)", "AG": "A + dendritic gain (G)", "ABG": "A+B + dendritic gain (G)",
              "gru": "GRU core (control)", "ann": "all-ANN + GRU (control)",
              "Athr025": "A, threshold 0.25", "Athr100": "A, threshold 1.0",
              "ABthr025": "A+B, threshold 0.25", "ABthr100": "A+B, threshold 1.0",
              "ABGcap": "A+B+G, rate capped at 0.06", "AGthr025": "A+G, threshold 0.25", "AGthr100": "A+G, threshold 1.0",
              "ABGthr025": "A+B+G, threshold 0.25", "ABGthr100": "A+B+G, threshold 1.0",
              "PD": "method prototype (PDRNN)", "PDeuler": "PDRNN, Euler input gain", "PDfix": "PDRNN, fixed tau",
              "PD2L": "ours (PDRNN, 2 layers)", "PDT1": "PDRNN, T = 1", "PDcap": "PDRNN, rate capped at 0.06",
              "PD2Leuler": "ours, Euler input gain", "PD2Lfix": "ours, fixed tau", "PD2Lcap": "ours, rate capped at 0.06",
              "LIF": "recurrent LIF (usual practice)", "LIFp": "LIF + persistent membrane",
              "LIFpg": "LIF + persistent + gain", "GRSN": "GRSN (AAAI 2025)"}
COND_COLOR = dict(base="#6b7280", A="#2563eb", B="#d97706", AB="#059669", gru="#a855f7", ann="#111827", G="#9ca3af",
                  AG="#0891b2", ABG="#dc2626", ABGcap="#f472b6", Athr025="#60a5fa", Athr100="#1e3a8a",
                  ABthr025="#34d399", ABthr100="#065f46", AGthr025="#67e8f9", AGthr100="#155e75",
                  ABGthr025="#fca5a5", ABGthr100="#7f1d1d", PD="#ea580c", PDeuler="#fdba74", PDfix="#c2410c",
                  PD2L="#9a3412", PDT1="#facc15", PDcap="#fb923c", PD2Leuler="#fed7aa", PD2Lfix="#7c2d12",
                  PD2Lcap="#f97316", LIF="#16a34a", LIFp="#4ade80", LIFpg="#14532d", GRSN="#7c3aed")


def load(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def collect(kind):
    """{task: {(cond, tau): {seed: rows}}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for metrics in sorted((RUNS / kind).glob("*/*/s*/metrics.jsonl")):
        seed_dir = metrics.parent
        cond, tau = seed_dir.parent.name.rsplit("_tau", 1)
        task = seed_dir.parent.parent.name
        out[task][(cond, int(tau))][seed_dir.name] = load(metrics)
    return out


def fmt(values):
    if not values:
        return "–"
    values = np.asarray(values)
    return f"{values.mean():.3f} ± {values.std():.3f} (n={len(values)})"


def probe_table(task, cells):
    lines = [f"### probe · {task}", "",
             "| condition | τ_d | AUC @ common update | best AUC | cue in h @ end | cue in mem @ end | firing rate | updates |",
             "|---|---|---|---|---|---|---|---|"]
    finished = [max(r["step"] for r in rows) for runs in cells.values() for rows in runs.values() if rows]
    common = min(finished) if finished else 0
    for cond in COND_ORDER:
        for tau in sorted({t for c, t in cells if c == cond}):
            runs = cells[(cond, tau)]
            at, best, rate, steps, cue_h, cue_mem = [], [], [], [], [], []
            for rows in runs.values():
                evals = [r for r in rows if "eval_auc" in r]
                if not evals:
                    continue
                at += [r["eval_auc"] for r in evals if r["step"] == common]
                best.append(max(r["eval_auc"] for r in evals))
                rate.append(evals[-1]["eval_deter_rate"])
                cue_h += [evals[-1]["eval_cue_h_end"]] if "eval_cue_h_end" in evals[-1] else []
                cue_mem += [evals[-1]["eval_cue_mem_end"]] if "eval_cue_mem_end" in evals[-1] else []
                steps.append(evals[-1]["step"])
            lines.append(f"| {COND_LABEL[cond]} | {tau} | {fmt(at)} | {fmt(best)} | {fmt(cue_h)} | {fmt(cue_mem)} | "
                         f"{np.mean(rate) if rate else float('nan'):.3f} | {min(steps) if steps else 0} |")
    return lines + ["", f"(common update = {common})", ""]


def rl_table(task, cells):
    lines = [f"### RL · {task}", "",
             "| condition | τ_d | eval return (last 3) | env steps |", "|---|---|---|---|"]
    for cond in COND_ORDER:
        for tau in sorted({t for c, t in cells if c == cond}):
            finals, steps = [], []
            for rows in cells[(cond, tau)].values():
                evals = [r for r in rows if "eval_return" in r]
                if evals:
                    finals.append(np.mean([r["eval_return"] for r in evals[-3:]]))
                    steps.append(evals[-1]["step"])
            lines.append(f"| {COND_LABEL[cond]} | {tau} | {fmt(finals)} | {min(steps) if steps else 0} |")
    return lines + [""]


def plot(kind, task, cells, key, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = sorted({t for _, t in cells})
    fig, axes = plt.subplots(1, len(taus), figsize=(4.2 * len(taus), 3.4), sharey=True, squeeze=False)
    colors = COND_COLOR
    for ax, tau in zip(axes[0], taus):
        for cond in COND_ORDER:
            for i, rows in enumerate(cells.get((cond, tau), {}).values()):
                evals = [r for r in rows if key in r]
                if evals:
                    ax.plot([r["step"] for r in evals], [r[key] for r in evals], color=colors[cond],
                            alpha=0.85, lw=1.4, label=COND_LABEL[cond] if i == 0 else None)
        ax.set_title(f"{task}  τ_d={tau}", fontsize=9)
        ax.set_xlabel("env step" if kind.startswith("rl") else "update")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    if kind in ("probe", "recall", "xor"):
        for ax in axes[0]:
            ax.axhline(0.5, color="k", lw=0.8, ls=":")
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = RUNS / "plots" / f"{kind}_{task}_{key}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_retention(task, cells):
    """E1 read-out: cue decodability from the state vs. rows after the cue (latest eval)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    styles = {2: "-", 16: "--", 128: ":", 512: "-.", 1000: "-"}
    colors = COND_COLOR
    for (cond, tau), runs in sorted(cells.items(), key=lambda x: (COND_ORDER.index(x[0][0]), x[0][1])):
        profiles = [[r for r in rows if "eval_cue_h" in r][-1]["eval_cue_h"] for rows in runs.values()
                    if any("eval_cue_h" in r for r in rows)]
        if profiles:
            ax.plot(np.mean(profiles, 0), color=colors[cond], ls=styles.get(tau, "-"), lw=1.5,
                    label=f"{cond} τ{tau}")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("row (cue shown in rows 0-1; reward at the last row)")
    ax.set_ylabel("cue decodability from h (held-out acc.)")
    ax.set_title(task, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = RUNS / "plots" / f"retention_{task}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def learning_speed(kind, thresh=0.9):
    """Rows: cells; columns: tasks. Per seed, the first update with reward AUC > thresh."""
    by_task = collect(kind)
    tasks = sorted(by_task, key=lambda t: int(t.split("-")[1]))
    keys = sorted({c for cells in by_task.values() for c in cells}, key=lambda c: (COND_ORDER.index(c[0]), c[1]))
    lines = [f"## {kind}: updates until reward AUC > {thresh} (per seed; '–' = not within the run)", "",
             "| condition | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for cond, tau in keys:
        row = []
        for task in tasks:
            runs = by_task[task].get((cond, tau), {})
            hits = []
            for seed in sorted(runs, key=lambda x: int(x[1:])):
                step = next((r["step"] for r in runs[seed] if r.get("eval_auc", 0) > thresh), None)
                last = max((r["step"] for r in runs[seed]), default=0)
                hits.append(str(step) if step is not None else f"–@{last}")
            rates = [r["eval_deter_rate"] for rows in runs.values() for r in rows[-1:] if "eval_deter_rate" in r]
            rate = f" ({np.mean(rates):.3f})" if rates and cond not in ("gru", "ann") else ""
            row.append(" / ".join(hits) + rate if hits else "")
        lines.append(f"| {cond} τ{tau} | " + " | ".join(row) + " |")
    return lines + [""]


def main():
    lines = ["# Phase 0 summary (auto-generated by scripts/summarize.py)", ""]
    status = RUNS / "jobs.status"
    if status.exists():
        s = json.loads(status.read_text())
        lines += [f"queue: {s['done']}/{s['total']} done, {len(s['running'])} running ({s['time']})", ""]
    for task, cells in sorted(collect("probe").items()):
        lines += probe_table(task, cells)
        plot("probe", task, cells, "eval_auc", "reward-prediction AUC")
    for task, cells in sorted(collect("recall").items()):
        lines += probe_table(task, cells)
        plot("recall", task, cells, "eval_auc", "reward-prediction AUC")
        plot("recall", task, cells, "eval_cue_h_end", "cue decodability from h, last row")
        plot_retention(task, cells)
    for task, cells in sorted(collect("rl").items()):
        lines += rl_table(task, cells)
        plot("rl", task, cells, "eval_return", "eval return")
    for task, cells in sorted(collect("rl2").items()):  # round 3/4: action-indexed reward head
        lines += rl_table(task, cells)
        plot("rl2", task, cells, "eval_return", "eval return")
    for task, cells in sorted(collect("xor").items()):
        lines += probe_table(task, cells)
        plot("xor", task, cells, "eval_auc", "reward-prediction AUC (cue XOR action)")
    speed = learning_speed("recall") + (learning_speed("xor") if collect("xor") else [])
    (RUNS / "learning_speed.md").write_text("\n".join(speed))
    (RUNS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
