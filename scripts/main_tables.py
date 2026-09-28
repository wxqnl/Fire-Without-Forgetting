"""Paper main tables from the finished and running runs -> RESULTS.md (repo root).

A cell of a run that is still going shows its current value and progress, e.g. "0.52 (60%)";
"·" = not started. Single seed (s0) for now.

  Table 2  memory tasks, RL: mean of the last 3 evaluation returns (BSuite, POPGym)
  Table 3  BSuite, env steps until the evaluation return first reaches 0.9
  Table 4  world-model probe (memlen-L-recall): updates until reward AUC > 0.9, final AUC
  Table 5  Memory Maze offline probe: objects MSE and walls accuracy, second half of the episode
  Table 6  DMC (proprioception), non-memory control
  Table 7  ablations of our model (probe, memlen-32-recall and memlen-64-recall)
  Table 8  operations and energy per environment step (energy.json of the runs)

usage: python scripts/main_tables.py
"""
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MAIN = [("base_tau2", "Spiking-WM"), ("LIF_tau2", "LIF"), ("GRSN_tau2", "GRSN"),
        ("ABG_tau128", "Spiking-WM + A+B+G"), ("PD2L_tau1000", "**Ours**"), ("ann_tau2", "DreamerV3 (ANN)")]
EXTRA = [("gru_tau2", "spiking enc. + GRU"), ("PD_tau1000", "ours, 1 layer")]
BSUITE = [f"bsuite_memlen-{n}" for n in (8, 16, 32, 64)]
POPGYM = ["popgym_RepeatFirstEasy", "popgym_RepeatPreviousEasy", "popgym_CountRecallEasy",
          "popgym_RepeatPreviousMedium", "popgym_AutoencodeEasy", "popgym_RepeatPreviousHard"]
DMC = ["dmc_walker_walk", "dmc_cheetah_run", "dmc_cartpole_swingup"]
ABLATIONS = [("PD2L_tau1000", "ours"), ("PD2Leuler_tau1000", "Euler input gain 1 − λ"),
             ("PD2L_tau128", "single τ = 128"), ("PD2Lfix_tau1000", "τ fixed (not learned)"),
             ("PD2Lcap_tau1000", "firing rate capped at 0.06"), ("PD_tau1000", "1 layer"),
             ("PDT1_tau1000", "1 layer, T = 1")]


def jobs():
    out = {}
    path = RUNS / "jobs.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        job = json.loads(line)
        out[job["name"]] = job
    return out


JOBS = jobs()


def rows(run):
    path = RUNS / run / "s0" / "metrics.jsonl"
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def exit_code(run):
    path = RUNS / run / "s0" / ".exit"
    return path.read_text().strip() if path.exists() else None


def budget(run):
    job = JOBS.get(f"{run}/s0")
    if not job:
        return None
    m = re.search(r"--(steps|mm_updates|updates) (\S+)", job["cmd"])
    return float(m.group(2)) if m else None


def progress(run, step):
    """'' when finished, else ' (NN%)'."""
    if exit_code(run) == "0":
        return ""
    total = budget(run)
    return f" ({100 * step / total:.0f}%)" if total else " (…)"


def rl_cell(run, key="eval_return", fmt="{:.2f}"):
    evals = [r for r in rows(run) if key in r]
    if not evals:
        return "·" if exit_code(run) is None else f"failed ({exit_code(run)})"
    value = np.mean([r[key] for r in evals[-3:]])
    return fmt.format(value) + progress(run, evals[-1]["step"])


def steps_to(run, thresh=0.9):
    evals = [r for r in rows(run) if "eval_return" in r]
    if not evals:
        return "·"
    hit = next((r["step"] for r in evals if r["eval_return"] >= thresh), None)
    if hit is not None:
        return f"{hit / 1e3:.0f}k"
    return f"> {evals[-1]['step'] / 1e3:.0f}k" + ("" if exit_code(run) == "0" else " …")


def probe_cells(run, thresh=0.9):
    evals = [r for r in rows(run) if "eval_auc" in r]
    if not evals:
        return "·", "·"
    hit = next((r["step"] for r in evals if r["eval_auc"] > thresh), None)
    last = evals[-1]["step"]
    speed = str(hit) if hit is not None else f"> {last}" + ("" if exit_code(run) == "0" else " …")
    final = f"{np.mean([r['eval_auc'] for r in evals[-3:]]):.3f} ({evals[-1]['eval_deter_rate']:.3f})"
    return speed, final + progress(run, last)


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body]


def short(task):
    return task.replace("bsuite_memlen-", "L").replace("popgym_", "").replace("dmc_", "")


def mm_cells(run):
    path = RUNS / run / "s0"
    res = path / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        return f"{r['final_eval_objects_mse']:.2f}", f"{r['final_eval_walls_acc']:.3f}"
    evals = [r for r in rows(run) if "eval_objects_mse" in r]
    if not evals:
        return "·", "·"
    p = progress(run, evals[-1]["step"])
    return f"{evals[-1]['eval_objects_mse']:.2f}{p}", f"{evals[-1]['eval_walls_acc']:.3f}{p}"


def mm_baseline():
    for path in sorted((RUNS / "mmaze").glob("*/*/s0/metrics.jsonl")):
        first = json.loads(path.read_text().splitlines()[0])
        if "baseline_objects_mse" in first:
            return first["baseline_objects_mse"], first["baseline_walls_acc"]
    return None


def energy_cell(run):
    path = RUNS / run / "s0" / "energy.json"
    if not path.exists():
        return ["·"] * 4
    e = json.loads(path.read_text())
    return [f"{e['ac_per_step'] / 1e6:.2f}", f"{e['mac_per_step'] / 1e6:.2f}", f"{e['energy_nj_per_step'] / 1e3:.2f}",
            f"{e['deter_rate']:.3f}"]


def main():
    models = MAIN + EXTRA
    out = ["# Main results (auto-generated by `scripts/main_tables.py`)", "",
           "Single seed (s0). A value with (NN%) is from a run still in progress; · = not started.", ""]
    status = RUNS / "jobs.status"
    if status.exists():
        s = json.loads(status.read_text())
        out += [f"Queue at {s['time']}: {s['done']}/{s['total']} done, {len(s['running'])} running, "
                f"{s['pending']} pending.", ""]

    out += ["## Table 2. Memory tasks, RL: evaluation return (mean of the last 3 evaluations)", ""]
    tasks = BSUITE + POPGYM
    out += table(["model"] + [short(t) for t in tasks],
                 [[label] + [rl_cell(f"rl2/{t}/{m}") for t in tasks] for m, label in models])
    out += ["", "BSuite memory length L: return 1 = always recalls the cue, 0 = chance. POPGym: return in [−1, 1].", ""]

    out += ["## Table 3. BSuite: env steps until the evaluation return first reaches 0.9", ""]
    out += table(["model"] + [short(t) for t in BSUITE], [[label] + [steps_to(f"rl2/{t}/{m}") for t in BSUITE]
                                                          for m, label in models])

    out += ["", "## Table 4. World-model probe (memlen-L-recall, random-policy data)", "",
            "Updates until reward AUC > 0.9 / final AUC (firing rate of the recurrent state).", ""]
    probe_tasks = [f"bsuite_memlen-{n}-recall" for n in (8, 16, 32, 64)]
    body = []
    for m, label in models:
        cells = [probe_cells(f"recall/{t}/{m}") for t in probe_tasks]
        body.append([label] + [c[0] for c in cells] + [c[1] for c in cells])
    out += table(["model"] + [f"L{n} updates" for n in (8, 16, 32, 64)] + [f"L{n} AUC" for n in (8, 16, 32, 64)], body)

    base = mm_baseline()
    out += ["", "## Table 5. Memory Maze 9×9, offline probe (second half of the episode)", ""]
    if base:
        out += [f"Constant predictor: objects MSE {base[0]:.2f}, walls accuracy {base[1]:.3f}.", ""]
    body = [[label, *mm_cells(f"mmaze/memory_maze_9x9/{m}")] for m, label in models]
    out += table(["model", "objects MSE ↓", "walls acc. ↑"], body)

    out += ["", "## Table 6. DMC (proprioception), non-memory control: evaluation return", ""]
    out += table(["model"] + [short(t) for t in DMC], [[label] + [rl_cell(f"rl2/{t}/{m}", fmt="{:.0f}") for t in DMC]
                                                       for m, label in models])

    out += ["", "## Table 7. Ablations of our model (probe, memlen-L-recall)", "",
            "Updates until reward AUC > 0.9 / final AUC (firing rate).", ""]
    body = []
    for m, label in ABLATIONS:
        l32, l64 = (probe_cells(f"recall/bsuite_memlen-{n}-recall/{m}") for n in (32, 64))
        body.append([label, *l32, *l64])
    out += table(["variant", "L32 updates", "L32 AUC (rate)", "L64 updates", "L64 AUC (rate)"], body)

    out += ["", "## Table 8. Operations and energy per environment step (world-model filter: encoder + posterior)", "",
            "45 nm: 0.9 pJ per AC, 4.6 pJ per MAC; memory access not counted. Models of the BSuite L32 RL runs.", ""]
    out += table(["model", "AC (M)", "MAC (M)", "energy (µJ)", "recurrent firing rate"],
                 [[label, *energy_cell(f"rl2/bsuite_memlen-32/{m}")] for m, label in models])
    text = "\n".join(out) + "\n"
    (ROOT / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
