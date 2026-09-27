"""Writes the Phase 0 job list (E2 causal intervention) in priority order.

Conditions (2x2): base = original Spiking-WM; A = persist_state (no reset of the
recurrent cell between env steps); B = no dendritic reset on somatic spikes; AB = both.
Each cell is crossed with the dendritic time constant tau_d in {2, 16, 128}.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

CONDS = {"base": [], "A": ["persist"], "B": ["nodendreset"], "AB": ["persist", "nodendreset"]}
# positive control: continuous GRU core (only used with tau 2, which it ignores)
CTRL = {"gru": ["grucell"]}
TAUS = {2: [], 16: ["taud16"], 128: ["taud128"], 512: ["taud512"]}


def suite(task):
    return task.split("_", 1)[0]


def probe_jobs(task, seeds=(0, 1), taus=(2, 16, 128), conds=CONDS):
    for cond, cfgs in conds.items():
        for tau in taus:
            for seed in seeds:
                name = f"probe/{task}/{cond}_tau{tau}/s{seed}"
                logdir = RUNS / name
                configs = " ".join([suite(task), *cfgs, *TAUS[tau]])
                cmd = (f"{{python}} probe_wm.py --configs {configs} --task {task} --seed {seed} "
                       f"--updates 20000 --eval_every 1000 --logdir {logdir}")
                yield dict(name=name, cmd=cmd, logdir=str(logdir), mem_gb=2.5)


def rl_jobs(task, steps, seeds=(0, 1, 2), taus=(2, 128), train_ratio=64, eval_every=None,
            conds=CONDS):
    for cond, cfgs in conds.items():
        for tau in taus:
            for seed in seeds:
                name = f"rl/{task}/{cond}_tau{tau}/s{seed}"
                logdir = RUNS / name
                configs = " ".join([suite(task), *cfgs, *TAUS[tau]])
                cmd = (f"{{python}} dreamer.py --configs {configs} --task {task} --seed {seed} "
                       f"--steps {steps:g} --train_ratio {train_ratio} --logdir {logdir}")
                if eval_every:
                    cmd += f" --eval_every {eval_every:g}"
                yield dict(name=name, cmd=cmd, logdir=str(logdir), mem_gb=14.5)


def pick(jobs, keep):
    """keep: set of "cond_tau" labels, e.g. {"base_tau2", "AB_tau128"}"""
    return [j for j in jobs if j["name"].split("/")[2] in keep]


def with_updates(jobs, updates):
    for j in jobs:
        j["cmd"] = j["cmd"].replace("--updates 20000", f"--updates {updates}")
    return jobs


# Ablation ladder (2026-09-27 12:00): even the GRU control stays at chance, although a
# plain GRU learns memlen-8 from the same batches in 200 updates. Short runs localise the
# failure: learning rate, reward-loss weight, spiking feed-forward layers (annwm).
LADDER = {
    "gru": (["grucell"], ""),
    "gru-lr3e-4": (["grucell"], "--model_lr 3e-4"),
    "gru-rs10": (["grucell"], "--reward_scale 10"),
    "ann": (["grucell", "annwm"], ""),
    "ann-lr3e-4": (["grucell", "annwm"], "--model_lr 3e-4"),
    "base": ([], ""),
    "AB128": (["persist", "nodendreset", "taud128"], ""),
}
# 12:25: every model is still above the no-memory floor of the reward loss after
# thousands of updates: the zero-initialised 255-bucket two-hot reward head learns even
# the memory-free part slowly. An MSE (symlog) reward head starts at the right answer for
# the zero-reward steps.
LADDER_MSE = {
    "gru-mse": (["grucell"], "--reward_head symlog_mse"),
    "gru-mse-lr3e-4": (["grucell"], "--reward_head symlog_mse --model_lr 3e-4"),
    "gru-lr1e-3": (["grucell"], "--model_lr 1e-3"),
    "ann-mse": (["grucell", "annwm"], "--reward_head symlog_mse"),
    "base-mse": ([], "--reward_head symlog_mse"),
    "AB128-mse": (["persist", "nodendreset", "taud128"], "--reward_head symlog_mse"),
    "ann-mse-lr1e-3": (["grucell", "annwm"], "--reward_head symlog_mse --model_lr 1e-3"),
    "ann-mse-rs10": (["grucell", "annwm"], "--reward_head symlog_mse --reward_scale 10"),
    "gru-mse-rs10": (["grucell"], "--reward_head symlog_mse --reward_scale 10"),
}


def ladder_jobs(task="bsuite_memlen-4", variants=LADDER, updates=4000, extra="--batch_length 32 --train_steps 100000"):
    for variant, (cfgs, flags) in variants.items():
        name = f"ladder/{task}/{variant}/s0"
        logdir = RUNS / name
        cmd = (f"{{python}} probe_wm.py --configs {' '.join([suite(task), *cfgs])} --task {task} "
               f"--seed 0 --updates {updates} --eval_every 250 {extra} {flags} --logdir {logdir}")
        yield dict(name=name, cmd=" ".join(cmd.split()), logdir=str(logdir), mem_gb=2.5)


def recall_jobs(task, cells, seeds=(0, 1), updates=12000, extra=""):
    """Main Phase 0 grid after 12:45 (see PHASE0.md 2.7): memlen-L-recall, MSE reward head,
    cue-retention probe. cells: {"cond_tauX": config blocks}."""
    for cell, cfgs in cells.items():
        for seed in seeds:
            name = f"recall/{task}/{cell}/s{seed}"
            logdir = RUNS / name
            cmd = (f"{{python}} probe_wm.py --configs {' '.join([suite(task), *cfgs])} --task {task} "
                   f"--seed {seed} --updates {updates} --eval_every 500 --reward_head symlog_mse "
                   f"{extra} --logdir {logdir}")
            yield dict(name=name, cmd=" ".join(cmd.split()), logdir=str(logdir), mem_gb=2.5)


def grid_cells():
    cells = {f"{c}_tau{t}": cfgs + TAUS[t] for c, cfgs in CONDS.items() for t in (2, 128)}
    cells.update({"base_tau16": TAUS[16], "AB_tau16": CONDS["AB"] + TAUS[16]})
    cells.update({"gru_tau2": ["grucell"], "ann_tau2": ["grucell", "annwm"]})
    return cells


def main():
    # 2026-09-27 11:15: GPUs 1-6 are available for 8 more hours (hard stop 18:50).
    # 12:45: the memlen-L reward is an XOR of cue and action. Trained world models keep the
    # cue in their state (linear probe 0.97-1.00) but their reward head does not learn the
    # XOR, so the XOR probe grid is confounded and was stopped (runs/_xor_probe). New grid:
    # the -recall variant (reward = cue), MSE reward head, cue-retention probe.
    jobs = []
    # quick check that the reward head learns the recall task
    jobs += ladder_jobs("bsuite_memlen-4-recall", {k: LADDER_MSE[k] for k in ("ann-mse", "gru-mse", "base-mse", "AB128-mse")},
                        updates=3000)
    # 13:25: B alone cannot help while A's reset wipes the state (memlen-8 B runs all at
    # chance); its memlen-16 runs make room for memlen-32 and the G-only control
    jobs += recall_jobs("bsuite_memlen-16-recall", {k: v for k, v in grid_cells().items() if not k.startswith("B_")})
    jobs += recall_jobs("bsuite_memlen-8-recall", grid_cells(), extra="--batch_length 32")
    # 13:10: with A, tau 128 keeps the cue in the dendrites at ~1% of threshold (write gain
    # 1/tau) and it never reaches the spikes; intervention G decouples the gain
    G = {"ABG_tau128": ["persist", "nodendreset", "taud128", "dendgain"],
         "ABG_tau16": ["persist", "nodendreset", "taud16", "dendgain"],
         "AG_tau128": ["persist", "taud128", "dendgain"]}
    jobs += recall_jobs("bsuite_memlen-8-recall", G, extra="--batch_length 32")
    jobs += recall_jobs("bsuite_memlen-16-recall", {k: G[k] for k in ("ABG_tau128", "ABG_tau16")})
    # is G alone enough? (inter-step reset kept)
    jobs += recall_jobs("bsuite_memlen-16-recall", {"G_tau128": ["taud128", "dendgain"]})
    jobs += recall_jobs("bsuite_memlen-32-recall", {
        "ABG_tau128": G["ABG_tau128"], "ABG_tau512": ["persist", "nodendreset", "taud512", "dendgain"],
        "AB_tau128": CONDS["AB"] + TAUS[128], "base_tau2": [], "gru_tau2": ["grucell"]})
    # 14:30 stretch point: an episode has 66 rows, so the training window must be longer
    jobs += recall_jobs("bsuite_memlen-64-recall", {
        "ABG_tau512": ["persist", "nodendreset", "taud512", "dendgain"], "gru_tau2": ["grucell"]},
        extra="--batch_length 80")
    RUNS.mkdir(exist_ok=True)
    out = RUNS / "jobs.jsonl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    tmp.replace(out)  # atomic: the running queue re-reads this file
    print(f"{len(jobs)} jobs -> {out}")


if __name__ == "__main__":
    main()
