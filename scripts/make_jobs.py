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
TAUS = {2: [], 16: ["taud16"], 128: ["taud128"]}


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


def main():
    # 2026-09-27 11:15: GPUs 1-6 are available for 8 more hours (hard stop 18:50).
    # Depth over breadth: keep the running memlen-16 grid, GRU controls and partial
    # RepeatPrevious grid; keep a minimal RL contrast; add short probes that can finish.
    core = {"base_tau2", "A_tau2", "B_tau2", "AB_tau2", "AB_tau128", "base_tau128"}
    jobs = []
    jobs += probe_jobs("bsuite_memlen-16")
    for task in ("bsuite_memlen-16", "popgym_RepeatPreviousEasy", "bsuite_memlen-8", "bsuite_memlen-32"):
        jobs += probe_jobs(task, taus=(2,), conds=CTRL)
    jobs += pick(rl_jobs("popgym_RepeatPreviousEasy", steps=5e5), {"base_tau2", "AB_tau128"})
    jobs += pick(probe_jobs("popgym_RepeatPreviousEasy"), {"base_tau2", "base_tau16", "base_tau128", "A_tau2"})
    jobs += with_updates(pick(probe_jobs("popgym_RepeatPreviousEasy"), {"B_tau2", "AB_tau2", "AB_tau128"}), 15000)
    jobs += with_updates(pick(probe_jobs("bsuite_memlen-8"), core), 15000)
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
