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
TAUS = {2: [], 16: ["taud16"], 128: ["taud128"]}


def suite(task):
    return task.split("_", 1)[0]


def probe_jobs(task, seeds=(0, 1), taus=(2, 16, 128)):
    for cond, cfgs in CONDS.items():
        for tau in taus:
            for seed in seeds:
                name = f"probe/{task}/{cond}_tau{tau}/s{seed}"
                logdir = RUNS / name
                configs = " ".join([suite(task), *cfgs, *TAUS[tau]])
                cmd = (f"{{python}} probe_wm.py --configs {configs} --task {task} --seed {seed} "
                       f"--updates 20000 --eval_every 1000 --logdir {logdir}")
                yield dict(name=name, cmd=cmd, logdir=str(logdir), mem_gb=3)


def rl_jobs(task, steps, seeds=(0, 1, 2), taus=(2, 128), train_ratio=64, eval_every=None):
    for cond, cfgs in CONDS.items():
        for tau in taus:
            for seed in seeds:
                name = f"rl/{task}/{cond}_tau{tau}/s{seed}"
                logdir = RUNS / name
                configs = " ".join([suite(task), *cfgs, *TAUS[tau]])
                cmd = (f"{{python}} dreamer.py --configs {configs} --task {task} --seed {seed} "
                       f"--steps {steps:g} --train_ratio {train_ratio} --logdir {logdir}")
                if eval_every:
                    cmd += f" --eval_every {eval_every:g}"
                yield dict(name=name, cmd=cmd, logdir=str(logdir), mem_gb=14)


def main():
    jobs = []
    # longest jobs (RL on POPGym) start early so they do not become the tail
    jobs += probe_jobs("bsuite_memlen-16")
    jobs += rl_jobs("popgym_RepeatPreviousEasy", steps=5e5)
    jobs += probe_jobs("popgym_RepeatPreviousEasy")
    jobs += rl_jobs("bsuite_memlen-16", steps=2e5)
    jobs += probe_jobs("bsuite_memlen-8")
    jobs += probe_jobs("bsuite_memlen-32")
    RUNS.mkdir(exist_ok=True)
    out = RUNS / "jobs.jsonl"
    with open(out, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    print(f"{len(jobs)} jobs -> {out}")


if __name__ == "__main__":
    main()
