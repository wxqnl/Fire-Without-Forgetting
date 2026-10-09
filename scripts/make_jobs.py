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


def recall_jobs(task, cells, seeds=(0, 1), updates=12000, extra="", group="recall", eval_every=500):
    """Main Phase 0 grid after 12:45 (see PHASE0.md 2.7): memlen-L-recall, MSE reward head,
    cue-retention probe. cells: {"cond_tauX": config blocks}."""
    for cell, cfgs in cells.items():
        for seed in seeds:
            name = f"{group}/{task}/{cell}/s{seed}"
            logdir = RUNS / name
            cmd = (f"{{python}} probe_wm.py --configs {' '.join([suite(task), *cfgs])} --task {task} "
                   f"--seed {seed} --updates {updates} --eval_every {eval_every} --reward_head symlog_mse "
                   f"{extra} --logdir {logdir}")
            yield dict(name=name, cmd=" ".join(cmd.split()), logdir=str(logdir), mem_gb=2.5)


def grid_cells():
    cells = {f"{c}_tau{t}": cfgs + TAUS[t] for c, cfgs in CONDS.items() for t in (2, 128)}
    cells.update({"base_tau16": TAUS[16], "AB_tau16": CONDS["AB"] + TAUS[16]})
    cells.update({"gru_tau2": ["grucell"], "ann_tau2": ["grucell", "annwm"]})
    return cells


def round1():
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
    return jobs


def round2():
    """2026-09-28 (PHASE0.md §5): 5 seeds for the headline cells at memlen-16/32; G with the
    firing rate capped at A+B tau128's (0.06); the threshold sweep of Pr4; and whether the world
    model learns the original XOR reward (cue x action) with a longer budget, which decides the
    tasks for the RL half of E2."""
    AB128 = CONDS["AB"] + TAUS[128]
    core = {"base_tau2": [], "A_tau128": CONDS["A"] + TAUS[128], "AB_tau128": AB128,
            "ABG_tau128": AB128 + ["dendgain"], "gru_tau2": ["grucell"]}
    # longest jobs first
    jobs = list(recall_jobs("bsuite_memlen-8", {"ann_tau2": ["grucell", "annwm"], "gru_tau2": ["grucell"],
                                                "ABG_tau128": core["ABG_tau128"]},
                            updates=20000, eval_every=1000, extra="--batch_length 32", group="xor"))
    for L in (16, 32):
        jobs += recall_jobs(f"bsuite_memlen-{L}-recall", core, seeds=(2, 3, 4))
    jobs += recall_jobs("bsuite_memlen-32-recall", {"A_tau128": core["A_tau128"]})
    for L in (16, 32):
        jobs += recall_jobs(f"bsuite_memlen-{L}-recall", {"ABGcap_tau128": core["ABG_tau128"] + ["ratecap"]})
    # Pr4: a lower threshold means more somatic spikes; with the dendritic reset (A) that should
    # shorten the memory, without it (A+B) it should not matter
    jobs += recall_jobs("bsuite_memlen-16-recall", {f"{c}thr{t}_tau128": CONDS[c] + TAUS[128] + [f"thr{t}"]
                                                    for c in ("A", "AB") for t in ("025", "100")})
    return jobs


def rl2_jobs(task, cells, steps=2e5, seed=0, train_ratio=64, extra="", eval_every=1e4):
    """RL with the action-indexed reward head (configs block rewardbyact, MSE reward head)."""
    for cell, cfgs in cells.items():
        name = f"rl2/{task}/{cell}/s{seed}"
        logdir = RUNS / name
        cmd = (f"{{python}} dreamer.py --configs {' '.join([suite(task), *cfgs, 'rewardbyact'])} --task {task} "
               f"--seed {seed} --steps {steps:g} --train_ratio {train_ratio} --reward_head symlog_mse "
               f"--eval_every {eval_every:g} {extra} --logdir {logdir}")
        yield dict(name=name, cmd=" ".join(cmd.split()), logdir=str(logdir), mem_gb=14.5)


def round3():
    """2026-09-28, GPUs 2-5, one seed (s0): close the open links of the idea instead of adding
    seeds (PHASE0.md §6). Mechanism: Pr4 threshold sweep, G at a matched firing rate, A tau128
    at memlen-32. RL: the action-indexed reward head turns the (cue x action) reward into a
    linear read-out; check it in the probe, then run E2 as RL on the standard memlen task."""
    A128, AB128 = CONDS["A"] + TAUS[128], CONDS["AB"] + TAUS[128]
    cells = {"base_tau2": [], "A_tau128": A128, "AB_tau128": AB128, "AG_tau128": A128 + ["dendgain"],
             "ABG_tau128": AB128 + ["dendgain"], "gru_tau2": ["grucell"]}
    jobs = list(rl2_jobs("bsuite_memlen-16", {k: cells[k] for k in ("base_tau2", "AB_tau128", "ABG_tau128", "gru_tau2")}))
    # the probe on the original XOR reward, with the action-indexed head
    jobs += recall_jobs("bsuite_memlen-8", {**{k: cells[k] + ["rewardbyact"] for k in ("base_tau2", "AB_tau128", "ABG_tau128", "gru_tau2")},
                                            "ann_tau2": ["grucell", "annwm", "rewardbyact"]},
                        seeds=(0,), updates=6000, extra="--batch_length 32", group="xor")
    jobs += recall_jobs("bsuite_memlen-32-recall", {"A_tau128": A128, "ABGcap_tau128": cells["ABG_tau128"] + ["ratecap"]}, seeds=(0,))
    jobs += recall_jobs("bsuite_memlen-16-recall", {"ABGcap_tau128": cells["ABG_tau128"] + ["ratecap"],
                                                    "AG_tau128": cells["AG_tau128"]}, seeds=(0,))
    # Pr4: a lower threshold means more somatic spikes; with the dendritic reset (A, AG) that should
    # shorten the memory, without it (AB, ABG) it should not matter
    jobs += recall_jobs("bsuite_memlen-16-recall", {f"{c}thr{t}_tau128": cells[f"{c}_tau128"] + [f"thr{t}"]
                                                    for c in ("A", "AB", "AG", "ABG") for t in ("025", "100")}, seeds=(0,))
    return jobs


def round4():
    """2026-09-28 04:15, one seed. (1) The method prototype (networks.PDRNN, research plan §4.2)
    on the recall probe, with ablations at memlen-32: Euler input gain instead of the LRU
    normalisation, a single tau of 128, fixed tau, two layers, T = 1, rate cap. (2) RL on the
    standard memlen task (cue x action reward) at 16 and 32. (3) Pr5: a non-memory control task,
    DMC walker-walk from proprioception."""
    PD = ["pdcell"]
    jobs = list(rl2_jobs("bsuite_memlen-16", {"PD_tau1000": PD}))
    rl_cells = {"base_tau2": [], "ABG_tau128": ["persist", "nodendreset", "taud128", "dendgain"],
                "PD_tau1000": PD, "gru_tau2": ["grucell"]}
    jobs += rl2_jobs("bsuite_memlen-32", rl_cells, steps=3e5)
    for cell, cfgs in rl_cells.items():  # continuous actions: no rewardbyact, original reward head
        name = f"rl2/dmc_walker_walk/{cell}/s0"
        cmd = (f"{{python}} dreamer.py --configs dmc_proprio {' '.join(cfgs)} --task dmc_walker_walk --seed 0 "
               f"--steps 2e5 --train_ratio 64 --eval_every 2e4 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=14.5))
    jobs += recall_jobs("bsuite_memlen-16-recall", {"PD_tau1000": PD}, seeds=(0,))
    jobs += recall_jobs("bsuite_memlen-32-recall", {
        "PD_tau1000": PD, "PDeuler_tau1000": PD + ["pdeuler"], "PD_tau128": PD + ["pdtau128"],
        "PDfix_tau1000": PD + ["pdfixtau"], "PD2L_tau1000": PD + ["pd2layer"],
        "PDcap_tau1000": PD + ["ratecap"]}, seeds=(0,))
    jobs += recall_jobs("bsuite_memlen-32-recall", {"PDT1_tau1000": PD}, seeds=(0,), extra="--spike_times 1")
    jobs += recall_jobs("bsuite_memlen-64-recall", {"PD_tau1000": PD}, seeds=(0,), extra="--batch_length 80")
    return jobs


# main comparison (EXPERIMENTS.md): the same world model with different recurrent cores
MAIN = {
    "base_tau2": [],                                                   # Spiking-WM (MCN), original
    "LIF_tau2": ["lifcell"],                                           # recurrent LIF, usual practice
    "GRSN_tau2": ["grsn"],                                             # GRSN (AAAI 2025), state kept
    "ABG_tau128": ["persist", "nodendreset", "taud128", "dendgain"],   # Spiking-WM + A+B+G
    "PD2L_tau1000": ["ours"],                                          # proposed model
    "ann_tau2": ["grucell", "annwm"],                                  # DreamerV3-style ANN
}
MEMLEN_EXTRA = {8: "--batch_length 32", 16: "", 32: "", 64: "--batch_length 80"}


def dmc_jobs(task, cells, steps=2e5, seed=0, train_ratio=64, group="rl2"):
    """DMC from proprioception, continuous actions: original reward head, no rewardbyact."""
    for cell, cfgs in cells.items():
        name = f"{group}/{task}/{cell}/s{seed}"
        cmd = (f"{{python}} dreamer.py --configs dmc_proprio {' '.join(cfgs)} --task {task} --seed {seed} "
               f"--steps {steps:g} --train_ratio {train_ratio} --eval_every 2e4 --logdir {RUNS / name}")
        yield dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=14.5)


def round5():
    """2026-09-28 05:30, one seed: the main experiments of the paper (EXPERIMENTS.md). Runs that
    already exist with the same settings (rl2 memlen-16/32 for base and A+B+G, walker for base and
    A+B+G, recall probes of round 1-4) are reused and not repeated."""
    M = MAIN
    jobs = []
    # 1. world-model memory probe (Fig. mechanism / retention; ablations of the proposed model)
    for L in (8, 16, 32, 64):
        cells = {"LIF_tau2": M["LIF_tau2"], "LIFp_tau2": M["LIF_tau2"] + ["persist"], "GRSN_tau2": M["GRSN_tau2"]}
        if L != 32:
            cells["PD2L_tau1000"] = M["PD2L_tau1000"]
        if L in (32, 64):
            cells["ann_tau2"] = M["ann_tau2"]
        if L == 64:
            cells.update(base_tau2=[], ABG_tau128=M["ABG_tau128"])
        if L in (16, 32):  # A+G for LIF: persistent membrane, long tau, input gain
            cells["LIFpg_tau128"] = M["LIF_tau2"] + ["persist", "liftau128", "lifgain"]
        jobs += recall_jobs(f"bsuite_memlen-{L}-recall", cells, seeds=(0,), extra=MEMLEN_EXTRA[L])
    jobs += recall_jobs("bsuite_memlen-32-recall", {
        "PD2Leuler_tau1000": ["ours", "pdeuler"], "PD2L_tau128": ["ours", "pdtau128"],
        "PD2Lfix_tau1000": ["ours", "pdfixtau"], "PD2Lcap_tau1000": ["ours", "ratecap"]}, seeds=(0,))
    # 2. RL on BSuite memory length (memory-length curve)
    have = {16: ("base_tau2", "ABG_tau128"), 32: ("base_tau2", "ABG_tau128")}
    for L, steps in ((8, 1.5e5), (16, 2e5), (32, 3e5), (64, 5e5)):
        cells = {k: v for k, v in M.items() if k not in have.get(L, ())}
        jobs += rl2_jobs(f"bsuite_memlen-{L}", cells, steps=steps, extra=MEMLEN_EXTRA[L])
    # 3. RL on POPGym; episodes of 51 steps fit the default window, the longer tasks get 128
    for task, steps, extra in (("RepeatFirstEasy", 3e5, ""), ("RepeatPreviousEasy", 3e5, ""),
                               ("CountRecallEasy", 3e5, ""), ("RepeatPreviousMedium", 5e5, "--batch_length 128"),
                               ("AutoencodeEasy", 5e5, "--batch_length 128"),
                               ("RepeatPreviousHard", 5e5, "--batch_length 128")):
        jobs += rl2_jobs(f"popgym_{task}", M, steps=steps, extra=extra, eval_every=2e4)
    # 4. non-memory control (Pr5): DMC from proprioception
    jobs += dmc_jobs("dmc_walker_walk", {k: v for k, v in M.items() if k not in ("base_tau2", "ABG_tau128")})
    for task in ("dmc_cheetah_run", "dmc_cartpole_swingup"):
        jobs += dmc_jobs(task, M)
    return jobs


def round6():
    """Memory Maze offline probe (probe_mm.py) for the six models of MAIN, one seed. Listed only
    once the data cache exists (gen_memory_maze.py, then probe_mm.py --prepare)."""
    data = ROOT / "data" / "memory_maze_9x9"
    if not all((data / f"{split}_cache" / "done").exists() for split in ("train", "eval")):
        return []
    jobs = []
    for cell, cfgs in MAIN.items():
        name = f"mmaze/memory_maze_9x9/{cell}/s0"
        cmd = (f"{{python}} probe_mm.py --configs mmprobe {' '.join(cfgs)} --data {data} --seed 0 "
               f"--mm_updates 20000 --mm_eval_every 2000 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=20))
    return jobs


def round7():
    """2026-09-28 09:30: the ablations of the proposed model again at L = 64. At L = 32 all 2-layer
    variants learn within 1000-2500 updates, so L = 32 does not separate them; at L = 64 only the
    proposed model learns (4000 updates; 1 layer, GRU core, ANN and A+B+G tau 512 do not)."""
    return list(recall_jobs("bsuite_memlen-64-recall", {
        "PD2Leuler_tau1000": ["ours", "pdeuler"], "PD2L_tau128": ["ours", "pdtau128"],
        "PD2Lfix_tau1000": ["ours", "pdfixtau"], "PD2Lcap_tau1000": ["ours", "ratecap"]},
        seeds=(0,), extra=MEMLEN_EXTRA[64]))


def round9():
    """2026-09-29: two controls at L = 64 that isolate the retention time inside the proposed model.
    PD2Lreset: the same model with its dendrites cleared at every environment step (the reset of the
    spiking world model, nothing else changed). PD2L_tau16: a single retention time shorter than the
    memory; with tau 128 (round 7) and the log-uniform spectrum it shows the memory reaches as far as
    the retention time, which the spectrum covers without tuning tau to the task."""
    return list(recall_jobs("bsuite_memlen-64-recall", {
        "PD2Lreset_tau1000": ["ours", "pdreset"], "PD2L_tau16": ["ours", "pdtau16"]},
        seeds=(0,), extra=MEMLEN_EXTRA[64]))


def round10():
    """2026-09-29 07:00: pipeline check for Memory Maze, not a paper result. Google Drive serves only
    a few MB of the official training archives (download quota), so the main runs wait for the full
    data (round 8). Meanwhile ours and DreamerV3 run 5k updates on the 1471 complete episodes at the
    head of the partial archives (scripts/salvage_memory_maze.py), with the official evaluation set
    and probe: checks the whole probe_mm.py path and measures the update speed for the budget of the
    full runs."""
    data = ROOT / "data" / "memory_maze_9x9_official_subset"
    if not (data / "train").is_dir() or (ROOT / "data" / "memory_maze_9x9_official" / "train_complete").exists():
        return []  # not needed once the full data is there
    jobs = []
    for cell in ("PD2L_tau1000", "ann_tau2"):
        name = f"mmaze_pilot/memory_maze_9x9/{cell}/s0"
        cmd = (f"{{python}} probe_mm.py --configs mmprobe {' '.join(MAIN[cell])} --data {data} --seed 0 "
               f"--mm_updates 5000 --mm_eval_every 2500 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=20))
    return jobs


def round11():
    """2026-09-29: the non-memory control of the paper (claim C4), replacing the DMC runs of round 5.
    At train ratio 64 and 200k steps Spiking-WM and A+B+G reached a return of only 110-130 on walker,
    too little training to show whether a model loses anything; now DreamerV3's train ratio for DMC
    from proprioception (512). Three models: the model we modify, ours, and the ANN world model.
    2026-09-30 (configuration review, EXPERIMENTS.md §14): DreamerV3's budget for DMC from
    proprioception, 500k env steps, and a different seed per parallel env; the runs at 200k steps
    (group dmc) had one seed for all 4 envs, so the deterministic evaluation played the same
    episode in each of them. Group dmc500."""
    cells = {k: MAIN[k] for k in ("base_tau2", "PD2L_tau1000", "ann_tau2")}
    jobs = []
    for task in ("dmc_walker_walk", "dmc_cheetah_run", "dmc_cartpole_swingup"):
        for job in dmc_jobs(task, cells, steps=5e5, train_ratio=512, group="dmc500"):
            job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
            jobs.append(job)
    return jobs


def round12():
    """2026-09-30: why ours does not learn POPGym RepeatPreviousEasy while DreamerV3 does. At 1M
    steps its reward head picks the suit of o_{t-4} at 37% (DreamerV3 75%) and its state holds the
    cards 1-4 steps back at 72/56/44/39% (DreamerV3 91/87/84/81%; scripts/diag_popgym.py). World
    model only, on random-action data, 10k updates: with unit rewards (x48) it shows whether the
    model can learn the 4-step memory at all; with the task's own +-1/48 rewards, as in the RL
    runs, whether the weak reward signal is what holds it back."""
    task = "popgym_RepeatPreviousEasy"
    cells = {k: MAIN[k] + ["rewardbyact"] for k in ("PD2L_tau1000", "ann_tau2", "base_tau2")}
    jobs = list(recall_jobs(task, cells, seeds=(0,), updates=10000, extra="--batch_length 64",
                            group="diag_rp", eval_every=1000))
    raw = {k: cells[k] for k in ("PD2L_tau1000", "ann_tau2")}
    jobs += list(recall_jobs(task, raw, seeds=(0,), updates=10000, extra="--batch_length 64 --raw_reward",
                             group="diag_rp_raw", eval_every=1000))
    # the same with unit rewards on Medium (the card 32 steps back), window 64 as in the RL runs:
    # whether a rerun with normalised rewards can separate the models there
    jobs += list(recall_jobs("popgym_RepeatPreviousMedium", raw, seeds=(0,), updates=10000,
                             extra="--batch_length 64", group="diag_rp", eval_every=1000))
    return jobs


def round13():
    """2026-09-30: POPGym RepeatPrevious as in R2I (configs.yaml of Recall2Imagine, block popgym):
    DreamerV3's default reward head (symlog two-hot, 255 bins, zero-initialised, reading [h, z]), no
    reward scaling, 8 parallel envs, train ratio 64, 2M env steps. Replaces the POPGym runs of round 8,
    which used the action-indexed MSE reward head of the BSuite runs: with POPGym's per-step rewards
    of +-1/(N-k) its loss is (N-k)^2 smaller than for unit rewards, and neither DreamerV3 nor ours
    learned the card memory from it within the probe budget (round 12). BSuite rewards are +-1 and
    keep that head. Order: ours and DreamerV3 first."""
    order = [("popgym_RepeatPreviousEasy", c) for c in ("PD2L_tau1000", "ann_tau2")]
    order += [("popgym_RepeatPreviousMedium", c) for c in ("PD2L_tau1000", "ann_tau2")]
    order += [("popgym_RepeatPreviousEasy", c) for c in ("base_tau2", "ABG_tau128", "LIF_tau2", "GRSN_tau2")]
    order += [("popgym_RepeatPreviousMedium", c) for c in ("base_tau2", "ABG_tau128")]
    jobs = []
    for task, cell in order:
        name = f"popgym/{task}/{cell}/s0"
        logdir = RUNS / name
        cmd = (f"{{python}} dreamer.py --configs popgym {' '.join(MAIN[cell])} --task {task} --seed 0 "
               f"--steps 2e6 --train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True "
               f"--eval_every 5e4 --logdir {logdir}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(logdir), mem_gb=14.5))
    return jobs


MM_UPDATES = 1000000  # Memory Maze gradient steps, as in the benchmark paper (200k until 2026-10-01; runs extended)


def round8():
    """2026-09-28 21:00, after the GPU reclaim: memory benchmarks of the world-model literature with
    published ANN results, run with their protocols (EXPERIMENTS.md §10). All with a different seed
    per parallel env (env_seed_per_env).

    1. Memory Maze offline probing (Pasukonis et al. 2023) on the official 9x9 dataset, replacing
       round 6 (own data, probe without the agent pose, 20k updates: no model left the no-memory
       level). Reference: RSSM / RSSM-TBTT / VAE+GRU at 1M steps.
    2. BSuite Memory Length as in R2I (Samsami et al. 2024, Fig. 3/7): 1M env steps per memory
       length; DreamerV3 fails beyond ~30 steps, R2I reaches ~70-100. The window covers the episode.
    3. POPGym RepeatPrevious as in R2I (Fig. 4/9): R2I learns Easy and Medium within the first
       1-2M steps while DreamerV3 stays near -0.5 on Medium until 20M; 2M steps each."""
    jobs = []
    data = ROOT / "data" / "memory_maze_9x9_official"
    for cell, cfgs in MAIN.items() if (data / "train_complete").exists() else ():  # set by download_all.sh
        name = f"mmaze_official/memory_maze_9x9/{cell}/s0"
        cmd = (f"{{python}} probe_mm.py --configs mmprobe {' '.join(cfgs)} --data {data} --seed 0 "
               f"--mm_updates {MM_UPDATES} --mm_eval_every 10000 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=20))
    long_cells = {k: MAIN[k] for k in ("base_tau2", "ABG_tau128", "PD2L_tau1000", "ann_tau2")}
    for L, window in ((51, 64), (71, 80), (101, 128)):
        for job in rl2_jobs(f"bsuite_memlen-{L}", long_cells, steps=1e6,
                            extra=f"--batch_length {window} --env_seed_per_env True", eval_every=2e4):
            job["name"] = job["name"].replace("rl2/", "r2i/")
            job["cmd"] = job["cmd"].replace("/runs/rl2/", "/runs/r2i/")
            job["logdir"] = job["logdir"].replace("/runs/rl2/", "/runs/r2i/")
            jobs.append(job)
    # Medium (2026-09-29): Spiking-WM, A+B+G, ours, DreamerV3; LIF and GRSN stay at chance on Easy
    medium = {k: MAIN[k] for k in ("base_tau2", "ABG_tau128", "PD2L_tau1000", "ann_tau2")}
    for task, window, cells in (("RepeatPreviousEasy", 64, MAIN), ("RepeatPreviousMedium", 64, medium)):
        for job in rl2_jobs(f"popgym_{task}", cells, steps=2e6,
                            extra=f"--batch_length {window} --env_seed_per_env True", eval_every=5e4):
            job["name"] = job["name"].replace("rl2/", "r2i/")
            job["cmd"] = job["cmd"].replace("/runs/rl2/", "/runs/r2i/")
            job["logdir"] = job["logdir"].replace("/runs/rl2/", "/runs/r2i/")
            jobs.append(job)
    return jobs


def round14():
    """2026-10-02: validation of the integrate-and-fire soma (configs block somaif, EXPERIMENTS.md §17)
    before any rerun. Memory Maze showed the dendrites of our model storing about as much object
    information as DreamerV3's state, but the single spike frame per environment step passing
    almost none of it on. Checks, with the protocols of the existing results:
      1. memlen-L-recall probes L8-L64 (table 4): memory must not get worse;
      2. Memory Maze to 1M updates, keeping the 80k and 200k world models: frozen probes on the
         spikes vs the full state, against the old soma at 80k / 200k."""
    cells = {"PD2Lif_tau1000": ["ours", "somaif"]}
    jobs = []
    for L, extra in ((8, "--batch_length 32"), (16, ""), (32, ""), (64, "--batch_length 80")):
        jobs += recall_jobs(f"bsuite_memlen-{L}-recall", cells, seeds=(0,), extra=extra)
    data = ROOT / "data" / "memory_maze_9x9_official"
    name = "mmaze_official/memory_maze_9x9/PD2Lif_tau1000/s0"
    cmd = (f"{{python}} probe_mm.py --configs mmprobe ours somaif --data {data} --seed 0 "
           f"--mm_updates {MM_UPDATES} --mm_eval_every 10000 --keep 80000 200000 --logdir {RUNS / name}")
    jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=20))
    return jobs


def round15():
    """2026-10-02: resonant dendrites (configs blocks resonant, resonanthalf; EXPERIMENTS.md §18).
    POPGym RepeatPreviousMedium: at the end of the RL runs our world model predicts whether the
    previous action matched the card 32 steps back at AUC 0.545 (DreamerV3 0.653), and the random-data
    probe of round 12 stays at 0.54; on Easy (4 steps back) it is the weakest of the spiking models.
    Integrator dendrites hold a value but blur when it arrived; recalling the input of exactly k
    steps ago needs channels that also encode the lag. Checks against the existing results of the
    old model: the RepeatPrevious probes of round 12 (10k updates) and the memlen-L-recall probes of
    table 4 (memory must not get worse)."""
    cells = {"PD2Lres_tau1000": ["ours", "resonant"], "PD2Lreshalf_tau1000": ["ours", "resonanthalf"]}
    jobs = []
    for task in ("popgym_RepeatPreviousMedium", "popgym_RepeatPreviousEasy"):
        jobs += recall_jobs(task, {k: v + ["rewardbyact"] for k, v in cells.items()}, seeds=(0,), updates=10000,
                            extra="--batch_length 64", group="diag_rp", eval_every=1000)
    for L in (32, 64):
        jobs += recall_jobs(f"bsuite_memlen-{L}-recall", cells, seeds=(0,), extra=MEMLEN_EXTRA[L])
    return jobs


PIX_MEM = {"base_tau2": 24, "ann_tau2": 6}  # GPU memory (GB) of the DMC-from-pixels runs (smoke test 2026-10-04: 20 and 4.3)
V2 = ["ours", "resonanthalf", "policymem"]  # our model after §18 (resonant dendrites, actor reads the dendrites)


def round16():
    """2026-10-02: our model after §18 on every benchmark, with the protocols of the existing rows
    (EXPERIMENTS.md §18.3). The world-model probes reuse round 15 (PD2Lreshalf_tau1000: policymem only
    changes what the actor reads). Longest runs first. Also the RepeatPrevious probes of round 12 for
    the spiking baselines (precise-lag recall for every model)."""
    data = ROOT / "data" / "memory_maze_9x9_official"
    name = "mmaze_official/memory_maze_9x9/PD2Lv2_tau1000/s0"
    jobs = [dict(name=name, logdir=str(RUNS / name), mem_gb=20, cmd=" ".join((
        f"{{python}} probe_mm.py --configs mmprobe {' '.join(V2)} --data {data} --seed 0 --mm_updates {MM_UPDATES} "
        f"--mm_eval_every 10000 --keep 80000 200000 --logdir {RUNS / name}").split()))]
    for task in ("popgym_RepeatPreviousEasy", "popgym_RepeatPreviousMedium"):
        name = f"popgym/{task}/PD2Lv2_tau1000/s0"
        cmd = (f"{{python}} dreamer.py --configs popgym {' '.join(V2)} --task {task} --seed 0 --steps 2e6 "
               f"--train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True --eval_every 5e4 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=14.5))
    for L, window in ((51, 64), (71, 80), (101, 128)):
        for job in rl2_jobs(f"bsuite_memlen-{L}", {"PD2Lv2_tau1000": V2}, steps=1e6,
                            extra=f"--batch_length {window} --env_seed_per_env True", eval_every=2e4):
            jobs.append({k: v.replace("rl2/", "r2i/") if isinstance(v, str) else v for k, v in job.items()})
    for task in ("dmc_walker_walk", "dmc_cheetah_run", "dmc_cartpole_swingup"):
        for job in dmc_jobs(task, {"PD2Lv2_tau1000": V2}, steps=5e5, train_ratio=512, group="dmc500"):
            job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
            jobs.append(job)
    for L, steps in ((8, 1.5e5), (16, 2e5), (32, 3e5), (64, 5e5)):
        jobs += rl2_jobs(f"bsuite_memlen-{L}", {"PD2Lv2_tau1000": V2}, steps=steps, extra=MEMLEN_EXTRA[L])
    # 2026-10-02 20:40: at L64 the new world model recalls the cue (terminal reward AUC 1.0 on the
    # evaluation episodes at 160k steps) but the policy has not learned to use it (old model: 82k steps).
    # Ablation: the same model, the actor reading the soma spikes only (no policymem).
    jobs += rl2_jobs("bsuite_memlen-64", {"PD2Lreshalf_tau1000": ["ours", "resonanthalf"]}, steps=5e5, extra=MEMLEN_EXTRA[64])
    # 2026-10-03 02:30: that ablation learns L64 at 102k (with the dendrites in the actor input: 192k, and
    # not stable; L101 not learned at 284k, old model 202k). R2I feeds its policy the output state on
    # BSuite and DMC and the hidden / full state on POPGym and Memory Maze. The actor reading the soma
    # spikes only, on the other BSuite lengths and DMC (§18.6).
    R = ["ours", "resonanthalf"]
    out = []
    for L, steps in ((8, 1.5e5), (16, 2e5), (32, 3e5)):
        out += rl2_jobs(f"bsuite_memlen-{L}", {"PD2Lreshalf_tau1000": R}, steps=steps, extra=MEMLEN_EXTRA[L])
    for L, window in ((51, 64), (71, 80), (101, 128)):
        for job in rl2_jobs(f"bsuite_memlen-{L}", {"PD2Lreshalf_tau1000": R}, steps=1e6,
                            extra=f"--batch_length {window} --env_seed_per_env True", eval_every=2e4):
            out.append({k: v.replace("rl2/", "r2i/") if isinstance(v, str) else v for k, v in job.items()})
    for task in ("dmc_walker_walk", "dmc_cheetah_run", "dmc_cartpole_swingup"):
        for job in dmc_jobs(task, {"PD2Lreshalf_tau1000": R}, steps=5e5, train_ratio=512, group="dmc500"):
            job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
            out.append(job)
    # and POPGym, so that either choice of §18.6 has its runs (R2I: hidden state there)
    for task in ("popgym_RepeatPreviousEasy", "popgym_RepeatPreviousMedium"):
        name = f"popgym/{task}/PD2Lreshalf_tau1000/s0"
        cmd = (f"{{python}} dreamer.py --configs popgym {' '.join(R)} --task {task} --seed 0 --steps 2e6 "
               f"--train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True --eval_every 5e4 --logdir {RUNS / name}")
        out.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name)))
    for job in out:
        job["mem_gb"] = 26 if "memlen-101" in job["name"] else 18
    jobs += out
    # table 4: the remaining lengths; table 7: the ablations on top of the new model
    R = ["ours", "resonanthalf"]
    for L in (8, 16):
        jobs += recall_jobs(f"bsuite_memlen-{L}-recall", {"PD2Lreshalf_tau1000": R}, seeds=(0,), extra=MEMLEN_EXTRA[L])
    abl = {"PD2Lreshalfeuler_tau1000": R + ["pdeuler"], "PD2Lreshalf_tau128": R + ["pdtau128"],
           "PD2Lreshalffix_tau1000": R + ["pdfixtau"], "PD2Lreshalfcap_tau1000": R + ["ratecap"],
           "PDreshalf_tau1000": ["pdcell", "resonanthalf"]}
    for L in (32, 64):
        jobs += recall_jobs(f"bsuite_memlen-{L}-recall", abl, seeds=(0,), extra=MEMLEN_EXTRA[L])
    jobs += recall_jobs("bsuite_memlen-64-recall", {"PD2Lreshalfreset_tau1000": R + ["pdreset"],
                                                   "PD2Lreshalf_tau16": R + ["pdtau16"]}, seeds=(0,), extra=MEMLEN_EXTRA[64])
    jobs += recall_jobs("bsuite_memlen-32-recall", {"PDT1reshalf_tau1000": ["pdcell", "resonanthalf"]}, seeds=(0,),
                        extra="--spike_times 1")
    # the actor and critic read 7680 features (old model: 1536): measured GPU memory of the RL runs (+10%)
    mem = {"popgym/": 26, "r2i/bsuite_memlen-51/": 28, "r2i/bsuite_memlen-71/": 32, "r2i/bsuite_memlen-101/": 47,
           "dmc500/": 30, "rl2/bsuite_memlen-8/": 14, "rl2/bsuite_memlen-16/": 28, "rl2/bsuite_memlen-32/": 28,
           "rl2/bsuite_memlen-64/": 32}
    for job in jobs:
        for prefix, gb in mem.items():
            if job["name"].startswith(prefix) and "/PD2Lv2_" in job["name"]:
                job["mem_gb"] = gb
    # precise-lag recall probes (round 12 protocol) for the spiking baselines (DreamerV3 and the old model: round 12)
    base = {k: MAIN[k] + ["rewardbyact"] for k in ("base_tau2", "LIF_tau2", "GRSN_tau2", "ABG_tau128")}
    for task in ("popgym_RepeatPreviousMedium", "popgym_RepeatPreviousEasy"):
        jobs += recall_jobs(task, base, seeds=(0,), updates=10000, extra="--batch_length 64", group="diag_rp", eval_every=1000)
    return jobs


def round17():  # stopped 2026-10-04 15:10 (user: "同意停掉"): superseded by round 19, FwF at the baselines' size
    """2026-10-04: capacity control for the actor-critic (EXPERIMENTS.md §19.7). FwF's actor and critic
    read the dendritic spikes and have 12.85M parameters on POPGym, the soma-only ones 3.42M. One task:
    soma spikes only, widened to 12.85M. First on RepeatPrevious Easy; moved to Medium on 2026-10-04
    (Stage 3 review: Medium is the task only FwF learns; user decision "1a")."""
    name = "popgym/popgym_RepeatPreviousMedium/PD2Lreshalfacwide_tau1000/s0"
    cmd = (f"{{python}} dreamer.py --configs popgym ours resonanthalf acwide --task popgym_RepeatPreviousMedium --seed 0 "
           f"--steps 2e6 --train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True --eval_every 5e4 "
           f"--logdir {RUNS / name}")
    return [dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=26)]


def round18():
    """2026-10-04, after the Stage 3 review (user decision "3做", EXPERIMENTS.md §19.8): world-model probes only.
    1. Held-out memory task: POPGym RepeatPrevious Hard (the card of exactly 64 steps ago), not used while
       the model was developed. Round 12 protocol with the window doubled with the lag (128) and 20k
       updates, the same for every model.
    2. Design ablations of FwF on memlen-64-recall and RepeatPrevious Medium (protocols of table 5):
       a soma with a membrane that is reset by its spike (pdsomalif), dendrites that also receive the
       layer's own spikes (pdfeedback)."""
    R = ["ours", "resonanthalf"]
    hard = {k: MAIN[k] + ["rewardbyact"] for k in ("base_tau2", "LIF_tau2", "GRSN_tau2", "ABG_tau128", "ann_tau2")}
    hard["PD2Lreshalf_tau1000"] = R + ["rewardbyact"]
    jobs = list(recall_jobs("popgym_RepeatPreviousHard", hard, seeds=(0,), updates=20000,
                            extra="--batch_length 128", group="diag_rp", eval_every=1000))
    abl = {"PD2Lreshalfsomalif_tau1000": R + ["pdsomalif"], "PD2Lreshalffeedback_tau1000": R + ["pdfeedback"]}
    jobs += recall_jobs("bsuite_memlen-64-recall", abl, seeds=(0,), extra=MEMLEN_EXTRA[64])
    jobs += recall_jobs("popgym_RepeatPreviousMedium", {k: v + ["rewardbyact"] for k, v in abl.items()}, seeds=(0,),
                        updates=10000, extra="--batch_length 64", group="diag_rp", eval_every=1000)
    for job in jobs:
        if "RepeatPreviousHard" in job["name"]:
            job["mem_gb"] = 5
    return jobs


def round19():
    """2026-10-04, user decisions after the Stage 4 status report (EXPERIMENTS.md §19.11):
    1. FwF's actor-critic at the baselines' size (user: "能不能也这么大"; chosen: shrink ours to 3.4M).
       Two candidates, validated on the three tasks where the actor input matters most: POPGym
       RepeatPrevious Medium, DMC walker and BSuite memlen-64. Decision rule fixed in advance: the
       width-142 version (3.41M, the same size as the baselines) unless it is clearly worse than the
       rank-64 version (4.7M); the chosen one then reruns every RL benchmark of FwF.
    2. DMC from pixels (user: "补像素 DMC"), the setting Spiking-WM was evaluated in: DreamerV3's
       dmc_vision configuration, 1M env frames, train ratio 512. Spiking-WM and DreamerV3 now; FwF once
       item 1 is decided."""
    jobs = []
    cand = {"PD2Lv2m_tau1000": V2 + ["acmatch"], "PD2Lv2r64_tau1000": V2 + ["aclowrank"]}
    for cell, cfgs in cand.items():
        name = f"popgym/popgym_RepeatPreviousMedium/{cell}/s0"
        cmd = (f"{{python}} dreamer.py --configs popgym {' '.join(cfgs)} --task popgym_RepeatPreviousMedium --seed 0 "
               f"--steps 2e6 --train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True --eval_every 5e4 "
               f"--logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=26))
    for job in dmc_jobs("dmc_walker_walk", cand, steps=5e5, train_ratio=512, group="dmc500"):
        job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
        job["mem_gb"] = 30
        jobs.append(job)
    for job in rl2_jobs("bsuite_memlen-64", cand, steps=5e5, extra=MEMLEN_EXTRA[64]):
        job["mem_gb"] = 32
        jobs.append(job)
    return jobs


def round20():
    """DMC from pixels (round19 item 2). MuJoCo renders with OSMesa on the CPU: EGL hangs on this
    machine and rendering aborted under the MPS GPU setup before (runs/_dmc_render_abort)."""
    jobs = []
    # 2026-10-04 15:10, user: "这个像素的，我觉得选一个任务不就行了…我们只是想可视化或者说往 cv 稍微靠一下而已":
    # walker walk only (the most reported DMC task, first in Spiking-WM's table); cheetah and cartpole stopped.
    for task in ("dmc_walker_walk",):
        for cell in ("base_tau2", "ann_tau2"):
            name = f"dmcpix/{task}/{cell}/s0"
            cmd = (f"MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa {{python}} dreamer.py --configs dmc_vision {' '.join(MAIN[cell])} --task {task} --seed 0 "
                   f"--steps 1e6 --train_ratio 512 --eval_every 2e4 --env_seed_per_env True --logdir {RUNS / name}")
            jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=PIX_MEM[cell]))
    return jobs


def round21():
    """2026-10-04 15:10, user: "赶紧把需要重跑的跑完". FwF with the actor-critic at the baselines' size
    (acmatch, 142 units, 3.41M) on its other RL benchmarks, started without waiting for the round 19
    validation (if acmatch turns out clearly worse than aclowrank, these are rerun with aclowrank).
    Protocols and GPU memory as the PD2Lv2 rows of round 16. Plus DMC walker from pixels (round 20)."""
    M = V2 + ["acmatch"]
    cell = {"PD2Lv2m_tau1000": M}
    jobs = []
    for L, steps in ((8, 1.5e5), (16, 2e5), (32, 3e5)):
        jobs += rl2_jobs(f"bsuite_memlen-{L}", cell, steps=steps, extra=MEMLEN_EXTRA[L])
    for L, window in ((51, 64), (71, 80), (101, 128)):
        for job in rl2_jobs(f"bsuite_memlen-{L}", cell, steps=1e6,
                            extra=f"--batch_length {window} --env_seed_per_env True", eval_every=2e4):
            jobs.append({k: v.replace("rl2/", "r2i/") if isinstance(v, str) else v for k, v in job.items()})
    name = "popgym/popgym_RepeatPreviousEasy/PD2Lv2m_tau1000/s0"
    cmd = (f"{{python}} dreamer.py --configs popgym {' '.join(M)} --task popgym_RepeatPreviousEasy --seed 0 --steps 2e6 "
           f"--train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True --eval_every 5e4 --logdir {RUNS / name}")
    jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=26))
    for task in ("dmc_cheetah_run", "dmc_cartpole_swingup"):
        for job in dmc_jobs(task, cell, steps=5e5, train_ratio=512, group="dmc500"):
            job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
            jobs.append(job)
    mem = {"r2i/bsuite_memlen-51/": 28, "r2i/bsuite_memlen-71/": 32, "r2i/bsuite_memlen-101/": 47, "dmc500/": 30,
           "rl2/bsuite_memlen-8/": 14, "rl2/bsuite_memlen-16/": 28, "rl2/bsuite_memlen-32/": 28}
    for job in jobs:
        for prefix, gb in mem.items():
            if job["name"].startswith(prefix):
                job["mem_gb"] = gb
    name = "dmcpix/dmc_walker_walk/PD2Lv2m_tau1000/s0"
    cmd = (f"MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa {{python}} dreamer.py --configs dmc_vision {' '.join(M)} "
           f"--task dmc_walker_walk --seed 0 --steps 1e6 --train_ratio 512 --eval_every 2e4 --env_seed_per_env True "
           f"--logdir {RUNS / name}")
    jobs.insert(0, dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=30))
    # 2026-10-06, user: "同意，按你的推荐来吧": FwF switches to aclowrank (EXPERIMENTS.md §19.16). acmatch stays
    # as the equal-size appendix row (BSuite, POPGym Medium, DMC walker); its other runs were stopped.
    stopped = {"dmc500/dmc_cheetah_run/PD2Lv2m_tau1000/s0", "dmc500/dmc_cartpole_swingup/PD2Lv2m_tau1000/s0",
               "popgym/popgym_RepeatPreviousEasy/PD2Lv2m_tau1000/s0", "dmcpix/dmc_walker_walk/PD2Lv2m_tau1000/s0"}
    return [j for j in jobs if j["name"] not in stopped]


def round22():
    """2026-10-05 09:40, insurance for the round 19 decision. On DMC walker the width-142 actor-critic (acmatch)
    trails the rank-64 one (aclowrank) at 45% (337 vs 544 at 220k steps); if acmatch ends clearly worse, FwF
    switches to aclowrank on every benchmark. User: "可以可以，按照你的推荐来". The aclowrank runs of the other
    RL benchmarks (L64, POPGym Medium and DMC walker are in round 19), longest first; if acmatch stays, they
    are an appendix ablation. GPU memory: measured aclowrank runs (28-34 GB) + 15%.
    2026-10-06: acmatch ended clearly worse on walker, aclowrank is FwF's configuration (EXPERIMENTS.md §19.16)."""
    M = V2 + ["aclowrank"]
    cell = {"PD2Lv2r64_tau1000": M}
    name = "dmcpix/dmc_walker_walk/PD2Lv2r64_tau1000/s0"
    cmd = (f"MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa {{python}} dreamer.py --configs dmc_vision {' '.join(M)} "
           f"--task dmc_walker_walk --seed 0 --steps 1e6 --train_ratio 512 --eval_every 2e4 --env_seed_per_env True "
           f"--logdir {RUNS / name}")
    jobs = [dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=34)]
    name = "popgym/popgym_RepeatPreviousEasy/PD2Lv2r64_tau1000/s0"
    cmd = (f"{{python}} dreamer.py --configs popgym {' '.join(M)} --task popgym_RepeatPreviousEasy --seed 0 --steps 2e6 "
           f"--train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True --eval_every 5e4 --logdir {RUNS / name}")
    jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=32))
    for task in ("dmc_cheetah_run", "dmc_cartpole_swingup"):
        for job in dmc_jobs(task, cell, steps=5e5, train_ratio=512, group="dmc500"):
            job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
            job["mem_gb"] = 34
            jobs.append(job)
    for L, window, gb in ((101, 128, 54), (71, 80, 37), (51, 64, 32)):
        for job in rl2_jobs(f"bsuite_memlen-{L}", cell, steps=1e6,
                            extra=f"--batch_length {window} --env_seed_per_env True", eval_every=2e4):
            job = {k: v.replace("rl2/", "r2i/") if isinstance(v, str) else v for k, v in job.items()}
            job["mem_gb"] = gb
            jobs.append(job)
    for L, steps, gb in ((8, 1.5e5, 16), (16, 2e5, 32), (32, 3e5, 32)):
        for job in rl2_jobs(f"bsuite_memlen-{L}", cell, steps=steps, extra=MEMLEN_EXTRA[L]):
            job["mem_gb"] = gb
            jobs.append(job)
    return jobs


def round23():
    """2026-10-06, user: "按照你的意见来吧". POPGym RepeatPrevious Medium for recurrent LIF and GRSN, with the
    protocol of round 13. They had been left out because at the time they were also at chance on Easy; both
    learned Easy by 2M steps (0.69, 0.33), so the Medium column of the main table needs them (EXPERIMENTS.md §19.17)."""
    jobs = []
    for cell in ("LIF_tau2", "GRSN_tau2"):
        name = f"popgym/popgym_RepeatPreviousMedium/{cell}/s0"
        logdir = RUNS / name
        cmd = (f"{{python}} dreamer.py --configs popgym {' '.join(MAIN[cell])} --task popgym_RepeatPreviousMedium --seed 0 "
               f"--steps 2e6 --train_ratio 64 --envs 8 --batch_length 64 --env_seed_per_env True "
               f"--eval_every 5e4 --logdir {logdir}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(logdir), mem_gb=14.5))
    return jobs


def round24():
    """2026-10-06, user: "可以补，把卡八卡都用起来，加加速". DMC from proprioception for the other spiking
    baselines (recurrent LIF, GRSN, Spiking-WM + A+B+G), with the protocol of the dmc500 group (500k steps, train
    ratio 512, a seed per env), so that the non-memory control compares FwF with every spiking model
    (EXPERIMENTS.md §19.18). GPU memory as for Spiking-WM."""
    cells = {k: MAIN[k] for k in ("LIF_tau2", "GRSN_tau2", "ABG_tau128")}
    jobs = []
    for task in ("dmc_walker_walk", "dmc_cheetah_run", "dmc_cartpole_swingup"):
        for job in dmc_jobs(task, cells, steps=5e5, train_ratio=512, group="dmc500"):
            job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
            jobs.append(job)
    return jobs


def round25():
    """2026-10-07, user: "可以，先诊断像素 walker，有结论再找我". Pixel walker world models trained offline on the
    same data, DreamerV3's replay buffer of its pixel walker run (diag_wm_offline.py; EXPERIMENTS.md §19.22): in
    RL each world model trains on the data of its own policy. Also FwF with the thermometer soma (T graded
    frames per step instead of one), the candidate fix if the binary soma limits velocity coding. GPU memory
    measured: Spiking-WM and LIF 13 GB, DreamerV3 3 GB."""
    data = RUNS / "dmcpix/dmc_walker_walk/ann_tau2/s0/train_eps"
    cells = {"base_tau2": ([], 16), "LIF_tau2": (["lifcell"], 16), "ann_tau2": (["grucell", "annwm"], 8),
             "PD2Lv2_tau1000": (["ours", "resonanthalf"], 18), "PD2Lv2thermo_tau1000": (["ours", "resonanthalf", "somathermo"], 18)}
    jobs = []
    for cell, (cfgs, gb) in cells.items():
        name = f"diag_pix/dmc_walker_walk/{cell}/s0"
        cmd = (f"{{python}} diag_wm_offline.py --configs dmc_vision {' '.join(cfgs)} --data {data} --updates 30000 "
               f"--eval_every 3000 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=gb))
    return jobs


def round26(jobs):
    """2026-10-07 15:30, user: "好的…我们得确认他已经改进到基本可以了" (EXPERIMENTS.md §19.23). RL check of the
    thermometer soma (somathermo) before any rerun: the 4.7M FwF runs of BSuite L32 and L64 (memory must not
    get worse), pixel walker (to 500k frames: compared at matched frames with the binary soma and Spiking-WM)
    and proprioceptive walker, each with the soma switched and nothing else changed. GPUs 0-3 only (user)."""
    picks = ("rl2/bsuite_memlen-32/", "dmcpix/dmc_walker_walk/", "rl2/bsuite_memlen-64/", "dmc500/dmc_walker_walk/")
    out = []
    for prefix in picks:
        job = next(j for j in jobs if j["name"] == f"{prefix}PD2Lv2r64_tau1000/s0")
        job = {k: (v.replace("PD2Lv2r64_tau1000", "PD2Lv2r64thermo_tau1000") if isinstance(v, str) else v)
               for k, v in job.items()}
        job["cmd"] = job["cmd"].replace(" aclowrank ", " aclowrank somathermo ")
        if prefix.startswith("dmcpix/"):
            job["cmd"] = job["cmd"].replace(" --steps 1e6 ", " --steps 5e5 ")
        assert "somathermo" in job["cmd"]
        out.append(job)
    # 2026-10-08 00:45 (EXPERIMENTS.md §19.26): thermo alone fails the L64 recall probe (AUC 0.50 at 9k updates; binary
    # soma 3.5k), so only its L32 run (already running) is kept; the RL check moves to thermo + feedback (round 28)
    return out[:1]


def round27():
    """2026-10-07 21:10, user: "现在的 Thermo 做到了当前任务里面的 spike sota 吗？如果还弱于别的 spike 的，其实还是有点
    问题的" (EXPERIMENTS.md §19.24). Offline on the same pixel walker data (round 25), thermo still trails LIF on
    every metric; LIF has a nonlinear recurrence (spikes fed back), our dendrites do not. Spike feedback into the
    dendrites (pdfeedback) cost almost nothing in memory in round 18 (L64 4k vs 3.5k updates, Medium 7k vs 6k).
    Cheap checks first: the pixel world model with thermo + feedback, and the memory probes of round 18 (Table 5
    protocol) for thermo and thermo + feedback."""
    data = RUNS / "dmcpix/dmc_walker_walk/ann_tau2/s0/train_eps"
    name = "diag_pix/dmc_walker_walk/PD2Lv2thermofb_tau1000/s0"
    cmd = (f"{{python}} diag_wm_offline.py --configs dmc_vision ours resonanthalf somathermo pdfeedback --data {data} "
           f"--updates 30000 --eval_every 3000 --logdir {RUNS / name}")
    jobs = [dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=18)]
    for cell, cfgs in (("PD2Lreshalfthermo_tau1000", "somathermo"), ("PD2Lreshalfthermofb_tau1000", "somathermo pdfeedback")):
        name = f"recall/bsuite_memlen-64-recall/{cell}/s0"
        cmd = (f"{{python}} probe_wm.py --configs bsuite ours resonanthalf {cfgs} --task bsuite_memlen-64-recall --seed 0 "
               f"--updates 12000 --eval_every 500 --reward_head symlog_mse --batch_length 80 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=cmd, logdir=str(RUNS / name), mem_gb=2.5))
        name = f"diag_rp/popgym_RepeatPreviousMedium/{cell}/s0"
        cmd = (f"{{python}} probe_wm.py --configs popgym ours resonanthalf {cfgs} rewardbyact --task popgym_RepeatPreviousMedium "
               f"--seed 0 --updates 10000 --eval_every 1000 --reward_head symlog_mse --batch_length 64 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=cmd, logdir=str(RUNS / name), mem_gb=2.5))
    return jobs


def round28():
    """2026-10-08 00:45, user: "所以现在有结论了吗" (EXPERIMENTS.md §19.26). Thermo + spike feedback keeps the memory probes
    (L64 5k, Medium 7k updates) and brings the pixel world model to LIF's level, but R adds 2.1M parameters to the world
    model (18.9M; ours now 16.8M, LIF 14.6M, Spiking-WM 15.4M). With two dendritic channels per neuron (pdch2) B and R
    together have the parameters of B alone: the same world-model size as now. Cheap checks of that variant first."""
    data = RUNS / "dmcpix/dmc_walker_walk/ann_tau2/s0/train_eps"
    cfg = "somathermo pdfeedback pdch2"
    cell = "PD2Lv2thermofbch2_tau1000"
    name = f"diag_pix/dmc_walker_walk/{cell}/s0"
    cmd = (f"{{python}} diag_wm_offline.py --configs dmc_vision ours resonanthalf {cfg} --data {data} "
           f"--updates 30000 --eval_every 3000 --logdir {RUNS / name}")
    jobs = [dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=18)]
    cell = "PD2Lreshalfthermofbch2_tau1000"
    name = f"recall/bsuite_memlen-64-recall/{cell}/s0"
    cmd = (f"{{python}} probe_wm.py --configs bsuite ours resonanthalf {cfg} --task bsuite_memlen-64-recall --seed 0 "
           f"--updates 12000 --eval_every 500 --reward_head symlog_mse --batch_length 80 --logdir {RUNS / name}")
    jobs.append(dict(name=name, cmd=cmd, logdir=str(RUNS / name), mem_gb=2.5))
    name = f"diag_rp/popgym_RepeatPreviousMedium/{cell}/s0"
    cmd = (f"{{python}} probe_wm.py --configs popgym ours resonanthalf {cfg} rewardbyact --task popgym_RepeatPreviousMedium "
           f"--seed 0 --updates 10000 --eval_every 1000 --reward_head symlog_mse --batch_length 64 --logdir {RUNS / name}")
    jobs.append(dict(name=name, cmd=cmd, logdir=str(RUNS / name), mem_gb=2.5))
    return jobs


def round29(jobs):
    """2026-10-08 03:00, user: "你觉得我们最终采用哪个配置和版本最好" (EXPERIMENTS.md §19.27). Thermo + feedback with two
    dendritic channels keeps the memory probes (L64 5.5k updates) and the world-model size (16.75M), and its pixel
    world model beats Spiking-WM and is level with LIF. The RL check of round 26 with this variant: the 4.7M runs of
    BSuite L32 and L64, pixel walker (to 500k frames) and proprioceptive walker, nothing else changed."""
    picks = ("rl2/bsuite_memlen-32/", "dmcpix/dmc_walker_walk/", "rl2/bsuite_memlen-64/", "dmc500/dmc_walker_walk/")
    out = []
    for prefix in picks:
        job = next(j for j in jobs if j["name"] == f"{prefix}PD2Lv2r64_tau1000/s0")
        job = {k: (v.replace("PD2Lv2r64_tau1000", "PD2Lv2r64thermofbch2_tau1000") if isinstance(v, str) else v)
               for k, v in job.items()}
        job["cmd"] = job["cmd"].replace(" aclowrank ", " aclowrank somathermo pdfeedback pdch2 ")
        if prefix.startswith("dmcpix/"):
            job["cmd"] = job["cmd"].replace(" --steps 1e6 ", " --steps 5e5 ")
        assert "pdch2" in job["cmd"]
        out.append(job)
    return out


def round30(jobs):
    """2026-10-08 05:40, user: "可以，按你说的两步走" (EXPERIMENTS.md §19.28). The L32 check of round 29 reached 1.0 at
    32.5k steps (binary soma 42.5k), so the memory side of the paper is rerun now with the final model, in parallel
    with the DMC checks of round 29: BSuite L8, L16 (L32, L64: round 29), L51, L71, L101, POPGym RepeatPrevious Easy
    and Medium, the world-model probes (L64 recall and Medium: round 28) and Memory Maze."""
    flags = "somathermo pdfeedback pdch2"
    renames = {"PD2Lv2r64_tau1000": ("PD2Lv2r64thermofbch2_tau1000", " aclowrank ", f" aclowrank {flags} "),
               "PD2Lreshalf_tau1000": ("PD2Lreshalfthermofbch2_tau1000", " resonanthalf ", f" resonanthalf {flags} "),
               "PD2Lv2_tau1000": ("PD2Lv2thermofbch2_tau1000", " policymem ", f" policymem {flags} ")}
    # the small probes first (they must not wait behind a large job), then the longest runs
    picks = [(f"recall/bsuite_memlen-{n}-recall/", "PD2Lreshalf_tau1000") for n in (8, 16, 32)]
    picks += [(f"diag_rp/popgym_RepeatPrevious{d}/", "PD2Lreshalf_tau1000") for d in ("Easy", "Hard")]
    picks += [("mmaze_official/memory_maze_9x9/", "PD2Lv2_tau1000"), ("r2i/bsuite_memlen-101/", "PD2Lv2r64_tau1000"),
              ("popgym/popgym_RepeatPreviousEasy/", "PD2Lv2r64_tau1000"),
              ("popgym/popgym_RepeatPreviousMedium/", "PD2Lv2r64_tau1000"), ("r2i/bsuite_memlen-71/", "PD2Lv2r64_tau1000"),
              ("r2i/bsuite_memlen-51/", "PD2Lv2r64_tau1000"), ("rl2/bsuite_memlen-16/", "PD2Lv2r64_tau1000"),
              ("rl2/bsuite_memlen-8/", "PD2Lv2r64_tau1000")]
    out = []
    for prefix, cell in picks:
        new, old_flag, new_flag = renames[cell]
        job = next(j for j in jobs if j["name"] == f"{prefix}{cell}/s0")
        job = {k: (v.replace(cell, new) if isinstance(v, str) else v) for k, v in job.items()}
        assert old_flag in job["cmd"]
        job["cmd"] = job["cmd"].replace(old_flag, new_flag, 1)
        out.append(job)
    return out


def round31(jobs):
    """2026-10-09 01:30, user: "可以，按你的建议来吧" (EXPERIMENTS.md §19.30), after the review of §19.29.
    1. Table 7 on the final model: each ablation changes one part of it, on the L32 and L64 recall probes and the
       RepeatPrevious Medium probe (protocols of Table 4); the soma, the spike feedback and the channel count
       included. Its own row: PD2Lreshalfthermofbch2 (rounds 28, 30).
    2. Pixel walker world-model diagnostics (round 25 protocol) without thermo and without feedback.
    3. BSuite L8-L64 again for every model with a different seed per parallel environment (rl2 shared one seed,
       §14.2), as rl3; with them our model with the baselines' actor-critic (soma spikes only, 3.41M).
    4. The same actor-critic control on POPGym Easy and Medium."""
    out = []
    final = ["ours", "resonanthalf", "somathermo", "pdfeedback", "pdch2"]
    abl = {"SDDreset_tau1000": final + ["pdreset"], "SDD_tau16": final + ["pdtau16"], "SDD_tau128": final + ["pdtau128"],
           "SDDeuler_tau1000": final + ["pdeuler"], "SDDfix_tau1000": final + ["pdfixtau"],
           "SDDcap_tau1000": final + ["ratecap"], "SDD1L_tau1000": ["pdcell"] + final[1:],
           "SDDres_tau1000": ["ours", "resonant"] + final[2:], "SDDnores_tau1000": ["ours"] + final[2:],
           "SDDsomalif_tau1000": ["ours", "resonanthalf", "pdsomalif", "pdfeedback", "pdch2"],
           "SDDstep_tau1000": ["ours", "resonanthalf", "pdfeedback", "pdch2"],
           "SDDnofb_tau1000": ["ours", "resonanthalf", "somathermo", "pdch2"],
           "SDDch4_tau1000": ["ours", "resonanthalf", "somathermo", "pdfeedback"]}
    t1 = {"SDD1LT1_tau1000": ["pdcell"] + final[1:]}
    for L in (32, 64):
        out += recall_jobs(f"bsuite_memlen-{L}-recall", abl, seeds=(0,), extra=MEMLEN_EXTRA[L])
        out += recall_jobs(f"bsuite_memlen-{L}-recall", t1, seeds=(0,), extra=MEMLEN_EXTRA[L] + " --spike_times 1")
    rp = {k: v + ["rewardbyact"] for k, v in abl.items()}
    out += recall_jobs("popgym_RepeatPreviousMedium", rp, seeds=(0,), updates=10000, extra="--batch_length 64",
                       group="diag_rp", eval_every=1000)
    out += recall_jobs("popgym_RepeatPreviousMedium", {k: v + ["rewardbyact"] for k, v in t1.items()}, seeds=(0,),
                       updates=10000, extra="--batch_length 64 --spike_times 1", group="diag_rp", eval_every=1000)
    data = RUNS / "dmcpix/dmc_walker_walk/ann_tau2/s0/train_eps"
    for cell, cfg in (("PD2Lv2stepfbch2_tau1000", "pdfeedback pdch2"), ("PD2Lv2thermoch2_tau1000", "somathermo pdch2")):
        name = f"diag_pix/dmc_walker_walk/{cell}/s0"
        cmd = (f"{{python}} diag_wm_offline.py --configs dmc_vision ours resonanthalf {cfg} --data {data} "
               f"--updates 30000 --eval_every 3000 --logdir {RUNS / name}")
        out.append(dict(name=name, cmd=cmd, logdir=str(RUNS / name), mem_gb=18))
    flags = "somathermo pdfeedback pdch2"
    by_name = {j["name"]: j for j in jobs}
    rl3 = []
    for L in (64, 32, 16, 8):  # longest first
        for cell in ("PD2Lv2r64thermofbch2_tau1000", "PD2Lreshalfthermofbch2_tau1000", "base_tau2", "LIF_tau2",
                     "GRSN_tau2", "ABG_tau128", "ann_tau2", "gru_tau2"):
            src = {"PD2Lv2r64thermofbch2_tau1000": "PD2Lv2r64_tau1000",
                   "PD2Lreshalfthermofbch2_tau1000": "PD2Lreshalf_tau1000"}.get(cell, cell)
            job = by_name.get(f"rl2/bsuite_memlen-{L}/{src}/s0")
            if job is None:
                continue
            job = {k: (v.replace("/rl2/", "/rl3/").replace("rl2/", "rl3/", 1).replace(src, cell)
                       if isinstance(v, str) else v) for k, v in job.items()}
            if src != cell:
                anchor = " aclowrank " if "aclowrank" in job["cmd"] else " resonanthalf "
                job["cmd"] = job["cmd"].replace(anchor, f"{anchor}{flags} ", 1)
                job["mem_gb"] = max(job["mem_gb"], 24)
            assert "--env_seed_per_env" not in job["cmd"]
            job["cmd"] = job["cmd"].replace(" --logdir ", " --env_seed_per_env True --logdir ")
            rl3.append(job)
    out += rl3
    for task in ("popgym_RepeatPreviousEasy", "popgym_RepeatPreviousMedium"):
        job = by_name[f"popgym/{task}/PD2Lreshalf_tau1000/s0"]
        job = {k: (v.replace("PD2Lreshalf_tau1000", "PD2Lreshalfthermofbch2_tau1000") if isinstance(v, str) else v)
               for k, v in job.items()}
        job["cmd"] = job["cmd"].replace(" resonanthalf ", f" resonanthalf {flags} ", 1)
        job["mem_gb"] = 26
        out.append(job)
    return out


def round32():
    """2026-10-09 02:40, user: "可以" (EXPERIMENTS.md §19.32). The final model never learns R2I L101 in RL (716k steps):
    its reward head stays at the predict-zero loss, while the old r64 learned it at 302k. Recall probe at L=101,
    window 128 as in the RL run: can the final world model carry the cue 101 steps, and if not, is it the spike
    feedback or the thermometer soma. The old r64's world model is PD2Lreshalf."""
    cells = {"PD2Lreshalfthermofbch2_tau1000": ["ours", "resonanthalf", "somathermo", "pdfeedback", "pdch2"],
             "SDDnofb_tau1000": ["ours", "resonanthalf", "somathermo", "pdch2"],
             "SDDstep_tau1000": ["ours", "resonanthalf", "pdfeedback", "pdch2"],
             "PD2Lreshalf_tau1000": ["ours", "resonanthalf"]}
    out = list(recall_jobs("bsuite_memlen-101-recall", cells, seeds=(0,), extra="--batch_length 128"))
    # 05:10: the final model passes the recall probe (AUC 1.0 at 3000 updates). In RL the reward at the last step is
    # +1/-1 by whether the action matches the cue (rewardbyact head, as in the RL run): the same probe on that task
    xor = {k: cells[k] + ["rewardbyact"] for k in ("PD2Lreshalfthermofbch2_tau1000", "PD2Lreshalf_tau1000")}
    out += recall_jobs("bsuite_memlen-101", xor, seeds=(0,), extra="--batch_length 128", group="xor")
    for job in out:
        job["mem_gb"] = 6  # peak 5.7 GB measured; the first try at 4 GB went out of memory on a full GPU
    return out


def round33(jobs):
    """2026-10-09 09:21 (EXPERIMENTS.md §19.33), step 2 of the plan of 2026-10-08 (user: "可以，按你说的两步走") for the
    proprioceptive side: the final model's walker run passed its check at 300k frames (mean over 100k-300k 828, LIF 731).
    The final model on cheetah run and cartpole swingup, and the actor-critic control (soma spikes only, as the
    baselines) on all three tasks. Longest jobs, first in the queue. The pixel side waits for its own check."""
    flags = "somathermo pdfeedback pdch2"
    by_name = {j["name"]: j for j in jobs}
    out = []
    for task in ("dmc_cheetah_run", "dmc_cartpole_swingup", "dmc_walker_walk"):
        for src, cell, anchor in (("PD2Lv2r64_tau1000", "PD2Lv2r64thermofbch2_tau1000", " aclowrank "),
                                  ("PD2Lreshalf_tau1000", "PD2Lreshalfthermofbch2_tau1000", " resonanthalf ")):
            if task == "dmc_walker_walk" and cell.startswith("PD2Lv2r64"):
                continue  # round 29
            job = by_name[f"dmc500/{task}/{src}/s0"]
            job = {k: (v.replace(src, cell) if isinstance(v, str) else v) for k, v in job.items()}
            job["cmd"] = job["cmd"].replace(anchor, f"{anchor}{flags} ", 1)
            assert job["cmd"].count("pdch2") == 1
            job["mem_gb"] = max(job["mem_gb"], 22)
            out.append(job)
    return out


def main():
    # priority: rounds 3-4, round 5 probes and BSuite RL, L64 ablations and controls (rounds 7, 9),
    # Memory Maze pipeline check (round 10), Memory Maze on the official data once downloaded, round 8
    # (BSuite long memory, POPGym as in R2I), then DMC (round 11).
    # Superseded: the POPGym runs of round 5 (tasks and budget not those of R2I) by round 8, round 6
    # (own Memory Maze data) by rounds 10 and 8, the DMC runs of round 5 (train ratio 64) by round 11.
    # The POPGym runs of round 8 (action-indexed MSE reward head) by round 13.
    r5 = round5()
    r5 = [j for j in r5 if "/popgym_" not in j["name"] and "/dmc_" not in j["name"]]
    r8 = round8()
    mm_full = [j for j in r8 if j["name"].startswith("mmaze_official/")]
    r8 = [j for j in r8 if j not in mm_full and "/popgym_" not in j["name"]]  # POPGym: round 13
    # Round 14 (integrate-and-fire soma) failed check V1 on 2026-10-02 (EXPERIMENTS.md §17.4): its
    # unfinished runs were stopped and are not queued again.
    r14 = [j for j in round14() if j["name"] == "recall/bsuite_memlen-8-recall/PD2Lif_tau1000/s0"]
    # 2026-10-06 (EXPERIMENTS.md §19.18): the short BSuite runs of aclowrank first (energy count at L=32), the
    # 54-GB L=101 run after the DMC baselines, so that it does not hold a GPU half empty while it waits for room.
    r22 = round22()
    short = [j for j in r22 if "/bsuite_memlen-16/" in j["name"] or "/bsuite_memlen-32/" in j["name"]]
    l101 = [j for j in r22 if "/bsuite_memlen-101/" in j["name"]]
    r22 = short + [j for j in r22 if j not in short and j not in l101]
    jobs = (round19() + round21() + r22 + round23() + round24() + round25() + l101 + round20() + round18() + round16() + round15() + r14 + round3() + round4() + r5 + round7() + round9() + round10() + round12() + mm_full + round13()
            + r8 + round11())
    # the old model's 1M Memory Maze continuation was stopped on 2026-10-02 (superseded, EXPERIMENTS.md §18.4)
    jobs = [j for j in jobs if j["name"] != "mmaze_official/memory_maze_9x9/PD2L_tau1000/s0"]
    jobs = round33(jobs) + round32() + round29(jobs) + round30(jobs) + round31(jobs) + round28() + round27() + round26(jobs) + jobs
    # 2026-10-09 (EXPERIMENTS.md §19.29), user: "可以停": the binary-soma Memory Maze run is superseded by the final
    # model's (round 30, which copies its job); stopped at 540k updates
    jobs = [j for j in jobs if j["name"] != "mmaze_official/memory_maze_9x9/PD2Lv2_tau1000/s0"]
    for job in jobs:  # DreamerV3 (ANN) needs about 4 GB (measured 2026-09-30), the spiking models 14-20 GB
        if "/ann_" in job["name"] and job["mem_gb"] > 6:
            job["mem_gb"] = 7 if job["name"].startswith("mmaze") else 6
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
