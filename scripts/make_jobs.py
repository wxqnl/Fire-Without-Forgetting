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


def round17():
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
    jobs = (round18() + round17() + round16() + round15() + r14 + round3() + round4() + r5 + round7() + round9() + round10() + round12() + mm_full + round13()
            + r8 + round11())
    # the old model's 1M Memory Maze continuation was stopped on 2026-10-02 (superseded, EXPERIMENTS.md §18.4)
    jobs = [j for j in jobs if j["name"] != "mmaze_official/memory_maze_9x9/PD2L_tau1000/s0"]
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
