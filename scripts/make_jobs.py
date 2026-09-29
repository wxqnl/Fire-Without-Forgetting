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


def dmc_jobs(task, cells, steps=2e5, seed=0):
    """DMC from proprioception, continuous actions: original reward head, no rewardbyact."""
    for cell, cfgs in cells.items():
        name = f"rl2/{task}/{cell}/s{seed}"
        cmd = (f"{{python}} dreamer.py --configs dmc_proprio {' '.join(cfgs)} --task {task} --seed {seed} "
               f"--steps {steps:g} --train_ratio 64 --eval_every 2e4 --logdir {RUNS / name}")
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
    """2026-09-29 06:30: Memory Maze offline probing on the part of the official training data that
    Google Drive served (its download quota refuses the rest): the 1471 complete episodes at the head
    of 9 of the 10 archives (scripts/salvage_memory_maze.py), 5% of the 29k, with the official
    evaluation set and probe. 50k updates of 32 x 48 pass over these 1.47M steps as often as the
    paper's 1M updates pass over its 29M (53 times). Ours and DreamerV3 first."""
    data = ROOT / "data" / "memory_maze_9x9_official_subset"
    if not (data / "train").is_dir():
        return []
    jobs = []
    for cell in ("PD2L_tau1000", "ann_tau2", "base_tau2", "ABG_tau128", "LIF_tau2", "GRSN_tau2"):
        name = f"mmaze_subset/memory_maze_9x9/{cell}/s0"
        cmd = (f"{{python}} probe_mm.py --configs mmprobe {' '.join(MAIN[cell])} --data {data} --seed 0 "
               f"--mm_updates 50000 --mm_eval_every 5000 --logdir {RUNS / name}")
        jobs.append(dict(name=name, cmd=" ".join(cmd.split()), logdir=str(RUNS / name), mem_gb=20))
    return jobs


MM_UPDATES = 200000  # Memory Maze gradient steps; the benchmark paper trains 1M


def round8():
    """2026-09-28 21:00, after the GPU reclaim: memory benchmarks of the world-model literature with
    published ANN results, run with their protocols (EXPERIMENTS.md §10). All with a different seed
    per parallel env (env_seed_per_env).

    1. Memory Maze offline probing (Pasukonis et al. 2023) on the official 9x9 dataset, replacing
       round 6 (own data, probe without the agent pose, 20k updates: no model left the no-memory
       level). Reference: RSSM / RSSM-TBTT / VAE+GRU at 1M steps.
    2. BSuite Memory Length as in R2I (Samsonov et al. 2024, Fig. 3/7): 1M env steps per memory
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
    for task, window in (("RepeatPreviousEasy", 64), ("RepeatPreviousMedium", 64)):
        for job in rl2_jobs(f"popgym_{task}", MAIN, steps=2e6,
                            extra=f"--batch_length {window} --env_seed_per_env True", eval_every=5e4):
            job["name"] = job["name"].replace("rl2/", "r2i/")
            job["cmd"] = job["cmd"].replace("/runs/rl2/", "/runs/r2i/")
            job["logdir"] = job["logdir"].replace("/runs/rl2/", "/runs/r2i/")
            jobs.append(job)
    return jobs


def main():
    # priority: rounds 3-4, round 5 probes and BSuite RL, L64 ablations and controls (rounds 7, 9),
    # Memory Maze on the served part of the official data (round 10), round 8 (Memory Maze official,
    # BSuite long memory, POPGym as in R2I), then DMC. Superseded by round 8: the POPGym runs of
    # round 5 (tasks and budget not those of R2I) and round 6 (own Memory Maze data).
    r5 = round5()
    popgym5 = [j for j in r5 if "/popgym_" in j["name"]]
    dmc = [j for j in r5 if "/dmc_" in j["name"]]
    jobs = (round3() + round4() + [j for j in r5 if j not in popgym5 + dmc] + round7() + round9() + round10() + round8() + dmc)
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
