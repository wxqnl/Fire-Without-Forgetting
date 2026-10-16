"""Minimal GPU job queue for the Phase 0 grid.

Jobs file: one JSON object per line: {"name", "cmd", "logdir", "mem_gb"}, optionally "gpus": the only GPUs the
job may start on (long runs placed on the GPUs kept to the end, 2026-10-09). File order is
priority. The file is re-read every poll, so jobs can be added while the queue runs
(write it atomically, see make_jobs.py).

Per job the queue keeps small files in <logdir>:
  .pid   pid of the job's wrapper shell   .gpu   physical GPU index
  .exit  exit code, written by the job wrapper itself (so it survives a queue restart)
A job is done when .exit holds 0; failed jobs are retried once. On start the queue
adopts jobs whose .pid is still alive instead of launching them twice.

Scheduling: memory is accounted with each job's mem_gb (plus a live nvidia-smi check
against other users). When the highest-priority pending job does not fit anywhere, the
GPU with the most headroom is reserved for it and smaller jobs backfill the others. After a launch the GPU gets no
other job for --cooldown seconds: a new job allocates its memory only after its start-up, so the live check would
otherwise see stale free memory and stack jobs (2026-10-11 01:52: 30 probes started in one pass, GPUs ran full).

GPUs in --idle_gpus are shared with other people: a job is started there only after the GPU
has had no process of anyone else for --idle_min minutes, and no further job once one appears
(processes already running are never touched). All work on the machine runs under one user
account, so "ours" means: in the session of one of our jobs (each job is its own session).

usage: setsid nohup python scripts/queue.py --jobs runs/jobs.jsonl --gpus 1,2,3,4,5,6 &
"""
import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def gpu_free_gb(gpu):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) / 1024 if out else 0.0


def session(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[3])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def foreign_procs(gpu, own_sessions):
    """Compute processes on the GPU that are not ours (MPS servers do not count: their clients do)."""
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return ["nvidia-smi failed"]
    procs = []
    for line in out.stdout.strip().splitlines():
        pid, name = (x.strip() for x in line.split(",", 1))
        if "nvidia-cuda-mps-server" in name:
            continue
        if session(int(pid)) not in own_sessions:
            procs.append(f"{pid} {name}")
    return procs


def mps_dir(root, gpu):
    return root / ".mps" / f"gpu{gpu}"


def ensure_mps(root, gpu):
    """One user-level MPS daemon per GPU: many small processes then share the GPU
    concurrently instead of time-slicing (about 3.4x more updates/s at 12 procs/GPU)."""
    d = mps_dir(root, gpu)
    (d / "pipe").mkdir(parents=True, exist_ok=True)
    (d / "log").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(d / "pipe"),
               CUDA_MPS_LOG_DIRECTORY=str(d / "log"), CUDA_VISIBLE_DEVICES=str(gpu))
    alive = subprocess.run("echo get_server_list | nvidia-cuda-mps-control", shell=True, env=env,
                           capture_output=True, text=True, timeout=30).returncode == 0
    if not alive:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
        log(f"started MPS daemon for gpu{gpu}")


def stop_mps(root, gpu):
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(mps_dir(root, gpu) / "pipe"))
    subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True, env=env, timeout=60)


def pid_alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def infer_exit(job):
    """Exit code for a job that ended without writing .exit (launched by an older queue)."""
    logdir = pathlib.Path(job["logdir"])
    if "probe_wm.py" in job["cmd"]:
        return 0 if (logdir / "results.json").exists() else 1
    target = re.search(r"--steps (\S+)", job["cmd"])
    steps = [json.loads(l).get("step", 0) for l in open(logdir / "metrics.jsonl")
             if "eval_return" in l] if (logdir / "metrics.jsonl").exists() else []
    return 0 if target and steps and max(steps) >= float(target.group(1)) else 1


class Handle:
    def __init__(self, job, pid, gpu):
        self.job, self.pid, self.gpu, self.mem = job, pid, gpu, job["mem_gb"]
        self.proc = None

    def poll(self):
        """None while running, else the job's exit code."""
        exit_file = pathlib.Path(self.job["logdir"]) / ".exit"
        if self.proc is not None:
            if self.proc.poll() is None:
                return None
        elif pid_alive(self.pid):
            return None
        if exit_file.exists() and exit_file.read_text().strip():
            return int(exit_file.read_text().strip())
        code = self.proc.returncode if self.proc is not None else infer_exit(self.job)
        if exit_file.parent.exists():  # a stopped job's logdir may have been moved away (2026-10-08)
            exit_file.write_text(str(code))
        return code


def launch(job, gpu, args, root):
    logdir = pathlib.Path(job["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / ".exit").unlink(missing_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONUNBUFFERED="1")
    if not args.no_mps:
        # under MPS the client sees only its daemon's GPU, as device 0
        env.update(CUDA_VISIBLE_DEVICES="0", CUDA_MPS_PIPE_DIRECTORY=str(mps_dir(root, gpu) / "pipe"))
    cmd = job["cmd"].replace("{python}", args.python)
    exit_file = shlex.quote(str(logdir / ".exit"))
    wrapped = f"{cmd}; code=$?; echo $code > {exit_file}.tmp && mv {exit_file}.tmp {exit_file}"
    with open(logdir / "stdout.log", "a") as out:
        proc = subprocess.Popen(["bash", "-c", wrapped], cwd=root, env=env, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    (logdir / ".pid").write_text(str(proc.pid))
    (logdir / ".gpu").write_text(str(gpu))
    handle = Handle(job, proc.pid, gpu)
    handle.proc = proc
    return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    parser.add_argument("--idle_gpus", default="", help="used only while nobody else is on them")
    parser.add_argument("--idle_min", type=float, default=10.0)
    parser.add_argument("--budget_gb", type=float, default=74.0)
    parser.add_argument("--max_per_gpu", type=int, default=10)
    parser.add_argument("--no_mps", action="store_true")
    parser.add_argument("--reserve_gb", type=float, default=4.0)
    parser.add_argument("--poll", type=float, default=20.0)
    parser.add_argument("--cooldown", type=float, default=180.0, help="seconds after a launch before the GPU gets another job")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    idle_gpus = [int(g) for g in args.idle_gpus.split(",") if g]
    idle_since = {g: None for g in idle_gpus}  # since when nobody else is on the GPU
    mps_started = set(gpus)
    root = pathlib.Path(__file__).resolve().parents[1]
    status_file = pathlib.Path(args.jobs).with_suffix(".status")
    running = {}  # name -> Handle
    attempts = {}
    last_launch = {}  # gpu -> time of the last launch there
    if not args.no_mps:
        for g in gpus:
            ensure_mps(root, g)

    # adopt jobs that are still running from a previous queue instance
    for line in open(args.jobs):
        if not line.strip() or line.startswith("#"):
            continue
        job = json.loads(line)
        logdir = pathlib.Path(job["logdir"])
        if (logdir / ".pid").exists() and not (logdir / ".exit").exists():
            pid = int((logdir / ".pid").read_text())
            if pid_alive(pid):
                running[job["name"]] = Handle(job, pid, int((logdir / ".gpu").read_text()))
                attempts[job["name"]] = 1
    log(f"adopted {len(running)} running jobs")

    while True:
        jobs = [json.loads(l) for l in open(args.jobs) if l.strip() and not l.startswith("#")]
        for name, handle in list(running.items()):
            code = handle.poll()
            if code is not None:
                log(f"finished {name} gpu{handle.gpu} exit={code}")
                del running[name]
        own = {h.pid for h in running.values()}
        usable = []
        for g in idle_gpus:
            others = foreign_procs(g, own)
            if others:
                if idle_since[g] is not None:
                    log(f"gpu{g}: other processes again ({others[0]}), no new jobs there")
                idle_since[g] = None
            else:
                if idle_since[g] is None:
                    idle_since[g] = time.time()
                    log(f"gpu{g}: nobody else on it, usable after {args.idle_min:.0f} min")
                if time.time() - idle_since[g] >= 60 * args.idle_min:
                    usable.append(g)
        pending = []
        for job in jobs:
            if job["name"] in running:
                continue
            exit_file = pathlib.Path(job["logdir"]) / ".exit"
            if exit_file.exists():
                code = exit_file.read_text().strip()
                if code == "0" or attempts.get(job["name"], 0) >= 2:
                    continue
            pending.append(job)

        reserved = None  # GPU held back for the first job that does not fit
        for job in pending:
            # adopted jobs may run on GPUs no longer in --gpus: counted, but nothing new is started there
            on = set(gpus + idle_gpus) | {h.gpu for h in running.values()}
            used = {g: 0.0 for g in on}
            count = {g: 0 for g in on}
            for h in running.values():
                used[h.gpu] += h.mem
                count[h.gpu] += 1
            choice = None
            pin = job.get("gpus")
            for g in sorted([g for g in gpus + usable if pin is None or g in pin], key=lambda g: used[g]):
                if g == reserved or time.time() - last_launch.get(g, 0.0) < args.cooldown:
                    continue
                if count[g] >= args.max_per_gpu or used[g] + job["mem_gb"] > args.budget_gb:
                    continue
                if gpu_free_gb(g) < job["mem_gb"] + args.reserve_gb:
                    continue  # other users' processes appeared on this GPU
                choice = g
                break
            if choice is None:
                if reserved is None:
                    keep = [g for g in gpus if pin is None or g in pin] or gpus
                    reserved = max(keep, key=lambda g: args.budget_gb - used[g])
                continue
            if not args.no_mps and choice not in mps_started:
                ensure_mps(root, choice)
                mps_started.add(choice)
            running[job["name"]] = launch(job, choice, args, root)
            last_launch[choice] = time.time()
            attempts[job["name"]] = attempts.get(job["name"], 0) + 1
            log(f"launched {job['name']} on gpu{choice}")
            time.sleep(2)

        done = sum(1 for j in jobs if (pathlib.Path(j["logdir"]) / ".exit").exists()
                   and (pathlib.Path(j["logdir"]) / ".exit").read_text().strip() == "0")
        tmp = status_file.with_suffix(".status.tmp")
        tmp.write_text(json.dumps({
            "time": time.strftime("%F %T"), "total": len(jobs), "done": done,
            "pending": len(pending) - sum(1 for j in pending if j["name"] in running),
            "running": {n: f"gpu{h.gpu}" for n, h in running.items()},
        }, indent=1))
        tmp.replace(status_file)
        if not running and not pending:
            log("queue empty")
            if not args.no_mps:
                for g in mps_started:
                    stop_mps(root, g)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
