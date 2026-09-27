"""Minimal GPU job queue for the Phase 0 grid.

Jobs file: one JSON object per line: {"name", "cmd", "logdir", "mem_gb"}.
The file is re-read every poll, so jobs can be appended while the queue runs.
A job is skipped when <logdir>/.exit holds 0; failed jobs are retried once.

usage: nohup python scripts/queue.py --jobs runs/jobs.jsonl --gpus 1,2,3,4,5,6 &
"""
import argparse
import json
import os
import pathlib
import subprocess
import time


def gpu_free_gb(gpu):
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) / 1024 if out else 0.0


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
        print(time.strftime("%F %T"), f"started MPS daemon for gpu{gpu}", flush=True)


def stop_mps(root, gpu):
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(mps_dir(root, gpu) / "pipe"))
    subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True, env=env, timeout=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    parser.add_argument("--budget_gb", type=float, default=74.0)
    parser.add_argument("--max_per_gpu", type=int, default=10)
    parser.add_argument("--no_mps", action="store_true")
    parser.add_argument("--reserve_gb", type=float, default=4.0)
    parser.add_argument("--poll", type=float, default=20.0)
    parser.add_argument("--python", default="/home/user01/miniconda3/envs/fwf/bin/python")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    root = pathlib.Path(__file__).resolve().parents[1]
    status_file = pathlib.Path(args.jobs).with_suffix(".status")
    running = {}  # name -> (proc, gpu, mem, job)
    attempts = {}
    if not args.no_mps:
        for g in gpus:
            ensure_mps(root, g)

    while True:
        jobs = [json.loads(l) for l in open(args.jobs) if l.strip() and not l.startswith("#")]
        # reap finished jobs
        for name, (proc, gpu, mem, job) in list(running.items()):
            code = proc.poll()
            if code is not None:
                (pathlib.Path(job["logdir"]) / ".exit").write_text(str(code))
                print(time.strftime("%F %T"), f"finished {name} gpu{gpu} exit={code}", flush=True)
                del running[name]
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
        # launch while capacity allows (jobs keep file order = priority)
        for job in pending:
            used = {g: 0.0 for g in gpus}
            count = {g: 0 for g in gpus}
            for proc, gpu, mem, _ in running.values():
                used[gpu] += mem
                count[gpu] += 1
            choice = None
            for g in sorted(gpus, key=lambda g: used[g]):
                if count[g] >= args.max_per_gpu or used[g] + job["mem_gb"] > args.budget_gb:
                    continue
                # protect against other users' processes appearing on our GPUs
                if gpu_free_gb(g) < job["mem_gb"] + args.reserve_gb:
                    continue
                choice = g
                break
            if choice is None:
                break
            logdir = pathlib.Path(job["logdir"])
            logdir.mkdir(parents=True, exist_ok=True)
            (logdir / ".exit").unlink(missing_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(choice), OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       PYTHONUNBUFFERED="1")
            if not args.no_mps:
                # under MPS the client sees only its daemon's GPU, as device 0
                env.update(CUDA_VISIBLE_DEVICES="0",
                           CUDA_MPS_PIPE_DIRECTORY=str(mps_dir(root, choice) / "pipe"))
            cmd = job["cmd"].replace("{python}", args.python)
            with open(logdir / "stdout.log", "a") as out:
                proc = subprocess.Popen(cmd, shell=True, cwd=root, env=env, stdout=out,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            attempts[job["name"]] = attempts.get(job["name"], 0) + 1
            running[job["name"]] = (proc, choice, job["mem_gb"], job)
            print(time.strftime("%F %T"), f"launched {job['name']} on gpu{choice}", flush=True)
            time.sleep(2)
        done = sum(1 for j in jobs if (pathlib.Path(j["logdir"]) / ".exit").exists()
                   and (pathlib.Path(j["logdir"]) / ".exit").read_text().strip() == "0")
        status_file.write_text(json.dumps({
            "time": time.strftime("%F %T"), "total": len(jobs), "done": done,
            "running": {n: f"gpu{g}" for n, (_, g, _, _) in running.items()},
        }, indent=1))
        if not running and not pending:
            print(time.strftime("%F %T"), "queue empty", flush=True)
            if not args.no_mps:
                for g in gpus:
                    stop_mps(root, g)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
