"""Extend the Memory Maze runs from 200k to the paper's 1M updates (2026-10-01, EXPERIMENTS.md §15).

The runs were started with --mm_updates 200000; jobs.jsonl now has 1000000. Whenever one of them
has finished (.exit 0, results.json at fewer updates than its job now asks for), keep its result
as results_<N>k.json and remove .exit: the queue then launches the job again, and probe_mm.py
resumes from model.pt. Exits once every Memory Maze job has been handed back to the queue.

usage: setsid nohup python scripts/extend_mm.py >> runs/extend_mm.log 2>&1 &
"""
import json
import pathlib
import re
import shutil
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    while True:
        jobs = [json.loads(l) for l in (ROOT / "runs" / "jobs.jsonl").open() if l.strip()]
        jobs = [j for j in jobs if j["name"].startswith("mmaze_official/")]
        waiting = 0
        for job in jobs:
            logdir = pathlib.Path(job["logdir"])
            target = int(re.search(r"--mm_updates (\d+)", job["cmd"]).group(1))
            exit_file, results = logdir / ".exit", logdir / "results.json"
            if not results.exists():
                waiting += 1  # still running towards its first budget
                continue
            step = json.loads(results.read_text())["step"]
            if step >= target or not exit_file.exists():
                continue  # done, or already handed back to the queue
            if exit_file.read_text().strip() != "0":
                waiting += 1
                continue
            keep = logdir / f"results_{step // 1000}k.json"
            if not keep.exists():
                shutil.copy(results, keep)
            exit_file.unlink()
            print(time.strftime("%F %T"), f"{job['name']}: kept {keep.name}, back to the queue for {target}",
                  flush=True)
        if waiting == 0:
            print(time.strftime("%F %T"), "all Memory Maze runs handed back", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
