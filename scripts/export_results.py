#!/usr/bin/env python3
"""Export a dated, lightweight record of real runs without loading GPU models."""
import argparse
import datetime
import json
import math
import shutil
from pathlib import Path

def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def arg_value(command, *keys):
    for key in keys:
        if key in command:
            return command[command.index(key) + 1]
    return None

def export(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise SystemExit("Use a new results destination to preserve previous exports.")
    destination.mkdir(parents=True)
    started = datetime.datetime.now().astimezone().isoformat()
    active = {}
    queues = {}
    for file in sorted((source / "runs").glob("queue*status.json")):
        queue = read_json(file)
        queues[file.name] = queue
        for job in queue.get("active", []):
            active[job["job"]] = job
        write_json(destination / "queues" / file.name, queue)

    summaries = []
    metadata_names = (
        "command.json", "job.json", "result.json", "resolved_config.json",
        "config.yaml", "model_summary.json", "final_checkpoint.json", "exclusion.json",
    )
    evaluation_keys = (
        "step", "train_action_steps", "updates", "update_count", "eval_return",
        "eval_return_std", "eval_length", "eval_episodes", "elapsed_seconds",
    )
    for group in ("phase0_initial", "v2"):
        group_dir = source / "runs" / group
        for run in sorted(x for x in group_dir.iterdir() if x.is_dir()):
            if not any((run / f).exists() for f in ("job.json", "command.json", "result.json", "metrics.jsonl")):
                continue
            out = destination / "runs" / group / run.name
            out.mkdir(parents=True)
            for name in metadata_names:
                if (run / name).is_file():
                    shutil.copy2(run / name, out / name)
            result = read_json(run / "result.json", {})
            exclusion = read_json(run / "exclusion.json")
            job = read_json(run / "job.json", {})
            command = result.get("command") or job.get("command") or read_json(run / "command.json", [])
            if isinstance(command, dict):
                command = command.get("command", [])
            config = read_json(run / "resolved_config.json", {})
            status = ("excluded" if exclusion else
                      "completed" if result.get("returncode") == 0 else
                      "failed" if "returncode" in result else
                      "running" if run.name in active else "unresolved")
            scope = ("integration" if "integration" in run.name else
                     "preliminary_100k" if group == "phase0_initial" else "bounded_stage")
            rows, malformed = [], 0
            metrics = run / "metrics.jsonl"
            source_size = metrics.stat().st_size if metrics.exists() else 0
            if metrics.exists():
                with metrics.open("rb") as stream:
                    payload = stream.read(source_size).decode("utf-8", errors="replace")
                for line in payload.splitlines():
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed += 1
            evals = [{k: row[k] for k in evaluation_keys if k in row}
                     for row in rows if "eval_return" in row]
            training = []
            bad = []
            for row in rows:
                if not any("loss" in key or "grad_norm" in key or key in ("model_grad_norm", "actor_grad_norm")
                           for key in row):
                    continue
                scalar = {key: val for key, val in row.items() if isinstance(val, (int, float, bool, str))}
                nonfinite = [key for key, val in scalar.items()
                             if isinstance(val, (int, float)) and not math.isfinite(val)]
                if nonfinite:
                    scalar["nonfinite_metric_names"] = nonfinite
                    bad.append({"step": row.get("step"), "metrics": nonfinite})
                training.append(clean(scalar))
            for name, records in (("evaluations.jsonl", evals), ("training_scalars.jsonl", training)):
                (out / name).write_text("".join(json.dumps(clean(x), ensure_ascii=False, allow_nan=False) + "\n" for x in records))
            if (run / "evaluations.jsonl").exists():
                shutil.copy2(run / "evaluations.jsonl", out / "evaluation_episodes.jsonl")
            log = run / "train.log"
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 128 * 1024))
                    tail = stream.read().decode("utf-8", errors="replace")
                (out / "log_tail.txt").write_text("\n".join(tail.splitlines()[-80:]) + "\n")
                if status in ("failed", "excluded") and log.stat().st_size < 2 * 1024**2:
                    shutil.copy2(log, out / "failure.log")
            step = max((row.get("step", 0) for row in rows), default=0)
            action_records = [row["train_action_steps"] for row in rows if "train_action_steps" in row]
            torch_run = group == "phase0_initial" or run.name.startswith(("mcn_", "persistent_ssm"))
            action_steps = max(action_records) if action_records else (step if torch_run else None)
            updates = max((val for row in rows for key, val in row.items()
                           if key in ("update_count", "updates")), default=0)
            budget = arg_value(command, "--run.steps", "--steps") or config.get("steps")
            code_path = next((x for x in command if isinstance(x, str) and "/code/" in x and x.endswith(".py")), None)
            code = code_path.split("/code/", 1)[1].split("/", 1)[0] if code_path else None
            last = evals[-1] if evals else None
            item = dict(
                run=run.name, group=group, status=status, scope=scope,
                seed=arg_value(command, "--seed") or config.get("seed", 0),
                task=arg_value(command, "--task") or config.get("task"), code_snapshot=code,
                configured_budget=int(budget) if budget is not None else None,
                budget_unit="actual_actions" if torch_run or action_records else "driver_transitions_including_reset",
                latest_logged_step=step, latest_logged_action_steps=action_steps,
                latest_logged_updates=updates, final_or_latest_evaluation=last,
                evaluation_points=len(evals), returncode=result.get("returncode"),
                elapsed_seconds=result.get("elapsed_seconds"), ended_at=result.get("ended_at"),
                exclusion=exclusion, final_checkpoint_metadata=read_json(run / "final_checkpoint.json"),
                metric_source_bytes=source_size, metric_source_records=len(rows),
                incomplete_or_malformed_lines=malformed, training_scalar_records=len(training),
                nonfinite_training_records=bad,
                exported_at=datetime.datetime.now().astimezone().isoformat(),
                source_run_directory=str(run), exported_directory=str(out.relative_to(destination)),
                gpu=active.get(run.name, {}).get("gpu", result.get("gpu")),
            )
            if run.name in ("r2i_repeat_easy_s0_cuda", "r2i_repeat_medium_s0_cuda"):
                item["checkpoint_limitation"] = "Final evaluation reached 1M; final async save did not finish. Actual saved weights: Easy949312, Medium926744."
            if run.name.startswith("mcn_") or group == "phase0_initial":
                item["checkpoint_limitation"] = "Weights only; no complete optimizer state for exact continuation."
            write_json(out / "summary.json", item)
            summaries.append(item)

    resources = source / "runs/resource_samples.jsonl"
    if resources.exists():
        size = resources.stat().st_size
        with resources.open("rb") as stream:
            data = stream.read(size)
        if data and not data.endswith(b"\n"):
            data = data.rsplit(b"\n", 1)[0] + b"\n"
        (destination / "resource_samples.jsonl").write_bytes(data)
    summary = dict(
        captured_at=started, export_finished_at=datetime.datetime.now().astimezone().isoformat(),
        source_root=str(source), seed_policy="seed 0 only",
        scope_note="Bounded-stage snapshot. Integrations, preliminary runs, exclusions and failures remain distinct. Running results are interim.",
        raw_artifacts_note="Checkpoints, replay, virtual environments and full per-episode logs stay on the source server. All logged evaluation points and loss-bearing scalar records are included.",
        nonfinite_note="Nonfinite scalars are JSON null with their metric names retained. R2I empty-subset pos/neg diagnostics may legitimately be nonfinite.",
        runs=summaries,
    )
    write_json(destination / "summary.json", summary)
    lines = ["# 实验结果快照", "", "采集开始：" + started, "",
             "每个条件只跑 seed 0。已完成和仍在运行的结果分开列出；这是阶段实验记录，不是完整论文结论。", "",
             "计步：Torch 和 BSuite 新适配按实际动作计；早期 R2I / Dreamer POPGym 按含 reset 首帧的 driver transition 计。不同预算/框架不能直接当成严格匹配比较。", ""]
    for title, select in (
        ("已完成阶段实验", lambda x: x["status"] == "completed" and x["scope"] == "bounded_stage"),
        ("早期 100k 四格初筛", lambda x: x["scope"] == "preliminary_100k"),
        ("正在运行：临时结果", lambda x: x["status"] == "running"),
        ("完整训练集成：不作为性能结果", lambda x: x["scope"] == "integration" and x["status"] == "completed"),
        ("失败、排除或未确认", lambda x: x["status"] in ("failed", "excluded", "unresolved")),
    ):
        lines += ["## " + title, "", "|运行|状态|最新计步|计步口径|最终/最近回报|评估回合|", "|---|---|---:|---|---:|---:|"]
        for item in filter(select, summaries):
            ev = item["final_or_latest_evaluation"] or {}
            score = f'{ev["eval_return"]:.6f}' if "eval_return" in ev else "—"
            step = item["latest_logged_action_steps"] if item["budget_unit"] == "actual_actions" else item["latest_logged_step"]
            link = item["exported_directory"] + "/summary.json"
            lines.append(f'|[{item["run"]}]({link})|{item["status"]}|{int(step or 0):,}|{item["budget_unit"]}|{score}|{int(ev.get("eval_episodes", 0)) or "—"}|')
        lines.append("")
    lines += ["## 阅读方式", "", "每个运行目录保留原始配置、命令、退出/排除记录，另有全部 evaluations.jsonl、training_scalars.jsonl 与日志末尾。Dreamer 的逐回合评估明细在 evaluation_episodes.jsonl。无凭据、权重或 replay。", "",
              "源代码/环境审计、最终检查点的独立读回证据及 E1–E5 对照见 ../docs/README.md。resource_samples.jsonl 只覆盖采样启动后的时段，共卡功率不能归于单个模型。"]
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"destination": str(destination), "captured_at": started, "runs": len(summaries),
                      "states": {s: sum(x["status"] == s for x in summaries) for s in sorted({x["status"] for x in summaries})},
                      "evaluation_points": sum(x["evaluation_points"] for x in summaries)}, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination)
