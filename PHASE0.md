# Phase 0 实验记录：复位失忆的因果检验（E2）

对应研究方案 `../snn_world_model_research_plan.md` 的 §5.1 E2、§9 Phase 0。代码基于 `Brain-Cog-Lab/Spiking-WM` commit `602a7c8`，改动在分支 `fwf-phase0`。

启动时间：2026-09-27 09:20，GPU 1–6（H100 80GB）。

## 1. 要回答的问题

判定门：干预 A、B 是否带来记忆提升（Pr2、Pr3）。同时检验：

- Pr1：原模型在记忆任务上对 τ 不敏感；
- Pr3 的后半句：去掉复位后，τ 开始显著影响性能。

## 2. 实验设置

### 2.1 干预（2×2）

| 条件 | 配置块 | 含义 |
|---|---|---|
| base | – | 原始 Spiking-WM：每个环境步开头清零循环单元的膜电位；胞体发放时树突一起复位 |
| A | `persist` | 循环单元（MCRNN）的三个区室膜电位写入 RSSM 的 state 字典，跨环境步保留，只在 `is_first` 时清零 |
| B | `nodendreset` | 胞体发放时只复位胞体，basal、apical 不复位 |
| AB | `persist nodendreset` | 两者都做 |

与 τ_d 交叉：τ_d = τ_basal = τ_apical ∈ {2, 16, 128}（配置块 `taud16`、`taud128`）。胞体 τ 固定为 2。

按欧拉更新、每个环境步 T=5 个 SNN 步计算，树突在一个环境步内保留的比例为 (1 − 1/τ_d)^5：

| τ_d | 每个环境步保留的比例 | 等效记忆（环境步） |
|---|---|---|
| 2 | 0.03 | ≈ 0 |
| 16 | 0.72 | ≈ 3.6 |
| 128 | 0.96 | ≈ 25 |

干预只作用于循环单元。前馈层（输入层、输出层、编码器、各预测头）里的 LIF 神经元仍在每次调用时复位，与原实现一致。

### 2.2 两类实验

**(a) 世界模型记忆探针（`probe_wm.py`）**：新增，是对方案 E1/E2 的补充。

- 数据：随机策略采集，训练 20 万步，测试 512 条完整 episode。同一任务的所有条件、种子用同一份数据（data_seed 12345）。
- 训练：只训练世界模型，20k 次更新，batch 16×64，其余超参与原论文相同。
- 指标：在测试 episode 上，从 episode 起点开始滤波，计算奖励预测对正、负奖励步的 AUC。
- 为什么有效：这些任务里，奖励取决于 k 步前的线索和当前动作。没有记忆的模型 AUC 只能是 0.5，所以 AUC 直接读出世界模型能记多远。这样也排除了策略探索这个混杂因素。
- 任务：BSuite memory_len（L = 8 / 16 / 32）、POPGym RepeatPreviousEasy（k=4）。
- 规模：4 条件 × 3 个 τ_d × 2 个种子。

**(b) RL（`dreamer.py`，E2 本体）**

- 任务：BSuite memlen-16（20 万步）、POPGym RepeatPreviousEasy（50 万步）。
- 规模：4 条件 × τ_d ∈ {2, 128} × 3 个种子。
- 指标：评估回报。POPGym 每 2 万步评估 16 条 episode；BSuite 每 1 万步评估 32 条。

### 2.3 与原方案、原论文设置的偏差

| 项 | 本实验 | 原因 |
|---|---|---|
| train_ratio（RL） | 64（原论文 DMC 为 512） | 实测每次更新约 1.3–1.5 秒（瓶颈在 CPU 端调度大量小 kernel）。512 时 10 万环境步需要 18 小时，网格跑不完。四个条件用同一设置，因果比较不受影响 |
| τ 扫描 | 只扫树突 τ_d，胞体 τ 固定为 2 | 假设说的是树突记忆；同时改胞体 τ 会改变发放动力学，引入混杂 |
| 代理梯度宽度 | 固定 sg_alpha = 2 | 原代码 MCNode 用 `alpha=tau`，扫 τ 时代理梯度宽度也会跟着变（混杂）。τ=2 时与原实现一致 |
| 种子数 | 探针 2 个、RL 3 个 | 先求出判定门信号；定稿实验按方案补到 5 个及以上 |

### 2.4 顺带修掉的原仓库问题（不影响原论文的 DMC 结果）

1. `MultiEncoder` 的向量观测分支用了 `config[act]`，argparse Namespace 不支持下标访问，会直接报错。另外它返回的是 T 步平均值而不是逐步脉冲列表，和 `torch.stack` 的调用对不上。原论文只用了图像观测，所以没有暴露。
2. `ImagBehavior._compute_actor_loss` 的 reinforce / both 分支里，`self.value(imag_feat[:-1])` 切的是脉冲时间维（T），而不是想象步维。已改为 `imag_feat[:, :-1]`。DMC 用 `dynamics` 梯度，不走这个分支。
3. `_policy`、`preprocess`、`simulate` 默认观测里有 `image` 键，已兼容纯向量环境。
4. `sample_episodes` 的随机种子写死为 0，已改为 `config.seed`。

以上改动都经过单元测试：A 打开时，非首步进入循环单元的膜电位等于上一步结束时的值，`is_first` 时为 0；base 条件下每步从 0 开始。

## 3. 运行方式

```bash
cd fwf
python scripts/make_jobs.py                  # 生成 runs/jobs.jsonl（144 个任务，按优先级排序）
setsid nohup python scripts/queue.py --jobs runs/jobs.jsonl --gpus 1,2,3,4,5,6 > runs/queue.log 2>&1 &
python scripts/summarize.py                  # 汇总到 runs/summary.md 和 runs/plots/
```

- **调度**：每张 GPU 起一个用户级 MPS 守护进程（`.mps/gpu<k>/`），同卡多进程并发执行。实测单卡 12 个进程时总吞吐是不开 MPS 时的 3.4 倍。每张卡最多 10 个进程。
- **显存**：按任务估计值记账（探针 3GB、RL 14GB），单卡预算 74GB。启动前再用 nvidia-smi 检查实际空闲显存，避让其他用户。
- **断点续跑**：
  - 每个任务自己把退出码写入 `<logdir>/.exit`，同时记录 `.pid` 和 `.gpu`。
  - 重启队列时：已成功的任务跳过；仍在运行的任务直接接管，不会重复启动；失败的任务重试一次。
  - RL 任务从 `latest_model.pt` 和已存的 episode 续跑。
- **调度**：队首任务放不下时，给它预留余量最大的那张卡，较小的任务回填其他卡，避免队头阻塞。
- **只停调度器**（任务继续跑，之后重启时会被接管）：`pkill -f scripts/queue.py`。
- **全部停止**：
  1. `pkill -f scripts/queue.py`；
  2. `pkill -f "probe_wm.py|dreamer.py"`；
  3. 对每张卡执行 `CUDA_MPS_PIPE_DIRECTORY=.mps/gpu<k>/pipe bash -c 'echo quit | nvidia-cuda-mps-control'`。
- **加任务**：编辑 `scripts/make_jobs.py` 后重新运行。它用原子方式重写 `runs/jobs.jsonl`，正在运行的队列下一轮会自动读到。

### 2.5 阳性对照

`grucell`：循环核换成连续状态的 GRU，编码器、预测头仍然是脉冲网络。它在四个探针任务上各跑 2 个种子，在两个 RL 任务上各跑 3 个种子。

用途：如果 GRU 核也学不会某个任务，那个任务上 SNN 的失败就不能归因于复位，只能说明训练预算或其他模块有问题。

## 4. 结果

（运行中；用 `scripts/summarize.py` 更新 `runs/summary.md`）
