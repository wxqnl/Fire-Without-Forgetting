# 历史README：2026-09-27初期状态

本文件仅保留历史记录；最新状态见仓库根README及docs/RESULTS.md。

# Fire Without Forgetting

研究方案原件：`docs/snn_world_model_research_plan.md`。本目录运行于 **New-H100-2（42），/data/Minko/fire-without-forgetting**。

当前阶段：v2 实验已启动。GPU 0/1 为官方 R2I Easy/Medium 学习对照，GPU 2 为 MCN Easy，GPU 3 为已实现的持久态脉冲 SSM Medium。全部 seed 0，首轮各 1M 训练器计步，R2I reset 计数差异见协议。每小时 Codex heartbeat 检查并推进，调度器在空卡释放后自动接续已准备任务。

当前协议：[experiment_protocol_v2.md](../experiment_protocol_v2.md)；状态：`python3 scripts/status_v2.py`；队列：`configs/experiments_v2.json`、`runs/queue_v2_status.json`；tmux：`fwf-v2`。

本方法首版在 `code/persistent-ssm-v2`：T=1、观测独立后验、32×32 WTA latent、3 层持久复数树突状态、精确仿射扫描和逐步想象、完整世界模型与 Actor-Critic。修正版已通过真实回放上的状态/梯度一致性、四类观测可区分、编码器梯度非零及完整更新检查，见 [persistent_ssm_v2_integration.json](../persistent_ssm_v2_integration.json)。初版在 T=1 下编码器不发放，已排除并保留失败记录；其训练后 E4 数据仅作调试记录。v2 的训练后 E4 和性能结论仍需后续检查点。

先前四组 100k 初筛的确定性后验探针显示 4 步线索差异（原版 28.3%，取消步间复位 42.2%）；按实际后验采样时，4 步读出均回到约 25%。两种结果必须分开解读：`docs/probes/` 是确定性补充，`docs/probes_stochastic/` 是实际采样口径。尚未验证在线记忆恢复。

## 首批运行记录

首批 phase0_initial 调度已完成；以下为历史记录。当前 v2 作业见上方。使用过 GPU **0、1、2、3**；入口 `scripts/run_phase0.py`；任务清单 `configs/phase0_initial.json`。

- 真实 POPGym RepeatPrevious **Medium**，2×2 复位组合，每格只跑 seed 0，共 4 次（用户要求优先节省时间）。
- 每次预算 100,000 环境步，包含 2,500 步预填充；每 10,000 步评估 20 回合；训练与评估环境使用分离的随机种子。
- 原版 MCN 宽度 512，32×32 离散隐变量，T=5，batch=16×128，想象长度 15，FP32；完整世界模型、重建/奖励/继续预测与 Actor-Critic。
- train_ratio=64（参考 R2I POPGym 的回放比率），其余核心网络规模保留 Spiking-WM 默认值。原版图像任务默认 train_ratio=512，不应把本配置写成未经适配的论文复现。
- 首批固定 τ=2、θ=0.5，目的是建立学习曲线和运行成本；10 万步不是最终论文预算。不能用未收敛的筛查否定研究假设。后续 E2 仍需 τ=2/8/32、阈值、BSuite 长度扫描和 DMC 对照。
- 使用 GPU 0、1、2、3 并行运行四种配置，每种配置只跑 seed 0。每次启动前确认该卡空闲；每次最多 12 小时。失败后停止新增任务，已启动的任务继续并保存结果。

查看状态（从本机执行）：

```bash
ssh New-H100-2 'cd /data/Minko && bash fire-without-forgetting/scripts/status.sh'
```

进入后台会话：

```bash
ssh -t New-H100-2 'cd /data/Minko && tmux attach -t fwf-phase0-multi'
```

运行输出在 `runs/phase0_initial/<实验名>/`，包含 `train.log`、`metrics.jsonl`、TensorBoard、完整配置、训练/评估轨迹。每个评估区间后保存 `latest_model.pt`，`models/` 对应链接指向该文件。首次保存前链接目标尚不存在。完成状态在 `result.json`。

原版检查点不保存优化器/完整采样状态；队列拒绝静默接着中断的检查点跑，以免把它当成无中断实验。需要重跑时保留旧输出，使用新的运行名。

## 源码与环境

| 内容 | 位置 | 状态 |
|---|---|---|
| [Spiking-WM](https://github.com/Brain-Cog-Lab/Spiking-WM) 官方源码 | sources/Spiking-WM | 下载保留 |
| 适配与干预实现 | code/spiking-wm | 首批训练使用 |
| [Recall2Imagine](https://github.com/chandar-lab/Recall2Imagine) | sources/Recall2Imagine | 官方代码；模型导入及 CUDA 设备识别通过，尚未训练 |
| [DreamerV3](https://github.com/danijar/dreamerv3) | sources/dreamerv3 | 官方代码；模型导入及 CUDA 设备识别通过，尚未训练 |
| [BSuite](https://github.com/google-deepmind/bsuite)、[POPGym](https://github.com/proroklab/popgym)、[Memory Maze](https://github.com/jurgisp/memory-maze) | sources/ 对应目录 | 官方代码；运行版以环境冻结列表为准 |
| PyTorch | envs/torch | Python 3.10；torch 2.4.1+cu121 |
| R2I | envs/r2i | Python 3.10；JAX 0.4.23、CUDA 12、cuDNN 8 |
| DreamerV3 | envs/dreamerv3 | Python 3.10；JAX 0.4.33、CUDA 12、cuDNN 9 |

环境相互独立，没有改动其他项目的虚拟环境。安装依赖输入为 `envs/*-requirements.txt`，实际完整版本为 `envs/*-freeze.txt`。`scripts/setup_*.sh` 可重建对应环境。R2I 使用 Python 3.10 兼容环境，并固定历史 JAX/Flax/Haiku；不是其原始 Python 3.8 容器。当前 JAX 准备包括任务所需依赖，没有安装 Atari ROM/Minecraft 等无关资源。

官方仓库以分支、下载日期及源码日期记录在 `docs/sources.json`。本项目未使用提交散列作运行标识。不要修改正在运行的代码；新修改应留到当前实验边界或单独版本后再排队。

## 数据与模型

BSuite、POPGym、DMC 为在线交互任务，训练数据保存在各运行的 replay 轨迹中；模型从头训练，无预训练权重依赖。10 个向量任务完成真实环境回合，DMC Walker 和 Memory Maze 9×9 完成真实 64×64 渲染，记录在 `docs/*benchmark_installation.json`。

Memory Maze **离线** 9×9 的 30,000 条轨迹尚未下载。官方 Google Drive 链接从服务器超时，已记录在 `data/assets.json`，魔搭公开数据集搜索接口可以访问；检索 memory-maze、memory_maze、MemoryMaze、9x9、jurgisp、memory maze 以及 maze，没有发现对应离线集合，结果记录在 `docs/modelscope_search.json`。用户已接手离线数据下载，收到服务器路径后接入。在线环境可用不代表离线数据已备齐。这不影响当前 E1/E2 的 POPGym/BSuite 实验。官方数据说明见 [Memory Maze](https://github.com/jurgisp/memory-maze#offline-dataset)。

## 实现范围与检查

`docs/spiking-wm-changes.patch` 为相对官方代码的全部 Python/YAML 改动。

1. 为原版添加官方 POPGym/Gymnasium、BSuite/dm_env 适配器；只给模型可观测量，不提供隐藏状态。
2. 修复原版向量编码器的配置访问、脉冲时间维返回，以及无图像观测时的初始化/预处理/日志；离散策略 REINFORCE 的 value 切片改为环境时间维。
3. 所有组合共享上述接口修复；`original` 表示原始 MCN 的两级复位设置。
4. A 干预将 basal/apical/soma 放进显式 RSSM state，在 episode 起点按 batch 掩码归零，在序列内通过 BPTT 保留梯度；交互、回放及想象均从传入的 state 恢复，防止模块缓存串扰。其他前馈 LIF 的复位保持一致。
5. B 干预只控制树突随发放复位，胞体仍复位。τ 扫描固定代理梯度 alpha=2，避免原实现 alpha=τ 带来的混杂。
6. 修复回放切片对原始 is_first 标志的原地写入，以及到达预算后额外训练一个区间的问题。
7. 四格均已在真实 POPGym 回放上完成完整世界模型和 Actor-Critic 更新，损失/梯度有限，并检查持久态在不同想象分支间不串扰；记录 `docs/phase0_integration.json`。这是运行检查，不能作为记忆能力的实验结果。

持久态 SSM 首版及 RepeatPrevious 线性记忆探针已实现；LIF/aLIF/TC-LIF/PSN 全部同框架对照、Memory Maze 离线探针和完整效率评价仍待后续阶段。用户要求维持单种子，不报告跨训练种子的 IQM/置信区间。当前公开 Spiking-WM 的 node.py 只包含 LIF/MCN，并没有文档所说的全部替换实现。完整 RSSM 还包含随机 latent，文中“只有 deter 承载跨步信息”的说法应在后续论证时细化。

2026-09-27：按用户要求取消所有 seed 1–4 的重复实验；后续默认单种子，只有用户明确要求才增加种子。首个训练进程由新调度器原位接管，参数更新未中断；旧调度器暂停并在当前训练完成后退出。队列在每次任务边界重读任务清单。

首批分配：GPU 0 原版，GPU 1 取消步间复位，GPU 2 两种复位都取消，GPU 3 取消树突发放复位。全部使用 seed 0，现均已完成。
