# Fire Without Forgetting

面向记忆型具身控制的持久态脉冲世界模型。当前仓库记录**正在进行的研究、完整模型实现和单种子阶段实验**，不是已完成论文的结果发布。

**当前结论：Phase 0 的因果验证尚未通过；本方法核心已实现，但尚无性能或能效优势证据。** 原研究方案中的预期结果不是已取得结果。按用户要求，所有训练条件只用 **seed 0**。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 原始研究问题、假设、E1–E5 | [研究方案](docs/snn_world_model_research_plan.md) |
| 原计划与实际执行的逐项对应 | [设计对照](docs/plan_alignment_20260928.md) |
| 当前结论、已完成结果与限制 | [结果解读](docs/RESULTS.md) |
| 全部运行、评估曲线、训练标量和失败记录 | [结果快照](results/README.md) · [机器可读汇总](results/summary.json) |
| 模型版本、代码入口和修复来源 | [代码地图](docs/CODE_MAP.md) |
| 依赖环境、运行命令和计步约定 | [复现与环境](docs/REPRODUCING.md) |
| 历史诊断、实施协议和巡检 | [文档索引](docs/README.md) |
| 发布包含什么、保留什么 | [快照说明](docs/REPOSITORY_SNAPSHOT.md) |

## 对照 E1–E5 的状态

| 实验 | 当前状态 | 尚未得到的证据 |
|---|---|---|
| E1：记忆长度诊断 | POPGym 部分基线与本方法已训练；BSuite 长度4/30/100子集已启动 | 原MCN及所有SNN的完整退化曲线、Autoencode等任务 |
| E2：两级复位因果干预 | 完成Medium、τ=2、100k四格初筛，四组均接近随机 | 完整τ/阈值扫描、恢复记忆的因果证据、DMC对照 |
| E3：新方法主结果 | 持久态SSM向量观测版本已实现；当前Easy学习效果较弱 | 同预算完整对照、接近R2I或优于SNN、Memory Maze视觉结果 |
| E4：训练与想象一致性 | 已训练模型在同输入路径下的扫描/逐步数值一致性通过 | SDN/masked PSN近似对照及完整长想象误差比较 |
| E5：效率 | 已有耗时、显存、脉冲率和部分资源采样 | 完整AC/MAC/状态访存账及性能–能耗优势 |

R2I/DreamerV3是ANN参照；修正R2I的episode边界错误，不等于验证MCN的步间复位/树突发放复位假设。不同任务、训练预算、回放机制、精度和计步方式在结果中分别记录，不把它们当成严格匹配的单因素比较。

## 本方法

当前训练版本：[persistent-ssm-longseq-v3](code/persistent-ssm-longseq-v3/)，核心实现为 [persistent_ssm.py](code/persistent-ssm-longseq-v3/persistent_ssm.py)。

- T=1，神经元时间与环境时间对齐。
- 三层稳定复数对角递推，树突跨环境步保留；仅在episode起点清零。
- 胞体读出不回写或复位树突；层间传递二值脉冲。
- 观测独立后验 q(z|o)，32×32 WTA离散潜变量。
- 可微仿射扫描训练与相同的一步递推；完整重建、奖励、继续预测及Actor-Critic。
- 实现另用了胞体PopNorm与时间尺度输入归一化，具体差异见执行协议。

已有验证针对向量观测任务；视觉训练、近似胞体对照与全部消融未完成。v1的T=1零发放编码器问题已有诊断并排除，原始失败和修正版代码同时保留。

## 仓库结构

~~~text
code/       实际运行的不可变模型快照，含历史及失败版本
sources/    上游源码及许可证；不包含嵌套Git历史
configs/    实际实验队列、预算和seed策略
scripts/    环境准备、调度、诊断、探针与结果导出
envs/       requirements和本次采集的完整freeze文件
docs/       研究方案、执行协议、诊断证据和历史巡检
results/    有时间戳的结果快照、每次评估和训练标量
data/       数据资产状态说明，不包含大型数据集
~~~

运行期间的模型权重、优化器、replay和完整训练流水保存在42号机 /data/Minko/fire-without-forgetting/runs/，未纳入Git。Memory Maze离线数据尚待用户提供；模型从头训练，不依赖预训练权重。

所有项目操作按现行约定先SSH到 New-H100-2，再 cd /data/Minko。仅使用获授权的GPU0–3，不使用41号机。发布分支为 **codex**；服务器调度和训练独立于这份冻结快照继续运行。

第三方源码及许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。上游目录内的论文分数与资源不是本项目新实验结果。
