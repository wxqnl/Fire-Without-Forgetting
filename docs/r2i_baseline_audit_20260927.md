# R2I 学习地板审计与资源安排：2026-09-27

## 当前证据

此前 R2I Easy/Medium 的 1M 运行回报 -0.5167/-0.4958，没有完成正对照校准。实际落盘检查点分别在 949,312/926,744 步，不能称作 1M 最终模型。

虽然模型和损失代码来自官方仓库，但运行条件被我们改成 batch 16×128、Uniform 1M replay、train_eval、FP32、prefill 2500。这不是已经复现成功的官方训练通路。公开 POPGym 配置是 4×1024、FIFO_LFS 10M、train 入口、8 个环境、FP16，train 入口的预填充至少 32768。原论文 POPGym 学习曲线展示到 50M，而配置文件的上限是 200M；两者不应混为一谈，1M 负结果不足以确定最终能力。来源：https://arxiv.org/pdf/2403.04253，附录 E/H/O。

## 已确认的实现不一致

在实际保存的 Easy 回放和 949312 步权重上，固定相同潜变量/输入路径，比较官方并行 SSM 与逐步递推：
- 不含内部 episode reset 时，隐藏状态最大差异 1.15e-5。
- 含真实 episode reset 时，隐藏状态最大差异 15.32，输出差异 1.18。
- 对应代码 ssm/common.py 在前面插入初始状态后，却把 reset 的占位插在 -1；其非零初值复位还没有直接编码成独立的仿射映射。
- nets.py 的并行后验路径在内部回合边界没有像 obs_step 一样清除上回合的潜变量输入。

这些代码与本地官方源文件相同，不是本次种子/评估适配引入。不能因来自官方就跳过训练/推理一致性校准。修复后在同一实际回放/已训练权重上，完整 RSSM 的 hidden/deter/prior logits 最大差异分别为 7.86e-6/7.15e-6/3.34e-6。证据：
- docs/r2i_core_audit_original.json
- docs/r2i_core_audit_fixed.json
- docs/r2i_rssm_audit_v2.json
- docs/r2i_rssm_audit_resetfix-v4.json
- docs/r2i_v4_reset_fix.patch

这是已定位并修复的一致性错误。是否是随机回报的主要原因，要由匹配训练条件下的原版/修正版曲线判断，不能仅由数值修复推断已经学会任务。我们的 persistent-ssm-v2 使用独立的仿射扫描实现，不继承这段 R2I 代码。

## 新的成对校准

code/r2i-recipe-reference-v4 和 code/r2i-resetfix-v4 恢复公开 POPGym 训练配置。二者网络/目标/数据/训练条件相同，代码差异只有 nets.py 和 ssm/common.py 的复位修正。原版和修正版都在 Easy/Medium 运行 seed 0，每项预设 5M 训练器计数步，每 25k 步独立评估 20 回合。每 600 秒同步保存中间检查点，比较时使用实际落盘步数和共同预算区间，不冒充恰好 1M 权重。5M 是有限的诊断阶段，不是完成了论文全部预算，也不与旧 SNN 1M 结果直接声称公平性能优劣。

共同适配包括：训练/评估 seed 分离；固定预填充动作 RNG；本地日志；在官方 train 入口增加独立评估；同步落盘和更新计数；允许外部设置显存分配上限。奖励尺度为 1，未乘 48。没有改变模型层数/宽度、batch4×1024、train_ratio64、FIFO_LFS 容量、训练前至少 32768 条数据等核心配置。没有给 R2I 的原版负结果安排相同条件的重复重跑。

之前排队的奖励乘48校准已推迟，因为更直接的复位问题尚待学习验证。E2 仍等待相同训练条件下的 Medium 正对照连续两次 > -0.25。

## GPU 吞吐和同卡并行

用户要求显存足够时用满。真实 4×1024 回放上的完整世界模型/actor/critic FP16 更新，预热后中位数 52.6ms，约 77,829 replay 位置/秒，进程显存 37,724MiB；测量时同卡已有本项目 SSM Hard。该值仅为更新吞吐，不等于环境步吞吐，也不能作为独占卡能效结果。证据 docs/r2i_longseq_gpu_profile_fresh.json。

第一次吞吐测试试图将旧 FP32 检查点直接载入 FP16 优化器，缺少动态梯度缩放状态而失败。日志 logs/r2i_longseq_gpu_profile.log 保留。改成真实回放加全新 seed0 FP16 初始化后完成 12 次完整更新，核心损失/梯度有限。这是测试初始化修正，不是一次失败训练的重复重跑。

新增调度器 scripts/run_r2i_recipe_queue.py，清单 configs/experiments_r2i_recipe_v4.json，状态 runs/queue_r2i_v4_status.json，tmux fwf-r2i-v4。只在 GPU0–3 上与 queue_v2 中已知本项目进程同卡，不与未知进程同卡；启动前核对进程和空闲显存。每项显存预算 40000MiB，额外预留6000MiB，JAX allocation fraction 0.45。Easy 修正版已通过实际 FIFO_LFS / train 入口，50,024 步完成269次更新，已启用其他配对任务。

GPU0: SSM Easy + R2I Easy 原版
GPU1: SSM Hard + R2I Easy 修正版
GPU2: MCN Easy + R2I Medium 原版
GPU3: SSM Medium + R2I Medium 修正版

现有四项继续原来的1M预算，不改正在运行的代码/批量。上述 R2I 配对为新增配置，不能混进旧条件。最终实际启动范围以队列状态/command.json/result.json 为准，不以清单代替实际运行。

资源采样 schema2 按设备记录所有同卡 job，并分开记录各进程显存。设备利用率和功率为共享测量，不归给某个模型。旧 schema1 在增加同卡任务之前的记录保留。status_v2.py 和检查点归档同时覆盖两条队列。R2I 保存模型/优化器和更新计数，但并不承诺环境/RNG/数据线程的逐位精确续跑。

## 尚未完成

正对照尚未证明学会；原版/修正后性能因果检验尚在启动阶段；全部 E1–E5、额外模型/任务和 Memory Maze 数据仍存在缺口。不得把此次修复和吞吐测试写成科学主结果。现有小时 heartbeat 继续检查两条队列，无需另建重复 monitor。

实际保存的旧 R2I 训练数据另作核验：Easy 600 个完整回合、31200 条记录，Medium 288 个完整回合、29952 条记录；使用 outgoing action[t-1] 对齐 reward[t] 后，官方奖励关系的最大误差均为0。见 docs/r2i_saved_training_alignment.json。这排除了这部分已保存轨迹的动作/奖励错位，不能单独证明所有 sampler 路径无问题。

23:46 运行记录：四组 R2I 均已实际启动；GPU0/1/2/3 总显存约47/47/70/47GB，八个本项目训练进程。SSM v2 Easy 最近独立评估为427500步 -0.23958、452500步 -0.17708，开始摆脱随机地板；Medium 未过学习门槛，E2不解禁。
