# BSuite 记忆长度阶段执行方案

本阶段属于原计划E1/E3的BSuite记忆长度曲线子集。由于POPGym Medium在已完成5M的R2I和DreamerV3中均未达到学习门槛，E2 Medium复位矩阵仍禁用；本阶段不代替它，也不依据困难任务地板效应支持/否定复位机制。

## 事前固定的范围

官方BSuite memory_len/3、memory_len/15、memory_len/22，memory_length分别4、30、100，num_bits=1。使用已安装官方模块和sweep.SETTINGS，原始三维观测、末尾±1奖励、动作空间和终止条件均保留。BSuite的随机策略期望回报为0，与POPGym RepeatPrevious的-0.5不同。

每个长度比较修正复位的完整R2I、官方DreamerV3 size12m、完整持久SSM；每个条件只跑训练seed0，三长度共9项，各自从新初始化训练。本轮尚未加入MCN等所有SNN或全部23种官方长度，不能写成完整BSuite曲线或全部E1完成。

每项1,000,000实际训练动作步；batch4×1024，train_ratio64回放位置/实际动作，8训练环境、1评估环境。每25k实际动作独立评估20回合，保存最终评估与checkpoint。训练环境seed=0+编号，评估seed=1000000，不反复播种。短回合的reset首帧不计入动作预算，也不推进采样更新比率。R2I/Dreamer新增显式action counter保存在检查点、评估和日志，原driver transition计数同时保留；Torch原本按action计数。可直接核对实际action_steps，而不以未经换算的driver步冒充公平预算。

R2I使用既定修正版模型、FP16、FIFO_LFS10M容量；Dreamer沿用完整size12m、BF16、uniform/online replay5M容量、replay_context1；本方法保留完整v3模型、FP32、episode replay10M。三个后端均约32768动作预热，SSM仍有原生100次pretrain更新，Dreamer预热使用原生未训练策略、R2I/SSM使用随机策略。保留原生评估动作方式（Dreamer采样、SSM argmax等）并记录。这些算法、回放、精度和初始更新差异不允许把比较称为单因素网络替换实验。

1M是有界阶段预算，不冒充官方BSuite按10,000回合的评分协议或论文完整复现。每回合真实action数为memory_length+1；20回合适配核对的实际步数分别100/620/2020。后续分析同时报告episode数和置信范围，不把单seed回合波动当成跨训练种子稳定性。

## 实现和放行

独立快照code/r2i-bsuite-v5及code/dreamerv3-bsuite-v2，只改环境适配、真实动作计步/预算和日志保存，没有改世界模型、损失或优化器。SSM复用正在执行其他条件的不可变code/persistent-ssm-longseq-v3，仅通过CLI选择官方BSuite任务。

三后端三个长度的观测、奖励、终止已逐步对照官方环境各20回合，通过结果为docs/bsuite_adapter_{r2i,dreamerv3,torch}.json。GPU0/1/3分别先执行R2I/Dreamer/SSM的length30完整模型集成，目标34816实际动作（含32768预热），确认真实优化更新、有限核心指标、20回合评估、最终模型和实际计数，再启用该后端的三长度主运行；失败保留原记录，不能原样重复。

队列仍为configs/experiments_dreamerv3_v1.json和tmux fwf-dreamerv3，包含本轮BSuite和未结束的POPGym SSM长序列；GPU2的旧MCN/SSM Easy继续既定预算。固定每个后端在各自GPU上按4→30→100接续，已验证任务自动衔接，避免依赖下一次桌面heartbeat才启动。

## 本次监控时间

收到的heartbeat标记2026-09-28 04:02（北京时间），本次实际执行约14:44，服务器时间与时钟工具一致。不能把这段没有执行巡检的时间称为已完成逐小时检查。原服务器训练和调度服务持续运行；Dreamer三项11:13–11:37完成，此后GPU0/1/3空闲至本次接续。原因未确定，不把自动化触发时间当作实际检查时间。报告记录实际执行时间。


## 15:01 放行记录

三项集成全部退出码0，读回最终检查点均34816实际动作：R2I31次更新、Dreamer33次更新、SSM132次更新（含原生100次pretrain）。R2I三优化器计数均31，Dreamer优化器计数33，SSM三优化器状态完整；核心指标有限。R2I/Dreamer的driver计数为35944，明确区别于34816实际action。

30秒资源采样记录的进程显存最大值：R2I37730MiB、Dreamer3708MiB、SSM18406MiB；这是采样最大值，不是分配峰值。证据docs/bsuite_training_integration.json及三个后端详细记录。

15:01:47–48已启动三个memory_length4主运行；每个后端随后自动接续30、100。主运行不是沿用集成权重，各自seed0重新初始化、按原定1M实际action预算执行。训练源快照保持不变。
