# 代码版本与实验对应

这些目录是实际运行快照。保持原路径是为了从每项结果的command.json/job.json定位对应源码；不把它们合并成一个会覆盖历史行为的工作目录。

|目录|用途与状态|
|---|---|
|code/spiking-wm|原MCN及两级复位四格的100k初筛；向量任务适配和干预实现|
|code/mcn-v2|完整原版MCN复位设置；Easy 1M学习校准|
|code/mcn-mechanism-v2|MCN 2×2复位和独立树突时间常数实现；后续Medium矩阵目前禁用|
|code/persistent-ssm-v1|初版方法；T=1前馈编码器不发放，运行已排除，保留诊断|
|code/persistent-ssm-v2|编码器修正后的完整方法；Easy/Medium/Hard各1M阶段已完成|
|code/persistent-ssm-longseq-v3|与v2相同模型/优化器，batch4×1024、8训练环境、独立1评估环境；Easy5M及BSuite阶段|
|code/r2i-v2|早期短序列R2I配置；1M评估完成但异步最终检查点未完整落盘|
|code/r2i-v3|同步保存及奖励尺度诊断候选；正式reward×48候选已被后续Easy学习校准替代|
|code/r2i-recipe-reference-v4|恢复官方长序列/FIFO/精度/训练入口的原版R2I；Easy/Medium各5M|
|code/r2i-resetfix-v4|相同配置的R2I复位修正版；Easy/Medium各5M|
|code/r2i-bsuite-v5|基于修正版接入官方BSuite和独立实际action计数；模型/优化器不变|
|code/dreamerv3-v1|官方完整size12m加真实POPGym适配和独立评估；三难度各5M|
|code/dreamerv3-bsuite-v2|官方BSuite适配、实际action预算及保存；保留Dreamer模型/优化器|

## 本方法入口

- code/persistent-ssm-longseq-v3/persistent_ssm.py：持久树突、可微扫描、一步递推、观测独立后验。
- code/persistent-ssm-longseq-v3/models.py：世界模型与行为学习接线。
- code/persistent-ssm-longseq-v3/dreamer.py：交互、回放、训练和独立评估。
- code/persistent-ssm-longseq-v3/envs/memory_tasks.py：实际官方向量环境适配。
- code/persistent-ssm-longseq-v3/configs.yaml：完整模型、persistent及longseq配置。

当前向量任务使用脉冲MLP编码器。没有用模拟数据、玩具模型或隐藏环境状态替代真实任务。视觉场景和全部同框架神经元消融尚未完成。

## 调度与分析

三个清单分别是 configs/experiments_v2.json、configs/experiments_r2i_recipe_v4.json、configs/experiments_dreamerv3_v1.json。最后一个也承载本方法长序列和BSuite队列，不仅包含Dreamer。

scripts/status_v2.py汇总实际在线状态；scripts/measure_resources.py与archive_checkpoints.py记录资源/归档。probe_memory.py、evaluate_duality.py、diagnose_*使用真实保存模型和数据；集成检查不当作性能结果。export_results.py将运行记录导出成不含权重/replay的独立结果快照。

相对上游的适配补丁与各次修复证据保存在docs内的.patch和审计JSON。sources/保存当时获取的上游源码；各目录中的论文资源和上游scores仅作来源，不混入本项目结果。
