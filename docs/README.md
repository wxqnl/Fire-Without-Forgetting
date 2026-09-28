# 文档索引

优先阅读：[结果解读](RESULTS.md)、[原方案对照](plan_alignment_20260928.md)、[代码地图](CODE_MAP.md)、[复现说明](REPRODUCING.md)。当前快照时间以 [results/summary.json](../results/summary.json) 为准。

## 方案与执行

- [原研究方案](snn_world_model_research_plan.md)：包含待验证假设和预期结果，不能当作实验事实。
- [原计划对照](plan_alignment_20260928.md)：E1–E5、Phase0判定门、已执行范围与缺口。
- [执行协议v2](experiment_protocol_v2.md)：追加式历史记录；后面的修订优先，早期段落描述当时状态。
- [BSuite阶段方案](bsuite_stage_v1.md)：三长度、三模型、实际action计步及固定预算。
- [发布快照说明](REPOSITORY_SNAPSHOT.md)：文件范围、数据导出口径和服务器保留内容。
- [历史README](history/README_20260927_initial.md)：保留原记录，其“当前”指早期执行时点。

## 基线审计与完成证据

- [R2I配置与复位审计](r2i_baseline_audit_20260927.md)
- [R2I原版/修正版5M完成记录](r2i_recipe_v4_completed.json)
- [DreamerV3三任务5M完成记录](dreamerv3_repeat_completed.json)
- [R2I BSuite长度4最终检查点](r2i_bsuite_memory4_completed.json)
- [最初四格100k结果](phase0_results.md) / [JSON](phase0_results.json)
- [本方法v2 Easy](ssm_v2_easy_completed.json)、[Medium](ssm_v2_medium_completed.json)、[Hard](ssm_v2_hard_completed.json)

## 方法实现与诊断

- [v1零发放编码器诊断](ssm_v1_encoder_diagnosis.json)：已排除运行的原因。
- [v2真实更新及一致性集成](persistent_ssm_v2_integration.json)
- [长序列版本完整训练集成](persistent_ssm_longseq_v3_integration.json)
- [最终Medium奖励学习诊断](ssm_v2_medium_step1000000_diagnosis.json)
- [最终Medium条件扫描/递推一致性](duality_v2_medium_step1000000.json)
- [确定性后验探针](probes/)：补充口径；早期短延迟提升不能代替在线后验采样证据。
- [实际采样后验探针](probes_stochastic/)：当前主要探针口径。
- [BSuite完整训练/保存集成](bsuite_training_integration.json)
- [历史巡检](monitor/)：按实际执行时间读取；存在已记录的监控间隔，未假定逐小时都成功执行。

## 环境与来源

- [上游来源](sources.json)：URL、分支和下载日期；保留源码许可证。
- [向量任务环境检查](benchmark_installation.json)
- [视觉环境安装检查](visual_benchmark_installation.json)：可渲染不代表已进行视觉训练。
- [数据资产状态](../data/assets.json)
- [第三方许可](../THIRD_PARTY.md)
