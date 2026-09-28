# 实验结果快照

采集开始：2026-09-28T15:30:38.987946+08:00

每个条件只跑 seed 0。已完成和仍在运行的结果分开列出；这是阶段实验记录，不是完整论文结论。

计步：Torch 和 BSuite 新适配按实际动作计；早期 R2I / Dreamer POPGym 按含 reset 首帧的 driver transition 计。不同预算/框架不能直接当成严格匹配比较。

## 已完成阶段实验

|运行|状态|最新计步|计步口径|最终/最近回报|评估回合|
|---|---|---:|---|---:|---:|
|[dreamerv3_size12m_repeat_easy_s0](runs/v2/dreamerv3_size12m_repeat_easy_s0/summary.json)|completed|5,000,000|driver_transitions_including_reset|0.262500|20|
|[dreamerv3_size12m_repeat_hard_s0](runs/v2/dreamerv3_size12m_repeat_hard_s0/summary.json)|completed|5,000,000|driver_transitions_including_reset|-0.484783|20|
|[dreamerv3_size12m_repeat_medium_s0](runs/v2/dreamerv3_size12m_repeat_medium_s0/summary.json)|completed|5,000,000|driver_transitions_including_reset|-0.534722|20|
|[persistent_ssm_v2_repeat_easy_s0](runs/v2/persistent_ssm_v2_repeat_easy_s0/summary.json)|completed|1,000,000|actual_actions|-0.154167|20|
|[persistent_ssm_v2_repeat_hard_s0](runs/v2/persistent_ssm_v2_repeat_hard_s0/summary.json)|completed|1,000,000|actual_actions|-0.500000|20|
|[persistent_ssm_v2_repeat_medium_s0](runs/v2/persistent_ssm_v2_repeat_medium_s0/summary.json)|completed|1,000,000|actual_actions|-0.495833|20|
|[r2i_bsuite_memory4_s0](runs/v2/r2i_bsuite_memory4_s0/summary.json)|completed|1,000,000|actual_actions|-0.100000|20|
|[r2i_recipe_reference_repeat_easy_s0](runs/v2/r2i_recipe_reference_repeat_easy_s0/summary.json)|completed|5,000,000|driver_transitions_including_reset|0.789583|20|
|[r2i_recipe_reference_repeat_medium_s0](runs/v2/r2i_recipe_reference_repeat_medium_s0/summary.json)|completed|5,000,000|driver_transitions_including_reset|-0.548611|20|
|[r2i_recipe_resetfix_repeat_easy_s0](runs/v2/r2i_recipe_resetfix_repeat_easy_s0/summary.json)|completed|5,000,000|driver_transitions_including_reset|0.816667|20|
|[r2i_recipe_resetfix_repeat_medium_s0](runs/v2/r2i_recipe_resetfix_repeat_medium_s0/summary.json)|completed|5,000,000|driver_transitions_including_reset|-0.465278|20|
|[r2i_repeat_easy_s0_cuda](runs/v2/r2i_repeat_easy_s0_cuda/summary.json)|completed|1,000,000|driver_transitions_including_reset|-0.516667|20|
|[r2i_repeat_medium_s0_cuda](runs/v2/r2i_repeat_medium_s0_cuda/summary.json)|completed|1,000,000|driver_transitions_including_reset|-0.495833|20|

## 早期 100k 四格初筛

|运行|状态|最新计步|计步口径|最终/最近回报|评估回合|
|---|---|---:|---|---:|---:|
|[no_dendrite_reset_tau2_seed0](runs/phase0_initial/no_dendrite_reset_tau2_seed0/summary.json)|completed|100,000|actual_actions|-0.504167|20|
|[original_tau2_seed0](runs/phase0_initial/original_tau2_seed0/summary.json)|completed|100,000|actual_actions|-0.531944|20|
|[persistent_no_dendrite_reset_tau2_seed0](runs/phase0_initial/persistent_no_dendrite_reset_tau2_seed0/summary.json)|completed|100,000|actual_actions|-0.495833|20|
|[persistent_tau2_seed0](runs/phase0_initial/persistent_tau2_seed0/summary.json)|completed|100,000|actual_actions|-0.498611|20|

## 正在运行：临时结果

|运行|状态|最新计步|计步口径|最终/最近回报|评估回合|
|---|---|---:|---|---:|---:|
|[dreamerv3_bsuite_memory4_s0](runs/v2/dreamerv3_bsuite_memory4_s0/summary.json)|running|278,664|actual_actions|0.300000|20|
|[mcn_repeat_easy_original_s0](runs/v2/mcn_repeat_easy_original_s0/summary.json)|running|605,523|actual_actions|0.268750|20|
|[persistent_ssm_bsuite_memory4_s0](runs/v2/persistent_ssm_bsuite_memory4_s0/summary.json)|running|93,160|actual_actions|0.400000|20|
|[persistent_ssm_v3_longseq_repeat_easy_s0](runs/v2/persistent_ssm_v3_longseq_repeat_easy_s0/summary.json)|running|2,909,040|actual_actions|-0.100000|20|
|[r2i_bsuite_memory30_s0](runs/v2/r2i_bsuite_memory30_s0/summary.json)|running|300,032|actual_actions|-0.300000|20|

## 完整训练集成：不作为性能结果

|运行|状态|最新计步|计步口径|最终/最近回报|评估回合|
|---|---|---:|---|---:|---:|
|[dreamerv3_bs30_integration_s0](runs/v2/dreamerv3_bs30_integration_s0/summary.json)|completed|34,816|actual_actions|-0.100000|20|
|[dreamerv3_v1_integration_easy_timer_s0](runs/v2/dreamerv3_v1_integration_easy_timer_s0/summary.json)|completed|16,384|driver_transitions_including_reset|-0.497917|20|
|[persistent_ssm_bs30_integration_s0](runs/v2/persistent_ssm_bs30_integration_s0/summary.json)|completed|34,816|actual_actions|-0.100000|20|
|[persistent_ssm_v3_longseq_integration_easy_s0](runs/v2/persistent_ssm_v3_longseq_integration_easy_s0/summary.json)|completed|33,024|actual_actions|-0.489583|20|
|[r2i_bs30_integration_s0](runs/v2/r2i_bs30_integration_s0/summary.json)|completed|34,816|actual_actions|0.000000|20|

## 失败、排除或未确认

|运行|状态|最新计步|计步口径|最终/最近回报|评估回合|
|---|---|---:|---|---:|---:|
|[dreamerv3_v1_integration_easy_s0](runs/v2/dreamerv3_v1_integration_easy_s0/summary.json)|failed|0|driver_transitions_including_reset|-0.468750|20|
|[persistent_ssm_repeat_medium_s0](runs/v2/persistent_ssm_repeat_medium_s0/summary.json)|excluded|119,500|actual_actions|-0.491667|20|
|[r2i_repeat_easy_s0](runs/v2/r2i_repeat_easy_s0/summary.json)|failed|0|driver_transitions_including_reset|—|—|
|[r2i_repeat_medium_s0](runs/v2/r2i_repeat_medium_s0/summary.json)|failed|0|driver_transitions_including_reset|—|—|

## 阅读方式

每个运行目录保留原始配置、命令、退出/排除记录，另有全部 evaluations.jsonl、training_scalars.jsonl 与日志末尾。Dreamer 的逐回合评估明细在 evaluation_episodes.jsonl。无凭据、权重或 replay。

源代码/环境审计、最终检查点的独立读回证据及 E1–E5 对照见 ../docs/README.md。resource_samples.jsonl 只覆盖采样启动后的时段，共卡功率不能归于单个模型。
