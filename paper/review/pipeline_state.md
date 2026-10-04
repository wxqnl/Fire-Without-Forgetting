# ARS 流水线状态（论文完善与润色）

- 启动：2026-10-03 11:40，用户指令："用这个来吧我们的 Paper 完整的完善和润色一下"（`/academic-research-skills:ars-full`）
- 入口：已有完整草稿，属于中途进入。按协议从 Stage 2.5 开始，跳过 Stage 1（研究）和 Stage 2（写作）
- 投稿目标：CVPR 2027（注册 2026-11-10，投稿 2026-11-16），LaTeX，CVPR author kit
- 不变的约束（来自此前的用户指令）：
  - 单种子，多种子暂不做；
  - 不写成"把 SSM 改成脉冲"，GRSN 只作为基线；
  - 表述直接、不防御；
  - 用原论文的标准配置，不调参；
  - Memory Maze 子集不进论文；
  - 本文模型只用一个配置（10 月 4 日，用户原话："我们的模型不应该就是一个配置吗，别写的花里花哨的"）：所有任务的策略都读树突脉冲，不按任务换

| 阶段 | 状态 | 产出 | 用户决定（原话） |
|---|---|---|---|
| 1 RESEARCH | 跳过 | — | — |
| 2 WRITE | 跳过（草稿已有） | — | — |
| 2.5 INTEGRITY | 完成：第 1 轮 FAIL，修正后 PASS WITH NOTES；Mode 4 SUSPECTED | `stage2_5_integrity.md`、`audit/` | 待确认 |
| 3 REVIEW | 未开始 | | |
| 4 REVISE | 未开始 | | |
| 3′ RE-REVIEW | 未开始 | | |
| 4.5 FINAL INTEGRITY | 未开始 | | |
| 5 FINALIZE | 未开始（输出为 CVPR LaTeX，不转 DOCX/APA） | | |
| 6 PROCESS SUMMARY | 未开始 | | |

修订稿往返次数（advisory）：
- 目前 1 次（Stage 2.5 第 1 轮修正）；
- 最坏情况约 8 次：Stage 4 和 4′ 各 1 次，Stage 2.5 和 4.5 的修正循环最多各 3 次。

Stage 2.5 之后、Stage 3 之前的用户改动（10 月 4 日）：
- 用户原话："我觉得我们就得确定一个配置作为主的呀"；"我们的模型不应该就是一个配置吗，别写的花里花哨的"。
- 处理：主配置统一为 `PD2Lv2`，论文对应修改见 EXPERIMENTS.md §19.6。
- 改动过的主张（L64 步数与稳定性、R2I 协议步数、能耗比 3.2 倍、附录 A/E）已按运行数据逐条核对。
- Mode 4 现在的情况：本文的 actor-critic 在所有任务上都读树突，因而更大（POPGym 上 12.9M，基线 3.4M）；附录 A 已披露。
- Mode 4 用户决定（10 月 4 日，原话）："b 可以跑，但是没必要都跑呀，他就是最多算一个消融，就在消融里面做一个任务就够了呀"。
  - 处理：在 POPGym RepeatPrevious Easy 上加跑一组只读胞体、actor-critic 加宽到 12.85M 的对照，放进附录 E（EXPERIMENTS.md §19.7）。
- 是否进入 Stage 3：尚未得到用户的"继续"。
