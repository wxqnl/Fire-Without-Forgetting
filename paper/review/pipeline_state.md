# ARS 流水线状态（论文完善与润色）

- 启动：2026-10-03 11:40，用户指令："用这个来吧我们的 Paper 完整的完善和润色一下"（`/academic-research-skills:ars-full`）
- 入口：已有完整草稿，属于中途进入。按协议从 Stage 2.5 开始，跳过 Stage 1（研究）和 Stage 2（写作）
- 投稿目标：CVPR 2027（注册 2026-11-10，投稿 2026-11-16），LaTeX，CVPR author kit
- 不变的约束（来自此前的用户指令）：
  - 单种子，多种子暂不做；
  - 不写成"把 SSM 改成脉冲"，GRSN 只作为基线；
  - 表述直接、不防御；
  - 用原论文的标准配置，不调参；
  - Memory Maze 子集不进论文

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
