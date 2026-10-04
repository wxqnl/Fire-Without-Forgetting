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
| 3 REVIEW | 完成（10 月 4 日）：Major Revision；评分 3/3/4/3/2（6 级）；面板由同一子 agent 依次写成，非独立 | `review/stage3/`（5 份报告、editorial_decision、revision_roadmap） | "你搞个子 agent 来继续模拟审稿" |
| 4 REVISE | 进行中（10 月 4 日） | | "1a 2A 3做" |
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
- 进入 Stage 3：用户原话 "你搞个子 agent 来继续模拟审稿"（10 月 4 日）。

Stage 3 结果核对（主 agent，10 月 4 日）：
- Spiking-WM 原文（arXiv 2503.00713 表 I）评测的是像素 DMC，1M 帧，约 18.8M 参数；walker walk / cheetah run / cartpole swingup 分别为 938.5 / 577.2 / 791.0。审稿所说属实；论文 §5.1 "on which Spiking-WM was evaluated" 需更正。
- R2I 在 POPGym 上 2M 步处于随机水平：与其原文 50M 步曲线（图 9）对照，Medium 一致，Easy 低于其 3 种子均值。属于预算差异，需在文中说明（EXPERIMENTS §19.5）。
- 用户在审稿决定检查点的选择（10 月 4 日，原话）："1a 2A 3做"。
  - 1a：网络大小对照只跑一个任务，从 RepeatPrevious Easy 换到 Medium；
  - 2A：DMC 只改文字（写明本体感觉输入，列出 Spiking-WM 原文的像素 DMC 数字作参考），不补跑像素 DMC；
  - 3 做：补 RepeatPrevious Hard 留出任务的世界模型探针（所有模型），以及两个设计消融探针（有状态并会重置的胞体；树突接收本层脉冲）。
- 进入 Stage 4 REVISE。

Stage 4 修订进度（10 月 4 日）：
- 已用现有材料完成：
  - RM-1：主张措辞改为"达到 0.9"并写明预算；不再写"按 R2I 的协议"；给出 L71/101 的最终回报。
  - RM-2：§5.1 写明 actor 读树突、12.9M 对 3.4M，并写明 BSuite 结果不依赖树突读出。
  - RM-3：R2I 的 50M 步曲线、Medium 一致、DreamerV3 一致，比较限定为样本效率。
  - RM-4：单次运行声明、评估次数和间隔、附录稳定性表。
  - RM-6A：写明 Spiking-WM 原文是像素 DMC（19M 参数，938/577/791）。
  - RM-7：补 FFM、LMU、S5-ICRL 三篇引用（NeurIPS 论文集已核实），§4.2 写明继承自 LRU 的部分和 PDN 新增的部分。
  - RM-8：脉冲表述限定为"世界模型内部"，写明 LayerNorm 和读出路径。
  - RM-9：能耗计入 actor，并修正计数错误（EXPERIMENTS §19.9）。
  - RM-10：L64 的预算限定，以及探针中 DreamerV3 同样失败。
  - RM-11：写明开发顺序，以 Hard 作留出任务。
  - RM-15：评估细节、EMA、共享种子、替代导数、代码发布。
  - RM-16：Spiking-WM 树突重置的脚注（官方代码 `MCNode.calc_spike`；原文公式只有胞体重置，已核对）。
  - RM-17：图 4 移到附录 C；表 4 去掉能耗列，修好溢出；正文控制在 8 页内。
- 等结果：POPGym 和 DMC 最终值（今天）；Hard 探针和两个设计消融；Medium 宽 actor（约 10 月 6 日）；Memory Maze（约 10 月 7–15 日，RM-5）。
