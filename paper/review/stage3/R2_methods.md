# R2 — 方法 / 实验设计审稿报告

稿件：*Fire Without Forgetting: Persistent Dendrites Give Spiking World Models Long-Term Memory*（CVPR 2027 投稿稿，2026-10-04 版本）
审稿轮次：第 1 轮（ARS Stage 3，full mode）
面板席位映射：本报告对应 ARS 面板的 Peer Reviewer 1（Methodology）席位（dispatch role = methodology），负责维度 D1，并与 DA 共同负责 D3。

## 审稿人信息

- **身份**：强化学习实验方法学方向的审稿人，长期关注深度 RL 的评估协议、基线公平性与统计报告（如单种子结果的可靠性、"steps-to-threshold"类指标的偏差、参数量/预算匹配）。
- **审稿重点**：协议与基线配置、比较是否公平（预算、配置、参数量、输入）、单种子下的统计报告、消融与干预是否把每个因果主张单独隔离、可复现性。
- **评分量表**：CVPR 近年审稿表风格的 6 级总体评分（6 Accept / 5 Weak Accept / 4 Borderline Accept / 3 Borderline Reject / 2 Weak Reject / 1 Reject）；置信度 1–5。
- **总体评分**：**3 — Borderline Reject**
- **ARS 建议**：Major Revision
- **置信度**：4

## 总体评价

论文在一个共享的脉冲世界模型中只替换循环核心，比较 Spiking-WM、循环 LIF、GRSN、去重置版 Spiking-WM-NR、同构 ANN（DreamerV3 架构）与 FwF，并用世界模型探针把记忆与探索/信用分配分开。实验设计的骨架是好的：探针数据对所有模型相同，干预配有排除替代解释的对照（线性解码、发放率上限、阈值扫描），消融逐项改变一个设计选择。

方法上的主要问题有四个。第一，RL 比较并非"只差循环核心"：FwF 的 actor-critic 为 12.9M 参数并多读 6144 路树突脉冲，基线为 3.4M；作者自己的 Table 7 显示 POPGym 与 DMC 的 RL 收益依赖这一读出，而容量对照仍在进行且只覆盖 Easy。第二，所有结果为单种子，"solve"判据是评估回报首次达到 0.9，而评估回合数未报告；图 1 的最终回报（L=101 约 0.40）显示这一判据偏宽松。第三，若干基线没有使用其源论文配置（Spiking-WM 的规模与 DMC 协议、GRSN 加了 LayerNorm、R2I 的预算），而预注册的 DMC 判据正是相对 Spiking-WM。第四，BSuite 短预算设置（L=64 只有 500k 步、并行环境共享种子）使"唯一解决 64 步"的结论依赖预算。

BSuite 主结论在策略只读 (z_t, h_t) 时依然成立，世界模型探针也独立支持 64 步记忆，这是论文最稳的证据。上述问题大多可用现有记录与少量追加运行修复，因此建议 Major Revision。

contract_role: methodology

## Dimension Scores

### D1: methodology_rigor
score: block
trigger: "标题性比较中各方法的训练预算、配置或容量不对等"
block_class: repairable

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: warn
trigger: "结论的表述强于证据所能支撑的程度，或依赖尚在进行中的临时结果"

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

以下优点与缺点均从实验设计与统计报告角度给出；每个缺点注明严重程度、证据锚点与可操作的修改建议。凡涉及作者已固定的决定（单种子、单一模型配置、不调参），本报告只陈述一次关切，并给出在这些限制内的缓解方案。

### S1: 世界模型探针设计合理，所有模型使用相同数据
探针用随机策略采集的同一批数据（200k 训练步、512 个留出回合）训练各模型的世界模型，只在回忆步计算奖励预测的 AUC，去掉了探索与信用分配（与 Ni et al. 的区分一致）。这使"世界模型能否保持信息"可以在等更新预算下直接比较。
**Evidence Anchor**: text: Supp. C "200k steps for training, 512 held-out episodes for evaluation, the same data for every model"

### S2: 干预配有排除替代解释的对照
A+B 在 32 步失败时，作者用线性探针证明膜电位中仍有线索（准确率 0.88），说明问题是"痕迹太弱"而非"痕迹不存在"；又用发放率上限 0.06 排除"G 只是提高了发放率"，用阈值扫描（Supp. D）排除"发放重置通过发放频率限制记忆"。这是本文方法上最好的部分。
**Evidence Anchor**: text: §3.2 "capped at a firing rate of 0.06, A+B+G still learns 32 steps in 2000 updates"

### S3: 逐项消融，包含模型内部的持久性检验
Table 5 每行只改一个设计选择，其中"dendrites cleared every step"（AUC 0.50）是在 FwF 内部对 reset-induced amnesia 的直接检验，与第 3 节在 Spiking-WM 上的干预形成双向证据。
**Evidence Anchor**: table: Table 5 — "dendrites cleared every step" 行，L=64 为 "-- (0.50)"

### S4: 策略输入对照已经存在，并支持 BSuite 主结论
Table 7 中只读 (z_t, h_t) 的策略在 BSuite 上同样解决全部长度（64 步 102k、101 步 362k），说明 BSuite 结果不依赖树突读出。
**Evidence Anchor**: table: Table 7 — "(z_t, h_t) only" 行 BSuite 列 12k–362k

### W1: actor-critic 容量与输入不对等，POPGym 与 DMC 的 RL 收益无法归因于世界模型
**Problem**: 正文称各模型只在循环核心上不同，但补充材料 A 显示 FwF 的 actor-critic 为 12.9M 参数（基线 3.4M，约 3.8 倍），并额外读取 6144 路树突脉冲（每路 T=5 帧）。Table 7 表明，去掉树突读出后 FwF 在 RepeatPrevious Easy 从 0.71 降到 0.11（低于 LIF 0.69、Spiking-WM 0.43、DreamerV3 0.39），Medium 从 0.45 降到 −0.03，DMC walker 从 880 降到 504（Spiking-WM 为 923）。宽 actor-critic 对照只在 Easy 上运行且尚未完成。
**Evidence Anchor**: text: Supp. A "the actor-critic has 3.4M for the baselines and 12.9M for FwF, which also reads the 6144 dendritic spikes"
**Why it matters**: POPGym 与 DMC 的 RL 结果目前同时受"世界模型记忆""策略可读到的分级状态"与"策略容量"三者影响，无法单独归因于前者。作者固定了单一模型配置，这一点我只陈述一次；问题不在于配置本身，而在于正文没有披露差异、也没有完成区分三种解释的对照。
**Suggestion**: (1) 在 §5.1 写明 actor-critic 输入与参数量，并把 Table 7 结论移入正文；(2) 完成宽 actor-critic 对照，并把它扩展到 Medium（Medium 是论文宣称独有的能力）；(3) 说明 DreamerV3 的 actor 已读取其完整连续状态 h_t，因此 FwF 与 DreamerV3 的比较在"actor 读取核心完整状态"这一意义上是对等的；(4) 可选的追加运行：对有持久状态的基线（GRSN、Spiking-WM-NR）用同一 ON/OFF 编码把其膜状态提供给 actor，在 RepeatPrevious Easy/Medium 上各跑一次。这不改变 FwF 的配置，只把同等信息提供给基线。
**Severity**: Major
**Confidence**: 5 — core expertise: RL 实验公平性

### W2: 单种子与宽松的"solve"判据
**Problem**: 除 Table 1（两个种子）外全部结果为单种子。BSuite 的指标是评估回报"first reaches 0.9"，每次评估的回合数与评估间隔未报告。图 1 显示 FwF 在 L=71、101 的最终回报约 0.46、0.40；图 3 中 71 与 101 的曲线在峰值后回落到 0.5–0.9 之间。
**Evidence Anchor**: text: §5.1 "Number of seeds: one so far (multiple seeds deferred, decision of Oct. 3); state the final count here."
**Why it matters**: "首次越过阈值"对单次运行的噪声敏感：若每次评估只有少量回合，一个平均正确率 80% 的策略也可能偶尔达到 0.9。作者已决定不做多种子，我只陈述一次这一关切；可信度需要靠单次运行内部的变异性报告来补足。
**Suggestion**: 均可由现有日志完成：(1) 报告每次评估的回合数，并为每个评估回报给出二项分布置信区间；(2) 增加更严格的判据，如 EMA ≥ 0.9 或连续 3 次评估 ≥ 0.9，与"首次达到"并列；(3) 报告最终回报、解决后的均值与标准差、≥0.9 与 ≥0.5 的评估比例（作者已对 L=64 给出 94% / 0.87，应对所有长度给出）；(4) 探针 AUC 用 512 个留出回合做 bootstrap 置信区间；(5) 在 Table 1 中两个种子结果高度一致（如 0.5k / 0.5k）可作为单次运行变异较小的旁证，应在正文点明。
**Severity**: Major
**Confidence**: 5 — core expertise: RL 统计报告

### W3: 若干基线未使用源论文配置，而预注册判据正是相对这些基线
**Problem**: Spiking-WM 在本文以 DreamerV3-S 规模（7.2M）运行于本体感受 DMC，而 Spiking-WM 原文使用像素 DMC（1M frames）与约 18.8M 参数；本文复现得到 cheetah run 343、cartpole swingup 497，远低于原文在更难的像素设置下的 577、791。GRSN 被加了 LayerNorm；R2I 的预算从官方配置改为 2M 步。§5.6 的 DMC 判据为"no degradation relative to Spiking-WM"。
**Evidence Anchor**: text: §5.1 "is a non-memory control on which Spiking-WM was evaluated"
**Why it matters**: 作者的原则是采用源论文的标准配置、不调参；当基线偏离其源配置且明显弱于其发表结果时，相对它的"无退化"判据几乎自动满足，读者无法据此判断 FwF 在非记忆任务上是否付出代价。
**Suggestion**: (1) 在表中并列 Spiking-WM 原文数字并说明协议差异；(2) 把 DMC 判据同时相对 DreamerV3（同构 ANN）陈述，FwF 目前在 run / swingup 上比 DreamerV3 低约 260 / 250 分，应直接写出；(3) 若 DMC 对照仍可调整，采用 Spiking-WM 原文的像素 DMC 协议正符合"源论文标准配置"原则（新运行）；(4) GRSN 的 LayerNorm 修改移入正文一句话说明。
**Severity**: Major
**Confidence**: 4 — core expertise: 基线公平性；Spiking-WM 原文数字取自其 Table I

### W4: BSuite 短预算设置使"唯一解决 64 步"依赖预算
**Problem**: L=64 的预算为 500k 步，而 L=51/71/101 为 1M 步；DreamerV3 在 L=51 需要 902k 步才解决。L≤64 的运行在四个并行环境间共享同一种子（非标准设置），L≥51 则每个环境不同种子。
**Evidence Anchor**: table: Table 2 — budget 行 L=64 为 500k，51†/71†/101† 为 1M；DreamerV3 51† 为 902k
**Why it matters**: "FwF 是唯一解决 64 步的模型"在 500k 预算下成立，但 DreamerV3 在 1M 预算下是否能解决 64 步未知；共享种子的影响也未说明。
**Suggestion**: 把主张改为直接的样本效率陈述（"FwF solves L=64 in 192k steps; no other model does within 500k"）；指出等更新预算的探针（Table 3）中 DreamerV3 在 64 步同样失败（AUC 0.50），这是不依赖 RL 预算的现有证据；说明共享种子的原因及其对所有模型相同；可选：在 1M 的 R2I 协议下补跑 DreamerV3 与 FwF 的 L=64（两次运行）。
**Severity**: Major
**Confidence**: 4 — core expertise: RL 评估协议

### W5: 干预表格不完整
**Problem**: Table 1 有大量空格（未运行）：A 单独在 L=32 未运行，A+G 只出现在 Supp. D 的阈值实验中，B only 只在 L=8、G only 只在 L=16；L=64 与 τ_d=512 的结果只在正文提及而不在表中。
**Evidence Anchor**: table: Table 1 — "B only" 与 "G only" 行仅各有一个已运行单元，L=32 列多数为空
**Why it matters**: "removing the dendritic reset or restoring the write gain has no effect while the step reset remains"这一结论各只在一个长度上检验；B 的独立贡献（A vs A+B）主要体现为速度（8k→4.5k），在 32 步两者都失败。
**Suggestion**: 用现有运行补齐 A+G（Supp. D 中阈值 0.5 的数据）与 L=64 / τ_d=512 行；补跑 A 在 L=32、B only 在 L=16、G only 在 L=8（各两次探针运行，成本低）；在正文把 B 的作用表述为"加速"而非"必需"。
**Severity**: Minor
**Confidence**: 4 — core expertise: 干预与消融设计

### W6: 保留时间谱的必要性未被证明
**Problem**: 单一 τ=128 在 64 步上与对数均匀谱同样好（3k vs 3.5k 次更新）；全部消融只在 L≤64、k=32 上做，且为单次运行。
**Evidence Anchor**: table: Table 5 — "single τ=128" 行 L=32 为 1k、L=64 为 3k
**Why it matters**: §4.2 主张对数均匀谱使"no time constant is tuned per task"，但现有消融无法区分"谱有用"与"只要有一个足够长的 τ 就行"。
**Suggestion**: 在一个 τ=128 不够的设置上（如 memlen-256 探针，或同时要求短延迟精确回忆与长延迟保持的任务）比较单一 τ 与对数均匀谱；探针成本低。
**Severity**: Minor
**Confidence**: 4 — core expertise: 消融设计

### W7: 复现所需的评估细节缺失
**Problem**: 未说明每次评估的回合数、评估频率、图 3 EMA 系数、RL 评估是否使用确定性策略，也没有代码/模型发布声明。
**Evidence Anchor**: absence: §5.1 Setup 与 Supp. A — expected 评估回合数、评估间隔、EMA 系数与代码发布声明; checked §5.1, §5.2, Table 2 caption, Figure 3 caption, Supp. A–H
**Why it matters**: 这些细节直接决定"first reaches 0.9"与"mean of the last three evaluations"的可信度。
**Suggestion**: 在 Supp. A 增加一张评估协议表，并在正文加入代码发布声明。
**Severity**: Minor
**Confidence**: 5 — core expertise: 可复现性

### W8: Memory Maze 探针的输入不对等
**Problem**: FwF 的 Memory Maze 探针读取与策略相同的特征（含 6144 路树突脉冲 × T 帧），基线读取 h_t 与 z_t。
**Evidence Anchor**: text: §4.3 "the Memory Maze probes read the same features"
**Why it matters**: 探针 MLP 容量固定时，输入维度与信息形式不同会影响解码结果，使 Memory Maze 的比较混入读出差异。
**Suggestion**: 按 TODO 计划同时报告 FwF "spikes only"与"spikes + dendritic spikes"两种探针，并与每个模型的 image-only 探针相减得到 memory gain；在正文写明各模型探针读取的特征。
**Severity**: Minor
**Confidence**: 4 — core expertise: 探针评估设计

### W9: POPGym 比较中的度量与时间点不一致
**Problem**: 基线为最后三次评估的平均，R2I 为最后 200k 步训练回合回报，FwF 当前为 65% / 68% 预算处的临时值。
**Evidence Anchor**: text: Table 2 caption "mean training-episode return over the last 200k steps (its code has no separate evaluation"
**Why it matters**: 不同度量与时间点会放大或缩小差异，Easy 上 FwF 0.71 与 LIF 0.69 的差距本身就在噪声范围内。
**Suggestion**: 最终版统一在 2M 步处报告；对 Easy 直接写"与 LIF 持平"；对所有模型同时给出训练回合回报，以便与 R2I 同口径比较。
**Severity**: Minor
**Confidence**: 4 — core expertise: RL 评估

### 算术核对（非 bounded procedure，供参考）

我核对了以下可复算的数字，均与文中一致：17.18 / 5.38 ≈ 3.19（"3.2×"）；Euler 增益下 (1−λ)/(1+λ) ≈ 1/(2τ)；G 的缩放 √(τ/2)·(1/τ) = ½√(2/τ) ≈ ½√(1−λ²)；6144 = 512 神经元 × 6 个状态变量（2 积分 + 2×2 共振）× ON/OFF 2；"nearly halves"（4.5k 对 8–8.5k，比值 0.53–0.56）。唯一偏差：0.004–0.009 相对阈值 0.5 为 0.8%–1.8%，"about 1%"略低估上界。

### 致作者的问题

1. 每次 BSuite 评估包含多少回合？在"连续 3 次评估 ≥ 0.9"判据下，FwF 在 71 与 101 步是否仍算解决？
2. 宽 actor-critic（12.9M，只读 (z_t, h_t)）在 Medium 上的结果是什么？若仍为 −0.5 左右，你们如何陈述 POPGym RL 收益的来源？
3. L≤64 的运行为何让四个并行环境共享同一种子？这是否意味着四个环境产生相同的线索序列？
4. Spiking-WM 为何以 S 规模、本体感受 DMC 运行，而非其原文的像素 DMC 与约 18.8M 参数配置？

### 总体评分与理由

总体评分 3（Borderline Reject），置信度 4。决定性的未解决准则是 D1：W1（actor-critic 不对等）与 W3（基线偏离源配置）都是 repairable 的，前者主要靠披露与完成已在进行的对照，后者靠并列源论文数字与调整 DMC 协议。W2 受作者固定决定限制，只能用单次运行内部的变异性报告缓解。BSuite 与探针的证据（S1、S4）是扎实的；若 W1–W4 按建议处理，我会升至 4。

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | reviewer_full D6 | NOT_ASSESSED | — | 由 R1 负责 | — | no |
| Methodological Rigor | reviewer_full D1 | PARTLY_MEETS | S1–S3; W1–W4 | 探针与干预设计好；RL 比较存在容量与配置不对等 | 单种子为作者固定决定 | yes |
| Evidence Sufficiency | reviewer_full D1 | PARTLY_MEETS | S4; W2 | BSuite 与探针证据充分；POPGym/DMC 归因不足 | POPGym/DMC 仍在运行 | yes |
| Argument Coherence | reviewer_full D3 | PARTLY_MEETS | W2; W4 | "solves"与"only model"依赖宽松判据与预算 | — | yes |
| Writing Quality | reviewer_full D5 | NOT_ASSESSED | — | 由 R1 负责 | — | no |
| Literature Integration | reviewer_full D2 | NOT_ASSESSED | — | 由 R4 负责 | — | no |
| Significance & Impact | reviewer_full D6 | NOT_ASSESSED | — | 由 R1 负责 | — | no |

## Arithmetic Receipts

no_recomputable_statistics: no test statistics, p-values, degrees of freedom or scale means with SDs are reported; results are RL returns, AUCs, update counts and energy estimates (checked Tables 1-8, Sections 3-5, Supp. A-H)
