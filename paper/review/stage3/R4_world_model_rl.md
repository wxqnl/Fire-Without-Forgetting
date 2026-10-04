# R4 — 领域审稿报告：世界模型 / 基于模型的强化学习 / RL 中的记忆

稿件：*Fire Without Forgetting: Persistent Dendrites Give Spiking World Models Long-Term Memory*（CVPR 2027 投稿稿，2026-10-04 版本）
审稿轮次：第 1 轮（ARS Stage 3，full mode）
面板席位映射：本报告对应 ARS 面板的 Peer Reviewer 2（Domain）席位（dispatch role = domain），负责维度 D2（领域准确性：前人工作与基准协议的表述、基线数字、相关工作覆盖）。

## 审稿人信息

- **身份**：基于模型的强化学习与部分可观测 RL 方向研究者，熟悉 DreamerV3 / RSSM、R2I（S4/S5 世界模型）、S4WM、Transformer 世界模型，以及 BSuite memory length、POPGym、Memory Maze、DMC 的协议与已发表结果；熟悉 RL 中的循环与状态空间记忆（GRU、LRU、S5、FFM）。
- **审稿重点**：前人方法与基准协议是否被准确描述；基线是否按源论文配置复现、数字是否与发表结果一致；相关工作覆盖；结论在世界模型领域的增量。
- **评分量表**：CVPR 近年审稿表风格的 6 级总体评分（6 Accept / 5 Weak Accept / 4 Borderline Accept / 3 Borderline Reject / 2 Weak Reject / 1 Reject）；置信度 1–5。
- **总体评分**：**3 — Borderline Reject**
- **ARS 建议**：Major Revision
- **置信度**：4

## 总体评价

本文在 DreamerV3 风格的脉冲世界模型中替换循环核心，在 BSuite memory length（含 R2I 的 51/71/101 步设定）、POPGym RepeatPrevious、Memory Maze 离线探针与 DMC 上评估。从世界模型角度，有几点做得好：奖励头分析（Supp. B）指出 BSuite 的"线索 × 动作"异或奖励无法被标量奖励头一阶学习，这是对 R2I 报告的 DreamerV3 失败长度的一个有价值的再解释；RepeatPrevious 探针区分了"长时间保持一个值"与"精确回忆某个延迟"两类记忆，后者确实是 GRU/DreamerV3 的弱项。

但作为 D2 的负责人，我认为对最接近的前人工作与基准协议的表述存在影响比较结论的问题：(i) R2I 的 POPGym 复现在 RepeatPrevious Easy 与 Medium 都停留在随机水平，而 R2I 原文报告在 RepeatPrevious Medium/Hard 上达到 SOTA，论文未解释这一矛盾；(ii) "under the protocol of R2I"实际改动了奖励头，并把单种子结果与 R2I 十种子中位数并列；(iii) 称 DMC 本体感受是"Spiking-WM 被评估过的基准"，而 Spiking-WM 原文使用像素 DMC，且本文复现数字远低于其发表值；(iv) Memory Maze 的临时结果中所有 S 规模模型都接近常数预测器，目前无法支持记忆结论；(v) 缺少 Fast and Forgetful Memory（FFM）等机制上紧密相关的 RL 记忆工作。这些都可修复，建议 Major Revision。

contract_role: domain

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: block
trigger: "前人工作或基准协议被误述且影响比较结论，或紧密相关的方法既未讨论也未比较"
block_class: repairable

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

以下从世界模型与 RL 记忆的领域角度给出优点与缺点。每个缺点注明严重程度、证据锚点与修改建议。

### S1: 奖励头分析是对 BSuite memory length 的有价值再解释
作者用探针表明，完全训练的 ANN 世界模型在 h_t 中能线性解码线索（0.96–1.00）与上一动作（0.94–0.97），奖励 AUC 却停在 0.49–0.52；改为按动作索引的奖励头后问题消失。这说明 R2I 报告的 DreamerV3 在约 30 步以上失败，部分源于奖励头而非记忆。作者对所有模型统一使用该头并明确披露其影响，处理得当。
**Evidence Anchor**: text: Supp. B "The obstacle is the reward head, not memory."

### S2: 两类记忆的区分有洞见
RepeatPrevious 探针要求回忆"恰好 k 步前"的输入，而非长时间保持一个值；k=32 时只有带共振通道的 FwF 学会（AUC 0.93），DreamerV3 与 GRSN 停在 0.50–0.55，且 Supp. F 显示 DreamerV3 在 RL 末期对 Medium 的奖励 AUC 只有 0.65。这为"世界模型在 POPGym Medium 上失败的原因"提供了具体诊断。
**Evidence Anchor**: text: §5.3 "RepeatPrevious asks for a different memory: not one value held for long, but the input of an exact lag among a stream of similar inputs."

### S3: 使用有已发表协议的基准，并核对了常数预测器
Memory Maze 使用官方离线数据集与官方探针代码，常数预测器（24.0 / 80.8%）与原文（23.9 / 80.8%）一致，说明评估管线搭建正确。
**Evidence Anchor**: text: Supp. G "our constant predictors (objects 24.0, walls 80.8%) match those of the paper (23.9, 80.8%)"

### S4: 对 R2I 运行细节的透明披露
作者说明 R2I 使用官方 POPGym 配置，只改了预算与回放容量（保证不淘汰样本），并因其代码没有独立评估而改用训练回合回报，同时给出两种度量在自家模型上的差异上界（0.04）。
**Evidence Anchor**: text: Supp. A "R2I itself runs with the official POPGym configuration of its code (batches of 4×1024 steps), changed only in the budget"

### W1: R2I 的 POPGym 复现与其发表结果相矛盾，且未被解释
**Problem**: Table 2 中 R2I 在 RepeatPrevious Easy 为 −0.50、Medium 为 −0.49，即完全随机；而在同一任务上，所有其他模型（包括从不记住 8 步线索的 Spiking-WM）都能学会 Easy。R2I 原文（Samsami et al., ICLR 2024, §4.2 与图 4）报告 R2I 在 RepeatPrevious 上表现稳定强劲，并在 Medium 与 Hard 上取得新的 SOTA。正文"consistent with R2I"只适用于 DreamerV3 在 Medium 上失败这一点。
**Evidence Anchor**: table: Table 2 — R2I (ANN) 行 RepeatPrevious Easy −0.50‡、Medium −0.49‡
**Why it matters**: 把一个明显失败的运行作为"state of the art among ANN world models"放进主表，会让读者误以为 R2I 在 RepeatPrevious 上失败，也会让熟悉 R2I 的审稿人质疑全部基线复现的可靠性。
**Suggestion**: (1) 排查运行：2M 步相对 R2I 官方配置（回放 10^7、batch 4×1024）可能远未达到其收敛所需步数；检查奖励缩放、观测格式与环境版本；(2) 在表中并列 R2I 原文曲线在 2M 步处与最终处的数值并注明来源；(3) 若在投稿前无法得到与原文一致的结果，应删除该行，改为在正文以原文数字作参考，而不是保留一个与原文矛盾的复现。
**Severity**: Major
**Confidence**: 5 — core expertise: R2I 与 POPGym

### W2: "protocol of R2I"被修改，且单种子结果与十种子中位数并列
**Problem**: BSuite 51/71/101 步使用了按动作索引的奖励头（R2I 使用 DreamerV3 默认头），且长度多计一步；摘要据此称 FwF 在"R2I 的协议下"解决 101 步、DreamerV3 超过 51 步失败。§5.2 把单种子的 FwF 与 R2I 十种子中位数（约 0.9 / 0.9 / 0.35）并列，作者自己也称"indicative"。
**Evidence Anchor**: text: §5.1 "all models therefore use a reward head indexed by the action"
**Why it matters**: 奖励头的改动足以把 DreamerV3 从约 30 步提升到 51 步，因此"R2I 的协议"这一说法会误导读者以为结果可与 R2I 的数字直接比较；按图 1，FwF 在 101 步的最终回报约 0.40，与 R2I 的约 0.35 处于同一水平。
**Suggestion**: 把协议改称"R2I's lengths, budget and seeding, with an action-indexed reward head"；在 Table 2 中以单独一行列出 R2I 原文在 1M 步时的中位数并注明"10 seeds, default reward head"；比较时使用最终回报与解决后的均值，而非"首次达到 0.9"。可选的追加运行：用 R2I 官方代码与其 BSuite 配置在 51/71/101 上各跑一次，作为同一环境实现下的直接参考。
**Severity**: Major
**Confidence**: 5 — core expertise: BSuite memory length 与 R2I 协议

### W3: Spiking-WM 的评估背景被误述，复现数字远低于其发表值
**Problem**: 正文称本体感受 DMC 是"Spiking-WM 被评估过的"非记忆对照。Spiking-WM 原文（PNAS 2025；arXiv 2503.00713 Table I）评估的是像素 DMC，1M frames，约 18.8M 参数，报告 walker walk 938.5、cheetah run 577.2、cartpole swingup 791.0。本文在更容易的本体感受设置下复现得到 923 / 343 / 497（S 规模，7.2M）。
**Evidence Anchor**: table: Table 4 — Spiking-WM 行 DMC walk 923、run 343、swingup 497
**Why it matters**: §5.6 的预注册判据是"no degradation relative to Spiking-WM, whose benchmark this is"。基准与配置都与源论文不同，且基线明显弱于其发表值，使判据失去意义。
**Suggestion**: 更正表述，并在 Table 4 中并列 Spiking-WM 原文数字与协议说明；若 DMC 对照仍可调整，改用 Spiking-WM 原文的像素 DMC 协议与其配置，这同时符合"采用源论文标准配置"的原则并提升与 CVPR 视觉读者的相关性（新运行）；判据同时相对 DreamerV3 陈述。
**Severity**: Major
**Confidence**: 5 — core expertise: DMC 与 Dreamer 系基线；Spiking-WM 原文数字已核对

### W4: Memory Maze 的当前结果无法支持记忆结论
**Problem**: 临时结果中，FwF 墙体准确率 81.6%（常数预测器 80.8%），低于全部脉冲基线（82.7–82.9%）与 DreamerV3（85.3%）；目标物体 MSE 10.7，优于脉冲基线（11.3–12.1）但不及 DreamerV3（9.8）。所有 S 规模模型都远离参考 RSSM（8.3 / 89.0%；TBTT 6.2 / 94.0%）。memory gain（相对仅当前图像嵌入的探针）尚未报告。
**Evidence Anchor**: text: Supp. G "All models use the size of DreamerV3-S, smaller than the RSSM of the reference results (2048-dimensional state)."
**Why it matters**: Memory Maze 是本文唯一的视觉、长回合（1000 步）记忆基准，也是贡献要点 3 的组成部分。若墙体布局几乎学不到，这个基准目前测的更多是模型规模与训练量，而非记忆机制。
**Suggestion**: 按计划报告所有模型的 memory gain（state 探针减 image-only 探针），并对 FwF 区分"spikes only"与"spikes + dendritic spikes"；以目标物体 MSE 为主要记忆指标（它更依赖跨视野记忆）；在正文直接陈述结论，包括"S 规模模型在 1M 次更新内未学到墙体布局"这一事实及其与参考 RSSM 的差距来源（状态维度 512 vs 2048）。
**Severity**: Major
**Confidence**: 4 — core expertise: Memory Maze 协议；结果仍为临时值

### W5: 缺少机制上紧密相关的 RL 记忆工作
**Problem**: Fast and Forgetful Memory（Morad et al., NeurIPS 2023）为部分可观测 RL 设计了带衰减与"contextual drift"频率的复值聚合器，以对数间隔的时间尺度初始化，并在 POPGym 上取得最佳 MMER；这与 PDN 的"spectrum of retention times"和"resonance tells time"高度相似。相关工作也未讨论 Legendre Memory Units（精确延迟窗口）、S5 用于 in-context RL 的工作（Lu et al., NeurIPS 2023，处理扫描中的回合边界重置），以及 POPGym 自身报告的 LRU/S4D 基线。
**Evidence Anchor**: absence: §2 Related Work 与 refs.bib — expected Fast and Forgetful Memory、LMU 与 S5 in-context RL 的引用与讨论; checked §2 全部四段, §4.2, refs.bib 全部 57 个条目
**Why it matters**: 论文把"按环境步计时的对数均匀保留时间 + 相位编码延迟"作为设计亮点；FFM 已在同一类 POMDP 基准上提出了结构相近的先验，审稿人会据此质疑新颖性。
**Suggestion**: 在 §2 与 §4.2 用直接陈述说明差异，例如 FFM 是带门控与复值输出的稠密 ANN 单元，PDN 的输出是二值脉冲、无门控、把实值积分与复值共振分开并以 LRU 方式归一化写入；可选：把 FFM 单元作为 ANN 参考加入低成本的世界模型探针。
**Severity**: Major
**Confidence**: 4 — core expertise: RL 记忆架构文献

### W6: "DreamerV3"指代的是本文的重实现
**Problem**: 文中的 DreamerV3 是 Spiking-WM 代码库中去掉脉冲的同构网络（S 规模、SiLU、单仿真步、BSuite 上换了奖励头），并非官方 DreamerV3。
**Evidence Anchor**: text: §5.1 "the same architecture without spikes (GRU core, SiLU activations, one simulation step), an ANN reference"
**Why it matters**: 读者会把"DreamerV3 fails beyond 51 steps"理解为官方 DreamerV3 的性质。
**Suggestion**: 在表与正文中称"DreamerV3 architecture (our implementation)"，并在首次出现时说明与官方实现的差异。
**Severity**: Minor
**Confidence**: 5 — core expertise: DreamerV3

### W7: POPGym 任务覆盖偏窄
**Problem**: 只评估 RepeatPrevious Easy/Medium；R2I 在同一类实验中还评估了 Hard 以及 Autoencode、Concentration。
**Evidence Anchor**: text: §5.1 "one of the three most memory-demanding POPGym tasks that R2I evaluates"
**Why it matters**: FwF 的保留时间可达 1000 步，Hard 是检验其优势的自然设置；只报告 Easy/Medium 会让人怀疑任务选择。
**Suggestion**: 在不改变模型配置的前提下补充 RepeatPrevious Hard（至少 FwF 与 DreamerV3），作为方法开发中未使用过的留出任务。
**Severity**: Minor
**Confidence**: 4 — core expertise: POPGym

### W8: Spiking-WM 的树突发放重置应指明出处
**Problem**: §3.1 称胞体发放时胞体与两个树突都被置零。Spiking-WM 论文的公式只写出胞体重置 U[t](1−S)；树突重置出现在其官方实现中（`node.py` 中 `MCNode.calc_spike` 对 basal 与 apical 同样乘以 1−S）。
**Evidence Anchor**: text: §3.1 "When the soma fires, the soma and both dendrites are set to zero."
**Why it matters**: 诊断的一半依赖这一重置；只读过 Spiking-WM 论文的读者可能认为描述有误。
**Suggestion**: 加脚注说明该重置来自官方代码并给出位置。
**Severity**: Minor
**Confidence**: 5 — 已核对 Spiking-WM 官方代码

### W9: BSuite 短预算设置的并行环境共享种子不符合常规
**Problem**: L≤64 的运行在四个并行环境间共享一个种子，L≥51 则按 R2I 每个环境不同种子。
**Evidence Anchor**: text: Supp. A "runs share one seed across the four parallel environments, identically for all models"
**Why it matters**: 共享种子可能使并行环境产生相关甚至相同的线索序列，影响数据多样性；读者难以把短预算结果与标准 BSuite 设置对照。
**Suggestion**: 说明原因与影响；若共享种子导致相同回合，在正文明确写出，并以 R2I 协议的结果为主要证据。
**Severity**: Minor
**Confidence**: 3 — 依赖实现细节

### 致作者的问题

1. R2I 在 RepeatPrevious Easy 上为 −0.50，你们认为原因是预算、配置还是实现差异？R2I 官方配置下达到其发表结果需要多少环境步？
2. 若用 R2I 官方代码与默认奖励头在你们的 BSuite 实现上运行 51/71/101 步，结果是否与其发表的中位数一致？
3. Memory Maze 上 FwF 的 memory gain 相对 DreamerV3 如何？墙体准确率接近常数预测器是否意味着 S 规模模型在 1M 次更新内没有学到布局？
4. FwF 与 FFM 的保留时间/频率参数化有何本质差异？

### 总体评分与理由

总体评分 3（Borderline Reject），置信度 4。决定性的未解决准则是 D2：W1–W3 都涉及对前人工作或协议的不准确表述，且直接影响比较结论；W4 取决于仍在运行的 Memory Maze；W5 影响新颖性定位。S1 与 S2 是对世界模型记忆研究有实际价值的发现。若 R2I 比较得到修正、协议命名准确、Spiking-WM 背景更正、Memory Maze 给出明确的 memory gain 结论、并补上 FFM 的定位，我会升至 4。

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | reviewer_full D2 | PARTLY_MEETS | S2; W5 | 两类记忆区分新颖；保留时间谱与相位编码在 FFM 中已有相近先验 | — | yes |
| Methodological Rigor | reviewer_full D1 | NOT_ASSESSED | — | 由 R2 负责 | — | no |
| Evidence Sufficiency | reviewer_full D2 | PARTLY_MEETS | W1; W4 | R2I 比较无效；Memory Maze 尚无结论 | Memory Maze/POPGym 仍在运行 | yes |
| Argument Coherence | reviewer_full D3 | NOT_ASSESSED | — | 由 DA 与 R2 负责 | — | no |
| Writing Quality | reviewer_full D5 | NOT_ASSESSED | — | 由 R1 负责 | — | no |
| Literature Integration | reviewer_full D2 | DOES_NOT_MEET | W5; W8 | 缺少 FFM、LMU、S5 in-context RL | — | yes |
| Significance & Impact | reviewer_full D2 | PARTLY_MEETS | S1; S2 | 对脉冲世界模型有意义；相对 ANN SOTA 的位置不明 | — | yes |
