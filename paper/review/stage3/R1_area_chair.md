# R1 — Area Chair / Editor-in-Chief 审稿报告

稿件：*Fire Without Forgetting: Persistent Dendrites Give Spiking World Models Long-Term Memory*（CVPR 2027 投稿稿，2026-10-04 版本，`main.pdf` 10 页 + `main_full.pdf` 13 页）
审稿轮次：第 1 轮（ARS Stage 3，full mode）
面板席位映射：本报告对应 ARS 面板的 Journal-Fit Reviewer 席位（dispatch role = eic），负责维度 D5、D6。

## 审稿人信息

- **身份**：CVPR Area Chair，研究方向为具身智能中的世界模型与高效神经网络（含类脑/脉冲网络在视觉与控制中的应用），长期处理 embodied AI 与 efficient ML 方向的投稿。
- **审稿重点**：CVPR 会议适配性、相对最接近前人工作的新颖性与显著性、标题性主张与证据是否相符、写作与版式（含 8 页正文上限）。方法细节与统计问题交由 R2，领域细节交由 R3/R4，此处只在影响录用判断时提及。
- **评分量表**：采用 CVPR 近年审稿表风格的 6 级总体评分：6 Accept / 5 Weak Accept / 4 Borderline Accept / 3 Borderline Reject / 2 Weak Reject / 1 Reject；置信度 1–5（5 = 完全在专业范围内）。同时给出 ARS 四级建议（Accept / Minor / Major / Reject）。
- **总体评分**：**3 — Borderline Reject**（按当前稿件；若 Revision Roadmap 中 P1 级条目完成，可升至 4）
- **ARS 建议**：Major Revision
- **置信度**：4

## 总体评价

论文指出现有脉冲世界模型（以 Spiking-WM 为代表）在每个环境步清零膜电位、并在每次胞体发放时重置树突，导致记忆只能以二值脉冲逐步重编码，时间常数形同虚设（"reset-induced amnesia"）；用 A/B/G 三种干预在世界模型探针上做了因果检验，并据此设计了持久树突神经元（PDN）：线性积分/共振树突通道从不重置、写入增益按 LRU 归一化、保留时间对数均匀分布、胞体无状态。实验覆盖 BSuite memory length、POPGym RepeatPrevious、Memory Maze 离线探针、DMC 与能耗。

从 AC 视角看，第 3 节的诊断是本文最有价值、最可能被引用的部分：机制清楚、干预设计有针对性、结果对比鲜明（Table 1）。方法本身简洁，消融覆盖了主要设计选择。主要问题在于：(i) 摘要与图 1 的若干标题性主张强于论文自身证据（"solves 101-step recall" 对应图 1 中 L=101 的 final return 约 0.40；"under the protocol of R2I" 实际改动了奖励头）；(ii) 与最接近的强基线 R2I 没有形成有效比较（BSuite 只引用已发表数字且协议不同，POPGym 复现结果在 Easy 上即处于随机水平，与 R2I 发表结果相矛盾）；(iii) 对 CVPR 读者而言，唯一的视觉 3D 基准 Memory Maze 仍在运行，当前数字显示 FwF 在墙体布局上仅比常数预测器高 0.8 个百分点；(iv) 正文称各模型"differ only in the recurrent core"，但补充材料显示 FwF 的 actor-critic 为 12.9M 参数并多读 6144 路树突脉冲，基线为 3.4M。

这些问题都可修复，核心想法有价值，因此我的建议是 Major Revision；按 CVPR 量表当前为 Borderline Reject。

contract_role: eic

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: warn
trigger: "记号、图或表存在不一致或难以阅读，或关键细节被推到补充材料而正文没有指引"

### D6: venue_fit_and_contribution
score: block
trigger: "标题性主张超出实验所能支撑的范围，但核心想法仍可能有价值"
block_class: repairable

## Review Body

以下按"优点—缺点—问题—评分理由"组织。每个缺点给出严重程度（Critical / Major / Minor）、证据锚点与修改建议。

### S1: 诊断清晰且有因果干预支撑
第 3 节把"脉冲世界模型为何记不住"拆解为步重置、发放重置与 Euler 写入增益三个可操作的机制，并分别用干预 A、B、G 检验；Table 1 中 Spiking-WM 在 τ_d∈{2,16,128} 下全部失败、A+B+G 在 32 步以内 2k–3k 次更新即学会，对比鲜明。这种"先诊断、再按诊断设计"的结构对 SNN 与世界模型两个社区都有参考价值。
**Evidence Anchor**: table: Table 1 — Spiking-WM 行全为 "-- / --"，A+B+G 行 0.5k / 0.5k / 2k–3k

### S2: 用世界模型探针把记忆与策略学习分离
Table 3 的探针去掉了探索与信用分配，RL 中的排序（LIF 8 步、GRSN 16 步、Spiking-WM-NR 与 DreamerV3 32 步、FwF 64 步）在探针中复现，支持"基线的上限来自世界模型本身"这一判断。
**Evidence Anchor**: text: §5.3 "The same ordering as in RL shows that the limits of the baselines are limits of their world models."

### S3: 对比较局限的表述大体坦诚
作者主动说明 BSuite 上与 R2I 已发表数字的比较只是参考、能耗只计算算术不计访存、所有模型都换用了按动作索引的奖励头。这种透明度值得肯定。
**Evidence Anchor**: text: §5.2 "so the comparison is indicative"

### S4: 图 2 清楚地传达了"记忆存在哪里"
图 2 并排对比 MCN 与 PDN 的信息流（步间只有脉冲跨越 vs 树突通道跨越），是一张好的方法示意图。
**Evidence Anchor**: figure: Figure 2 — (a) "only spikes cross" 与 (b) "never reset" 两个面板

### W1: 标题性主张强于论文自身证据
**Problem**: 摘要称 FwF "solves 101-step recall"，图 1 caption 称 "FwF recalls across all lengths"。但"solve"在 Table 2 中定义为评估回报"first reaches 0.9"（单种子），而图 1 右侧（final return）显示 FwF 在 L=71 与 L=101 的最终回报约为 0.46 与 0.40，图 3 的学习曲线在峰值后明显回落。0.40 的最终回报约等于 R2I 在 101 步报告的中位数（约 0.35），并不能支撑"solves"。另外，摘要中的 "under the protocol of R2I" 实际采用了与 R2I 不同的奖励头（Supp. B），使 DreamerV3 从 R2I 报告的约 30 步提升到 51 步。
**Evidence Anchor**: text: Abstract "it solves 101-step recall under the protocol of R2I, where DreamerV3 fails beyond 51 steps"
**Why it matters**: CVPR 审稿人首先看摘要与图 1。图 1 自身的数值与"solves / recalls across all lengths"不符，会直接损害全文可信度，并把讨论从诊断的价值转移到夸大问题上。
**Suggestion**: 摘要与图 1 caption 改为与证据一致的直接陈述，例如"at L=101, FwF reaches return 0.9 at 404k steps and averages 0.75 afterwards, where all other models stay near 0"；在 Table 2 增加"final return"或"post-solve mean / fraction of evaluations ≥ 0.9"列；把协议称为"R2I's lengths and budget with an action-indexed reward head"。这些都可由现有运行记录完成。
**Severity**: Major
**Confidence**: 4 — AC 视角：主张与证据的一致性

### W2: 对 CVPR 读者的适配论证偏弱，唯一的视觉基准尚不支持记忆优势
**Problem**: 引言用头戴摄像头的具身智能体作动机，但主要结果来自向量观测任务（BSuite、POPGym、本体感受 DMC）。唯一的视觉 3D 基准 Memory Maze 仍为临时结果：FwF 墙体准确率 81.6%，常数预测器 80.8%，低于全部脉冲基线（82.7–82.9%）与 DreamerV3（85.3%）；所有 S 规模模型都远低于参考 RSSM（89.0% / 94.0%）。
**Evidence Anchor**: table: Table 4 — Memory Maze 列，FwF walls 81.6（pending）对比 constant predictor 80.8%、DreamerV3 85.3
**Why it matters**: 对 CVPR 而言，视觉输入下的长时记忆是本文与会议读者之间最直接的桥梁。若最终 Memory Maze 结果仍与常数预测器接近，贡献要点 3 中"first evaluation ... on standard memory benchmarks (..., Memory Maze)"只能支撑"做了评估"，不能支撑"有记忆优势"。
**Suggestion**: 按计划报告 state 探针与仅当前图像嵌入探针之差（memory gain），并对所有模型给出；在正文明确写出 Memory Maze 的结论（无论正负）。若计划中的 DMC 对照仍有替换空间，可考虑改为 Spiking-WM 原文所用的像素 DMC 协议（同时提升视觉相关性与基线保真度）。
**Severity**: Major
**Confidence**: 4 — AC 视角：会议适配

### W3: 相对最接近前人工作的增量没有被有效证明
**Problem**: 文中把 R2I 作为"state of the art among ANN world models on memory tasks"，但 BSuite 上只引用 R2I 已发表数字（奖励头不同），POPGym 上 R2I 复现在 RepeatPrevious Easy 与 Medium 都停留在随机水平（−0.50 / −0.49），而 R2I 原文报告在 RepeatPrevious Medium/Hard 达到 SOTA。此外，PDN 的写入归一化与初始化直接取自 LRU（文中已说明），与 spiking state-space layers、以及 RL 中的 Fast and Forgetful Memory（FFM）在机制上相近，而正文对差异的论证主要基于"应用场景不同"。
**Evidence Anchor**: text: §5.1 "As the state of the art among ANN world models on memory tasks we also report R2I"
**Why it matters**: CVPR 的录用判断依赖"相对最强前人工作的增量"。当前最强 ANN 记忆世界模型实际上不在任何受控比较中，读者无法判断 FwF 的记忆能力相对 SOTA 处于什么位置。
**Suggestion**: 解释并修正 R2I 的 POPGym 复现，或在表中改用 R2I 已发表曲线在相同步数处的数值并注明来源；在相关工作中用一两句直接说明 PDN 与 LRU/spiking SSM/FFM 的具体机制差异，并指向已有的支撑证据（例如 Table 5 中"all channels resonant"——最接近全复数 LRU 的变体——在 64 步上需要近两倍更新且不稳定）。
**Severity**: Major
**Confidence**: 4 — AC 视角：新颖性与相对前人的定位

### W4: 正文对比较设置的描述不完整
**Problem**: §5.1 称所有模型"differ only in the recurrent core"，并只给出世界模型参数量（FwF 8.6M、DreamerV3 7.5M、Spiking-WM 7.2M）。补充材料 A 才说明 FwF 的 actor-critic 为 12.9M（基线 3.4M），并额外读取 6144 路树突脉冲；Table 7 显示这一输入对 POPGym 与 DMC 的 RL 结果影响很大（RepeatPrevious Easy 0.71 vs 0.11）。
**Evidence Anchor**: text: §5.1 "all models use identical hyperparameters, budgets and evaluation and differ only in the recurrent core"
**Why it matters**: 审稿人会把这一差异视为比较不公平或选择性报告。BSuite 结果不依赖这一输入（Table 7），这是论文的有利证据，却被放在补充材料里。
**Suggestion**: 在 §5.1 直接写明 actor-critic 的输入与参数量差异，并把 Table 7 的核心行移入正文或在正文中引用其结论："BSuite results are unchanged when the actor reads (z_t, h_t) only; POPGym and DMC gains require the dendritic readout"。
**Severity**: Major
**Confidence**: 4 — AC 视角：表述完整性

### W5: 页数余量为零且存在版式溢出
**Problem**: 正文恰好在第 8 页末结束（Limitations 到第 544 行），而 Memory Maze 讨论、种子数说明、POPGym/DMC 最终结果文字仍是 TODO；Table 4 在编译日志中有 37pt 的 overfull hbox。
**Evidence Anchor**: text: §5.4 "Text after the results: compare FwF with the spiking baselines and DreamerV3"
**Why it matters**: CVPR 正文上限 8 页（不含参考文献），超页将被直接拒稿；补齐内容时容易被迫删减关键证据。
**Suggestion**: 预先规划压缩：例如合并 Table 3 与 Table 5 的部分行、把 §2 "Dendrites" 段并入其他段落、缩短图 2 caption；修正 Table 4 宽度（缩小列间距或拆表）。
**Severity**: Minor
**Confidence**: 5 — 会议格式规范

### W6: 方法开发与评测任务存在重叠，正文未说明
**Problem**: 补充材料 F 说明共振通道是在"earlier version with integrator channels only"于 RepeatPrevious Medium 失败后加入的，而 RepeatPrevious Medium 随后作为主要证据之一。正文没有提及这一开发顺序。
**Evidence Anchor**: text: Supp. F "This diagnosis motivated the resonant channels"
**Why it matters**: 读者会关心设计是否针对评测任务调整；若不说明，会被 DA 型审稿人当作过拟合评测集的证据。
**Suggestion**: 在 §5.3 直接写出开发顺序，并补充一个开发阶段未使用过的记忆任务（例如 RepeatPrevious Hard 或另一个 POPGym 记忆任务）作为留出验证，模型配置保持不变。
**Severity**: Minor
**Confidence**: 3 — AC 视角：研究流程透明度

### 致作者的问题

1. 在 L=71 与 L=101 上，FwF 的最终回报（图 1 约 0.46 / 0.40）与"post-solve mean"（0.86 / 0.75）差异较大。能否给出每次评估的回合数、评估间隔，以及更严格判据（例如连续 k 次评估 ≥ 0.9 或 EMA ≥ 0.9）下的结果？
2. R2I 官方代码在 RepeatPrevious Easy 上停留在 −0.50，而其他所有模型（包括 Spiking-WM）都能学会 Easy。你们如何解释这一与 R2I 原文相矛盾的结果？
3. Memory Maze 的最终结论预期是什么？若 FwF 的 memory gain 不超过 DreamerV3，论文打算如何陈述？
4. 对 CVPR 读者而言，你们希望他们带走的最主要结论是诊断（reset-induced amnesia）还是神经元设计？标题与摘要目前更强调后者。

### 总体评分与理由

总体评分 3（Borderline Reject），置信度 4。决定性的未解决准则是 D6（贡献与主张—证据一致性）：W1 与 W3 都可通过改写与补充分析修复，W2 取决于仍在运行的 Memory Maze。诊断部分（S1、S2）足以构成一篇 CVPR 论文的核心，但当前摘要把重点放在"solves 101 / only model"这类与自身图表不完全相符的主张上。若完成 Revision Roadmap 中的 P1 条目（主张与证据对齐、R2I 比较修正、比较设置完整披露、Memory Maze 结论明确），我会把评分提高到 4。

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | reviewer_full D6 | PARTLY_MEETS | S1; W3 | 诊断新颖；神经元组件多来自 LRU，差异需证据 | 不评估 SNN 文献细节（交 R3） | yes |
| Methodological Rigor | reviewer_full D1 | NOT_ASSESSED | — | 由 R2 负责 | — | no |
| Evidence Sufficiency | reviewer_full D6 | PARTLY_MEETS | W1; W2 | BSuite 与探针证据扎实；101 步与 Memory Maze 不足 | Memory Maze 仍在运行 | yes |
| Argument Coherence | reviewer_full D3 | PARTLY_MEETS | W1; W4 | 诊断到设计的推理清楚；标题性主张偏强 | DA 负责深入挑战 | yes |
| Writing Quality | reviewer_full D5 | MEETS | S4; W5 | 写作清楚紧凑；页数无余量 | — | no |
| Literature Integration | reviewer_full D6 | PARTLY_MEETS | W3 | 缺 FFM 等 RL 记忆工作的定位 | 细节交 R4 | yes |
| Significance & Impact | reviewer_full D6 | PARTLY_MEETS | W2 | 对 SNN 社区意义明确；对 CVPR 视觉读者的意义待 Memory Maze | — | yes |

## 次要问题（非发现通道）

- 正文 §5.6 的 DMC 数字为 "[--]"，而 Table 4 已给出蓝色临时值 880 / 438 / 620，两处应同步。
- 图 1 纵轴为 "final return"，Table 2 为 "env. steps to return 0.9"，建议在图 1 caption 中说明两种度量的关系。
- 摘要中 "with time constants spanning 1 to 1000 environment steps" 描述的是初始化，训练后保留时间会改变，建议注明或报告学习后的分布。
- 补充材料目前以 `make full` 附在参考文献后；提交时需作为独立文件上传。
