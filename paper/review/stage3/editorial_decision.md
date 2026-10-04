# Editorial Decision（ARS Stage 3，full mode）

## 稿件信息

- **标题**：Fire Without Forgetting: Persistent Dendrites Give Spiking World Models Long-Term Memory
- **目标会议**：CVPR 2027（正文上限 8 页，不含参考文献）
- **审阅版本**：2026-10-04 编译的 `main.pdf`（10 页）与 `main_full.pdf`（13 页）及其 LaTeX 源
- **决定日期**：2026-10-04
- **审稿轮次**：第 1 轮
- **稿件状态说明**：蓝色 `\pend{}` 数字（POPGym、DMC、Memory Maze、一项附录消融）与红色 TODO 为进行中的运行，未因其缺失而扣分；但已依赖临时数字的主张被单独标出。

## 审稿面板与执行溯源（#540/#740）

面板席位与本次任务的审稿人编号对应如下（ARS 内部席位 ID 在括号中）：

| 本任务编号 | ARS 席位 | dispatch role | 负责维度 | 报告文件 |
|---|---|---|---|---|
| R1 Area Chair / Editor-in-Chief | EIC（Journal-Fit Reviewer） | eic | D5, D6 | `R1_area_chair.md` |
| R2 方法 / 实验设计 | R1（Methodology） | methodology | D1, D3 | `R2_methods.md` |
| R3 SNN / 类脑计算 | R3（Perspective） | perspective | D4 | `R3_snn.md` |
| R4 世界模型 / MBRL / RL 记忆 | R2（Domain） | domain | D2 | `R4_world_model_rl.md` |
| R5 Devil's Advocate | DA | da | D3 | `R5_devils_advocate.md` |

- **Typed artifact**：`review/stage3/contract/review_panel_provenance.json`（`review_panel_provenance.py validate`：PASS）
- **Artifact SHA-256**：`26ec29dfa592e30975f2605efba7d23e60ec0db5bacbf0f1d35bdedf4231bdae`
- **Panel ID**：`fwf-cvpr2027-stage3-round1`
- **Normalized manifest SHA-256**：`18667c76c411ae35a658fb20cca8a6f36e9eb6c6b2bf849dc7038f874a0a207a`
- **Execution topology SHA-256**：`e72b69338e5c4e02cadc88769b1dbae949ebc5dd3016cd0c3e34b1a7c5140bac`
- **Fresh-context scope**：`within_panel_attempt_only`

| Seat | Role ID | Actor type | Context ID | Peer outputs visible | Model family | Provider | Human reviewer ID |
|---|---|---|---|---|---|---|---|
| EIC | eic | model | single-agent-context-78e6225c | true | claude-opus-5-5 | anthropic | unknown（null） |
| R1 | methodology | model | single-agent-context-78e6225c | true | claude-opus-5-5 | anthropic | unknown（null） |
| R2 | domain | model | single-agent-context-78e6225c | true | claude-opus-5-5 | anthropic | unknown（null） |
| R3 | perspective | model | single-agent-context-78e6225c | true | claude-opus-5-5 | anthropic | unknown（null） |
| DA | da | model | single-agent-context-78e6225c | true | claude-opus-5-5 | anthropic | unknown（null） |

| Provenance axis | Status |
|---|---|
| Role-separated | true |
| Within-panel invocation-context separation | false |
| Blind to peer outputs | false |
| Model-family distinct | false |
| Provider distinct | false |
| Human-reviewer distinct | false |

- **Binary independence claim**：Not computed. Persona or role diversity proves only `role_separated`; do not relabel the panel as independent.
- **Correlated-error disclosure**：All model-executed review seats used one model family; role separation does not remove correlated-error risk.
- **执行方式说明**：五个席位由同一个 agent 在同一上下文中依次完成，因此"上下文分离"与"对同伴输出盲"两个轴均为 false。Phase 1（paper-blind 预承诺，`review/stage3/contract/phase1_*.md`）在阅读稿件之前写成，并通过 `check_phase_conformance.py --phase1-only`；但撰写者此前已看到任务说明中对论文方法与实验的概要，因此 Phase 1 只对稿件正文是盲的，对论文内容并非完全盲。五份 Phase 2 报告均通过 `check_phase_conformance.py`，面板通过 `check_panel_synthesis.py --layer1-only`。
- **Criteria binding**：`criteria_binding_unavailable`（未提供 #684 目标准则绑定；本决定不作基于外部准则登记表的会议适配声明，会议适配判断来自 R1 的 D6 评审）。
- **Cross-model 检查**：未启用（`ARS_CROSS_MODEL` 未设置，且无跨模型上传同意）。

---

## 决定

### Major Revision

（按 CVPR 的录用语境：当前版本若直接投稿，面板的多数评分处于 Borderline Reject；问题均可修复，修复后有望进入 Borderline Accept 及以上。）

---

## 阻碍录用的问题（3 项，按 Roadmap 来源顺序）

| Transport ref | 阻碍问题 | 来源审稿人 | 证据锚点 | 对应 Roadmap 条目 |
|---|---|---|---|---|
| B1 | 标题性主张强于论文自身图表（"solves 101-step recall"、"recalls across all lengths"、"under the protocol of R2I"、"only model ... 64-step"） | R1, R2, R4, DA | figure: Figure 1 right panel — FwF final return ≈0.46 (L=71), ≈0.40 (L=101) | RM-1, RM-4 |
| B2 | 正文称各模型"differ only in the recurrent core"，但 FwF 的 actor-critic 为 12.9M（基线 3.4M）且多读 6144 路树突编码输入；POPGym/DMC 的 RL 收益依赖该输入 | R1, R2, R3, DA（C1） | text: §5.1 "differ only in the recurrent core, whose state the actor and critic read" | RM-2 |
| B3 | 作为 ANN 记忆 SOTA 的 R2I 未被有效比较：POPGym 复现为随机水平，与其发表结果矛盾；BSuite 只用已发表数字且协议不同 | R1, R2, R4, DA（C2） | table: Table 2 — R2I (ANN) 行 −0.50‡ / −0.49‡ | RM-3 |

---

## 审稿人汇总

评分量表：CVPR 近年审稿表风格 6 级（6 Accept / 5 Weak Accept / 4 Borderline Accept / 3 Borderline Reject / 2 Weak Reject / 1 Reject）；置信度 1–5。

| 审稿人 | 身份 | CVPR 评分 | ARS 建议 | 置信度 | 维度评分 |
|---|---|---|---|---|---|
| R1 Area Chair | CVPR AC，具身世界模型与高效网络 | 3 Borderline Reject | Major Revision | 4 | D5 warn；D6 block（repairable） |
| R2 方法 | RL 实验方法学 | 3 Borderline Reject | Major Revision | 4 | D1 block（repairable）；D3 warn |
| R3 SNN | 脉冲网络与神经形态计算 | 4 Borderline Accept | Major Revision | 4 | D4 warn |
| R4 世界模型/RL | MBRL 与 POMDP 记忆 | 3 Borderline Reject | Major Revision | 4 | D2 block（repairable） |
| R5 Devil's Advocate | 固定对抗席位 | 2 Weak Reject（应任务要求给出） | N/A — 只给发现 | 4 | D3 block（repairable）；CRITICAL C1、C2 |

---

## 共识分析

按 ARS 规则，共识只在四位非 DA 审稿人（R1、R2、R3、R4）之间按子主张计算；"未提及"不计为同意。DA 的发现单独处理。

### [CONSENSUS-3]（3/4 同意，第 4 位未提及）

1. **SC-1 标题性主张"solves 101 / recalls across all lengths"强于证据**——R1（W1：图 1 最终回报约 0.40）、R2（W2：单种子与"first reaches 0.9"判据宽松）、R4（W2：0.40 与 R2I 的约 0.35 同一水平）；R3 未提及。DA M1 同向。
2. **SC-2 actor-critic 容量与输入不对等，且正文未披露**——R1（W4）、R2（W1）、R3（W1：控制性能依赖 11 级树突读出；W3：actor 约 3.8 倍）；R4 未提及。DA C1 同向。
3. **SC-3 与 R2I 的比较无效**——R1（W3）、R4（W1 POPGym 复现矛盾；W2 BSuite 协议）、R2（W3：R2I 预算偏离官方配置）；R3 未提及。DA C2 同向。
4. **SC-6 相对最接近前人工作（LRU、spiking linear-recurrence 层、FFM）的机制差异缺少证据**——R1（W3）、R3（W4）、R4（W5）；R2 未提及。DA M5 同向。

### 双人佐证的发现（2/4，无冲突）

- **SC-1b "protocol of R2I"命名不准确（奖励头已改）**：R1 W1、R4 W2。
- **SC-4 Spiking-WM 基线偏离源配置、DMC 背景被误述**：R2 W3、R4 W3（R1 W2 在建议中提到像素 DMC，但未作为缺陷提出）。DA M8 同向。
- **SC-5 Memory Maze 当前结果不支持记忆结论 / CVPR 适配**：R1 W2、R4 W4。DA M4 同向。
- **SC-9 单种子缺乏变异性报告**：R2 W2、R1 W1。DA M9 同向。
- **SC-10 开发与评测任务重叠，缺留出任务**：R1 W6、R4 W7。DA M3 同向。
- **SC-19 L≤64 并行环境共享种子**：R2 W4、R4 W9。

### 单人发现（1/4，无冲突，按其准则与证据评估）

- **SC-7 "只有二值脉冲离开神经元""分级值不跨突触"与 ON/OFF 读出及 LayerNorm 不符**（R3 W1、W2；DA M6 同向）——证据锚点充分，保留为 Major。
- **SC-8 能耗范围只含滤波器、只测 BSuite L=32**（R3 W3；DA M7 同向）——保留为 Major。
- **SC-11 "only model 64"依赖 500k 预算**（R2 W4；DA M2 同向）——保留为 Major。
- **SC-5b Memory Maze 探针输入不对等**（R2 W8）、**干预表不完整**（R2 W5）、**保留时间谱必要性**（R2 W6）、**复现细节**（R2 W7）、**POPGym 度量不一致**（R2 W9）、**页数余量与版式**（R1 W5）、**发放稀疏需量化 / 学习后 τ、θ / 生物类比 / 突触延迟 / 替代梯度细节**（R3 W5–W9）、**DreamerV3 命名 / Spiking-WM 树突重置出处**（R4 W6、W8）——均为 Minor。

### 分歧

**分歧 1：总体评分方向**
- **R3 观点**：第 3 节诊断对 SNN 社区有独立发表价值，能耗报告惯例诚实，问题主要是主张精确性，给 4（Borderline Accept）。
- **R1/R2/R4 观点**：诊断有价值，但标题性主张、比较设置与 R2I 比较存在可修复但阻碍录用的问题，给 3（Borderline Reject）。DA 给 2。
- **分歧类型**：Perspective difference。
- **编辑裁决**：两方对缺陷本身没有争议（R3 自己的 W1–W4 也是 Major），差别在于对诊断价值的权重。决定按机械规则为 Major Revision，与两方都相容。
- **裁决理由**：证据优先——所有阻碍问题都有锚点且被至少三位审稿人或 DA 支持；R3 的正面判断（S1–S4）同样被 R1 的 S1、R2 的 S2 佐证，应在修改中保留并突出。

**分歧 2：actor-critic 不对等的严重程度（R2 Major vs DA Critical）**
- **R2 观点**：BSuite 主结论在策略只读 (z_t, h_t) 时依然成立，核心主张存活，故为 Major。
- **DA 观点**：正文错误描述了核心比较协议，未更正时无法录用，故为 Critical。
- **分歧类型**：Severity disagreement。
- **编辑裁决**：两者相容：该缺陷不推翻 BSuite 与探针支持的核心主张（R2 正确），但在更正前构成录用障碍（DA 正确）。按 Critical 的"未更正即无法录用"定义，DA 的 C1 被判定为 VALIDATED，且为 repairable。

**分歧 3：R2I 比较问题的严重程度（R4 Major vs DA Critical）**
- **R4 观点**：核心主张不依赖 R2I，问题在于前人工作表述与比较定位，故为 Major。
- **DA 观点**：主表对最强前人方法作出与其发表结果相反的陈述，未更正时无法录用，故为 Critical。
- **分歧类型**：Severity disagreement。
- **编辑裁决**：同分歧 2 的逻辑，C2 判定为 VALIDATED（repairable）。修复路径（解释或移除失败运行、并列已发表数字）明确。

---

## Devil's Advocate CRITICAL 发现的裁决

**C1（比较设置被错误描述；POPGym/DMC 的 RL 收益无法归因于世界模型）**
- DA 论点：§5.1 称只差循环核心，实为 actor-critic 12.9M vs 3.4M 且多读 6144 路输入；Table 7 显示去掉读出后 Easy 0.71→0.11、Medium 0.45→−0.03、walker 880→504。
- 其他审稿人佐证：R1 W4、R2 W1、R3 W1/W3。
- Journal-Fit Reviewer（R1）评估：成立。BSuite 结论不受影响；POPGym/DMC 的 RL 主张需披露并以完成的容量对照支撑。
- 裁决：**VALIDATED**（repairable；对应 RM-2）。作者必须回应。

**C2（R2I 复现与其发表结果矛盾）**
- DA 论点：R2I 在 RepeatPrevious Easy/Medium 均为随机水平，而 R2I 原文报告其在 RepeatPrevious Medium/Hard 达到 SOTA；Easy 上连 Spiking-WM 都能学会。
- 其他审稿人佐证：R1 W3、R4 W1（已核对 R2I 原文 §4.2 的结论表述）。
- Journal-Fit Reviewer（R1）评估：成立。主表不能保留一个与原文矛盾且未解释的复现。
- 裁决：**VALIDATED**（repairable；对应 RM-3）。作者必须回应。

---

## 决定理由

机械规则（reviewer_full v2 合同）先行：D1（R2）、D2（R4）、D3（DA）与 D6（R1）四个 mandatory 维度均有 repairable block，F2（"any mandatory dimension scores 'block'"）触发，严重度 90，动作 Major Revision；F3 与 F5 也触发但严重度更低；没有任何 fatal block，故不触发 Reject。

实质上，四位非 DA 审稿人在三点上形成 CONSENSUS-3：标题性主张强于证据（SC-1）、actor-critic 不对等未披露（SC-2）、R2I 比较无效（SC-3）；DA 的两项 CRITICAL 与 SC-2、SC-3 重合并被判定成立。这些缺陷都可修复：BSuite 结论在策略只读脉冲时依然成立（R2 S4），世界模型探针独立支持 64 步记忆（R1 S2、R2 S1），第 3 节诊断被所有审稿人认可（R1 S1、R2 S2、R3 S1–S2）。因此不选择 Reject。

不选择 Minor Revision，是因为修复需要实质性的再分析与少量新运行：R2I 比较的重新处理、宽 actor 对照扩展到 Medium、Memory Maze 结论的完成、Spiking-WM/DMC 对照的更正，以及与 LRU/FFM/spiking linear-recurrence 层的机制差异证据（SC-6）。此外 Memory Maze 与 POPGym 的最终数字尚未产生，其结论可能改变贡献要点 3 的措辞。

作者已固定的决定（单种子、FwF 单一配置、不调参、源论文标准配置、不做事件相机、不以 spiking SSM 定位）在 Roadmap 中均被尊重：相关关切各只陈述一次，缓解方案限于这些约束之内。

---

## 必须修改（must_fix）

| Transport ref | 修改项 | 子主张 | 严重度（转运） | 证据锚点 | 置信度 | 来源 | Obligation | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 标题性主张与图表对齐 | SC-1, SC-1b, SC-11 | major | figure: Figure 1 right panel — FwF final return ≈0.40 at L=101 | 4–5 | EIC, R1(方法), R2(领域), DA | must_fix | sentence + section（摘要、图 1、§1、§5.2、§6） | 关闭 B1 |
| R2 | 披露并对照 actor-critic 不对等 | SC-2 | critical（DA）/ major | text: §5.1 "differ only in the recurrent core" | 5 | EIC, R1(方法), R3, DA | must_fix | section + new_data（Medium 宽 actor 对照） | 关闭 B2 / C1 |
| R3 | 修正 R2I 比较 | SC-3, SC-1b | critical（DA）/ major | table: Table 2 — R2I 行 −0.50‡ / −0.49‡ | 5 | EIC, R1(方法), R2(领域), DA | must_fix | re_analysis（可选 new_data） | 关闭 B3 / C2 |
| R4 | 单种子下的变异性报告与更严格判据 | SC-9 | major | text: §5.1 "Number of seeds: one so far" | 5 | R1(方法), EIC, DA | must_fix | re_analysis（现有日志） | 支撑 B1 |
| R5 | 完成 Memory Maze 并直接陈述结论 | SC-5, SC-5b | major | table: Table 4 — FwF walls 81.6 vs constant 80.8% | 4 | EIC, R2(领域), R1(方法), DA | must_fix | section（计划中的运行） | 贡献要点 3 的措辞 |
| R6 | 更正 Spiking-WM 背景与 DMC 对照 | SC-4 | major | text: §5.1 "is a non-memory control on which Spiking-WM was evaluated" | 4–5 | R1(方法), R2(领域), DA | must_fix | sentence + table（可选 new_data） | DMC 判据有效性 |

（表中"R1(方法)""R2(领域)"为 ARS 席位编号；对应本任务的 R2、R4。）

### 必须修改项详情

**R1：标题性主张与图表对齐**
- **Problem**：摘要/图 1 的"solves 101""recalls across all lengths""under the protocol of R2I""only model ... 64"强于图 1 最终回报与预算设置。
- **Source**：R1 W1；R2 W2、W4；R4 W2；DA M1、M2。
- **Requirement**：按 Roadmap RM-1 改写，并在 Table 2 增加最终回报与解决后统计。
- **Acceptance criteria**：摘要、图 1 caption、引言与结论中每个数字性主张都能在表或图中找到同一度量下的直接对应，且协议命名与 Supp. B 一致。

**R2：披露并对照 actor-critic 不对等**
- **Problem**：正文称只差循环核心，实际 actor-critic 与输入不同；POPGym/DMC 收益依赖读出。
- **Source**：R1 W4；R2 W1；R3 W1、W3；DA C1。
- **Requirement**：按 RM-2 在 §5.1 披露、把 Table 7 结论移入正文、完成 Easy 与 Medium 的宽 actor 对照。
- **Acceptance criteria**：§5.1 写明两种 actor 的输入与参数量，正文明确区分"不依赖读出的结论（BSuite、探针）"与"依赖读出的结论（POPGym/DMC RL）"，且宽 actor 对照在 Easy 与 Medium 上有最终数字。

**R3：修正 R2I 比较**
- **Problem**：R2I POPGym 复现与原文矛盾；BSuite 比较协议不同。
- **Source**：R1 W3；R4 W1、W2；R2 W3；DA C2。
- **Requirement**：按 RM-3 处理（排查或移除失败运行、并列已发表数字、重命名协议）。
- **Acceptance criteria**：Table 2 中不存在与 R2I 原文结论矛盾且未解释的条目；R2I 的已发表数字以注明来源与种子数的单独行出现。

**R4：单种子下的变异性报告与更严格判据**
- **Problem**：单种子、首次越过 0.9 的判据、评估回合数未报告。
- **Source**：R2 W2；R1 W1；DA M9。
- **Requirement**：按 RM-4 从现有日志补充。
- **Acceptance criteria**：每个 BSuite 长度都报告评估回合数、最终回报、解决后均值/标准差与 ≥0.9 比例，并给出至少一种更严格判据下的结果；探针 AUC 有 bootstrap 区间。

**R5：完成 Memory Maze 并直接陈述结论**
- **Problem**：临时结果中墙体准确率接近常数预测器，memory gain 未报告，探针输入不对等。
- **Source**：R1 W2；R4 W4；R2 W8；DA M4。
- **Requirement**：按 RM-5 完成。
- **Acceptance criteria**：正文报告所有模型的 memory gain 与 FwF 两种特征集的探针结果，并有一句直接结论；贡献要点 3 的措辞与该结论一致。

**R6：更正 Spiking-WM 背景与 DMC 对照**
- **Problem**：Spiking-WM 原文为像素 DMC、约 18.8M 参数，本文复现数字远低于其发表值，而 DMC 判据相对它设定。
- **Source**：R2 W3；R4 W3；DA M8。
- **Requirement**：按 RM-6 处理。
- **Acceptance criteria**：Table 4 或其注释并列 Spiking-WM 原文数字与协议差异；DMC 结论同时相对 DreamerV3 陈述。

---

## 建议修改（should_fix / consider）

| Transport ref | 修改项 | 子主张 | 严重度 | 来源 | Obligation | Cost scope |
|---|---|---|---|---|---|---|
| S1 | 以证据支撑与 LRU/FFM/spiking linear-recurrence 层的机制差异 | SC-6 | major | EIC, R2(领域), R3, DA | should_fix | section + new_data（低成本探针） |
| S2 | 精确化"只有脉冲离开神经元""分级值不跨突触"的表述 | SC-7 | major | R3, DA | should_fix | sentence |
| S3 | 扩展能耗计量范围（滤波器 + actor，按基准） | SC-8 | major | R3, DA | should_fix | re_analysis（现有检查点） |
| S4 | 预算限定"only model 64"并说明共享种子 | SC-11, SC-19 | major | R1(方法), R2(领域), DA | should_fix | sentence（可选 new_data） |
| S5 | 披露开发顺序并增加留出记忆任务 | SC-10 | minor | EIC, R2(领域), DA | should_fix | section + new_data |
| S6 | 补全干预表、保留时间谱必要性探针、学习后 τ/θ 分析 | — | minor | R1(方法), R3 | consider | re_analysis + 低成本 new_data |
| S7 | 复现细节、术语、引用出处、版式与页数规划 | — | minor | EIC, R1(方法), R2(领域), R3 | should_fix | sentence |

---

## Revision Roadmap：来源追溯清单（不代表工作顺序）

> 本清单按来源顺序列出、不排工作优先级（ARS 规范）。按本次任务要求另行给出的**优先级 Roadmap** 见 `revision_roadmap.md`。

- [ ] R1 — obligation `must_fix`：标题性主张与图表对齐（RM-1）
- [ ] R2 — obligation `must_fix`：披露并对照 actor-critic 不对等（RM-2）
- [ ] R3 — obligation `must_fix`：修正 R2I 比较（RM-3）
- [ ] R4 — obligation `must_fix`：单种子下的变异性报告（RM-4）
- [ ] R5 — obligation `must_fix`：完成 Memory Maze 结论（RM-5）
- [ ] R6 — obligation `must_fix`：更正 Spiking-WM/DMC 对照（RM-6）
- [ ] S1 — obligation `should_fix`：机制差异证据（RM-7）
- [ ] S2 — obligation `should_fix`：脉冲相关表述精确化（RM-8）
- [ ] S3 — obligation `should_fix`：能耗范围（RM-9）
- [ ] S4 — obligation `should_fix`：BSuite 预算与种子（RM-10）
- [ ] S5 — obligation `should_fix`：留出任务（RM-11）
- [ ] S6 — obligation `consider`：干预表与机制分析（RM-12 至 RM-14）
- [ ] S7 — obligation `should_fix`：复现、术语、版式（RM-15 至 RM-18）

机器可读的 `revision-roadmap/1.0` 核心未生成：该格式要求绑定稿件的块级清单（block manifest），本次未提供；`revision_roadmap.md` 为人可读版本。

---

## 期限

- **会议给定期限**：NOT PROVIDED（不推断工作量或时长；投稿截止日期由作者掌握）。

---

## 回复信要求

请逐条回应每一项 must_fix 与 should_fix（采纳与否及理由），对两项 DA CRITICAL（C1、C2）给出明确回应，标出修改位置，并说明哪些数字来自新运行。

---

## 机械合成审计行（reviewer_full v2）

dimension_verdicts: [D1=block, D2=block, D3=block, D4=warn, D5=warn, D6=block]
fired_conditions: [F2, F3, F5]
da_critical_adjudications: [C1=VALIDATED, C2=VALIDATED]
editorial_decision=major_revision

（Step 1：D1 仅 methodology 评分 block；D2 仅 domain 评分 block；D3 由 methodology warn 与 da block 取最坏为 block；D4 perspective warn；D5 eic warn；D6 eic block；无 fatal。Step 2：F1 未触发；F2 触发（D1/D2/D3/D6 有 block）；F3 触发（majority：D1、D2、D6 单席位即为 owner，D3 两席位均为 warn 或更差，共 4 个 mandatory 维度）；F4 未触发（D4 为 warn）；F5 触发；F0 未触发。Step 3：最高严重度为 F2（90）。）

---

## 附录：完整审稿报告

见同目录 `R1_area_chair.md`、`R2_methods.md`、`R3_snn.md`、`R4_world_model_rl.md`、`R5_devils_advocate.md`；Phase 1 预承诺见 `contract/phase1_*.md`。

---

## Attachment: Acronym Check (advisory, #849)

输入为串接后的 LaTeX 源文件（sec/*.tex、tables/*.tex、supp/supp.tex），而非 PDF 文本；以下为脚本原样输出（脚本只支持 en 与 zh-TW，故为繁体）。该附件不产生任何审稿发现或修改条目。

### 縮寫檢查（僅供參考，不必回覆）
範圍：正文（部分）
此檔沒有：英文摘要、中文摘要。
未檢查：
- 正文第 24 行：ANN（括號前各字的字首拼不出這個縮寫）
- 正文第 53 行：PDN（括號前各字的字首拼不出這個縮寫）
- 正文第 110 行：MCN（括號前各字的字首拼不出這個縮寫）
- 正文第 303 行：MSE（括號前各字的字首拼不出這個縮寫）
- 正文第 388 行：AUC（括號前各字的字首拼不出這個縮寫）
- 正文第 589 行：GRSN（括號前各字的字首拼不出這個縮寫）
- 正文第 589 行：LIF（括號前各字的字首拼不出這個縮寫）

| 範圍 | 行 | 問題 | 縮寫 | 次數 |
|---|---|---|---|---|
| 正文 | 11 | 未定義 | R2I | 24 |
| 正文 | 12 | 未定義 | POPGym | 19 |
| 正文 | 34 | 重複定義 | SNN | 4 |
| 正文 | 36 | 未定義 | WM | 1 |
| 正文 | 41 | 未定義 | RL | 8 |
| 正文 | 49 | 未定義 | GRU | 10 |
| 正文 | 69 | 未定義 | DMC | 12 |
| 正文 | 76 | 未定義 | S4WM | 1 |
| 正文 | 85 | 未定義 | POMDP | 1 |
| 正文 | 224 | 未定義 | LRU | 2 |
| 正文 | 264 | 未定義 | OFF | 1 |
| 正文 | 264 | 未定義 | ON | 1 |
| 正文 | 292 | 未定義 | SiLU | 2 |
| 正文 | 312 | 未定義 | TODO | 9 |
| 正文 | 373 | 未定義 | RSSM | 4 |
| 正文 | 444 | 未定義 | PHASE0 | 1 |
| 正文 | 479 | 未定義 | TBTT | 1 |
| 正文 | 548 | 未定義 | CNN | 1 |
| 正文 | 548 | 未定義 | MLP | 2 |
| 正文 | 551 | 未定義 | KL | 1 |
| 正文 | 679 | 未定義 | ELU | 1 |
| 正文 | 689 | 未定義 | AC | 2 |
| 正文 | 689 | 未定義 | MAC | 3 |
