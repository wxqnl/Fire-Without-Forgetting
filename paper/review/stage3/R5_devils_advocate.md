# R5 — Devil's Advocate 审稿报告

稿件：*Fire Without Forgetting: Persistent Dendrites Give Spiking World Models Long-Term Memory*（CVPR 2027 投稿稿，2026-10-04 版本）
审稿轮次：第 1 轮（ARS Stage 3，full mode）
面板席位映射：本报告对应 ARS 面板固定的 Devil's Advocate 席位（dispatch role = da），负责维度 D3（论证自洽），发现不受维度限制。

## 审稿人信息

- **身份**：固定的对抗性审稿席位。任务是构造拒稿的最强理由：攻击中心主张、寻找未被排除的替代解释、选择性报告、过度泛化与混杂因素。
- **评分量表**：CVPR 近年审稿表风格的 6 级总体评分（6 Accept / 5 Weak Accept / 4 Borderline Accept / 3 Borderline Reject / 2 Weak Reject / 1 Reject）；置信度 1–5。按 ARS 规范，DA 不给出 Accept/Minor/Major/Reject 建议，只给出发现；应本次任务要求另给出 CVPR 评分。
- **总体评分**：**2 — Weak Reject**（代表"拒稿的最强理由"成立的程度，不是平衡意见）
- **置信度**：4

## Calibration Status

`NOT_CALIBRATED`

## Criterion-Bound Judgements

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| 中心主张与证据一致 | reviewer_full D3 | PARTLY_MEETS | C1; M1; M2 | BSuite 与探针支持"比基线记得更久"；"solves 101""only model""differ only in the core"超出证据 | 单种子 | yes：标题性主张需改写 |
| 替代解释已被排除 | reviewer_full D3 | DOES_NOT_MEET | C1; M5 | POPGym/DMC 的 RL 收益可由 actor 读出与容量解释；机制收益可由"任何持久线性递推"解释 | 宽 actor 对照仍在运行 | yes |
| 前人工作表述准确 | reviewer_full D2（发现层面） | DOES_NOT_MEET | C2; M8 | R2I 与 Spiking-WM 的复现与发表结果矛盾 | — | yes |
| 无选择性报告 | reviewer_full D3 | PARTLY_MEETS | M3; M4 | 开发与评测任务重叠；Memory Maze 临时结果不利 | Memory Maze 仍在运行 | yes |
| "那又怎样"测试 | reviewer_full D6（发现层面） | PARTLY_MEETS | M6; M7 | 若控制依赖分级读出、能耗只计滤波器，"脉冲"带来的部署收益未被证明 | — | no：不单独决定 |

contract_role: da

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: block
trigger: "一个论文未排除的替代解释能与所提机制同样好地解释标题性结果"
block_class: repairable

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

### 先肯定真实的长处

论文确实做对了几件事：第 3 节的干预在 Spiking-WM 上给出了干净的阳性/阴性对比；世界模型探针在等更新预算下独立显示 FwF 能回忆 64 步，且这一结论在 actor 只读 (z_t, h_t) 时依然成立；作者对奖励头的改动、能耗不计访存、与 R2I 的比较只是参考等都做了披露。下面的攻击不否认这些，而是针对标题、摘要与比较设置。

### 最强反驳论点

本文声称持久树突让脉冲世界模型获得长时记忆，并在"只替换循环核心"的受控比较中优于所有脉冲基线与 DreamerV3。最强的反驳是：受控比较并不受控，而标题性数字比论文自身的图表更乐观。

第一，RL 比较中 FwF 的 actor-critic 是基线的约 3.8 倍（12.9M vs 3.4M），并多读 6144 路由分级树突状态编码而来的输入；去掉这一输入后，FwF 在 RepeatPrevious Easy 只有 0.11，低于 LIF、Spiking-WM 与 DreamerV3，Medium 回到 −0.03，DMC walker 跌到 504。POPGym 与 DMC 的 RL 收益因此至少同样可以解释为"actor 拿到了一个近似模拟量的状态和更多参数"。第二，唯一被列为 ANN SOTA 的 R2I 在 RepeatPrevious Easy 上停在随机水平，这与 R2I 原文相矛盾，说明比较中最强的对手实际上缺席。第三，"solves 101-step recall"建立在单种子、评估回报首次触及 0.9 的判据上，而图 1 显示 L=101 的最终回报约 0.40，与 R2I 报告的约 0.35 同一水平。第四，机制上，PDN 的记忆路径是 LRU 参数化的对角线性递推；现有消融（单一 τ=128 在 64 步与对数谱同样好、全共振与半共振在 k=32 上同样好）只证明了"持久线性递推有用"，没有隔离任何 PDN 特有元素的贡献。第五，唯一的视觉 3D 基准 Memory Maze 的临时结果中，FwF 的墙体准确率仅比常数预测器高 0.8 个百分点，且低于所有脉冲基线。

把这些放在一起，可以得到一个更简洁的叙述：任何持久的线性递推加上足够宽、能读到其状态的 actor，都能在这些向量观测记忆任务上取得类似结果；"脉冲"在其中既不是记忆的来源，也尚未被证明带来部署收益。

### 被忽视的替代解释

1. **读出与容量解释**：POPGym/DMC 的 RL 收益来自 actor 读取 11 级树突编码与 3.8 倍参数，而非世界模型记忆（Table 7 已部分支持这一解释；宽 actor 对照只覆盖 Easy 且未完成）。
2. **线性递推解释**：64 步记忆来自任何持久、归一化写入的线性递推（LRU/S5/FFM 类）。一个 ANN LRU 核心或已发表的 spiking linear-recurrence 层在同一探针中可能同样成功；论文没有做这一比较。
3. **预算解释**："唯一解决 64 步"来自 500k 预算，DreamerV3 在 51 步需要 902k 步；等更新预算的探针部分排除了这一解释，但 RL 中的表述没有限定预算。
4. **弱基线解释**：R2I（POPGym）与 Spiking-WM（DMC）都明显弱于其发表结果，"优于基线"部分来自基线复现不足。
5. **开发选择解释**：共振通道是在旧版 FwF 于 RepeatPrevious Medium 失败后加入的，随后又以同一任务作为主要证据；没有留出任务检验这一设计的泛化。

### 缺失的利益相关方视角

- **神经形态硬件工程师**：PDN 内的 LayerNorm 与 actor 前的层标准化需要跨神经元的分级统计量；论文以 Loihi 2 的多变量神经元论证可部署性，但没有讨论这些全局操作。
- **具身视觉研究者（CVPR 读者）**：主要证据来自向量观测；像素输入下的记忆只有尚未完成的 Memory Maze。
- **基准维护者与被比较方法的作者**：R2I 与 Spiking-WM 的复现与其发表结果不一致，且协议命名（"protocol of R2I"）与实际设置不同。

### 未被检验的前提

论文的动机是 SNN 能在低功耗硬件上运行世界模型。但能耗只按算术计数，作者引用的硬件感知研究指出访存可能占主导；而 FwF 的 actor 参数是基线的 3.8 倍、输入多出 6144 路 × T 帧。在计入访存与 actor 之后，"脉冲"是否仍带来能耗优势，是全文没有检验的前提。

### 非缺陷观察

- 第 3 节诊断的价值不依赖于 PDN 是否优于 SSM 类核心；即使 M5 成立，诊断本身仍是对 SNN 记忆研究的有用贡献。
- 阈值扫描（Supp. D）与发放率上限对照是少见的、主动排除替代解释的实验，值得保留在正文。

### 次要问题（不入表）

- 名称"reset-induced amnesia"把失败主要归因于重置，但 Table 1 显示在 32 步上决定性的是写入增益 G，且 A+B+G 在 64 步仍失败；"resets are necessary but not sufficient"应在摘要层面写明。置信度 4。
- 文中的"DreamerV3"是在 Spiking-WM 代码库中的同构重实现，应命名区分。置信度 4。
- §3.1 对 Spiking-WM 树突发放重置的描述来自其官方代码而非论文公式，应注明。置信度 4。

#### CRITICAL
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|
| C1 | Logic Chain Break（比较设置） | 正文声称各模型只在循环核心上不同，但 FwF 的 actor-critic 为 12.9M（基线 3.4M）且多读 6144 路树突编码输入；Table 7 显示去掉该输入后 POPGym Easy 0.71→0.11、Medium 0.45→−0.03、DMC walker 880→504。未更正时，论文错误描述了其核心比较协议，POPGym/DMC 的 RL 结论无法归因于世界模型记忆。BSuite 结论不受影响。 | text: §5.1 "differ only in the recurrent core, whose state the actor and critic read" | 5 — 直接由论文正文与 Supp. A/Table 7 对照得出 | | |
| C2 | Data-Conclusion Mismatch（前人工作） | 主表把 R2I 作为 ANN 记忆 SOTA 列出，但其 RepeatPrevious Easy −0.50、Medium −0.49 均为随机水平，与 R2I 原文在 RepeatPrevious Medium/Hard 上的 SOTA 结果矛盾，且 Easy 上连 Spiking-WM 都能学会。未更正时，论文对最强前人方法作出与其发表结果相反的陈述。 | table: Table 2 — R2I (ANN) 行 −0.50‡ / −0.49‡ | 5 — 已核对 R2I 原文 §4.2 结论 | | |

#### MAJOR
| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|
| M1 | Overgeneralization | 摘要称"solves 101-step recall"、图 1 称"recalls across all lengths"，但图 1 中 FwF 在 L=71、101 的最终回报约 0.46、0.40，图 3 曲线峰值后回落；"solve"是单种子评估回报首次达到 0.9。与 R2I 报告的 101 步约 0.35 相比是持平而非超越。 | figure: Figure 1 right panel — FwF final return ≈0.46 at L=71 and ≈0.40 at L=101 | 5 — 图表数值直读 | | |
| M2 | Overgeneralization | "the only model ... that solves 64-step recall"依赖 L=64 的 500k 预算（L=51 为 1M，DreamerV3 在 51 步需 902k），且 L≤64 运行在并行环境间共享种子。主张应限定预算，或以等更新预算的探针为证据。 | text: Abstract "FwF is the only model in our comparison, spiking or not, that solves 64-step recall" | 4 — 预算对照来自 Table 2 | | |
| M3 | Cherry-picking risk | 共振通道是在旧版（仅积分通道）于 RepeatPrevious Medium 失败后加入的，随后 RepeatPrevious Medium 成为"唯一能回忆精确 32 步延迟"的主要证据；没有开发中未使用的留出记忆任务。 | text: Supp. F "FwF here is the earlier version with integrator channels only." | 4 — 研究流程推断 | | |
| M4 | Data-Conclusion Mismatch（临时） | 唯一的视觉 3D 基准 Memory Maze：FwF 墙体准确率 81.6%，常数预测器 80.8%，低于全部脉冲基线（82.7–82.9%）与 DreamerV3（85.3%）；贡献要点 3 将其列为评估之一。若最终结果维持，视觉记忆证据与"长时记忆"叙述相反。 | table: Table 4 — FwF walls 81.6 (pending) vs constant predictor 80.8% | 3 — 结果为临时值 | | |
| M5 | Alternative Path（机制） | 第 4 节的设计规则（持久、归一化写入、递推不过阈值）即 LRU/SSM 的配方；PDN 的写入归一化与初始化取自 LRU，FFM 已有对数间隔衰减与相位编码。现有消融中单一 τ=128 与对数谱同样好、全共振与半共振在 k=32 同样好，未隔离任何 PDN 特有元素（无状态胞体、无脉冲回馈）的贡献。 | text: §4.2 "The normalization of the LRU thus keeps channels with retention times from 1 to 1000 steps on the same scale." | 4 — SSM/LRU 文献与 Table 5 对照 | | |
| M6 | Internal Contradiction | 摘要称二值脉冲是离开神经元的唯一信号，但 actor/critic 与 Memory Maze 探针读取层内标准化后以 ON/OFF 阈值编码的树突分级状态（11 级），且 PDN 内部使用 LayerNorm；控制性能依赖这一读出（Table 7）。 | text: §4.3 "each state variable is standardized over the layer and encoded by T ON and T OFF thresholds" | 5 — 正文前后对照 | | |
| M7 | Overgeneralization（能耗） | "3.2× less energy per step"只计世界模型滤波器、只在 BSuite L=32 模型上测量，不含 FwF 放大约 3.8 倍的 actor；在计入 actor 与访存后，能耗优势未被证明。 | text: Table 4 caption "Energy of the world-model filter per environment step" | 4 — SNN 能耗惯例 | | |
| M8 | Data-Conclusion Mismatch（非记忆对照） | 预注册判据"no degradation relative to Spiking-WM"所对照的 Spiking-WM 既非其源协议（原文为像素 DMC、约 18.8M 参数），数字也远低于其发表值；即便如此，FwF walker 880 已低于 Spiking-WM 923，run/swingup 比 DreamerV3 低约 260/250 分；只读 (z_t, h_t) 时为 504/314/528。 | text: §5.6 "Criterion (fixed in advance): no degradation relative to Spiking-WM, whose benchmark this is." | 4 — Spiking-WM 原文 Table I 已核对 | | |
| M9 | Evidence Gap（统计） | 全部 RL 与探针结果为单种子（仅 Table 1 为两种子），且没有任何区间估计；在作者已固定不做多种子的前提下，"only model / solves"类排他性主张缺乏统计支撑。 | text: §5.1 "Number of seeds: one so far" | 4 — RL 评估文献 | 深度 RL 文献普遍要求多种子与区间估计（Henderson et al., AAAI 2018；Agarwal et al., NeurIPS 2021） | 在单种子下，排他性主张（"only model"）需要单次运行内部的变异性证据（评估回合置信区间、解决后稳定性）才能跨越证据门槛 |

## 致作者的问题

1. 在只读 (z_t, h_t) 且 actor-critic 加宽到 12.9M 的条件下，FwF 在 RepeatPrevious Medium 上能达到多少？如果仍接近随机，POPGym RL 的收益应归于世界模型还是树突读出？
2. 一个 ANN LRU 核心（相同的对数均匀保留时间与 LRU 归一化）放进同一世界模型，在 memlen-64 与 RepeatPrevious k=32 探针上表现如何？如果与 FwF 相当，PDN 特有的哪一个元素是必要的？
3. L=71 与 L=101 的学习曲线在峰值后回落，原因是什么（策略退化、世界模型漂移、评估噪声）？世界模型在这些时刻的奖励 AUC 是否仍为 1.0？
4. 若 Memory Maze 最终墙体准确率仍接近常数预测器，你们是否仍会把 Memory Maze 列为贡献要点 3 的证据？

## 总体评分与理由

总体评分 2（Weak Reject），置信度 4。拒稿的最强理由是 C1 与 C2：前者使 POPGym/DMC 的 RL 比较无法归因，后者使与 ANN SOTA 的比较失效；M1、M2 使摘要中的两个排他性主张强于论文自身的图表。所有这些都可修复——它们是"块状但可修复"的（D3 block, repairable）：BSuite 结论在策略只读脉冲时依然成立，世界模型探针独立支持 64 步记忆，第 3 节诊断不依赖任何受攻击的比较。若作者把标题性主张改写到与图表一致、披露并完成读出/容量对照、修正 R2I 比较，C1 与 C2 即可关闭。
