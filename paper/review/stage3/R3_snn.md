# R3 — 领域审稿报告：脉冲神经网络 / 类脑计算

稿件：*Fire Without Forgetting: Persistent Dendrites Give Spiking World Models Long-Term Memory*（CVPR 2027 投稿稿，2026-10-04 版本）
审稿轮次：第 1 轮（ARS Stage 3，full mode）
面板席位映射：本报告对应 ARS 面板的 Peer Reviewer 3（Perspective）席位（dispatch role = perspective），负责维度 D4。本文横跨 SNN 与世界模型两个社区，SNN 一侧的主张（二值通信、能耗、神经形态可部署性、树突/共振类比）由本席位评估。

## 审稿人信息

- **身份**：脉冲神经网络与神经形态计算方向研究者，熟悉多房室神经元、自适应阈值与共振神经元（LSNN、TC-LIF、DH-SNN、BRF/D-RF）、替代梯度训练、spiking SSM 系列工作，以及 SNN 能耗估计惯例（Horowitz 45nm 常数、AC/MAC 计数）与 Loihi 2 类硬件约束。
- **审稿重点**：神经元模型与训练的正确性；"只有二值脉冲离开神经元"等主张是否成立；能耗计量的范围与惯例；在神经形态硬件上的可部署性；与 SNN 记忆相关工作的关系。
- **评分量表**：CVPR 近年审稿表风格的 6 级总体评分（6 Accept / 5 Weak Accept / 4 Borderline Accept / 3 Borderline Reject / 2 Weak Reject / 1 Reject）；置信度 1–5。
- **总体评分**：**4 — Borderline Accept**
- **ARS 建议**：Major Revision
- **置信度**：4

## 总体评价

从 SNN 角度，本文最重要的贡献是把"脉冲网络的时间常数为何在 RL 世界模型中不起作用"讲清楚并做了干预检验：步重置让核心退化为二值循环网络，发放重置截断树突痕迹（Eq. 2），而 Euler 离散化的写入增益 1/τ 使长时间常数只以约 1% 阈值的幅度存储线索。阈值扫描（Supp. D）进一步排除了"发放频率限制记忆"的解释。这一分析对整个"用房室/树突动力学提供记忆"的 SNN 研究方向都有警示意义，我认为它本身就值得发表。

PDN 的设计（无状态胞体、不受脉冲回馈的线性树突、按环境步计时）在记忆路径上避开了替代梯度，思路干净。我的主要保留意见集中在 SNN 一侧主张的精确性：(i) 树突的分级状态通过层内标准化后的 ON/OFF 阈值编码（2T+1=11 级）提供给 actor/critic 与 Memory Maze 探针，与摘要"the only signal that leaves the neuron, a binary spike"不符；(ii) PDN 内部的 LayerNorm 需要跨神经元的分级统计量，与"graded values never cross a synapse"不符；(iii) 能耗只计世界模型滤波器、只在 BSuite L=32 模型上测，不含 FwF 放大了约 3.8 倍的 actor；(iv) 与 spiking state-space 层的机制差异只从应用场景角度论证。这些问题主要靠精确改写与基于现有模型的重新计数解决，(iv) 需要少量追加的探针运行。总体为 Borderline Accept，建议 Major Revision。

contract_role: perspective

## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: warn
trigger: "跨领域术语使用松散，或能耗估计依赖简化惯例却未说明其局限"

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

以下从 SNN / 神经形态计算角度给出优点与缺点。每个缺点注明严重程度、证据锚点与修改建议。

### S1: 写入增益分析定量且可检验
文中指出 Euler 更新以 1/τ 写入，τ=128 的树突对线索的响应仅为 0.004–0.009（阈值 0.5），并给出白噪声输入下稳态方差 (1−λ)/(1+λ) ≈ 1/(2τ) 的推导。这把"长时间常数为何无用"从经验观察变成了可检验的机制，对使用 Euler 离散 LIF/房室模型的 SNN 记忆研究有普遍意义。
**Evidence Anchor**: text: §3.1 "A long time constant thus retains a cue but stores it faintly."

### S2: 阈值扫描排除了"发放频率"解释
若发放重置通过发放频率限制记忆，降低阈值应缩短记忆；Supp. D 的结果相反，且加入 G 后几乎不再依赖阈值。这是区分两种机制的漂亮对照。
**Evidence Anchor**: text: Supp. D "The limit is the write gain, not the firing frequency."

### S3: 记忆路径不经过替代梯度
§4.2 给出积分通道 ∂d_t/∂w_{t−k} = √(1−λ²) λ^k，与期间是否发放无关；梯度在时间上不穿过替代导数，只有每步重新计算的输出脉冲穿过一次。这是对"替代梯度沿时间累积失真"问题的原则性回应。
**Evidence Anchor**: equation: §4.2 "No reset in the memory path" — ∂d_t/∂w_{t−k} = √(1−λ²) λ^k

### S4: 能耗计量的惯例交代清楚、对所有模型一致
Supp. H 说明 AC/MAC 的划分、归一化按每元素 2 MAC 计入、各类神经元更新的计数，并对所有模型统一不计访存，同时引用了硬件感知研究指出访存可能占主导，还报告了各模型的发放率。3.2× 的比值也没有夸大成数量级。
**Evidence Anchor**: text: §5.6 "These estimates count arithmetic only, the convention of SNN energy reports"

### S5: 稀疏性与记忆的权衡如实报告
发放率上限 0.06 保留 32 步、失去 64 步，作者没有隐藏这一代价。
**Evidence Anchor**: table: Table 5 — "firing rate capped at 0.06" 行，L=64 为 "-- (0.52)"

### W1: 树突分级状态被编码后输出给 actor 与探针，与"只有二值脉冲离开神经元"不符
**Problem**: 摘要与 §4.1 强调胞体输出的二值脉冲是神经元唯一的对外信号；但 §4.3 中 actor/critic（以及 Memory Maze 探针）在每个任务上都读取第二层树突状态，方式是先在层内标准化，再用 T 个 ON 与 T 个 OFF 阈值编码为 2T 路二值帧，表示 2T+1=11 个等级。这本质上是对分级状态的约 3.5 bit 模数转换读出。Table 7 显示 POPGym 与 DMC 的控制性能依赖这一读出。
**Evidence Anchor**: text: Abstract "A stateless soma emits the only signal that leaves the neuron, a binary spike."
**Why it matters**: 对 SNN 读者而言，"树突存储、脉冲通信"是本文的核心卖点；若控制性能依赖把分级状态以多级编码送出神经元，这一卖点需要精确限定，否则会被视为概念上的不一致。
**Suggestion**: 在摘要与 §4.1 直接写明两条输出路径："within the world model, neurons communicate only by binary spikes; the actor and critic additionally read an 11-level spike encoding of the dendritic state"；在 §4.3 给出该编码的维度（6144 路 × T 帧）与其能耗；在 §5 引用 Table 7 说明 BSuite 不需要该读出、POPGym 与 DMC 需要。
**Severity**: Major
**Confidence**: 5 — core expertise: 脉冲编码与神经元模型

### W2: PDN 内部的层归一化需要跨神经元的分级统计量
**Problem**: 树突输入为 w_t = LN(B x_t)，胞体为 s_t = H(N(u_t) − ϑ)，ON/OFF 编码前也做层内标准化。这三处都需要在整层神经元上计算分级量的均值与方差，即分级信息在神经元之间汇聚。§4.2 却称分级值从不跨越突触、始终留在神经元内部，并以 Loihi 2 的多变量神经元作为可部署性依据。
**Evidence Anchor**: text: §4.2 "Graded values never cross a synapse; they remain inside the neuron, as state variables of neuromorphic cores that support multi-variable neuron models"
**Why it matters**: LayerNorm 在所有比较的脉冲世界模型中都存在（继承自 Spiking-WM），不影响相对比较；但"局部性"是 PDN 特有的主张，直接关系到神经形态硬件上的可部署性论证。N(u_t) 使阈值相对于每步的层统计量，这在功能上接近 k-WTA 式的全局竞争，其硬件实现成本与普通 LIF 阈值不同。
**Suggestion**: 用一句话准确列出非局部操作（"all compared spiking world models, including FwF, use layer normalization; in FwF it appears in the dendritic input, before the somatic threshold and in the readout encoding"）；如可行，评估推理时用运行统计量折叠为逐通道仿射（类似 BN folding）后的性能（基于现有检查点即可评估）；在 Limitations 中把"deployment needs multi-variable neurons"扩展为同时需要层归一化支持。
**Severity**: Major
**Confidence**: 5 — core expertise: 神经形态硬件约束

### W3: 能耗计量范围不足以支撑摘要中的"每步能耗"比较
**Problem**: 能耗只计算世界模型滤波器（编码器与后验更新），不含 actor；而 FwF 的 actor-critic 为 12.9M 参数（基线 3.4M），并多读 6144 路 × T 帧的树突脉冲。测量只在 BSuite L=32 的模型上进行（向量观测、输入层极小），没有 Memory Maze（CNN 编码器）或 DMC。另外，PDN 的输出脉冲在 T=5 个仿真步上重复，下游 LIF 层接收 5 帧相同输入，其 AC 是否按 5 倍计入未说明。
**Evidence Anchor**: text: Supp. H "We count the operations of the world-model filter per environment step, i.e., the encoder and the posterior update"
**Why it matters**: 部署时 actor 每步都要运行。摘要写"uses 3.2× less energy per step than DreamerV3"，读者会理解为整个智能体的每步能耗。若 actor 计入，FwF 相对 DreamerV3 的优势可能缩小；在 CNN 编码器主导的视觉任务上比值也会不同。
**Suggestion**: 基于现有检查点做操作计数（无需训练）：按基准分别报告"滤波器"与"滤波器 + actor"的每步能耗；注明 Horowitz 常数对应 FP32 加法 0.9 pJ、乘加 4.6 pJ 的假设；说明重复脉冲帧的计数方式；摘要改为"3.2× less energy per step for the world-model filter on BSuite"或换成全智能体数字。
**Severity**: Major
**Confidence**: 5 — core expertise: SNN 能耗估计惯例

### W4: 与 spiking state-space 层及 LRU 型神经元的机制差异缺少证据
**Problem**: PDN 的树突是对角线性递推，积分通道对应实特征值、共振通道对应复特征值，写入归一化与初始化取自 LRU；相关工作中作者把 spiking state-space 层概括为"attach spiking outputs to linear recurrences"，并以"那些工作做离线序列分类、本文做在线滤波"来区分。机制层面的差异（无状态胞体、树突不接收脉冲回馈、按环境步计时、积分与共振混合）没有逐项与这类层做受控比较。
**Evidence Anchor**: text: §2 "These neurons are trained and tested on offline sequence classification, where the network sees the whole sequence and is reset between samples."
**Why it matters**: SNN 审稿人会首先问"这与 SpikingSSM / P-SpikeSSM / SiLIF 有何本质不同"。以应用场景作答不足以说服。作者的定位不是 spiking SSM，这一点我尊重；正因如此，差异应以直接陈述加证据的形式给出。
**Suggestion**: 用现有证据直接陈述已证实的差异：Table 5 中"all channels resonant"（最接近全复数 LRU 的变体）在 64 步需要近两倍更新且 AUC 不稳定，说明积分通道对保持一个值是必要的；Spiking-WM-NR 的失败说明脉冲回馈进入树突、以仿真步计时的递推不能提供 64 步记忆。追加两项低成本探针（memlen-64、RepeatPrevious k=32）：(a) 胞体带膜电位与重置的 PDN 变体，检验"无状态胞体"；(b) 以一个已发表的 spiking linear-recurrence 层（如 SiLIF 或 P-SpikeSSM 的层）作为核心。
**Severity**: Major
**Confidence**: 4 — core expertise: SNN 长程记忆神经元文献

### W5: "firing sparsely"需要相对基线量化
**Problem**: FwF 的循环状态发放率为 0.15（探针 0.18），是所有脉冲模型中最高的（基线 0.09–0.14）；发放率限制到 0.06 会失去 64 步记忆。结论却以"while firing sparsely"作结。
**Evidence Anchor**: text: §6 "recalls exact lags that DreamerV3 cannot, while firing sparsely"
**Why it matters**: SNN 读者会把"sparse"与能效直接挂钩；当 FwF 的发放率高于所有脉冲基线时，这一措辞会被视为不准确。
**Suggestion**: 直接写出数值："at a recurrent firing rate of 0.15, the highest among the spiking models but 3.2× cheaper than DreamerV3 in arithmetic"；并解释 FwF 的能耗低于 Spiking-WM 的来源（PDN 每环境步更新一次，而 MCN 每仿真步更新 4 次）。
**Severity**: Minor
**Confidence**: 5 — core expertise: SNN 稀疏性与能耗

### W6: 未报告学习后的保留时间与相位分布
**Problem**: τ 与 θ 都是可学习参数，但没有报告训练后的分布，也没有分析哪些通道承载 32 步精确延迟。
**Evidence Anchor**: absence: §4.1 Parameters 与 §5.5 Ablations — expected 训练后 τ 与 θ 的分布及承载延迟信息的通道分析; checked §4.1, §4.2, §5.3, §5.5, Table 5, Supp. A–H
**Why it matters**: "A spectrum of retention times"与"Resonance tells time"是机制性主张；学习后长 τ 是否保留、θ 是否集中在与 k=32 相关的频率，是最直接的机制证据。
**Suggestion**: 用现有检查点画出 τ 与 θ 的初始化/训练后直方图，并对 RepeatPrevious k=32 的模型做按通道组的线性解码（积分 vs 共振、按 τ 分箱）。
**Severity**: Minor
**Confidence**: 4 — core expertise: 神经元动力学分析

### W7: 生物学类比应精确化
**Problem**: PDN 的"树突"是抽象的线性状态变量，与胞体之间没有电导耦合；共振通道与 Hutcheon & Yost 的阈下共振只有功能上的相似。
**Evidence Anchor**: text: §2 "PDNs borrow both properties"
**Why it matters**: 神经科学背景的读者可能把"dendritic channels"理解为生物物理房室；精确的术语能避免不必要的争议。
**Suggestion**: 在 §4.1 首次出现处直接定义："dendritic channels are linear state variables of a neuron; the analogy to dendrites is functional (local, persistent, subthreshold), not biophysical"。
**Severity**: Minor
**Confidence**: 4 — adjacent field: 计算神经科学

### W8: 精确延迟回忆与 SNN 原生的突触/轴突延迟机制未讨论
**Problem**: 回忆"恰好 k 步前"的输入在 SNN 中有原生机制——可学习的突触或轴突延迟（如 learnable delays / dilated convolutions with learnable spacings 一类工作）。文中只讨论了共振相位。
**Evidence Anchor**: absence: §2 Related Work 与 §4.2 "Resonance tells time" — expected 对可学习突触/轴突延迟的讨论; checked §2 全部四段, §4.2, refs.bib 条目
**Why it matters**: 这是 SNN 社区对"when"问题最常见的解法，审稿人会问为何选择共振而非延迟。
**Suggestion**: 在相关工作中加一两句直接比较：延迟需要按任务设定最大延迟并为每个延迟付出存储，共振通道以固定状态维度编码任意延迟（随 λ^k 衰减）。
**Severity**: Minor
**Confidence**: 4 — core expertise: SNN 时间编码

### W9: 替代梯度与胞体前归一化的细节缺失
**Problem**: PDN 胞体使用哪种替代导数、宽度多少、N(u_t) 对替代梯度的影响均未说明；补充材料只给出世界模型其余部分使用 QGateGrad（宽度 2）。
**Evidence Anchor**: text: §4.1 "The Heaviside step is trained with a surrogate derivative, and all states are reset only at the start of an episode."
**Why it matters**: 替代导数的形状与宽度影响可训练性，复现需要这些信息。
**Suggestion**: 在 Supp. A 补一句说明，若与其余层相同则直接写明。
**Severity**: Minor
**Confidence**: 5 — core expertise: 替代梯度训练

### 致作者的问题

1. 若 actor 只读 (z_t, h_t)，FwF 在 POPGym 与 DMC 上大幅下降（Table 7）。你们如何在"只有脉冲离开神经元"的叙述下解释控制性能对 11 级树突读出的依赖？
2. 推理时能否把 PDN 内的 LayerNorm 折叠为固定仿射而不损失记忆？若不能，部署到 Loihi 2 类硬件需要什么额外支持？
3. 若把 actor 计入，FwF 与 DreamerV3 的每步能耗比是多少？在 Memory Maze（CNN 编码器）上又是多少？
4. 训练后保留时间分布是否仍覆盖 1–1000 步？共振通道的 θ 是否向某些频率集中？

### 总体评分与理由

总体评分 4（Borderline Accept），置信度 4。我给出偏正面的评分，是因为第 3 节的诊断（S1–S3）对 SNN 社区有独立价值，且能耗报告的惯例是诚实的（S4）。D4 的未解决部分（W1–W3）主要是主张的精确性与计量范围，可以通过改写与基于现有检查点的重新计数完成；W4 需要少量追加探针。若这些完成，我会升至 5。

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | reviewer_full D4 | PARTLY_MEETS | S1; W4 | 诊断新颖；神经元与 spiking linear-recurrence 层的机制差异待证据 | 不评估 RL 文献（交 R4） | yes |
| Methodological Rigor | reviewer_full D1 | NOT_ASSESSED | — | 由 R2 负责 | — | no |
| Evidence Sufficiency | reviewer_full D4 | PARTLY_MEETS | S4; W3 | 能耗惯例清楚，但范围只覆盖滤波器与一个基准 | — | yes |
| Argument Coherence | reviewer_full D4 | PARTLY_MEETS | W1; W2 | "只有脉冲离开神经元""分级值不跨突触"与实现不符 | — | yes |
| Writing Quality | reviewer_full D5 | NOT_ASSESSED | — | 由 R1 负责 | — | no |
| Literature Integration | reviewer_full D4 | PARTLY_MEETS | W4; W8 | SNN 记忆文献覆盖较全，缺延迟机制 | — | no |
| Significance & Impact | reviewer_full D4 | MEETS | S1; S2 | 对 SNN 记忆研究方向有明确警示与设计规则 | — | yes |
