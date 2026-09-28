# Fire Without Forgetting：面向记忆型具身控制的持久态脉冲世界模型

**英文工作标题**：*Fire Without Forgetting: Persistent Dendritic State for Memory-Capable Spiking World Models*

> 日期：2026-09-27 ｜ 状态：研究方案，结果待实验 ｜ 起点论文：Spiking-WM（Sun et al., PNAS 2025）

---

## 摘要

### Abstract

> 方括号 `[ ]` 内为待实验结果的占位符。"we show" 类表述须在 E1、E2 实验确认后才能保留。

World models allow embodied agents to imagine and plan, but embodied perception is inherently partial: an agent must remember what it can no longer see. Spiking neural networks (SNNs) are attractive substrates for world models on power-constrained agents, and their neuronal dynamics are often credited with an intrinsic capacity for temporal memory. We show that current spiking world models leave this capacity unused. Neuronal states are re-initialized at every environment step, and dendritic potentials are reset at every somatic spike. Memory across time is therefore carried only by binary spikes fed back as input, and membrane time constants become functionally irrelevant. We term this failure mode *reset-induced amnesia* and show that it limits spiking world models on memory-demanding tasks. To address it, we propose [Name], a spiking world model whose memory lives in persistent, non-resetting dendritic compartments with linear recurrence. Somatic spikes carry all inter-layer communication and also serve as stochastic discrete latents. Aligning neuronal time with environment time lets the neuronal state itself serve as the belief state. The linear dendritic recurrence makes parallel-scan training exactly equivalent to step-by-step imagination. On BSuite, POPGym and Memory Maze, [Name] [outperforms existing spiking world models by X] and [reaches Y% of the ANN state-of-the-art R2I], while [requiring Z× fewer synaptic operations under hardware-aware energy accounting]. Our results suggest that for spiking world models, whether neuronal state persists matters more than how neurons spike.

### 中文摘要

世界模型让具身智能体能够想象和规划，但具身感知天然是部分可观测的：智能体必须记住它已经看不到的东西。脉冲神经网络（SNN）适合在功耗受限的智能体上实现世界模型，其神经元动力学也常被认为天然具备时间记忆能力。我们发现，现有脉冲世界模型并没有用上这种能力：神经元状态在每个环境步都被重新初始化，树突电位在每次胞体发放时也被复位。因此，跨时间的记忆只能靠作为输入回传的二值脉冲来承载，膜时间常数在功能上变得无关紧要。我们把这一失效模式称为"复位导致的失忆"（reset-induced amnesia），并证明它限制了脉冲世界模型在记忆型任务上的表现。为此，我们提出 [Name]：记忆存放在持久、不复位、线性递推的树突区室中，胞体脉冲负责全部层间通信，同时充当随机离散隐变量。神经元时间与环境时间对齐后，神经元状态本身就是信念状态；线性树突递推使并行扫描训练与逐步想象在数学上完全等价。在 BSuite、POPGym 和 Memory Maze 上，[Name] [比现有脉冲世界模型提升 X]，[达到 ANN SOTA 方法 R2I 的 Y%]，并且在硬件感知的能耗口径下 [突触运算减少 Z 倍]。结果表明，对脉冲世界模型而言，神经元状态能否持久保留，比神经元如何发放更重要。

---

## 1. 研究问题

### 1.1 核心问题

现有脉冲世界模型用"神经元动力学带来长时记忆"作为动机，但在实现上，神经元状态并不跨环境步保留。结果是，SNN 最有特色的时间动力学，恰恰在世界模型最需要记忆的场景中没有被用上。

### 1.2 研究假设

**H**：脉冲世界模型在记忆型任务上的瓶颈来自两级复位：

- **步间复位**：每个环境步开始时，所有膜电位清零；
- **步内发放复位**：胞体发放时，树突电位一并清零。

两者共同导致记忆只能靠二值脉冲回传，时间常数与记忆无关。解除这两级复位，并让记忆状态持久化，就能恢复记忆能力。

### 1.3 研究问题

| 编号 | 研究问题 | 类型 |
|---|---|---|
| RQ1 | 现有脉冲世界模型在部分可观测、记忆型任务上是否失效？失效程度是否随记忆长度增加？ | 诊断 |
| RQ2 | 失效是否由两级复位导致？去掉复位能否因果性地恢复记忆，并使时间常数重新起作用？ | 机制 |
| RQ3 | 以"持久态 + 记忆与发放解耦 + 精确对偶"为原则设计的脉冲世界模型，能否接近 ANN 记忆世界模型的性能，同时保持脉冲的稀疏性？ | 方法 |

### 1.4 为什么值得做

1. **动机与机制之间存在落差**：前作以树突长时记忆为卖点，实现上却没有用到它。指出并修复这一落差，比单纯提出一个新神经元更有说服力。
2. **与具身场景直接相关**：第一人称视角、遮挡和视野受限，使具身任务天然部分可观测；功耗约束又使 SNN 有部署价值。
3. **结论可推广**："每个环境步独立仿真 T 步、步间清零"很可能是脉冲 RL 的常见做法（需在综述中进一步核实）。如果这一点成立，结论适用于整个脉冲 RL 方向，而不只是一篇论文。

---

## 2. 证据链：来自 Spiking-WM 原文与代码

代码版本：`Brain-Cog-Lab/Spiking-WM`，commit `602a7c8`（2025-12-14），于 2026-09-27 核查。

| 编号 | 事实 | 出处 |
|---|---|---|
| F1 | 论文动机是长时记忆："Dendritic computation is key to the brain's ability to integrate information over long timescales … to enhance long-term memory and planning." | PNAS Significance Statement |
| F2 | 只在 DMC 19 个任务和语音分类上评测，没有部分可观测或记忆型 RL 任务；DMC 平均得分为 Dreamer(GRU) 的 90.4% | 论文实验部分 |
| F3 | 每个环境步开始时，所有神经元的膜电位（胞体、basal、apical）清零 | `networks.py`：`MCRNN.forward` 与 `img_step` 首行都调用 `self.reset()`；`node.py`：`BaseMCNode.n_reset` 将所有区室置为 `v_reset` |
| F4 | 跨环境步只传递二值脉冲张量 `deter` ∈ {0,1}^(T×N)，其中 T=5、N=512 | `networks.py`：`return outputs, [torch.stack(outputs, dim=0)]`；`configs.yaml`：`spike_times: 5`，`dyn_deter: 512` |
| F5 | 胞体发放时树突同时复位："For simplicity, both somatic and dendritic membrane potentials are reset simultaneously after a spike." | 论文 §4.1 Limitation；`node.py` 第 341–343 行 |
| F6 | 默认 τ = τ_basal = τ_apical = 2.0；按欧拉更新，每个 SNN 步衰减 50% | `configs.yaml` 第 82 行；`node.py` 第 332–336 行 |
| F7 | "adjustments to different membrane time constants in the MCN model do not substantially impact overall model performance." | 论文 §3.3.1 |

**推论**：由 F3–F6，神经元时间常数只影响单个环境步内 5 个 SNN 步的瞬态，跨步记忆完全依赖二值脉冲回传。这为 F7 提供了直接的机制解释，也说明 F1 所说的"树突长时记忆"在实现层面并没有发生。

**措辞提醒**：写进论文时要注明代码版本、保持客观，建议投稿前和原作者沟通确认。

---

## 3. 机制分析：复位导致的失忆

### 3.1 步间复位：世界模型退化为二值状态 RNN

设 $\Phi_T$ 为单个环境步内的 T 步 SNN 仿真。每个环境步从零膜电位开始：

$$
\mathbf{m}_t^{(0)} = \mathbf{0}, \qquad \mathbf{s}_t = \Phi_T\big(\mathbf{x}_t,\ \mathbf{s}_{t-1}\big) \in \{0,1\}^{T \times N}
$$

世界模型的循环状态只有 $\mathbf{s}_{t-1}$。它本质上是一个二值状态 RNN：

- 每一步都必须把记忆重新编码进脉冲；
- 跨步梯度只能经过代理梯度函数传播。

### 3.2 步内发放复位：记忆窗口被发放间隔截断

对一个带发放复位的树突区室（$\alpha = 1 - \Delta t/\tau$）：

$$
v[k] = \alpha\,\big(1 - s[k-1]\big)\,v[k-1] + I[k]
$$

展开后，第 $k-j$ 步输入对当前电位的贡献为：

$$
\frac{\partial v[k]}{\partial I[k-j]} = \alpha^{j} \prod_{i=1}^{j} \big(1 - s[k-i]\big)
$$

只要窗口内发放过一次，这个贡献就为零。因此有效记忆窗口为：

$$
H_{\text{eff}} = \min\big(H_\tau,\ k - k_{\text{last spike}}\big), \qquad \mathbb{E}\big[k - k_{\text{last spike}}\big] \approx 1/r
$$

其中 r 为每步发放概率。当 $r \gg 1/H_\tau$ 时，$H_{\text{eff}} \approx 1/r$，与 τ 无关。

这带来一个**活动–记忆权衡**：神经元发放越多（向下游传递的信息越多），记住的历史就越短。

### 3.3 可证伪预测

| 预测 | 内容 | 检验 |
|---|---|---|
| Pr1 | 在记忆型任务上扫描 τ，现有模型的性能基本不变（在部分可观测任务上复现 F7） | E2 |
| Pr2 | 只去掉步间复位（膜电位跨步保留），长延迟记忆即显著提升 | E2 干预 A |
| Pr3 | 再去掉树突的发放复位，记忆进一步提升，且 τ 开始显著影响性能 | E2 干预 B |
| Pr4 | 保留复位时，降低阈值 θ（提高发放率）会缩短记忆窗口；解耦后这一权衡消失 | E2 阈值扫描 |
| Pr5 | 在 DMC（非记忆任务）上去掉复位不损害性能 | E2 对照 |
| Pr6 | 持久态模型在记忆任务上学到的时间常数会向长时间尺度扩展 | E3 机制分析 |

---

## 4. 方法

### 4.1 设计原则

1. **状态持久**：神经元状态跨环境步保留，SNN 时间与环境时间对齐（T=1），神经元状态本身就是世界模型的信念状态。
2. **记忆与发放解耦**：记忆存放在不复位的树突区室中；胞体的发放和复位不回写树突。
3. **精确对偶**：树突递推是线性的（对角，可取复数），非线性只出现在层与层之间。因此并行扫描训练与逐步想象在数学上等价，不存在近似误差。

### 4.2 持久态双区室脉冲神经元

树突区室（不复位；$\boldsymbol{\Lambda}$ 为对角矩阵，$|\lambda_i| < 1$）：

$$
\mathbf{d}_t = \boldsymbol{\Lambda}\,\mathbf{d}_{t-1} + \mathbf{B}\,\mathbf{x}_t
$$

胞体区室：

$$
u_t = \mathrm{Re}\big(\mathbf{C}\,\mathbf{d}_t\big) + D\,\mathbf{x}_t, \qquad s_t = \mathcal{H}\big(u_t - \theta\big)
$$

- **初始化**：$\boldsymbol{\Lambda}$ 按对数间隔的时间尺度初始化（S5 / HiPPO 风格），覆盖从几步到数千步的记忆。
- **复数取值**：对应亚阈值膜共振，与 BRF、D-RF 一脉相承。
- **胞体**：默认无状态。作为消融，另设一个 LIF 胞体变体，用 SDN 近似实现并行。
- **生物合理性**：Spiking-WM 作者自己在局限部分指出，反向传播动作电位使树突去极化而不是复位。因此"树突不复位"比前作更符合生物事实。

### 4.3 世界模型结构

| 模块 | 设计 | 来源 / 理由 |
|---|---|---|
| 编码器 | 脉冲 CNN | 沿用 Spiking-WM |
| 后验 | $q(z_t \mid o_t)$，不依赖循环状态 | 沿用 R2I：去掉对 $h_t$ 的依赖后，所有时间步的后验可同时计算 |
| 随机隐变量 | 32×32 one-hot，即每组恰好一个脉冲的 WTA 群体编码；直通梯度 | DreamerV3 的离散隐变量本身就是二值稀疏码，直接当作脉冲使用 |
| 序列核 | L 层持久态双区室神经元 | 本方法核心 |
| 预测头 | 奖励、终止、观测重建 | 沿用 Dreamer |
| Actor-Critic | 脉冲 MLP，读取胞体脉冲和隐变量 | 沿用 Spiking-WM |

训练时沿序列长度做并行扫描；想象时逐步递推 H 步，两种模式输出一致。

### 4.4 能耗账（硬件感知口径）

| 运算 | 规模（每步） | 类型 | 说明 |
|---|---|---|---|
| 层间突触 | O(N_in × N_out) | 累加（AC），只在有脉冲时发生 | SNN 能效的来源 |
| 树突递推 | O(N × M) | 乘加（MAC），稠密 | 必须如实计入 |
| 状态读写 | O(N × M) | 访存 | 硬件感知研究指出访存占能耗的重要部分 |

报告"每突触每次推理的脉冲数"，并与 Dampfhoffer et al. 给出的 0.15–1.38 阈值比较。只有低于这个阈值，SNN 才比 ANN 更省能。

---

## 5. 实验设计

### 5.1 实验与研究问题的对应

| 实验 | 回答 | 设置 | 预期结果 |
|---|---|---|---|
| E1 诊断 | RQ1 | BSuite Memory Length 扫描记忆长度；POPGym RepeatPrevious、Autoencode（Easy/Medium/Hard）；对比 Spiking-WM(MCN)、LIF、aLIF、TC-LIF、PSN、DreamerV3、R2I | 脉冲世界模型随记忆长度增加显著退化 |
| E2 因果干预 | RQ2 | 在 Spiking-WM 上做 2×2 析因：{步间复位 有/无} × {树突发放复位 有/无}；每格扫描 τ（如 2 / 8 / 32）；另扫描阈值 θ；DMC 作对照 | 验证 Pr1–Pr5 |
| E3 主结果 | RQ3 | 本方法对比全部 baseline：BSuite、POPGym、Memory Maze 离线探针（9×9，必要时 15×15）、DMC 子集 | 明显优于 SNN baseline，接近 R2I |
| E4 训练–想象一致性 | RQ3 | 精确对偶（本方法）对比近似并行（LIF 胞体 + SDN、masked PSN）；测量并行模式与逐步模式的隐状态轨迹偏差，以及想象回报误差随想象步长 H 的增长 | 近似误差随 H 累积，精确对偶无此问题 |
| E5 效率 | RQ3 | 硬件感知能耗、每突触脉冲数、训练耗时（对比串行的 Spiking-WM 和 R2I）、回报–能耗 Pareto 前沿 | 在 Pareto 前沿上占优 |

**E2 的实现方式**（改动量小，可作为第一周的实验）：

- **干预 A**（去掉步间复位）：把各区室的 `mems` 加入 RSSM 的 state 字典，删去 `MCRNN.forward` 和 `img_step` 中的 `self.reset()`，只在 episode 起点（`is_first`）清零。
- **干预 B**（去掉树突发放复位）：删除 `node.py` 第 342–343 行。

### 5.2 Benchmark

| Benchmark | 设置 | 用途 |
|---|---|---|
| BSuite | Memory Length、Discounting Chain（奖励延迟最长 30 步） | 记忆长度曲线（E1、E2） |
| POPGym | RepeatPrevious（需记住 4 / 32 / 64 个事件）、Autoencode（52 / 104 / 156）、Concentration；各分 Easy / Medium / Hard | 主结果，向量观测，开销小 |
| Memory Maze 离线探针 | 9×9 数据集：3 万条轨迹、共 3000 万步、每条 1000 步、64×64 RGB、约 100GB；预测墙体布局和目标物体位置 | 第一人称 3D 视觉记忆，具身属性最强 |
| DMC | Spiking-WM 的 19 个任务或其子集 | 非记忆任务上不退化；与前作直接对比 |

### 5.3 Baseline

| 类别 | 方法 | 作用 |
|---|---|---|
| ANN 世界模型 | DreamerV3、R2I | 性能上界；R2I 是记忆任务 SOTA |
| SNN 世界模型 | Spiking-WM（MCN） | 直接前作 |
| 脉冲神经元（同一框架内替换） | LIF、aLIF、TC-LIF、PSN | 前作中已有实现，可复用 |
| 近似并行脉冲神经元 | LIF 胞体 + SDN（SpikingSSM 式） | 用于 E4 |

### 5.4 评价指标

- 回报 / 成功率：IQM + 95% bootstrap 置信区间，每组至少 5 个种子。
- 记忆探针：从状态线性解码 k 步前的线索，画出准确率随延迟 k 的衰减曲线。
- Memory Maze：墙体布局和物体位置的探针准确率。
- 效率：发放率、每突触脉冲数、能耗估计、训练耗时、每步想象延迟。

### 5.5 消融

| 消融项 | 要回答的问题 |
|---|---|
| 树突复位：硬复位 / 软复位 / 不复位 | 记忆损失来自哪种复位 |
| Λ 取实数 / 复数 | 亚阈值共振是否有帮助 |
| 时间尺度初始化：均匀 / 对数间隔 | 多时间尺度覆盖的作用 |
| 胞体：无状态 / LIF + SDN | 精确对偶的必要性（与 E4 对应） |
| 隐变量：脉冲 / 连续 | 全脉冲化的代价 |
| 每个环境步 T=1 / T=5 | 时间对齐的收益与代价 |

---

## 6. 预期贡献

1. **发现**：首次在部分可观测、记忆型基准上系统评测脉冲世界模型，揭示其性能随记忆长度增加而退化。
2. **机制**：指出"复位导致的失忆"：步间复位和树突发放复位使神经元动力学不参与跨步记忆。给出活动–记忆权衡的形式化分析，用因果干预加以验证，并解释了前作中时间常数不敏感的现象。
3. **方法**：提出持久态、记忆与发放解耦、训练与想象精确对偶的脉冲世界模型，并将脉冲直接作为随机离散隐变量。
4. **评测**：在 BSuite、POPGym、Memory Maze 上给出结果，以及按硬件感知口径计算的能耗账。

---

## 7. 与已有工作的区别

| 方法 | 设定 | 跨环境步保留神经元状态 | 发放时复位记忆状态 | 并行训练 | 记忆型 RL 评测 |
|---|---|---|---|---|---|
| Spiking-WM（PNAS 2025） | 基于模型的 RL | 否（代码中每步清零） | 是（树突随胞体复位） | 否（STBP 逐步） | 否 |
| SpikingSSM（AAAI 2025） | 序列分类、语言建模 | 不适用（序列内 SSM 状态持续） | SSM 状态不复位；LIF 硬复位只作用于自身 | 是（SDN 近似，脉冲准确率 >99.9%） | 否 |
| PSN（NeurIPS 2023） | 序列分类 | 不适用 | 去掉复位 | 是 | 否 |
| TC-LIF / BRF / SiLIF / D-RF | 序列分类、语音 | 不适用 | 各自改进神经元动力学 | 部分支持 | 否 |
| R2I（ICLR 2024） | 基于模型的 RL（ANN） | 是 | 不适用 | 是（并行扫描） | 是 |
| **本方法** | 基于模型的 RL | 是 | 否（仅胞体） | 是（精确） | 是 |

**定位**：脉冲 SSM 在结构上已经隐含"记忆状态不随脉冲复位"，但它们只在离线分类任务上验证过。本工作的新意不在于发明这种结构，而在于三点：

1. 指出脉冲世界模型缺的正是这一点，并用因果实验证明；
2. 在闭环、自回归想象的设定下检验精确对偶的必要性，这是分类任务中不会暴露的问题；
3. 给出首个部分可观测记忆基准上的系统评测。

---

## 8. 预判审稿意见与回应

| 可能的质疑 | 回应 |
|---|---|
| 这不就是把 SpikingSSM / SiLIF 的结构搬进 R2I 吗？ | 承认结构上的继承关系，并把脉冲 SSM 作为支持设计原则的证据。贡献在于诊断与因果验证（E1、E2）、闭环想象中的精确对偶（E4），以及首个记忆型 RL 评测 |
| 无状态胞体只是二值化的 SSM，不算脉冲神经元 | 层间通信全部是二值事件；记忆位于亚阈值树突电位，符合前作作者指出的生物事实（反向传播动作电位使树突去极化而非复位）；另附 LIF 胞体变体 |
| 能耗优势是否可信 | 采用硬件感知口径，计入树突状态更新和访存；报告每突触脉冲数并与 0.15–1.38 阈值比较；不以理想化的 AC/MAC 比例作为唯一证据 |
| 为什么不直接用 R2I | 目标是神经形态部署；贡献体现在回报–能耗 Pareto 前沿，而不是在绝对回报上超过 R2I |
| 理论分析太简单 | 价值在于解释了已发表的反常现象（τ 不敏感），并给出可证伪预测 Pr1–Pr6，逐条用实验检验 |
| 具身性体现在哪里 | Memory Maze 是第一人称 3D 导航，部分可观测来自视野受限和遮挡；遮挡条件下的操作任务可作为扩展 |
| 对前作代码的批评是否公允 | 注明代码版本，客观陈述，投稿前与作者沟通；若"步间清零"是脉冲 RL 的普遍做法，就表述为领域惯例而非某一篇论文的问题 |

---

## 9. 时间线与判定门

总周期约 13–16 周。

| 阶段 | 时长 | 内容 | 判定门 |
|---|---|---|---|
| Phase 0 | 3 周 | E1 诊断 + E2 因果干预 | 干预 A/B 是否带来记忆提升（Pr2、Pr3）。否 → 假设被证伪，启用兜底方案 |
| Phase 1 | 4–6 周 | 实现方法；跑 BSuite、POPGym | POPGym Medium 上是否明显超过 SNN baseline |
| Phase 2 | 3–4 周 | Memory Maze 离线探针；E4 训练–想象一致性 | — |
| Phase 3 | 3 周 | E5 能耗；DMC 不退化验证；写作 | — |

**兜底方案**：

- 若 E2 证伪复位假设：问题改为"二值循环状态的容量瓶颈"，方法主线不变；或转向附录 A 的方向 B。
- 若性能明显不及 R2I：主张改为 Pareto 前沿，即"用 X% 的能耗达到 R2I 的 Y%"。
- 撞车风险：持续关注 Brain-Cog-Lab 及脉冲 SSM 方向的新工作。

---

## 附录 A：备选方向 B（简）

**用神经元快变量实现世界模型的上下文自适应。**

- **问题**：世界模型在重力、质量等动力学参数变化时泛化差；DALI（NeurIPS 2025）需要额外训练 context encoder。
- **思路**：把自适应阈值、短时突触可塑性、内禀可塑性当作前向传播中的快权重，不做梯度更新，在线推断上下文。
- **Benchmark**：CARL-DMC，沿用 DALI 设置。
  - Ball-in-Cup 变重力和绳长；Walker 变重力和执行器强度。
  - 训练和测试分内插、外推、混合三种区间。
  - 预算：Ball-in-Cup 20 万步，Walker 50 万步，10 个种子。
- **Baseline**：Dreamer-DR、cRSSM-S/D（oracle）、DALI、Spiking-WM、SpikeWorld。
- **与 A 的关系**：可共用 A 的持久态骨干，适合作为第二篇论文。

---

## 附录 B：SNN 论文的问题陈述范式

| 论文 | 开篇 | 提出问题 | 核心洞察 | 证据形式 |
|---|---|---|---|---|
| PSN（NeurIPS 2023） | 直接点出机制缺陷 | "charge-fire-reset … can only be simulated serially and can hardly learn long-time dependencies" | "We find that when removing reset, the neuronal dynamics can be … parallelized" | 速度与精度；"first study about …" |
| TC-LIF（AAAI 2024） | 从任务需求切入：有用线索被长延迟分隔 | "remains a challenging task for SOTA SNNs to establish long-term temporal dependency" | 生物启发的双区室设计 | 梯度传播理论分析 + 分类实验 + 能效 |
| SiLIF（2025） | 多状态神经元的优势 | "gradient propagation through these dynamics often leads to instabilities" | 借鉴 SSM 的参数化与初始化 | 脉冲神经元中的 SOTA + 性能–效率权衡 |
| Spiking-WM（PNAS 2025） | 神经科学事实："Dendritic computation is key …" | 需要具备长时记忆的脉冲神经元 | 多区室树突整合 | DMC + 语音 |

**共同结构**：SNN 的优势 → "However" 加一个具体的机制缺陷 → "We find / identify" 根因 → 提出方法（通常是生物启发）→ 理论分析 → 实验（SNN 中的 SOTA + 效率）→ 一句意义陈述。

**本方案摘要的对应**：具身需求 → SNN 的承诺 → 发现这一能力没被用上 → 根因（两级复位）→ 方法（持久态、解耦、精确对偶）→ 结果 → 一句结论。

---

## 参考文献

- Sun, Y., Zhao, F., Lyu, M., & Zeng, Y. (2025). Spiking world model with multicompartment neurons for model-based reinforcement learning. *PNAS*. [论文](https://www.pnas.org/doi/10.1073/pnas.2513319122) ｜ [PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12718366/) ｜ [arXiv 2503.00713](https://arxiv.org/abs/2503.00713) ｜ [代码](https://github.com/Brain-Cog-Lab/Spiking-WM)（commit 602a7c8）
- Samsami, M. R., et al. (2024). Mastering memory tasks with world models (R2I). *ICLR 2024*. [arXiv 2403.04253](https://arxiv.org/abs/2403.04253) ｜ [代码](https://github.com/chandar-lab/Recall2Imagine)
- Pasukonis, J., Lillicrap, T., & Hafner, D. (2022). Evaluating long-term memory in 3D mazes. [arXiv 2210.13383](https://arxiv.org/abs/2210.13383) ｜ [代码与数据](https://github.com/jurgisp/memory-maze)
- Morad, S., et al. (2023). POPGym: Benchmarking partially observable reinforcement learning. [arXiv 2303.01859](https://arxiv.org/pdf/2303.01859)
- Shen, S., et al. SpikingSSMs: Learning long sequences with sparse and parallel spiking state space models. *AAAI*. [arXiv 2408.14909](https://arxiv.org/abs/2408.14909)
- Fang, W., et al. (2023). Parallel spiking neurons with high efficiency and ability to learn long-term dependencies (PSN). *NeurIPS 2023*. [arXiv 2304.12760](https://arxiv.org/abs/2304.12760)
- Zhang, S., et al. (2024). TC-LIF: A two-compartment spiking neuron model for long-term sequential modelling. *AAAI 2024*. [arXiv 2308.13250](https://arxiv.org/abs/2308.13250)
- Higuchi, S., Kairat, S., Bohte, S., & Otte, S. (2024). Balanced resonate-and-fire neurons. *ICML 2024*. [PMLR](https://proceedings.mlr.press/v235/higuchi24a.html)
- SiLIF: Structured state space model dynamics and parametrization for spiking neural networks. [arXiv 2506.06374](https://arxiv.org/abs/2506.06374)
- Dendritic resonate-and-fire neuron for effective and efficient long sequence modeling. [arXiv 2509.17186](https://arxiv.org/abs/2509.17186)
- P-SpikeSSM. *ICLR 2025*. [arXiv 2406.02923](https://arxiv.org/pdf/2406.02923)
- Dampfhoffer, M., et al. (2023). Are SNNs really more energy-efficient than ANNs? An in-depth hardware-aware study. IEEE. [IEEE Xplore](https://ieeexplore.ieee.org/document/9927729/)
- DALI: Dynamics-aligned latent imagination in contextual world models for zero-shot generalization. *NeurIPS 2025*. [arXiv 2508.20294](https://arxiv.org/abs/2508.20294) ｜ [代码](https://github.com/frankroeder/DALI)
- SpikeWorld: Fast-state adaptation for frozen spiking world models. [arXiv 2608.07712](https://arxiv.org/abs/2608.07712)

*说明：本方案由 AI 辅助调研整理，检索为非系统性检索。PSN、SpikingSSM、POPGym、TC-LIF 的作者与完整标题未在本次调研中逐一核对，投稿前请核对全部书目信息。*
