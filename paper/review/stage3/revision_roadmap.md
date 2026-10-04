# Revision Roadmap（按优先级）

稿件：*Fire Without Forgetting*（CVPR 2027，投稿截止 2026-11-16）
来源：`editorial_decision.md`（决定：Major Revision）与五份审稿报告。
本文件按本次任务要求**按优先级排序**；ARS 规范要求的来源顺序追溯清单见 `editorial_decision.md` 末尾（两者条目一一对应）。

## 使用说明

- **优先级**：P1 = 阻碍录用，投稿前必须完成；P2 = 显著加强中心主张，强烈建议；P3 = 次要，按页数与时间取舍。
- **严重度**：沿用审稿报告中的 Critical / Major / Minor；"Critical（DA）"表示 Devil's Advocate 判为 Critical 且编辑裁决为 VALIDATED（可修复）。
- **材料**：「现有材料」= 只需改写、重新统计现有日志或对现有检查点做分析；「新运行」= 需要新的训练运行（注明规模，均来自论文中已有的预算设定）。
- **审稿人编号**：R1 = Area Chair，R2 = 方法，R3 = SNN，R4 = 世界模型/RL，R5 = Devil's Advocate。

## 作者已固定的决定（本 Roadmap 不提议推翻）

| 固定决定 | 审稿中的关切（各陈述一次） | 本 Roadmap 的处理 |
|---|---|---|
| 所有运行单种子 | R2 W2、R5 M9 | RM-4：单次运行内部的变异性报告、更严格判据、直接的措辞 |
| FwF 在所有任务上只用一种配置（actor/critic 读胞体脉冲 + ON/OFF 编码的树突状态） | R2 W1、R3 W1、R5 C1 | RM-2：不改 FwF 配置，只做披露与对照（对照放在附录消融或基线侧） |
| 方法不定位为 spiking SSM / R2I 的脉冲版 | R3 W4、R4 W5、R5 M5 | RM-7：不改定位，用直接陈述 + 证据说明 PDN 的具体机制差异 |
| 采用源论文标准配置，不调参 | R2 W3、R4 W1/W3 | RM-3、RM-6：让基线回到其源配置/源协议，或并列源论文数字 |
| 不做事件相机实验 | R1 W2（CVPR 适配） | RM-5、RM-6：用 Memory Maze 与（可选）像素 DMC 加强视觉相关性 |

---

## P1 — 阻碍录用（必须完成）

### RM-1　标题性主张与论文自身图表对齐
- **问题**：摘要/图 1/引言/结论中的"solves 101-step recall""recalls across all lengths""under the protocol of R2I""the only model ... that solves 64-step recall""3.2× less energy per step""the only signal that leaves the neuron""while firing sparsely"强于图表证据：图 1 中 FwF 在 L=71、101 的最终回报约 0.46、0.40（与 R2I 报告的 101 步约 0.35 同一水平）；"solve"为单种子首次越过 0.9；协议改了奖励头；64 步结论依赖 500k 预算；能耗只计滤波器；actor 读取树突编码。
- **提出者**：R1 W1；R2 W2、W4；R3 W1、W3、W5；R4 W2；R5 M1、M2、M6、M7（对应阻碍问题 B1）。
- **严重度**：Major
- **具体修改**：
  1. **统一度量**：BSuite 与 POPGym 使用同一个预先固定的最终度量（论文已在 POPGym 上使用"mean of the last three evaluations"，建议 BSuite 也用它），图 1 纵轴改为该度量；"steps to 0.9"保留为样本效率指标。
  2. **摘要改写示例**（数字以统一度量重新填写）：
     "On BSuite memory length, FwF is the only model we train, spiking or not, that reaches a return of 0.9 at 64 steps within 500k environment steps. With R2I's lengths and budget, it reaches 0.9 at 51, 71 and 101 steps and ends at [x], [y], [z], where all other models stay near chance at 71 and 101."
  3. 图 1 caption："FwF recalls across all lengths" → "FwF is the only model above chance at L ≥ 64"。
  4. 协议命名："under the protocol of R2I" → "with R2I's lengths, budget and per-environment seeds, and an action-indexed reward head (Supp. B)"。
  5. 能耗句：在 RM-9 完成后改为全智能体数字，或限定为"its world-model filter uses 3.2× less arithmetic energy per step than DreamerV3 (45 nm estimate)"。
  6. 脉冲句（与 RM-8 一致）："Within the world model, neurons exchange only binary spikes; the actor and critic additionally read the dendritic state through an 11-level spike encoding."
  7. 结论："while firing sparsely" → "at a recurrent firing rate of 0.15"。
  8. 贡献要点 3 的措辞待 RM-5 的 Memory Maze 结论确定后同步。
- **材料**：现有材料（改写 + 用现有日志重算统一度量）。

### RM-2　披露 actor-critic 的输入与容量差异，并完成区分对照
- **问题**：§5.1 称各模型"differ only in the recurrent core"，并只给世界模型参数量；Supp. A 显示 FwF 的 actor-critic 为 12.9M（基线 3.4M），并多读 6144 路树突编码输入。Table 7 显示去掉该输入后 POPGym Easy 0.71→0.11、Medium 0.45→−0.03、DMC walker 880→504；宽 actor 对照只在 Easy 上运行且未完成。
- **提出者**：R1 W4；R2 W1；R3 W1、W3；R5 C1（编辑裁决 VALIDATED；阻碍问题 B2）。
- **严重度**：Critical（DA）/ Major
- **具体修改**：
  1. §5.1 加入直接陈述："FwF's actor and critic also read the dendritic state (6144 binary channels, Sec. 4.3) and have 12.9M parameters, against 3.4M for the baselines, whose actors read (z_t, h_t). DreamerV3's h_t is its full continuous GRU state."
  2. §5.2 加入 Table 7 的结论（这是论文的有利证据）："BSuite results do not depend on the dendritic readout: with an actor that reads (z_t, h_t) only, FwF solves every length (Supp. Table 7). POPGym and DMC returns require it."
  3. 把 Table 7 的两行压缩进正文 Table 2 的底部（或在正文给出数字），完整表留在补充材料。
  4. 完成宽 actor-critic（12.9M，只读 (z_t, h_t)）对照，并**扩展到 RepeatPrevious Medium**——Medium 是论文宣称独有的能力。
  5. 可选（加强对照、但不改 FwF 配置）：对有持久状态的基线（GRSN、Spiking-WM-NR）用相同 ON/OFF 编码把其持久膜状态提供给 actor，在 RepeatPrevious Easy/Medium 上运行。结果无论正负都可直接陈述："the readout helps only when the state it reads holds the memory"。
- **材料**：1–3 现有材料；4 中 Easy 已在运行，Medium 为**新运行**（1 次，2M 步）；5 为**可选新运行**（4 次，2M 步）。

### RM-3　修正与 R2I 的比较
- **问题**：Table 2 中 R2I（官方代码）在 RepeatPrevious Easy −0.50、Medium −0.49，均为随机水平；R2I 原文报告其在 RepeatPrevious Medium/Hard 达到 SOTA，而本文中连 Spiking-WM 都能学会 Easy。BSuite 上只引用 R2I 已发表数字，且奖励头不同、种子数不同（单种子 vs 十种子中位数）。正文"consistent with R2I"只适用于 DreamerV3。
- **提出者**：R1 W3；R2 W3；R4 W1、W2；R5 C2（编辑裁决 VALIDATED；阻碍问题 B3）。
- **严重度**：Critical（DA）/ Major
- **具体修改**：
  1. **排查 POPGym 运行**：与 R2I 官方 POPGym 配置逐项对照（奖励缩放、观测格式、环境版本、train ratio、回放）；对照 R2I 附录中的学习曲线，确认其在 2M 步时是否已开始学习 Easy。
  2. 若 R2I 原文在 2M 步内已学会 Easy，则本文运行存在问题：修复后重跑，或从主表删除该行。不要在主表保留与原文矛盾且未解释的复现。
  3. Table 2 增加"R2I (published)"行，填入 R2I 原文在相同步数与最终处的数值，注明来源与种子数；BSuite 51/71/101 列填入原文 1M 步中位数（约 0.9 / 0.9 / 0.35，10 seeds，default reward head）。
  4. §5.2 用同一度量（RM-1 的统一度量）比较 FwF 与 R2I 已发表数字，并直接写出结论（例如 101 步上两者处于同一水平、FwF 为脉冲模型且只用 [x] 步达到 0.9）。
  5. 修改"consistent with R2I"一句，使其只指 DreamerV3 在 Medium 上失败。
  6. 可选：用 R2I 官方代码与其**源论文 BSuite 配置**（默认奖励头）在 51/71/101 上各运行一次，作为同一环境实现下的直接参考；这符合"源论文标准配置"原则。
- **材料**：1、3–5 现有材料；2 视排查结果可能需要**新运行**（R2I POPGym，Easy + Medium）；6 为**可选新运行**（3 次，1M 步）。

### RM-4　单种子下的变异性报告与更严格判据
- **问题**：除 Table 1 外均为单种子；"first reaches 0.9"判据对噪声敏感；每次评估的回合数、评估间隔、EMA 系数未报告；图 3 中 L=71、101 曲线峰值后回落。
- **提出者**：R2 W2、W7；R1 W1；R5 M9。
- **严重度**：Major（关切只陈述一次；作者已固定单种子）
- **具体修改**（全部基于现有日志）：
  1. 用直接陈述替换 TODO："All RL and probe results are single training runs per setting; Table 1 uses two seeds per cell."
  2. 报告每次评估的回合数与评估间隔；对每个评估回报给出 95% 二项分布（Wilson）区间，在图 3 中作为阴影带。
  3. 补充表（可放 Supp.）：每个 BSuite 长度的 steps to first 0.9、steps to 3 consecutive evaluations ≥ 0.9（或 EMA ≥ 0.9）、最终度量、解决后均值 ± 标准差、≥0.9 与 ≥0.5 的评估比例（论文已对 L=64 给出 94% / 0.87，扩展到所有长度）。
  4. 探针 AUC 用 512 个留出回合做 bootstrap 区间。
  5. 解释 L=71、101 的峰值后回落：若日志中有世界模型在同一时刻的奖励 AUC，报告它以区分"策略退化"与"世界模型遗忘"。
  6. 在正文点明 Table 1 两个种子高度一致（如 0.5k / 0.5k），作为单次运行变异较小的旁证。
- **材料**：现有材料。

### RM-5　完成 Memory Maze 并直接陈述结论
- **问题**：临时结果中 FwF 墙体准确率 81.6%（常数预测器 80.8%），低于全部脉冲基线（82.7–82.9%）与 DreamerV3（85.3%）；目标物体 MSE 10.7 优于脉冲基线、不及 DreamerV3（9.8）；所有 S 规模模型远离参考 RSSM（89.0% / 94.0%）；memory gain 未报告；FwF 的探针多读树突编码，输入不对等。
- **提出者**：R1 W2；R2 W8；R4 W4；R5 M4。
- **严重度**：Major
- **具体修改**：
  1. 按既定计划报告所有模型的 memory gain（state 探针 − image-only 探针）。
  2. FwF 同时报告"spikes only"与"spikes + dendritic spikes"两种特征集，在正文写明每个模型的探针读取哪些特征。
  3. 以目标物体 MSE 为主要记忆指标（更依赖跨视野记忆），墙体准确率为辅。
  4. 在 §5.4 写一句直接结论，包括"在 DreamerV3-S 规模与 1M 次更新下，所有模型的墙体准确率都远低于参考 RSSM（512 vs 2048 维状态）"。
  5. 若 FwF 的 memory gain 不超过 DreamerV3，贡献要点 3 改为"we evaluate ... on Memory Maze, where [直接结论]"，不把 Memory Maze 列为记忆优势的证据。
- **材料**：计划中的运行（Oct 10–13 完成）+ image-only 探针（已在计划中）；无额外新运行。

### RM-6　更正 Spiking-WM 的评估背景与 DMC 对照
- **问题**：§5.1 称本体感受 DMC 是"Spiking-WM 被评估过的"非记忆对照；Spiking-WM 原文（PNAS 2025；arXiv 2503.00713 Table I）评估的是像素 DMC（1M frames，约 18.8M 参数，walker walk 938.5、cheetah run 577.2、cartpole swingup 791.0）。本文在本体感受、S 规模下复现得到 923 / 343 / 497。预注册判据"no degradation relative to Spiking-WM"因此对照的是非源配置且明显偏弱的基线；FwF（临时）walker 880 < 923，run/swingup 比 DreamerV3 低约 260/250 分。
- **提出者**：R2 W3；R4 W3；R5 M8；R1 W2（建议层面）。
- **严重度**：Major
- **具体修改**：
  1. 更正：§5.1 改为"DMC from proprioception is our non-memory control; Spiking-WM was evaluated on DMC from pixels [39]."；Table 4 增加脚注或一行列出 Spiking-WM 原文数字与协议差异。
  2. DMC 结论同时相对 DreamerV3（同构 ANN）陈述，并直接写出差距。
  3. 二选一：
     - **A（现有材料）**：保留本体感受 DMC，按 1–2 改写；把判据表述为"we report DMC to measure the cost of the memory design on non-memory control"。
     - **B（新运行，推荐在 DMC 仍可调整时采用）**：采用 Spiking-WM 原文的像素 DMC 协议运行三项任务（Spiking-WM、DreamerV3、FwF），既符合"源论文标准配置"原则，也增加一组视觉输入结果，提升 CVPR 适配性。
- **材料**：A 为现有材料；B 为**新运行**（3 任务 × 3 模型）。

---

## P2 — 显著加强中心主张（强烈建议）

### RM-7　用证据直接陈述 PDN 与 LRU / FFM / spiking linear-recurrence 层的机制差异
- **问题**：PDN 的写入归一化与初始化取自 LRU（文中已说明）；FFM（Morad et al., NeurIPS 2023）在 POMDP RL 中已使用对数间隔衰减与相位（"contextual drift"）；相关工作把 spiking state-space 层与 PDN 的区别归于应用场景（离线分类 vs 在线滤波），而非机制。现有消融（单一 τ=128 在 64 步与对数谱同样好；全共振与半共振在 k=32 同样好）没有隔离 PDN 特有元素。缺 FFM、LMU、S5 in-context RL（Lu et al., NeurIPS 2023）的引用。
- **提出者**：R1 W3；R3 W4；R4 W5；R5 M5。
- **严重度**：Major
- **具体修改**（不改变定位，不写防御性语句）：
  1. §2 增加"Memory cells for RL"一段，引用 FFM、LMU、S5 in-context RL，并逐一给出直接差异陈述（例如："FFM is a dense ANN cell with gated complex-valued traces; PDN neurons communicate only by binary spikes, have no gates, and keep real integrator and complex resonant channels separate."）。
  2. §4.2 直接写出继承与新增："We adopt the LRU's normalization and initialization. PDN adds four elements: binary spikes as the only signal between neurons inside the world model, no spike feedback into the dendrites, a stateless soma, and separate integrator and resonant channels."
  3. 为每个新增元素指向证据：
     - 积分/共振分离：Table 5 中"all channels resonant"（最接近全复数 LRU 的变体）在 64 步需要近两倍更新且 AUC 不稳定——现有材料；
     - 无脉冲回馈、按环境步计时：Spiking-WM-NR 在 64 步失败——现有材料；
     - 无状态胞体：**新增探针消融**"soma with membrane and reset"（memlen-64、RepeatPrevious k=32）；
     - 无脉冲回馈的直接检验：**新增探针消融**"dendrites also receive the layer's spikes"。
  4. 可选：在探针中加入一个已发表的 spiking linear-recurrence 层（如 SiLIF 或 P-SpikeSSM 的层）作为核心，以及一个 ANN LRU 或 FFM 核心的 DreamerV3 参考。无论结果如何，都能支撑一条直接陈述（例如"FwF matches an ANN linear recurrence in memory while communicating by binary spikes"或"FwF remembers longer than ..."）。
- **材料**：1–3 中前两项为现有材料；3 中两项消融为**新运行**（低成本探针，每项 12k / 10k 次更新）；4 为**可选新运行**（低成本探针）。

### RM-8　精确化脉冲相关表述（分级读出与层归一化）
- **问题**：摘要与 §4.1 称二值脉冲是离开神经元的唯一信号，§1/§4.2 称分级值从不跨越突触；但 actor/critic 与 Memory Maze 探针读取层内标准化后 ON/OFF 编码的树突状态（11 级），PDN 内部的 LN(B x_t)、N(u_t) 需要跨神经元的分级统计量。
- **提出者**：R3 W1、W2；R5 M6。
- **严重度**：Major
- **具体修改**：
  1. 摘要、§1（"Graded state never crosses a synapse; it stays inside the neuron"）、§4.2（"Graded values never cross a synapse"）统一改为准确陈述，例如："Inside the world model, neurons exchange only binary spikes; graded dendritic state stays inside each neuron except through layer normalization, which all compared spiking world models use, and through the actor's 11-level spike readout."
  2. §4.3 给出读出的维度（512 × 6 状态变量 × ON/OFF = 6144 路，每路 T 帧）。
  3. Limitations 扩展为："deployment needs neuromorphic cores with multi-variable neurons and support for layer normalization"。
  4. 可选：在现有检查点上评估把 LN 折叠为固定逐通道仿射（用运行统计量）后的探针 AUC；若记忆不受影响，可直接写出这一事实。
- **材料**：1–3 现有材料；4 为现有检查点上的评估（无需训练）。

### RM-9　扩展能耗计量范围
- **问题**：能耗只计世界模型滤波器、只在 BSuite L=32 模型上测；不含 FwF 放大约 3.8 倍的 actor（及其 6144 × T 路输入）；未说明 PDN 输出脉冲在 T 个仿真步上重复时下游 AC 的计数方式；未说明 FP32 假设。
- **提出者**：R3 W3；R5 M7（及其"未被检验的前提"）。
- **严重度**：Major
- **具体修改**：
  1. 在现有检查点上做操作计数，按基准（BSuite、POPGym、DMC、Memory Maze）分别报告"滤波器"与"滤波器 + actor"的每步能耗与比值。
  2. 注明 Horowitz 常数对应 FP32（加法 0.9 pJ，乘加 4.6 pJ），说明重复脉冲帧的计数方式。
  3. 可选：按每步参数读取量给出访存的粗略估计（同一假设应用于所有模型），直接回应"计入访存后是否仍有优势"。
  4. 摘要与 §5.6 的数字按 1 更新。
- **材料**：现有材料（操作计数，无需训练）。

### RM-10　BSuite 预算限定与共享种子说明
- **问题**："the only model ... that solves 64-step recall"依赖 L=64 的 500k 预算（DreamerV3 在 L=51 需要 902k）；L≤64 运行在四个并行环境间共享种子，非标准设置。
- **提出者**：R2 W4；R4 W9；R5 M2。
- **严重度**：Major
- **具体修改**：
  1. 主张加预算限定（见 RM-1 摘要示例），并指出等更新预算的探针（Table 3）中 DreamerV3 在 64 步同样失败（AUC 0.50）——这是不依赖 RL 预算的现有证据。
  2. 在 Supp. A 说明共享种子的原因，以及四个环境是否因此产生相同的线索序列。
  3. 可选：在 1M 的 R2I 协议下补跑 DreamerV3 与 FwF 的 L=64。
- **材料**：1–2 现有材料；3 为**可选新运行**（2 次，1M 步）。

### RM-11　披露开发顺序并增加一个留出记忆任务
- **问题**：共振通道是在旧版（仅积分通道）于 RepeatPrevious Medium 失败后加入的（Supp. F），随后 RepeatPrevious Medium 成为主要证据；没有开发中未使用过的记忆任务。
- **提出者**：R1 W6；R4 W7；R5 M3。
- **严重度**：Minor（R1、R4）/ Major（R5）
- **具体修改**：
  1. §5.3 直接写出开发顺序："Resonant channels were added after the integrator-only model failed RepeatPrevious Medium (Supp. F); the configuration has not been changed since."
  2. 在不改模型配置的前提下增加一个留出任务：最低成本是 RepeatPrevious Hard 的**世界模型探针**（FwF 与所有基线）；若时间允许，再做 RepeatPrevious Hard 的 RL（至少 FwF 与 DreamerV3）。
- **材料**：1 现有材料；2 为**新运行**（探针低成本；RL 为可选）。

---

## P3 — 次要修改（按页数与时间取舍）

### RM-12　补全干预表，并准确描述各重置的作用
- **提出者**：R2 W5；R5（次要问题）。**严重度**：Minor
- **修改**：Table 1 并入 A+G（Supp. D 阈值 0.5 的数据）与 L=64 / τ_d=512 行；补跑 A 在 L=32、B only 在 L=16、G only 在 L=8（各两次探针）；正文把 B 描述为"加速"，并在摘要层面写明"resets are necessary but not sufficient; the write gain decides recall beyond 16 steps"。
- **材料**：大部分现有材料；三个单元为低成本**新运行**（探针）。

### RM-13　保留时间谱必要性的探针
- **提出者**：R2 W6。**严重度**：Minor
- **修改**：在 τ=128 不够的设置（如 memlen-256 探针）上比较单一 τ=128 与对数均匀谱；若对数谱胜出，§4.2 的"no time constant is tuned per task"就有直接证据。
- **材料**：低成本**新运行**（探针）。

### RM-14　学习后保留时间与相位的机制分析
- **提出者**：R3 W6。**严重度**：Minor
- **修改**：画出 τ、θ 的初始化/训练后直方图；在 RepeatPrevious k=32 模型上按通道组（积分 vs 共振、按 τ 分箱）做线性解码，直接展示"共振相位编码延迟"。
- **材料**：现有检查点。

### RM-15　复现细节
- **提出者**：R2 W7；R3 W9。**严重度**：Minor
- **修改**：Supp. A 增加评估协议表（每次评估回合数、评估间隔、确定性/随机策略、图 3 EMA 系数）；写明 PDN 胞体的替代导数与宽度；正文加入代码发布声明。
- **材料**：现有材料。

### RM-16　术语与引用出处
- **提出者**：R4 W6、W8；R3 W5、W7、W8；R5（次要问题）。**严重度**：Minor
- **修改**：
  - "DreamerV3"在表与正文中称"DreamerV3 architecture (our implementation)"，首次出现时说明与官方实现的差异；
  - §3.1 加脚注：Spiking-WM 的树突发放重置来自其官方实现（`node.py`，`MCNode.calc_spike`），论文公式只写出胞体重置；
  - §4.1 首次出现"dendritic channels"时给出功能性定义（"linear state variables of a neuron; the analogy to dendrites is functional, not biophysical"）；
  - §2 增加一两句关于可学习突触/轴突延迟的直接比较（延迟需按任务设定最大延迟并为每个延迟付出存储；共振通道以固定状态维度编码任意延迟）；
  - 解释 FwF 能耗低于 Spiking-WM 的来源（PDN 每环境步更新一次，MCN 每仿真步更新 4 次）。
- **材料**：现有材料。

### RM-17　版式与页数规划
- **提出者**：R1 W5 及其次要问题清单。**严重度**：Minor
- **修改**：修正 Table 4 的 37pt overfull（缩小列间距或拆分）；同步 §5.6 文字与 Table 4 的 DMC 数字；为 RM-2（Table 7 两行进正文）、RM-5（Memory Maze 结论）与种子说明预留空间——可合并 Table 3 与 Table 5 的部分行、把 §2 "Dendrites" 段并入 §4.2、缩短图 2 caption；补充材料作为独立文件提交；摘要注明 1–1000 步是保留时间的初始化范围。
- **材料**：现有材料。

### RM-18　POPGym 度量一致性
- **提出者**：R2 W9。**严重度**：Minor
- **修改**：最终版统一在 2M 步处报告；所有模型同时给出训练回合回报以与 R2I 同口径比较；Easy 上直接写"FwF is on par with recurrent LIF (0.71 vs 0.69)"。
- **材料**：现有材料（待运行完成）。

---

## 汇总

| 条目 | 优先级 | 严重度 | 提出者 | 现有材料 | 新运行 |
|---|---|---|---|---|---|
| RM-1 主张与图表对齐 | P1 | Major | R1, R2, R3, R4, R5 | 是 | — |
| RM-2 actor-critic 披露与对照 | P1 | Critical（DA）/ Major | R1, R2, R3, R5 | 是 | Medium 宽 actor 1 次；可选基线读出 4 次 |
| RM-3 R2I 比较 | P1 | Critical（DA）/ Major | R1, R2, R4, R5 | 是 | 视排查结果；可选 R2I BSuite 3 次 |
| RM-4 单种子变异性报告 | P1 | Major | R1, R2, R5 | 是 | — |
| RM-5 Memory Maze 结论 | P1 | Major | R1, R2, R4, R5 | 计划中运行 | — |
| RM-6 Spiking-WM / DMC | P1 | Major | R1, R2, R4, R5 | 方案 A | 方案 B：像素 DMC 3×3 |
| RM-7 机制差异证据 | P2 | Major | R1, R3, R4, R5 | 部分 | 2 项低成本探针；可选参考核心 |
| RM-8 脉冲表述精确化 | P2 | Major | R3, R5 | 是 | — |
| RM-9 能耗范围 | P2 | Major | R3, R5 | 是 | — |
| RM-10 BSuite 预算与种子 | P2 | Major | R2, R4, R5 | 是 | 可选 L=64 @1M 2 次 |
| RM-11 留出任务 | P2 | Minor / Major（R5） | R1, R4, R5 | 部分 | Hard 探针；可选 Hard RL |
| RM-12 干预表 | P3 | Minor | R2, R5 | 大部分 | 3 个探针单元 |
| RM-13 谱必要性 | P3 | Minor | R2 | — | 1 组探针 |
| RM-14 τ/θ 机制分析 | P3 | Minor | R3 | 是 | — |
| RM-15 复现细节 | P3 | Minor | R2, R3 | 是 | — |
| RM-16 术语与引用 | P3 | Minor | R3, R4, R5 | 是 | — |
| RM-17 版式与页数 | P3 | Minor | R1 | 是 | — |
| RM-18 POPGym 度量 | P3 | Minor | R2 | 是 | — |

**最小必要路径**：P1 的六项中，RM-1、RM-4、RM-6（方案 A）与 RM-3 的改写部分只需现有材料；RM-2 需要一次新运行（Medium 宽 actor）；RM-5 依赖已在计划中的运行；RM-3 是否需要重跑 R2I 取决于排查结果。完成这些即可关闭三项阻碍问题（B1–B3）与两项 DA CRITICAL（C1、C2）。
