# Stage 2.5 诚信核查报告（Mode 1，送审前）

- 对象：`paper/`，基线 commit 9e7e629（2026-10-03 11:40 的草稿）
- 核查时间：2026-10-03 11:45–12:00
- 流程：ARS academic-pipeline v3.22.2，`integrity_verification_agent` 的 Mode 1，以及 7 类 AI 研究失效模式清单
- **结论：第 1 轮 FAIL（2 项 SERIOUS，8 项 MEDIUM）。全部修正后复核，结果为 PASS WITH NOTES。** 失效模式清单中 Mode 4 为 SUSPECTED，进入 Stage 3 前需要作者确认（见第 7 节）。

## 1. Phase A：参考文献

| 检查 | 结果 |
|---|---|
| A0/A1 存在性，A2 著录信息 | **57/57 VERIFIED**：26 条按 DOI 查 Crossref，31 条按标题查 arXiv API。判定条件：标题相似度 > 0.85，第一作者一致；有 DOI 的条目年份也须一致。脚本和逐条输出：`review/audit/verify_refs.py`、`review/audit/phaseA.txt` |
| 会议名称与年份 | 今天早些时候已逐条核对（Crossref、arXiv 注释、会议页面），本轮未发现变化 |
| A3 幽灵引用 | 0 条正文引用缺少 bib 条目；9 条 bib 条目未被引用（bellec2020eprop、fang2021plif、fang2021sew、gu2022s4、hausknecht2015drqn、maass1997networks、savva2019habitat、yao2023sdt、zhou2023spikformer）。BibTeX 不会输出它们，对 PDF 没有影响 |
| 作者完整性 | davies2018loihi 只列 10 位作者加 "others"，Crossref 记录为 23 位。MINOR，ieeenat 样式本来也会截断 |

## 2. Phase B：引用语境抽查（16/48 = 33%）

| 引用 | 正文说法 | 核对依据 | 结论 |
|---|---|---|---|
| sun2025spikingwm | 在 DMC 和语音分类上评测；改变时间常数对性能影响不大 | 研究方案 F2，以及原文 §3.3.1 原句 "adjustments to different membrane time constants … do not substantially impact overall model performance" | 准确 |
| samsami2024r2i | BSuite 和本体感受 DMC 上策略用 (z,h)，POPGym 用隐状态 | R2I 原文 §4（"we leverage the output state policy"；"For R2I, we use the hidden state policy"；DMC proprio 用 output state） | 准确 |
| samsami2024r2i | BSuite 结果出自图 7，10 个种子的中位数 | 原文图 7 图注 "averaged over 10 independent runs … (Median reward plotted)" | 准确 |
| samsami2024r2i | RepeatPrevious 是"三个记忆需求最高的 POPGym 任务"之一 | 原文 "we select the three most memory-intensive environments: RepeatPrevious, Autoencode, and Concentration" | 准确 |
| samsami2024r2i | Easy k=4，Medium k=32 | 原文表 4 "Parallel events to recall 4 / 32 / 64" | 准确 |
| samsami2024r2i | 训练窗口须覆盖整个 episode | 原文附录 "the success of our model is strongly contingent on accommodating the entire episode within a single sequence" | 准确；本轮新写入附录 A |
| samsami2024r2i | POPGym 官方配置 | 官方代码 `configs.yaml`：8 个环境，train_ratio 64，batch 4×1024 | **原文说法过强，已修正（M4）** |
| pasukonis2023memorymaze | 常数基线 23.9 MSE / 80.8%；RSSM 8.3 / 89.0；RSSM TBTT 6.2 / 94.0 | 原文表 4 | 准确 |
| pasukonis2023memorymaze | 29k 训练局 + 1k 评估局；取后半段（500 步）；1M 次更新；RSSM 状态 2048 维 | 原文 §3、§5，附录 E | 准确 |
| huebotter2025spiking | 循环 LIF 网络端到端训练的脉冲模型控制 | arXiv 摘要 | 准确 |
| qin2025grsn | 门控循环脉冲神经元，状态跨步保留，用于 POMDP 的无模型 RL | 标题、DOI 记录，以及研究方案中的文献摘录 | 准确 |
| orvieto2023lru | 归一化 √(1−λ²)；相位 U[0,2π) 初始化 | LRU 原文（此前已核对） | 准确 |
| horowitz2014energy | 45 nm 下 AC 0.9 pJ，MAC 4.6 pJ（加法 0.9 + 乘法 3.7） | 常用数值 | 准确 |
| dampfhoffer2023energy | 访存能耗可能占主导 | 论文标题和结论 | 准确 |
| ni2023transformers | 区分记忆长度与信用分配长度 | 标题 "Decoupling Memory from Credit Assignment" | 准确 |
| hafner2025dreamerv3 | DMC proprio 500k 步，train ratio 512 | 本仓库 `configs.yaml` 的 `dmc_proprio` 块（dreamerv3-torch 复现设置） | 准确 |

## 3. Phase C：数据核对（正文中所有数字与 `runs/` 日志对照）

用 `review/audit/probe_check.py` 从 `runs/recall` 和 `runs/diag_rp` 重算全部探针数值；RL、能耗、参数量直接读日志、`energy.json` 和 `scripts/review_configs.py --params`。

- 与日志一致：表 1（Phase 0，两个种子）、表 3、表 5、附录 D 的阈值扫描、附录 F 的诊断表、能耗（5.88 / 17.18 µJ，2.9×）、参数量、第 3 节的机制数值（0.004–0.009、0.19、0.88、发放率上限 0.06 时 2000 次更新）、附录 A 的训练超参数。
- 发现的不一致见第 6 节的 M1、M2、M3、M7、m3–m5。
- 一处误报：`fig/*-1.png` 是 03:13 的旧预览图，其中 FwF 曲线不完整。LaTeX 实际用的是 11:32 生成的 PDF 版本，数据正确。旧 PNG 已删除，避免再被误读。

## 4. Phase D：原创性

- D1：正文和附录与 R2I、Memory Maze 两篇原文全文做 8-gram 比对，共享 8-gram 为 0（本文共 6662 个 8-gram）。
- 局限：没有使用商业查重库。全文是本项目中从零撰写的。
- D2：作者信息为匿名，不适用。

## 5. Phase E：高影响主张核对（全部为本文自己的实验结果，依据是 `runs/` 日志）

| 主张 | 依据 | 结论 |
|---|---|---|
| "only model … that solves 64-step recall"（摘要、引言） | 表 2。R2I 没有在 L=64 上跑，而它公开的结果在 71 步约为 0.9 | **范围过宽，已限定为"我们比较的模型中"（S1）** |
| "removing both resets restores memory up to 32 steps"（摘要） | 表 1：只去掉复位（A+B）在 32 步失败，还要加上 G | **错误归因，已改（S2）** |
| 101 步：FwF 用 362k 步达到 0.9；DreamerV3 在 51 步以上失败 | `runs/r2i` 日志 | 准确（FwF L101 最近几次评估为 0.88–0.94） |
| 只有 FwF 能回忆恰好 32 步前的牌（AUC 0.93） | `runs/diag_rp`：6k 次更新时 0.931 | 准确 |
| 能耗低 2.9× | `energy.json` | 准确 |
| "remembers further than its ANN counterpart on standard memory benchmarks"（结论） | Memory Maze 当前数值是 DreamerV3 更好（9.8 vs 11.5，仍在运行） | **过强，已改为限定在 BSuite 和精确延迟回忆上（M6）** |
| 第 3 节"有复位时时间常数不起作用" | 表 1 | 准确，但 L32 有一个种子成功、正文没交代（M1，已补） |

## 6. 问题清单与处理

| ID | 级别 | 问题 | 处理 |
|---|---|---|---|
| S1 | SERIOUS | 摘要和引言写"唯一解决 64 步的模型（含 ANN）"，但 R2I 未在 64 步上评测，它公开的 71 步结果约 0.9 | 改为"我们比较的模型中唯一" |
| S2 | SERIOUS | 摘要把 32 步的恢复归因于"去掉两种复位"；实际需要去掉复位并归一化写入增益 | 已改 |
| M1 | MEDIUM | Spiking-WM 在 L32 探针上有一个种子成功（表 1、表 3 的 12k），正文没有解释 | 第 3.2 节补一句：靠自持循环发放，只存在于脉冲中，学得慢且不可靠；表 3 该格附最终 AUC 0.66 |
| M2 | MEDIUM | 第 5.2 节的 POPGym 数值过期（正文 0.61 / 0.03，表中 0.64 / 0.21） | 正文改用 `tables/numbers.tex` 中的自动生成宏，`make data` 时随日志更新 |
| M3 | MEDIUM（Mode 6） | 附录表 8 写 L51–101 的 batch 为 16×64，实际为 64/80/128 步窗口 | 已改，并说明窗口覆盖整个 episode（与 R2I 一致） |
| M4 | MEDIUM（Mode 6） | 写"使用 R2I 的配置"，实际只用了环境侧设置；R2I 的 batch 是 4×1024 | 表 2 图注、第 5.1 节、附录 A 都改为"R2I 代码的环境设置"；附录写明 R2I 本身的配置和两处改动 |
| M5 | MEDIUM（Mode 4） | POPGym 上 FwF 的 actor-critic 读取 6144 个树突脉冲，参数 12.9M，基线为 3.4M，文中没有披露；第 5.1 节写"只有循环核不同" | 附录 A 写明 12.9M；第 5.1 节改为"只有循环核不同，FwF 在 POPGym 上策略输入也不同" |
| M6 | MEDIUM | 结论"在标准记忆基准上比 ANN 记得更远"包括了 Memory Maze，而该任务当前是 DreamerV3 更好 | 已限定 |
| M7 | MEDIUM | L64 上 FwF 首次达到 0.9 后在 0.5–1.0 间波动（最终 0.73），正文只写了"solves" | 补充说明（其他模型都不超过 0.5） |
| M8 | MEDIUM（Mode 6） | 第 4.3 节说 Memory Maze 上"把树突状态送给 actor 和 critic"，但 Memory Maze 是离线探针，没有 actor | 改为"探针读取同样的特征"；附录 E 图注同步修改 |
| m1 | MINOR | 图 1 图注"Spiking baselines fail at 8–32 steps"不精确 | 改为 "stop at 32 steps or fewer" |
| m2 | MINOR | 图 3 图注称所有曲线都是评估回报，但 R2I 曲线是训练回报 | 已注明 |
| m3 | MINOR | "learns every length fastest"：L8、L16 上有并列 | 改为 "at least as fast as any other model" |
| m4 | MINOR | 消融："adds nothing"（3.5k vs 4k）、"twice"（6.5k vs 3.5k） | 改为 "changes little" 和 "almost twice"，并注明全共振在 L64 上没有保持在 0.9 以上 |
| m5 | MINOR | 第 3.2 节 "halves"（8–8.5k → 4.5k） | 改为 "nearly halves"，给出数值 |
| m6 | MINOR | 参数量没说明是哪个 benchmark | 注明 "On POPGym" |
| m7 | MINOR | `fig/*-1.png` 是旧预览 | 已删除 |
| m8 | MINOR，未改 | Loihi 条目作者列表截断；图 2（TikZ）溢出版心 37 pt | 留到 Stage 5 排版时处理 |
| m9 | MINOR，未改 | 9 条未引用的 bib 条目 | 不影响输出；Stage 5 清理 |

第 1 轮修正后复核：上表中改动过的 18 处都已对照日志或原文重新确认；`make` 和 `make full` 编译通过，无未定义引用。正文页数不变（参考文献从第 8 页开始）。

## 7. AI 研究失效模式清单（7 类）

| 模式 | 状态 | 依据 |
|---|---|---|
| 1 实现错误通过自查 | CLEAR | 每个数字都对应到有退出码的运行日志。此前发现的两个 bug（`is_first` 写回、固定 τ）对应的运行已隔离到 `runs/_invalid_*`，并已重跑。同一 benchmark 内所有模型的设置逐项对照过（`scripts/review_configs.py`） |
| 2 引用幻觉 | CLEAR | 57/57 VERIFIED |
| 3 结果幻觉 | CLEAR | 表格由脚本从日志生成；正文中手写的数字已逐个复核（第 3 节）。种子数如实写明（表 1 两个种子，其余一个，正文保留 TODO） |
| 4 依赖捷径 | **SUSPECTED（披露层面）** | POPGym 上 FwF 的 actor-critic 是基线的 3.8 倍（12.9M vs 3.4M）。世界模型探针不经过策略，证明了记忆来自世界模型；但 POPGym 的 RL 回报可能部分来自更大的策略网络。同样 3.4M 策略、只读胞体脉冲的对照（`popgym/*/PD2Lreshalf_tau1000`）正在跑，目前在 202k 步：Easy 约 0，Medium 为 −0.5。已披露参数量，**需要你确认处理方式** |
| 5 把 bug 包装成发现 | CLEAR | 反直觉的结果（阈值越低记忆越好；Spiking-WM 在 L32 上偶尔成功）都有机制测量支持（线索幅值 0.004–0.009，保留曲线），并与 Spiking-WM 原文自己报告的"τ 不敏感"一致 |
| 6 方法描述与实际不符 | 第 1 轮 SUSPECTED，修正后 CLEAR | M3、M4、M8 已按实际运行配置改正 |
| 7 早期框架锁定 | CLEAR（附提醒） | 主张已限定在已完成的结果上。POPGym、Memory Maze、DMC 的最终结果出来后，第 5.2/5.4/5.6 节和结论要再核一次 |

## 8. 工具局限

- 原创性：只比对了手头有全文的两篇来源和 arXiv 摘要，没有查重库。
- Phase B：Spiking-WM、GRSN 两篇期刊文章没有拿到全文，依据的是早前研究方案里的原文摘录和 DOI 记录。
- 仍在运行的结果（`\pend`）目前只核对了"当前值与日志一致"；最终值出来后要重新核对（Stage 4.5）。
