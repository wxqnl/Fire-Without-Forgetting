# 改论文会话的交接说明（2026-10-12 00:30）

本文件给新开的"改论文"会话用：读完就能接手，不必翻之前的对话。原会话继续负责实验（监控队列、释放 GPU、写 EXPERIMENTS.md 的运行状态），本会话只改论文。

## 1. 项目

- **Spike-Driven Dreamer（SD-Dreamer，宏 `\method` / `\methodfull`）**：CVPR 2027。注册截止 2026-11-10，投稿截止 2026-11-16。
- **路径：** 代码在 `/home/user01/Minko/calude/fwf`，论文在 `fwf/paper`。编译和生成图表见 `paper/README.md`：
  - `make data` 从 `../runs` 重新生成图和表；
  - `make` 编译正文；
  - `make full` 编译正文加附录。
- **记录：**
  - 实验与决策记录在 `fwf/EXPERIMENTS.md`，最新是 §19.50–19.55。
  - 内部状态表在 `fwf/RESULTS.md`，由 `scripts/main_tables.py` 生成。注意：这里 POPGym 用最后 3 次评估平均，论文表用 MMER。
- **完整的旧对话：** `/home/user01/Minko/.claude-personal/config/projects/-home-user01-Minko-calude/78e6225c-7b28-4236-b968-2967c664017d.jsonl`。只在需要查原话时搜。

## 2. 两个会话的分工（必须遵守）

- **本会话可以：**
  - 改 `paper/` 下的文件（sec、supp、preamble、refs.bib、fig、scripts/paper_tables.py、scripts/make_figures.py）；
  - 随时运行 `make data`，它只读 `runs/`。
- **本会话不要碰：**
  - `runs/`、队列（`scripts/queue.py`、`runs/jobs.jsonl`）、`scripts/make_jobs.py`；
  - GPU 和任何进程。服务器是多人共用的 Unix 账号 user01，**别人的进程绝对不能动**。
- **提交：**
  - 只 `git add` 自己改的文件，写明路径，不用 `git add -A`，也不用 `commit -a`。两个会话在同一个工作区提交。
  - 提交信息结尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
  - 不 push（用户："先不管 push 了"）。GitHub token 不写进任何文件、remote 或 git config。
- **EXPERIMENTS.md：** 论文层面的决定可以在末尾追加新的 §19.x 小节，写清时间和用户原话。编辑前先重新读文件末尾，只追加，不改别的小节，因为原会话也在往里写。
- **权限：** 被拒绝的操作不要绕过。

## 3. 用户的长期要求

- **回复：** 用中文，语气肯定，不防御。
- **论文写作：**
  - 不写防御性的话，不堆限定，不写单次运行的免责声明；
  - 用平实的词，不用大而空的抽象词；
  - 正文不用破折号，不用连字符自造复合词（通用术语和名字可以保留：model-based、first-person、two-compartment、resonate-and-fire、actor-critic、DH-SNN、TC-LIF、SD-Dreamer）；
  - 范围写成 "to"，不写 `--`。
- **固定术语：**
  - reset amnesia；
  - cue recall（原 memlen-L-recall）；
  - Spiking-WM A+B+G（宏 `\swmnr`，原 Spiking-WM-NR）；
  - persistent dendritic neuron（PDN）；
  - stateless soma；
  - spike count；
  - retention time；
  - write gain；
  - multicompartment。
- **定位：** 方法和 R2I 本质不同。不要写成"把 SSM 脉冲化"。
- **指标：** 用主流指标，不自造标准。BSuite 按 R2I 的约定：预算内任一次评估 ≥ 0.9 算学会（✓）。
- **模型配置：** 我们的模型在所有任务上只用一套配置，变体只作为附录消融。配置用原论文的，不在测试 benchmark 上调参。
- **额外种子：** 不提议额外种子。唯一例外是 R2I 重跑规则：每个模型在最短的没学会的长度上换种子重跑一次，学会了再跑下一个长度。
- **模型大小：** 和基线对齐时缩小我们的模型，不放大基线。
- **取舍：** 记忆任务优先。DMC cheetah、cartpole 和 cue recall L64 学得慢，用户已经接受。
- **范围：** 不考虑事件相机。GRSN 只作为基线。
- **不能做的：** 不能只给我们换一种口径，也不能隐藏计算方法。这一点之前讨论过，见 §19.52。

## 4. 最终模型（论文里唯一的配置）

- **运行名：**
  - RL：`PD2Lv2r64thermofbch2_tau1000`；
  - 探针：`PD2Lreshalfthermofbch2`；
  - Memory Maze：`PD2Lv2thermofbch2`。
- **结构：**
  - 两层 PDN，每层 N = 512，每个神经元 M = 2 个通道（一个积分、一个共振）；
  - 保留时间 1 到 1000 个环境步，写入增益归一化；
  - 无状态胞体按 thermometer 编码，T = 5，发出脉冲计数；
  - 输入电流 `LN(B x_t + R s̄_{t−1})`，带脉冲反馈，反馈只写入、不清零。
- **策略：** actor-critic 通过秩 64 的因子读第二层树突的 3072 个 ON/OFF 输入，共 4.1M 参数。世界模型 8.56M，其中核心 2.64M。
- **注意：** `runs/` 里不带 `thermofbch2` 的 PD 系列都是旧模型。

## 5. 论文现状

- 10 月 11 日做过一次全文去防御润色，摘要按 Spiking-WM 的结构重写过，见 §19.50。表格和方法部分已换成最终模型（§19.48、§19.49）。主图 `fig/overview_svg.py` 已重画；要导出，用 `python fig/overview_svg.py /home/user01/miniconda3/envs/seamdet/bin/rsvg-convert`。
- 11:08 用户说"先别改 Paper 了"，之后论文就停在提交 `7b1284d` 的状态。现在用户要在本会话里接着改。
- **页数：** 正文超出 8 页约 1/4 页，第 9 页左栏约 29 行，其中约 8 行是 TODO。
- **图：** `make_figures.py` 的图 1、3、4 还没有用最终模型和新图例重新生成，图里还是旧模型。
- **待用户决定的事：**
  1. **POPGym 指标。** 现在的提交是 MMER（max-mean episodic return，POPGym 论文自己的指标），所有模型同一口径，表注一句说明。另一个选项是最后 3 次评估平均，`git revert 7b1284d` 即可。用户说"最后再说"。两种口径下我们分别是：
     - MMER：Easy 0.72，Medium 0.48；
     - 最后 3 次平均：Easy 0.66，Medium 0.32。最终模型在 1.9M–1.95M 有一次暂时回落，见 §19.51。
  2. **TC-LIF 的定位。** 用户问"能不能不比他了"。我建议保留，理由和写法见第 7 节；用户还没回复，动笔前先确认。

## 6. 当前数字（2026-10-12 00:10；最新值看 `make data` 或 RESULTS.md）

**BSuite memory length（✓ = 预算内任一次评估 ≥ 0.9；括号里是首次达到的步数）**

| 模型 | L8 | L16 | L32 | L64 | R2I L51 | L71 | L101 |
|---|---|---|---|---|---|---|---|
| Spiking-WM | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| LIF | ✓ | ✗ | ✗ | ✗ | 在跑（最高 0.31） | — | — |
| GRSN | ✓ | ✓ | ✗ | ✗ | 在跑（最高 0.44） | — | — |
| DH-SNN | ✓ | ✓ | ✗ | ✗ | 在跑（最高 0.81） | — | — |
| TC-LIF | ✓ | ✓ | ✓ | ✓ (112k) | ✓ (102k) | 在跑 | 在跑（222k 时最高 0.88） |
| **本文** | ✓ | ✓ | ✓ | ✓ (82k) | ✓ (62k) | ✓ (182k) | ✓ (142k) |
| DreamerV3 | ✓ | ✓ | ✓ | ✗ | ✓ (902k) | ✗ | ✗ |
| A+B+G（附录） | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |

**POPGym RepeatPrevious（MMER，随机水平 −0.5；Easy 记住 4 步前，Medium 记住 32 步前）**

| 模型 | Easy | Medium |
|---|---|---|
| 本文 | 0.72 | 0.48 |
| LIF | 0.74 | −0.44 |
| Spiking-WM | 0.64 | −0.46 |
| DreamerV3 | 0.42 | −0.41 |
| GRSN | 0.37 | −0.47 |
| R2I（官方代码，2M 步） | −0.48 | −0.46 |
| DH-SNN / TC-LIF | 跑到 45–60%，都在随机水平：Easy −0.44 / −0.30，Medium −0.46 / −0.48 | |

**世界模型探针（final AUC）**

- **cue recall L64：** 本文 1.00，TC-LIF 1.00，DH-SNN 0.52，其余约 0.50。
- **精确延迟 Easy / Medium / Hard（k = 4 / 32 / 64）：**
  - 本文 0.98 / 0.92 / 0.94；
  - TC-LIF 0.86 / 0.50 / 0.57；
  - DH-SNN 0.63 / 0.50 / 0.50；
  - DreamerV3 0.95 / 0.50 / 0.50。
- Hard 没有用于任何设计选择。
- **消融：** 去掉共振通道后 Medium 探针 0.57；每步清空树突后全部降到 0.5。

**Memory Maze 9×9（objects MSE ↓ / walls 准确率 % ↑）**

- **本文：** 310k 次更新时 9.37 / 82.8，还在跑，要到 1M，约一周。
- **DreamerV3（1M）：** 9.8 / 85.3。
- **其他基线：** Spiking-WM 11.8 / 83.8，LIF 11.7 / 83.7，GRSN 11.5 / 83.3。
- **参照：** 无记忆探针约 12.0；常数预测 24.0 / 80.8。
- DH-SNN 和 TC-LIF 没有 Memory Maze 运行。

**DMC 本体感觉，500k 步（walker / cheetah / cartpole）**

| 模型 | walker | cheetah | cartpole |
|---|---|---|---|
| 本文 | 923 | 487 | 676 |
| actor 只读胞体 | 904 | 473 | 813 |
| LIF | 807 | 532 | 811 |
| GRSN | 863 | 392 | 652 |
| Spiking-WM | 923 | 343 | 497 |
| DreamerV3 | 953 | 699 | 871 |

**像素 walker，1M 步**

- 本文 405k 步时 842；同一区间 Spiking-WM 716。
- Spiking-WM 终值 917，DreamerV3 终值 968。
- LIF、GRSN、DH-SNN、TC-LIF 在 145k–225k，还要 3–4 天。

**能耗（BSuite L32 智能体，每个环境步；45 nm，AC 0.9 pJ，MAC 4.6 pJ，不计访存）**

- **本文：** 4.74 µJ，比 DreamerV3（22.02 µJ）少 **4.6 倍**。
- **脉冲基线：** GRSN 2.41，DH-SNN 2.49，LIF 2.75，TC-LIF 3.24，Spiking-WM 3.43 µJ。
- **我们在脉冲模型里最高，原因有两个：**
  - 发放率 0.18，基线约 0.10；
  - actor 读树突输入，多 0.85 µJ。
- 这一点照实写。`numbers.tex` 的能耗宏要等 `make data` 才会填上。

**其他诊断（可放附录）**

- **去掉脉冲反馈：** R2I L71 在 1M 步内没学会（最高 0.44），最终模型 182.5k 学会；探针里去掉反馈不影响记忆。
- **actor 只读胞体：**
  - R2I L71：202.5k 学会；
  - R2I L101：162.5k 学会，之后多次回落；
  - POPGym：Easy 0.58，Medium 0.22，都还在跑。

## 7. 必须跟着新结果改的地方

1. **§5.6 第 137 行和附录第 146 行。** 现在写的是"只读 (z,h) 的 actor 在 cheetah、cartpole 上更高（499、787 对 433、517），所以差距在策略输入"。按终值：
   - cheetah：本文 487，对照 473，这个说法不再成立；
   - cartpole：676 对 813，仍然成立。
   - 要改成照实的说法：读树突对 walker 和 cheetah 没有损失，只在 cartpole 上较低；我们在 cheetah 上低于 LIF（532）的原因不在策略输入。
2. **能耗段**（§5.6 第 140–143 行）：填上 4.6 倍。加一句：脉冲基线 2.4–3.4 µJ，我们因为发放率高、actor 读树突，能耗更高。
3. **DH-SNN：**
   - RL 止于 16 步：L32 最后 0.48，L64 0.06。第 63 行的 TODO 已可写。
   - 探针 cue recall 止于 32 步，第 86 行已写对。
   - R2I L51 还在跑（最高 0.81）。
4. **TC-LIF**（建议，需用户确认）：
   - **它的性质：** TC-LIF 两个隔室都没有泄漏，脉冲只减去固定量，状态本质上是一个持久的积分器，能记住"看到过什么"。正文已经把它当作支持诊断的证据（引言第 38 行，§5.2 第 61 行），保留这个写法。
   - **我们的优势写在它做不到的地方：** 记住确切的时间间隔。
     - POPGym：TC-LIF 连 Easy 都在随机水平，Medium 只有我们学会；
     - 精确延迟探针：k = 32 时 0.50 对 0.92，k = 64 时 0.57 对 0.94；
     - 消融：去掉共振通道，我们的 Medium 探针也掉到 0.57，所以差距来自共振通道。
   - **BSuite：** 两者都学会，我们更快（L64 82k 对 112k，L51 62k 对 102k）。如果 TC-LIF 也学会 L101，就写两者都到 101 步；主线对比 DreamerV3 止于 51 步，不受影响。
   - **能耗：** TC-LIF 比我们低，照实写。
   - **核对：** 摘要、引言、结论里所有 "only model" 的说法都要和 TC-LIF 对一遍。摘要第 3 行的 LaTeX 注释就是这个提醒。目前在 POPGym 和精确延迟上都成立。
5. **图 1 图注**（引言第 11 行）：写的是 "GRSN at 16"，DH-SNN 也止于 16，要补上。
6. **其余 TODO**（§5 四处、附录三处）：等对应运行结束再填。ETA：
   - R2I L51 基线：约 10 月 12 日 07:00；
   - TC-LIF L71、L101：约 10 月 12 日下午；
   - POPGym 的 DH-SNN、TC-LIF 和只读胞体对照：约 10 月 13 日；
   - 像素 walker：10 月 14–15 日；
   - Memory Maze 本文：约 10 月 19 日。
7. **最后：** 压回 8 页；用 `make data` 重新生成图 1、3、4；投稿前在 `preamble.tex` 关掉 `\pend` 和 `\draftnote`，并确认没有 `\TODO`。
