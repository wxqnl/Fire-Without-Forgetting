# 论文（CVPR 2027 投稿稿）

CVPR 2027：注册 2026-11-10，投稿 2026-11-16。模板用 CVPR 官方 author kit（github.com/cvpr-org/author-kit）。2027 版发布前先用 2026 版，发布后替换 `cvpr.sty` 和 `ieeenat_fullname.bst`。

## 编译

```bash
cd paper
make data   # 从 ../runs 重新生成图（fig/*.pdf）和表（tables/*.tex）
make        # main.pdf：正文 + 参考文献，用于投稿
make full   # main_full.pdf：正文 + 参考文献 + 附录，交叉引用全部解析
```

- 投稿时附录单独交 PDF：从 `main_full.pdf` 拆出参考文献之后的页。
- `make` 只编译正文，此时正文中对附录的引用显示为固定字母（`\suppref`）。

## 文件

| 文件 | 内容 |
|---|---|
| `main.tex`、`preamble.tex` | 主文件、宏包与宏 |
| `sec/0_abstract.tex` … `sec/6_conclusion.tex` | 正文各节 |
| `supp/supp.tex` | 附录 A–H |
| `fig/neurons.tex` | 图 2：神经元示意图（TikZ） |
| `scripts/make_figures.py` | 图 1、3、4，读 `runs/` |
| `scripts/paper_tables.py` | 表 2–6 和附录表 7，读 `runs/`，与 `scripts/main_tables.py` 用同一套运行映射 |
| `tables/interventions.tex` | 表 1：Phase 0 干预结果，两个种子（`PHASE0.md` §4.1），手写 |
| `refs.bib` | 参考文献 |

## 标记

- `\pend{…}`（蓝色）：来自仍在运行的任务。`make data` 后自动更新；任务跑完后蓝色自动消失。
- `\TODO{…}`（红色）：待补的文字，大多等结果出来后再写。
- 投稿前在 `preamble.tex` 取消注释两行 `\renewcommand`，关掉这两种标记；确认正文中不再有 `\pend` 和 `\TODO`。

## 参考文献核对（2026-10-03）

每一条都逐条核对过：
- 有 DOI 的，查 Crossref，核对标题、作者、卷期、页码；
- arXiv 论文，按 arXiv 号查 arXiv API，核对标题和作者；
- 会议名称：已知会议的按 arXiv 注释或会议页面确认；Memory Maze、Wijmans et al.、POPGym 用网页搜索确认为 ICLR 2023。

核对中发现并改正了两处页码（PLIF、Habitat）。新增引用时按同样方式核对。

## 当前状态

见 `EXPERIMENTS.md` §19。
