## Contract Paraphrase

D1 (methodology_rigor)：从方法严谨性的角度，这是我负责的核心维度：实验设计是否能回答所提问题，基线与对照是否公平（训练预算、配置、参数量、调参），评估协议是否标准，随机种子与变异性如何报告，消融是否覆盖每个被依赖的组件，复现所需细节是否给出。

D2 (domain_accuracy)：领域事实与前人工作的表述是否准确。该维度不由我打分，但若前人方法的配置被误用，会直接影响我对比较公平性的判断。

D3 (argumentative_coherence)：我与 Devil's Advocate 共同负责。我关注的是每个因果性结论是否被一个受控干预单独隔离，文字、表格与图中的数字是否一致，结论强度是否与证据强度相称。

D4 (cross_disciplinary_relevance)：跨领域可读性与跨学科主张的实据，由视角审稿人负责；与我相关的部分是能耗等跨领域主张背后的计量方法是否可检验。

D5 (writing_and_structure)：组织与表述由编辑负责；我只关注方法与协议细节是否写到可复现的程度。

D6 (venue_fit_and_contribution)：会议适配与贡献由编辑负责，不在我的打分范围。

## Scoring Plan

### D1: methodology_rigor
dimension_id: D1
what_to_look_for: 基线的来源与配置、环境步数与参数量与计算量是否对齐、种子数量与变异性报告、评估协议是否沿用标准设定、每个关键组件是否有消融、超参数与代码等复现信息
what_triggers_block: 标题性比较中各方法的训练预算、配置或容量不对等，或某个机制性主张没有受控干预加以隔离
what_triggers_warn: 结果只基于单个随机种子且没有任何变异性估计，或关键超参数与协议细节缺失
what_triggers_fatal: 所报告的数字不可能由所述协议产生，或评估过程把训练信息泄漏进测试条件

### D3: argumentative_coherence
dimension_id: D3
what_to_look_for: 从问题诊断到方法再到证据的推理链、每个结论是否由所给结果推出、正文与表格数字是否一致、尚未完成的结果是否被用来支撑结论
what_triggers_block: 某个中心结论被论文自己的表格反驳或缺乏支撑，或标题性结果在正文与表格中数字不一致
what_triggers_warn: 结论的表述强于证据所能支撑的程度，或依赖尚在进行中的临时结果
what_triggers_fatal: 中心论点在逻辑上被论文自身报告的证据所否定

criteria_binding_unavailable

[CONTRACT-ACKNOWLEDGED]
