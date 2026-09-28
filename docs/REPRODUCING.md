# 环境、运行记录与复现

## 运行位置与权限

现有实验实际运行在42号机。任何项目操作先SSH，再进入 /data/Minko：

~~~bash
ssh New-H100-2
cd /data/Minko
python3 fire-without-forgetting/scripts/status_v2.py
~~~

只使用已获授权且可用的GPU0–3，不使用41号机，不操作其他项目。所有条件训练seed0，评估随机种子与训练分开；评估多回合不等于多个训练seed。

## 环境记录

envs/仅收录安装输入及发布时重新读取的完整包版本，不收录虚拟环境本体。

|环境|Python|核心运行栈|文件|
|---|---|---|---|
|torch|3.10|PyTorch2.4.1+cu121|torch-requirements.txt、torch-freeze.txt|
|r2i|3.10|JAX0.4.23、CUDA12/cuDNN8|r2i-requirements.txt、r2i-freeze.txt|
|dreamerv3|3.10|JAX0.4.33、CUDA12/cuDNN9|dreamerv3-requirements.txt、dreamerv3-freeze.txt|

*-requirements.txt是安装输入；*-freeze.txt是实际版本记录，尤其用于收紧Dreamer安装输入中的版本区间。实际包版本优先看freeze。基线运行采用独立环境，避免两套JAX/CUDA依赖相互覆盖。

scripts/setup_*.sh是42号机历史安装脚本，内含原部署绝对路径；它们不是通用安装器。现有训练使用的环境已经准备好，复现时新建独立环境，不覆盖正在使用的环境。

## 取得发布代码

在42号机 /data/Minko 下选择一个新的目录：

~~~bash
git clone --branch codex --single-branch \
  https://github.com/wxqnl/Fire-Without-Forgetting.git \
  fire-without-forgetting-review
~~~

该检出目录用于查看冻结代码和结果。原调度器仍指向 /data/Minko/fire-without-forgetting，不会因为克隆新目录而自动迁移。

如需在新目录运行，使用该目录内的代码/新虚拟环境，并把运行记录中原工程前缀替换成新前缀；为输出分配新的run目录。历史队列中的绝对路径、CPU affinity、CUDA库路径均是原机器配置，不能不经调整直接启动。不会把已有运行的检查点静默接到新实验名下。

## 精确配置与命令在哪里

每项结果位于 results/runs/<组>/<名称>/：

- command.json / job.json / result.json：真实命令、任务、预算、来源快照、设备和退出状态；可用文件取决于运行版本。
- resolved_config.json 或 config.yaml：当时解析的完整配置。
- evaluations.jsonl：全部已记录评估点，包含计步、平均回报与回合数。
- training_scalars.jsonl：包含损失/梯度的训练标量记录。
- final_checkpoint.json：运行端保存完成记录；独立读回验证见docs对应completed文件。
- exclusion.json / failure.log：无效运行和失败依据。

JAX任务还需要job.json内的CUDA库/XLA环境变量，Dreamer需保留logger.timer=True。CPU/GPU亲和性按可用资源选择，但不得扩大授权GPU范围。发布整理没有重新训练模型。

## 预算和比较口径

- 早期MCN/本方法按实际环境动作计步。
- 早期R2I和Dreamer POPGym按Driver transition计步，包含episode首帧。不能直接把计数器值当成完全相同的动作预算。
- 本轮BSuite三后端统一1M实际训练动作，reset首帧不计入；R2I/Dreamer独立action counter与原driver counter都保存。
- 回报只在同一任务、同一奖励口径内比较。POPGym RepeatPrevious随机期望-0.5；BSuite memory_len随机期望0。
- 各框架仍有模型、精度、回放、预训练更新和评估动作方式差异。不能把当前跨框架表当成只有神经元类型不同的严格消融。
- 每次评估通常20回合，只表示该训练seed下的评估波动。

## 数据、模型与续跑

BSuite/POPGym使用官方环境在线生成交互，模型从头训练。Memory Maze9×9离线数据尚待用户提供；真实环境可渲染不表示已完成离线视觉实验。

权重、优化器、replay、训练轨迹和TensorBoard文件留在原服务器runs/，不在Git中。当前SSM保存完整优化器/RNG；原MCN只保存权重。早期R2I的两个1M运行最终评估与实际保存步数不一致，详见结果解读。任何模型均不承诺未保存环境状态下的逐步精确恢复。

## 更新结果快照

导出器只读取运行文件，不启动训练、不加载GPU模型。为每次导出使用新目录：

~~~bash
ssh New-H100-2
cd /data/Minko
python3 fire-without-forgetting/scripts/export_results.py \
  --source /data/Minko/fire-without-forgetting \
  --destination /data/Minko/fwf-results-next
~~~

当前仓库快照的采集时间记录在results/summary.json。后续训练继续推进，不会自动改变GitHub内容。不要把旧快照的running理解为永久在线状态。
