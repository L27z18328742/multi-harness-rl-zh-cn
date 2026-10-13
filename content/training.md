## 跨 harness 训练小模型

我们用上一章的系统训练 [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B)：一次仅在 OpenCode 中训练，另一次跨四种 harness 训练，再将两次实验与基于更大模型 rollout 的监督微调比较。最早的实验使用 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，这些实验影响了 LFM 的训练设置，放在本章后面的折叠部分中。

### 设置

- **任务。** 1,000 个 [SmolDataEnvs](https://huggingface.co/collections/FineEnvs/smoldataenvs) 训练任务，其中 400 个中等、600 个困难；两次实验以相同的固定顺序使用它们。测试集包含 250 个留出任务，notebook、问题和指令都没有重叠。
- **两次实验。** 一次仅在 OpenCode 中训练。另一次为多 harness 训练，包括 [OpenCode](https://opencode.ai)、[Claude Code](https://code.claude.com/docs/en/overview)、[Codex](https://github.com/openai/codex)、[Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent)；每个包含八次 rollout 的 GRPO [@shao2024deepseekmath] 组使用四者之一。
- **评估。** 两次实验采用相同评估方式，包括 OpenCode-only 实验从未训练过的另外三种 harness。每 100 步，在四种 harness 下将 250 个测试任务各运行一次，总计 1,000 个评测单元。一个单元的首次已评分尝试正确时，就算解出。
- **训练方案。** 在 TRL 中使用 [Async GRPO](https://huggingface.co/docs/trl/async_grpo_trainer) 训练 1,000 步。异步意味着生成与训练分开运行：一张 H100 训练，另一张用 vLLM [@kwon2023vllm] 服务策略并持续产生 rollout，因此训练无需等待最慢的智能体。代价是 rollout 通常来自落后两个优化器步骤的权重，最多落后四步。
- **硬件与时间。** 每次实验使用 Hugging Face 集群上的两张 H100；每次 rollout 使用一个 [E2B](https://e2b.dev) 沙箱，配有 1 CPU 和 4 GB 内存。OpenCode-only 训练约用 32 小时，多 harness 约用 46 小时，不包含评估、排队和重启。我们没有足够细致地记录总成本，因此不给出成本数字。

配套教程提供了易读的[多 harness](https://github.com/adithya-s-k/FineEnvs/blob/main/05-multi-harness-rl/train/multi_harness.py)和[原生 OpenCode](https://github.com/adithya-s-k/FineEnvs/blob/main/05-multi-harness-rl/train/opencode.py)训练脚本，以及 [HF Jobs 与 Slurm 说明](https://github.com/adithya-s-k/FineEnvs/blob/main/05-multi-harness-rl/REPRODUCE.md)。报告中的两次 LFM 实验都使用 Harbor 和 E2B 沙箱。当前教程使用 Daytona，并将原生 OpenCode 作为单独比较，因此并不是对历史实验的精确重放。

每个评测单元只运行一次，因此相邻 checkpoint 间一两个百分点的差距属于噪声。应更关注跨 checkpoint 的趋势，而非某个单点。

### 基础模型的起点

<Embed n="0" title="训练前 LFM2.5-2.6B 在测试集上的表现" desc="按 harness 展示训练前的 pass@1。" />

基线评估已经体现出前文的问题：相同权重在 Mini-SWE-Agent 中解出 62% 的任务，在 Claude Code 中却只有 33%。根据发布说明，LFM2.5-2.6B 没有在我们的四种 harness 中训练过；Liquid AI 列出的[训练 harness](https://www.liquid.ai/blog/lfm2-5-2-6b)是 Hermes Agent 和 OpenClaw。

### 我们奖励什么

我们给 LFM 两种奖励：

- **正确性：** 答案正确得 1，错误得 0。
- **工具效率：** 对调用次数更少的正确答案，给予最多 0.1 的小额加分。错误答案依旧得 0，因此模型不能靠提前放弃来获得加分。

这个加分项源于 Qwen 实验。那些实验只奖励正确性，模型得到答案后没有理由停止探索；一次早期 Qwen 实验中，每个 rollout 的工具调用数从 13 慢慢涨到 41。

<Embed n="1" title="一个 rollout 的奖励" desc="仅奖励正确性，与 LFM 使用的“正确性加工具调用奖励”对比。" />

加分最多只有 0.1，但 GRPO 从组内差异学习，因此小额加分也有作用。GRPO 对同一个任务运行八次 rollout，根据各次结果与组内平均值的比较进行训练。如果八次全都正确，单纯的正确性奖励让它们完全相同，这个组就无法提供学习信号。此时，加分是区分它们的唯一信息，会偏向更短的解题过程。

下图选取多 harness 实验中的两个真实组，分别用两种奖励打分。

<Embed n="2" title="同一个 GRPO 组，用两种方式评分" desc="同一任务的八次真实 rollout，以及在两种奖励下各自的 reward 和 advantage。" />

在 555 个 OpenCode-only 组中，有 97 个（17.5%）；在 686 个多 harness 组中，有 155 个（22.6%）：所有可评分 rollout 都正确，但工具调用数不同。效率加分为这些组提供了唯一的奖励差异。这些[审计计数](/data/reward-group-audit.json)覆盖了实际进入保留训练更新的组，不包括恢复运行时丢弃的工作。工具调用数来自 harness 自身的轨迹。奖励审计没有发现公式不一致，也没有发现对错误答案发放加分。

如下方曲线所示，两次 LFM 实验中，每个 rollout 的工具调用数都在训练期间下降，并且每种 harness 都如此。我们没有运行不带加分项的 LFM 实验，因此无法说这一现象单独由加分造成。

### 训练曲线

<Embed n="3" title="LFM2.5-2.6B 训练曲线" desc="每个优化器步骤的奖励、正确答案和工具调用数。拖动滑块可调整平滑程度。" />

两次 LFM 实验中，正确 rollout 比例在前几百步上升。此后，它会随各段任务的难度上下波动；由于任务顺序一致，两次实验面对的难度变化也一致。工具调用数持续下降。在测试集上，OpenCode-only 模型在 OpenCode 中优于多 harness 模型（58% 对 50%）；多 harness 模型则在 Claude Code（49% 对 42%）和 Codex（54% 对 43%）中更好。

训练与测试中的工具调用曲线，看起来方向相反。训练时，多 harness 实验每个 rollout 的调用数更多，因为平均值包括 Claude Code、Codex、Mini-SWE-Agent；即使训练之前，它们也比 OpenCode 使用更多调用。测试时，两个模型都在全部四种 harness 下运行，而多 harness 模型在每一种中都使用更少调用。OpenCode-only 模型学会在 OpenCode 中减少调用，这种能力只有部分迁移到其他 harness，主要是 Codex。

### 准确率

<Embed n="4" title="LFM2.5-2.6B 训练过程中的 pass@1" desc="每个已评估 checkpoint 的总体 pass@1，以及各 harness 下的 pass@1。" />

两次实验的准确率都上升：OpenCode-only 提升 10 个百分点，多 harness 提升 12 个百分点；主要提升发生在前 200 步，并保持到第 1,000 步。最终，OpenCode-only 为 52.3%，多 harness 为 54.2%。相对基线的两项提升都明显超出噪声范围，但两次实验相差的 1.9 个百分点处于噪声范围之内，因此总体准确率无法区分两者。

<Embed n="5" title="第 1,000 步时 LFM2.5-2.6B 在各 harness 下的表现" desc="逐 harness 对比 pass@1 与基础模型。" />

分 harness 的结果最清楚。OpenCode-only 模型在它训练过的 OpenCode 中最好；多 harness 模型在 Claude Code、Codex 中更好；在 Mini-SWE-Agent 下两者持平。OpenCode-only 的提升大部分发生在 OpenCode（34 → 58%），而多 harness 模型在四者中都有提升。

### 工具调用与 token

<Embed n="6" title="每个 LFM checkpoint 完成任务的工作量" desc="每个 checkpoint 在测试任务上的工具调用与 token 用量，包括总体和各 harness。" />

两个模型都用更少的工具调用解出更多任务，多 harness 模型减少得更多：在它与基础模型都能解出的任务上，调用数比基础模型少 31%；OpenCode-only 则少 11%。更少调用也意味着每次调用重发的历史更少，因此输入 token 也下降。最大节省出现在 Codex，多 harness 模型只使用约一半的调用。明显的例外是 OpenCode-only 在未经训练的 Claude Code 中：调用数和 token 都比基础模型更多。

<Embed n="7" title="各 checkpoint、各 harness 相对基础模型的节省" desc="在两个模型都能解出的测试任务上，工具调用、生成 token、输入 token 各减少了多少百分比。悬停可查看任务数。" />

多 harness 模型在 Codex 下的节省，几乎在每个 checkpoint 都增长；到第 1,000 步时，所有 harness 在三项指标上都呈蓝色。OpenCode-only 在 OpenCode 中稳定节省，在 Codex 中也越来越多。但在 Claude Code 中，它几乎在每个 checkpoint 都比基础模型生成更多 token，第 500 步时最多增加 62%。

### 每次实验看到了多少数据

两次实验都进行了 1,000 个优化器步骤，但看到的数据量不同。

| | 仅 OpenCode | 多 harness |
|---|---:|---:|
| 不同训练任务数 | 555 | 626 |
| 参与监督的 token | 12.5M | 18.7M |
| 训练处理的 token | 162M | 451M |

Claude Code 在运行过程中会改写自己的历史，因此每个 rollout 会变成约八条训练记录，每条都带有自己的上下文副本；其他 harness 平均约为一条。每次实验只有一个随机种子，且数据暴露量不同，因此两次实验之间的比较属于观察性比较。

### SFT 与 RL

RL 并非从 harness 学习的唯一方式。为 RL 保留精确 token 的采集过程，也记录完整轨迹，因此更大模型的成功执行可以用于监督微调（SFT）。我们在 LFM2.5-2.6B 上尝试这种方法，以比较模仿学习与上述 RL 实验各能走多远。

#### 实验

我们让 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 作为教师，使用同样的 OpenEnv 和 Harbor 系统，在四种 harness 中执行 SmolDataEnvs 训练任务；沙箱放在 [Daytona](https://www.daytona.io) 上。每个任务、每种 harness 最多尝试三次，保留验证器判为正确的第一次。这得到来自 888 个任务的 3,189 次成功 rollout，其中 377 个任务中等、511 个困难。

rollout 中每个 assistant 轮次都成为一个训练样本。截至这一轮的对话，包括工具输出，作为上下文；损失只作用于下一条 assistant 响应。样本使用 LFM 自己的聊天模板分词，而非教师模板。

我们从与 RL 相同的 LFM2.5-2.6B 基础模型出发，训练两个 SFT 模型：

- **OpenCode SFT：** 使用 801 次 OpenCode rollout，共 4,825 个样本。
- **多 harness SFT：** 使用全部 3,189 次 rollout，共 17,929 个样本；混合打乱，不在 harness 之间做平衡。

两者都使用 TRL 的 SFT trainer 全参数微调两个 epoch，学习率为 3e-6，有效批量大小为 8。每个 epoch 都在与 RL 相同的 1,000 个测试单元上评估。

数据已公开为 [SmolDataEnvs-multiharness-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft)：每种 harness 一个 config，`all` config 包含四者，还有预先为 LFM2.5-2.6B、Qwen3.5-2B 分词的 config，以及[训练脚本](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft/blob/main/train_sft.py)。教师原始 rollout 位于 [qwen38-27b-harbor-rollouts](https://huggingface.co/datasets/FineEnvs/qwen38-27b-harbor-rollouts)。

两种方法从不同内容学习：

| | SFT | RL |
|---|---|---|
| 学习来源 | 教师的成功 rollout | 自己的 rollout |
| 训练信号 | 教师的下一条响应 | 正确性加工具调用奖励 |
| 是否奖励更少工具调用 | 否 | 是 |
| 训练任务数（OpenCode / 多 harness） | 801 / 888 | 555 / 626 |
| 全程参与监督的 token | 1.2M / 8.2M | 12.5M / 18.7M |
| 训练时间 | 1.5 小时 / 6.5 小时 | 32 小时 / 46 小时 |

SFT 时间只计算训练器自身的循环，不包括使用 27B 教师模型在四种 harness 内采集 rollout 的时间。

#### 结果

<Embed n="8" title="LFM2.5-2.6B 上的 SFT 与 RL" desc="总体 pass@1 与相对基础模型减少的工具调用数。" />

对于 RL，我们使用各次实验在测试集上的最佳 checkpoint：OpenCode-only 为第 900 步，多 harness 为第 700 步。在测试集上选择会稍微美化 RL，但两者与各自第 1,000 步分数的差距都不到半个百分点。

**RL 最终领先于两次 SFT 实验。** 多 harness RL 达到 54.6%。较好的 SFT 模型 OpenCode SFT 达到 47.5%，多 harness SFT 为 43.1%，比在相同四种 harness 中训练的 RL 模型低 11.5 个百分点。

**OpenCode SFT 像是提升幅度较小的 OpenCode RL。** 几乎所有提升都在 OpenCode 中，从 33.6% 到 51.6%，接近 OpenCode RL 的 56.0%。在 Claude Code 中，它停留在基础模型的 33.2%；在 Mini-SWE-Agent 中略降，从 62.1% 到 58.8%。无论模仿还是 RL，只在一种 harness 中训练，主要帮助的都是那一种。

**多 harness SFT 分散了提升，却又在 Mini-SWE-Agent 中失去这些收益。** 它在 Claude Code（33.2 → 42.4%）、Codex（40.0 → 46.8%）、OpenCode（33.6 → 38.0%）中提升；在 Mini-SWE-Agent 中却从 62.1% 降至 45.2%，抵消其他收益。因此总体为 43.1%，与基础模型 42.2% 的差距处于噪声范围内。第一个 epoch 后，它曾降至基础模型之下的 38.3%，OpenCode 下降到 22.4%。多 harness RL 则在包括 Mini-SWE-Agent 在内的四者中都有提升。我们尚未弄清 SFT 为何损害 Mini-SWE-Agent 表现。

<Embed n="9" title="按 harness 比较 LFM2.5-2.6B 上的 SFT 与 RL" desc="每种 harness 下的 pass@1 或工具调用减少量。可切换 SFT 的第一个和第二个 epoch。" />

**SFT 即使没有效率奖励，也减少工具调用。** SFT 没有效率项，但在它与基础模型都解出的任务上，多 harness SFT 的工具调用减少 24.2%，且每种 harness 都减少。OpenCode SFT 总体减少 8.5%，但在 Claude Code 中多 10.5%，在 Mini-SWE-Agent 中多 4.8%。RL 也呈现同样模式：跨四种 harness 训练，在所有 harness 中减少调用；多 harness RL 第 700 步总体减少 26.7%，到第 1,000 步减少 31.1%；仅 OpenCode 训练的减少则不均匀。

**这项比较无法告诉我们什么。** 两次 SFT 实验的差异不止在 harness 混合方式。多 harness 实验监督 token 约为七倍，步骤接近四倍。SFT 与 RL 的目标、数据来源、任务池和计算量也不同，每个设置只运行一次。这些是观察结果，不是方法排名。接下来我们希望比较：先 SFT 再 RL 与从基础模型直接 RL；在教师四种 harness 都能解出的 692 个任务上比较 harness 混合方式；以及各 harness 监督 token 相同的 SFT 混合方式。

<details>
<summary>Qwen3.5-2B 实验，以及它们带来的经验</summary>

#### 实验设置

Qwen 实验更早进行，只奖励正确性，任务池也更容易：1,000 个任务包含 150 个简单、600 个中等、250 个困难。在相同测试集上，它的起点只有 LFM 的约三分之一；Qwen3.5 模型卡没有说明训练 harness。

<Embed n="10" title="两个基础模型在测试集上的表现" desc="按 harness 展示训练前的 pass@1。" />

共有三次 Qwen 实验，均在相同的 1,000 个测试单元上评估：

- **Harbor 多 harness：** 在 GRPO 组之间轮换四种 harness。
- **Harbor OpenCode-only：** 相同流水线，但只用一种 harness。
- **独立 OpenCode：** OpenCode 在 Harbor 之外的独立环境中运行，使用 [Daytona](https://www.daytona.io)。

<Embed n="11" title="Qwen3.5-2B 训练过程中的 pass@1" desc="三次仅奖励正确性的 Qwen 实验。上图为全部 1,000 个测试单元，下图按 harness 展示；独立 OpenCode 实验有单独测得的 15.9% 基线。" />

<Embed n="12" title="Qwen3.5-2B 训练曲线" desc="每个优化器步骤的奖励与工具调用数，可调整平滑程度。Qwen 的奖励就是正确性。" />

#### 先上升，后下降

三次实验开始都提升，说明只要拥有精确 token，就能通过未经修改的 harness 训练。多 harness 从 14.6% 提升至第 500 步的 37.0%，超过两倍；OpenCode-only 在第 700 步达到 39.5% 峰值。随后，两者都在第 1,000 步回落至约 26%。独立实验上升较慢，最终达到其最佳值 29.8%。

独立实验只在 OpenCode 中训练，但在 Claude Code、Codex、Mini-SWE-Agent 中的提升，约为在 OpenCode 中的两倍。

#### 为什么两次 Harbor 实验下降

两次下降在轨迹中呈现出不同特点。

- **多 harness 耗尽了输出预算。** 训练后期，答案越来越长，直到撞上评估使用的 4,096-token 输出限制。1,000 个单元中被这一限制截断的数量，从 9 增至 556；模型常常写出长篇解释，而本应进行计算。训练允许最多 16,384 个响应 token，因此模型学会了评估时会被截断的习惯。在 OpenCode 中，每任务调用数从 16 暴跌至 3。
- **OpenCode-only 持续工作，却不完成。** 每任务调用数从 17 增至 21；在跟踪提交行为的单元中，提交答案的比例从 69% 降至 41%。
- **恢复运行的 bug 重放了旧任务。** 多 harness 实验在第 684 步重启后，几乎所有新 rollout 都来自已经见过的任务。这改变了后期训练，但下降从重启之前的第 500 步已经开始，所以这个 bug 无法解释下降的起点。

#### 没有加分时的工具调用

只奖励正确性时，Qwen OpenCode-only 实验每个 rollout 的工具调用，在训练期间从约 10 翻倍至 20；Qwen 多 harness 保持在 11 到 16。两次带加分项的 LFM 实验中，调用数则下降。

<Embed n="13" title="训练期间 Qwen 与 LFM 的工具调用" desc="两次仅奖励正确性的 Qwen 实验，与两次带效率加分的 LFM 实验，逐优化器步骤对比。" />

#### Qwen 实验还揭示了什么

- **很多步骤没有学到任何东西。** 35% 到 58% 的优化器步骤里，所有组都没有奖励差异，因为八次 rollout 要么全对，要么全错。
- **Claude Code 主导训练数据。** 每种 harness 约占四分之一的 rollout，但 Claude Code 产生了 77% 的训练记录，以及 35% 的监督 token。
- **SETA：** 在普通 bash 环境中采用同步训练、使用独立评估器的实验，在 150 步内从 18.8% 提升至 38.0%，随后被我们主动停止。
- **更难的数据没能挽救下降。** 从多 harness 第 500 步 checkpoint 出发，使用 500 个新困难任务，通过 [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) 在 H200 上继续训练，没能恢复峰值。近三分之二的步骤没有奖励差异；即使把所有未评分单元都算作正确，最终分数也达不到起点的 37.0%。

#### LFM 做了哪些改变

LFM 设置改变了三点。奖励加入工具调用加分，因此全对的组依然携带信号，漫长、未完成的循环会付出代价。训练池改用中等和困难任务。训练和评估的输出上限都设为 4,096 token，因此模型无法学会评估时会被截断的答案。

</details>

### 复现

- [SmolDataEnvs 多 harness RL 合集](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212)：将下方资源及基础模型集中在一处。
- 已训练模型：第 1,000 步的 LFM [多 harness RL](https://huggingface.co/FineEnvs/LFM2.5-2.6B-multiharness-RL) 和 [OpenCode-only RL](https://huggingface.co/FineEnvs/LFM2.5-2.6B-opencode-RL)，最佳 checkpoint（第 700 和 900 步）分别在 `step-700` 和 `step-900` 分支；两个 epoch 后的 LFM [多 harness SFT](https://huggingface.co/FineEnvs/LFM2.5-2.6B-multiharness-SFT) 和 [OpenCode SFT](https://huggingface.co/FineEnvs/LFM2.5-2.6B-opencode-SFT)；以及最佳 Qwen checkpoint：[多 harness](https://huggingface.co/FineEnvs/Qwen3.5-2B-multiharness-RL)（第 500 步）、[OpenCode-only](https://huggingface.co/FineEnvs/Qwen3.5-2B-opencode-RL)（第 700 步）、[独立 OpenCode](https://huggingface.co/FineEnvs/Qwen3.5-2B-opencode-standalone-RL)（第 1,000 步）。
- [训练比较仪表盘](https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio)：全部训练曲线与 checkpoint 评估。
- [Harbor 环境](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-harbor)和[独立 OpenCode 环境](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-opencode)：当前配套环境服务器。独立的 `opencode_env` 自 OpenEnv 0.7.0 起已弃用，只为历史比较保留；新实验应使用 Harbor 并设置 `harness="opencode"`。
- [SmolDataEnvs-multiharness-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft)：SFT 数据，按 harness 分组，并提供为 LFM2.5-2.6B、Qwen3.5-2B 预先分词的版本及训练脚本。
- 已合并集成：[OpenEnv #1036](https://github.com/huggingface/OpenEnv/pull/1036) 提供采集代理和 Harbor；[OpenEnv #1280](https://github.com/huggingface/OpenEnv/pull/1280) 提供有类型的 `TrainingTrace` API；[TRL #6947](https://github.com/huggingface/trl/pull/6947) 提供 worker。TRL 的 [Harbor 示例](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor)展示完整训练器接线；worker 正式发布前，需要从 main 安装 TRL。

<aside class="note"><strong>更大规模的实验即将到来</strong>

本文每次实验都使用 20 亿至 26 亿参数的小模型，以单个随机种子训练 1,000 步。我们正在同一技术栈上准备更大模型、更大规模的实验，结果出来后会在这里补充。敬请关注。

</aside>
