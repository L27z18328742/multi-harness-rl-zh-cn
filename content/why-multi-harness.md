## 为什么要跨 harness 训练

### 基准分数开始附带 harness 信息

模型卡开始说明分数来自哪种 harness。GLM-4.7 声称，在 Claude Code、Kilo Code、Cline、Roo Code 等主流智能体框架中执行复杂任务时，性能显著提升 [@zai2025glm47]。Kimi K2 报告了两组 Terminal-Bench [@merrill2026terminalbench] 得分：在 Terminus 下为 25.0，在 Moonshot 自有框架下为 30.0 [@kimi2025k2]。MiniMax M2 几乎为列出的每个基准都注明了 harness [@minimax2025m2card]。DeepSeek-V3.2 的思考模式甚至无法在 Terminus 下运行，因此它的 Terminal-Bench 分数只能来自另一种 harness [@deepseek2025v32]。

harness 甚至可以改变模型的领先顺序。@zhang2026stopcomparing 选取了公开编码排行榜上彼此相差不超过三分的三个模型：GLM-5.1、GPT-5.4、Kimi K2.6。在其余条件固定的情况下，研究者让它们在三种 harness 配置下执行相同的 100 个 SWE-bench Verified [@jimenez2024swebench] 任务。三种配置逐步叠加：**Minimal（最简）**使用冗长的工具，不做上下文压缩，也不重试；**Improved（改进）**精简工具，并加入压缩和重试；**Full（完整）**再加入自检、偏移检查和回滚。

<Embed n="0" title="相同模型、相同任务，三种 harness 配置" desc="三种模型分别在不同的 harness 配置下取得第一。" />

每个模型都在一种不同的 harness 下胜出。更换 harness 让 GLM-5.1 的分数变化了 13 个百分点；在相同 harness 内更换模型，分数只变化 2.5 到 5 个百分点。[Joel Niklaus 在 SWE-bench Pro 的十种 harness 上也发现了同样的现象](https://x.com/joelniklaus/status/2085725862142623875)。

在受控实验之外也能看到这一点。同样的 Claude Opus 4.5 权重，在 Scale 的 [SEAL 排行榜](https://labs.scale.com/leaderboard/swe_bench_pro)上取得 45.9% 的 SWE-bench Pro [@deng2025swebenchpro] 得分，而在 Claude Code 内取得 55.4% [@zhang2026stopcomparing]。

### 当模型离开训练时使用的 harness

模型可能在训练时使用的 harness 内表现良好，换到另一种就遇到困难。Orchard 论文测量了这种跨 harness 迁移 [@peng2026orchard]。它把两个在一到两种 harness 中训练的开放编码模型，放到三种 harness 下运行，其中包括所有被比较系统在训练时都未用过的 Kimi-CLI。Orchard 自己的模型也参与比较：它先使用 OpenHands 和 Mini-SWE-Agent 的轨迹微调，再在 Mini-SWE-Agent 中进行 RL 训练。

<Embed n="1" title="只在一到两种 harness 中训练的代价" desc="三个系统分别在三种 harness 下的任务解决率。" />

将 OpenSWE-32B 从论文视为其原生 harness 的 [OpenHands](https://github.com/OpenHands/OpenHands)，迁移到 [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent)，分数下降 7.5 个百分点。放到 [Kimi-CLI](https://github.com/MoonshotAI/kimi-cli) 下，SWE-bench Verified 分数下降 58.8 个百分点至 3.6%，Terminal-Bench 2.0 得分为零。Scale-SWE 在原生 harness 之外无法生成有效工具调用。Orchard 的模型则在三种 harness 下，都能保持 45.0 至 64.3 的 SWE-bench Verified 得分。

论文将这些情况分别称为**任务解决率下降**——模型仍能工作，但解出的任务变少；以及**灾难性的格式失败**——模型输出完全无法使用。图中的每一行都是固定的同一个模型，而每个基准在所有 harness 下都使用相同任务，因此下降来自 harness。

### 为什么模型会对单一 harness 过拟合

KwaiKAT 团队指出了原因 [@kwaikat2026katcoder]：

> 在 agentic RL 中，如果训练只依赖一个固定的 harness，模型学到的往往不是“如何解决任务”，而是“如何按照这个特定 harness 的接口约定解决任务”。

他们将其分为三种过拟合：对 harness 的动作**格式**、**上下文结构**以及**控制流**的过拟合。控制流包括模型学会依赖的重试方式和停止条件。

<Embed n="2" title="模型对 harness 过拟合的三种方式" desc="模型依赖了什么；当这些条件改变时，哪里会失效。" />

harness 恰好在这些维度上存在差异。它们使用不同的 **API 方言**，即与模型服务器通信时采用的请求和响应格式：OpenAI chat-completions、OpenAI Responses、Anthropic Messages、Google Gemini。它们也在是否使用工具调用、如何压缩上下文，以及替模型接管多少控制流方面存在差异。

<Embed n="3" title="harness 生态概览" desc="每种 harness 使用哪些模型 API。悬停可查看开发者及其特点。" />

Mini-SWE-Agent 的 [README](https://github.com/SWE-agent/mini-swe-agent) 将避免 scaffold 过拟合列为采用简单设计的原因之一。[Aider](https://aider.chat/) 使用另一种接口：它要求模型在普通文本中编写[编辑块](https://aider.chat/docs/more/edit-formats.html)，而非结构化工具调用。受过工具调用训练的模型，也必须学习这种输出格式。

实践中，这类失败往往直接表现为报错，而非分数下降。模型会使用训练时 harness 的工具名称和参数名称发起调用；如果新 harness 的命名不同，调用还没真正执行就会被拒绝。

<Embed n="4" title="harness 过拟合的实际表现" desc="同一个工具调用，在训练时的 harness 中被接受，在未见过的 harness 中被拒绝。" />

harness 的数量还在增长。[Harbor 0.22.0](https://github.com/harbor-framework/harbor/releases/tag/v0.22.0) 提供了四十多种 harness 的适配器，[OpenEnv](https://github.com/huggingface/OpenEnv) 已对十种完成端到端验证。因此，只在少数 harness 中训练的模型，会不断遇到自己未见过的 harness。

### 前沿实验室开始有意识地跨 harness 训练

KwaiKAT 将其应对办法称为 harness scaling：训练时改变 harness。Poolside 的 Laguna 报告采用了这种做法 [@poolside2026laguna]：

> 为了促进对多种 agent harness 的泛化，我们的训练数据包含来自 OpenHands、OpenCode、Mini-SWE-Agent 等外部框架的轨迹。

他们在监督训练中加入了 13 亿 token 的这类轨迹，并有意保留各 harness 自身的习惯。随后，强化学习只在生产环境交付的 `pool` harness 内进行；报告没有说明为何其他 harness 不继续参与。

Kimi K3 的 RL 环境通过可组合模块构建 Claude Code、[Codex](https://github.com/openai/codex) 等 harness，让模型能在多种配置中训练，而不是只训练一种 [@kimi2026k3]。Qwen3-Coder-Next 在六种 harness 内生成智能体编码数据 [@qwen2026codernext]；MiMo-V2.6 则在一组适配不同任务的 mini-harness 中训练 [@xiaomi2026mimov26]。Liquid AI 在小得多的模型规模上也这样做。对于本文后面训练的 [LFM2.5-2.6B](https://www.liquid.ai/blog/lfm2-5-2-6b)，Liquid 为每个训练任务随机选择一个 harness。发布文章给出的理由是：直接在 Hermes Agent、OpenClaw 等 harness 内训练，可以让模型接触它们的工具、系统提示和交互模式，帮助模型在不同智能体环境中可靠工作。

OpenForgeRL 检查了这种多样性是否会牺牲峰值性能。它让同一个模型分别在一种和三种 harness 内训练；三 harness 模型即使在单 harness 模型的主场，分数也更高（48.5 对 46.0），在 Codex 下更是接近其两倍 [@yu2026openforgerl]。

<Embed n="5" title="直接查看论文中的论据" desc="八份 2026 年报告，以及引用段落所在的页面。论文截图保留原文。" />

模型报告直到最近才开始说明某个分数来自哪种 harness。2025 年，多数报告在每个基准上不注明 harness，或只注明一种。到了 2026 年，下方时间线上的每份报告都使用多种 harness 训练。

<Embed n="6" title="从不披露 harness，到跨多种 harness 训练" desc="按各报告对 harness 的最高披露程度展示近期模型发布。悬停或点击圆点可查看详情。" />

### 还缺少什么

仍然缺少的，是一种能用于任意 harness 的开放、可靠的方法。它必须从你无法控制的 harness 中采集 RL 训练所需的全部信息，并接入你已经使用的训练器。

这也不必消耗前沿模型级别的预算。Agent Lightning 用大约 6,000 个训练样本，将 Qwen3.5-9B 的 SWE-bench Verified 得分从 41.8% 提升至 56.4% [@he2026agentlightning]。同一篇论文也指出了难点：循环由 harness 掌握，训练器只能看到经过的请求与响应；如何将它们重新转为训练样本，仍是开放问题。

我们基于 OpenEnv [@openenv2026] 构建了这套系统，核心是位于 harness 与模型之间的采集代理（capture proxy）。智能体的每次调用都经过代理，代理记录模型实际看到的提示 token、实际生成的补全 token，以及对应对数概率。它支持编码智能体使用的 API 格式，因此 Claude Code、Codex、[OpenCode](https://opencode.ai) 无需修改就能运行，自定义 harness 只需指向代理。所有这些能力都位于 OpenEnv 的开放环境接口之后，因此同一套系统可以为不同训练器提供数据；我们使用的是 TRL [@trl]。

在此基础上，Harbor 集成 [@openenv2026harbor] 将 [Harbor](https://harborframework.com/) 的容器化任务作为 OpenEnv 环境提供服务，因此 harness 和沙箱都成为可在每次 rollout 中选择的配置。
