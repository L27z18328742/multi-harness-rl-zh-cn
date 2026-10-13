## 导论

如果你用 AI 写代码，很可能在不止一种工具里用过同一个模型：例如在 [Claude Code](https://code.claude.com/docs/en/overview)、[Cursor](https://cursor.com)、[Pi](https://pi.dev) 或 [Cline](https://cline.bot) 中使用 Claude。你也可能注意到，它在这些工具里的表现并不相同：规划方式不同，选择的工具不同；在一种工具中能完成的任务，换到另一种工具里可能就会卡住。

这种差异可以测量，来源是包裹在模型周围的程序，也就是 **agent harness（智能体运行框架，下文简称 harness）**。它负责运行循环、决定模型可用的工具、构造模型读取的上下文、解释模型返回的内容，并决定何时停止。更换 harness，就会改变模型看到的信息和获准执行的操作，结果也随之改变。在 SWE-bench Pro 上，[Joel Niklaus 测得](https://x.com/joelniklaus/status/2085725862142623875)，同一个 GLM-5.2 模型在一种 harness 中的得分为 23%，在另一种中为 52%。harness 的排名也不会在不同模型之间保持一致：对 GLM-5.2 而言，Codex 在十种 harness 中排名第二；对 Gemma 4 26B-A4B 而言，却排名第九。

<SourceImage />

对于自行部署的开放权重模型，harness 的影响尤其明显。如果模型从未在你的 harness 中接受训练，就可能表现不佳：它会调用不存在的工具，或者生成 harness 无法解析的输出。只在一种 harness 中训练也不能彻底解决问题，因为模型会学会这种 harness 的习惯，在其他 harness 中依然遇到困难。

解决办法是直接在这些 harness 内训练模型，针对多步工具使用开展强化学习，这称为 **agentic reinforcement learning（智能体强化学习，agentic RL）**。本研究要回答的问题是：能否拿一个小型开放模型，在人们实际使用的 Claude Code、Codex、OpenCode 等 harness 内这样训练，让它在所有这些 harness 中都得到提升？这些 harness 都按其发布时的方式运行：它们原本并非为训练设计，我们也没有修改它们的代码。本文介绍了一个实现这一目标的开放框架：以 [OpenEnv](https://github.com/huggingface/OpenEnv) 为基础，由 [Harbor](https://harborframework.com/) 提供任务和沙箱，由 [TRL](https://github.com/huggingface/trl) 负责训练。

本文分为两部分。第一部分是指南：先介绍 harness、agentic RL 和 RL 环境的[基础知识](#the-basics)，再结合近期论文和模型报告，解释[为什么模型需要跨 harness 训练](#why-train-across-harnesses)。第二部分介绍我们自己的工作：[框架如何运作](#how-the-framework-works)，以及用它在四种 harness 中同时训练小型开放模型 [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) 后的[实验结果](#training-small-models-across-harnesses)。跨四种 harness 训练后，模型在留出任务上的平均解题率由 42% 提升至 54%；对于它原本就能解出的任务，工具调用次数减少了 31%。同一个模型如果只在 OpenCode 中训练，提升主要集中在 OpenCode，在另外三种未经训练的 harness 中提升较小。我们也比较了 RL 与基于更大模型成功 rollout 的**监督微调（SFT）**，后者带来的提升更少。使用 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 的早期实验，放在训练章节的折叠部分中。
